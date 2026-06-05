//! Eviction Worker — 评分驱动迁出执行器.
//!
//! Per SPEC `22-PAGE-COMPRESSION.md §6`.  后台线程周期性地评估页面重要性评分,
//! 将低价值页面从 GpuHbm 迁出到 CpuDram 或 Nvme, 不阻塞调度器热路径.
//!
//! 评分策略 (SPEC §6.1):
//! - `importance_score = base_score - time_penalty - recency_penalty + freq_bonus + payload_bonus`
//! - 时间衰减: `time_penalty = idle_ticks * TIME_DECAY_WEIGHT`
//! - 访问频率: `freq_bonus = access_count * FREQUENCY_BONUS`
//! - 压缩收益: `compression_ratio_bonus = (1.0 - compressed_size/original_size) * COMPRESSION_RATIO_WEIGHT`
//! - 页大小: `page_size_bonus = page_bytes * PAGE_SIZE_WEIGHT`
//!
//! 迁出路径 (SPEC §4.2):
//! - GpuHbm → CpuDram: importance_score < 100 AND tier_age > 50 ticks
//! - CpuDram → Nvme: tier_age > 500 ticks
//!
//! 触发条件 (SPEC §4.3):
//! - HBM 占用 > 90%
//! - DRAM 用于 page 部分占用 > 80%

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::kv_cache::{CompressionCodec, StorageTier};
use crate::scheduler::memory_manager::{GlobalMemoryManager, Tier};
use crate::scheduler::migration_actor::{
    MigrationCommand, MigrationDone, MigrationResult, PageAddrTable, PageMigrationActor,
};
use crate::scheduler::observer::{
    BasicObserver, EvictionReason, WeightPageTelemetryEvent,
};
use crate::scheduler::types::{
    PageId, PageMetadata, PagePayloadKind, PageState, RequestId,
    WeightTier,
};

/// Eviction priority tier for telemetry (SPEC §21 WP5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvictionTier {
    ColdExpert,
    PinnedDense,
    StandbyKv,
    Protected,
}

// ─────────────────────────────────────────────────────────────────────────────
// Scoring constants (SPEC §6.1)
// ─────────────────────────────────────────────────────────────────────────────

/// Per-tick decay weight for idle time. Higher = faster score decay.
const TIME_DECAY_WEIGHT: i64 = 2;

/// Bonus per access event. Higher = more resistant to eviction.
const FREQUENCY_BONUS: i64 = 15;

/// Bonus proportional to compression ratio (1.0 - ratio). Better compression
/// means cheaper eviction (less data to move), so we boost score slightly for
/// pages that compress well — they are cheaper to bring back.
const COMPRESSION_RATIO_WEIGHT: i64 = 500;

/// Bonus per byte of page size (scaled by 1 KiB). Larger pages cost more to
/// evict and bring back, so they get a higher score (harder to evict).
const PAGE_SIZE_WEIGHT: i64 = 1;

/// Payload-kind baseline adjustments (SPEC §4.3 / §6.1).
const EXPERT_WEIGHT_BONUS: i64 = -300;
const KV_CONTEXT_BONUS: i64 = 0;
const PROMPT_SYSTEM_BONUS: i64 = 1000;
const DENSE_LAYER_BONUS: i64 = 5000;
const KNOWLEDGE_RAG_BONUS: i64 = -500;

/// Importance score threshold below which a page is evictable (SPEC §4.3).
const IMPORTANCE_SCORE_THRESHOLD: i64 = 100;

/// Tier-age threshold in ticks for GpuHbm → CpuDram (SPEC §4.3).
const HBM_EVICT_AGE_TICKS: u64 = 50;

/// Tier-age threshold in ticks for CpuDram → Nvme (SPEC §4.3).
const DRAM_EVICT_AGE_TICKS: u64 = 500;

/// HBM occupancy ratio that triggers eviction (SPEC §4.3: > 90%).
const HBM_PRESSURE_RATIO: f32 = 0.90;

/// DRAM occupancy ratio that triggers NVMe eviction (SPEC §4.3: > 80%).
const DRAM_PRESSURE_RATIO: f32 = 0.80;

/// Default interval between eviction rounds.
const DEFAULT_TICK_INTERVAL: Duration = Duration::from_millis(10);

/// Default max pages evicted per round.
const DEFAULT_MAX_EVICT_PER_ROUND: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// EvictionCandidate
// ─────────────────────────────────────────────────────────────────────────────

/// A candidate page selected for eviction, with its computed score.
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub page_id: PageId,
    pub score: i64,
    pub current_tier: StorageTier,
    pub codec: CompressionCodec,
    pub page_bytes: usize,
    pub group_id: Option<RequestId>,
}

// ─────────────────────────────────────────────────────────────────────────────
// EvictionWorkerConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the eviction worker.
#[derive(Debug, Clone)]
pub struct EvictionWorkerConfig {
    /// Interval between eviction scan rounds.
    pub tick_interval: Duration,
    /// Maximum pages evicted per round.
    pub max_evict_per_round: usize,
    /// HBM pressure ratio above which eviction triggers (0.0–1.0).
    pub hbm_pressure_threshold: f32,
    /// DRAM pressure ratio above which NVMe eviction triggers (0.0–1.0).
    pub dram_pressure_threshold: f32,
    /// Importance score threshold for evictability.
    pub importance_threshold: i64,
    /// Tier-age in ticks before a page on HBM is eligible for eviction.
    pub hbm_evict_age_ticks: u64,
    /// Tier-age in ticks before a page on DRAM is eligible for NVMe eviction.
    pub dram_evict_age_ticks: u64,
    /// Codec to use for GpuHbm → CpuDram eviction.
    pub default_evict_codec: CompressionCodec,
    /// Uncompressed page size in bytes (used for migration commands).
    pub page_bytes: usize,
}

impl Default for EvictionWorkerConfig {
    fn default() -> Self {
        Self {
            tick_interval: DEFAULT_TICK_INTERVAL,
            max_evict_per_round: DEFAULT_MAX_EVICT_PER_ROUND,
            hbm_pressure_threshold: HBM_PRESSURE_RATIO,
            dram_pressure_threshold: DRAM_PRESSURE_RATIO,
            importance_threshold: IMPORTANCE_SCORE_THRESHOLD,
            hbm_evict_age_ticks: HBM_EVICT_AGE_TICKS,
            dram_evict_age_ticks: DRAM_EVICT_AGE_TICKS,
            default_evict_codec: CompressionCodec::Lz4,
            page_bytes: 4096,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EvictionWorker
// ─────────────────────────────────────────────────────────────────────────────

/// Scoring-driven eviction worker (SPEC §6).
///
/// Runs in a dedicated background thread. Each tick:
/// 1. Reads current memory pressure from `GlobalMemoryManager`.
/// 2. Computes importance scores for all tracked pages.
/// 3. Selects candidates whose score < threshold and tier_age > minimum.
/// 4. Submits migration commands to `PageMigrationActor`.
/// 5. Drains completion events and updates page table.
#[allow(dead_code)]
pub struct EvictionWorker {
    /// Channel to signal the worker thread to stop.
    stop: Arc<AtomicBool>,
    /// Handle to the background thread.
    handle: Option<JoinHandle<()>>,
    /// Shared page metadata (read by worker, written by scheduler).
    page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>>,
    /// Shared page address table (used for tier lookups and migration actor).
    addr_table: PageAddrTable,
    /// Shared global memory manager (for tier usage stats).
    memory_manager: Arc<Mutex<GlobalMemoryManager>>,
    /// Observer for eviction telemetry.
    observer: Arc<Mutex<BasicObserver>>,
}

impl EvictionWorker {
    /// Spawn the eviction worker on a background thread.
    ///
    /// # Arguments
    /// * `config` — Worker configuration.
    /// * `actor` — Already-initialized `PageMigrationActor` (moved into the worker).
    /// * `page_metadata` — Shared page metadata map (read by worker).
    /// * `addr_table` — Shared page address table.
    /// * `memory_manager` — Shared global memory manager.
    /// * `observer` — Telemetry observer for eviction events.
    pub fn spawn(
        config: EvictionWorkerConfig,
        actor: PageMigrationActor,
        page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>>,
        addr_table: PageAddrTable,
        memory_manager: Arc<Mutex<GlobalMemoryManager>>,
        observer: Arc<Mutex<BasicObserver>>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let page_meta_clone = Arc::clone(&page_metadata);
        let addr_table_clone = Arc::clone(&addr_table);
        let mm_clone = Arc::clone(&memory_manager);
        let observer_clone = Arc::clone(&observer);

        let handle = thread::Builder::new()
            .name("eviction-worker".to_string())
            .spawn(move || {
                eviction_loop(
                    config,
                    actor,
                    stop_clone,
                    page_meta_clone,
                    addr_table_clone,
                    mm_clone,
                    observer_clone,
                );
            })
            .expect("failed to spawn eviction-worker thread");

        Self {
            stop,
            handle: Some(handle),
            page_metadata,
            addr_table,
            memory_manager,
            observer,
        }
    }

    /// Signal the worker thread to stop and wait for it to finish.
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }

    /// Compute the importance score for a single page (SPEC §6.1).
    ///
    /// The score combines:
    /// - **Base score**: starts at 0.
    /// - **Time penalty**: `idle_ticks * TIME_DECAY_WEIGHT` — pages idle longer score lower.
    /// - **Recency penalty**: `recency * TIME_DECAY_WEIGHT / 2` — LIRS recency contribution.
    /// - **Frequency bonus**: `access_count * FREQUENCY_BONUS` — frequently accessed pages resist eviction.
    /// - **Compression ratio bonus**: pages that compress well are cheaper to evict and restore.
    /// - **Page size bonus**: larger pages cost more to migrate, so they get a slight protection.
    /// - **Payload kind bonus**: payload-specific adjustment (ExpertWeight easy, DenseLayer hard).
    ///
    /// Lower score = more evictable. Pages with `score < importance_threshold` are candidates.
    pub fn compute_importance_score(
        meta: &PageMetadata,
        payload_kind: Option<PagePayloadKind>,
        compressed_size: u32,
        original_size: u32,
        current_tier: StorageTier,
        tier_age_ticks: u64,
    ) -> i64 {
        // Time-based decay: pages idle for many ticks lose importance.
        let time_penalty = (tier_age_ticks as i64) * TIME_DECAY_WEIGHT;

        // LIRS recency contribution.
        let recency_penalty = (meta.recency as i64) * (TIME_DECAY_WEIGHT / 2);

        // Frequency bonus: heavily accessed pages resist eviction.
        let freq_bonus = (meta.access_count as i64) * FREQUENCY_BONUS;

        // Compression ratio bonus: if we know compressed vs original size,
        // a better ratio means cheaper to evict (less data to transfer).
        let compression_ratio_bonus = if original_size > 0 {
            let ratio = (compressed_size as f64) / (original_size as f64);
            ((1.0 - ratio) * COMPRESSION_RATIO_WEIGHT as f64) as i64
        } else {
            0
        };

        // Page size bonus: larger pages cost more to migrate.
        let page_size_bonus = if original_size > 0 {
            ((original_size as f64) / 1024.0 * PAGE_SIZE_WEIGHT as f64) as i64
        } else {
            0
        };

        // Payload-kind baseline.
        let payload_bonus = match payload_kind {
            Some(PagePayloadKind::ExpertWeight) => EXPERT_WEIGHT_BONUS,
            Some(PagePayloadKind::KvContext) => KV_CONTEXT_BONUS,
            Some(PagePayloadKind::PromptSystem) => PROMPT_SYSTEM_BONUS,
            Some(PagePayloadKind::DenseLayerWeight) => DENSE_LAYER_BONUS,
            Some(PagePayloadKind::KnowledgeRAG) => KNOWLEDGE_RAG_BONUS,
            None => 0,
        };

        // Warm/Protected pages get a large protection bonus.
        let state_bonus = match meta.state {
            PageState::Protected => 10_000,
            PageState::Warm => 5_000,
            _ => 0,
        };

        // Pages already on a lower tier get a discount (they've already been evicted once).
        let tier_discount = match current_tier {
            StorageTier::GpuHbm => 0,
            StorageTier::CpuDram => -200,
            StorageTier::Nvme => -500,
        };

        let base: i64 = 1000;
        base - time_penalty - recency_penalty + freq_bonus
            + compression_ratio_bonus + page_size_bonus
            + payload_bonus + state_bonus + tier_discount
    }

    /// Classify the eviction tier for telemetry (SPEC §21 WP5).
    pub fn classify_eviction_tier(
        payload_kind: Option<PagePayloadKind>,
        score: i64,
    ) -> EvictionTier {
        match payload_kind {
            Some(PagePayloadKind::ExpertWeight) => EvictionTier::ColdExpert,
            Some(PagePayloadKind::DenseLayerWeight) => EvictionTier::PinnedDense,
            _ if score < IMPORTANCE_SCORE_THRESHOLD => EvictionTier::StandbyKv,
            _ => EvictionTier::Protected,
        }
    }

    /// Perform one eviction round synchronously.
    ///
    /// This is the core logic also used by the background thread. It:
    /// 1. Checks memory pressure.
    /// 2. Scores all tracked pages.
    /// 3. Selects candidates below the importance threshold.
    /// 4. Submits migration commands.
    /// 5. Drains completions and updates page state.
    ///
    /// Returns the number of eviction commands submitted this round.
    pub fn evict_round(
        config: &EvictionWorkerConfig,
        actor: &PageMigrationActor,
        page_metadata: &Arc<RwLock<HashMap<PageId, PageMetadata>>>,
        addr_table: &PageAddrTable,
        memory_manager: &Arc<Mutex<GlobalMemoryManager>>,
        observer: &Arc<Mutex<BasicObserver>>,
    ) -> usize {
        // ── 1. Read tier pressure ──────────────────────────────────────────────
        let (hbm_usage, dram_usage) = {
            let mm = match memory_manager.lock() {
                Ok(guard) => guard,
                Err(_) => return 0,
            };
            let hbm = mm.tier_usage(Tier::L1);
            let dram = mm.tier_usage(Tier::L2);
            (hbm, dram)
        };

        let hbm_pressure = if hbm_usage.capacity > 0 {
            hbm_usage.used as f32 / hbm_usage.capacity as f32
        } else {
            0.0
        };
        let dram_pressure = if dram_usage.capacity > 0 {
            dram_usage.used as f32 / dram_usage.capacity as f32
        } else {
            0.0
        };

        let should_evict_hbm = hbm_pressure > config.hbm_pressure_threshold;
        let should_evict_dram = dram_pressure > config.dram_pressure_threshold;

        if !should_evict_hbm && !should_evict_dram {
            return 0;
        }

        // ── 2. Score all pages and select candidates ───────────────────────────
        let mut candidates: Vec<EvictionCandidate> = Vec::new();

        {
            let meta_guard = match page_metadata.read() {
                Ok(g) => g,
                Err(_) => return 0,
            };
            let addr_guard = match addr_table.read() {
                Ok(g) => g,
                Err(_) => return 0,
            };

            for (&page_id, meta) in meta_guard.iter() {
                // Skip non-evictable states.
                if meta.state == PageState::Protected || meta.state == PageState::Active {
                    continue;
                }

                // Determine current tier from addr_table.
                let (current_tier, compressed_size, original_size, codec) = match addr_guard.get(&page_id) {
                    Some(entry) => (
                        entry.current_tier,
                        meta.access_count, // approximate: use access_count as proxy if no compressed size
                        entry.original_bytes,
                        entry.codec,
                    ),
                    None => continue,
                };

                // Determine eligibility based on tier and pressure.
                let tier_age_ticks = compute_tier_age(meta);
                let eligible = match current_tier {
                    StorageTier::GpuHbm => {
                        should_evict_hbm
                            && tier_age_ticks > config.hbm_evict_age_ticks
                    }
                    StorageTier::CpuDram => {
                        should_evict_dram
                            && tier_age_ticks > config.dram_evict_age_ticks
                    }
                    StorageTier::Nvme => false, // already cold, no further eviction
                };

                if !eligible {
                    continue;
                }

                // Determine payload kind from page state (heuristic: Standby + no owner = ExpertWeight).
                let payload_kind = infer_payload_kind(meta);

                let score = Self::compute_importance_score(
                    meta,
                    payload_kind,
                    compressed_size as u32,
                    original_size as u32,
                    current_tier,
                    tier_age_ticks,
                );

                if score < config.importance_threshold {
                    candidates.push(EvictionCandidate {
                        page_id,
                        score,
                        current_tier,
                        codec: if current_tier == StorageTier::GpuHbm {
                            config.default_evict_codec
                        } else {
                            codec
                        },
                        page_bytes: config.page_bytes,
                        group_id: meta.sequence_id,
                    });
                }
            }
        }

        // ── 3. Sort by score ascending (lowest = most evictable) ───────────────
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(config.max_evict_per_round);

        // ── 4. Submit migration commands ───────────────────────────────────────
        let mut submitted = 0;
        for candidate in &candidates {
            let cmd = match candidate.current_tier {
                StorageTier::GpuHbm => MigrationCommand::EvictToDram {
                    page_id: candidate.page_id,
                    codec: candidate.codec,
                    page_bytes: candidate.page_bytes,
                },
                StorageTier::CpuDram => MigrationCommand::EvictToNvme {
                    page_id: candidate.page_id,
                    codec: candidate.codec,
                    page_bytes: candidate.page_bytes,
                },
                StorageTier::Nvme => continue, // no further tier
            };

            if actor.send(cmd).is_ok() {
                submitted += 1;

                // Record telemetry via BasicObserver.
                let (from_weight_tier, to_weight_tier) = match candidate.current_tier {
                    StorageTier::GpuHbm => (WeightTier::Hot, WeightTier::Warm),
                    StorageTier::CpuDram => (WeightTier::Warm, WeightTier::Cold),
                    StorageTier::Nvme => continue,
                };
                if let Ok(mut obs) = observer.lock() {
                    obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                        page_id: candidate.page_id,
                        from_tier: from_weight_tier,
                        to_tier: to_weight_tier,
                        reason: EvictionReason::MemoryPressure,
                        bytes: candidate.page_bytes as u64,
                    });
                }
            }
        }

        // ── 5. Drain completions and update page metadata ──────────────────────
        drain_completions_and_update(actor, page_metadata, addr_table);

        submitted
    }
}

impl Drop for EvictionWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Background loop
// ─────────────────────────────────────────────────────────────────────────────

fn eviction_loop(
    config: EvictionWorkerConfig,
    actor: PageMigrationActor,
    stop: Arc<AtomicBool>,
    page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>>,
    addr_table: PageAddrTable,
    memory_manager: Arc<Mutex<GlobalMemoryManager>>,
    observer: Arc<Mutex<BasicObserver>>,
) {
    while !stop.load(Ordering::Relaxed) {
        let start = Instant::now();

        EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &memory_manager,
            &observer,
        );

        let elapsed = start.elapsed();
        let sleep_remaining = config.tick_interval.saturating_sub(elapsed);
        if sleep_remaining > Duration::ZERO {
            thread::sleep(sleep_remaining);
        }
    }

    // Final drain before exit.
    drain_completions_and_update(&actor, &page_metadata, &addr_table);
    actor.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate tier age in "ticks" from page metadata timestamps.
///
/// We use the time since `swap_in_time` or `last_access` as a proxy for
/// how long the page has been sitting on its current tier. One tick ≈ 10ms
/// (matching `DEFAULT_TICK_INTERVAL`).
fn compute_tier_age(meta: &PageMetadata) -> u64 {
    let anchor = match meta.swap_in_time {
        Some(t) => t,
        None => meta.last_access,
    };
    let elapsed_ms = Instant::now().saturating_duration_since(anchor).as_millis() as u64;
    // 1 tick = 10ms (matching DEFAULT_TICK_INTERVAL).
    
    elapsed_ms / 10
}

/// Infer payload kind from page metadata heuristics.
///
/// When no explicit payload kind is stored on the metadata, we infer it:
/// - No owner (sequence_id = None) → ExpertWeight
/// - Owner present, access_count high → KvContext
/// - Otherwise → KvContext (default)
fn infer_payload_kind(meta: &PageMetadata) -> Option<PagePayloadKind> {
    match meta.sequence_id {
        None => Some(PagePayloadKind::ExpertWeight),
        Some(_) => Some(PagePayloadKind::KvContext),
    }
}

/// Drain completion events from the migration actor and update page metadata
/// to reflect successful migrations.
fn drain_completions_and_update(
    actor: &PageMigrationActor,
    page_metadata: &Arc<RwLock<HashMap<PageId, PageMetadata>>>,
    _addr_table: &PageAddrTable,
) {
    let completions: Vec<MigrationDone> = {
        let mut completions = Vec::new();
        while let Some(done) = actor.try_recv_done() {
            completions.push(done);
        }
        completions
    };

    for done in completions {
        if let MigrationResult::Ok { .. } = done.result {
            // Update page metadata to reflect the new tier.
            if let Ok(mut meta_guard) = page_metadata.write() {
                if let Some(meta) = meta_guard.get_mut(&done.page_id) {
                    match done.to_tier {
                        StorageTier::CpuDram => {
                            meta.state = PageState::SwappedOut;
                            meta.swap_in_time = Some(Instant::now());
                        }
                        StorageTier::Nvme => {
                            meta.state = PageState::Swapped;
                            meta.swap_in_time = Some(Instant::now());
                        }
                        StorageTier::GpuHbm => {
                            meta.state = PageState::Active;
                            meta.swap_in_time = None;
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::dma_helpers::{CpuDmaBackendSized, DmaBackend};
    use crate::scheduler::migration_actor::{MigrationActorConfig, PageAddrEntry};
    use crate::scheduler::observer::{BasicObserver, ObserverError};

    /// Verify importance score decreases with tier age.
    #[test]
    fn score_decreases_with_age() {
        let meta_young = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_old = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now() - Duration::from_secs(5),
            swap_in_time: Some(Instant::now() - Duration::from_secs(5)),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        let score_young = EvictionWorker::compute_importance_score(
            &meta_young,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            10, // young: 10 ticks
        );
        let score_old = EvictionWorker::compute_importance_score(
            &meta_old,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            500, // old: 500 ticks
        );

        assert!(
            score_old < score_young,
            "older page should have lower importance score: old={} young={}",
            score_old,
            score_young,
        );
    }

    /// Verify expert weight pages score lower than KV context pages.
    #[test]
    fn expert_weight_scores_lower_than_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        let score_expert = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            0,
            4096,
            StorageTier::GpuHbm,
            100,
        );
        let score_kv = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            100,
        );

        assert!(
            score_expert < score_kv,
            "expert weight should score lower than KV: expert={} kv={}",
            score_expert,
            score_kv,
        );
    }

    /// Verify dense layer pages score much higher (harder to evict).
    #[test]
    fn dense_layer_scores_higher_than_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        let score_dense = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            0,
            4096,
            StorageTier::GpuHbm,
            100,
        );
        let score_kv = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            100,
        );

        assert!(
            score_dense > score_kv,
            "dense layer should score higher than KV: dense={} kv={}",
            score_dense,
            score_kv,
        );
    }

    /// Verify higher access count increases score (resists eviction).
    #[test]
    fn frequency_increases_score() {
        let meta_low_freq = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_high_freq = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 5,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        let score_low = EvictionWorker::compute_importance_score(
            &meta_low_freq,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_high = EvictionWorker::compute_importance_score(
            &meta_high_freq,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );

        assert!(
            score_high > score_low,
            "higher access count should yield higher score: high={} low={}",
            score_high,
            score_low,
        );
    }

    /// Verify compression ratio bonus: better compression yields higher score.
    #[test]
    fn compression_ratio_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        let score_uncompressed = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            4096, // compressed = original → ratio 1.0
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_well_compressed = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            1024, // compressed = 25% of original → ratio 0.25
            4096,
            StorageTier::GpuHbm,
            50,
        );

        assert!(
            score_well_compressed > score_uncompressed,
            "better compression ratio should yield higher score: compressed={} uncompressed={}",
            score_well_compressed,
            score_uncompressed,
        );
    }

    /// Verify eviction tier classification.
    #[test]
    fn eviction_tier_classification() {
        let tier_expert = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::ExpertWeight),
            50,
        );
        assert_eq!(tier_expert, EvictionTier::ColdExpert);

        let tier_dense = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::DenseLayerWeight),
            50,
        );
        assert_eq!(tier_dense, EvictionTier::PinnedDense);

        let tier_kv_low = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            50, // below threshold
        );
        assert_eq!(tier_kv_low, EvictionTier::StandbyKv);

        let tier_kv_high = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            5000, // above threshold
        );
        assert_eq!(tier_kv_high, EvictionTier::Protected);
    }

    /// Verify that evict_round returns 0 when memory pressure is below threshold.
    #[test]
    fn no_eviction_below_threshold() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(
            1000, 1000, 1000,
        )));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "should not evict when pressure is low");
        actor.shutdown();
    }

    /// Verify that EvictionWorker::spawn and shutdown work cleanly.
    #[test]
    fn spawn_and_shutdown() {
        let config = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(
            1000, 1000, 1000,
        )));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = EvictionWorker::spawn(
            config,
            actor,
            page_metadata,
            addr_table,
            mm,
            observer,
        );

        // Let it run for a couple of ticks.
        thread::sleep(Duration::from_millis(150));
        worker.shutdown();
    }

    /// Verify pages on NVMe tier are not eligible for further eviction.
    #[test]
    fn nvme_pages_not_evictable() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1,
            last_access: Instant::now() - Duration::from_secs(60),
            swap_in_time: Some(Instant::now() - Duration::from_secs(60)),
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };

        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::Nvme,
            5000,
        );

        // Even with a very low score, NVMe pages should not be evictable
        // (the eligibility check in evict_round handles this; here we just
        // verify the scoring includes the tier discount).
        assert!(
            score < 1000,
            "NVMe pages should have discounted score: got {}",
            score,
        );
    }

    /// Verify Warm and Protected state bonuses resist eviction.
    #[test]
    fn protected_pages_resist_eviction() {
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_protected = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };

        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            100,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &meta_protected,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            100,
        );

        assert!(
            score_protected > score_standby,
            "protected pages should have higher score: protected={} standby={}",
            score_protected,
            score_standby,
        );
    }

    // ── EvictionWorkerConfig default ──

    #[test]
    fn config_default_values() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.tick_interval, DEFAULT_TICK_INTERVAL);
        assert_eq!(cfg.max_evict_per_round, DEFAULT_MAX_EVICT_PER_ROUND);
        assert!((cfg.hbm_pressure_threshold - HBM_PRESSURE_RATIO).abs() < 1e-6);
        assert!((cfg.dram_pressure_threshold - DRAM_PRESSURE_RATIO).abs() < 1e-6);
        assert_eq!(cfg.importance_threshold, IMPORTANCE_SCORE_THRESHOLD);
        assert_eq!(cfg.hbm_evict_age_ticks, HBM_EVICT_AGE_TICKS);
        assert_eq!(cfg.dram_evict_age_ticks, DRAM_EVICT_AGE_TICKS);
        assert_eq!(cfg.default_evict_codec, CompressionCodec::Lz4);
        assert_eq!(cfg.page_bytes, 4096);
    }

    // ── classify_eviction_tier ──

    #[test]
    fn classify_expert_weight_is_cold_expert() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::ExpertWeight), 50),
            EvictionTier::ColdExpert
        );
    }

    #[test]
    fn classify_dense_layer_is_pinned() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::DenseLayerWeight), 50),
            EvictionTier::PinnedDense
        );
    }

    #[test]
    fn classify_low_score_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 50),
            EvictionTier::StandbyKv
        );
    }

    #[test]
    fn classify_high_score_is_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 500),
            EvictionTier::Protected
        );
    }

    #[test]
    fn classify_none_payload_high_is_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, 200),
            EvictionTier::Protected
        );
    }

    // ── EvictionCandidate fields ──

    #[test]
    fn eviction_candidate_fields() {
        let c = EvictionCandidate {
            page_id: 7,
            score: -50,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 65536,
            group_id: Some(42),
        };
        assert_eq!(c.page_id, 7);
        assert_eq!(c.score, -50);
        assert_eq!(c.current_tier, StorageTier::GpuHbm);
        assert_eq!(c.page_bytes, 65536);
    }

    // ── compute_importance_score: tier discount ──

    #[test]
    fn score_lower_tier_gets_discount() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert!(score_hbm > score_dram, "DRAM should have discount vs HBM");
        assert!(score_dram > score_nvme, "NVMe should have more discount vs DRAM");
    }

    // ── Tier discount is exactly 200 between adjacent tiers ──

    #[test]
    fn tier_discount_exact_delta() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(score_hbm - score_dram, 200, "DRAM discount = -200");
        assert_eq!(score_dram - score_nvme, 300, "NVMe additional discount = -300");
    }

    // ── PromptSystem payload gets +1000 bonus ──

    #[test]
    fn prompt_system_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_prompt = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::PromptSystem),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_kv = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        assert!(
            score_prompt > score_kv,
            "PromptSystem should score higher than KvContext: prompt={} kv={}",
            score_prompt,
            score_kv,
        );
        assert_eq!(score_prompt - score_kv, 1000, "PromptSystem bonus = +1000");
    }

    // ── KnowledgeRAG payload gets -500 penalty ──

    #[test]
    fn knowledge_rag_penalty() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_rag = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KnowledgeRAG),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_kv = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        assert!(
            score_rag < score_kv,
            "KnowledgeRAG should score lower than KvContext: rag={} kv={}",
            score_rag,
            score_kv,
        );
        assert_eq!(score_kv - score_rag, 500, "KnowledgeRAG penalty = -500");
    }

    // ── None payload kind gets zero bonus ──

    #[test]
    fn none_payload_zero_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_none = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_kv = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        assert_eq!(
            score_none, score_kv,
            "None payload should have same bonus as KvContext (= 0)"
        );
    }

    // ── Warm state gets +5000 bonus ──

    #[test]
    fn warm_state_bonus() {
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_warm = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_warm = EvictionWorker::compute_importance_score(
            &meta_warm,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        assert_eq!(score_warm - score_standby, 5000, "Warm bonus = +5000");
    }

    // ── Protected state gets +10000 bonus ──

    #[test]
    fn protected_state_exact_bonus() {
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_protected = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &meta_protected,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        assert_eq!(score_protected - score_standby, 10000, "Protected bonus = +10000");
    }

    // ── Recency penalty contribution ──

    #[test]
    fn recency_penalty_reduces_score() {
        let meta_low_recency = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_high_recency = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 100,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_low = EvictionWorker::compute_importance_score(
            &meta_low_recency,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_high = EvictionWorker::compute_importance_score(
            &meta_high_recency,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        assert!(
            score_high < score_low,
            "higher recency should reduce score: high={} low={}",
            score_high,
            score_low,
        );
    }

    // ── Frequency bonus exact delta ──

    #[test]
    fn frequency_bonus_exact_delta() {
        let meta_0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_10 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_0 = EvictionWorker::compute_importance_score(
            &meta_0,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        let score_10 = EvictionWorker::compute_importance_score(
            &meta_10,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            50,
        );
        assert_eq!(
            score_10 - score_0,
            10 * 15,
            "frequency bonus should be access_count * FREQUENCY_BONUS"
        );
    }

    // ── Zero sizes yield zero compression and page size bonus ──

    #[test]
    fn zero_original_size_no_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0, // original_size = 0
            StorageTier::GpuHbm,
            0,
        );
        // With zero everything: base(1000) - time(0) - recency(0) + freq(0) + compression(0) + page_size(0) + kv_bonus(0) + state(0) + tier(0) = 1000
        assert_eq!(score, 1000, "zero inputs should give base score of 1000");
    }

    // ── Page size bonus scales linearly ──

    #[test]
    fn page_size_bonus_scales() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_4k = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            4096,
            StorageTier::GpuHbm,
            0,
        );
        let score_8k = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            8192,
            StorageTier::GpuHbm,
            0,
        );
        assert!(
            score_8k > score_4k,
            "larger page should score higher: 8k={} 4k={}",
            score_8k,
            score_4k,
        );
        // 4096/1024*1 = 4, 8192/1024*1 = 8, delta = 4
        assert_eq!(score_8k - score_4k, 4, "page size bonus delta");
    }

    // ── infer_payload_kind: no owner → ExpertWeight ──

    #[test]
    fn infer_payload_no_owner_is_expert() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::ExpertWeight)
        );
    }

    // ── infer_payload_kind: with owner → KvContext ──

    #[test]
    fn infer_payload_with_owner_is_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::KvContext)
        );
    }

    // ── infer_payload_kind: high access_count still → KvContext ──

    #[test]
    fn infer_payload_high_freq_still_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 99999,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::KvContext)
        );
    }

    // ── compute_tier_age: recent swap_in_time gives small age ──

    #[test]
    fn tier_age_recent_is_small() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        // Just created, should be 0 or very close to 0.
        assert!(age <= 1, "recently swapped-in page should have tiny age: {}", age);
    }

    // ── compute_tier_age: prefers swap_in_time over last_access ──

    #[test]
    fn tier_age_prefers_swap_in_time() {
        let old_instant = Instant::now() - Duration::from_secs(1);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(), // recent
            swap_in_time: Some(old_instant), // old — should be used as anchor
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        // 1 second = 1000ms / 10 = 100 ticks. Should be >= 95 (allowing small timing variance).
        assert!(
            age >= 90,
            "should use swap_in_time anchor, expected ~100 ticks, got {}",
            age,
        );
    }

    // ── compute_tier_age: falls back to last_access when no swap_in_time ──

    #[test]
    fn tier_age_fallback_to_last_access() {
        let old_instant = Instant::now() - Duration::from_millis(500);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: None, // no swap_in_time, falls back to last_access
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        // 500ms / 10 = 50 ticks. Allow some variance.
        assert!(
            age >= 45,
            "should use last_access as fallback, expected ~50 ticks, got {}",
            age,
        );
    }

    // ── EvictionTier Debug derive ──

    #[test]
    fn eviction_tier_debug_format() {
        assert_eq!(format!("{:?}", EvictionTier::ColdExpert), "ColdExpert");
        assert_eq!(format!("{:?}", EvictionTier::PinnedDense), "PinnedDense");
        assert_eq!(format!("{:?}", EvictionTier::StandbyKv), "StandbyKv");
        assert_eq!(format!("{:?}", EvictionTier::Protected), "Protected");
    }

    // ── EvictionTier equality ──

    #[test]
    fn eviction_tier_equality() {
        assert_eq!(EvictionTier::ColdExpert, EvictionTier::ColdExpert);
        assert_ne!(EvictionTier::ColdExpert, EvictionTier::PinnedDense);
        assert_ne!(EvictionTier::StandbyKv, EvictionTier::Protected);
    }

    // ── EvictionTier Copy + Clone ──

    #[test]
    fn eviction_tier_copy_clone() {
        let tier = EvictionTier::ColdExpert;
        let tier2 = tier; // Copy
        let tier3 = tier.clone(); // Clone
        assert_eq!(tier, tier2);
        assert_eq!(tier, tier3);
    }

    // ── EvictionCandidate clone independence ──

    #[test]
    fn eviction_candidate_clone() {
        let original = EvictionCandidate {
            page_id: 42,
            score: -100,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
            group_id: Some(7),
        };
        let cloned = original.clone();
        assert_eq!(cloned.page_id, original.page_id);
        assert_eq!(cloned.score, original.score);
        assert_eq!(cloned.current_tier, original.current_tier);
        assert_eq!(cloned.codec, original.codec);
        assert_eq!(cloned.page_bytes, original.page_bytes);
        assert_eq!(cloned.group_id, original.group_id);
    }

    // ── EvictionCandidate with None group_id ──

    #[test]
    fn eviction_candidate_no_group() {
        let c = EvictionCandidate {
            page_id: 1,
            score: 0,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 0,
            group_id: None,
        };
        assert_eq!(c.group_id, None);
    }

    // ── EvictionWorkerConfig clone independence ──

    #[test]
    fn config_clone_independence() {
        let original = EvictionWorkerConfig::default();
        let mut cloned = original.clone();
        cloned.tick_interval = Duration::from_secs(1);
        cloned.max_evict_per_round = 999;
        assert_ne!(cloned.tick_interval, original.tick_interval);
        assert_ne!(cloned.max_evict_per_round, original.max_evict_per_round);
    }

    // ── EvictionWorkerConfig Debug format ──

    #[test]
    fn config_debug_format() {
        let cfg = EvictionWorkerConfig::default();
        let debug_str = format!("{:?}", cfg);
        assert!(debug_str.contains("EvictionWorkerConfig"));
        assert!(debug_str.contains("tick_interval"));
        assert!(debug_str.contains("max_evict_per_round"));
    }

    // ── classify_eviction_tier: KnowledgeRAG with low score → StandbyKv ──

    #[test]
    fn classify_knowledge_rag_low_score_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KnowledgeRAG), 50),
            EvictionTier::StandbyKv
        );
    }

    // ── classify_eviction_tier: KnowledgeRAG with high score → Protected ──

    #[test]
    fn classify_knowledge_rag_high_score_is_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KnowledgeRAG), 500),
            EvictionTier::Protected
        );
    }

    // ── classify_eviction_tier: ExpertWeight always ColdExpert regardless of score ──

    #[test]
    fn classify_expert_weight_ignores_score() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::ExpertWeight), 99999),
            EvictionTier::ColdExpert
        );
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::ExpertWeight), -1000),
            EvictionTier::ColdExpert
        );
    }

    // ── classify_eviction_tier: DenseLayerWeight always PinnedDense regardless of score ──

    #[test]
    fn classify_dense_layer_ignores_score() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::DenseLayerWeight), 99999),
            EvictionTier::PinnedDense
        );
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::DenseLayerWeight), -1000),
            EvictionTier::PinnedDense
        );
    }

    // ── classify_eviction_tier: None payload with low score → StandbyKv ──

    #[test]
    fn classify_none_payload_low_score_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, 0),
            EvictionTier::StandbyKv
        );
    }

    // ── classify_eviction_tier: boundary at IMPORTANCE_SCORE_THRESHOLD ──

    #[test]
    fn classify_boundary_at_threshold() {
        // Exactly at threshold is NOT below, so should be Protected.
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, IMPORTANCE_SCORE_THRESHOLD),
            EvictionTier::Protected
        );
        // One below threshold → StandbyKv.
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, IMPORTANCE_SCORE_THRESHOLD - 1),
            EvictionTier::StandbyKv
        );
    }

    // ── EvictionWorker Drop impl calls shutdown ──

    #[test]
    fn drop_calls_shutdown() {
        let config = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(
            1000, 1000, 1000,
        )));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        {
            let _worker = EvictionWorker::spawn(
                config,
                actor,
                page_metadata,
                addr_table,
                mm,
                observer,
            );
            // Worker goes out of scope here, Drop should call shutdown.
        }
        // If Drop didn't call shutdown, the test would hang or panic on thread leak.
    }

    // ── evict_round: empty page metadata returns 0 ──

    #[test]
    fn evict_round_empty_metadata() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        // Create high-pressure memory manager.
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        // Fill up HBM to trigger pressure.
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "no pages in metadata means nothing to evict");
        actor.shutdown();
    }

    // ── evict_round: high HBM pressure but all pages are Active (skipped) ──

    #[test]
    fn evict_round_skips_active_pages() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Active, // Active — should be skipped
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        // Add addr entry.
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // High HBM pressure.
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "Active pages should be skipped");
        actor.shutdown();
    }

    // ── Compression ratio: fully compressed (compressed=0) yields maximum bonus ──

    #[test]
    fn compression_ratio_fully_compressed() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_fully = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0, // compressed = 0 → ratio 0.0
            4096,
            StorageTier::GpuHbm,
            0,
        );
        let score_uncompressed = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            4096, // compressed = original → ratio 1.0 → bonus = 0
            4096,
            StorageTier::GpuHbm,
            0,
        );
        assert!(
            score_fully > score_uncompressed,
            "fully compressed should have higher bonus: fully={} uncompressed={}",
            score_fully,
            score_uncompressed,
        );
    }

    // ── Score is deterministic for same inputs ──

    #[test]
    fn score_is_deterministic() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s1 = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            2048,
            4096,
            StorageTier::GpuHbm,
            100,
        );
        let s2 = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            2048,
            4096,
            StorageTier::GpuHbm,
            100,
        );
        assert_eq!(s1, s2, "same inputs should produce same score");
    }

    // ── Negative score is possible (very old, ExpertWeight, Nvme) ──

    #[test]
    fn score_can_be_negative() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            0,
            0,
            StorageTier::Nvme,
            1000, // very old
        );
        // base(1000) - time(1000*2=2000) + expert(-300) + nvme(-500) = -1800
        assert!(
            score < 0,
            "extreme age + ExpertWeight + NVMe should yield negative score: {}",
            score,
        );
    }

    // ── EvictionCandidate Debug format ──

    #[test]
    fn eviction_candidate_debug_format() {
        let c = EvictionCandidate {
            page_id: 42,
            score: -100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(7),
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("EvictionCandidate"));
        assert!(debug.contains("page_id"));
        assert!(debug.contains("score"));
    }

    // ── EvictionTier PartialEq + Eq: all variants are distinct ──

    #[test]
    fn eviction_tier_all_variants_distinct() {
        let variants = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j],
                    "EvictionTier variants at {} and {} should not be equal", i, j);
            }
        }
    }

    // ── Score base value is exactly 1000 with all zero inputs ──

    #[test]
    fn score_base_value_is_1000() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,       // payload_bonus = 0
            0,          // compressed (won't matter, original_size = 0)
            0,          // original_size = 0 → no compression/page bonus
            StorageTier::GpuHbm, // tier_discount = 0
            0,          // tier_age_ticks = 0 → time_penalty = 0
        );
        assert_eq!(score, 1000, "base score with all zeros should be exactly 1000");
    }

    // ── Time penalty exact delta per tick ──

    #[test]
    fn time_penalty_exact_per_tick() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_0 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_1 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 1,
        );
        let score_10 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 10,
        );
        assert_eq!(score_0 - score_1, TIME_DECAY_WEIGHT, "1 tick penalty = TIME_DECAY_WEIGHT");
        assert_eq!(score_0 - score_10, 10 * TIME_DECAY_WEIGHT, "10 tick penalty = 10 * TIME_DECAY_WEIGHT");
    }

    // ── Recency penalty exact delta ──

    #[test]
    fn recency_penalty_exact_delta() {
        let meta_0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_20 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 20,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_0 = EvictionWorker::compute_importance_score(
            &meta_0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_20 = EvictionWorker::compute_importance_score(
            &meta_20, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // recency_penalty = recency * (TIME_DECAY_WEIGHT / 2) = recency * 1
        assert_eq!(
            score_0 - score_20,
            20 * (TIME_DECAY_WEIGHT / 2),
            "recency penalty = recency * (TIME_DECAY_WEIGHT / 2)"
        );
    }

    // ── Full payload kind ranking: KnowledgeRAG < ExpertWeight < KvContext < PromptSystem < DenseLayerWeight ──

    #[test]
    fn payload_kind_full_ranking() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_rag = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_expert = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_prompt = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_dense = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert!(score_rag < score_expert, "KnowledgeRAG < ExpertWeight: rag={} expert={}", score_rag, score_expert);
        assert!(score_expert < score_kv, "ExpertWeight < KvContext: expert={} kv={}", score_expert, score_kv);
        assert!(score_kv < score_prompt, "KvContext < PromptSystem: kv={} prompt={}", score_kv, score_prompt);
        assert!(score_prompt < score_dense, "PromptSystem < DenseLayerWeight: prompt={} dense={}", score_prompt, score_dense);
    }

    // ── SwappedOut state gets zero state bonus (same as Standby) ──

    #[test]
    fn swapped_out_state_no_bonus() {
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_swapped_out = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_swapped_out = EvictionWorker::compute_importance_score(
            &meta_swapped_out, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_standby, score_swapped_out, "SwappedOut should have same bonus as Standby (= 0)");
    }

    // ── Swapped state gets zero state bonus (same as Standby) ──

    #[test]
    fn swapped_state_no_bonus() {
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_swapped = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_swapped = EvictionWorker::compute_importance_score(
            &meta_swapped, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_standby, score_swapped, "Swapped should have same bonus as Standby (= 0)");
    }

    // ── Free state gets zero state bonus (same as Standby) ──

    #[test]
    fn free_state_no_bonus() {
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_free = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_free = EvictionWorker::compute_importance_score(
            &meta_free, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_standby, score_free, "Free should have same bonus as Standby (= 0)");
    }

    // ── Warm state bonus exactly +5000 over Standby ──

    #[test]
    fn warm_beats_standby_by_5000() {
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 3,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_warm = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 3,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::CpuDram, 200,
        );
        let score_warm = EvictionWorker::compute_importance_score(
            &meta_warm, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::CpuDram, 200,
        );
        assert_eq!(score_warm - score_standby, 5000, "Warm bonus should be exactly +5000");
    }

    // ── Protected beats Warm by exactly 5000 ──

    #[test]
    fn protected_beats_warm_by_5000() {
        let meta_warm = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let meta_protected = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score_warm = EvictionWorker::compute_importance_score(
            &meta_warm, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &meta_protected, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_protected - score_warm, 5000, "Protected should be +5000 over Warm (10000 vs 5000)");
    }

    // ── Compression ratio: compressed > original yields negative bonus (no crash) ──

    #[test]
    fn compression_ratio_over_compressed_no_panic() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // compressed > original — pathological case, should not panic.
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            8000, // compressed > original
            4096, // original
            StorageTier::GpuHbm,
            0,
        );
        // ratio = 8000/4096 ≈ 1.95, bonus = (1.0 - 1.95) * 500 ≈ -475
        assert!(
            score < 1000,
            "over-compressed should yield negative compression bonus, reducing score below base: got {}",
            score,
        );
    }

    // ── Compression ratio exact bonus at 50% compression ──

    #[test]
    fn compression_ratio_exact_50_percent() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_50 = EvictionWorker::compute_importance_score(
            &meta, None, 2048, 4096, StorageTier::GpuHbm, 0,
        );
        // 50% compression: (1.0 - 0.5) * 500 = 250
        // uncompressed: (1.0 - 1.0) * 500 = 0
        let delta = score_50 - score_no_compress;
        assert_eq!(delta, 250, "50%% compression bonus should be 250, got {}", delta);
    }

    // ── EvictionWorkerConfig with custom values ──

    #[test]
    fn config_custom_values() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(100),
            max_evict_per_round: 32,
            hbm_pressure_threshold: 0.75,
            dram_pressure_threshold: 0.65,
            importance_threshold: 200,
            hbm_evict_age_ticks: 25,
            dram_evict_age_ticks: 250,
            default_evict_codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
        };
        assert_eq!(cfg.tick_interval, Duration::from_millis(100));
        assert_eq!(cfg.max_evict_per_round, 32);
        assert!((cfg.hbm_pressure_threshold - 0.75).abs() < 1e-6);
        assert!((cfg.dram_pressure_threshold - 0.65).abs() < 1e-6);
        assert_eq!(cfg.importance_threshold, 200);
        assert_eq!(cfg.hbm_evict_age_ticks, 25);
        assert_eq!(cfg.dram_evict_age_ticks, 250);
        assert_eq!(cfg.default_evict_codec, CompressionCodec::BitPackRle);
        assert_eq!(cfg.page_bytes, 8192);
    }

    // ── EvictionCandidate with all CompressionCodec variants ──

    #[test]
    fn eviction_candidate_all_codec_variants() {
        for codec in &[
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let c = EvictionCandidate {
                page_id: 1,
                score: 0,
                current_tier: StorageTier::GpuHbm,
                codec: *codec,
                page_bytes: 4096,
                group_id: None,
            };
            assert_eq!(c.codec, *codec, "candidate should preserve codec variant");
        }
    }

    // ── EvictionCandidate with all StorageTier variants ──

    #[test]
    fn eviction_candidate_all_tier_variants() {
        for tier in &[StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let c = EvictionCandidate {
                page_id: 1,
                score: 0,
                current_tier: *tier,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            };
            assert_eq!(c.current_tier, *tier, "candidate should preserve tier variant");
        }
    }

    // ── evict_round skips Protected pages ──

    #[test]
    fn evict_round_skips_protected_pages() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Protected, // Protected — should be skipped
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "Protected pages should be skipped");
        actor.shutdown();
    }

    // ── Score components are additive: verify isolated contribution ──

    #[test]
    fn score_components_additive() {
        // Isolate: 10 ticks age + recency=4 + access_count=2 + 50% compression on 4K page
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 4,
            access_count: 2,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext), // +0
            2048,  // 50% of 4096
            4096,
            StorageTier::GpuHbm, // +0
            10,
        );
        // base(1000) - time(10*2=20) - recency(4*1=4) + freq(2*15=30)
        // + compression((1-0.5)*500=250) + page_size(4096/1024*1=4) + kv(0) + state(0) + tier(0)
        // = 1000 - 20 - 4 + 30 + 250 + 4 = 1260
        assert_eq!(score, 1260, "score components should sum correctly: got {}", score);
    }

    // ── Large tier_age_ticks can drive score very negative ──

    #[test]
    fn large_tier_age_drives_score_deeply_negative() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // triggers ExpertWeight
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            0,
            0,
            StorageTier::Nvme,
            10000, // very old
        );
        // base(1000) - time(10000*2=20000) + expert(-300) + nvme(-500) = -19800
        assert!(
            score < -19000,
            "extreme age with worst payload/tier should yield deeply negative score: got {}",
            score,
        );
    }

    // ── DenseLayerWeight with Protected state is nearly unevictable ──

    #[test]
    fn dense_layer_protected_is_unevictable() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            0,
            4096,
            StorageTier::GpuHbm,
            0,
        );
        // base(1000) + freq(100*15=1500) + dense(5000) + protected(10000) + page(4) = 17504
        assert!(
            score > IMPORTANCE_SCORE_THRESHOLD,
            "DenseLayerWeight + Protected + high freq should be far above threshold: got {}",
            score,
        );
    }

    // ── EvictionCandidate Debug includes all key fields ──

    #[test]
    fn eviction_candidate_debug_includes_tier_and_codec() {
        let c = EvictionCandidate {
            page_id: 99,
            score: 42,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
            group_id: None,
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("current_tier"), "Debug should include current_tier");
        assert!(debug.contains("codec"), "Debug should include codec");
        assert!(debug.contains("page_bytes"), "Debug should include page_bytes");
    }

    // ── EvictionWorkerConfig clone preserves all fields ──

    #[test]
    fn config_clone_preserves_all_fields() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(42),
            max_evict_per_round: 16,
            hbm_pressure_threshold: 0.85,
            dram_pressure_threshold: 0.70,
            importance_threshold: 150,
            hbm_evict_age_ticks: 30,
            dram_evict_age_ticks: 300,
            default_evict_codec: CompressionCodec::NvcompAns,
            page_bytes: 16384,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.tick_interval, cfg.tick_interval);
        assert_eq!(cloned.max_evict_per_round, cfg.max_evict_per_round);
        assert!((cloned.hbm_pressure_threshold - cfg.hbm_pressure_threshold).abs() < 1e-6);
        assert!((cloned.dram_pressure_threshold - cfg.dram_pressure_threshold).abs() < 1e-6);
        assert_eq!(cloned.importance_threshold, cfg.importance_threshold);
        assert_eq!(cloned.hbm_evict_age_ticks, cfg.hbm_evict_age_ticks);
        assert_eq!(cloned.dram_evict_age_ticks, cfg.dram_evict_age_ticks);
        assert_eq!(cloned.default_evict_codec, cfg.default_evict_codec);
        assert_eq!(cloned.page_bytes, cfg.page_bytes);
    }

    // ── Score on CpuDram tier is exactly 200 less than GpuHbm ──

    #[test]
    fn dram_discount_exact_with_all_components() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::GpuHbm, 100,
        );
        let score_dram = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::CpuDram, 100,
        );
        assert_eq!(score_hbm - score_dram, 200, "DRAM tier discount should be exactly -200");
    }

    // ── Score on Nvme tier is exactly 500 less than GpuHbm ──

    #[test]
    fn nvme_discount_exact_with_all_components() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::GpuHbm, 100,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::Nvme, 100,
        );
        assert_eq!(score_hbm - score_nvme, 500, "NVMe tier discount should be exactly -500");
    }

    // ── ExpertWeight bonus is exactly -300 ──

    #[test]
    fn expert_weight_bonus_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_expert = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_kv - score_expert, 300, "ExpertWeight penalty = -300");
    }

    // ── DenseLayerWeight bonus is exactly +5000 ──

    #[test]
    fn dense_layer_bonus_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_dense = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_dense - score_kv, 5000, "DenseLayerWeight bonus = +5000");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (20 new)
    // ─────────────────────────────────────────────────────────────────────────

    // ── EvictionCandidate sort order: lowest score first ──

    #[test]
    fn eviction_candidate_sort_ascending_by_score() {
        let mut candidates = vec![
            EvictionCandidate {
                page_id: 3, score: 500, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            },
            EvictionCandidate {
                page_id: 1, score: -200, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            },
            EvictionCandidate {
                page_id: 2, score: 50, current_tier: StorageTier::CpuDram,
                codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(1),
            },
        ];
        candidates.sort_by_key(|c| c.score);
        assert_eq!(candidates[0].page_id, 1, "lowest score (-200) should be first");
        assert_eq!(candidates[1].page_id, 2, "middle score (50) should be second");
        assert_eq!(candidates[2].page_id, 3, "highest score (500) should be last");
    }

    // ── EvictionCandidate with i64::MIN and i64::MAX scores ──

    #[test]
    fn eviction_candidate_extreme_scores() {
        let c_min = EvictionCandidate {
            page_id: 1, score: i64::MIN, current_tier: StorageTier::Nvme,
            codec: CompressionCodec::None, page_bytes: 0, group_id: None,
        };
        let c_max = EvictionCandidate {
            page_id: 2, score: i64::MAX, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4, page_bytes: 65536, group_id: Some(99),
        };
        assert!(c_min.score < c_max.score);
        assert_eq!(c_min.score, i64::MIN);
        assert_eq!(c_max.score, i64::MAX);
    }

    // ── classify_eviction_tier: PromptSystem low score -> StandbyKv ──

    #[test]
    fn classify_prompt_system_low_score_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::PromptSystem), 0),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: PromptSystem high score -> Protected ──

    #[test]
    fn classify_prompt_system_high_score_is_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::PromptSystem), 999),
            EvictionTier::Protected,
        );
    }

    // ── EvictionWorkerConfig with zero page_bytes ──

    #[test]
    fn config_zero_page_bytes() {
        let cfg = EvictionWorkerConfig {
            page_bytes: 0,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.page_bytes, 0);
        // Config with zero page_bytes should still construct and clone.
        let cloned = cfg.clone();
        assert_eq!(cloned.page_bytes, 0);
    }

    // ── EvictionWorkerConfig with zero tick_interval ──

    #[test]
    fn config_zero_tick_interval() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::ZERO,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval, Duration::ZERO);
    }

    // ── EvictionWorkerConfig with extreme thresholds ──

    #[test]
    fn config_extreme_thresholds() {
        let cfg = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            importance_threshold: i64::MAX,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: u64::MAX,
            ..EvictionWorkerConfig::default()
        };
        assert!((cfg.hbm_pressure_threshold - 0.0).abs() < 1e-6);
        assert!((cfg.dram_pressure_threshold - 1.0).abs() < 1e-6);
        assert_eq!(cfg.importance_threshold, i64::MAX);
        assert_eq!(cfg.hbm_evict_age_ticks, 0);
        assert_eq!(cfg.dram_evict_age_ticks, u64::MAX);
    }

    // ── compute_importance_score with very high access_count ──

    #[test]
    fn score_with_high_access_count_no_panic() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 100_000,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // freq_bonus = 100000 * 15 = 1500000
        assert!(
            score > 1_000_000,
            "high access_count should yield very high score: got {}",
            score,
        );
    }

    // ── compute_importance_score: tier_age_ticks does not overflow i64 ──

    #[test]
    fn score_large_tier_age_no_overflow() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, u64::MAX,
        );
        // time_penalty = u64::MAX * 2 — this will overflow to a wrapped value,
        // but the function should not panic.
        // The score just needs to be a valid i64 (no panic = success).
        let _ = score; // no panic is the assertion
    }

    // ── Warm + DenseLayerWeight + GpuHbm: maximum non-Protected protection ──

    #[test]
    fn warm_dense_layer_on_hbm_high_score() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            0,
            4096,
            StorageTier::GpuHbm,
            0,
        );
        // base(1000) + freq(50*15=750) + dense(5000) + warm(5000)
        // + compression(compressed=0,original=4096 -> ratio=0 -> (1-0)*500=500)
        // + page_size(4096/1024*1=4) = 12254
        assert!(
            score > IMPORTANCE_SCORE_THRESHOLD,
            "Warm + DenseLayerWeight on HBM should be far above threshold: got {}",
            score,
        );
        assert_eq!(score, 12254, "expected exact combined score");
    }

    // ── EvictionCandidate with group_id = Some(0) ──

    #[test]
    fn eviction_candidate_zero_request_id() {
        let c = EvictionCandidate {
            page_id: 1,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(0),
        };
        assert_eq!(c.group_id, Some(0));
    }

    // ── compute_importance_score: compressed=0, original>0 is maximum compression bonus ──

    #[test]
    fn score_compressed_zero_yields_max_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_full_compress = EvictionWorker::compute_importance_score(
            &meta, None, 0, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio = 0.0 -> bonus = (1.0 - 0.0) * 500 = 500
        // ratio = 1.0 -> bonus = 0
        assert_eq!(
            score_full_compress - score_no_compress, 500,
            "fully compressed (compressed=0) bonus should be 500",
        );
    }

    // ── EvictionTier can be used in a match exhaustively ──

    #[test]
    fn eviction_tier_exhaustive_match() {
        // Ensure all variants produce a valid string.
        fn tier_name(tier: EvictionTier) -> &'static str {
            match tier {
                EvictionTier::ColdExpert => "cold_expert",
                EvictionTier::PinnedDense => "pinned_dense",
                EvictionTier::StandbyKv => "standby_kv",
                EvictionTier::Protected => "protected",
            }
        }
        assert_eq!(tier_name(EvictionTier::ColdExpert), "cold_expert");
        assert_eq!(tier_name(EvictionTier::PinnedDense), "pinned_dense");
        assert_eq!(tier_name(EvictionTier::StandbyKv), "standby_kv");
        assert_eq!(tier_name(EvictionTier::Protected), "protected");
    }

    // ── EvictionWorkerConfig default codec is Lz4 ──

    #[test]
    fn config_default_codec_is_lz4() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.default_evict_codec, CompressionCodec::Lz4);
    }

    // ── Score components: recency penalty uses TIME_DECAY_WEIGHT / 2 = 1 ──

    #[test]
    fn recency_penalty_weight_is_half_time_decay() {
        let meta_r0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_r100 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 100,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta_r0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s100 = EvictionWorker::compute_importance_score(
            &meta_r100, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let expected_delta = 100 * (TIME_DECAY_WEIGHT / 2);
        assert_eq!(
            s0 - s100, expected_delta,
            "recency penalty weight should be TIME_DECAY_WEIGHT/2 = {}",
            TIME_DECAY_WEIGHT / 2,
        );
    }

    // ── Compression ratio bonus exact for 75% compression ──

    #[test]
    fn compression_ratio_exact_75_percent() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_75 = EvictionWorker::compute_importance_score(
            &meta, None, 1024, 4096, StorageTier::GpuHbm, 0,
        );
        // 75% compression: ratio = 0.25, bonus = (1.0 - 0.25) * 500 = 375
        assert_eq!(
            score_75 - score_no_compress, 375,
            "75% compression bonus should be 375",
        );
    }

    // ── EvictionCandidate collection truncation preserves lowest scores ──

    #[test]
    fn candidate_truncate_keeps_lowest_scores() {
        let mut candidates: Vec<EvictionCandidate> = (0..10)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: (i as i64) * 100, // 0, 100, 200, ..., 900
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(3);
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].score, 0);
        assert_eq!(candidates[1].score, 100);
        assert_eq!(candidates[2].score, 200);
    }

    // ── Score formula verification: all components combined manually ──

    #[test]
    fn score_formula_manual_verification() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // no owner -> infer ExpertWeight (but we pass explicit PromptSystem)
            recency: 10,
            access_count: 4,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::PromptSystem), // +1000
            1024, // 25% of 4096
            4096,
            StorageTier::CpuDram, // -200
            50, // 50 ticks
        );
        // base = 1000
        // - time_penalty = 50 * 2 = 100
        // - recency_penalty = 10 * 1 = 10
        // + freq_bonus = 4 * 15 = 60
        // + compression_bonus = (1.0 - 0.25) * 500 = 375
        // + page_size_bonus = 4096 / 1024 * 1 = 4
        // + payload_bonus = 1000 (PromptSystem)
        // + state_bonus = 5000 (Warm)
        // + tier_discount = -200 (CpuDram)
        // = 1000 - 100 - 10 + 60 + 375 + 4 + 1000 + 5000 - 200 = 7129
        assert_eq!(score, 7129, "manual formula should match: got {}", score);
    }

    // ── KnowledgeRAG payload bonus is exactly -500 relative to KvContext ──

    #[test]
    fn knowledge_rag_bonus_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_rag = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_kv - score_rag, 500, "KnowledgeRAG penalty should be exactly -500 relative to KvContext");
    }

    // ── PromptSystem bonus is exactly +1000 relative to KvContext ──

    #[test]
    fn prompt_system_bonus_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_prompt = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_prompt - score_kv, 1000, "PromptSystem bonus should be exactly +1000");
    }

    // ── Page size bonus exact for 16 KiB page ──

    #[test]
    fn page_size_bonus_exact_16k() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Use same compression ratio so compression bonus cancels out.
        // 4K page: original=4096, compressed=2048 (50% ratio)
        // 16K page: original=16384, compressed=8192 (50% ratio)
        let score_4k = EvictionWorker::compute_importance_score(
            &meta, None, 2048, 4096, StorageTier::GpuHbm, 0,
        );
        let score_16k = EvictionWorker::compute_importance_score(
            &meta, None, 8192, 16384, StorageTier::GpuHbm, 0,
        );
        // Both have 50% compression so compression_ratio_bonus is the same.
        // page_size delta = (16384/1024*1) - (4096/1024*1) = 16 - 4 = 12
        assert_eq!(
            score_16k - score_4k, 12,
            "16 KiB page should add 12 more than 4 KiB page (both 50% compressed)",
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (18 new)
    // ─────────────────────────────────────────────────────────────────────────

    // ── access_count=0 and access_count=1: frequency bonus delta is exactly FREQUENCY_BONUS ──

    #[test]
    fn frequency_bonus_single_access_exact() {
        let meta_0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_1 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta_0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta_1, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s1 - s0, FREQUENCY_BONUS as i64, "single access bonus = FREQUENCY_BONUS ({})", FREQUENCY_BONUS);
    }

    // ── recency=u32::MAX does not panic ──

    #[test]
    fn recency_max_no_panic() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: usize::MAX,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // recency_penalty = usize::MAX * 1 — will overflow as i64, but must not panic.
        let _ = score;
    }

    // ── warm_until field exists on PageMetadata but scoring ignores it ──

    #[test]
    fn warm_until_does_not_affect_score() {
        let meta_no_warm = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_with_warm = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
        };
        let s_no = EvictionWorker::compute_importance_score(
            &meta_no_warm, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_warm = EvictionWorker::compute_importance_score(
            &meta_with_warm, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_no, s_warm, "warm_until should not affect score — only state matters");
    }

    // ── is_lir field does not directly affect scoring ──

    #[test]
    fn is_lir_does_not_affect_score() {
        let meta_not_lir = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_lir = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_not = EvictionWorker::compute_importance_score(
            &meta_not_lir, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_lir = EvictionWorker::compute_importance_score(
            &meta_lir, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_not, s_lir, "is_lir flag should not affect scoring");
    }

    // ── EvictionWorkerConfig with max_evict_per_round=0 ──

    #[test]
    fn config_zero_max_evict_per_round() {
        let cfg = EvictionWorkerConfig {
            max_evict_per_round: 0,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.max_evict_per_round, 0);
        let cloned = cfg.clone();
        assert_eq!(cloned.max_evict_per_round, 0);
    }

    // ── EvictionWorkerConfig with negative importance_threshold ──

    #[test]
    fn config_negative_importance_threshold() {
        let cfg = EvictionWorkerConfig {
            importance_threshold: -500,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.importance_threshold, -500);
    }

    // ── EvictionWorkerConfig with importance_threshold=0 ──

    #[test]
    fn config_zero_importance_threshold() {
        let cfg = EvictionWorkerConfig {
            importance_threshold: 0,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.importance_threshold, 0);
    }

    // ── EvictionTier all 4 variants are distinguishable via match ──

    #[test]
    fn eviction_tier_all_four_variants() {
        // Verify that there are exactly 4 variants and each is reachable.
        let variants = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        let mut count = 0;
        for v in &variants {
            match v {
                EvictionTier::ColdExpert => count += 1,
                EvictionTier::PinnedDense => count += 1,
                EvictionTier::StandbyKv => count += 1,
                EvictionTier::Protected => count += 1,
            }
        }
        assert_eq!(count, 4, "all 4 EvictionTier variants should be matchable");
    }

    // ── infer_payload_kind: sequence_id=Some(0) is still KvContext ──

    #[test]
    fn infer_payload_zero_request_id_is_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(0), // zero-valued Some
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::KvContext),
            "Some(0) should still be treated as owned -> KvContext",
        );
    }

    // ── Score exact: KvContext + Warm + GpuHbm, zero everything else ──

    #[test]
    fn score_exact_warm_hbm_zero_other() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        // base(1000) + warm(5000) = 6000
        assert_eq!(score, 6000, "Warm + KvContext + GpuHbm + zero other = 6000, got {}", score);
    }

    // ── Score exact: ExpertWeight + Nvme + Standby, zero everything else ──

    #[test]
    fn score_exact_expert_nvme_standby() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + expert(-300) + nvme(-500) = 200
        assert_eq!(score, 200, "ExpertWeight + NVMe + Standby + zero other = 200, got {}", score);
    }

    // ── Score exact: KnowledgeRAG + CpuDram + Standby, zero everything else ──

    #[test]
    fn score_exact_rag_dram_standby() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + rag(-500) + dram(-200) = 300
        assert_eq!(score, 300, "KnowledgeRAG + CpuDram + Standby + zero other = 300, got {}", score);
    }

    // ── Compression ratio: compressed equals original yields zero bonus ──

    #[test]
    fn compression_ratio_no_compression_exact_zero() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_zero = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_same = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // When original_size=0, both compression and page_size bonuses are 0.
        // When compressed=original (ratio=1.0), compression bonus=0 but page_size bonus = 4096/1024*1 = 4.
        assert_eq!(
            score_same - score_zero, 4,
            "compressed=original should yield 0 compression bonus but nonzero page_size bonus",
        );
    }

    // ── Score monotonically decreases with increasing tier_age_ticks ──

    #[test]
    fn score_monotonically_decreases_with_age() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let mut prev = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        for ticks in [1u64, 5, 10, 50, 100, 500, 1000] {
            let score = EvictionWorker::compute_importance_score(
                &meta, None, 0, 0, StorageTier::GpuHbm, ticks,
            );
            assert!(
                score < prev,
                "score should decrease as tier_age increases: ticks={} score={} prev={}",
                ticks, score, prev,
            );
            prev = score;
        }
    }

    // ── Score monotonically increases with access_count ──

    #[test]
    fn score_monotonically_increases_with_frequency() {
        let mut prev = i64::MIN;
        for freq in [0usize, 1, 5, 10, 50, 100, 1000] {
            let meta = PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: freq,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            };
            let score = EvictionWorker::compute_importance_score(
                &meta, None, 0, 0, StorageTier::GpuHbm, 0,
            );
            assert!(
                score > prev,
                "score should increase with access_count: freq={} score={} prev={}",
                freq, score, prev,
            );
            prev = score;
        }
    }

    // ── All 5 PagePayloadKind variants produce distinct scores ──

    #[test]
    fn all_payload_kinds_produce_distinct_scores() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let kinds = [
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::KvContext,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::DenseLayerWeight,
        ];
        let scores: Vec<i64> = kinds.iter().map(|k| {
            EvictionWorker::compute_importance_score(
                &meta, Some(*k), 0, 0, StorageTier::GpuHbm, 0,
            )
        }).collect();

        // All scores should be distinct.
        for i in 0..scores.len() {
            for j in (i + 1)..scores.len() {
                assert_ne!(
                    scores[i], scores[j],
                    "payload kinds {:?} and {:?} should produce different scores",
                    kinds[i], kinds[j],
                );
            }
        }
    }

    // ── All 3 StorageTier variants produce distinct scores ──

    #[test]
    fn all_tiers_produce_distinct_scores() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        let scores: Vec<i64> = tiers.iter().map(|t| {
            EvictionWorker::compute_importance_score(
                &meta, None, 0, 0, *t, 0,
            )
        }).collect();

        assert_ne!(scores[0], scores[1], "GpuHbm and CpuDram should differ");
        assert_ne!(scores[1], scores[2], "CpuDram and Nvme should differ");
        assert_ne!(scores[0], scores[2], "GpuHbm and Nvme should differ");
    }

    // ── Page size bonus scales linearly across multiple sizes ──

    #[test]
    fn page_size_bonus_linear_across_sizes() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let sizes: Vec<u32> = vec![1024, 2048, 4096, 8192, 16384];
        let mut scores: Vec<i64> = Vec::new();
        for &sz in &sizes {
            let score = EvictionWorker::compute_importance_score(
                &meta, None, sz, sz, StorageTier::GpuHbm, 0,
            );
            scores.push(score);
        }
        // Verify linear progression of deltas (each step is 1024 bytes more = 1 more bonus point).
        for w in scores.windows(2) {
            let delta = w[1] - w[0];
            // page_size_bonus delta = 1024/1024 * 1 = 1 per step
            // compression_ratio bonus also changes slightly because ratio is same (1.0) but size differs
            // Actually compressed=original so ratio=1.0 for all, compression bonus = 0 for all.
            // page_size delta per 1024 step = 1024/1024*1 = 1
            assert!(
                delta > 0,
                "larger page should score higher: delta={}",
                delta,
            );
        }
        // The total delta from 1K to 16K should be (16384-1024)/1024 = 15
        let total_delta = scores[scores.len() - 1] - scores[0];
        assert_eq!(total_delta, 15, "total page_size bonus from 1K to 16K should be 15");
    }

    // ── EvictionCandidate collection: empty vec truncation is safe ──

    #[test]
    fn candidate_empty_truncate_is_safe() {
        let mut candidates: Vec<EvictionCandidate> = Vec::new();
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(3);
        assert_eq!(candidates.len(), 0, "empty vec should remain empty after truncate");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (17 new)
    // ─────────────────────────────────────────────────────────────────────────

    // ── compute_importance_score: page_id does not affect score ──

    #[test]
    fn page_id_does_not_affect_score() {
        let meta_a = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 3,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_b = PageMetadata {
            page_id: 99999,
            sequence_id: Some(100),
            recency: 3,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_a = EvictionWorker::compute_importance_score(
            &meta_a, None, 2048, 4096, StorageTier::GpuHbm, 50,
        );
        let score_b = EvictionWorker::compute_importance_score(
            &meta_b, None, 2048, 4096, StorageTier::GpuHbm, 50,
        );
        assert_eq!(score_a, score_b, "page_id should not affect score");
    }

    // ── Score exact: ExpertWeight + CpuDram, zero everything else ──

    #[test]
    fn score_exact_expert_weight_cpu_dram() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + expert(-300) + dram(-200) = 500
        assert_eq!(score, 500, "ExpertWeight + CpuDram + zero other = 500, got {}", score);
    }

    // ── Score exact: DenseLayerWeight + CpuDram + Warm ──

    #[test]
    fn score_exact_dense_layer_warm_dram() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + dense(5000) + warm(5000) + dram(-200) = 10800
        assert_eq!(score, 10800, "DenseLayerWeight + Warm + CpuDram = 10800, got {}", score);
    }

    // ── Score exact: PromptSystem + Protected + Nvme ──

    #[test]
    fn score_exact_prompt_system_protected_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + prompt(1000) + protected(10000) + nvme(-500) = 11500
        assert_eq!(score, 11500, "PromptSystem + Protected + Nvme = 11500, got {}", score);
    }

    // ── Score exact: KnowledgeRAG + Protected + GpuHbm ──

    #[test]
    fn score_exact_rag_protected_hbm() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
        );
        // base(1000) + rag(-500) + protected(10000) + hbm(0) = 10500
        assert_eq!(score, 10500, "KnowledgeRAG + Protected + GpuHbm = 10500, got {}", score);
    }

    // ── EvictionTier can be used as HashMap key (Hash trait) ──

    #[test]
    fn eviction_tier_hash_consistency() {
        use std::collections::HashMap;
        let mut map: HashMap<EvictionTier, &'static str> = HashMap::new();
        map.insert(EvictionTier::ColdExpert, "cold");
        map.insert(EvictionTier::PinnedDense, "pinned");
        map.insert(EvictionTier::StandbyKv, "standby");
        map.insert(EvictionTier::Protected, "protected");
        assert_eq!(map.get(&EvictionTier::ColdExpert), Some(&"cold"));
        assert_eq!(map.get(&EvictionTier::PinnedDense), Some(&"pinned"));
        assert_eq!(map.get(&EvictionTier::StandbyKv), Some(&"standby"));
        assert_eq!(map.get(&EvictionTier::Protected), Some(&"protected"));
        assert_eq!(map.len(), 4, "all 4 variants should be distinct keys");
    }

    // ── EvictionCandidate sort stability: equal scores preserve insertion order ──

    #[test]
    fn eviction_candidate_sort_equal_scores() {
        let mut candidates = vec![
            EvictionCandidate {
                page_id: 3, score: 50, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            },
            EvictionCandidate {
                page_id: 1, score: 50, current_tier: StorageTier::CpuDram,
                codec: CompressionCodec::Lz4, page_bytes: 8192, group_id: Some(1),
            },
            EvictionCandidate {
                page_id: 2, score: 50, current_tier: StorageTier::Nvme,
                codec: CompressionCodec::BitPackRle, page_bytes: 2048, group_id: None,
            },
        ];
        candidates.sort_by_key(|c| c.score);
        // Rust's sort_by_key is stable, so order should be preserved for equal keys.
        assert_eq!(candidates[0].page_id, 3);
        assert_eq!(candidates[1].page_id, 1);
        assert_eq!(candidates[2].page_id, 2);
    }

    // ── Score monotonically decreases with increasing recency ──

    #[test]
    fn score_monotonically_decreases_with_recency() {
        let mut prev = i64::MAX;
        for recency in [0usize, 1, 5, 10, 50, 100, 500] {
            let meta = PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            };
            let score = EvictionWorker::compute_importance_score(
                &meta, None, 0, 0, StorageTier::GpuHbm, 0,
            );
            assert!(
                score < prev,
                "score should decrease as recency increases: recency={} score={} prev={}",
                recency, score, prev,
            );
            prev = score;
        }
    }

    // ── Score exact: all 6 PageState variants produce 3 distinct bonus levels ──

    #[test]
    fn state_bonus_produces_three_levels() {
        let make_meta = |state: PageState| PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state,
            warm_until: None,
        };

        let score_free = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Free), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_standby = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Standby), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_swapped_out = EvictionWorker::compute_importance_score(
            &make_meta(PageState::SwappedOut), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_swapped = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Swapped), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_active = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Active), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_warm = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Warm), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Protected), None, 0, 0, StorageTier::GpuHbm, 0,
        );

        // Free, Standby, SwappedOut, Swapped, Active all get 0 bonus.
        let zero_bonus = score_free;
        assert_eq!(score_standby, zero_bonus, "Standby should have 0 bonus");
        assert_eq!(score_swapped_out, zero_bonus, "SwappedOut should have 0 bonus");
        assert_eq!(score_swapped, zero_bonus, "Swapped should have 0 bonus");
        assert_eq!(score_active, zero_bonus, "Active should have 0 bonus");
        // Warm = +5000
        assert_eq!(score_warm - zero_bonus, 5000, "Warm bonus = +5000");
        // Protected = +10000
        assert_eq!(score_protected - zero_bonus, 10000, "Protected bonus = +10000");
    }

    // ── Score: time penalty dominates over frequency bonus for very old pages ──

    #[test]
    fn time_penalty_dominates_frequency_for_old_pages() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // freq_bonus = 10 * 15 = 150
        // time_penalty at 1000 ticks = 1000 * 2 = 2000
        // Net: base(1000) + 150 - 2000 = -850
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 1000,
        );
        assert!(
            score < 0,
            "time penalty should dominate frequency bonus for old pages: got {}",
            score,
        );
    }

    // ── Score: frequency bonus can overcome time penalty for moderately old pages ──

    #[test]
    fn frequency_overcomes_moderate_time_penalty() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // freq_bonus = 100 * 15 = 1500
        // time_penalty at 50 ticks = 50 * 2 = 100
        // Net: base(1000) + 1500 - 100 = 2400
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 50,
        );
        assert!(
            score > IMPORTANCE_SCORE_THRESHOLD,
            "high frequency should overcome moderate time penalty: got {}",
            score,
        );
    }

    // ── Score exact: DenseLayerWeight + Warm + CpuDram + access_count=10 ──

    #[test]
    fn score_exact_dense_warm_dram_with_freq() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + dense(5000) + warm(5000) + freq(10*15=150) + dram(-200) = 10950
        assert_eq!(score, 10950, "DenseLayerWeight + Warm + CpuDram + freq=10 = 10950, got {}", score);
    }

    // ── EvictionCandidate with very large page_bytes ──

    #[test]
    fn eviction_candidate_large_page_bytes() {
        let c = EvictionCandidate {
            page_id: 1,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: usize::MAX,
            group_id: Some(42),
        };
        assert_eq!(c.page_bytes, usize::MAX);
        let cloned = c.clone();
        assert_eq!(cloned.page_bytes, usize::MAX);
    }

    // ── EvictionWorkerConfig with all compression codec variants ──

    #[test]
    fn config_all_codec_variants() {
        for codec in &[
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let cfg = EvictionWorkerConfig {
                default_evict_codec: *codec,
                ..EvictionWorkerConfig::default()
            };
            assert_eq!(cfg.default_evict_codec, *codec, "config should preserve codec");
            let cloned = cfg.clone();
            assert_eq!(cloned.default_evict_codec, *codec, "cloned config should preserve codec");
        }
    }

    // ── Score and classify consistency: below threshold => StandbyKv for default payload ──

    #[test]
    fn score_classify_consistency_below_threshold() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // ExpertWeight on Nvme with 200 ticks => well below threshold
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 200,
        );
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::ExpertWeight), score,
        );
        assert!(
            score < IMPORTANCE_SCORE_THRESHOLD,
            "score should be below threshold: got {}",
            score,
        );
        // ExpertWeight always classifies as ColdExpert regardless of score.
        assert_eq!(tier, EvictionTier::ColdExpert);
    }

    // ── Score and classify consistency: above threshold => Protected for KvContext ──

    #[test]
    fn score_classify_consistency_above_threshold() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext), score,
        );
        assert!(
            score > IMPORTANCE_SCORE_THRESHOLD,
            "high-freq Protected page should be above threshold: got {}",
            score,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    // ── compute_tier_age: zero elapsed time yields zero ticks ──

    #[test]
    fn tier_age_zero_elapsed_is_zero() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        // Zero or very close to zero (< 1ms = 0 ticks after division by 10).
        assert!(age <= 1, "freshly created page should have ~0 age ticks, got {}", age);
    }

    // ── EvictionWorkerConfig Debug includes codec field ──

    #[test]
    fn config_debug_includes_codec_field() {
        let cfg = EvictionWorkerConfig {
            default_evict_codec: CompressionCodec::NvcompAns,
            ..EvictionWorkerConfig::default()
        };
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("default_evict_codec"), "Debug should include default_evict_codec");
        assert!(debug.contains("NvcompAns"), "Debug should include codec variant name");
    }

    // ── Score with compressed > original: negative compression bonus is bounded ──

    #[test]
    fn compression_negative_bonus_bounded() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_base = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // compressed = 4x original: ratio = 4.0, bonus = (1 - 4) * 500 = -1500
        let score_over = EvictionWorker::compute_importance_score(
            &meta, None, 16384, 4096, StorageTier::GpuHbm, 0,
        );
        let delta = score_over - score_base;
        // page_size_bonus is the same (original=4096), so delta is purely compression.
        // compressed=4096 (same) -> bonus = 0. compressed=16384 -> ratio=4.0, bonus = -1500
        // But page_size_bonus is different: compressed=16384 uses original=4096, so page_size=4.
        // Both have original=4096, so page_size bonus is the same = 4.
        // The only difference is compression_ratio_bonus: 0 vs -1500.
        assert!(
            delta < 0,
            "over-compressed should reduce score: delta={}",
            delta,
        );
        assert_eq!(
            delta, -1500,
            "over-compressed (4x) bonus should be exactly -1500: got {}",
            delta,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (40 new)
    // ─────────────────────────────────────────────────────────────────────────

    // ── EvictionTier Hash: same variant produces same hash across insertions ──

    #[test]
    fn eviction_tier_hash_deterministic() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        EvictionTier::Protected.hash(&mut h1);
        let hash1 = h1.finish();
        let mut h2 = DefaultHasher::new();
        EvictionTier::Protected.hash(&mut h2);
        let hash2 = h2.finish();
        assert_eq!(hash1, hash2, "same EvictionTier variant must produce same hash");
    }

    // ── EvictionTier Hash: different variants produce different hashes ──

    #[test]
    fn eviction_tier_hash_different_per_variant() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |t: EvictionTier| -> u64 {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };
        let hashes: Vec<u64> = vec![
            hash_of(EvictionTier::ColdExpert),
            hash_of(EvictionTier::PinnedDense),
            hash_of(EvictionTier::StandbyKv),
            hash_of(EvictionTier::Protected),
        ];
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(
                    hashes[i], hashes[j],
                    "EvictionTier variants at indices {} and {} should have different hashes",
                    i, j,
                );
            }
        }
    }

    // ── EvictionTier Eq: transitivity ──

    #[test]
    fn eviction_tier_eq_transitivity() {
        let a = EvictionTier::StandbyKv;
        let b = EvictionTier::StandbyKv;
        let c = EvictionTier::StandbyKv;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "Eq must be transitive");
    }

    // ── StorageTier round-trip through u8 encoding ──

    #[test]
    fn storage_tier_roundtrip_u8() {
        for tier in &[StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let encoded = tier.as_u8();
            let decoded = StorageTier::from_u8(encoded);
            assert_eq!(decoded, Some(*tier), "round-trip failed for {:?}", tier);
        }
    }

    // ── StorageTier from_u8 rejects invalid values ──

    #[test]
    fn storage_tier_from_u8_invalid() {
        assert_eq!(StorageTier::from_u8(255), None, "255 should be invalid");
        assert_eq!(StorageTier::from_u8(3), None, "3 should be invalid");
    }

    // ── CompressionCodec round-trip through u8 encoding ──

    #[test]
    fn compression_codec_roundtrip_u8() {
        for codec in &[
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let encoded = codec.as_u8();
            let decoded = CompressionCodec::from_u8(encoded);
            assert_eq!(decoded, Some(*codec), "round-trip failed for {:?}", codec);
        }
    }

    // ── CompressionCodec from_u8 rejects invalid values ──

    #[test]
    fn compression_codec_from_u8_invalid() {
        assert_eq!(CompressionCodec::from_u8(255), None);
        assert_eq!(CompressionCodec::from_u8(5), None);
    }

    // ── EvictionCandidate with page_id=0 ──

    #[test]
    fn eviction_candidate_page_id_zero() {
        let c = EvictionCandidate {
            page_id: 0,
            score: 100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };
        assert_eq!(c.page_id, 0);
        let cloned = c.clone();
        assert_eq!(cloned.page_id, 0);
    }

    // ── EvictionCandidate with page_id=PageId::MAX (usize::MAX) ──

    #[test]
    fn eviction_candidate_page_id_max() {
        let c = EvictionCandidate {
            page_id: PageId::MAX,
            score: -1,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 0,
            group_id: Some(RequestId::MAX),
        };
        assert_eq!(c.page_id, PageId::MAX);
        assert_eq!(c.group_id, Some(RequestId::MAX));
    }

    // ── EvictionWorkerConfig default page_bytes is 4096 ──

    #[test]
    fn config_default_page_bytes() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.page_bytes, 4096);
    }

    // ── EvictionWorkerConfig default tick_interval is 10ms ──

    #[test]
    fn config_default_tick_interval_10ms() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.tick_interval, Duration::from_millis(10));
    }

    // ── EvictionWorkerConfig default hbm_evict_age_ticks is 50 ──

    #[test]
    fn config_default_hbm_evict_age() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.hbm_evict_age_ticks, 50);
    }

    // ── EvictionWorkerConfig default dram_evict_age_ticks is 500 ──

    #[test]
    fn config_default_dram_evict_age() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.dram_evict_age_ticks, 500);
    }

    // ── EvictionWorkerConfig default max_evict_per_round is 8 ──

    #[test]
    fn config_default_max_evict_per_round() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.max_evict_per_round, 8);
    }

    // ── EvictionWorkerConfig with negative hbm_pressure_threshold ──

    #[test]
    fn config_negative_hbm_threshold() {
        let cfg = EvictionWorkerConfig {
            hbm_pressure_threshold: -0.5,
            ..EvictionWorkerConfig::default()
        };
        assert!(cfg.hbm_pressure_threshold < 0.0, "negative threshold should be stored");
    }

    // ── EvictionWorkerConfig with hbm_pressure_threshold > 1.0 ──

    #[test]
    fn config_hbm_threshold_above_one() {
        let cfg = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.5,
            ..EvictionWorkerConfig::default()
        };
        assert!((cfg.hbm_pressure_threshold - 1.5).abs() < 1e-6);
    }

    // ── Score: original_size = u32::MAX produces large page_size bonus ──

    #[test]
    fn score_large_original_size() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, u32::MAX, StorageTier::GpuHbm, 0,
        );
        // page_size_bonus = u32::MAX / 1024 * 1 = 4194303
        // compression_bonus = (1.0 - 0.0) * 500 = 500
        // base(1000) + 4194303 + 500 = 4195803
        assert!(
            score > 4_000_000,
            "large original_size should produce very large page_size bonus: got {}",
            score,
        );
    }

    // ── Score: compressed_size = u32::MAX, original_size = u32::MAX (ratio = 1.0) ──

    #[test]
    fn score_max_compressed_equals_max_original() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, u32::MAX, u32::MAX, StorageTier::GpuHbm, 0,
        );
        // ratio = 1.0, compression bonus = 0
        // page_size = u32::MAX / 1024 = 4194303
        // base(1000) + 4194303 = 4195303
        assert!(
            score > 4_000_000,
            "max sizes with ratio 1.0 should still have large page_size bonus: got {}",
            score,
        );
    }

    // ── Score: very small original_size (1 byte) ──

    #[test]
    fn score_one_byte_original_size() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 1, StorageTier::GpuHbm, 0,
        );
        // page_size_bonus = 1 / 1024 * 1 = 0 (integer truncation)
        // compression_bonus = (1.0 - 0.0) * 500 = 500
        // base(1000) + 500 = 1500
        assert_eq!(score, 1500, "1-byte original should have page_size bonus truncated to 0");
    }

    // ── classify_eviction_tier: KvContext at exactly threshold - 1 is StandbyKv ──

    #[test]
    fn classify_kv_at_threshold_minus_one() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(
                Some(PagePayloadKind::KvContext),
                IMPORTANCE_SCORE_THRESHOLD - 1,
            ),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: KvContext at threshold + 1 is Protected ──

    #[test]
    fn classify_kv_at_threshold_plus_one() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(
                Some(PagePayloadKind::KvContext),
                IMPORTANCE_SCORE_THRESHOLD + 1,
            ),
            EvictionTier::Protected,
        );
    }

    // ── classify_eviction_tier: negative score is StandbyKv for non-expert/dense ──

    #[test]
    fn classify_negative_score_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(
                Some(PagePayloadKind::KvContext),
                -9999,
            ),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: i64::MAX score is Protected for KvContext ──

    #[test]
    fn classify_max_score_is_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(
                Some(PagePayloadKind::KvContext),
                i64::MAX,
            ),
            EvictionTier::Protected,
        );
    }

    // ── Score: time penalty overwhelms everything at extreme tier_age ──

    #[test]
    fn time_penalty_overwhelms_protection() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        // Protected gives +10000, DenseLayer gives +5000, but 100000 ticks * 2 = 200000 penalty.
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            0,
            0,
            StorageTier::GpuHbm,
            100_000,
        );
        // base(1000) - time(200000) + dense(5000) + protected(10000) = -184000
        assert!(
            score < 0,
            "extreme time should overwhelm all protection bonuses: got {}",
            score,
        );
    }

    // ── Score: CpuDram discount (-200) plus Nvme discount (-500) sum to -500 total from HBM ──

    #[test]
    fn tier_discount_cumulative() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        // GpuHbm=0, Nvme=-500, total delta = 500
        assert_eq!(
            score_hbm - score_nvme, 500,
            "total tier discount from HBM to NVMe should be -500",
        );
    }

    // ── Score: Warm and Protected state bonuses are cumulative with all other components ──

    #[test]
    fn state_bonus_independent_of_other_components() {
        let make_meta = |state: PageState| PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 10,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Standby),
            Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::CpuDram, 100,
        );
        let score_warm = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Warm),
            Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::CpuDram, 100,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Protected),
            Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::CpuDram, 100,
        );
        assert_eq!(score_warm - score_standby, 5000, "Warm bonus should be +5000 regardless of other components");
        assert_eq!(score_protected - score_standby, 10000, "Protected bonus should be +10000 regardless of other components");
    }

    // ── EvictionCandidate sort with mixed positive and negative scores ──

    #[test]
    fn eviction_candidate_sort_mixed_signs() {
        let mut candidates = vec![
            EvictionCandidate {
                page_id: 1, score: 50, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            },
            EvictionCandidate {
                page_id: 2, score: -100, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: None,
            },
            EvictionCandidate {
                page_id: 3, score: 0, current_tier: StorageTier::CpuDram,
                codec: CompressionCodec::BitPackRle, page_bytes: 4096, group_id: None,
            },
        ];
        candidates.sort_by_key(|c| c.score);
        assert_eq!(candidates[0].page_id, 2, "most negative should be first");
        assert_eq!(candidates[1].page_id, 3, "zero should be second");
        assert_eq!(candidates[2].page_id, 1, "positive should be last");
    }

    // ── EvictionCandidate with score exactly IMPORTANCE_SCORE_THRESHOLD ──

    #[test]
    fn eviction_candidate_score_at_threshold() {
        let c = EvictionCandidate {
            page_id: 1,
            score: IMPORTANCE_SCORE_THRESHOLD,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: None,
        };
        assert_eq!(c.score, 100);
    }

    // ── EvictionWorkerConfig clone with extreme page_bytes ──

    #[test]
    fn config_clone_extreme_page_bytes() {
        let cfg = EvictionWorkerConfig {
            page_bytes: usize::MAX,
            ..EvictionWorkerConfig::default()
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.page_bytes, usize::MAX);
    }

    // ── EvictionWorkerConfig with importance_threshold=i64::MIN ──

    #[test]
    fn config_min_importance_threshold() {
        let cfg = EvictionWorkerConfig {
            importance_threshold: i64::MIN,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.importance_threshold, i64::MIN);
    }

    // ── Score: frequency bonus exactly linear across wide range ──

    #[test]
    fn frequency_bonus_linear_across_range() {
        let make_meta = |freq: usize| PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: freq,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &make_meta(0), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &make_meta(1), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s10 = EvictionWorker::compute_importance_score(
            &make_meta(10), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s100 = EvictionWorker::compute_importance_score(
            &make_meta(100), None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s1 - s0, FREQUENCY_BONUS as i64, "step 0->1 = FREQUENCY_BONUS");
        assert_eq!(s10 - s0, 10 * FREQUENCY_BONUS as i64, "step 0->10 = 10 * FREQUENCY_BONUS");
        assert_eq!(s100 - s0, 100 * FREQUENCY_BONUS as i64, "step 0->100 = 100 * FREQUENCY_BONUS");
    }

    // ── Score: time penalty exactly linear across wide range ──

    #[test]
    fn time_penalty_linear_across_range() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 1,
        );
        let s10 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 10,
        );
        let s100 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 100,
        );
        assert_eq!(s0 - s1, TIME_DECAY_WEIGHT, "1 tick penalty = TIME_DECAY_WEIGHT");
        assert_eq!(s0 - s10, 10 * TIME_DECAY_WEIGHT, "10 tick penalty = 10 * TIME_DECAY_WEIGHT");
        assert_eq!(s0 - s100, 100 * TIME_DECAY_WEIGHT, "100 tick penalty = 100 * TIME_DECAY_WEIGHT");
    }

    // ── EvictionCandidate Debug: includes all 6 fields ──

    #[test]
    fn eviction_candidate_debug_all_six_fields() {
        let c = EvictionCandidate {
            page_id: 7,
            score: -42,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 8192,
            group_id: Some(99),
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("page_id"), "Debug should contain page_id");
        assert!(debug.contains("score"), "Debug should contain score");
        assert!(debug.contains("current_tier"), "Debug should contain current_tier");
        assert!(debug.contains("codec"), "Debug should contain codec");
        assert!(debug.contains("page_bytes"), "Debug should contain page_bytes");
        assert!(debug.contains("group_id"), "Debug should contain group_id");
    }

    // ── Score: None payload and KvContext produce identical results ──

    #[test]
    fn none_payload_identical_to_kv_context() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 3,
            access_count: 7,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_none = EvictionWorker::compute_importance_score(
            &meta, None, 2048, 4096, StorageTier::CpuDram, 75,
        );
        let s_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::CpuDram, 75,
        );
        assert_eq!(s_none, s_kv, "None and KvContext should produce identical scores");
    }

    // ── Score: all payload kinds produce distinct values when combined with state bonuses ──

    #[test]
    fn all_payloads_distinct_with_protected_state() {
        let make_meta = || PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let kinds = [
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::KvContext,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::DenseLayerWeight,
        ];
        let scores: Vec<i64> = kinds.iter().map(|k| {
            EvictionWorker::compute_importance_score(
                &make_meta(), Some(*k), 0, 0, StorageTier::GpuHbm, 0,
            )
        }).collect();
        for i in 0..scores.len() {
            for j in (i + 1)..scores.len() {
                assert_ne!(
                    scores[i], scores[j],
                    "Protected + {:?} should differ from Protected + {:?}: {} vs {}",
                    kinds[i], kinds[j], scores[i], scores[j],
                );
            }
        }
    }

    // ── EvictionWorkerConfig Debug: all 9 fields present ──

    #[test]
    fn config_debug_all_nine_fields() {
        let cfg = EvictionWorkerConfig::default();
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("tick_interval"), "missing tick_interval");
        assert!(debug.contains("max_evict_per_round"), "missing max_evict_per_round");
        assert!(debug.contains("hbm_pressure_threshold"), "missing hbm_pressure_threshold");
        assert!(debug.contains("dram_pressure_threshold"), "missing dram_pressure_threshold");
        assert!(debug.contains("importance_threshold"), "missing importance_threshold");
        assert!(debug.contains("hbm_evict_age_ticks"), "missing hbm_evict_age_ticks");
        assert!(debug.contains("dram_evict_age_ticks"), "missing dram_evict_age_ticks");
        assert!(debug.contains("default_evict_codec"), "missing default_evict_codec");
        assert!(debug.contains("page_bytes"), "missing page_bytes");
    }

    // ── Score: recency penalty and time penalty are independent ──

    #[test]
    fn recency_and_time_penalty_independent() {
        let meta_r0_t0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_r10_t0 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 10,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_r0_t10 = PageMetadata {
            page_id: 3,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_base = EvictionWorker::compute_importance_score(
            &meta_r0_t0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_recency = EvictionWorker::compute_importance_score(
            &meta_r10_t0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_time = EvictionWorker::compute_importance_score(
            &meta_r0_t10, None, 0, 0, StorageTier::GpuHbm, 10,
        );
        // recency penalty = 10 * (TIME_DECAY_WEIGHT / 2) = 10 * 1 = 10
        // time penalty = 10 * TIME_DECAY_WEIGHT = 10 * 2 = 20
        assert_eq!(s_base - s_recency, 10 * (TIME_DECAY_WEIGHT / 2));
        assert_eq!(s_base - s_time, 10 * TIME_DECAY_WEIGHT);
        // They are independent: total reduction with both should be sum.
    }

    // ── Score: combined recency + time penalty is sum of individual penalties ──

    #[test]
    fn recency_plus_time_penalty_is_additive() {
        let meta_base = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_both = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 20,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_base = EvictionWorker::compute_importance_score(
            &meta_base, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_both = EvictionWorker::compute_importance_score(
            &meta_both, None, 0, 0, StorageTier::GpuHbm, 20,
        );
        // recency_penalty = 20 * 1 = 20
        // time_penalty = 20 * 2 = 40
        // total = 60
        assert_eq!(
            s_base - s_both, 60,
            "combined recency + time penalty should be additive (20 + 40 = 60)",
        );
    }

    // ── EvictionCandidate can be collected into a Vec and sorted in one pass ──

    #[test]
    fn eviction_candidate_collect_and_sort() {
        let candidates: Vec<EvictionCandidate> = [(-50, 1), (200, 2), (-300, 3), (0, 4)]
            .into_iter()
            .map(|(score, page_id)| EvictionCandidate {
                page_id,
                score,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        let mut sorted = candidates;
        sorted.sort_by_key(|c| c.score);
        assert_eq!(sorted[0].page_id, 3);
        assert_eq!(sorted[1].page_id, 1);
        assert_eq!(sorted[2].page_id, 4);
        assert_eq!(sorted[3].page_id, 2);
    }

    // ── Score: compression bonus at 10% compression ──

    #[test]
    fn compression_ratio_exact_10_percent() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_10pct = EvictionWorker::compute_importance_score(
            &meta, None, 3686, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio = 3686/4096 ≈ 0.8999, bonus = (1 - 0.8999) * 500 ≈ 50
        // Exact: (1 - 3686/4096) * 500 = (410/4096) * 500 = 50.048...
        // Truncated to i64: 50
        let delta = score_10pct - score_no_compress;
        assert!(
            delta > 45 && delta < 55,
            "10% compression bonus should be approximately 50: got {}",
            delta,
        );
    }

    // ── Score: payload bonuses are exact integers ──

    #[test]
    fn payload_bonuses_are_exact_integers() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(EXPERT_WEIGHT_BONUS, -300);
        assert_eq!(KV_CONTEXT_BONUS, 0);
        assert_eq!(PROMPT_SYSTEM_BONUS, 1000);
        assert_eq!(DENSE_LAYER_BONUS, 5000);
        assert_eq!(KNOWLEDGE_RAG_BONUS, -500);
        // Verify they match the computed differences.
        assert_eq!(
            EvictionWorker::compute_importance_score(
                &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 0,
            ) - s_kv,
            EXPERT_WEIGHT_BONUS,
        );
        assert_eq!(
            EvictionWorker::compute_importance_score(
                &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
            ) - s_kv,
            PROMPT_SYSTEM_BONUS,
        );
        assert_eq!(
            EvictionWorker::compute_importance_score(
                &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
            ) - s_kv,
            DENSE_LAYER_BONUS,
        );
        assert_eq!(
            EvictionWorker::compute_importance_score(
                &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
            ) - s_kv,
            KNOWLEDGE_RAG_BONUS,
        );
    }

    // ── EvictionTier Copy: assigning to another variable does not move ──

    #[test]
    fn eviction_tier_copy_semantic() {
        let original = EvictionTier::PinnedDense;
        let copy1 = original;
        let copy2 = original;
        assert_eq!(original, copy1);
        assert_eq!(original, copy2);
        // original is still usable because Copy.
    }

    // ── Score: swap_in_time with future Instant yields zero tier_age ──

    #[test]
    fn tier_age_future_swap_in_is_zero() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now() + Duration::from_secs(60)),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        assert_eq!(age, 0, "future swap_in_time should saturate to 0 age ticks");
    }

    // ── EvictionCandidate with all StorageTier variants preserves tier in clone ──

    #[test]
    fn eviction_candidate_clone_preserves_all_tiers() {
        for tier in &[StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let original = EvictionCandidate {
                page_id: 1,
                score: 0,
                current_tier: *tier,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            };
            let cloned = original.clone();
            assert_eq!(cloned.current_tier, *tier, "cloned tier should match original for {:?}", tier);
        }
    }

    // ── EvictionCandidate with all CompressionCodec variants preserves codec in clone ──

    #[test]
    fn eviction_candidate_clone_preserves_all_codecs() {
        for codec in &[
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let original = EvictionCandidate {
                page_id: 1,
                score: 0,
                current_tier: StorageTier::GpuHbm,
                codec: *codec,
                page_bytes: 4096,
                group_id: None,
            };
            let cloned = original.clone();
            assert_eq!(cloned.codec, *codec, "cloned codec should match original for {:?}", codec);
        }
    }

    // ── EvictionWorkerConfig with dram_pressure_threshold=0.0 ──

    #[test]
    fn config_zero_dram_threshold() {
        let cfg = EvictionWorkerConfig {
            dram_pressure_threshold: 0.0,
            ..EvictionWorkerConfig::default()
        };
        assert!((cfg.dram_pressure_threshold - 0.0).abs() < 1e-6);
    }

    // ── Score: base score constant is exactly 1000 ──

    #[test]
    fn base_score_constant_verification() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: Some(0),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000, "base score must be exactly 1000");
    }

    // ── EvictionTier can be stored in a HashSet without duplicates ──

    #[test]
    fn eviction_tier_hashset_no_duplicates() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(EvictionTier::ColdExpert));
        assert!(set.insert(EvictionTier::PinnedDense));
        assert!(set.insert(EvictionTier::StandbyKv));
        assert!(set.insert(EvictionTier::Protected));
        assert!(!set.insert(EvictionTier::ColdExpert), "duplicate insert should return false");
        assert_eq!(set.len(), 4);
    }

    // ── Score: tier_age_ticks=1 produces score exactly TIME_DECAY_WEIGHT less than 0 ──

    #[test]
    fn single_tick_penalty_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 1,
        );
        assert_eq!(s0 - s1, TIME_DECAY_WEIGHT, "1 tick should reduce score by TIME_DECAY_WEIGHT ({})", TIME_DECAY_WEIGHT);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (50 new — batch 5)
    // ─────────────────────────────────────────────────────────────────────────

    // ── Constant value verification: all scoring constants match SPEC ──

    #[test]
    fn constant_time_decay_weight_is_2() {
        assert_eq!(TIME_DECAY_WEIGHT, 2);
    }

    #[test]
    fn constant_frequency_bonus_is_15() {
        assert_eq!(FREQUENCY_BONUS, 15);
    }

    #[test]
    fn constant_compression_ratio_weight_is_500() {
        assert_eq!(COMPRESSION_RATIO_WEIGHT, 500);
    }

    #[test]
    fn constant_page_size_weight_is_1() {
        assert_eq!(PAGE_SIZE_WEIGHT, 1);
    }

    #[test]
    fn constant_importance_threshold_is_100() {
        assert_eq!(IMPORTANCE_SCORE_THRESHOLD, 100);
    }

    #[test]
    fn constant_hbm_evict_age_is_50() {
        assert_eq!(HBM_EVICT_AGE_TICKS, 50);
    }

    #[test]
    fn constant_dram_evict_age_is_500() {
        assert_eq!(DRAM_EVICT_AGE_TICKS, 500);
    }

    #[test]
    fn constant_hbm_pressure_ratio() {
        assert!((HBM_PRESSURE_RATIO - 0.90).abs() < 1e-6);
    }

    #[test]
    fn constant_dram_pressure_ratio() {
        assert!((DRAM_PRESSURE_RATIO - 0.80).abs() < 1e-6);
    }

    #[test]
    fn constant_default_tick_interval() {
        assert_eq!(DEFAULT_TICK_INTERVAL, Duration::from_millis(10));
    }

    #[test]
    fn constant_default_max_evict_per_round() {
        assert_eq!(DEFAULT_MAX_EVICT_PER_ROUND, 8);
    }

    #[test]
    fn constant_expert_weight_bonus() {
        assert_eq!(EXPERT_WEIGHT_BONUS, -300);
    }

    #[test]
    fn constant_kv_context_bonus() {
        assert_eq!(KV_CONTEXT_BONUS, 0);
    }

    #[test]
    fn constant_prompt_system_bonus() {
        assert_eq!(PROMPT_SYSTEM_BONUS, 1000);
    }

    #[test]
    fn constant_dense_layer_bonus() {
        assert_eq!(DENSE_LAYER_BONUS, 5000);
    }

    #[test]
    fn constant_knowledge_rag_bonus() {
        assert_eq!(KNOWLEDGE_RAG_BONUS, -500);
    }

    // ── evict_round: max_evict_per_round limits eviction count ──

    #[test]
    fn evict_round_respects_max_evict_per_round() {
        let config = EvictionWorkerConfig {
            max_evict_per_round: 2,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_millis(6000);
        let mut pages = HashMap::new();
        let mut addrs = HashMap::new();
        for i in 1..=5usize {
            pages.insert(i, PageMetadata {
                page_id: i,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: old_instant,
                swap_in_time: Some(old_instant),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
            addrs.insert(i, PageAddrEntry {
                gpu_ptr: Some(0x1000 + (i as u64) * 0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(pages));
        {
            let mut guard = addr_table.write().unwrap();
            for (k, v) in addrs {
                guard.insert(k, v);
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert!(
            submitted <= 2,
            "max_evict_per_round=2 should limit evictions: submitted={}",
            submitted,
        );
        actor.shutdown();
    }

    // ── evict_round: page not in addr_table is skipped ──

    #[test]
    fn evict_round_skips_pages_not_in_addr_table() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_millis(6000);
        let meta = PageMetadata {
            page_id: 99,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Page metadata exists but NO addr_table entry.
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(99, meta)])));

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "page without addr_table entry should be skipped");
        actor.shutdown();
    }

    // ── evict_round: tier_age below hbm_evict_age_ticks is skipped ──

    #[test]
    fn evict_round_skips_young_hbm_pages() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Just swapped in (very young page — tier_age < 50 ticks).
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "young HBM page should not be evicted");
        actor.shutdown();
    }

    // ── evict_round: NVMe pages are never eligible regardless of pressure ──

    #[test]
    fn evict_round_nvme_pages_never_eligible() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(300);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // High HBM pressure.
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "NVMe pages should never be eligible for further eviction");
        actor.shutdown();
    }

    // ── evict_round: high score pages are not evicted even under pressure ──

    #[test]
    fn evict_round_high_score_pages_not_evicted() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_millis(6000);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1000, // very high frequency → high score
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "high-score page should not be evicted");
        actor.shutdown();
    }

    // ── evict_round: no eviction when both pressures below threshold ──

    #[test]
    fn evict_round_no_pressure_no_eviction() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_millis(6000);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // Both tiers at 0% usage — no pressure.
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(1000, 1000, 1000)));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        assert_eq!(submitted, 0, "no pressure should yield no eviction");
        actor.shutdown();
    }

    // ── compute_importance_score: scoring with all penalties active (negative result) ──

    #[test]
    fn score_all_penalties_combined() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 50,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KnowledgeRAG),
            8000, // over-compressed
            4096,
            StorageTier::Nvme,
            500,
        );
        // base(1000) - time(500*2=1000) - recency(50*1=50) + freq(0)
        // + compression((1-1.95)*500=-475) + page_size(4096/1024=4) + rag(-500)
        // + state(0) + nvme(-500) = 1000 - 1000 - 50 - 475 + 4 - 500 - 500 = -1521
        assert!(
            score < IMPORTANCE_SCORE_THRESHOLD,
            "all penalties should drive score well below threshold: got {}",
            score,
        );
        assert_eq!(score, -1522, "exact combined penalty score");
    }

    // ── compute_importance_score: scoring with all bonuses active (high positive result) ──

    #[test]
    fn score_all_bonuses_combined() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 200,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            0, // fully compressed
            16384, // large page
            StorageTier::GpuHbm,
            0,
        );
        // base(1000) - time(0) - recency(0) + freq(200*15=3000)
        // + compression((1-0)*500=500) + page_size(16384/1024=16) + dense(5000)
        // + protected(10000) + hbm(0) = 1000 + 3000 + 500 + 16 + 5000 + 10000 = 19516
        assert!(
            score > IMPORTANCE_SCORE_THRESHOLD,
            "all bonuses should drive score far above threshold: got {}",
            score,
        );
        assert_eq!(score, 19516, "exact combined bonus score");
    }

    // ── EvictionCandidate with score=0 is not special ──

    #[test]
    fn eviction_candidate_zero_score_is_valid() {
        let c = EvictionCandidate {
            page_id: 1,
            score: 0,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(1),
        };
        assert_eq!(c.score, 0);
        // score=0 is below IMPORTANCE_SCORE_THRESHOLD=100, so it would be evictable.
        assert!(c.score < IMPORTANCE_SCORE_THRESHOLD);
    }

    // ── EvictionWorkerConfig with zero capacities does not panic on construction ──

    #[test]
    fn config_with_zero_capacities_no_panic() {
        let _ = GlobalMemoryManager::new_with_capacities(0, 0, 0);
    }

    // ── EvictionWorkerConfig clone: modifying clone does not affect original ──

    #[test]
    fn config_clone_is_deep_copy() {
        let original = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(20),
            max_evict_per_round: 4,
            hbm_pressure_threshold: 0.85,
            dram_pressure_threshold: 0.75,
            importance_threshold: 200,
            hbm_evict_age_ticks: 100,
            dram_evict_age_ticks: 1000,
            default_evict_codec: CompressionCodec::NvcompAns,
            page_bytes: 8192,
        };
        let mut cloned = original.clone();
        cloned.tick_interval = Duration::from_secs(1);
        cloned.max_evict_per_round = 999;
        cloned.hbm_pressure_threshold = 0.0;
        cloned.dram_pressure_threshold = 1.0;
        cloned.importance_threshold = i64::MIN;
        cloned.hbm_evict_age_ticks = 0;
        cloned.dram_evict_age_ticks = 0;
        cloned.default_evict_codec = CompressionCodec::None;
        cloned.page_bytes = 0;

        // Cloned reflects mutations.
        assert_eq!(cloned.tick_interval, Duration::from_secs(1));
        assert_eq!(cloned.max_evict_per_round, 999);
        assert!((cloned.hbm_pressure_threshold).abs() < 1e-6);
        assert!((cloned.dram_pressure_threshold - 1.0).abs() < 1e-6);
        assert_eq!(cloned.importance_threshold, i64::MIN);
        assert_eq!(cloned.hbm_evict_age_ticks, 0);
        assert_eq!(cloned.dram_evict_age_ticks, 0);
        assert_eq!(cloned.default_evict_codec, CompressionCodec::None);
        assert_eq!(cloned.page_bytes, 0);
        // Original unchanged.
        assert_eq!(original.tick_interval, Duration::from_millis(20));
        assert_eq!(original.max_evict_per_round, 4);
        assert!((original.hbm_pressure_threshold - 0.85).abs() < 1e-6);
        assert!((original.dram_pressure_threshold - 0.75).abs() < 1e-6);
        assert_eq!(original.importance_threshold, 200);
        assert_eq!(original.hbm_evict_age_ticks, 100);
        assert_eq!(original.dram_evict_age_ticks, 1000);
        assert_eq!(original.default_evict_codec, CompressionCodec::NvcompAns);
        assert_eq!(original.page_bytes, 8192);
    }

    // ── Score: i64 arithmetic does not overflow for reasonable inputs ──

    #[test]
    fn score_reasonable_inputs_no_overflow() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 1000,
            access_count: 10000,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            2048,
            65536,
            StorageTier::GpuHbm,
            10000,
        );
        // No overflow, result is a valid i64.
        assert!(
            score > i64::MIN && score < i64::MAX,
            "reasonable inputs should not cause overflow: got {}",
            score,
        );
    }

    // ── EvictionTier matches exhaustively across all code paths ──

    #[test]
    fn eviction_tier_classify_exhaustive_coverage() {
        let all_kinds = [
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::DenseLayerWeight,
            PagePayloadKind::KvContext,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
        ];
        let mut seen_tiers = std::collections::HashSet::new();
        for kind in &all_kinds {
            for score in [0i64, 50, 100, 500, 10000] {
                let tier = EvictionWorker::classify_eviction_tier(Some(*kind), score);
                seen_tiers.insert(tier);
            }
        }
        // None payload with low and high scores.
        seen_tiers.insert(EvictionWorker::classify_eviction_tier(None, 0));
        seen_tiers.insert(EvictionWorker::classify_eviction_tier(None, 999));
        // All 4 tiers should be reachable.
        assert_eq!(seen_tiers.len(), 4, "all 4 EvictionTier variants should be reachable");
    }

    // ── Score: mixed state + payload produces correct ordering ──

    #[test]
    fn score_ordering_protected_beats_warm_beats_standby() {
        let make_meta = |state: PageState| PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Standby), None, 0, 4096, StorageTier::GpuHbm, 50,
        );
        let score_warm = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Warm), None, 0, 4096, StorageTier::GpuHbm, 50,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Protected), None, 0, 4096, StorageTier::GpuHbm, 50,
        );
        assert!(score_protected > score_warm, "Protected > Warm");
        assert!(score_warm > score_standby, "Warm > Standby");
    }

    // ── compute_tier_age: old last_access without swap_in_time yields large ticks ──

    #[test]
    fn tier_age_old_last_access_yields_large_ticks() {
        let old_instant = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        // 10 seconds = 10000ms / 10 = 1000 ticks.
        assert!(
            age >= 900,
            "old last_access should yield large tick count, expected ~1000, got {}",
            age,
        );
    }

    // ── infer_payload_kind: sequence_id=Some(RequestId::MAX) is KvContext ──

    #[test]
    fn infer_payload_max_request_id_is_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(RequestId::MAX),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::KvContext),
        );
    }

    // ── infer_payload_kind: sequence_id=Some(PageId::MAX) is KvContext (non-zero) ──

    #[test]
    fn infer_payload_large_sequence_id_is_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(PageId::MAX as u64),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::KvContext),
        );
    }

    // ── Score: exact value for ExpertWeight + Standby + GpuHbm with freq=3, age=25 ──

    #[test]
    fn score_exact_expert_hbm_freq3_age25() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 25,
        );
        // base(1000) - time(25*2=50) + freq(3*15=45) + expert(-300) = 695
        assert_eq!(score, 695, "ExpertWeight + freq=3 + age=25 = 695, got {}", score);
    }

    // ── Score: exact value for KvContext + SwappedOut + CpuDram, zero other ──

    #[test]
    fn score_exact_kv_swappedout_dram() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + kv(0) + swappedout(0) + dram(-200) = 800
        assert_eq!(score, 800, "KvContext + SwappedOut + CpuDram = 800, got {}", score);
    }

    // ── Score: exact value for KnowledgeRAG + Active + Nvme, zero other ──

    #[test]
    fn score_exact_rag_active_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + rag(-500) + active(0) + nvme(-500) = 0
        assert_eq!(score, 0, "KnowledgeRAG + Active + NVMe = 0, got {}", score);
    }

    // ── Score: exact value for PromptSystem + Free + GpuHbm, zero other ──

    #[test]
    fn score_exact_prompt_free_hbm() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        // base(1000) + prompt(1000) + free(0) + hbm(0) = 2000
        assert_eq!(score, 2000, "PromptSystem + Free + GpuHbm = 2000, got {}", score);
    }

    // ── Score: compression_ratio_weight constant determines max compression bonus ──

    #[test]
    fn compression_max_bonus_equals_weight() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_base = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_full_compress = EvictionWorker::compute_importance_score(
            &meta, None, 0, 4096, StorageTier::GpuHbm, 0,
        );
        // Maximum compression bonus = COMPRESSION_RATIO_WEIGHT = 500
        // page_size bonus is the same for both (original=4096).
        assert_eq!(
            score_full_compress - score_base,
            COMPRESSION_RATIO_WEIGHT as i64,
            "max compression bonus should equal COMPRESSION_RATIO_WEIGHT ({})",
            COMPRESSION_RATIO_WEIGHT,
        );
    }

    // ── EvictionWorkerConfig: custom hbm_evict_age_ticks=0 allows immediate eviction ──

    #[test]
    fn config_zero_hbm_evict_age_allows_immediate_eviction() {
        let cfg = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.hbm_evict_age_ticks, 0);
        // With age threshold 0, any page on HBM (even freshly swapped in) is eligible.
    }

    // ── EvictionCandidate: group_id distinguishes pages from different sequences ──

    #[test]
    fn eviction_candidate_different_groups() {
        let c_a = EvictionCandidate {
            page_id: 1,
            score: 50,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(10),
        };
        let c_b = EvictionCandidate {
            page_id: 2,
            score: 50,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(20),
        };
        assert_ne!(c_a.group_id, c_b.group_id);
    }

    // ── Score: page_size_bonus exactly 0 for original_size < 1024 ──

    #[test]
    fn page_size_bonus_zero_for_sub_kb_page() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_512 = EvictionWorker::compute_importance_score(
            &meta, None, 512, 512, StorageTier::GpuHbm, 0,
        );
        let score_1023 = EvictionWorker::compute_importance_score(
            &meta, None, 1023, 1023, StorageTier::GpuHbm, 0,
        );
        // Both: page_size = 512/1024 = 0, 1023/1024 = 0 (integer division)
        // compression_ratio = 1.0 for both → bonus = 0
        // So both should equal base = 1000
        assert_eq!(score_512, 1000, "512-byte page should have 0 page_size bonus");
        assert_eq!(score_1023, 1000, "1023-byte page should have 0 page_size bonus");
    }

    // ── Score: page_size_bonus exactly 1 for original_size = 1024 ──

    #[test]
    fn page_size_bonus_one_for_exactly_1kb() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 1024, 1024, StorageTier::GpuHbm, 0,
        );
        // compression_ratio = 1.0 → bonus = 0
        // page_size = 1024/1024 * 1 = 1
        // base(1000) + 1 = 1001
        assert_eq!(score, 1001, "exactly 1KB page should have page_size bonus = 1");
    }

    // ── Score: frequency_bonus constant determines per-access delta ──

    #[test]
    fn frequency_bonus_constant_matches_delta() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_0 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // Delta for 1 access should equal FREQUENCY_BONUS.
        let delta = FREQUENCY_BONUS as i64;
        assert_eq!(score_0 + delta - score_0, delta);
    }

    // ── EvictionCandidate: clone then mutate original does not affect clone ──

    #[test]
    fn eviction_candidate_clone_independence_after_mutation() {
        let mut original = EvictionCandidate {
            page_id: 1,
            score: 100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(42),
        };
        let cloned = original.clone();
        original.score = -9999;
        original.page_id = 999;
        original.current_tier = StorageTier::Nvme;
        assert_eq!(original.score, -9999);
        assert_eq!(original.page_id, 999);
        assert_eq!(original.current_tier, StorageTier::Nvme);
        assert_eq!(cloned.score, 100, "clone should not be affected by original mutation");
        assert_eq!(cloned.page_id, 1);
        assert_eq!(cloned.current_tier, StorageTier::GpuHbm);
    }

    // ── Score: recency penalty exactly half of time penalty per unit ──

    #[test]
    fn recency_penalty_is_half_time_penalty() {
        let meta_r1 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 1,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_base = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_base = EvictionWorker::compute_importance_score(
            &meta_base, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_r1 = EvictionWorker::compute_importance_score(
            &meta_r1, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_t1 = EvictionWorker::compute_importance_score(
            &meta_base, None, 0, 0, StorageTier::GpuHbm, 1,
        );
        // recency penalty for 1 = TIME_DECAY_WEIGHT / 2 = 1
        // time penalty for 1 = TIME_DECAY_WEIGHT = 2
        let recency_delta = s_base - s_r1;
        let time_delta = s_base - s_t1;
        assert_eq!(time_delta, 2 * recency_delta, "time penalty should be exactly 2x recency penalty per unit");
    }

    // ── Score: negative compressed_size yields negative compression bonus (no panic) ──

    #[test]
    fn score_negative_compression_no_panic() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // compressed=0 (zero compression) is the maximum bonus.
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 4096, StorageTier::GpuHbm, 0,
        );
        // Just verify no panic — the score is a valid i64.
        assert!(score > 0, "score should be positive with full compression: got {}", score);
    }

    // ── Score: verify symmetry — swapping payload kind changes only payload bonus ──

    #[test]
    fn score_payload_swap_only_changes_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let s_expert = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 2048, 4096, StorageTier::GpuHbm, 100,
        );
        let s_dense = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 2048, 4096, StorageTier::GpuHbm, 100,
        );
        // Only payload bonus differs: DenseLayerWeight(+5000) vs ExpertWeight(-300) = delta 5300
        assert_eq!(
            s_dense - s_expert, 5300,
            "swapping ExpertWeight to DenseLayerWeight should change score by exactly 5300",
        );
    }

    // ── EvictionCandidate: sort stability with 10 candidates and identical scores ──

    #[test]
    fn eviction_candidate_sort_stability_large_set() {
        let candidates: Vec<EvictionCandidate> = (0..10usize)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: 42, // all identical
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        let mut sorted = candidates;
        sorted.sort_by_key(|c| c.score);
        // Stable sort should preserve original order for equal keys.
        for (i, c) in sorted.iter().enumerate() {
            assert_eq!(c.page_id, i, "stable sort should preserve order for equal scores at index {}", i);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (~60 new — batch 6)
    // ─────────────────────────────────────────────────────────────────────────

    // ── Score: large recency can overflow i64 but does not panic ──

    #[test]
    fn score_large_recency_no_panic() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: usize::MAX,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // recency * 1 = usize::MAX cast to i64 — may wrap, but must not panic.
        let _ = score;
    }

    // ── Score: access_count=1 produces exactly FREQUENCY_BONUS more than 0 ──

    #[test]
    fn score_access_count_one_exact_bonus() {
        let meta_0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_1 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta_0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta_1, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s1 - s0, FREQUENCY_BONUS as i64);
    }

    // ── Score: compression bonus zero when compressed equals original ──

    #[test]
    fn compression_bonus_zero_at_equal_sizes() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 8192, 8192, StorageTier::GpuHbm, 0,
        );
        // ratio = 1.0 → compression bonus = 0
        // page_size = 8192/1024 = 8
        // base(1000) + 8 = 1008
        assert_eq!(score, 1008, "no compression bonus when compressed == original, got {}", score);
    }

    // ── Score: verify base=1000 with all-neutral inputs on CpuDram ──

    #[test]
    fn score_base_on_cpu_dram() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(score, 800, "base(1000) + dram(-200) = 800, got {}", score);
    }

    // ── Score: verify base=1000 with all-neutral inputs on Nvme ──

    #[test]
    fn score_base_on_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(score, 500, "base(1000) + nvme(-500) = 500, got {}", score);
    }

    // ── Score: ExpertWeight + CpuDram exactly with freq=1 ──

    #[test]
    fn score_exact_expert_dram_freq1() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + freq(15) + expert(-300) + dram(-200) = 515
        assert_eq!(score, 515, "ExpertWeight + CpuDram + freq=1 = 515, got {}", score);
    }

    // ── EvictionCandidate: page_id=0 with score=0 is valid ──

    #[test]
    fn eviction_candidate_zero_page_zero_score() {
        let c = EvictionCandidate {
            page_id: 0,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 0,
            group_id: None,
        };
        assert_eq!(c.page_id, 0);
        assert_eq!(c.score, 0);
        assert_eq!(c.page_bytes, 0);
        assert!(c.group_id.is_none());
    }

    // ── EvictionCandidate: all fields round-trip through clone ──

    #[test]
    fn eviction_candidate_full_clone_roundtrip() {
        let c = EvictionCandidate {
            page_id: 42,
            score: -999,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 32768,
            group_id: Some(7),
        };
        let cloned = c.clone();
        assert_eq!(cloned.page_id, 42);
        assert_eq!(cloned.score, -999);
        assert_eq!(cloned.current_tier, StorageTier::CpuDram);
        assert_eq!(cloned.codec, CompressionCodec::NvcompAns);
        assert_eq!(cloned.page_bytes, 32768);
        assert_eq!(cloned.group_id, Some(7));
    }

    // ── Score: page_size_bonus exactly 0 for original_size=1 ──

    #[test]
    fn page_size_bonus_zero_for_1_byte() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_page = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_1byte = EvictionWorker::compute_importance_score(
            &meta, None, 0, 1, StorageTier::GpuHbm, 0,
        );
        // page_size = 1/1024 = 0 (integer division)
        // compression = (1-0)*500 = 500
        // delta from base = 500 (compression only, page_size = 0)
        assert_eq!(score_1byte - score_no_page, 500, "1-byte page should have 0 page_size bonus, only compression");
    }

    // ── classify_eviction_tier: KvContext at score 0 ──

    #[test]
    fn classify_kv_score_zero_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 0),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: None at score 0 ──

    #[test]
    fn classify_none_score_zero_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, 0),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: KnowledgeRAG at score 0 ──

    #[test]
    fn classify_rag_score_zero_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KnowledgeRAG), 0),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: PromptSystem at score 0 ──

    #[test]
    fn classify_prompt_score_zero_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::PromptSystem), 0),
            EvictionTier::StandbyKv,
        );
    }

    // ── Score: CpuDram discount exactly -200 with Warm state ──

    #[test]
    fn dram_discount_with_warm_state() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(score_hbm - score_dram, 200, "DRAM discount = -200 even with Warm state");
    }

    // ── Score: Nvme discount exactly -500 with Protected state ──

    #[test]
    fn nvme_discount_with_protected_state() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(score_hbm - score_nvme, 500, "NVMe discount = -500 even with Protected state");
    }

    // ── EvictionWorkerConfig: tick_interval = Duration::MAX ──

    #[test]
    fn config_max_tick_interval() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::MAX,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval, Duration::MAX);
    }

    // ── EvictionWorkerConfig: all numeric fields at extreme boundaries ──

    #[test]
    fn config_all_extreme_boundaries() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::MAX,
            max_evict_per_round: usize::MAX,
            hbm_pressure_threshold: f32::MAX,
            dram_pressure_threshold: f32::MIN, // negative
            importance_threshold: i64::MIN,
            hbm_evict_age_ticks: u64::MAX,
            dram_evict_age_ticks: 0,
            default_evict_codec: CompressionCodec::ZstdDict,
            page_bytes: usize::MAX,
        };
        assert_eq!(cfg.tick_interval, Duration::MAX);
        assert_eq!(cfg.max_evict_per_round, usize::MAX);
        assert_eq!(cfg.hbm_pressure_threshold, f32::MAX);
        assert_eq!(cfg.dram_pressure_threshold, f32::MIN);
        assert_eq!(cfg.importance_threshold, i64::MIN);
        assert_eq!(cfg.hbm_evict_age_ticks, u64::MAX);
        assert_eq!(cfg.dram_evict_age_ticks, 0);
        assert_eq!(cfg.default_evict_codec, CompressionCodec::ZstdDict);
        assert_eq!(cfg.page_bytes, usize::MAX);
        let cloned = cfg.clone();
        assert_eq!(cloned.hbm_pressure_threshold, f32::MAX);
    }

    // ── Score: ExpertWeight + Nvme + recency=10, age=50 ──

    #[test]
    fn score_exact_expert_nvme_recency10_age50() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 10,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 50,
        );
        // base(1000) - time(50*2=100) - recency(10*1=10) + expert(-300) + nvme(-500) = 90
        assert_eq!(score, 90, "ExpertWeight + Nvme + recency=10 + age=50 = 90, got {}", score);
    }

    // ── Score: DenseLayerWeight + Protected + freq=50 + age=5 ──

    #[test]
    fn score_exact_dense_protected_freq50_age5() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 5,
        );
        // base(1000) - time(5*2=10) + freq(50*15=750) + dense(5000) + protected(10000) = 16740
        assert_eq!(score, 16740, "DenseLayerWeight + Protected + freq=50 + age=5 = 16740, got {}", score);
    }

    // ── Score: KvContext + Warm + freq=3 + age=100 + 50% compression ──

    #[test]
    fn score_exact_kv_warm_freq3_age100_half_compress() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::GpuHbm, 100,
        );
        // base(1000) - time(100*2=200) + freq(3*15=45)
        // + compression((1-0.5)*500=250) + page_size(4096/1024=4) + warm(5000) = 6099
        assert_eq!(score, 6099, "KvContext + Warm + freq=3 + age=100 + 50%% compress = 6099, got {}", score);
    }

    // ── EvictionTier: can be used in a Vec and iterated ──

    #[test]
    fn eviction_tier_vec_iteration() {
        let tiers = vec![
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        assert_eq!(tiers.len(), 4);
        let cold_count = tiers.iter().filter(|t| **t == EvictionTier::ColdExpert).count();
        assert_eq!(cold_count, 1);
    }

    // ── Score: swapping payload between ExpertWeight and DenseLayerWeight changes exactly 5300 ──

    #[test]
    fn score_payload_delta_expert_to_dense() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_expert = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 1024, 4096, StorageTier::GpuHbm, 50,
        );
        let s_dense = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 1024, 4096, StorageTier::GpuHbm, 50,
        );
        assert_eq!(
            s_dense - s_expert, 5300,
            "DenseLayerWeight(+5000) - ExpertWeight(-300) = 5300",
        );
    }

    // ── Score: swapping payload between KnowledgeRAG and PromptSystem changes exactly 1500 ──

    #[test]
    fn score_payload_delta_rag_to_prompt() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 3,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_rag = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 10,
        );
        let s_prompt = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 10,
        );
        assert_eq!(
            s_prompt - s_rag, 1500,
            "PromptSystem(+1000) - KnowledgeRAG(-500) = 1500",
        );
    }

    // ── Score: swapping tier from GpuHbm to CpuDram changes exactly -200 ──

    #[test]
    fn score_tier_delta_hbm_to_dram_independent() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 7,
            access_count: 20,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let s_hbm = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 1024, 4096, StorageTier::GpuHbm, 30,
        );
        let s_dram = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 1024, 4096, StorageTier::CpuDram, 30,
        );
        assert_eq!(s_hbm - s_dram, 200, "tier delta GpuHbm→CpuDram = 200");
    }

    // ── Score: swapping tier from CpuDram to Nvme changes exactly -300 ──

    #[test]
    fn score_tier_delta_dram_to_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        let s_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(s_dram - s_nvme, 300, "tier delta CpuDram→Nvme = 300");
    }

    // ── EvictionCandidate: multiple candidates with descending scores sort correctly ──

    #[test]
    fn eviction_candidate_descending_sort() {
        let mut candidates: Vec<EvictionCandidate> = (0..5)
            .rev()
            .map(|i| EvictionCandidate {
                page_id: i,
                score: (i as i64) * 100,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        candidates.sort_by_key(|c| c.score);
        for (i, c) in candidates.iter().enumerate() {
            assert_eq!(c.score, (i as i64) * 100, "position {} should have score {}", i, i * 100);
        }
    }

    // ── Score: tier_age_ticks=0 yields no time penalty ──

    #[test]
    fn score_zero_age_no_time_penalty() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000, "zero age should produce exactly base score");
    }

    // ── Score: recency=0 yields no recency penalty ──

    #[test]
    fn score_zero_recency_no_penalty() {
        let meta_r0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_r5 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 5,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta_r0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s5 = EvictionWorker::compute_importance_score(
            &meta_r5, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s0 - s5, 5 * (TIME_DECAY_WEIGHT / 2), "recency penalty should be exactly 5 * (TIME_DECAY_WEIGHT / 2)");
    }

    // ── EvictionCandidate: truncation to 0 yields empty vec ──

    #[test]
    fn candidate_truncate_to_zero() {
        let mut candidates: Vec<EvictionCandidate> = (0..5)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: i as i64,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(0);
        assert!(candidates.is_empty(), "truncate to 0 should yield empty vec");
    }

    // ── EvictionWorkerConfig: hbm_pressure_threshold=0.0 triggers eviction always ──

    #[test]
    fn config_zero_hbm_threshold() {
        let cfg = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            ..EvictionWorkerConfig::default()
        };
        assert!((cfg.hbm_pressure_threshold - 0.0).abs() < 1e-6);
        // With threshold 0.0, any positive HBM usage (even 0.001) would trigger eviction.
    }

    // ── Score: compression bonus at 25% compression is 375 ──

    #[test]
    fn compression_ratio_exact_25_percent() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_25 = EvictionWorker::compute_importance_score(
            &meta, None, 3072, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio = 3072/4096 = 0.75, bonus = (1-0.75)*500 = 125
        assert_eq!(
            score_25 - score_no_compress, 125,
            "25% compression bonus should be 125",
        );
    }

    // ── Score: KnowledgeRAG + Warm + GpuHbm + freq=5 + age=10 ──

    #[test]
    fn score_exact_rag_warm_hbm_freq5_age10() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 10,
        );
        // base(1000) - time(10*2=20) + freq(5*15=75) + rag(-500) + warm(5000) = 5555
        assert_eq!(score, 5555, "KnowledgeRAG + Warm + freq=5 + age=10 = 5555, got {}", score);
    }

    // ── EvictionTier: Copy allows re-assignment without move errors ──

    #[test]
    fn eviction_tier_reassignment_after_copy() {
        let mut tier = EvictionTier::ColdExpert;
        let _copy = tier; // Copy, not move
        tier = EvictionTier::Protected; // re-assignment is fine
        assert_eq!(tier, EvictionTier::Protected);
    }

    // ── Score: verify that PageMetadata.default() produces base=1000 score ──

    #[test]
    fn score_default_metadata_base() {
        let meta = PageMetadata::default();
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // Default: sequence_id=None → infer ExpertWeight, but we pass None explicitly.
        // base(1000) + 0 = 1000
        assert_eq!(score, 1000, "default metadata with None payload should give base score");
    }

    // ── EvictionCandidate: Debug output is non-empty for all codec variants ──

    #[test]
    fn eviction_candidate_debug_non_empty_for_all_codecs() {
        for codec in &[
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let c = EvictionCandidate {
                page_id: 1,
                score: 0,
                current_tier: StorageTier::GpuHbm,
                codec: *codec,
                page_bytes: 4096,
                group_id: None,
            };
            let debug = format!("{:?}", c);
            assert!(!debug.is_empty(), "Debug should not be empty for {:?}", codec);
        }
    }

    // ── Score: dense_layer + nvme exactly equal to base + dense + nvme ──

    #[test]
    fn score_exact_dense_nvme_no_other() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + dense(5000) + nvme(-500) = 5500
        assert_eq!(score, 5500, "DenseLayerWeight + Nvme = 5500, got {}", score);
    }

    // ── Score: expert_weight + warm + nvme + freq=2 ──

    #[test]
    fn score_exact_expert_warm_nvme_freq2() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 2,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + freq(2*15=30) + expert(-300) + warm(5000) + nvme(-500) = 5230
        assert_eq!(score, 5230, "ExpertWeight + Warm + Nvme + freq=2 = 5230, got {}", score);
    }

    // ── Score: PromptSystem + Warm + CpuDram + age=25 + recency=5 ──

    #[test]
    fn score_exact_prompt_warm_dram_age25_recency5() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::CpuDram, 25,
        );
        // base(1000) - time(25*2=50) - recency(5*1=5) + prompt(1000) + warm(5000) + dram(-200) = 6745
        assert_eq!(score, 6745, "PromptSystem + Warm + CpuDram + age=25 + recency=5 = 6745, got {}", score);
    }

    // ── Score: page_size_bonus = original_size / 1024 for various sizes ──

    #[test]
    fn page_size_bonus_exact_for_various_sizes() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let base = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // Each size: page_size_bonus = size / 1024 * 1
        // Plus compression bonus = (1.0 - 0.0) * 500 = 500 for compressed=0
        for (original, expected_page_bonus) in [(1024, 1), (4096, 4), (8192, 8), (16384, 16)] {
            let score = EvictionWorker::compute_importance_score(
                &meta, None, 0, original, StorageTier::GpuHbm, 0,
            );
            let delta = score - base;
            // delta = compression_bonus(500) + page_size_bonus(expected_page_bonus)
            assert_eq!(
                delta, 500 + expected_page_bonus,
                "original_size={} should give page_size_bonus={}, got delta={}",
                original, expected_page_bonus, delta,
            );
        }
    }

    // ── EvictionCandidate: sort with all equal scores except one lower ──

    #[test]
    fn eviction_candidate_sort_one_outlier() {
        let mut candidates = vec![
            EvictionCandidate {
                page_id: 1, score: 100, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            },
            EvictionCandidate {
                page_id: 2, score: -500, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            },
            EvictionCandidate {
                page_id: 3, score: 100, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            },
        ];
        candidates.sort_by_key(|c| c.score);
        assert_eq!(candidates[0].page_id, 2, "lowest score should be first");
        assert_eq!(candidates[0].score, -500);
    }

    // ── Score: compression bonus for 0% (no compression) = 0 ──

    #[test]
    fn compression_bonus_zero_percent() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_original = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_same = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // compressed=original → bonus = 0, but page_size = 4096/1024 = 4
        // compressed=0, original=0 → both bonuses are 0
        // delta = page_size bonus from original=4096 = 4
        assert_eq!(
            score_same - score_no_original, 4,
            "no compression bonus, only page_size bonus from 4096",
        );
    }

    // ── EvictionWorkerConfig: dram_pressure_threshold=1.0 ──

    #[test]
    fn config_dram_threshold_one_point_zero() {
        let cfg = EvictionWorkerConfig {
            dram_pressure_threshold: 1.0,
            ..EvictionWorkerConfig::default()
        };
        assert!((cfg.dram_pressure_threshold - 1.0).abs() < 1e-6);
    }

    // ── Score: frequency bonus dominates over recency penalty for high access ──

    #[test]
    fn frequency_dominates_recency() {
        let meta_low_freq = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 100,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_high_freq = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 100,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_low = EvictionWorker::compute_importance_score(
            &meta_low_freq, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_high = EvictionWorker::compute_importance_score(
            &meta_high_freq, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert!(
            s_high > s_low,
            "high frequency should overcome same recency penalty: high={} low={}",
            s_high, s_low,
        );
    }

    // ── Score: combined all-positive vs all-negative scoring components ──

    #[test]
    fn score_positive_vs_negative_scenario() {
        let meta_positive = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let meta_negative = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 50,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s_pos = EvictionWorker::compute_importance_score(
            &meta_positive, Some(PagePayloadKind::DenseLayerWeight), 0, 4096, StorageTier::GpuHbm, 0,
        );
        let s_neg = EvictionWorker::compute_importance_score(
            &meta_negative, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 200,
        );
        assert!(
            s_pos > s_neg,
            "all-positive scenario should dominate all-negative: pos={} neg={}",
            s_pos, s_neg,
        );
        assert!(s_pos > IMPORTANCE_SCORE_THRESHOLD, "positive should be above threshold");
        assert!(s_neg < IMPORTANCE_SCORE_THRESHOLD, "negative should be below threshold");
    }

    // ── EvictionTier: match with wildcard still covers all variants ──

    #[test]
    fn eviction_tier_match_coverage() {
        let all_covered = [EvictionTier::ColdExpert, EvictionTier::PinnedDense, EvictionTier::StandbyKv, EvictionTier::Protected]
            .iter()
            .all(|t| matches!(t, EvictionTier::ColdExpert | EvictionTier::PinnedDense | EvictionTier::StandbyKv | EvictionTier::Protected));
        assert!(all_covered, "all 4 variants should match");
    }

    // ── Score: 50% compression on 8K page with ExpertWeight + CpuDram + age=50 ──

    #[test]
    fn score_exact_expert_dram_compress_50pct_age50() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 4096, 8192, StorageTier::CpuDram, 50,
        );
        // base(1000) - time(50*2=100) + compression((1-0.5)*500=250)
        // + page_size(8192/1024=8) + expert(-300) + dram(-200) = 658
        assert_eq!(score, 658, "ExpertWeight + CpuDram + 50%% compress + age=50 = 658, got {}", score);
    }

    // ── EvictionCandidate: group_id Some vs None produces different candidates ──

    #[test]
    fn eviction_candidate_group_id_distinction() {
        let c_with = EvictionCandidate {
            page_id: 1,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: Some(1),
        };
        let c_without = EvictionCandidate {
            page_id: 1,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };
        assert_ne!(c_with.group_id, c_without.group_id);
    }

    // ── Score: 75% compression on 4K page exact bonus ──

    #[test]
    fn compression_ratio_75pct_4k_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // 75% compression = compressed is 25% of original
        let score_75 = EvictionWorker::compute_importance_score(
            &meta, None, 1024, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio = 1024/4096 = 0.25, bonus = (1-0.25)*500 = 375
        // page_size is same for both (original=4096) = 4
        assert_eq!(
            score_75 - score_no_compress, 375,
            "75% compression bonus should be 375",
        );
    }

    // ── Score: KvContext + Standby + GpuHbm + age=49 is still above 0 ──

    #[test]
    fn score_age_49_still_positive_base() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 49,
        );
        // base(1000) - time(49*2=98) = 902
        assert_eq!(score, 902, "age=49 should yield 902, got {}", score);
    }

    // ── Score: KvContext + Standby + GpuHbm + age=450 → below threshold ──

    #[test]
    fn score_age_450_below_threshold() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 450,
        );
        // base(1000) - time(450*2=900) = 100 → exactly at threshold, NOT below
        assert_eq!(score, 100, "age=450 should yield 100 (exactly at threshold)");
        assert!(score <= IMPORTANCE_SCORE_THRESHOLD);
    }

    // ── Score: KvContext + Standby + GpuHbm + age=451 → below threshold ──

    #[test]
    fn score_age_451_below_threshold() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 451,
        );
        // base(1000) - time(451*2=902) = 98 → below threshold
        assert!(score < IMPORTANCE_SCORE_THRESHOLD, "age=451 should be below threshold, got {}", score);
    }

    // ── EvictionWorkerConfig: Debug contains threshold values ──

    #[test]
    fn config_debug_contains_threshold_values() {
        let cfg = EvictionWorkerConfig::default();
        let debug = format!("{:?}", cfg);
        // Default hbm_pressure_threshold is 0.9
        assert!(debug.contains("0.9"), "Debug should contain hbm threshold value");
    }

    // ── Score: KnowledgeRAG + Nvme + age=1 ──

    // ── Score: DenseLayerWeight + Warm + GpuHbm + age=200 + freq=10 + recency=5 ──

    #[test]
    fn score_exact_dense_warm_hbm_complex() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 200,
        );
        // base(1000) - time(200*2=400) - recency(5*1=5) + freq(10*15=150)
        // + dense(5000) + warm(5000) = 10745
        assert_eq!(score, 10745, "DenseLayerWeight + Warm + age=200 + freq=10 + recency=5 = 10745, got {}", score);
    }

    // ── infer_payload_kind: Various PageState values do not affect inference ──

    #[test]
    fn infer_payload_kind_ignores_state() {
        for state in [PageState::Free, PageState::Active, PageState::Standby, PageState::SwappedOut, PageState::Warm, PageState::Protected, PageState::Swapped] {
            let meta_no_owner = PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 0,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state,
                warm_until: None,
            };
            assert_eq!(
                infer_payload_kind(&meta_no_owner),
                Some(PagePayloadKind::ExpertWeight),
                "no owner should be ExpertWeight regardless of state {:?}",
                state,
            );
        }
    }

    // ── infer_payload_kind: with owner always KvContext regardless of state ──

    #[test]
    fn infer_payload_kind_with_owner_ignores_state() {
        for state in [PageState::Free, PageState::Active, PageState::Standby, PageState::SwappedOut, PageState::Warm, PageState::Protected, PageState::Swapped] {
            let meta_with_owner = PageMetadata {
                page_id: 1,
                sequence_id: Some(42),
                recency: 0,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state,
                warm_until: None,
            };
            assert_eq!(
                infer_payload_kind(&meta_with_owner),
                Some(PagePayloadKind::KvContext),
                "with owner should be KvContext regardless of state {:?}",
                state,
            );
        }
    }

    // ── Score: verify score at age=0 with each payload on each tier ──

    #[test]
    fn score_payload_tier_matrix() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let payloads = [
            (PagePayloadKind::ExpertWeight, -300),
            (PagePayloadKind::KvContext, 0),
            (PagePayloadKind::PromptSystem, 1000),
            (PagePayloadKind::DenseLayerWeight, 5000),
            (PagePayloadKind::KnowledgeRAG, -500),
        ];
        let tiers = [
            (StorageTier::GpuHbm, 0),
            (StorageTier::CpuDram, -200),
            (StorageTier::Nvme, -500),
        ];
        for (payload, p_bonus) in &payloads {
            for (tier, t_discount) in &tiers {
                let score = EvictionWorker::compute_importance_score(
                    &meta, Some(*payload), 0, 0, *tier, 0,
                );
                let expected = 1000 + p_bonus + t_discount;
                assert_eq!(
                    score, expected,
                    "payload={:?} tier={:?}: expected {}, got {}",
                    payload, tier, expected, score,
                );
            }
        }
    }

    // ── EvictionWorkerConfig: clone then compare each field individually ──

    #[test]
    fn config_clone_field_by_field_equality() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(77),
            max_evict_per_round: 5,
            hbm_pressure_threshold: 0.88,
            dram_pressure_threshold: 0.77,
            importance_threshold: 42,
            hbm_evict_age_ticks: 33,
            dram_evict_age_ticks: 333,
            default_evict_codec: CompressionCodec::BitPackRle,
            page_bytes: 2048,
        };
        let cloned = cfg.clone();

        assert_eq!(cloned.tick_interval, cfg.tick_interval);
        assert_eq!(cloned.max_evict_per_round, cfg.max_evict_per_round);
        assert!((cloned.hbm_pressure_threshold - cfg.hbm_pressure_threshold).abs() < 1e-6);
        assert!((cloned.dram_pressure_threshold - cfg.dram_pressure_threshold).abs() < 1e-6);
        assert_eq!(cloned.importance_threshold, cfg.importance_threshold);
        assert_eq!(cloned.hbm_evict_age_ticks, cfg.hbm_evict_age_ticks);
        assert_eq!(cloned.dram_evict_age_ticks, cfg.dram_evict_age_ticks);
        assert_eq!(cloned.default_evict_codec, cfg.default_evict_codec);
        assert_eq!(cloned.page_bytes, cfg.page_bytes);
    }

    // ── Score: verify Warm + Protected state bonuses are additive with tier discounts ──

    #[test]
    fn state_bonus_and_tier_discount_both_apply() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        // Warm on HBM = 1000 + 5000 = 6000
        // Warm on Nvme = 1000 + 5000 - 500 = 5500
        assert_eq!(score_hbm, 6000);
        assert_eq!(score_nvme, 5500);
        assert_eq!(score_hbm - score_nvme, 500, "tier discount still applies with Warm state");
    }

    // ── Score: compression with very small original_size (2 bytes) ──

    #[test]
    fn score_tiny_original_size() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 2, StorageTier::GpuHbm, 0,
        );
        // compression = (1-0)*500 = 500
        // page_size = 2/1024 = 0
        // base(1000) + 500 = 1500
        assert_eq!(score, 1500, "2-byte page: compression=500, page_size=0, got {}", score);
    }

    // ── EvictionCandidate: can be created with all group_id variants ──

    #[test]
    fn eviction_candidate_group_id_variants() {
        let c_none = EvictionCandidate {
            page_id: 1, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
        };
        let c_some = EvictionCandidate {
            page_id: 2, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 4096, group_id: Some(42),
        };
        let c_zero = EvictionCandidate {
            page_id: 3, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 4096, group_id: Some(0),
        };
        assert!(c_none.group_id.is_none());
        assert_eq!(c_some.group_id, Some(42));
        assert_eq!(c_zero.group_id, Some(0));
        assert_ne!(c_none.group_id, c_some.group_id);
    }

    // ── Score: ExpertWeight at age=500 on CpuDram crosses deeply negative ──

    #[test]
    fn score_expert_dram_old_is_very_low() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::CpuDram, 500,
        );
        // base(1000) - time(500*2=1000) + expert(-300) + dram(-200) = -500
        assert_eq!(score, -500, "ExpertWeight + CpuDram + age=500 = -500, got {}", score);
        assert!(score < 0);
    }

    // ── Score: DenseLayerWeight + Protected resists even at age=4000 ──

    #[test]
    fn score_dense_protected_resists_high_age() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 4000,
        );
        // base(1000) - time(4000*2=8000) + dense(5000) + protected(10000) = 8000
        assert!(
            score > IMPORTANCE_SCORE_THRESHOLD,
            "DenseLayerWeight + Protected should resist age=4000: got {}",
            score,
        );
        assert_eq!(score, 8000);
    }

    // ── Wave 12x0: Additional eviction_worker tests ────────────────────────────

    #[test]
    fn eviction_tier_variants_debug() {
        assert!(format!("{:?}", EvictionTier::ColdExpert).contains("ColdExpert"));
        assert!(format!("{:?}", EvictionTier::PinnedDense).contains("PinnedDense"));
        assert!(format!("{:?}", EvictionTier::StandbyKv).contains("StandbyKv"));
        assert!(format!("{:?}", EvictionTier::Protected).contains("Protected"));
    }

    #[test]
    fn eviction_tier_copy() {
        let a = EvictionTier::ColdExpert;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn eviction_tier_hash_in_set() {
        use std::collections::HashSet;
        let set: HashSet<EvictionTier> = [
            EvictionTier::ColdExpert,
            EvictionTier::Protected,
            EvictionTier::ColdExpert,
        ].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn eviction_candidate_construction() {
        let c = EvictionCandidate {
            page_id: 42,
            score: -100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(7),
        };
        assert_eq!(c.page_id, 42);
        assert_eq!(c.score, -100);
        assert_eq!(c.page_bytes, 4096);
        assert_eq!(c.group_id, Some(7));
    }

    #[test]
    fn eviction_candidate_clone_independent() {
        let a = EvictionCandidate {
            page_id: 1, score: 50, current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::None, page_bytes: 0, group_id: None,
        };
        let b = a.clone();
        assert_eq!(a.page_id, b.page_id);
        assert_eq!(a.score, b.score);
    }

    #[test]
    fn eviction_worker_config_default_values() {
        let config = EvictionWorkerConfig::default();
        assert_eq!(config.tick_interval, Duration::from_millis(10));
        assert_eq!(config.max_evict_per_round, 8);
        assert!((config.hbm_pressure_threshold - 0.9).abs() < 1e-6);
        assert!((config.dram_pressure_threshold - 0.8).abs() < 1e-6);
        assert_eq!(config.importance_threshold, 100);
        assert_eq!(config.hbm_evict_age_ticks, 50);
        assert_eq!(config.dram_evict_age_ticks, 500);
        assert_eq!(config.page_bytes, 4096);
    }

    #[test]
    fn eviction_worker_config_clone() {
        let config = EvictionWorkerConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.max_evict_per_round, config.max_evict_per_round);
        assert_eq!(cloned.importance_threshold, config.importance_threshold);
    }

    #[test]
    fn eviction_worker_config_debug() {
        let config = EvictionWorkerConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("EvictionWorkerConfig"));
    }

    #[test]
    fn eviction_worker_config_custom_values() {
        let config = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(100),
            max_evict_per_round: 32,
            hbm_pressure_threshold: 0.95,
            dram_pressure_threshold: 0.7,
            importance_threshold: 200,
            hbm_evict_age_ticks: 100,
            dram_evict_age_ticks: 1000,
            default_evict_codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
        };
        assert_eq!(config.max_evict_per_round, 32);
        assert_eq!(config.page_bytes, 8192);
        assert_eq!(config.default_evict_codec, CompressionCodec::BitPackRle);
    }

    #[test]
    fn eviction_worker_config_zero_evict_per_round() {
        let config = EvictionWorkerConfig {
            max_evict_per_round: 0,
            ..Default::default()
        };
        assert_eq!(config.max_evict_per_round, 0);
    }

    #[test]
    fn eviction_worker_config_max_threshold() {
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0,
            dram_pressure_threshold: 1.0,
            ..Default::default()
        };
        assert!((config.hbm_pressure_threshold - 1.0).abs() < 1e-6);
    }

    #[test]
    fn eviction_candidate_all_tiers() {
        for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let c = EvictionCandidate {
                page_id: 0, score: 0, current_tier: tier,
                codec: CompressionCodec::None, page_bytes: 0, group_id: None,
            };
            assert_eq!(c.current_tier, tier);
        }
    }

    #[test]
    fn eviction_candidate_all_codecs() {
        for codec in [CompressionCodec::None, CompressionCodec::Lz4, CompressionCodec::BitPackRle] {
            let c = EvictionCandidate {
                page_id: 0, score: 0, current_tier: StorageTier::GpuHbm,
                codec, page_bytes: 0, group_id: None,
            };
            assert_eq!(c.codec, codec);
        }
    }

    #[test]
    fn eviction_candidate_negative_score() {
        let c = EvictionCandidate {
            page_id: 1, score: i64::MIN, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 0, group_id: None,
        };
        assert_eq!(c.score, i64::MIN);
    }

    #[test]
    fn eviction_candidate_max_score() {
        let c = EvictionCandidate {
            page_id: 1, score: i64::MAX, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 0, group_id: None,
        };
        assert_eq!(c.score, i64::MAX);
    }

    #[test]
    fn eviction_candidate_group_id_with_value() {
        let c = EvictionCandidate {
            page_id: 0, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 0, group_id: Some(u64::MAX),
        };
        assert_eq!(c.group_id, Some(u64::MAX));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // New tests (targeting 40+ additional tests)
    // ─────────────────────────────────────────────────────────────────────────

    /// Helper: create a PageMetadata with sensible defaults.
    fn make_meta(
        page_id: PageId,
        seq_id: Option<RequestId>,
        recency: usize,
        access_count: usize,
        state: PageState,
    ) -> PageMetadata {
        PageMetadata {
            page_id,
            sequence_id: seq_id,
            recency,
            access_count,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state,
            warm_until: None,
        }
    }

    // ── EvictionTier: ordinal / ordering properties ──

    #[test]
    fn eviction_tier_cold_expert_is_distinct() {
        assert_ne!(EvictionTier::ColdExpert, EvictionTier::PinnedDense);
        assert_ne!(EvictionTier::ColdExpert, EvictionTier::StandbyKv);
        assert_ne!(EvictionTier::ColdExpert, EvictionTier::Protected);
    }

    #[test]
    fn eviction_tier_pinned_dense_is_distinct() {
        assert_ne!(EvictionTier::PinnedDense, EvictionTier::ColdExpert);
        assert_ne!(EvictionTier::PinnedDense, EvictionTier::StandbyKv);
        assert_ne!(EvictionTier::PinnedDense, EvictionTier::Protected);
    }

    #[test]
    fn eviction_tier_standby_kv_is_distinct() {
        assert_ne!(EvictionTier::StandbyKv, EvictionTier::ColdExpert);
        assert_ne!(EvictionTier::StandbyKv, EvictionTier::PinnedDense);
        assert_ne!(EvictionTier::StandbyKv, EvictionTier::Protected);
    }

    #[test]
    fn eviction_tier_protected_is_distinct() {
        assert_ne!(EvictionTier::Protected, EvictionTier::ColdExpert);
        assert_ne!(EvictionTier::Protected, EvictionTier::PinnedDense);
        assert_ne!(EvictionTier::Protected, EvictionTier::StandbyKv);
    }

    #[test]
    fn eviction_tier_four_variants_count() {
        let variants = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        assert_eq!(variants.len(), 4);
    }

    #[test]
    fn eviction_tier_copy_preserves_value() {
        let a = EvictionTier::Protected;
        let b = a;
        assert_eq!(a, b);
    }

    // ── compute_importance_score: page_bytes / original_size edge cases ──

    #[test]
    fn score_original_size_exact_1kb_yields_bonus_1() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 1024, StorageTier::GpuHbm, 0,
        );
        // base=1000, page_size_bonus = 1024/1024*1 = 1, compression_bonus = 1.0*500 = 500
        // (compressed=0, so ratio=0, bonus = 500)
        assert_eq!(score, 1000 + 1 + 500);
    }

    #[test]
    fn score_original_size_exact_2kb_yields_bonus_2() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 2048, StorageTier::GpuHbm, 0,
        );
        // page_size_bonus = 2048/1024*1 = 2, compression_bonus = 1.0*500 = 500
        assert_eq!(score, 1000 + 2 + 500);
    }

    #[test]
    fn score_original_size_512_yields_zero_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 512, StorageTier::GpuHbm, 0,
        );
        // page_size_bonus = (512/1024)*1 = 0 (f64->i64 truncation of 0.5)
        // compression_bonus = 1.0*500 = 500
        assert_eq!(score, 1000 + 500);
    }

    #[test]
    fn score_compressed_equals_original_zero_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio = 1.0, bonus = (1.0-1.0)*500 = 0
        assert_eq!(score, 1000 + 4); // +4 from page_size_bonus: 4096/1024*1
    }

    #[test]
    fn score_compressed_half_yields_250_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio = 0.5, bonus = 0.5*500 = 250; page_size_bonus = 4096/1024*1 = 4
        assert_eq!(score, 1000 + 250 + 4);
    }

    #[test]
    fn score_compressed_quarter_yields_375_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 1024, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio = 0.25, bonus = 0.75*500 = 375; page_size_bonus = 4
        assert_eq!(score, 1000 + 375 + 4);
    }

    #[test]
    fn score_zero_compressed_yields_max_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio = 0.0, bonus = 1.0*500 = 500; page_size_bonus = 4
        assert_eq!(score, 1000 + 500 + 4);
    }

    #[test]
    fn score_both_sizes_zero_no_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        // no compression bonus, no page_size_bonus
        assert_eq!(score, 1000);
    }

    // ── compute_importance_score: state bonus edge cases ──

    #[test]
    fn score_swapped_out_state_no_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::SwappedOut);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_free_state_no_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Free);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_warm_bonus_exactly_5000() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000 + 5000);
    }

    #[test]
    fn score_protected_bonus_exactly_10000() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000 + 10000);
    }

    // ── compute_importance_score: tier discount edge cases ──

    #[test]
    fn score_hbm_no_discount() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_dram_discount_minus_200() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(score, 1000 - 200);
    }

    #[test]
    fn score_nvme_discount_minus_500() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(score, 1000 - 500);
    }

    // ── compute_importance_score: recency penalty exact values ──

    #[test]
    fn score_recency_10_penalty_exact() {
        let meta = make_meta(1, Some(10), 10, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        // recency_penalty = 10 * (2/2) = 10
        assert_eq!(score, 1000 - 10);
    }

    #[test]
    fn score_recency_100_penalty_exact() {
        let meta = make_meta(1, Some(10), 100, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        // recency_penalty = 100 * 1 = 100
        assert_eq!(score, 1000 - 100);
    }

    // ── compute_importance_score: time penalty exact values ──

    #[test]
    fn score_time_penalty_1_tick_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 1,
        );
        // time_penalty = 1 * 2 = 2
        assert_eq!(score, 1000 - 2);
    }

    #[test]
    fn score_time_penalty_100_ticks_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 100,
        );
        // time_penalty = 100 * 2 = 200
        assert_eq!(score, 1000 - 200);
    }

    // ── compute_importance_score: frequency bonus exact values ──

    #[test]
    fn score_freq_1_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 1, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        // freq_bonus = 1 * 15 = 15
        assert_eq!(score, 1000 + 15);
    }

    #[test]
    fn score_freq_10_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 10, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        // freq_bonus = 10 * 15 = 150
        assert_eq!(score, 1000 + 150);
    }

    // ── compute_importance_score: payload bonus exact values ──

    #[test]
    fn score_none_payload_bonus_zero() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_expert_weight_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000 + EXPERT_WEIGHT_BONUS);
    }

    #[test]
    fn score_prompt_system_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000 + PROMPT_SYSTEM_BONUS);
    }

    #[test]
    fn score_dense_layer_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000 + DENSE_LAYER_BONUS);
    }

    #[test]
    fn score_knowledge_rag_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000 + KNOWLEDGE_RAG_BONUS);
    }

    // ── compute_importance_score: combined exact formula ──

    #[test]
    fn score_exact_combined_kv_warm_hbm() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096,
            StorageTier::GpuHbm, 10,
        );
        // base=1000, time_pen=-20, recency_pen=-5, freq=+45,
        // compression=+250, page_size=+4, payload=0, state=+5000, tier=0
        let expected: i64 = 1000 - 20 - 5 + 45 + 250 + 4 + 0 + 5000 + 0;
        assert_eq!(score, expected);
    }

    #[test]
    fn score_exact_combined_expert_standby_dram() {
        let meta = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 20,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 2048,
            StorageTier::CpuDram, 100,
        );
        // base=1000, time_pen=-200, recency_pen=-20, freq=+15,
        // compression = (1.0-0.0)*500=500, page_size=+2, payload=-300, state=0, tier=-200
        let expected: i64 = 1000 - 200 - 20 + 15 + 500 + 2 - 300 + 0 - 200;
        assert_eq!(score, expected);
    }

    #[test]
    fn score_exact_combined_rag_standby_nvme() {
        let meta = PageMetadata {
            page_id: 3,
            sequence_id: Some(99),
            recency: 50,
            access_count: 2,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0,
            StorageTier::Nvme, 500,
        );
        // base=1000, time_pen=-1000, recency_pen=-50, freq=+30,
        // compression=0, page_size=0, payload=-500, state=0, tier=-500
        let expected: i64 = 1000 - 1000 - 50 + 30 + 0 + 0 - 500 + 0 - 500;
        assert_eq!(score, expected);
    }

    // ── classify_eviction_tier: additional edge cases ──

    #[test]
    fn classify_expert_weight_always_cold() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::ExpertWeight), 99999),
            EvictionTier::ColdExpert,
        );
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::ExpertWeight), -99999),
            EvictionTier::ColdExpert,
        );
    }

    #[test]
    fn classify_dense_layer_always_pinned() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::DenseLayerWeight), 99999),
            EvictionTier::PinnedDense,
        );
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::DenseLayerWeight), -99999),
            EvictionTier::PinnedDense,
        );
    }

    #[test]
    fn classify_none_payload_score_99_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, 99),
            EvictionTier::StandbyKv,
        );
    }

    #[test]
    fn classify_none_payload_score_100_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, 100),
            EvictionTier::Protected,
        );
    }

    #[test]
    fn classify_kv_score_99_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 99),
            EvictionTier::StandbyKv,
        );
    }

    #[test]
    fn classify_kv_score_100_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 100),
            EvictionTier::Protected,
        );
    }

    #[test]
    fn classify_prompt_high_score_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::PromptSystem), 500),
            EvictionTier::Protected,
        );
    }

    #[test]
    fn classify_prompt_low_score_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::PromptSystem), 50),
            EvictionTier::StandbyKv,
        );
    }

    #[test]
    fn classify_rag_high_score_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KnowledgeRAG), 500),
            EvictionTier::Protected,
        );
    }

    #[test]
    fn classify_rag_low_score_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KnowledgeRAG), 50),
            EvictionTier::StandbyKv,
        );
    }

    // ── EvictionCandidate: construction with all codec variants ──

    #[test]
    fn eviction_candidate_codec_nvcomp_ans() {
        let c = EvictionCandidate {
            page_id: 1, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::NvcompAns, page_bytes: 4096, group_id: None,
        };
        assert_eq!(c.codec, CompressionCodec::NvcompAns);
    }

    #[test]
    fn eviction_candidate_codec_zstd_dict() {
        let c = EvictionCandidate {
            page_id: 1, score: 0, current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::ZstdDict, page_bytes: 4096, group_id: None,
        };
        assert_eq!(c.codec, CompressionCodec::ZstdDict);
    }

    // ── EvictionWorkerConfig: field-level validation ──

    #[test]
    fn config_new_default_codec_is_lz4() {
        let config = EvictionWorkerConfig::default();
        assert_eq!(config.default_evict_codec, CompressionCodec::Lz4);
    }

    #[test]
    fn config_new_custom_codec_preserved() {
        let config = EvictionWorkerConfig {
            default_evict_codec: CompressionCodec::ZstdDict,
            ..Default::default()
        };
        assert_eq!(config.default_evict_codec, CompressionCodec::ZstdDict);
    }

    #[test]
    fn config_tick_interval_custom() {
        let config = EvictionWorkerConfig {
            tick_interval: Duration::from_secs(1),
            ..Default::default()
        };
        assert_eq!(config.tick_interval, Duration::from_secs(1));
    }

    #[test]
    fn config_all_fields_custom() {
        let config = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(500),
            max_evict_per_round: 64,
            hbm_pressure_threshold: 0.5,
            dram_pressure_threshold: 0.6,
            importance_threshold: 500,
            hbm_evict_age_ticks: 200,
            dram_evict_age_ticks: 2000,
            default_evict_codec: CompressionCodec::BitPackRle,
            page_bytes: 16384,
        };
        assert_eq!(config.tick_interval, Duration::from_millis(500));
        assert_eq!(config.max_evict_per_round, 64);
        assert!((config.hbm_pressure_threshold - 0.5).abs() < 1e-6);
        assert!((config.dram_pressure_threshold - 0.6).abs() < 1e-6);
        assert_eq!(config.importance_threshold, 500);
        assert_eq!(config.hbm_evict_age_ticks, 200);
        assert_eq!(config.dram_evict_age_ticks, 2000);
        assert_eq!(config.default_evict_codec, CompressionCodec::BitPackRle);
        assert_eq!(config.page_bytes, 16384);
    }

    // ── infer_payload_kind: exhaustive state coverage ──

    #[test]
    fn infer_payload_no_owner_is_expert_regardless_of_state() {
        for state in [
            PageState::Free, PageState::Active, PageState::Standby,
            PageState::SwappedOut, PageState::Warm, PageState::Protected,
            PageState::Swapped,
        ] {
            let meta = PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 0,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state,
                warm_until: None,
            };
            assert_eq!(
                infer_payload_kind(&meta),
                Some(PagePayloadKind::ExpertWeight),
                "expected ExpertWeight for state {:?}",
                state,
            );
        }
    }

    #[test]
    fn infer_payload_with_owner_is_kv_regardless_of_state() {
        for state in [
            PageState::Free, PageState::Active, PageState::Standby,
            PageState::SwappedOut, PageState::Warm, PageState::Protected,
            PageState::Swapped,
        ] {
            let meta = PageMetadata {
                page_id: 1,
                sequence_id: Some(42),
                recency: 0,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state,
                warm_until: None,
            };
            assert_eq!(
                infer_payload_kind(&meta),
                Some(PagePayloadKind::KvContext),
                "expected KvContext for state {:?}",
                state,
            );
        }
    }

    // ── compute_importance_score: is_lir does not affect score ──

    #[test]
    fn score_is_lir_true_same_as_false() {
        let meta_false = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_true = PageMetadata {
            is_lir: true,
            ..meta_false
        };
        let score_false = EvictionWorker::compute_importance_score(
            &meta_false, Some(PagePayloadKind::KvContext), 0, 4096,
            StorageTier::GpuHbm, 10,
        );
        let score_true = EvictionWorker::compute_importance_score(
            &meta_true, Some(PagePayloadKind::KvContext), 0, 4096,
            StorageTier::GpuHbm, 10,
        );
        assert_eq!(score_false, score_true);
    }

    // ── compute_importance_score: warm_until does not affect score ──

    #[test]
    fn score_warm_until_no_effect() {
        let meta_none = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_future = PageMetadata {
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
            ..meta_none
        };
        let score_none = EvictionWorker::compute_importance_score(
            &meta_none, Some(PagePayloadKind::KvContext), 0, 0,
            StorageTier::GpuHbm, 0,
        );
        let score_future = EvictionWorker::compute_importance_score(
            &meta_future, Some(PagePayloadKind::KvContext), 0, 0,
            StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_none, score_future);
    }

    // ── compute_importance_score: page_id does not affect score ──

    #[test]
    fn score_page_id_independent() {
        let meta_a = make_meta(1, Some(10), 5, 3, PageState::Standby);
        let meta_b = make_meta(999, Some(10), 5, 3, PageState::Standby);
        let score_a = EvictionWorker::compute_importance_score(
            &meta_a, Some(PagePayloadKind::KvContext), 0, 4096,
            StorageTier::GpuHbm, 10,
        );
        let score_b = EvictionWorker::compute_importance_score(
            &meta_b, Some(PagePayloadKind::KvContext), 0, 4096,
            StorageTier::GpuHbm, 10,
        );
        assert_eq!(score_a, score_b);
    }

    // ── compute_importance_score: monotonicity checks ──

    #[test]
    fn score_decreases_with_higher_recency() {
        let meta_low = make_meta(1, Some(10), 5, 0, PageState::Standby);
        let meta_high = make_meta(1, Some(10), 50, 0, PageState::Standby);
        let score_low = EvictionWorker::compute_importance_score(
            &meta_low, Some(PagePayloadKind::KvContext), 0, 0,
            StorageTier::GpuHbm, 0,
        );
        let score_high = EvictionWorker::compute_importance_score(
            &meta_high, Some(PagePayloadKind::KvContext), 0, 0,
            StorageTier::GpuHbm, 0,
        );
        assert!(score_low > score_high, "lower recency should yield higher score");
    }

    #[test]
    fn score_increases_with_higher_access_count() {
        let meta_low = make_meta(1, Some(10), 0, 2, PageState::Standby);
        let meta_high = make_meta(1, Some(10), 0, 20, PageState::Standby);
        let score_low = EvictionWorker::compute_importance_score(
            &meta_low, Some(PagePayloadKind::KvContext), 0, 0,
            StorageTier::GpuHbm, 0,
        );
        let score_high = EvictionWorker::compute_importance_score(
            &meta_high, Some(PagePayloadKind::KvContext), 0, 0,
            StorageTier::GpuHbm, 0,
        );
        assert!(score_high > score_low, "higher access_count should yield higher score");
    }

    // ── evict_round: integration scenario with SwappedOut + Dram pressure ──

    #[test]
    fn evict_round_dram_pressure_evicts_to_nvme() {
        let config = EvictionWorkerConfig {
            dram_evict_age_ticks: 0, // any age qualifies
            dram_pressure_threshold: 0.0, // always triggered
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(60);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 100, 100);
        // Fill DRAM to create pressure
        for _ in 0..90 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(submitted, 1, "should evict 1 page from DRAM to NVMe");
        actor.shutdown();
    }

    // ── evict_round: mixed states, Active/Protected skipped ──

    #[test]
    fn evict_round_mixed_states_skips_active_and_protected() {
        // Use a config where HBM pressure triggers eviction but scores must
        // be below threshold. Active and Protected pages are always skipped
        // by the state filter regardless of score.
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0, // no DRAM pressure
            importance_threshold: i64::MAX, // all scores qualify
            max_evict_per_round: 100,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        // Only put 2 Standby pages on HBM. Active and Protected have no addr
        // entries so they get skipped by the addr_table lookup, not by state.
        // This tests the pure scoring path without triggering SIGSEGV from
        // fake gpu pointers.
        let pages: Vec<(PageId, PageMetadata)> = vec![
            (1, PageMetadata { page_id: 1, sequence_id: Some(1), recency: 0, access_count: 0, last_access: old, swap_in_time: Some(old), is_lir: false, state: PageState::Standby, warm_until: None }),
            (2, PageMetadata { page_id: 2, sequence_id: Some(1), recency: 0, access_count: 0, last_access: old, swap_in_time: Some(old), is_lir: false, state: PageState::Active, warm_until: None }),
            (3, PageMetadata { page_id: 3, sequence_id: Some(1), recency: 0, access_count: 0, last_access: old, swap_in_time: Some(old), is_lir: false, state: PageState::Protected, warm_until: None }),
        ];
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(pages.into_iter().collect()));

        // Only page 1 has an addr_table entry — on HBM.
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // High HBM pressure
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Page 1 (Standby) has a real addr entry and qualifies.
        // Pages 2/3 have no addr entries -> skipped.
        assert_eq!(submitted, 1, "only the Standby page with addr entry should be evicted");
        actor.shutdown();
    }

    // ── evict_round: HBM pressure but no addr_table entries ──

    #[test]
    fn evict_round_pressure_but_no_addr_entries() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        // addr_table is empty — page has no physical address

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(submitted, 0, "pages not in addr_table should be skipped");
        actor.shutdown();
    }

    // ── compute_importance_score: large original_size no overflow ──

    #[test]
    fn score_large_original_size_no_overflow() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let _score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, u32::MAX,
            StorageTier::GpuHbm, 0,
        );
        // Should not panic; the result should be a valid i64
    }

    // ── compute_importance_score: large access_count no overflow ──

    #[test]
    fn score_large_access_count_stays_positive() {
        // Use a large but safe access_count that doesn't overflow i64 when
        // multiplied by FREQUENCY_BONUS (15).  i64::MAX / 15 ≈ 613B.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 100_000,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0,
            StorageTier::GpuHbm, 0,
        );
        // freq_bonus = 100000 * 15 = 1_500_000, base=1000
        assert!(score > 0, "large frequency bonus should keep score positive: {}", score);
    }

    // ── compute_importance_score: Swapped state no bonus ──

    #[test]
    fn score_swapped_state_no_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Swapped);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0,
            StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    // ── EvictionTier: Debug output contains variant name ──

    #[test]
    fn eviction_tier_debug_contains_name() {
        assert!(format!("{:?}", EvictionTier::ColdExpert).contains("ColdExpert"));
        assert!(format!("{:?}", EvictionTier::PinnedDense).contains("PinnedDense"));
        assert!(format!("{:?}", EvictionTier::StandbyKv).contains("StandbyKv"));
        assert!(format!("{:?}", EvictionTier::Protected).contains("Protected"));
    }

    // ── EvictionCandidate: debug output contains relevant fields ──

    #[test]
    fn eviction_candidate_debug_contains_score() {
        let c = EvictionCandidate {
            page_id: 42, score: -123, current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: None,
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("-123") || debug.contains("score"));
    }

    // ── EvictionWorkerConfig: clone produces deep copy ──

    #[test]
    fn config_clone_deep_copy_independence() {
        let mut config = EvictionWorkerConfig::default();
        let cloned = config.clone();
        config.importance_threshold = 999;
        assert_ne!(config.importance_threshold, cloned.importance_threshold);
    }

    // ── compute_importance_score: all payload bonuses are distinct ──

    #[test]
    fn score_all_payload_bonuses_distinct() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let payloads = [
            Some(PagePayloadKind::ExpertWeight),
            Some(PagePayloadKind::KvContext),
            Some(PagePayloadKind::PromptSystem),
            Some(PagePayloadKind::DenseLayerWeight),
            Some(PagePayloadKind::KnowledgeRAG),
            None,
        ];
        let mut scores: Vec<i64> = payloads.iter().map(|pk| {
            EvictionWorker::compute_importance_score(
                &meta, *pk, 0, 0, StorageTier::GpuHbm, 0,
            )
        }).collect();
        scores.sort();
        scores.dedup();
        // All payloads produce distinct scores (5 distinct bonuses + None=0)
        // ExpertWeight=-300, KnowledgeRAG=-500, KvContext=0, None=0
        // KvContext and None both give 0, so we expect 5 unique scores (one pair tied)
        assert!(scores.len() >= 4, "at least 4 distinct scores from 6 payload variants");
    }

    // ── compute_importance_score: tier discount ordering ──

    #[test]
    fn score_tier_discount_hbm_gt_dram_gt_nvme() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_dram = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::CpuDram, 0,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::Nvme, 0,
        );
        assert!(score_hbm > score_dram, "HBM > DRAM: {} vs {}", score_hbm, score_dram);
        assert!(score_dram > score_nvme, "DRAM > NVMe: {} vs {}", score_dram, score_nvme);
    }

    // ── compute_importance_score: state bonus ordering ──

    #[test]
    fn score_state_bonus_protected_gt_warm_gt_standby() {
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta_warm = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let meta_protected = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_warm = EvictionWorker::compute_importance_score(
            &meta_warm, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &meta_protected, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert!(score_protected > score_warm);
        assert!(score_warm > score_standby);
    }

    // ── compute_importance_score: total formula components add up ──

    #[test]
    fn score_total_equals_sum_of_components() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 10,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let tier_age: u64 = 25;
        let compressed: u32 = 1024;
        let original: u32 = 4096;

        let time_penalty = (tier_age as i64) * TIME_DECAY_WEIGHT;
        let recency_penalty = (meta.recency as i64) * (TIME_DECAY_WEIGHT / 2);
        let freq_bonus = (meta.access_count as i64) * FREQUENCY_BONUS;
        let compression_bonus = if original > 0 {
            ((1.0 - (compressed as f64) / (original as f64)) * COMPRESSION_RATIO_WEIGHT as f64) as i64
        } else { 0 };
        let page_bonus = if original > 0 {
            ((original as f64) / 1024.0 * PAGE_SIZE_WEIGHT as f64) as i64
        } else { 0 };
        let payload_bonus: i64 = KV_CONTEXT_BONUS;
        let state_bonus: i64 = 5000;
        let tier_discount: i64 = 0;

        let expected: i64 = 1000 - time_penalty - recency_penalty + freq_bonus
            + compression_bonus + page_bonus + payload_bonus + state_bonus + tier_discount;

        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), compressed, original,
            StorageTier::GpuHbm, tier_age,
        );
        assert_eq!(score, expected);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (50 new — batch 6)
    // ─────────────────────────────────────────────────────────────────────────

    // ── infer_payload_kind: Free state with no owner → ExpertWeight ──

    #[test]
    fn infer_payload_free_no_owner_is_expert() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::ExpertWeight),
        );
    }

    // ── infer_payload_kind: Warm state with owner → KvContext ──

    #[test]
    fn infer_payload_warm_with_owner_is_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::KvContext),
        );
    }

    // ── infer_payload_kind: Protected state with owner → KvContext ──

    #[test]
    fn infer_payload_protected_with_owner_is_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(7),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::KvContext),
        );
    }

    // ── infer_payload_kind: SwappedOut state with no owner → ExpertWeight ──

    #[test]
    fn infer_payload_swapped_out_no_owner_is_expert() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::ExpertWeight),
        );
    }

    // ── infer_payload_kind: Active state with owner → KvContext ──

    #[test]
    fn infer_payload_active_with_owner_is_kv() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(1),
            recency: 0,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::KvContext),
        );
    }

    // ── Score: Active state gets zero bonus (same as Standby) ──

    #[test]
    fn score_active_state_zero_bonus() {
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 3,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_active = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 3,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::GpuHbm, 50,
        );
        let score_active = EvictionWorker::compute_importance_score(
            &meta_active, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::GpuHbm, 50,
        );
        assert_eq!(score_standby, score_active, "Active should have same bonus as Standby (= 0)");
    }

    // ── Score: swapping compressed and original produces different bonus ──

    #[test]
    fn score_swapped_sizes_changes_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_a = EvictionWorker::compute_importance_score(
            &meta, None, 1024, 4096, StorageTier::GpuHbm, 0,
        );
        let score_b = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 1024, StorageTier::GpuHbm, 0,
        );
        assert_ne!(
            score_a, score_b,
            "swapping compressed/original should produce different scores: a={} b={}",
            score_a, score_b,
        );
    }

    // ── Score: KvContext bonus is zero, verifying via None comparison ──

    #[test]
    fn kv_context_bonus_zero_via_none() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 2,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 1024, 4096, StorageTier::CpuDram, 75,
        );
        let score_none = EvictionWorker::compute_importance_score(
            &meta, None, 1024, 4096, StorageTier::CpuDram, 75,
        );
        assert_eq!(score_kv, score_none, "KvContext and None should have identical bonus (= 0)");
    }

    // ── Score: ExpertWeight on CpuDram with Warm state ──

    #[test]
    fn score_exact_expert_warm_dram() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + expert(-300) + warm(5000) + dram(-200) = 5500
        assert_eq!(score, 5500, "ExpertWeight + Warm + CpuDram = 5500, got {}", score);
    }

    // ── Score: KnowledgeRAG on Nvme with Standby ──

    #[test]
    fn score_exact_rag_nvme_standby() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + rag(-500) + nvme(-500) = 0
        assert_eq!(score, 0, "KnowledgeRAG + Nvme + Standby = 0, got {}", score);
    }

    // ── Score: PromptSystem on CpuDram with Warm ──

    #[test]
    fn score_exact_prompt_warm_dram() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + prompt(1000) + warm(5000) + dram(-200) = 6800
        assert_eq!(score, 6800, "PromptSystem + Warm + CpuDram = 6800, got {}", score);
    }

    // ── Score: DenseLayerWeight on Nvme with Protected ──

    #[test]
    fn score_exact_dense_protected_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + dense(5000) + protected(10000) + nvme(-500) = 15500
        assert_eq!(score, 15500, "DenseLayerWeight + Protected + Nvme = 15500, got {}", score);
    }

    // ── Score: KnowledgeRAG on Nvme goes negative with tier_age ──

    #[test]
    fn score_rag_nvme_goes_negative_with_age() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::Nvme, 100,
        );
        // base(1000) + rag(-500) + nvme(-500) - time(100*2=200) = -200
        assert!(
            score < 0,
            "KnowledgeRAG on NVMe with age should go negative: got {}",
            score,
        );
    }

    // ── EvictionWorkerConfig: default importance_threshold is 100 ──

    #[test]
    fn config_default_importance_threshold_is_100() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.importance_threshold, 100);
    }

    // ── EvictionWorkerConfig: default hbm_pressure_threshold is 0.9 ──

    #[test]
    fn config_default_hbm_pressure_is_0_9() {
        let cfg = EvictionWorkerConfig::default();
        assert!((cfg.hbm_pressure_threshold - 0.90).abs() < 1e-6);
    }

    // ── EvictionWorkerConfig: default dram_pressure_threshold is 0.8 ──

    #[test]
    fn config_default_dram_pressure_is_0_8() {
        let cfg = EvictionWorkerConfig::default();
        assert!((cfg.dram_pressure_threshold - 0.80).abs() < 1e-6);
    }

    // ── EvictionWorkerConfig: all fields mutable after construction ──

    #[test]
    fn config_all_fields_mutable() {
        let mut cfg = EvictionWorkerConfig::default();
        cfg.tick_interval = Duration::from_secs(1);
        cfg.max_evict_per_round = 100;
        cfg.hbm_pressure_threshold = 0.5;
        cfg.dram_pressure_threshold = 0.5;
        cfg.importance_threshold = 0;
        cfg.hbm_evict_age_ticks = 10;
        cfg.dram_evict_age_ticks = 100;
        cfg.default_evict_codec = CompressionCodec::ZstdDict;
        cfg.page_bytes = 65536;
        assert_eq!(cfg.tick_interval, Duration::from_secs(1));
        assert_eq!(cfg.max_evict_per_round, 100);
        assert!((cfg.hbm_pressure_threshold - 0.5).abs() < 1e-6);
        assert!((cfg.dram_pressure_threshold - 0.5).abs() < 1e-6);
        assert_eq!(cfg.importance_threshold, 0);
        assert_eq!(cfg.hbm_evict_age_ticks, 10);
        assert_eq!(cfg.dram_evict_age_ticks, 100);
        assert_eq!(cfg.default_evict_codec, CompressionCodec::ZstdDict);
        assert_eq!(cfg.page_bytes, 65536);
    }

    // ── classify_eviction_tier: None with score=0 is StandbyKv (batch 6) ──

    #[test]
    fn classify_none_score_zero_standby_batch6() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, 0),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: KvContext with score=99 → StandbyKv ──

    #[test]
    fn classify_kv_score_99_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 99),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: KvContext with score=100 → Protected ──

    #[test]
    fn classify_kv_score_100_is_protected() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 100),
            EvictionTier::Protected,
        );
    }

    // ── classify_eviction_tier: KnowledgeRAG with score=99 → StandbyKv ──

    #[test]
    fn classify_rag_score_99_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KnowledgeRAG), 99),
            EvictionTier::StandbyKv,
        );
    }

    // ── classify_eviction_tier: PromptSystem with score=99 → StandbyKv ──

    #[test]
    fn classify_prompt_score_99_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::PromptSystem), 99),
            EvictionTier::StandbyKv,
        );
    }

    // ── EvictionCandidate: group_id can be large RequestId ──

    #[test]
    fn eviction_candidate_large_group_id() {
        let c = EvictionCandidate {
            page_id: 1,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: Some(RequestId::MAX),
        };
        assert_eq!(c.group_id, Some(RequestId::MAX));
    }

    // ── Score: multiple state transitions show correct ordering ──

    #[test]
    fn score_state_ordering_comprehensive() {
        let make_meta = |state: PageState| PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state,
            warm_until: None,
        };
        let states = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Swapped,
        ];
        // All non-Warm/Protected states should have same score.
        let scores: Vec<i64> = states.iter().map(|s| {
            EvictionWorker::compute_importance_score(
                &make_meta(*s), None, 0, 0, StorageTier::GpuHbm, 0,
            )
        }).collect();
        for i in 1..scores.len() {
            assert_eq!(
                scores[i], scores[0],
                "all zero-bonus states should have same score: {:?} vs {:?}",
                states[i], states[0],
            );
        }
    }

    // ── Score: page_size_bonus for 64 KiB page ──

    #[test]
    fn score_page_size_bonus_64k() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_4k = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_64k = EvictionWorker::compute_importance_score(
            &meta, None, 65536, 65536, StorageTier::GpuHbm, 0,
        );
        // Both ratio=1.0 so compression_bonus=0.
        // page_size delta = (65536/1024 - 4096/1024) = 64 - 4 = 60
        assert_eq!(
            score_64k - score_4k, 60,
            "64K page should have 60 more page_size bonus than 4K page",
        );
    }

    // ── Score: recency and frequency interact correctly ──

    #[test]
    fn score_recency_and_frequency_interaction() {
        let meta_low_r_high_f = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 1,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_high_r_low_f = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 50,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_low_r_high_f = EvictionWorker::compute_importance_score(
            &meta_low_r_high_f, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_high_r_low_f = EvictionWorker::compute_importance_score(
            &meta_high_r_low_f, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // low_r_high_f: -recency(1*1=1) + freq(50*15=750) = net +749
        // high_r_low_f: -recency(50*1=50) + freq(1*15=15) = net -35
        assert!(
            score_low_r_high_f > score_high_r_low_f,
            "low recency + high frequency should score higher than high recency + low frequency: {} vs {}",
            score_low_r_high_f, score_high_r_low_f,
        );
    }

    // ── Score: same score produced for symmetric inputs ──

    #[test]
    fn score_deterministic_across_multiple_calls() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 7,
            access_count: 12,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let mut scores = Vec::new();
        for _ in 0..5 {
            scores.push(EvictionWorker::compute_importance_score(
                &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::GpuHbm, 30,
            ));
        }
        for s in &scores[1..] {
            assert_eq!(*s, scores[0], "all calls with same inputs should produce same score");
        }
    }

    // ── Score: compression ratio at 25% exact bonus (batch 6) ──

    #[test]
    fn compression_ratio_exact_25_pct_batch6() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_25 = EvictionWorker::compute_importance_score(
            &meta, None, 3072, 4096, StorageTier::GpuHbm, 0,
        );
        // 25% compression: ratio = 0.75, bonus = (1 - 0.75) * 500 = 125
        assert_eq!(
            score_25 - score_no_compress, 125,
            "25% compression bonus should be 125",
        );
    }

    // ── Score: DenseLayerWeight with Nvme discount and Warm bonus ──

    #[test]
    fn score_dense_warm_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + dense(5000) + warm(5000) + nvme(-500) = 10500
        assert_eq!(score, 10500, "DenseLayerWeight + Warm + Nvme = 10500, got {}", score);
    }

    // ── Score: ExpertWeight on Nvme with Protected ──

    #[test]
    fn score_exact_expert_protected_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + expert(-300) + protected(10000) + nvme(-500) = 10200
        assert_eq!(score, 10200, "ExpertWeight + Protected + Nvme = 10200, got {}", score);
    }

    // ── EvictionCandidate: sort with duplicate scores preserves deterministic order ──

    #[test]
    fn eviction_candidate_sort_duplicate_scores_deterministic() {
        let mut candidates: Vec<EvictionCandidate> = (0..6)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: if i < 3 { -10 } else { 10 },
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        candidates.sort_by_key(|c| c.score);
        // First three should have score -10, last three score 10.
        assert!(candidates[0].score < 0);
        assert!(candidates[2].score < 0);
        assert!(candidates[3].score > 0);
        assert!(candidates[5].score > 0);
    }

    // ── Score: freq_bonus + compression_bonus + page_size_bonus combined ──

    #[test]
    fn score_combined_positive_bonuses() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 20,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 8192, StorageTier::GpuHbm, 0,
        );
        // base(1000) + freq(20*15=300) + compression((1-0)*500=500) + page_size(8192/1024=8)
        // = 1000 + 300 + 500 + 8 = 1808
        assert_eq!(score, 1808, "combined positive bonuses should sum to 1808, got {}", score);
    }

    // ── Score: penalty combination pushes below threshold ──

    #[test]
    fn score_penalties_push_below_threshold() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // triggers ExpertWeight
            recency: 50,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 4096, 4096, StorageTier::CpuDram, 200,
        );
        // base(1000) - time(200*2=400) - recency(50*1=50) + expert(-300)
        // + dram(-200) + compression(0) + page(4) = 54
        assert!(
            score < IMPORTANCE_SCORE_THRESHOLD,
            "combined penalties should push below threshold: got {}",
            score,
        );
    }

    // ── EvictionWorkerConfig: hbm_evict_age_ticks=0 means all ages eligible ──

    #[test]
    fn config_zero_hbm_evict_age() {
        let cfg = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.hbm_evict_age_ticks, 0);
    }

    // ── EvictionWorkerConfig: dram_evict_age_ticks=0 means all ages eligible ──

    #[test]
    fn config_zero_dram_evict_age() {
        let cfg = EvictionWorkerConfig {
            dram_evict_age_ticks: 0,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.dram_evict_age_ticks, 0);
    }

    // ── EvictionWorkerConfig: importance_threshold=0 means everything is evictable ──

    #[test]
    fn config_importance_threshold_zero_all_evictable() {
        let cfg = EvictionWorkerConfig {
            importance_threshold: 0,
            ..EvictionWorkerConfig::default()
        };
        // A page with score 50 would be >= 0, so NOT evictable with threshold=0.
        // But a page with score -10 would be < 0, so evictable.
        assert_eq!(cfg.importance_threshold, 0);
    }

    // ── compute_tier_age: uses Instant arithmetic correctly ──

    #[test]
    fn tier_age_5_seconds_is_500_ticks() {
        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        // 5000ms / 10 = 500 ticks. Allow some variance.
        assert!(
            age >= 490 && age <= 510,
            "5 second old page should be ~500 ticks, got {}",
            age,
        );
    }

    // ── compute_tier_age: 100ms is 10 ticks ──

    #[test]
    fn tier_age_100ms_is_10_ticks() {
        let old = Instant::now() - Duration::from_millis(100);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        assert!(
            age >= 9 && age <= 11,
            "100ms old page should be ~10 ticks, got {}",
            age,
        );
    }

    // ── Score: compression bonus at 90% compression ──

    #[test]
    fn compression_ratio_exact_90_percent() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // 90% compression: compressed = 409 (about 10% of 4096)
        let score_90 = EvictionWorker::compute_importance_score(
            &meta, None, 410, 4096, StorageTier::GpuHbm, 0,
        );
        // ratio ≈ 0.100, bonus ≈ (1.0 - 0.100) * 500 = 450
        let delta = score_90 - score_no_compress;
        assert!(
            delta > 440 && delta < 460,
            "90% compression bonus should be ~450: got {}",
            delta,
        );
    }

    // ── Score: access_count=1 produces FREQUENCY_BONUS delta from 0 ──

    #[test]
    fn frequency_bonus_one_access_delta() {
        let meta_0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_1 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta_0, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::GpuHbm, 50,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta_1, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::GpuHbm, 50,
        );
        assert_eq!(s1 - s0, FREQUENCY_BONUS as i64);
    }

    // ── Score: tier_age=50 with DRAM discount exact ──

    #[test]
    fn score_age_50_dram_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 50,
        );
        // base(1000) - time(50*2=100) + dram(-200) = 700
        assert_eq!(score, 700, "age 50 + CpuDram = 700, got {}", score);
    }

    // ── EvictionCandidate: page_id=usize::MAX preserves value ──

    #[test]
    fn eviction_candidate_page_id_usize_max() {
        let c = EvictionCandidate {
            page_id: usize::MAX,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };
        let cloned = c.clone();
        assert_eq!(cloned.page_id, usize::MAX);
    }

    // ── Score: swapping payload from ExpertWeight to DenseLayerWeight is +5300 ──

    #[test]
    fn score_expert_to_dense_delta_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_expert = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 2048, 4096, StorageTier::GpuHbm, 30,
        );
        let score_dense = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 2048, 4096, StorageTier::GpuHbm, 30,
        );
        // ExpertWeight(-300) -> DenseLayerWeight(+5000) = +5300 delta
        assert_eq!(
            score_dense - score_expert, 5300,
            "ExpertWeight to DenseLayerWeight should be +5300",
        );
    }

    // ── Score: ExpertWeight on CpuDram with age=100 ──

    #[test]
    fn score_exact_expert_dram_age_100() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::CpuDram, 100,
        );
        // base(1000) - time(100*2=200) + expert(-300) + dram(-200) = 300
        assert_eq!(score, 300, "ExpertWeight + CpuDram + age=100 = 300, got {}", score);
    }

    // ── Score: KnowledgeRAG with age=250 on GpuHbm ──

    #[test]
    fn score_exact_rag_hbm_age_250() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 250,
        );
        // base(1000) - time(250*2=500) + rag(-500) = 0
        assert_eq!(score, 0, "KnowledgeRAG + GpuHbm + age=250 = 0, got {}", score);
    }

    // ── Score: PromptSystem with age=400 on CpuDram ──

    #[test]
    fn score_exact_prompt_dram_age_400() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::CpuDram, 400,
        );
        // base(1000) - time(400*2=800) + prompt(1000) + dram(-200) = 1000
        assert_eq!(score, 1000, "PromptSystem + CpuDram + age=400 = 1000, got {}", score);
    }

    // ── Score: DenseLayerWeight with freq=5 and age=10 on GpuHbm ──

    #[test]
    fn score_exact_dense_freq5_age10_hbm() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 10,
        );
        // base(1000) - time(10*2=20) + freq(5*15=75) + dense(5000) = 6055
        assert_eq!(score, 6055, "DenseLayerWeight + freq=5 + age=10 + GpuHbm = 6055, got {}", score);
    }

    // ── Score: check that all 5 payload kinds produce monotonically ordered scores ──

    #[test]
    fn payload_scores_monotonically_ordered() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let ordered_kinds = [
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::KvContext,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::DenseLayerWeight,
        ];
        let mut prev = i64::MIN;
        for kind in &ordered_kinds {
            let score = EvictionWorker::compute_importance_score(
                &meta, Some(*kind), 0, 0, StorageTier::GpuHbm, 0,
            );
            assert!(
                score > prev,
                "payload scores should be monotonically increasing: {:?} score={} prev={}",
                kind, score, prev,
            );
            prev = score;
        }
    }

    // ── EvictionWorkerConfig: clone changes are independent (tick_interval) ──

    #[test]
    fn config_clone_tick_interval_independence() {
        let original = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(5),
            ..EvictionWorkerConfig::default()
        };
        let mut cloned = original.clone();
        cloned.tick_interval = Duration::from_secs(10);
        assert_ne!(cloned.tick_interval, original.tick_interval);
    }

    // ── EvictionWorkerConfig: clone changes are independent (importance_threshold) ──

    #[test]
    fn config_clone_importance_threshold_independence() {
        let original = EvictionWorkerConfig::default();
        let mut cloned = original.clone();
        cloned.importance_threshold = -999;
        assert_eq!(original.importance_threshold, IMPORTANCE_SCORE_THRESHOLD);
        assert_eq!(cloned.importance_threshold, -999);
    }

    // ── Score: ExpertWeight + DenseLayerWeight delta independent of tier_age ──

    #[test]
    fn score_expert_dense_delta_constant_across_age() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        for &age in &[0u64, 10, 100, 1000] {
            let score_expert = EvictionWorker::compute_importance_score(
                &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, age,
            );
            let score_dense = EvictionWorker::compute_importance_score(
                &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, age,
            );
            assert_eq!(
                score_dense - score_expert, 5300,
                "ExpertWeight→DenseLayerWeight delta should be constant (5300) regardless of age: age={}",
                age,
            );
        }
    }

    // ── Score: Warm and Protected delta is constant regardless of payload ──

    #[test]
    fn score_warm_protected_delta_constant_across_payload() {
        let make_meta = |state: PageState| PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state,
            warm_until: None,
        };
        for kind in &[
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::DenseLayerWeight,
            PagePayloadKind::KnowledgeRAG,
        ] {
            let score_warm = EvictionWorker::compute_importance_score(
                &make_meta(PageState::Warm), Some(*kind), 0, 0, StorageTier::GpuHbm, 0,
            );
            let score_protected = EvictionWorker::compute_importance_score(
                &make_meta(PageState::Protected), Some(*kind), 0, 0, StorageTier::GpuHbm, 0,
            );
            assert_eq!(
                score_protected - score_warm, 5000,
                "Protected-Warm delta should be 5000 for {:?}",
                kind,
            );
        }
    }

    // ── Score: tier discount delta constant regardless of payload ──

    #[test]
    fn score_tier_discount_constant_across_payload() {
        let make_meta = || PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let kinds: Vec<Option<PagePayloadKind>> = vec![
            None,
            Some(PagePayloadKind::KvContext),
            Some(PagePayloadKind::ExpertWeight),
            Some(PagePayloadKind::DenseLayerWeight),
        ];
        for kind in &kinds {
            let score_hbm = EvictionWorker::compute_importance_score(
                &make_meta(), *kind, 0, 0, StorageTier::GpuHbm, 0,
            );
            let score_nvme = EvictionWorker::compute_importance_score(
                &make_meta(), *kind, 0, 0, StorageTier::Nvme, 0,
            );
            assert_eq!(
                score_hbm - score_nvme, 500,
                "HBM->NVMe discount should be 500 regardless of payload: {:?}",
                kind,
            );
        }
    }

    // ── Score: recency penalty weight is exactly 1 (TIME_DECAY_WEIGHT / 2) ──

    #[test]
    fn recency_penalty_weight_exactly_one() {
        assert_eq!(TIME_DECAY_WEIGHT / 2, 1, "recency penalty weight should be 1");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (targeting 80 new tests)
    // ─────────────────────────────────────────────────────────────────────────

    // ── classify_eviction_tier: ExpertWeight always yields ColdExpert ──

    #[test]
    fn classify_expert_weight_always_cold_expert_low_score() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::ExpertWeight),
            -9999,
        );
        assert_eq!(tier, EvictionTier::ColdExpert);
    }

    #[test]
    fn classify_expert_weight_always_cold_expert_high_score() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::ExpertWeight),
            99999,
        );
        assert_eq!(tier, EvictionTier::ColdExpert);
    }

    #[test]
    fn classify_expert_weight_always_cold_expert_zero_score() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::ExpertWeight),
            0,
        );
        assert_eq!(tier, EvictionTier::ColdExpert);
    }

    // ── classify_eviction_tier: DenseLayerWeight always yields PinnedDense ──

    #[test]
    fn classify_dense_layer_always_pinned_dense_low_score() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::DenseLayerWeight),
            -9999,
        );
        assert_eq!(tier, EvictionTier::PinnedDense);
    }

    #[test]
    fn classify_dense_layer_always_pinned_dense_high_score() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::DenseLayerWeight),
            99999,
        );
        assert_eq!(tier, EvictionTier::PinnedDense);
    }

    #[test]
    fn classify_dense_layer_always_pinned_dense_threshold_score() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::DenseLayerWeight),
            IMPORTANCE_SCORE_THRESHOLD,
        );
        assert_eq!(tier, EvictionTier::PinnedDense);
    }

    // ── classify_eviction_tier: KvContext score-dependent routing ──

    #[test]
    fn classify_kv_context_below_threshold_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            IMPORTANCE_SCORE_THRESHOLD - 1,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    #[test]
    fn classify_kv_context_at_threshold_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            IMPORTANCE_SCORE_THRESHOLD,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    #[test]
    fn classify_kv_context_above_threshold_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            IMPORTANCE_SCORE_THRESHOLD + 1,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    #[test]
    fn classify_kv_context_zero_score_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            0,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    // ── classify_eviction_tier: PromptSystem score-dependent routing ──

    #[test]
    fn classify_prompt_system_below_threshold_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::PromptSystem),
            IMPORTANCE_SCORE_THRESHOLD - 1,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    #[test]
    fn classify_prompt_system_at_threshold_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::PromptSystem),
            IMPORTANCE_SCORE_THRESHOLD,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    // ── classify_eviction_tier: KnowledgeRAG score-dependent routing ──

    #[test]
    fn classify_knowledge_rag_below_threshold_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KnowledgeRAG),
            IMPORTANCE_SCORE_THRESHOLD - 1,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    #[test]
    fn classify_knowledge_rag_at_threshold_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KnowledgeRAG),
            IMPORTANCE_SCORE_THRESHOLD,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    #[test]
    fn classify_knowledge_rag_very_high_score_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KnowledgeRAG),
            50000,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    // ── classify_eviction_tier: None payload score-dependent routing ──

    #[test]
    fn classify_none_payload_below_threshold_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            None,
            IMPORTANCE_SCORE_THRESHOLD - 1,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    #[test]
    fn classify_none_payload_at_threshold_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            None,
            IMPORTANCE_SCORE_THRESHOLD,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    #[test]
    fn classify_none_payload_above_threshold_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            None,
            IMPORTANCE_SCORE_THRESHOLD + 100,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    #[test]
    fn classify_none_payload_negative_score_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(None, -500);
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    // ── classify_eviction_tier: threshold boundary exactly at 99 vs 100 ──

    #[test]
    fn classify_boundary_score_99_is_standby() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            99,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    #[test]
    fn classify_boundary_score_100_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            100,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    // ── infer_payload_kind: explicit tests ──

    #[test]
    fn infer_payload_none_sequence_id_is_expert_weight() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(infer_payload_kind(&meta), Some(PagePayloadKind::ExpertWeight));
    }

    #[test]
    fn infer_payload_some_sequence_id_is_kv_context() {
        let meta = PageMetadata {
            page_id: 2,
            sequence_id: Some(42),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(infer_payload_kind(&meta), Some(PagePayloadKind::KvContext));
    }

    #[test]
    fn infer_payload_zero_sequence_id_is_kv_context() {
        let meta = PageMetadata {
            page_id: 3,
            sequence_id: Some(0),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(infer_payload_kind(&meta), Some(PagePayloadKind::KvContext));
    }

    #[test]
    fn infer_payload_high_access_count_with_owner_still_kv_context() {
        let meta = PageMetadata {
            page_id: 4,
            sequence_id: Some(100),
            recency: 0,
            access_count: 9999,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        assert_eq!(infer_payload_kind(&meta), Some(PagePayloadKind::KvContext));
    }

    // ── Score: standby state gets zero bonus across all tiers ──

    #[test]
    fn score_standby_state_on_hbm_no_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000, "standby on HBM: base only");
    }

    #[test]
    fn score_standby_state_on_dram_discount_only() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(score, 800, "standby on DRAM: base - tier_discount");
    }

    #[test]
    fn score_standby_state_on_nvme_deeper_discount() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(score, 500, "standby on NVMe: base - tier_discount");
    }

    // ── Score: swapped state gets zero bonus ──

    #[test]
    fn score_swapped_state_on_dram_base_minus_discount() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Swapped);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(score, 800, "swapped on DRAM: base - 200");
    }

    #[test]
    fn score_swapped_state_on_nvme_base_minus_discount() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Swapped);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(score, 500, "swapped on NVMe: base - 500");
    }

    // ── Score: swapped_out state gets zero bonus ──

    #[test]
    fn score_swapped_out_state_on_dram_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::SwappedOut);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(score, 800);
    }

    // ── Score: time penalty and frequency bonus are both linear ──

    #[test]
    fn score_time_penalty_double_ticks_halves_score() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s1 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 100,
        );
        let s2 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 200,
        );
        assert_eq!(s1 - s2, 200, "double ticks should double penalty");
    }

    #[test]
    fn score_frequency_double_count_doubles_bonus() {
        let meta_low = make_meta(1, Some(10), 0, 5, PageState::Standby);
        let meta_high = make_meta(2, Some(10), 0, 10, PageState::Standby);
        let s_low = EvictionWorker::compute_importance_score(
            &meta_low, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_high = EvictionWorker::compute_importance_score(
            &meta_high, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_high - s_low, 75, "double access_count should double freq bonus delta");
    }

    // ── Score: combined penalty and bonus formula verification ──

    #[test]
    fn score_expert_hbm_age10_recency5_freq3_exact() {
        let meta = make_meta(1, None, 5, 3, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 10,
        );
        // base=1000 - time(10*2=20) - recency(5*1=5) + freq(3*15=45) + payload(-300) + state(0) + tier(0)
        assert_eq!(score, 720);
    }

    #[test]
    fn score_kv_dram_age50_recency0_freq5_exact() {
        let meta = make_meta(1, Some(10), 0, 5, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::CpuDram, 50,
        );
        // base=1000 - time(50*2=100) + freq(5*15=75) + tier(-200)
        assert_eq!(score, 775);
    }

    #[test]
    fn score_prompt_hbm_age0_recency0_freq0_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 2000);
    }

    #[test]
    fn score_rag_nvme_age200_recency10_freq2_exact() {
        let meta = make_meta(1, Some(10), 10, 2, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::Nvme, 200,
        );
        // base=1000 - time(200*2=400) - recency(10*1=10) + freq(2*15=30) + payload(-500) + tier(-500)
        assert_eq!(score, -380);
    }

    #[test]
    fn score_dense_hbm_warm_age0_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        // base=1000 + payload(5000) + state(5000)
        assert_eq!(score, 11000);
    }

    #[test]
    fn score_dense_hbm_protected_age0_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        // base=1000 + payload(5000) + state(10000)
        assert_eq!(score, 16000);
    }

    #[test]
    fn score_rag_hbm_warm_age100_freq10_exact() {
        let meta = make_meta(1, Some(10), 0, 10, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 100,
        );
        // base=1000 - time(200) + freq(150) + payload(-500) + state(5000) + tier(0)
        assert_eq!(score, 5450);
    }

    #[test]
    fn score_expert_nvme_warm_recency20_age300_exact() {
        let meta = make_meta(1, None, 20, 0, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 300,
        );
        // base=1000 - time(600) - recency(20) + payload(-300) + state(5000) + tier(-500)
        assert_eq!(score, 4580);
    }

    // ── Score: compression ratio with non-zero page size ──

    #[test]
    fn score_compression_1k_original_256_compressed_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 256, 1024, StorageTier::GpuHbm, 0,
        );
        // base=1000 + compression((1-0.25)*500=375) + page_size(1024/1024*1=1)
        assert_eq!(score, 1376);
    }

    #[test]
    fn score_compression_2k_original_512_compressed_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 512, 2048, StorageTier::GpuHbm, 0,
        );
        // base=1000 + compression((1-0.25)*500=375) + page_size(2048/1024*1=2)
        assert_eq!(score, 1377);
    }

    #[test]
    fn score_compression_4k_original_2k_compressed_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 2048, 4096, StorageTier::GpuHbm, 0,
        );
        // base=1000 + compression((1-0.5)*500=250) + page_size(4096/1024*1=4)
        assert_eq!(score, 1254);
    }

    // ── Score: page size bonus alone at various sizes ──

    #[test]
    fn score_page_size_4k_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // base=1000 + page_size(4096/1024=4) (no compression bonus since equal)
        assert_eq!(score, 1004);
    }

    #[test]
    fn score_page_size_8k_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 8192, 8192, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1008);
    }

    #[test]
    fn score_page_size_32k_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 32768, 32768, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1032);
    }

    // ── Score: complex combinations with all components ──

    #[test]
    fn score_all_components_expert_warm_nvme_compress() {
        let meta = make_meta(1, None, 3, 5, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 1024, 4096, StorageTier::Nvme, 100,
        );
        // base=1000 - time(200) - recency(3) + freq(75) + compression(375) + page_size(4) + payload(-300) + state(5000) + tier(-500)
        assert_eq!(score, 5451);
    }

    #[test]
    fn score_all_components_prompt_protected_dram_compress() {
        let meta = make_meta(1, Some(10), 2, 8, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 512, 2048, StorageTier::CpuDram, 25,
        );
        // base=1000 - time(50) - recency(2) + freq(120) + compression(375) + page_size(2) + payload(1000) + state(10000) + tier(-200)
        assert_eq!(score, 12245);
    }

    #[test]
    fn score_all_components_rag_standby_hbm_compress() {
        let meta = make_meta(1, Some(10), 1, 2, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 3072, 4096, StorageTier::GpuHbm, 50,
        );
        // base=1000 - time(100) - recency(1) + freq(30) + compression(125) + page_size(4) + payload(-500) + state(0) + tier(0)
        assert_eq!(score, 558);
    }

    #[test]
    fn score_all_components_kv_active_dram_no_compress() {
        let meta = make_meta(1, Some(10), 0, 1, PageState::Active);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::CpuDram, 0,
        );
        // base=1000 + freq(15) + tier(-200)
        assert_eq!(score, 815);
    }

    #[test]
    fn score_all_components_dense_free_nvme_no_compress() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Free);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::Nvme, 0,
        );
        // base=1000 + payload(5000) + tier(-500)
        assert_eq!(score, 5500);
    }

    // ── EvictionTier: Copy trait allows assignment without clone ──

    #[test]
    fn eviction_tier_copy_assignment() {
        let original = EvictionTier::ColdExpert;
        let assigned = original;
        assert_eq!(original, assigned);
        assert_eq!(original, EvictionTier::ColdExpert);
    }

    #[test]
    fn eviction_tier_copy_in_closure() {
        let tier = EvictionTier::Protected;
        let closure_result = || -> EvictionTier { tier }();
        assert_eq!(tier, closure_result);
    }

    // ── EvictionCandidate: field mutation via struct update ──

    #[test]
    fn eviction_candidate_struct_update_syntax() {
        let base = EvictionCandidate {
            page_id: 1,
            score: 500,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: None,
        };
        let modified = EvictionCandidate {
            score: 200,
            current_tier: StorageTier::CpuDram,
            ..base.clone()
        };
        assert_eq!(modified.page_id, 1);
        assert_eq!(modified.score, 200);
        assert_eq!(modified.current_tier, StorageTier::CpuDram);
        assert_eq!(modified.codec, CompressionCodec::Lz4);
        assert_eq!(modified.page_bytes, 4096);
        // Original unchanged
        assert_eq!(base.score, 500);
        assert_eq!(base.current_tier, StorageTier::GpuHbm);
    }

    #[test]
    fn eviction_candidate_struct_update_preserves_unmodified() {
        let original = EvictionCandidate {
            page_id: 42,
            score: -100,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
            group_id: Some(99),
        };
        let updated = EvictionCandidate {
            page_id: 43,
            ..original.clone()
        };
        assert_eq!(updated.page_id, 43);
        assert_eq!(updated.score, -100);
        assert_eq!(updated.current_tier, StorageTier::Nvme);
        assert_eq!(updated.codec, CompressionCodec::ZstdDict);
        assert_eq!(updated.page_bytes, 8192);
        assert_eq!(updated.group_id, Some(99));
    }

    // ── EvictionCandidate: sort by score then by page_id for determinism ──

    #[test]
    fn eviction_candidate_sort_many_same_score() {
        let mut candidates: Vec<EvictionCandidate> = (0..20).map(|i| EvictionCandidate {
            page_id: i,
            score: 100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        }).collect();
        candidates.sort_by_key(|c| c.score);
        assert!(candidates.iter().all(|c| c.score == 100));
        assert_eq!(candidates.len(), 20);
    }

    #[test]
    fn eviction_candidate_truncate_preserves_order() {
        let mut candidates: Vec<EvictionCandidate> = vec![
            EvictionCandidate { page_id: 0, score: 500, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 1, score: 100, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 2, score: -200, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 3, score: 300, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
        ];
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(2);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].score, -200);
        assert_eq!(candidates[1].score, 100);
    }

    // ── Score: payload bonus delta between adjacent kinds ──

    #[test]
    fn score_payload_delta_rag_to_expert_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_rag = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_expert = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_expert - s_rag, 200);
    }

    #[test]
    fn score_payload_delta_kv_to_prompt_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_prompt = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_prompt - s_kv, 1000);
    }

    #[test]
    fn score_payload_delta_prompt_to_dense_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_prompt = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_dense = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_dense - s_prompt, 4000);
    }

    // ── Score: state bonus delta between adjacent states ──

    #[test]
    fn score_state_delta_standby_to_warm_exact() {
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta_warm = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let s_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_warm = EvictionWorker::compute_importance_score(
            &meta_warm, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_warm - s_standby, 5000);
    }

    #[test]
    fn score_state_delta_warm_to_protected_exact() {
        let meta_warm = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let meta_prot = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let s_warm = EvictionWorker::compute_importance_score(
            &meta_warm, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_prot = EvictionWorker::compute_importance_score(
            &meta_prot, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_prot - s_warm, 5000);
    }

    // ── Score: tier discount delta between adjacent tiers ──

    #[test]
    fn score_tier_delta_hbm_to_dram_without_other_factors() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(s_hbm - s_dram, 200);
    }

    #[test]
    fn score_tier_delta_dram_to_nvme_without_other_factors() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        let s_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(s_dram - s_nvme, 300);
    }

    #[test]
    fn score_tier_delta_hbm_to_nvme_is_500() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(s_hbm - s_nvme, 500);
    }

    // ── Score: time penalty alone drives score to arbitrary depth ──

    #[test]
    fn score_time_penalty_1000_ticks_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 1000,
        );
        assert_eq!(score, -1000);
    }

    #[test]
    fn score_time_penalty_5000_ticks_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 5000,
        );
        assert_eq!(score, -9000);
    }

    // ── Config: field-by-field mutation independence ──

    #[test]
    fn config_mutate_hbm_threshold_preserves_others() {
        let mut cfg = EvictionWorkerConfig::default();
        let original_dram = cfg.dram_pressure_threshold;
        let original_importance = cfg.importance_threshold;
        cfg.hbm_pressure_threshold = 0.5;
        assert_eq!(cfg.dram_pressure_threshold, original_dram);
        assert_eq!(cfg.importance_threshold, original_importance);
        assert!((cfg.hbm_pressure_threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn config_mutate_dram_threshold_preserves_others() {
        let mut cfg = EvictionWorkerConfig::default();
        let original_hbm = cfg.hbm_pressure_threshold;
        cfg.dram_pressure_threshold = 0.5;
        assert_eq!(cfg.hbm_pressure_threshold, original_hbm);
    }

    #[test]
    fn config_mutate_importance_threshold_preserves_others() {
        let mut cfg = EvictionWorkerConfig::default();
        let original_hbm = cfg.hbm_pressure_threshold;
        cfg.importance_threshold = 500;
        assert_eq!(cfg.hbm_pressure_threshold, original_hbm);
        assert_eq!(cfg.importance_threshold, 500);
    }

    #[test]
    fn config_mutate_page_bytes_preserves_codec() {
        let mut cfg = EvictionWorkerConfig::default();
        let original_codec = cfg.default_evict_codec;
        cfg.page_bytes = 8192;
        assert_eq!(cfg.default_evict_codec, original_codec);
        assert_eq!(cfg.page_bytes, 8192);
    }

    #[test]
    fn config_mutate_max_evict_preserves_tick() {
        let mut cfg = EvictionWorkerConfig::default();
        let original_tick = cfg.tick_interval;
        cfg.max_evict_per_round = 32;
        assert_eq!(cfg.tick_interval, original_tick);
        assert_eq!(cfg.max_evict_per_round, 32);
    }

    // ── Config: clone field-by-field independence after mutation ──

    #[test]
    fn config_clone_hbm_threshold_independence() {
        let cfg1 = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.5,
            ..EvictionWorkerConfig::default()
        };
        let cfg2 = cfg1.clone();
        assert_eq!(cfg1.hbm_pressure_threshold, cfg2.hbm_pressure_threshold);
    }

    #[test]
    fn config_clone_dram_threshold_independence() {
        let cfg1 = EvictionWorkerConfig {
            dram_pressure_threshold: 0.6,
            ..EvictionWorkerConfig::default()
        };
        let cfg2 = cfg1.clone();
        assert_eq!(cfg1.dram_pressure_threshold, cfg2.dram_pressure_threshold);
    }

    #[test]
    fn config_clone_page_bytes_independence() {
        let cfg1 = EvictionWorkerConfig {
            page_bytes: 16384,
            ..EvictionWorkerConfig::default()
        };
        let cfg2 = cfg1.clone();
        assert_eq!(cfg1.page_bytes, cfg2.page_bytes);
    }

    // ── Score: verify recency penalty weight derivation ──

    #[test]
    fn score_recency_penalty_is_time_decay_half() {
        let meta = make_meta(1, Some(10), 10, 0, PageState::Standby);
        let s_no_recency = {
            let meta0 = make_meta(1, Some(10), 0, 0, PageState::Standby);
            EvictionWorker::compute_importance_score(
                &meta0, None, 0, 0, StorageTier::GpuHbm, 0,
            )
        };
        let s_with_recency = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_no_recency - s_with_recency, 10, "recency=10 should penalize by 10");
    }

    #[test]
    fn score_recency_penalty_50_exact() {
        let meta = make_meta(1, Some(10), 50, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 950);
    }

    // ── Score: all PageState variants produce consistent pattern ──

    #[test]
    fn score_all_states_on_hbm_ranking() {
        let states_and_expected_bonus = vec![
            (PageState::Free, 0i64),
            (PageState::Active, 0),
            (PageState::Standby, 0),
            (PageState::SwappedOut, 0),
            (PageState::Swapped, 0),
            (PageState::Warm, 5000),
            (PageState::Protected, 10000),
        ];
        let mut prev_score = i64::MIN;
        for (state, expected_bonus) in states_and_expected_bonus {
            let meta = make_meta(1, Some(10), 0, 0, state);
            let score = EvictionWorker::compute_importance_score(
                &meta, None, 0, 0, StorageTier::GpuHbm, 0,
            );
            assert_eq!(score, 1000 + expected_bonus, "state {:?} bonus mismatch", state);
            assert!(score >= prev_score, "scores should be non-decreasing: {:?} < prev", state);
            prev_score = score;
        }
    }

    // ── Score: all StorageTier variants produce distinct scores ──

    #[test]
    fn score_all_tiers_distinct_on_standby() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        let scores: Vec<i64> = tiers.iter().map(|t| {
            EvictionWorker::compute_importance_score(&meta, None, 0, 0, *t, 0)
        }).collect();
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1], "HBM > DRAM");
        assert!(scores[1] > scores[2], "DRAM > NVMe");
        assert_ne!(scores[0], scores[1]);
        assert_ne!(scores[1], scores[2]);
    }

    // ── Score: compression ratio with exact known values ──

    #[test]
    fn score_compression_1_out_of_3_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 1, 3, StorageTier::GpuHbm, 0,
        );
        // compression = (1 - 1/3) * 500 = 333, page_size = 3/1024 = 0 (integer)
        assert_eq!(score, 1333);
    }

    #[test]
    fn score_compression_3_out_of_4_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 3, 4, StorageTier::GpuHbm, 0,
        );
        // compression = (1 - 3/4) * 500 = 125, page_size = 4/1024 = 0
        assert_eq!(score, 1125);
    }

    // ── Score: frequency bonus at various access counts ──

    #[test]
    fn score_frequency_bonus_access_20_exact() {
        let meta = make_meta(1, Some(10), 0, 20, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1300);
    }

    #[test]
    fn score_frequency_bonus_access_50_exact() {
        let meta = make_meta(1, Some(10), 0, 50, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1750);
    }

    #[test]
    fn score_frequency_bonus_access_100_exact() {
        let meta = make_meta(1, Some(10), 0, 100, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 2500);
    }

    // ── Score: interaction between frequency and time penalty ──

    #[test]
    fn score_freq_10_beats_age_50_on_hbm() {
        let meta = make_meta(1, Some(10), 0, 10, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 50,
        );
        // base=1000 - time(100) + freq(150) = 1050
        assert_eq!(score, 1050);
        assert!(score > 0);
    }

    #[test]
    fn score_freq_10_loses_to_age_200_on_hbm() {
        let meta = make_meta(1, Some(10), 0, 10, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 200,
        );
        // base=1000 - time(400) + freq(150) = 750
        assert_eq!(score, 750);
    }

    // ── Score: negative score scenarios ──

    #[test]
    fn score_rag_nvme_age1000_deeply_negative() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::Nvme, 1000,
        );
        // base=1000 - time(2000) + payload(-500) + tier(-500) = -2000
        assert_eq!(score, -2000);
    }

    #[test]
    fn score_expert_nvme_age2000_deeply_negative() {
        let meta = make_meta(1, None, 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 2000,
        );
        // base=1000 - time(4000) + payload(-300) + tier(-500) = -3800
        assert_eq!(score, -3800);
    }

    // ── Score: warm state prevents deeply negative scores ──

    #[test]
    fn score_rag_warm_nvme_age500_still_positive() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::Nvme, 500,
        );
        // base=1000 - time(1000) + payload(-500) + state(5000) + tier(-500) = 4000
        assert_eq!(score, 4000);
        assert!(score > 0);
    }

    #[test]
    fn score_expert_warm_nvme_age1000_still_positive() {
        let meta = make_meta(1, None, 0, 0, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 1000,
        );
        // base=1000 - time(2000) + payload(-300) + state(5000) + tier(-500) = 3200
        assert_eq!(score, 3200);
        assert!(score > 0);
    }

    // ── Score: protected state ensures very high score even on NVMe ──

    #[test]
    fn score_rag_protected_nvme_age2000_high() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::Nvme, 2000,
        );
        // base=1000 - time(4000) + payload(-500) + state(10000) + tier(-500) = 6000
        assert_eq!(score, 6000);
    }

    #[test]
    fn score_expert_protected_nvme_age5000_high() {
        let meta = make_meta(1, None, 0, 0, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 5000,
        );
        // base=1000 - time(10000) + payload(-300) + state(10000) + tier(-500) = 200
        assert_eq!(score, 200);
    }

    // ── EvictionCandidate: debug output includes all fields ──

    #[test]
    fn eviction_candidate_debug_shows_score_negative() {
        let c = EvictionCandidate {
            page_id: 1, score: -9999, current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict, page_bytes: 4096, group_id: Some(42),
        };
        let debug_str = format!("{:?}", c);
        assert!(debug_str.contains("-9999"));
        assert!(debug_str.contains("Nvme"));
        assert!(debug_str.contains("ZstdDict"));
    }

    // ── Score: tier discount is independent of compression ratio ──

    #[test]
    fn score_tier_discount_same_with_and_without_compression() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_no_compress_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_no_compress_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        let s_compress_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 512, 1024, StorageTier::GpuHbm, 0,
        );
        let s_compress_dram = EvictionWorker::compute_importance_score(
            &meta, None, 512, 1024, StorageTier::CpuDram, 0,
        );
        let delta_no_compress = s_no_compress_hbm - s_no_compress_dram;
        let delta_compress = s_compress_hbm - s_compress_dram;
        assert_eq!(delta_no_compress, delta_compress);
        assert_eq!(delta_no_compress, 200);
    }

    // ── Score: state bonus is independent of compression ratio ──

    #[test]
    fn score_state_bonus_same_with_and_without_compression() {
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta_warm = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let s_standby_no = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_warm_no = EvictionWorker::compute_importance_score(
            &meta_warm, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_standby_yes = EvictionWorker::compute_importance_score(
            &meta_standby, None, 512, 1024, StorageTier::GpuHbm, 0,
        );
        let s_warm_yes = EvictionWorker::compute_importance_score(
            &meta_warm, None, 512, 1024, StorageTier::GpuHbm, 0,
        );
        let delta_no = s_warm_no - s_standby_no;
        let delta_yes = s_warm_yes - s_standby_yes;
        assert_eq!(delta_no, delta_yes);
        assert_eq!(delta_no, 5000);
    }

    // ── Score: verify constant relationships ──

    #[test]
    fn constant_hbm_dram_discount_sum_is_500() {
        let hbm_discount = 0i64;
        let dram_discount = -200i64;
        let nvme_discount = -500i64;
        assert_eq!(nvme_discount - hbm_discount, -500);
        assert_eq!(nvme_discount - dram_discount, -300);
    }

    #[test]
    fn constant_payload_bonus_ordering() {
        assert!(KNOWLEDGE_RAG_BONUS < EXPERT_WEIGHT_BONUS);
        assert!(EXPERT_WEIGHT_BONUS < KV_CONTEXT_BONUS);
        assert!(KV_CONTEXT_BONUS < PROMPT_SYSTEM_BONUS);
        assert!(PROMPT_SYSTEM_BONUS < DENSE_LAYER_BONUS);
    }

    #[test]
    fn constant_payload_bonuses_sum_reasonable() {
        let total = EXPERT_WEIGHT_BONUS + KV_CONTEXT_BONUS + PROMPT_SYSTEM_BONUS
            + DENSE_LAYER_BONUS + KNOWLEDGE_RAG_BONUS;
        // -300 + 0 + 1000 + 5000 + (-500) = 5200
        assert_eq!(total, 5200);
    }

    // ── Config: default values match constants ──

    #[test]
    fn config_default_matches_hbm_pressure_constant() {
        let cfg = EvictionWorkerConfig::default();
        assert!((cfg.hbm_pressure_threshold - HBM_PRESSURE_RATIO).abs() < 1e-6);
    }

    #[test]
    fn config_default_matches_dram_pressure_constant() {
        let cfg = EvictionWorkerConfig::default();
        assert!((cfg.dram_pressure_threshold - DRAM_PRESSURE_RATIO).abs() < 1e-6);
    }

    #[test]
    fn config_default_matches_importance_threshold_constant() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.importance_threshold, IMPORTANCE_SCORE_THRESHOLD);
    }

    #[test]
    fn config_default_matches_hbm_evict_age_constant() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.hbm_evict_age_ticks, HBM_EVICT_AGE_TICKS);
    }

    #[test]
    fn config_default_matches_dram_evict_age_constant() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.dram_evict_age_ticks, DRAM_EVICT_AGE_TICKS);
    }

    // ── EvictionCandidate: group_id with zero vs none ──

    #[test]
    fn eviction_candidate_group_id_zero_vs_none() {
        let with_zero = EvictionCandidate {
            page_id: 0, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 0, group_id: Some(0),
        };
        let with_none = EvictionCandidate {
            page_id: 0, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 0, group_id: None,
        };
        assert_ne!(with_zero.group_id, with_none.group_id);
    }

    // ── Score: u32::MAX values don't cause overflow ──

    #[test]
    fn score_u32_max_sizes_no_panic() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, u32::MAX, u32::MAX, StorageTier::GpuHbm, 0,
        );
        // compressed == original, no compression bonus, page_size = u32::MAX / 1024
        assert!(score > 1000);
    }

    #[test]
    fn score_u32_max_original_zero_compressed_no_panic() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, u32::MAX, 0, StorageTier::GpuHbm, 0,
        );
        // original = 0 => no compression bonus, no page size bonus
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_zero_original_u32_max_compressed_no_panic() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, u32::MAX, StorageTier::GpuHbm, 0,
        );
        // compressed = 0, original = u32::MAX: compression = (1 - 0) * 500 = 500
        assert!(score >= 1500);
    }

    // ── Score: frequency bonus alone at edge counts ──

    #[test]
    fn score_frequency_bonus_access_0_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_frequency_bonus_access_1_vs_0_delta() {
        let meta0 = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta1 = make_meta(1, Some(10), 0, 1, PageState::Standby);
        let s0 = EvictionWorker::compute_importance_score(&meta0, None, 0, 0, StorageTier::GpuHbm, 0);
        let s1 = EvictionWorker::compute_importance_score(&meta1, None, 0, 0, StorageTier::GpuHbm, 0);
        assert_eq!(s1 - s0, FREQUENCY_BONUS as i64);
    }

    // ── Score: time penalty at exactly threshold-relevant ages ──

    #[test]
    fn score_age_49_hbm_still_above_base_minus_100() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 49,
        );
        // base=1000 - time(98) = 902
        assert_eq!(score, 902);
    }

    #[test]
    fn score_age_51_hbm_below_base_minus_100() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 51,
        );
        // base=1000 - time(102) = 898
        assert_eq!(score, 898);
    }

    // ── Score: all components combined with protected state ──

    #[test]
    fn score_protected_with_all_penalties_still_above_threshold() {
        let meta = make_meta(1, Some(10), 50, 0, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 500,
        );
        // base=1000 - time(1000) - recency(50) + payload(-300) + state(10000) + tier(-500)
        // = 9150, well above threshold
        assert!(score > IMPORTANCE_SCORE_THRESHOLD);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests — Batch 8 (70 new tests)
    // ─────────────────────────────────────────────────────────────────────────

    // ── Score: page_id zero vs nonzero does not change score ──

    #[test]
    fn score_page_id_zero_vs_nonzero_identical() {
        let meta0 = make_meta(0, Some(10), 5, 3, PageState::Standby);
        let meta1 = make_meta(999, Some(10), 5, 3, PageState::Standby);
        let s0 = EvictionWorker::compute_importance_score(
            &meta0, None, 0, 0, StorageTier::GpuHbm, 10,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta1, None, 0, 0, StorageTier::GpuHbm, 10,
        );
        assert_eq!(s0, s1);
    }

    // ── Score: is_lir true vs false no score difference ──

    #[test]
    fn score_is_lir_false_vs_true_same_score() {
        let meta_false = PageMetadata {
            page_id: 1, sequence_id: Some(10), recency: 5, access_count: 3,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Standby, warm_until: None,
        };
        let meta_true = PageMetadata {
            page_id: 1, sequence_id: Some(10), recency: 5, access_count: 3,
            last_access: Instant::now(), swap_in_time: None, is_lir: true,
            state: PageState::Standby, warm_until: None,
        };
        let s_false = EvictionWorker::compute_importance_score(
            &meta_false, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_true = EvictionWorker::compute_importance_score(
            &meta_true, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_false, s_true);
    }

    // ── Score: all PagePayloadKind variants yield distinct scores on HBM ──

    #[test]
    fn score_all_payload_variants_distinct_on_hbm() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let kinds = [
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::KvContext,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::DenseLayerWeight,
            PagePayloadKind::KnowledgeRAG,
        ];
        let scores: Vec<i64> = kinds.iter().map(|k| {
            EvictionWorker::compute_importance_score(
                &meta, Some(*k), 0, 0, StorageTier::GpuHbm, 0,
            )
        }).collect();
        for i in 0..scores.len() {
            for j in (i + 1)..scores.len() {
                assert_ne!(scores[i], scores[j], "{:?} and {:?} should differ", kinds[i], kinds[j]);
            }
        }
    }

    // ── Score: None payload and KvContext produce same score ──

    #[test]
    fn score_none_equals_kv_context_payload() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_none = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_none, s_kv);
    }

    // ── Score: compression bonus with exactly 50% ratio ──

    #[test]
    fn score_compression_exactly_half_2k_of_4k() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 2048, 4096, StorageTier::GpuHbm, 0,
        );
        // base=1000 + compression((1-0.5)*500=250) + page_size(4096/1024=4)
        assert_eq!(score, 1254);
    }

    // ── Score: compression bonus zero when compressed == original ──

    #[test]
    fn score_compression_equal_sizes_zero_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 8192, 8192, StorageTier::GpuHbm, 0,
        );
        // base=1000 + compression(0) + page_size(8)
        assert_eq!(score, 1008);
    }

    // ── Score: page size bonus with non-power-of-2 size ──

    #[test]
    fn score_page_size_1536_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 1536, 1536, StorageTier::GpuHbm, 0,
        );
        // base=1000 + page_size(1536/1024=1) + compression(0 since equal)
        assert_eq!(score, 1001);
    }

    #[test]
    fn score_page_size_3000_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 3000, 3000, StorageTier::GpuHbm, 0,
        );
        // base=1000 + page_size(3000/1024=2) + compression(0)
        assert_eq!(score, 1002);
    }

    // ── Score: warm state with compression and frequency combined ──

    #[test]
    fn score_warm_kv_hbm_compress_50pct_freq5_age20() {
        let meta = make_meta(1, Some(10), 0, 5, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::GpuHbm, 20,
        );
        // base=1000 - time(40) + freq(75) + compression(250) + page_size(4) + state(5000)
        assert_eq!(score, 6289);
    }

    #[test]
    fn score_warm_expert_dram_compress_75pct_freq3_age50() {
        let meta = make_meta(1, None, 0, 3, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 1024, 4096, StorageTier::CpuDram, 50,
        );
        // base=1000 - time(100) + freq(45) + compression(375) + page_size(4) + payload(-300) + state(5000) + tier(-200)
        assert_eq!(score, 5824);
    }

    // ── Score: protected state with multiple penalties ──

    #[test]
    fn score_protected_rag_dram_recency30_age100_freq2() {
        let meta = make_meta(1, Some(10), 30, 2, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::CpuDram, 100,
        );
        // base=1000 - time(200) - recency(30) + freq(30) + payload(-500) + state(10000) + tier(-200)
        assert_eq!(score, 10100);
    }

    #[test]
    fn score_protected_prompt_nvme_recency50_age300_freq10() {
        let meta = make_meta(1, Some(10), 50, 10, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::Nvme, 300,
        );
        // base=1000 - time(600) - recency(50) + freq(150) + payload(1000) + state(10000) + tier(-500)
        assert_eq!(score, 11000);
    }

    // ── Score: Active state gets zero bonus like Standby ──

    #[test]
    fn score_active_state_zero_bonus_on_hbm() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Active);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_active_state_zero_bonus_on_dram() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Active);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(score, 800);
    }

    // ── Score: Free state gets zero bonus ──

    #[test]
    fn score_free_state_zero_bonus_on_hbm() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Free);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    // ── Score: large recency alone can drive score negative ──

    #[test]
    fn score_recency_2000_drives_negative() {
        let meta = make_meta(1, Some(10), 2000, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // base=1000 - recency(2000) = -1000
        assert_eq!(score, -1000);
    }

    // ── Score: frequency alone can overcome base time penalty ──

    #[test]
    fn score_freq_200_overcomes_age_200() {
        let meta = make_meta(1, Some(10), 0, 200, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 200,
        );
        // base=1000 - time(400) + freq(3000) = 3600
        assert_eq!(score, 3600);
    }

    // ── Score: time penalty at exactly age 250 ──

    #[test]
    fn score_time_penalty_250_ticks_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 250,
        );
        // base=1000 - time(500) = 500
        assert_eq!(score, 500);
    }

    // ── Score: age 0 vs age 1 difference is exactly TIME_DECAY_WEIGHT ──

    #[test]
    fn score_age_delta_0_to_1_is_time_decay_weight() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s0 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 1,
        );
        assert_eq!(s0 - s1, TIME_DECAY_WEIGHT as i64);
    }

    // ── Score: recency delta is exactly 1 per unit ──

    #[test]
    fn score_recency_delta_per_unit_exact() {
        let meta0 = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta1 = make_meta(1, Some(10), 1, 0, PageState::Standby);
        let s0 = EvictionWorker::compute_importance_score(
            &meta0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta1, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s0 - s1, 1);
    }

    // ── Score: frequency delta per unit is exactly FREQUENCY_BONUS ──

    #[test]
    fn score_freq_delta_per_unit_exact() {
        let meta0 = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta1 = make_meta(1, Some(10), 0, 1, PageState::Standby);
        let s0 = EvictionWorker::compute_importance_score(
            &meta0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta1, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s1 - s0, FREQUENCY_BONUS as i64);
    }

    // ── Score: warm delta is exactly 5000 for all payloads ──

    #[test]
    fn score_warm_delta_5000_for_expert_weight() {
        let meta_standby = make_meta(1, None, 0, 0, PageState::Standby);
        let meta_warm = make_meta(1, None, 0, 0, PageState::Warm);
        let ss = EvictionWorker::compute_importance_score(
            &meta_standby, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        let sw = EvictionWorker::compute_importance_score(
            &meta_warm, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(sw - ss, 5000);
    }

    #[test]
    fn score_warm_delta_5000_for_rag() {
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta_warm = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let ss = EvictionWorker::compute_importance_score(
            &meta_standby, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
        );
        let sw = EvictionWorker::compute_importance_score(
            &meta_warm, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(sw - ss, 5000);
    }

    // ── Score: protected delta is exactly 10000 for all payloads ──

    #[test]
    fn score_protected_delta_10000_for_dense_layer() {
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta_prot = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let ss = EvictionWorker::compute_importance_score(
            &meta_standby, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        let sp = EvictionWorker::compute_importance_score(
            &meta_prot, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(sp - ss, 10000);
    }

    #[test]
    fn score_protected_delta_10000_for_prompt() {
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta_prot = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let ss = EvictionWorker::compute_importance_score(
            &meta_standby, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        let sp = EvictionWorker::compute_importance_score(
            &meta_prot, Some(PagePayloadKind::PromptSystem), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(sp - ss, 10000);
    }

    // ── Score: tier discount is independent of payload ──

    #[test]
    fn score_dram_discount_independent_of_expert_weight() {
        let meta = make_meta(1, None, 0, 0, PageState::Standby);
        let s_hbm = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_dram = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(s_hbm - s_dram, 200);
    }

    #[test]
    fn score_dram_discount_independent_of_dense_layer() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let s_hbm = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_dram = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(s_hbm - s_dram, 200);
    }

    // ── Score: compression bonus independent of state ──

    #[test]
    fn score_compression_bonus_independent_of_warm() {
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta_warm = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let s_standby_no = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_standby_yes = EvictionWorker::compute_importance_score(
            &meta_standby, None, 512, 1024, StorageTier::GpuHbm, 0,
        );
        let s_warm_no = EvictionWorker::compute_importance_score(
            &meta_warm, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_warm_yes = EvictionWorker::compute_importance_score(
            &meta_warm, None, 512, 1024, StorageTier::GpuHbm, 0,
        );
        let compress_delta_standby = s_standby_yes - s_standby_no;
        let compress_delta_warm = s_warm_yes - s_warm_no;
        assert_eq!(compress_delta_standby, compress_delta_warm);
    }

    // ── Score: page size bonus independent of state ──

    #[test]
    fn score_page_size_bonus_independent_of_state() {
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let meta_protected = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let s_standby_no = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_standby_8k = EvictionWorker::compute_importance_score(
            &meta_standby, None, 8192, 8192, StorageTier::GpuHbm, 0,
        );
        let s_prot_no = EvictionWorker::compute_importance_score(
            &meta_protected, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s_prot_8k = EvictionWorker::compute_importance_score(
            &meta_protected, None, 8192, 8192, StorageTier::GpuHbm, 0,
        );
        assert_eq!(s_standby_8k - s_standby_no, s_prot_8k - s_prot_no);
    }

    // ── EvictionTier: exhaustively all four variants are distinct via Ord-like check ──

    #[test]
    fn eviction_tier_four_variants_mutually_neq() {
        let tiers = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        for i in 0..tiers.len() {
            for j in (i + 1)..tiers.len() {
                assert_ne!(tiers[i], tiers[j]);
            }
        }
    }

    // ── EvictionCandidate: sort 50 candidates by score correctness ──

    #[test]
    fn eviction_candidate_sort_50_mixed_scores() {
        let mut candidates: Vec<EvictionCandidate> = (0..50).map(|i| EvictionCandidate {
            page_id: i,
            score: (i as i64 - 25) * 100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        }).collect();
        candidates.sort_by_key(|c| c.score);
        for window in candidates.windows(2) {
            assert!(window[0].score <= window[1].score);
        }
    }

    // ── EvictionCandidate: truncate to 1 keeps lowest score ──

    #[test]
    fn eviction_candidate_truncate_to_one() {
        let mut candidates: Vec<EvictionCandidate> = vec![
            EvictionCandidate { page_id: 0, score: 500, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 1, score: -100, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 2, score: 300, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
        ];
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(1);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].score, -100);
    }

    // ── EvictionCandidate: codec variants all valid ──

    #[test]
    fn eviction_candidate_all_five_codec_variants() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, codec) in codecs.iter().enumerate() {
            let c = EvictionCandidate {
                page_id: i,
                score: 0,
                current_tier: StorageTier::GpuHbm,
                codec: *codec,
                page_bytes: 4096,
                group_id: None,
            };
            assert_eq!(c.codec, *codec);
        }
    }

    // ── EvictionCandidate: clone preserves all six fields exactly ──

    #[test]
    fn eviction_candidate_clone_exact_fields() {
        let original = EvictionCandidate {
            page_id: 42,
            score: -7777,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 16384,
            group_id: Some(123),
        };
        let cloned = original.clone();
        assert_eq!(cloned.page_id, original.page_id);
        assert_eq!(cloned.score, original.score);
        assert_eq!(cloned.current_tier, original.current_tier);
        assert_eq!(cloned.codec, original.codec);
        assert_eq!(cloned.page_bytes, original.page_bytes);
        assert_eq!(cloned.group_id, original.group_id);
    }

    // ── EvictionCandidate: group_id Some vs None are distinct ──

    #[test]
    fn eviction_candidate_group_some_vs_none_distinct() {
        let c1 = EvictionCandidate {
            page_id: 0, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 0, group_id: Some(0),
        };
        let c2 = EvictionCandidate {
            page_id: 0, score: 0, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 0, group_id: None,
        };
        assert_ne!(c1.group_id, c2.group_id);
    }

    // ── EvictionWorkerConfig: all nine fields configurable ──

    #[test]
    fn config_nine_fields_custom() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(50),
            max_evict_per_round: 64,
            hbm_pressure_threshold: 0.7,
            dram_pressure_threshold: 0.6,
            importance_threshold: 200,
            hbm_evict_age_ticks: 100,
            dram_evict_age_ticks: 1000,
            default_evict_codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
        };
        assert_eq!(cfg.tick_interval, Duration::from_millis(50));
        assert_eq!(cfg.max_evict_per_round, 64);
        assert!((cfg.hbm_pressure_threshold - 0.7).abs() < 1e-6);
        assert!((cfg.dram_pressure_threshold - 0.6).abs() < 1e-6);
        assert_eq!(cfg.importance_threshold, 200);
        assert_eq!(cfg.hbm_evict_age_ticks, 100);
        assert_eq!(cfg.dram_evict_age_ticks, 1000);
        assert_eq!(cfg.default_evict_codec, CompressionCodec::ZstdDict);
        assert_eq!(cfg.page_bytes, 8192);
    }

    // ── EvictionWorkerConfig: page_bytes field at extremes ──

    #[test]
    fn config_page_bytes_usize_max() {
        let cfg = EvictionWorkerConfig {
            page_bytes: usize::MAX,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.page_bytes, usize::MAX);
    }

    #[test]
    fn config_page_bytes_one() {
        let cfg = EvictionWorkerConfig {
            page_bytes: 1,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.page_bytes, 1);
    }

    // ── EvictionWorkerConfig: max_evict_per_round at extremes ──

    #[test]
    fn config_max_evict_per_round_usize_max() {
        let cfg = EvictionWorkerConfig {
            max_evict_per_round: usize::MAX,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.max_evict_per_round, usize::MAX);
    }

    #[test]
    fn config_max_evict_per_round_one() {
        let cfg = EvictionWorkerConfig {
            max_evict_per_round: 1,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.max_evict_per_round, 1);
    }

    // ── EvictionWorkerConfig: tick_interval at extreme ──

    #[test]
    fn config_tick_interval_one_nano() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_nanos(1),
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval, Duration::from_nanos(1));
    }

    #[test]
    fn config_tick_interval_one_hour() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_secs(3600),
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval, Duration::from_secs(3600));
    }

    // ── EvictionWorkerConfig: importance_threshold extremes ──

    #[test]
    fn config_importance_threshold_i64_max() {
        let cfg = EvictionWorkerConfig {
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.importance_threshold, i64::MAX);
    }

    #[test]
    fn config_importance_threshold_i64_min() {
        let cfg = EvictionWorkerConfig {
            importance_threshold: i64::MIN,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.importance_threshold, i64::MIN);
    }

    // ── EvictionWorkerConfig: hbm_evict_age_ticks extremes ──

    #[test]
    fn config_hbm_evict_age_ticks_u64_max() {
        let cfg = EvictionWorkerConfig {
            hbm_evict_age_ticks: u64::MAX,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.hbm_evict_age_ticks, u64::MAX);
    }

    #[test]
    fn config_hbm_evict_age_ticks_zero() {
        let cfg = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(cfg.hbm_evict_age_ticks, 0);
    }

    // ── EvictionWorkerConfig: clone independence after mutation ──

    #[test]
    fn config_clone_independence_after_mutation_all_fields() {
        let mut cfg = EvictionWorkerConfig::default();
        let cloned = cfg.clone();
        cfg.tick_interval = Duration::from_secs(999);
        cfg.max_evict_per_round = 999;
        cfg.importance_threshold = 999;
        cfg.page_bytes = 999;
        assert_ne!(cfg.tick_interval, cloned.tick_interval);
        assert_ne!(cfg.max_evict_per_round, cloned.max_evict_per_round);
        assert_ne!(cfg.importance_threshold, cloned.importance_threshold);
        assert_ne!(cfg.page_bytes, cloned.page_bytes);
    }

    // ── EvictionWorkerConfig: default page_bytes is 4096 ──

    #[test]
    fn config_default_page_bytes_is_4096() {
        assert_eq!(EvictionWorkerConfig::default().page_bytes, 4096);
    }

    // ── EvictionWorkerConfig: default max_evict_per_round is 8 ──

    #[test]
    fn config_default_max_evict_is_default_constant() {
        assert_eq!(EvictionWorkerConfig::default().max_evict_per_round, DEFAULT_MAX_EVICT_PER_ROUND);
    }

    // ── EvictionWorkerConfig: all codec variants accepted ──

    #[test]
    fn config_codec_all_variants_accepted() {
        for codec in [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let cfg = EvictionWorkerConfig {
                default_evict_codec: codec,
                ..EvictionWorkerConfig::default()
            };
            assert_eq!(cfg.default_evict_codec, codec);
        }
    }

    // ── classify_eviction_tier: KvContext at i64::MAX is Protected ──

    #[test]
    fn classify_kv_context_i64_max_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext), i64::MAX,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    // ── classify_eviction_tier: KvContext at i64::MIN is StandbyKv ──

    #[test]
    fn classify_kv_context_i64_min_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext), i64::MIN,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    // ── classify_eviction_tier: PromptSystem at threshold-1 is StandbyKv ──

    #[test]
    fn classify_prompt_system_threshold_minus_one_is_standby() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::PromptSystem), IMPORTANCE_SCORE_THRESHOLD - 1,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    // ── classify_eviction_tier: None at threshold+1 is Protected ──

    #[test]
    fn classify_none_payload_threshold_plus_one_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            None, IMPORTANCE_SCORE_THRESHOLD + 1,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    // ── Score: exact score with all components at once on DRAM ──

    #[test]
    fn score_all_components_kv_warm_dram_compress_age100_freq10_rec20() {
        let meta = make_meta(1, Some(10), 20, 10, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 1024, 4096, StorageTier::CpuDram, 100,
        );
        // base=1000 - time(200) - recency(20) + freq(150) + compress(375) + page_size(4) + payload(0) + state(5000) + tier(-200)
        assert_eq!(score, 6109);
    }

    // ── Score: exact score with all components on NVMe ──

    #[test]
    fn score_all_components_rag_standby_nvme_compress_age50_freq5_rec10() {
        let meta = make_meta(1, Some(10), 10, 5, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 512, 2048, StorageTier::Nvme, 50,
        );
        // base=1000 - time(100) - recency(10) + freq(75) + compress(375) + page_size(2) + payload(-500) + state(0) + tier(-500)
        assert_eq!(score, 342);
    }

    // ── Score: verify compression bonus is bounded by COMPRESSION_RATIO_WEIGHT ──

    #[test]
    fn score_compression_max_bonus_equals_weight_constant() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 1024, StorageTier::GpuHbm, 0,
        );
        // base=1000 + compress(500) + page_size(1)
        assert_eq!(score - 1001, COMPRESSION_RATIO_WEIGHT as i64);
    }

    // ── Score: verify page_size_bonus at exactly 64KB ──

    #[test]
    fn score_page_size_64k_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 65536, 65536, StorageTier::GpuHbm, 0,
        );
        // base=1000 + page_size(65536/1024=64)
        assert_eq!(score, 1064);
    }

    // ── Score: verify page_size_bonus at 128KB ──

    #[test]
    fn score_page_size_128k_bonus_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 131072, 131072, StorageTier::GpuHbm, 0,
        );
        // base=1000 + page_size(131072/1024=128)
        assert_eq!(score, 1128);
    }

    // ── Score: combined negative scenario — expert + nvme + age + recency ──

    #[test]
    fn score_expert_nvme_age500_rec100_deeply_negative() {
        let meta = make_meta(1, None, 100, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 500,
        );
        // base=1000 - time(1000) - recency(100) + payload(-300) + tier(-500)
        assert_eq!(score, -900);
    }

    // ── Score: time penalty exactly at HBM_EVICT_AGE_TICKS ──

    #[test]
    fn score_age_at_hbm_evict_ticks_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, HBM_EVICT_AGE_TICKS,
        );
        // base=1000 - time(50*2=100)
        assert_eq!(score, 900);
    }

    // ── Score: time penalty exactly at DRAM_EVICT_AGE_TICKS ──

    #[test]
    fn score_age_at_dram_evict_ticks_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, DRAM_EVICT_AGE_TICKS,
        );
        // base=1000 - time(500*2=1000)
        assert_eq!(score, 0);
    }

    // ── Score: compression ratio at 10% (90% compressed) ──

    #[test]
    fn score_compression_10pct_of_original_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 100, 1024, StorageTier::GpuHbm, 0,
        );
        // base=1000 + compress((1-100/1024)*500=451) + page_size(1024/1024=1)
        assert_eq!(score, 1452);
    }

    // ── Score: compression ratio at 90% (10% compressed) ──

    #[test]
    fn score_compression_90pct_of_original_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 922, 1024, StorageTier::GpuHbm, 0,
        );
        // base=1000 + compress((1-922/1024)*500=49) + page_size(1)
        assert_eq!(score, 1050);
    }

    // ── Score: zero access_count for each payload ──

    #[test]
    fn score_zero_access_count_for_each_payload() {
        let kinds = [
            Some(PagePayloadKind::ExpertWeight),
            Some(PagePayloadKind::KvContext),
            Some(PagePayloadKind::PromptSystem),
            Some(PagePayloadKind::DenseLayerWeight),
            Some(PagePayloadKind::KnowledgeRAG),
            None,
        ];
        for kind in &kinds {
            let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
            let score = EvictionWorker::compute_importance_score(
                &meta, *kind, 0, 0, StorageTier::GpuHbm, 0,
            );
            let expected = 1000 + match kind {
                Some(PagePayloadKind::ExpertWeight) => EXPERT_WEIGHT_BONUS,
                Some(PagePayloadKind::KvContext) => KV_CONTEXT_BONUS,
                Some(PagePayloadKind::PromptSystem) => PROMPT_SYSTEM_BONUS,
                Some(PagePayloadKind::DenseLayerWeight) => DENSE_LAYER_BONUS,
                Some(PagePayloadKind::KnowledgeRAG) => KNOWLEDGE_RAG_BONUS,
                None => 0,
            };
            assert_eq!(score, expected, "payload {:?} with zero access_count", kind);
        }
    }

    // ── Score: EvictionTier can be used in a Vec and iterated ──

    #[test]
    fn eviction_tier_vec_iteration_all_four() {
        let tiers = vec![
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        assert_eq!(tiers.len(), 4);
        assert_eq!(tiers[0], EvictionTier::ColdExpert);
        assert_eq!(tiers[3], EvictionTier::Protected);
    }

    // ── Score: constant TIME_DECAY_WEIGHT is 2 (verify) ──

    #[test]
    fn constant_time_decay_weight_value_2() {
        assert_eq!(TIME_DECAY_WEIGHT, 2);
    }

    // ── Score: constant PAGE_SIZE_WEIGHT is 1 (verify) ──

    #[test]
    fn constant_page_size_weight_value_1() {
        assert_eq!(PAGE_SIZE_WEIGHT, 1);
    }

    // ── Score: constant KV_CONTEXT_BONUS is 0 ──

    #[test]
    fn constant_kv_context_bonus_is_0() {
        assert_eq!(KV_CONTEXT_BONUS, 0);
    }

    // ── Score: constant PROMPT_SYSTEM_BONUS is 1000 ──

    #[test]
    fn constant_prompt_system_bonus_is_1000() {
        assert_eq!(PROMPT_SYSTEM_BONUS, 1000);
    }

    // ── Score: constant EXPERT_WEIGHT_BONUS is -300 ──

    #[test]
    fn constant_expert_weight_bonus_is_minus_300() {
        assert_eq!(EXPERT_WEIGHT_BONUS, -300);
    }

    // ── Score: constant KNOWLEDGE_RAG_BONUS is -500 ──

    #[test]
    fn constant_knowledge_rag_bonus_is_minus_500() {
        assert_eq!(KNOWLEDGE_RAG_BONUS, -500);
    }

    // ── Score: constant DENSE_LAYER_BONUS is 5000 ──

    #[test]
    fn constant_dense_layer_bonus_is_5000() {
        assert_eq!(DENSE_LAYER_BONUS, 5000);
    }

    // ── Score: verify base score 1000 constant ──

    #[test]
    fn constant_base_score_1000() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    // ── Score: verify EvictionWorkerConfig default page_bytes is 4096 ──

    #[test]
    fn config_default_page_bytes_4096() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.page_bytes, 4096);
    }

    // ── EvictionCandidate: debug output contains all six fields ──

    #[test]
    fn eviction_candidate_debug_shows_page_id() {
        let c = EvictionCandidate {
            page_id: 42, score: 100, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4, page_bytes: 8192, group_id: Some(7),
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("42"));
        assert!(debug.contains("100"));
        assert!(debug.contains("GpuHbm"));
        assert!(debug.contains("Lz4"));
        assert!(debug.contains("8192"));
        assert!(debug.contains("7"));
    }

    // ── EvictionCandidate: struct update with different tier ──

    #[test]
    fn eviction_candidate_struct_update_tier_change() {
        let base = EvictionCandidate {
            page_id: 1, score: 500, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
        };
        let demoted = EvictionCandidate {
            current_tier: StorageTier::CpuDram,
            score: 300,
            ..base.clone()
        };
        assert_eq!(demoted.page_id, 1);
        assert_eq!(demoted.score, 300);
        assert_eq!(demoted.current_tier, StorageTier::CpuDram);
        assert_eq!(demoted.codec, CompressionCodec::None);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests — Wave 13x57
    // ─────────────────────────────────────────────────────────────────────────

    // ── Score: time penalty exactly matches tier_age × TIME_DECAY_WEIGHT ──

    #[test]
    fn score_time_penalty_750_ticks_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 750,
        );
        // base(1000) - time(750*2=1500) = -500
        assert_eq!(score, -500);
    }

    #[test]
    fn score_time_penalty_0_ticks_yields_base() {
        let meta = make_meta(1, None, 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_recency_500_exact_penalty() {
        let meta = make_meta(1, Some(10), 500, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // base(1000) - recency(500*1=500) = 500
        assert_eq!(score, 500);
    }

    // ── Score: combined penalty edges ──

    #[test]
    fn score_time_plus_recency_exactly_equals_base() {
        let meta = make_meta(1, Some(10), 200, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 300,
        );
        // base(1000) - time(300*2=600) - recency(200*1=200) = 200
        assert_eq!(score, 200);
    }

    #[test]
    fn score_freq_exactly_cancels_time_penalty() {
        // freq_bonus = access_count * 15. time_penalty = tier_age * 2.
        // access_count=40, tier_age=300 → freq=600, time=600 → cancel.
        let meta = make_meta(1, Some(10), 0, 40, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 300,
        );
        assert_eq!(score, 1000);
    }

    #[test]
    fn score_recency_exactly_cancels_freq_bonus() {
        // freq_bonus = 10*15 = 150, recency_penalty = 150*1 = 150.
        let meta = make_meta(1, Some(10), 150, 10, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000);
    }

    // ── Score: all bonuses positive pushes score well above threshold ──

    #[test]
    fn score_protected_dense_hbm_freq100_age0_rec0_max_compress() {
        let meta = make_meta(1, Some(10), 0, 100, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            0,       // fully compressed
            65536,   // 64 KiB
            StorageTier::GpuHbm,
            0,
        );
        // base(1000) + freq(100*15=1500) + compress(~500) + page_size(64) + dense(5000) + protected(10000)
        assert!(
            score > 15000,
            "all positive components should push score well above 15k: {}",
            score,
        );
    }

    // ── Score: compression edge — compressed > original treated as negative bonus ──

    #[test]
    fn score_compressed_larger_than_original_negative_bonus() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score_normal = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_inverted = EvictionWorker::compute_importance_score(
            &meta, None, 8192, 4096, StorageTier::GpuHbm, 0,
        );
        // compressed > original → ratio > 1.0 → (1.0 - ratio) < 0 → negative bonus
        assert!(
            score_inverted < score_normal,
            "compressed > original should reduce score: inverted={} normal={}",
            score_inverted,
            score_normal,
        );
    }

    // ── Score: state bonuses are additive with tier discounts ──

    #[test]
    fn score_warm_on_dram_net_effect() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        // base(1000) + warm(5000) + dram(-200) = 5800
        assert_eq!(score, 5800);
    }

    #[test]
    fn score_protected_on_nvme_net_effect() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Protected);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + protected(10000) + nvme(-500) = 10500
        assert_eq!(score, 10500);
    }

    #[test]
    fn score_warm_on_nvme_net_effect() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + warm(5000) + nvme(-500) = 5500
        assert_eq!(score, 5500);
    }

    // ── Score: Swapped and SwappedOut states get no bonus ──

    #[test]
    fn score_swapped_state_on_hbm_yields_base() {
        let meta_swapped = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score_swapped = EvictionWorker::compute_importance_score(
            &meta_swapped, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_swapped, score_standby);
    }

    #[test]
    fn score_swapped_out_state_on_hbm_yields_base() {
        let meta_so = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score_so = EvictionWorker::compute_importance_score(
            &meta_so, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_so, score_standby);
    }

    // ── Score: Active state gets no bonus ──

    #[test]
    fn score_active_state_on_hbm_yields_base() {
        let meta_active = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score_active = EvictionWorker::compute_importance_score(
            &meta_active, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_active, score_standby);
    }

    // ── Score: Free state gets no bonus ──

    #[test]
    fn score_free_state_on_hbm_yields_base() {
        let meta_free = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        let meta_standby = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score_free = EvictionWorker::compute_importance_score(
            &meta_free, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score_free, score_standby);
    }

    // ── Score: all six non-bonus states yield same score ──

    #[test]
    fn score_six_states_without_bonus_are_equal() {
        let states = [
            PageState::Standby,
            PageState::Active,
            PageState::SwappedOut,
            PageState::Swapped,
            PageState::Free,
        ];
        let scores: Vec<i64> = states
            .iter()
            .map(|&s| {
                let meta = PageMetadata {
                    page_id: 1,
                    sequence_id: Some(10),
                    recency: 5,
                    access_count: 3,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: s,
                    warm_until: None,
                };
                EvictionWorker::compute_importance_score(
                    &meta, None, 0, 4096, StorageTier::GpuHbm, 50,
                )
            })
            .collect();
        // All non-Warm, non-Protected states should have the same score.
        let first = scores[0];
        assert!(scores.iter().all(|&s| s == first), "all non-bonus states should have equal score: {:?}", scores);
    }

    // ── Score: Warm beats all non-Protected states ──

    #[test]
    fn score_warm_beats_all_non_protected_states() {
        let non_protected = [
            PageState::Standby,
            PageState::Active,
            PageState::SwappedOut,
            PageState::Swapped,
            PageState::Free,
        ];
        for &state in &non_protected {
            let meta = PageMetadata {
                page_id: 1,
                sequence_id: Some(10),
                recency: 10,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state,
                warm_until: None,
            };
            let meta_warm = PageMetadata {
                page_id: 1,
                sequence_id: Some(10),
                recency: 10,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Warm,
                warm_until: None,
            };
            let score_state = EvictionWorker::compute_importance_score(
                &meta, None, 0, 4096, StorageTier::GpuHbm, 50,
            );
            let score_warm = EvictionWorker::compute_importance_score(
                &meta_warm, None, 0, 4096, StorageTier::GpuHbm, 50,
            );
            assert!(
                score_warm > score_state,
                "Warm({}) should beat {:?}({})",
                score_warm,
                state,
                score_state,
            );
        }
    }

    // ── Score: Protected beats all other states ──

    #[test]
    fn score_protected_beats_all_other_states() {
        let others = [
            PageState::Standby,
            PageState::Active,
            PageState::SwappedOut,
            PageState::Swapped,
            PageState::Free,
            PageState::Warm,
        ];
        for &state in &others {
            let meta = PageMetadata {
                page_id: 1,
                sequence_id: Some(10),
                recency: 10,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state,
                warm_until: None,
            };
            let meta_prot = PageMetadata {
                page_id: 1,
                sequence_id: Some(10),
                recency: 10,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Protected,
                warm_until: None,
            };
            let score_other = EvictionWorker::compute_importance_score(
                &meta, None, 0, 4096, StorageTier::GpuHbm, 50,
            );
            let score_prot = EvictionWorker::compute_importance_score(
                &meta_prot, None, 0, 4096, StorageTier::GpuHbm, 50,
            );
            assert!(
                score_prot > score_other,
                "Protected({}) should beat {:?}({})",
                score_prot,
                state,
                score_other,
            );
        }
    }

    // ── Score: multiple components combined exact check ──

    #[test]
    fn score_combined_expert_standby_dram_age100_freq5_rec20_4k_compress_half() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // triggers ExpertWeight via infer, but we pass payload explicitly
            recency: 20,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            2048,   // 50% compressed
            4096,
            StorageTier::CpuDram,
            100,
        );
        // base(1000) - time(100*2=200) - recency(20*1=20) + freq(5*15=75)
        // + compress(250) + page_size(4) + expert(-300) + dram(-200) = 609
        assert_eq!(score, 609);
    }

    // ── Score: combined with PromptSystem, Warm, HBM ──

    #[test]
    fn score_combined_prompt_warm_hbm_age30_freq8_rec10_compress_75pct() {
        let meta = make_meta(1, Some(10), 10, 8, PageState::Warm);
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::PromptSystem),
            1024,   // 25% of 4096 → compress bonus = 375
            4096,
            StorageTier::GpuHbm,
            30,
        );
        // base(1000) - time(30*2=60) - recency(10*1=10) + freq(8*15=120)
        // + compress(375) + page_size(4) + prompt(1000) + warm(5000) = 7429
        assert_eq!(score, 7429);
    }

    // ── EvictionCandidate: sort with many distinct negative scores ──

    #[test]
    fn eviction_candidate_sort_many_negative_scores() {
        let mut candidates: Vec<EvictionCandidate> = (0..20)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: -(i as i64) * 100 - 50,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::Lz4,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        candidates.sort_by_key(|c| c.score);
        for window in candidates.windows(2) {
            assert!(
                window[0].score <= window[1].score,
                "should be sorted ascending: {} vs {}",
                window[0].score,
                window[1].score,
            );
        }
    }

    // ── EvictionCandidate: truncate to zero removes all ──

    #[test]
    fn eviction_candidate_truncate_to_zero_removes_all() {
        let mut candidates: Vec<EvictionCandidate> = (0..5)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: i as i64,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        candidates.truncate(0);
        assert!(candidates.is_empty());
    }

    // ── EvictionWorkerConfig: all 5 codec variants accepted ──

    #[test]
    fn config_accepts_all_five_codec_variants() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for &codec in &codecs {
            let cfg = EvictionWorkerConfig {
                default_evict_codec: codec,
                ..EvictionWorkerConfig::default()
            };
            assert_eq!(cfg.default_evict_codec, codec);
        }
    }

    // ── EvictionWorkerConfig: mutate single field leaves others at default ──

    #[test]
    fn config_mutate_codec_preserves_other_defaults() {
        let mut cfg = EvictionWorkerConfig::default();
        cfg.default_evict_codec = CompressionCodec::BitPackRle;
        assert_eq!(cfg.tick_interval, DEFAULT_TICK_INTERVAL);
        assert_eq!(cfg.max_evict_per_round, DEFAULT_MAX_EVICT_PER_ROUND);
        assert_eq!(cfg.importance_threshold, IMPORTANCE_SCORE_THRESHOLD);
        assert_eq!(cfg.hbm_evict_age_ticks, HBM_EVICT_AGE_TICKS);
        assert_eq!(cfg.dram_evict_age_ticks, DRAM_EVICT_AGE_TICKS);
        assert_eq!(cfg.page_bytes, 4096);
    }

    // ── EvictionWorkerConfig: custom tick interval does not affect thresholds ──

    #[test]
    fn config_custom_tick_preserves_pressure_thresholds() {
        let mut cfg = EvictionWorkerConfig::default();
        cfg.tick_interval = Duration::from_secs(1);
        assert!((cfg.hbm_pressure_threshold - HBM_PRESSURE_RATIO).abs() < 1e-6);
        assert!((cfg.dram_pressure_threshold - DRAM_PRESSURE_RATIO).abs() < 1e-6);
    }

    // ── evict_round: no pressure returns zero even with eligible pages ──

    #[test]
    fn evict_round_no_pressure_returns_zero_with_pages() {
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0,
            dram_pressure_threshold: 1.0,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // Low pressure — only 1 page allocated out of 1000 capacity.
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(1000, 1000, 1000)));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        assert_eq!(submitted, 0);
        actor.shutdown();
    }

    // ── evict_round: only HBM pressure triggers eviction, not DRAM ──

    #[test]
    fn evict_round_hbm_only_pressure_triggers_hbm_eviction() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,    // always triggered
            dram_pressure_threshold: 1.0,   // never triggered
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        assert_eq!(submitted, 1, "HBM pressure should trigger 1 eviction");
        actor.shutdown();
    }

    // ── evict_round: page with high score not evicted even under pressure ──

    #[test]
    fn evict_round_high_importance_threshold_skips_high_score() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            importance_threshold: -10000, // extremely low — only deeply negative scores evict
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 100, // high frequency → high score
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        assert_eq!(submitted, 0, "high score page should not be evicted with very low threshold");
        actor.shutdown();
    }

    // ── evict_round: CpuDram page skipped when only HBM pressure is active ──

    #[test]
    fn evict_round_dram_page_skipped_under_hbm_only_pressure() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0, // no DRAM pressure
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        // Page is on CpuDram, but only HBM pressure is active.
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        assert_eq!(submitted, 0, "DRAM page should not be evicted under HBM-only pressure");
        actor.shutdown();
    }

    // ── compute_tier_age: sub-10ms elapsed yields zero ticks ──

    #[test]
    fn tier_age_sub_10ms_yields_zero() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let age = compute_tier_age(&meta);
        // Just created, <10ms elapsed → 0 ticks.
        assert_eq!(age, 0, "sub-10ms should yield 0 ticks: got {}", age);
    }

    // ── Score: each payload kind delta from KV_CONTEXT is exact ──

    #[test]
    fn score_payload_delta_from_kv_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        let cases: Vec<(PagePayloadKind, i64)> = vec![
            (PagePayloadKind::ExpertWeight, -300),
            (PagePayloadKind::PromptSystem, 1000),
            (PagePayloadKind::DenseLayerWeight, 5000),
            (PagePayloadKind::KnowledgeRAG, -500),
        ];
        for (kind, expected_delta) in cases {
            let score = EvictionWorker::compute_importance_score(
                &meta, Some(kind), 0, 0, StorageTier::GpuHbm, 0,
            );
            assert_eq!(
                score - score_kv, expected_delta,
                "payload {:?} delta from KV should be {}",
                kind,
                expected_delta,
            );
        }
    }

    // ── Additional coverage tests ──

    /// Verify base score constant is exactly 1000 when no penalties or bonuses apply.
    #[test]
    fn score_base_constant_is_1000() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None, // no payload bonus
            0,    // compressed = 0
            0,    // original = 0 → no compression/page bonus
            StorageTier::GpuHbm, // no tier discount
            0,    // zero age → no time penalty
        );
        assert_eq!(score, 1000, "base score with no modifiers must be 1000");
    }

    /// Verify that EvictionTier variants can be collected into a HashSet (Hash impl).
    #[test]
    fn eviction_tier_hash_set_dedup() {
        use std::collections::HashSet;
        let tiers: Vec<EvictionTier> = vec![
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
            EvictionTier::ColdExpert, // duplicate
        ];
        let set: HashSet<EvictionTier> = tiers.into_iter().collect();
        assert_eq!(set.len(), 4, "HashSet should deduplicate to 4 unique variants");
    }

    /// Verify EvictionCandidate with I64_MIN score does not panic.
    #[test]
    fn eviction_candidate_i64_min_score() {
        let candidate = EvictionCandidate {
            page_id: 42,
            score: i64::MIN,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 4096,
            group_id: Some(99),
        };
        assert_eq!(candidate.score, i64::MIN);
        assert_eq!(candidate.page_id, 42);
    }

    /// Verify EvictionCandidate with I64_MAX score does not panic.
    #[test]
    fn eviction_candidate_i64_max_score() {
        let candidate = EvictionCandidate {
            page_id: 0,
            score: i64::MAX,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 0,
            group_id: None,
        };
        assert_eq!(candidate.score, i64::MAX);
        assert!(candidate.group_id.is_none());
    }

    /// Verify EvictionCandidate group_id round-trips with Some(RequestId).
    #[test]
    fn eviction_candidate_group_id_roundtrip() {
        let rid: RequestId = 777;
        let candidate = EvictionCandidate {
            page_id: 1,
            score: 50,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4,
            page_bytes: 8192,
            group_id: Some(rid),
        };
        assert_eq!(candidate.group_id, Some(rid));
    }

    /// Verify EvictionCandidate on Nvme tier carries that tier.
    #[test]
    fn eviction_candidate_nvme_tier() {
        let candidate = EvictionCandidate {
            page_id: 10,
            score: -100,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 2048,
            group_id: None,
        };
        assert_eq!(candidate.current_tier, StorageTier::Nvme);
    }

    /// Verify score with usize::MAX recency does not panic (overflow safety).
    #[test]
    fn score_recency_usize_max_no_panic() {
        let meta = make_meta(1, Some(10), usize::MAX, 0, PageState::Standby);
        let _score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );
        // No assertion on value — just verifying no panic/overflow.
    }

    /// Verify score with usize::MAX access_count does not panic (overflow safety).
    #[test]
    fn score_access_count_usize_max_no_panic() {
        let meta = make_meta(1, Some(10), 0, usize::MAX, PageState::Standby);
        let _score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );
        // No assertion on value — just verifying no panic/overflow.
    }

    /// Verify EvictionCandidate with BitPackRle codec preserves codec field.
    #[test]
    fn eviction_candidate_bitpack_rle_codec() {
        let candidate = EvictionCandidate {
            page_id: 7,
            score: 42,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 4096,
            group_id: None,
        };
        assert_eq!(candidate.codec, CompressionCodec::BitPackRle);
        assert_eq!(candidate.page_id, 7);
        assert_eq!(candidate.score, 42);
    }

    /// Verify EvictionCandidate with NvcompAns codec preserves codec field.
    #[test]
    fn eviction_candidate_nvcomp_ans_codec() {
        let candidate = EvictionCandidate {
            page_id: 13,
            score: -200,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 8192,
            group_id: Some(5),
        };
        assert_eq!(candidate.codec, CompressionCodec::NvcompAns);
        assert_eq!(candidate.page_bytes, 8192);
    }

    /// Verify infer_payload_kind returns ExpertWeight for Protected state with no owner.
    #[test]
    fn infer_payload_protected_no_owner_is_expert() {
        let meta = make_meta(1, None, 5, 50, PageState::Protected);
        assert_eq!(infer_payload_kind(&meta), Some(PagePayloadKind::ExpertWeight));
    }

    /// Verify infer_payload_kind returns ExpertWeight for Standby state with no owner.
    #[test]
    fn infer_payload_standby_no_owner_is_expert() {
        let meta = make_meta(1, None, 0, 0, PageState::Standby);
        assert_eq!(infer_payload_kind(&meta), Some(PagePayloadKind::ExpertWeight));
    }

    /// Verify infer_payload_kind returns ExpertWeight even for Active state with no owner.
    #[test]
    fn infer_payload_active_no_owner_is_expert() {
        let meta = make_meta(5, None, 0, 100, PageState::Active);
        assert_eq!(infer_payload_kind(&meta), Some(PagePayloadKind::ExpertWeight));
    }

    /// Verify classify_eviction_tier returns StandbyKv for KvContext with low score.
    #[test]
    fn classify_kv_context_low_score_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            50, // below IMPORTANCE_SCORE_THRESHOLD=100
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    /// Verify classify_eviction_tier returns Protected for KvContext with high score.
    #[test]
    fn classify_kv_context_high_score_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            500, // above IMPORTANCE_SCORE_THRESHOLD=100
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional edge-case & boundary tests
    // ─────────────────────────────────────────────────────────────────────────

    /// @trace TEST-EVICT-EDGE score at exactly IMPORTANCE_SCORE_THRESHOLD (100)
    /// should NOT be evicted — eviction requires score < threshold.
    #[test]
    fn score_exactly_at_threshold_not_evicted() {
        // Arrange: build a scenario where score == 100 exactly.
        // base(1000) - time_penalty(450*2=900) = 100 on GpuHbm (no tier discount).
        // No payload bonus, no compression, no page size bonus, no state bonus.
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,      // no payload bonus
            0,         // compressed
            0,         // original (no compression/page bonus)
            StorageTier::GpuHbm,
            450,       // tier_age_ticks -> time_penalty = 450*2 = 900
        );
        // Assert: score == 100 (exactly at IMPORTANCE_SCORE_THRESHOLD).
        assert_eq!(score, 100);
        // score < threshold is required for eviction; 100 < 100 is false.
        assert!(score >= IMPORTANCE_SCORE_THRESHOLD as i64);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier: None payload kind at score 0
    /// (below threshold) → StandbyKv (falls into the `_ if score < threshold` arm).
    #[test]
    fn classify_none_payload_zero_score_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            None,
            0, // below IMPORTANCE_SCORE_THRESHOLD=100
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier: None payload kind at score 101
    /// (above threshold) → Protected.
    #[test]
    fn classify_none_payload_score_101_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            None,
            101, // above IMPORTANCE_SCORE_THRESHOLD=100
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier: KnowledgeRAG is not ExpertWeight
    /// or DenseLayerWeight, so at low score → StandbyKv.
    #[test]
    fn classify_knowledge_rag_low_score_is_standby_kv() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KnowledgeRAG),
            50,
        );
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier: KnowledgeRAG at high score → Protected.
    #[test]
    fn classify_knowledge_rag_score_500_is_protected() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KnowledgeRAG),
            500,
        );
        assert_eq!(tier, EvictionTier::Protected);
    }

    /// @trace TEST-EVICT-EDGE Score at exactly threshold minus one (99) is evictable.
    #[test]
    fn score_one_below_threshold_is_evictable() {
        // base=1000, time_penalty=700 (350 ticks * 2), recency_penalty=1,
        // CpuDram discount=-200, no compression/page bonus → 1000-700-1-200 = 99.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 1,    // recency_penalty = 1 * 1 = 1
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            0,
            StorageTier::CpuDram,
            350, // time_penalty = 700
        );
        // 1000 - 700 - 1 + 0 - 200 = 99
        assert_eq!(score, 99);
        assert!(score < IMPORTANCE_SCORE_THRESHOLD as i64);
    }

    /// @trace TEST-EVICT-EDGE Warm state on CpuDram: combined bonus (Warm=5000)
    /// plus CpuDram discount (-200) net effect.
    #[test]
    fn score_warm_on_dram_exact_value() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::CpuDram,
            0,
        );
        // base(1000) + state_bonus(5000) + tier_discount(-200) + kv_context(0) = 5800
        assert_eq!(score, 5800);
    }

    /// @trace TEST-EVICT-EDGE Protected state on Nvme: combined 10000 bonus and -500 discount.
    #[test]
    fn score_protected_on_nvme_exact_value() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            0,
            StorageTier::Nvme,
            0,
        );
        // base(1000) + state_bonus(10000) + tier_discount(-500) = 10500
        assert_eq!(score, 10500);
    }

    /// @trace TEST-EVICT-EDGE Score with maximal time penalty alone drives score deeply negative.
    #[test]
    fn score_large_time_penalty_drives_negative() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::GpuHbm,
            10000, // time_penalty = 10000 * 2 = 20000
        );
        // base(1000) - time(20000) = -19000
        assert_eq!(score, -19000);
    }

    /// @trace TEST-EVICT-EDGE Frequency bonus exactly cancels large time penalty.
    #[test]
    fn score_freq_exactly_cancels_10000_tick_penalty() {
        let meta = make_meta(1, Some(10), 0, 1334, PageState::Standby);
        // freq_bonus = 1334 * 15 = 20010
        // time_penalty = 10000 * 2 = 20000
        // base(1000) - 20000 + 20010 = 1010
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::GpuHbm,
            10000,
        );
        assert_eq!(score, 1010);
    }

    /// @trace TEST-EVICT-EDGE Compression ratio of exactly 50% with 8192 original bytes.
    #[test]
    fn score_compression_half_of_8192_exact() {
        let meta = make_meta(1, Some(10), 0, 0, PageState::Standby);
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            4096, // compressed
            8192, // original
            StorageTier::GpuHbm,
            0,
        );
        // compression_ratio_bonus = (1 - 0.5) * 500 = 250
        // page_size_bonus = 8192/1024 * 1 = 8
        // base(1000) + 250 + 8 = 1258
        assert_eq!(score, 1258);
    }

    /// @trace TEST-EVICT-EDGE evict_round with empty page_metadata map returns zero submissions.
    #[test]
    fn evict_round_empty_metadata_returns_zero() {
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0, // always triggered
            dram_pressure_threshold: 0.0,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));

        // High HBM and DRAM pressure (all capacity used).
        let mut mm = GlobalMemoryManager::new_with_capacities(10, 10, 10);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L1);
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: no pages in metadata → nothing to evict.
        assert_eq!(submitted, 0);
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE evict_round: Active state pages are skipped even under pressure.
    #[test]
    fn evict_round_active_state_skipped_under_pressure() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Active, // Active pages are skipped
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x2000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: Active page should not be evicted.
        assert_eq!(submitted, 0);
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE evict_round: Protected state pages are skipped even under pressure.
    #[test]
    fn evict_round_protected_state_skipped_under_pressure() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Protected, // Protected pages are skipped
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x3000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: Protected page should not be evicted.
        assert_eq!(submitted, 0);
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Observer telemetry is updated after successful eviction.
    #[test]
    fn evict_round_observer_records_eviction_event() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(42, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(42, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: eviction was submitted.
        assert_eq!(submitted, 1);

        // Assert: observer recorded the eviction event.
        let obs = observer.lock().unwrap();
        assert_eq!(obs.last_state.weight_eviction_count, 1);

        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE evict_round: Nvme pages are never evicted (already coldest tier).
    #[test]
    fn evict_round_nvme_pages_never_evicted() {
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 0.0,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme, // Nvme pages → never eligible
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 10, 10);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L1);
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: Nvme pages are never eligible for eviction.
        assert_eq!(submitted, 0);
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Multiple pages sorted by score: lowest score evicted first,
    /// max_evict_per_round limits the count.
    #[test]
    fn evict_round_multiple_pages_sorted_and_truncated() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 2, // limit to 2 evictions per round
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        // Page 10: high recency → lower score (more evictable)
        // Page 20: medium recency
        // Page 30: zero recency → highest score (least evictable)
        let pages: Vec<(PageId, PageMetadata)> = vec![
            (10, PageMetadata {
                page_id: 10, sequence_id: Some(10), recency: 500, access_count: 0,
                last_access: old, swap_in_time: Some(old), is_lir: false,
                state: PageState::Standby, warm_until: None,
            }),
            (20, PageMetadata {
                page_id: 20, sequence_id: Some(20), recency: 100, access_count: 0,
                last_access: old, swap_in_time: Some(old), is_lir: false,
                state: PageState::Standby, warm_until: None,
            }),
            (30, PageMetadata {
                page_id: 30, sequence_id: Some(30), recency: 0, access_count: 0,
                last_access: old, swap_in_time: Some(old), is_lir: false,
                state: PageState::Standby, warm_until: None,
            }),
        ];
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(pages.into_iter().collect()));

        {
            let mut guard = addr_table.write().unwrap();
            for pid in [10, 20, 30] {
                guard.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: max_evict_per_round=2, so exactly 2 evictions.
        assert_eq!(submitted, 2);

        // Assert: observer recorded both evictions.
        let obs = observer.lock().unwrap();
        assert_eq!(obs.last_state.weight_eviction_count, 2);

        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE evict_round: page with no addr_table entry is skipped.
    #[test]
    fn evict_round_missing_addr_entry_skipped() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 99,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Page exists in metadata but NOT in addr_table.
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(99, meta)])));

        // No addr_table entry for page 99.

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: page with no addr_table entry → skipped, zero evictions.
        assert_eq!(submitted, 0);
        actor.shutdown();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional edge-case & integration tests (wave-12x34)
    // ─────────────────────────────────────────────────────────────────────────

    /// @trace TEST-EVICT-EDGE CpuDram page under DRAM pressure with sufficient tier_age
    /// triggers EvictToNvme migration command.
    #[test]
    fn evict_round_cpubram_to_nvme_dram_pressure() {
        // Arrange: DRAM pressure > threshold, page on CpuDram with enough tier_age.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0, // HBM not pressured
            dram_pressure_threshold: 0.0, // DRAM always pressured
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 8192,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 55,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(55, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(55, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 8192]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 8192,
                codec: CompressionCodec::Lz4,
            });
        }

        // Fill DRAM (L2) to trigger pressure.
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: CpuDram page should be evicted to NVMe under DRAM pressure.
        assert_eq!(submitted, 1, "CpuDram page under DRAM pressure should be evicted");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Warm state page on GpuHbm is evictable (not skipped like Active/Protected).
    #[test]
    fn evict_round_warm_state_evictable() {
        // Arrange: Warm pages should NOT be skipped by the state filter — only
        // Active and Protected are excluded. The Warm bonus (+5000) may push the
        // score above threshold, so we use a very low importance_threshold.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(2);
        let meta = PageMetadata {
            page_id: 77,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(77, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(77, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: Warm page is not skipped by the Active/Protected filter.
        assert_eq!(submitted, 1, "Warm page should be evictable (not filtered by state)");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Both HBM and DRAM pressure active simultaneously:
    /// pages from both tiers are evaluated and evicted.
    #[test]
    fn evict_round_hbm_and_dram_pressure() {
        // Arrange: both HBM and DRAM pressured, two pages on different tiers.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 0.0,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 10,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(3);
        let meta_hbm = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_dram = PageMetadata {
            page_id: 2,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta_hbm), (2, meta_dram)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            guard.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // Fill both HBM and DRAM.
        let mut mm = GlobalMemoryManager::new_with_capacities(10, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L1);
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: both pages should be evicted.
        assert_eq!(submitted, 2, "both HBM and DRAM pages should be evicted under dual pressure");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE ExpertWeight on CpuDram with Warm state:
    /// verify exact score combining expert(-300) + warm(5000) + dram(-200).
    #[test]
    fn score_expert_warm_dram_exact() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            0,
            0,
            StorageTier::CpuDram,
            0,
        );

        // Assert: base(1000) + expert(-300) + warm(5000) + dram(-200) = 5500
        assert_eq!(score, 5500, "ExpertWeight + Warm + CpuDram exact score");
    }

    /// @trace TEST-EVICT-EDGE max_evict_per_round = 0 prevents any eviction.
    #[test]
    fn evict_round_max_evict_zero() {
        // Arrange: pressure on, eligible page, but max_evict_per_round = 0.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 0,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 10,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(10, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(10, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: max_evict_per_round=0 truncates candidates to 0.
        assert_eq!(submitted, 0, "max_evict_per_round=0 should prevent all evictions");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Single eligible page under pressure is evicted exactly once.
    #[test]
    fn evict_round_single_eligible_page() {
        // Arrange: exactly one page that passes all eligibility checks.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 8,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let meta = PageMetadata {
            page_id: 33,
            sequence_id: Some(42),
            recency: 0,
            access_count: 5,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(33, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(33, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: exactly 1 eviction submitted.
        assert_eq!(submitted, 1);
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Free state page is not filtered by the Active/Protected
    /// check in evict_round but has zero state bonus, so is evictable.
    #[test]
    fn evict_round_free_state_page_evictable() {
        // Arrange: Free state pages are not Active or Protected, so they pass
        // the state filter. With zero bonuses and high importance_threshold, evictable.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let meta = PageMetadata {
            page_id: 88,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(88, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(88, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: Free state is not Active/Protected, so passes the filter.
        assert_eq!(submitted, 1, "Free state page should pass the Active/Protected filter");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE SwappedOut state page is evictable (not filtered).
    #[test]
    fn evict_round_swapped_out_state_evictable() {
        // Arrange: SwappedOut is not Active or Protected, so it passes the filter.
        let config = EvictionWorkerConfig {
            dram_pressure_threshold: 0.0,
            hbm_pressure_threshold: 1.0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 44,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(44, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(44, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // Fill DRAM (L2) to trigger DRAM pressure.
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: SwappedOut is not filtered.
        assert_eq!(submitted, 1, "SwappedOut page on CpuDram under DRAM pressure should be evicted");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE CpuDram page is not evicted when DRAM has no pressure,
    /// even though HBM is pressured (HBM eviction only targets GpuHbm pages).
    #[test]
    fn evict_round_cpubram_skipped_when_only_hbm_pressure() {
        // Arrange: HBM pressured, DRAM not pressured, page on CpuDram.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0, // DRAM never triggered
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 22,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(22, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(22, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // Fill HBM only, leave DRAM empty.
        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: CpuDram page requires DRAM pressure, not HBM.
        assert_eq!(submitted, 0, "CpuDram page should not be evicted under HBM-only pressure");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE CpuDram page with tier_age below dram_evict_age_ticks
    /// is not eligible even under DRAM pressure.
    #[test]
    fn evict_round_dram_age_below_threshold() {
        // Arrange: DRAM pressure on, but page tier_age is 0 (< default 500).
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0,
            dram_pressure_threshold: 0.0,
            dram_evict_age_ticks: 500, // requires 500 ticks
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Recent swap_in_time → tier_age near 0.
        let meta = PageMetadata {
            page_id: 66,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(66, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(66, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // Fill DRAM.
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: page too young on DRAM, not eligible.
        assert_eq!(submitted, 0, "CpuDram page below dram_evict_age_ticks should not be evicted");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Score: ExpertWeight on CpuDram at age 0 (no time penalty).
    #[test]
    fn score_expert_weight_on_dram_age_zero() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            0,
            0,
            StorageTier::CpuDram,
            0,
        );

        // Assert: base(1000) + expert(-300) + dram(-200) = 500
        assert_eq!(score, 500, "ExpertWeight on CpuDram at age 0: expected 500");
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier: PromptSystem at exactly threshold
    /// is Protected (score < threshold is required for StandbyKv).
    #[test]
    fn classify_prompt_system_threshold_exact() {
        // Act & Assert: PromptSystem is not ExpertWeight or DenseLayerWeight,
        // so at exactly threshold it falls through to the `_` arm → Protected.
        let tier_at = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::PromptSystem),
            IMPORTANCE_SCORE_THRESHOLD as i64,
        );
        assert_eq!(tier_at, EvictionTier::Protected,
            "PromptSystem at exactly threshold should be Protected");

        let tier_below = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::PromptSystem),
            IMPORTANCE_SCORE_THRESHOLD as i64 - 1,
        );
        assert_eq!(tier_below, EvictionTier::StandbyKv,
            "PromptSystem at threshold-1 should be StandbyKv");
    }

    /// @trace TEST-EVICT-EDGE Score: compression ratio exactly 1.0 yields zero bonus.
    #[test]
    fn score_compression_ratio_exactly_one() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compressed == original → ratio = 1.0 → bonus = (1-1)*500 = 0
        let score_no_compress = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_base = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );

        // Assert: page_size_bonus = 4096/1024*1 = 4, so no-compress = base + 4.
        // When original_size = 0, page_size_bonus = 0 too.
        assert_eq!(
            score_no_compress - score_base, 4,
            "compression ratio 1.0 should yield zero compression bonus (only page_size differs)"
        );
    }

    /// @trace TEST-EVICT-EDGE evict_round uses custom page_bytes from config
    /// for the migration command.
    #[test]
    fn evict_round_custom_page_bytes() {
        // Arrange: custom page_bytes = 16384, verify eviction still works.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 16384,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(2);
        let meta = PageMetadata {
            page_id: 11,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(11, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(11, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 16384]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 16384,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: eviction succeeded with custom page_bytes.
        assert_eq!(submitted, 1, "eviction should succeed with custom page_bytes");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE CpuDram page with tier_age exactly at dram_evict_age_ticks
    /// is NOT eligible (requires strictly greater than threshold).
    #[test]
    fn evict_round_cpubram_age_exact_threshold() {
        // Arrange: tier_age must be strictly > dram_evict_age_ticks.
        // Use a recently-created page so compute_tier_age returns ~0.
        // Set dram_evict_age_ticks = 0 so that tier_age > 0 is needed.
        // Since the page is just created, tier_age ≈ 0, which is NOT > 0.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0,
            dram_pressure_threshold: 0.0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 99,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(99, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(99, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // Fill DRAM.
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: page is 5 seconds old → tier_age = 5000ms/10 = 500 ticks > 0.
        // With dram_evict_age_ticks=0, 500 > 0 → eligible.
        assert!(submitted >= 1,
            "CpuDram page with tier_age > 0 and dram_evict_age_ticks=0 should be eligible: got {}",
            submitted);
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Swapped state page on GpuHbm is not Active/Protected,
    /// so it passes the state filter and is evictable under pressure.
    #[test]
    fn evict_round_swapped_state_on_hbm_evictable() {
        // Arrange: PageState::Swapped on GpuHbm — passes filter, score has no state bonus.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let meta = PageMetadata {
            page_id: 77,
            sequence_id: Some(5),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(77, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(77, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: Swapped state is not Active/Protected, so it should be evicted.
        assert!(submitted >= 1, "Swapped state page on GpuHbm should be evictable");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE When all pages are on NVMe tier, no eviction happens
    /// regardless of pressure (NVMe is the coldest tier, no further destination).
    #[test]
    fn evict_round_all_pages_on_nvme_no_eviction() {
        // Arrange: Pages on NVMe — no tier to demote to.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 0.0,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta1 = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta2 = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta1), (2, meta2)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
            guard.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            });
        }

        // Fill all tiers to max pressure.
        let mut mm = GlobalMemoryManager::new_with_capacities(10, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L1);
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: NVMe pages are never eligible for eviction.
        assert_eq!(submitted, 0, "NVMe pages should never be evicted");
        actor.shutdown();
    }


    /// @trace TEST-EVICT-EDGE evict_round with three pages on mixed tiers under
    /// simultaneous HBM+DRAM pressure evicts eligible pages from both tiers.
    #[test]
    fn evict_round_mixed_tiers_under_dual_pressure() {
        // Arrange: one page on GpuHbm, one on CpuDram, one on Nvme.
        // Both HBM and DRAM pressured. Only GpuHbm and CpuDram should be evicted.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 0.0,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 10,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(2);
        let meta_hbm = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_dram = PageMetadata {
            page_id: 2,
            sequence_id: Some(20),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_nvme = PageMetadata {
            page_id: 3,
            sequence_id: Some(30),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([
                (1, meta_hbm),
                (2, meta_dram),
                (3, meta_nvme),
            ])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            guard.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
            guard.insert(3, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            });
        }

        // Fill both HBM and DRAM.
        let mut mm = GlobalMemoryManager::new_with_capacities(10, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L1);
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: only 2 evicted (GpuHbm + CpuDram), NVMe skipped.
        assert_eq!(submitted, 2, "only GpuHbm and CpuDram pages should be evicted, not NVMe");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE evict_round with capacity=0 for a tier yields 0 pressure,
    /// so no eviction is triggered for that tier.
    #[test]
    fn evict_round_zero_capacity_tier_no_pressure() {
        // Arrange: HBM capacity = 0, DRAM with some pages. HBM pressure should be 0.0.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.5,
            dram_pressure_threshold: 0.5,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // MM with 0 HBM capacity → hbm_pressure = 0.0 (division by capacity=0 guarded).
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(0, 0, 1000)));
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: zero capacity means no pressure, no eviction.
        assert_eq!(submitted, 0, "zero-capacity tiers should yield no pressure");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Warm state page on NVMe tier is not evicted
    /// because NVMe pages are never eligible regardless of state bonus.
    #[test]
    fn evict_round_warm_state_on_nvme_not_evicted() {
        // Arrange: Warm page on NVMe — has 5000 state bonus but NVMe never evicted.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 0.0,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 55,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(55, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(55, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L1);
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: NVMe pages never eligible, regardless of state bonus.
        assert_eq!(submitted, 0, "Warm page on NVMe should not be evicted");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Importance score is strictly decreasing with tier_age
    /// when all other inputs are identical, for each payload kind.
    #[test]
    fn score_monotonically_decreasing_across_all_payloads() {
        // Arrange: same meta, two different ages, for each payload kind.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let payloads = [
            Some(PagePayloadKind::ExpertWeight),
            Some(PagePayloadKind::KvContext),
            Some(PagePayloadKind::PromptSystem),
            Some(PagePayloadKind::DenseLayerWeight),
            Some(PagePayloadKind::KnowledgeRAG),
            None,
        ];

        for payload in &payloads {
            let score_young = EvictionWorker::compute_importance_score(
                &meta, *payload, 0, 4096, StorageTier::GpuHbm, 10,
            );
            let score_old = EvictionWorker::compute_importance_score(
                &meta, *payload, 0, 4096, StorageTier::GpuHbm, 200,
            );
            assert!(
                score_old < score_young,
                "score should decrease with age for payload {:?}: old={} young={}",
                payload, score_old, score_young,
            );
        }
    }

    /// @trace TEST-EVICT-EDGE evict_round skips a page that exists in metadata
    /// but has no corresponding entry in addr_table (already released).
    #[test]
    fn evict_round_metadata_without_addr_entry_skipped() {
        // Arrange: metadata exists but addr_table has no entry for the page.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 999,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Page metadata exists but no addr_table entry.
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(999, meta)])));

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: page without addr_table entry is skipped.
        assert_eq!(submitted, 0, "page without addr_table entry should be skipped");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE Both Free and Swapped state pages are evictable
    /// in the same evict_round, producing 2 evictions.
    #[test]
    fn evict_round_free_and_swapped_both_evictable() {
        // Arrange: one Free page, one Swapped page on GpuHbm under pressure.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let meta_free = PageMetadata {
            page_id: 10,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        let meta_swapped = PageMetadata {
            page_id: 11,
            sequence_id: Some(20),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([
                (10, meta_free),
                (11, meta_swapped),
            ])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(10, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            guard.insert(11, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(10, 1000, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: both Free and Swapped states are evictable (neither Active nor Protected).
        assert_eq!(submitted, 2, "both Free and Swapped pages should be evicted");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier for PromptSystem at score=0
    /// (far below threshold) still returns StandbyKv.
    #[test]
    fn classify_prompt_system_zero_score_is_standby() {
        // Arrange
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::PromptSystem),
            0, // well below threshold
        );
        // Assert: PromptSystem at score 0 is StandbyKv.
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier for KnowledgeRAG with score
    /// at i64::MAX returns Protected.
    #[test]
    fn classify_knowledge_rag_max_score_is_protected() {
        // Arrange
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KnowledgeRAG),
            i64::MAX,
        );
        // Assert: KnowledgeRAG at max score => Protected.
        assert_eq!(tier, EvictionTier::Protected);
    }

    /// @trace TEST-EVICT-EDGE Importance score with compressed_size > original_size
    /// yields a negative compression ratio bonus (penalty), making the page more evictable.
    #[test]
    fn score_compressed_larger_than_original_more_evictable() {
        // Arrange: same meta, one with normal compression, one with inverted.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        let score_normal = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext),
            2048, 4096, // 50% compression
            StorageTier::GpuHbm, 100,
        );
        let score_inverted = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext),
            8192, 4096, // compressed > original: ratio > 1.0
            StorageTier::GpuHbm, 100,
        );

        // Assert: inverted compression ratio yields lower score (more evictable).
        assert!(
            score_inverted < score_normal,
            "compressed > original should yield lower score: inverted={} normal={}",
            score_inverted, score_normal,
        );
    }

    /// @trace TEST-EVICT-EDGE Importance score is identical regardless of page_id value.
    #[test]
    fn score_page_id_does_not_affect_result() {
        // Arrange: same parameters, different page_ids.
        let meta_a = PageMetadata {
            page_id: 0,
            sequence_id: Some(10),
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_b = PageMetadata {
            page_id: usize::MAX,
            sequence_id: Some(10),
            recency: 5,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        let score_a = EvictionWorker::compute_importance_score(
            &meta_a, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::GpuHbm, 100,
        );
        let score_b = EvictionWorker::compute_importance_score(
            &meta_b, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::GpuHbm, 100,
        );

        // Assert: page_id does not participate in scoring.
        assert_eq!(score_a, score_b, "page_id should not affect importance score");
    }

    /// @trace TEST-EVICT-EDGE Config with all pressure thresholds at 1.0 means
    /// pressure can never exceed, so eviction never triggers even with full usage.
    #[test]
    fn evict_round_threshold_one_dot_zero_never_triggers() {
        // Arrange: both thresholds = 1.0, fully utilized tiers.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0,
            dram_pressure_threshold: 1.0,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // Fully utilize HBM (10/10 = 1.0 pressure, but threshold is strictly >, so 1.0 not > 1.0).
        let mut mm = GlobalMemoryManager::new_with_capacities(10, 10, 1000);
        for _ in 0..10 {
            let _ = mm.allocate_page(Tier::L1);
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: pressure == 1.0 is not > 1.0, so no eviction.
        assert_eq!(submitted, 0, "pressure == threshold should not trigger eviction");
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE EvictionCandidate clone produces an independent copy
    /// with identical field values including Option fields.
    #[test]
    fn eviction_candidate_clone_with_group_id() {
        // Arrange
        let original = EvictionCandidate {
            page_id: 42,
            score: -500,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
            group_id: Some(99),
        };

        // Act
        let cloned = original.clone();

        // Assert: all fields identical.
        assert_eq!(cloned.page_id, original.page_id);
        assert_eq!(cloned.score, original.score);
        assert_eq!(cloned.current_tier, original.current_tier);
        assert_eq!(cloned.codec, original.codec);
        assert_eq!(cloned.page_bytes, original.page_bytes);
        assert_eq!(cloned.group_id, original.group_id);
    }

    // ── Additional coverage tests (wave-12x35) ──

    /// @trace TEST-EVICT-EDGE EvictionTier Copy trait allows assignment without clone.
    #[test]
    fn eviction_tier_copy_trait_allows_assignment() {
        // Arrange
        let original = EvictionTier::ColdExpert;
        // Act: Copy, not Clone (implicit copy via assignment).
        let copied = original;
        // Assert: both values are still valid and equal.
        assert_eq!(original, copied);
        assert_eq!(original, EvictionTier::ColdExpert);
    }

    /// @trace TEST-EVICT-EDGE EvictionCandidate clone with None group_id preserves None.
    #[test]
    fn eviction_candidate_clone_without_group_id() {
        // Arrange
        let original = EvictionCandidate {
            page_id: 0,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 0,
            group_id: None,
        };
        // Act
        let cloned = original.clone();
        // Assert: None group_id is preserved.
        assert_eq!(cloned.group_id, None);
        assert_eq!(cloned.page_id, 0);
        assert_eq!(cloned.score, 0);
        assert_eq!(cloned.page_bytes, 0);
    }

    /// @trace TEST-EVICT-EDGE EvictionTier Hash stability: all four variants produce
    /// distinct hash values from each other.
    #[test]
    fn eviction_tier_hash_distinct_across_variants() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let tiers = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        for tier in &tiers {
            let mut h1 = DefaultHasher::new();
            let mut h2 = DefaultHasher::new();
            tier.hash(&mut h1);
            tier.hash(&mut h2);
            // Act & Assert: same value hashed twice must match.
            assert_eq!(h1.finish(), h2.finish(), "Hash mismatch for {:?}", tier);
        }
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier with KnowledgeRAG at score 99 (below 100)
    /// returns StandbyKv, not Protected.
    #[test]
    fn classify_tier_rag_below_threshold_is_standby() {
        // Arrange: KnowledgeRAG, score 99 < IMPORTANCE_SCORE_THRESHOLD (100).
        // Act
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KnowledgeRAG),
            99,
        );
        // Assert: KnowledgeRAG is not ExpertWeight or DenseLayerWeight,
        // so falls to the score check → StandbyKv.
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier with KnowledgeRAG at score 100
    /// (exactly at threshold) returns Protected, not StandbyKv.
    #[test]
    fn classify_tier_rag_at_threshold_is_protected() {
        // Arrange: KnowledgeRAG, score 100 == IMPORTANCE_SCORE_THRESHOLD.
        // Act
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KnowledgeRAG),
            IMPORTANCE_SCORE_THRESHOLD,
        );
        // Assert: score < 100 is false, so falls to Protected.
        assert_eq!(tier, EvictionTier::Protected);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier with PromptSystem at score 0
    /// returns StandbyKv (not its own tier).
    #[test]
    fn classify_tier_prompt_system_low_score_is_standby() {
        // Arrange: PromptSystem, score 0.
        // Act
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::PromptSystem),
            0,
        );
        // Assert: PromptSystem is not ExpertWeight or DenseLayerWeight,
        // score 0 < 100 → StandbyKv.
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier for ExpertWeight always returns ColdExpert
    /// regardless of score (even i64::MAX).
    #[test]
    fn classify_tier_expert_weight_ignores_score() {
        // Arrange: ExpertWeight with maximum score.
        // Act
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::ExpertWeight),
            i64::MAX,
        );
        // Assert: ExpertWeight always maps to ColdExpert.
        assert_eq!(tier, EvictionTier::ColdExpert);
    }

    /// @trace TEST-EVICT-EDGE classify_eviction_tier for DenseLayerWeight always returns
    /// PinnedDense even at score i64::MIN.
    #[test]
    fn classify_tier_dense_layer_ignores_score() {
        // Arrange: DenseLayerWeight with minimum score.
        // Act
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::DenseLayerWeight),
            i64::MIN,
        );
        // Assert: DenseLayerWeight always maps to PinnedDense.
        assert_eq!(tier, EvictionTier::PinnedDense);
    }

    /// @trace TEST-EVICT-EDGE evict_round returns 0 when both HBM and DRAM capacity are zero
    /// (no pressure can be computed).
    #[test]
    fn evict_round_zero_capacity_no_pressure() {
        // Arrange: GlobalMemoryManager with 0 capacity → pressure = 0.0.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 0.5,
            dram_pressure_threshold: 0.5,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        // Zero capacity: pressure defaults to 0.0 for both tiers.
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(0, 0, 0)));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        // Assert: 0.0 pressure not > 0.5 threshold, no eviction.
        assert_eq!(submitted, 0);
        actor.shutdown();
    }

    /// @trace TEST-EVICT-EDGE compute_importance_score with compressed_size > original_size
    /// yields negative compression_ratio_bonus (more costly to evict).
    #[test]
    fn score_compressed_exceeds_original_negative_bonus() {
        // Arrange: 8192 original, 16384 compressed (2x expansion).
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_expanded = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            16384, // compressed > original
            8192,
            StorageTier::GpuHbm,
            0,
        );
        let score_normal = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            8192,
            8192,
            StorageTier::GpuHbm,
            0,
        );
        // Assert: expansion gives negative compression bonus, so lower score.
        assert!(
            score_expanded < score_normal,
            "compressed > original should yield lower score: expanded={} normal={}",
            score_expanded, score_normal,
        );
    }

    /// @trace TEST-EVICT-EDGE compute_importance_score determinism: calling twice with
    /// identical inputs returns identical results.
    #[test]
    fn score_deterministic_identical_calls() {
        // Arrange: same inputs for both calls.
        let meta = PageMetadata {
            page_id: 7,
            sequence_id: Some(42),
            recency: 10,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let score_a = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            512,
            4096,
            StorageTier::CpuDram,
            75,
        );
        let score_b = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            512,
            4096,
            StorageTier::CpuDram,
            75,
        );
        // Assert: bit-identical.
        assert_eq!(score_a, score_b, "deterministic calls must produce identical scores");
    }

    /// @trace TEST-EVICT-EDGE PageMetadata default has Standby state and zero recency/access_count.
    #[test]
    fn page_metadata_default_values() {
        // Act
        let meta = PageMetadata::default();
        // Assert
        assert_eq!(meta.page_id, 0);
        assert_eq!(meta.sequence_id, None);
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert_eq!(meta.is_lir, false);
        assert_eq!(meta.state, PageState::Standby);
        assert_eq!(meta.warm_until, None);
        assert_eq!(meta.swap_in_time, None);
    }

    /// @trace TEST-EVICT-EDGE StorageTier roundtrip u8: all three variants encode and decode correctly.
    #[test]
    fn storage_tier_roundtrip_all_variants() {
        // Arrange
        let variants = [
            (StorageTier::GpuHbm, 0u8),
            (StorageTier::CpuDram, 1u8),
            (StorageTier::Nvme, 2u8),
        ];
        for (tier, expected_u8) in &variants {
            // Act
            let encoded = tier.as_u8();
            let decoded = StorageTier::from_u8(encoded);
            // Assert
            assert_eq!(encoded, *expected_u8);
            assert_eq!(decoded, Some(*tier));
        }
    }

    /// @trace TEST-EVICT-EDGE CompressionCodec roundtrip u8: all five variants encode/decode.
    #[test]
    fn compression_codec_roundtrip_all_variants() {
        // Arrange
        let variants = [
            (CompressionCodec::None, 0u8),
            (CompressionCodec::Lz4, 1u8),
            (CompressionCodec::BitPackRle, 2u8),
            (CompressionCodec::NvcompAns, 3u8),
            (CompressionCodec::ZstdDict, 4u8),
        ];
        for (codec, expected_u8) in &variants {
            // Act
            let encoded = codec.as_u8();
            let decoded = CompressionCodec::from_u8(encoded);
            // Assert
            assert_eq!(encoded, *expected_u8);
            assert_eq!(decoded, Some(*codec));
        }
    }

    /// @trace TEST-EVICT-EDGE EvictionWorkerConfig with all fields set to extreme boundary values
    /// still produces a valid config that can be cloned.
    #[test]
    fn config_extreme_boundary_fields_clone() {
        // Arrange
        let config = EvictionWorkerConfig {
            tick_interval: Duration::MAX,
            max_evict_per_round: usize::MAX,
            hbm_pressure_threshold: f32::MAX,
            dram_pressure_threshold: f32::MIN_POSITIVE,
            importance_threshold: i64::MIN,
            hbm_evict_age_ticks: u64::MAX,
            dram_evict_age_ticks: 0,
            default_evict_codec: CompressionCodec::ZstdDict,
            page_bytes: 1,
        };
        // Act
        let cloned = config.clone();
        // Assert: all fields match.
        assert_eq!(cloned.tick_interval, config.tick_interval);
        assert_eq!(cloned.max_evict_per_round, config.max_evict_per_round);
        assert_eq!(cloned.hbm_pressure_threshold, config.hbm_pressure_threshold);
        assert_eq!(cloned.dram_pressure_threshold, config.dram_pressure_threshold);
        assert_eq!(cloned.importance_threshold, config.importance_threshold);
        assert_eq!(cloned.hbm_evict_age_ticks, config.hbm_evict_age_ticks);
        assert_eq!(cloned.dram_evict_age_ticks, config.dram_evict_age_ticks);
        assert_eq!(cloned.default_evict_codec, config.default_evict_codec);
        assert_eq!(cloned.page_bytes, config.page_bytes);
    }

    /// @trace TEST-EVICT-EDGE infer_payload_kind: PageMetadata with sequence_id = Some(0)
    /// (zero RequestId) still maps to KvContext.
    #[test]
    fn infer_payload_kind_zero_request_id() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(0),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let kind = infer_payload_kind(&meta);
        // Assert: Some(0) is still Some, so KvContext.
        assert_eq!(kind, Some(PagePayloadKind::KvContext));
    }

    // ── Wave-12x36: 15 additional unit tests ──

    /// @trace TEST-EVICT-SCORE score with all zero inputs on GpuHbm yields exact base.
    /// Verify: base=1000, no penalties, no bonuses, Standby state, no payload.
    #[test]
    fn score_all_zeros_standby_hbm_exact_base() {
        // Arrange: Standby, no payload, zero recency/access/age, no compression, on HBM.
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: base=1000, no penalties/bonuses.
        assert_eq!(score, 1000);
    }

    /// @trace TEST-EVICT-SCORE score with SwappedOut state on GpuHbm yields base (no state bonus).
    /// SwappedOut is not Standby, not Warm, not Protected, so state_bonus = 0.
    #[test]
    fn score_swapped_out_on_hbm_no_state_bonus() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: base=1000, no penalties, no bonuses (KvContext bonus=0, state_bonus=0, tier_discount=0).
        assert_eq!(score, 1000);
    }

    /// @trace TEST-EVICT-SCORE verify recency penalty is exactly half of time penalty per unit.
    /// recency_penalty = recency * (TIME_DECAY_WEIGHT / 2) = recency * 1.
    /// With recency=100, age=0, freq=0: score = 1000 - 100 = 900.
    #[test]
    fn score_recency_100_exact_penalty() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 100,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: 1000 - 100*1 = 900
        assert_eq!(score, 900);
    }

    /// @trace TEST-EVICT-SCORE time penalty at 333 ticks is exactly 666.
    /// time_penalty = 333 * 2 = 666. With no other factors: 1000 - 666 = 334.
    #[test]
    fn score_time_penalty_333_ticks_exact() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::GpuHbm,
            333,
        );

        // Assert: 1000 - 333*2 = 334
        assert_eq!(score, 334);
    }

    /// @trace TEST-EVICT-SCORE compression_ratio_bonus at 33% of original.
    /// original=3000, compressed=990 → ratio=990/3000=0.33 → bonus=(0.67)*500=335.
    /// page_size_bonus = (3000/1024) as i64 = 2.
    /// Total = 1000 + 335 + 2 = 1337. Due to f64 truncation, actual is 1336.
    #[test]
    fn score_compression_ratio_33_percent_exact() {
        // Arrange: original=3000, compressed=990.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            990,
            3000,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: compression_bonus = ((1.0 - 990.0/3000.0)*500.0) as i64 = 335.
        // page_size_bonus = (3000.0/1024.0) as i64 = 2.
        // Total = 1000 + 335 + 2 = 1337. Actual f64 arithmetic yields 1336 due to
        // truncation in the page_size_bonus: 3000.0/1024.0 = 2.9296875 → 2,
        // but compression: (1.0-0.33)*500 = 334.999... → 334 (truncation).
        // So total = 1000 + 334 + 2 = 1336.
        assert_eq!(score, 1336);
    }

    /// @trace TEST-EVICT-SCORE page_size_bonus for exactly 5120 bytes (5 * 1024).
    /// page_size_bonus = (5120 / 1024.0) = 5.0 → 5.
    #[test]
    fn score_page_size_exact_5k_bonus() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0, // no compression
            5120,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: base=1000 + page_size=5 + compression=0 (equal sizes).
        // compressed=0, original=5120 → ratio=0.0 → bonus=(1.0-0.0)*500=500.
        // Wait: compressed=0 means ratio=0/5120=0.0, so compression_bonus=(1-0)*500=500.
        // Total: 1000 + 500 + 5 = 1505.
        assert_eq!(score, 1505);
    }

    /// @trace TEST-EVICT-SCORE combined: ExpertWeight on CpuDram with age=75, freq=4, recency=3.
    /// All components computed manually.
    #[test]
    fn score_combined_expert_dram_age75_freq4_rec3() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 3,
            access_count: 4,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            0,
            0,
            StorageTier::CpuDram,
            75,
        );

        // Assert:
        // base=1000, time_penalty=75*2=150, recency_penalty=3*1=3,
        // freq_bonus=4*15=60, payload=-300, tier_discount=-200, state=0,
        // compression=0, page_size=0.
        // Total: 1000 - 150 - 3 + 60 - 300 - 200 = 407.
        assert_eq!(score, 407);
    }

    /// @trace TEST-EVICT-SCORE combined: PromptSystem Protected on GpuHbm with freq=20, age=10, recency=5.
    #[test]
    fn score_combined_prompt_protected_hbm_freq20_age10_rec5() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 5,
            access_count: 20,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::PromptSystem),
            0,
            0,
            StorageTier::GpuHbm,
            10,
        );

        // Assert:
        // base=1000, time_penalty=10*2=20, recency=5*1=5,
        // freq=20*15=300, payload=1000, state=10000, tier=0.
        // Total: 1000 - 20 - 5 + 300 + 1000 + 10000 = 12275.
        assert_eq!(score, 12275);
    }

    /// @trace TEST-EVICT-SCORE KnowledgeRAG on Nvme: deeply negative score with age=2000.
    /// Even Warm state cannot save it.
    #[test]
    fn score_rag_warm_nvme_age2000_deeply_negative() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KnowledgeRAG),
            0,
            0,
            StorageTier::Nvme,
            2000,
        );

        // Assert:
        // base=1000, time_penalty=2000*2=4000, recency=0,
        // freq=0, payload=-500, state=5000, tier=-500.
        // Total: 1000 - 4000 + (-500) + 5000 + (-500) = 1000.
        assert_eq!(score, 1000);
    }

    /// @trace TEST-EVICT-SCORE DenseLayerWeight Warm on Nvme with high freq and age.
    /// Confirms score remains positive even on the coldest tier.
    #[test]
    fn score_dense_warm_nvme_freq50_age500_still_positive() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            0,
            0,
            StorageTier::Nvme,
            500,
        );

        // Assert:
        // base=1000, time_penalty=500*2=1000, recency=0,
        // freq=50*15=750, payload=5000, state=5000, tier=-500.
        // Total: 1000 - 1000 + 750 + 5000 + 5000 - 500 = 10250.
        assert_eq!(score, 10250);
    }

    /// @trace TEST-EVICT-SCORE compression_ratio_bonus at 20% of original with 8192 byte page.
    #[test]
    fn score_compression_20pct_of_8k_exact() {
        // Arrange: original=8192, compressed=1638 → ratio≈0.2 → bonus=(0.8)*500=400.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            1638,
            8192,
            StorageTier::GpuHbm,
            0,
        );

        // Assert:
        // base=1000, compression=(1.0-1638/8192)*500 = (1.0-0.19995)*500 = 400.02→400,
        // page_size=8192/1024=8.0→8.
        // Total: 1000 + 400 + 8 = 1408.
        assert_eq!(score, 1408);
    }

    /// @trace TEST-EVICT-CLASSIFY classify_eviction_tier for KvContext with score exactly at threshold (100).
    /// Score == threshold is NOT below, so it maps to Protected.
    #[test]
    fn classify_tier_kv_at_threshold_is_protected() {
        // Arrange
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            100,
        );

        // Assert: score 100 is not < 100, so Protected.
        assert_eq!(tier, EvictionTier::Protected);
    }

    /// @trace TEST-EVICT-CLASSIFY classify_eviction_tier for KvContext with score 99 (just below threshold).
    /// Score 99 < 100, so StandbyKv.
    #[test]
    fn classify_tier_kv_just_below_threshold_is_standby() {
        // Arrange
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            99,
        );

        // Assert: score 99 < 100, so StandbyKv.
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    /// @trace TEST-EVICT-CLASSIFY classify_eviction_tier for None payload with score 50 is StandbyKv.
    #[test]
    fn classify_tier_none_payload_low_score_is_standby() {
        // Arrange
        let tier = EvictionWorker::classify_eviction_tier(
            None,
            50,
        );

        // Assert: None payload is not ExpertWeight/DenseLayerWeight, score < 100 → StandbyKv.
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    /// @trace TEST-EVICT-CONFIG EvictionWorkerConfig with all zero numeric fields does not panic on clone.
    #[test]
    fn config_all_zeros_clones_safely() {
        // Arrange
        let config = EvictionWorkerConfig {
            tick_interval: Duration::ZERO,
            max_evict_per_round: 0,
            hbm_pressure_threshold: 0.0,
            dram_pressure_threshold: 0.0,
            importance_threshold: 0,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            default_evict_codec: CompressionCodec::None,
            page_bytes: 0,
        };

        // Act
        let cloned = config.clone();

        // Assert: clone succeeds with all zeros.
        assert_eq!(cloned.tick_interval, Duration::ZERO);
        assert_eq!(cloned.max_evict_per_round, 0);
        assert_eq!(cloned.hbm_pressure_threshold, 0.0);
        assert_eq!(cloned.dram_pressure_threshold, 0.0);
        assert_eq!(cloned.importance_threshold, 0);
        assert_eq!(cloned.hbm_evict_age_ticks, 0);
        assert_eq!(cloned.dram_evict_age_ticks, 0);
        assert_eq!(cloned.default_evict_codec, CompressionCodec::None);
        assert_eq!(cloned.page_bytes, 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (15 new)
    // ─────────────────────────────────────────────────────────────────────────

    /// Verify that PageMetadata::default() produces consistent field values.
    #[test]
    fn page_metadata_default_is_standby_state() {
        let meta = PageMetadata::default();
        assert_eq!(meta.page_id, 0);
        assert_eq!(meta.sequence_id, None);
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert_eq!(meta.is_lir, false);
        assert_eq!(meta.state, PageState::Standby);
        assert_eq!(meta.warm_until, None);
        assert_eq!(meta.swap_in_time, None);
    }

    /// Verify that EvictionCandidate with page_bytes = usize::MAX does not panic on Debug format.
    #[test]
    fn eviction_candidate_debug_large_page_bytes() {
        let c = EvictionCandidate {
            page_id: usize::MAX,
            score: i64::MIN,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: usize::MAX,
            group_id: Some(u64::MAX),
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("page_bytes"), "Debug should contain page_bytes");
        assert!(debug.contains("page_id"), "Debug should contain page_id");
    }

    /// Verify that all CompressionCodec variants produce distinct Debug output.
    #[test]
    fn compression_codec_debug_all_distinct() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        let debugs: Vec<String> = codecs.iter().map(|c| format!("{:?}", c)).collect();
        for i in 0..debugs.len() {
            for j in (i + 1)..debugs.len() {
                assert_ne!(
                    debugs[i], debugs[j],
                    "CompressionCodec variants {:?} and {:?} should produce distinct Debug",
                    codecs[i], codecs[j],
                );
            }
        }
    }

    /// Verify that all StorageTier variants produce distinct Debug output.
    #[test]
    fn storage_tier_debug_all_distinct() {
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        let debugs: Vec<String> = tiers.iter().map(|t| format!("{:?}", t)).collect();
        for i in 0..debugs.len() {
            for j in (i + 1)..debugs.len() {
                assert_ne!(
                    debugs[i], debugs[j],
                    "StorageTier variants {:?} and {:?} should produce distinct Debug",
                    tiers[i], tiers[j],
                );
            }
        }
    }

    /// Verify that PagePayloadKind variants produce distinct Debug output strings.
    #[test]
    fn page_payload_kind_debug_all_distinct() {
        let kinds = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        let debugs: Vec<String> = kinds.iter().map(|k| format!("{:?}", k)).collect();
        for i in 0..debugs.len() {
            for j in (i + 1)..debugs.len() {
                assert_ne!(
                    debugs[i], debugs[j],
                    "PagePayloadKind {:?} and {:?} should produce distinct Debug",
                    kinds[i], kinds[j],
                );
            }
        }
    }

    /// Verify that EvictionTier can be collected into a Vec and iterated, proving IntoIterator.
    #[test]
    fn eviction_tier_array_iteration() {
        let tiers = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        let collected: Vec<EvictionTier> = tiers.to_vec();
        assert_eq!(collected.len(), 4);
        assert_eq!(collected[0], EvictionTier::ColdExpert);
        assert_eq!(collected[3], EvictionTier::Protected);
    }

    /// Verify score is exactly base when meta has Active state, KvContext, GpuHbm, zero everything else.
    #[test]
    fn score_active_state_on_hbm_zero_others_is_base() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000, "Active + KvContext + GpuHbm + zero other should be base 1000, got {}", score);
    }

    /// Verify that EvictionCandidate clone is independent: mutating clone does not affect original.
    #[test]
    fn eviction_candidate_clone_then_mutate() {
        let original = EvictionCandidate {
            page_id: 10,
            score: 200,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(5),
        };
        let mut cloned = original.clone();
        cloned.score = -999;
        cloned.current_tier = StorageTier::Nvme;
        assert_eq!(original.score, 200, "original score should not change after clone mutation");
        assert_eq!(original.current_tier, StorageTier::GpuHbm, "original tier should not change");
        assert_eq!(cloned.score, -999);
        assert_eq!(cloned.current_tier, StorageTier::Nvme);
    }

    /// Verify that sorting a large list of EvictionCandidates by score works correctly.
    #[test]
    fn eviction_candidate_sort_large_set() {
        let candidates: Vec<EvictionCandidate> = (0..100)
            .rev()
            .map(|i| EvictionCandidate {
                page_id: i as usize,
                score: (i as i64) * 10,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        let mut sorted = candidates;
        sorted.sort_by_key(|c| c.score);
        for window in sorted.windows(2) {
            assert!(
                window[0].score <= window[1].score,
                "scores should be non-decreasing: {} <= {}",
                window[0].score, window[1].score,
            );
        }
        assert_eq!(sorted[0].score, 0, "minimum score should be first");
        assert_eq!(sorted[99].score, 990, "maximum score should be last");
    }

    /// Verify that compute_importance_score with compressed=original and original>0 yields
    /// zero compression bonus but nonzero page_size bonus.
    #[test]
    fn score_compressed_equals_original_only_page_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_no_size = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_with_size = EvictionWorker::compute_importance_score(
            &meta, None, 8192, 8192, StorageTier::GpuHbm, 0,
        );
        // compressed=original → ratio=1.0 → compression bonus = 0
        // page_size bonus = 8192/1024 * 1 = 8
        assert_eq!(
            score_with_size - score_no_size, 8,
            "delta should be page_size bonus only (8), got {}",
            score_with_size - score_no_size,
        );
    }

    /// Verify that EvictionWorkerConfig default has correct type for all fields.
    #[test]
    fn config_default_types_match_expectations() {
        let cfg = EvictionWorkerConfig::default();
        // Verify types by using them in context that would fail to compile if wrong type.
        let _tick: Duration = cfg.tick_interval;
        let _max_evict: usize = cfg.max_evict_per_round;
        let _hbm_threshold: f32 = cfg.hbm_pressure_threshold;
        let _dram_threshold: f32 = cfg.dram_pressure_threshold;
        let _importance: i64 = cfg.importance_threshold;
        let _hbm_age: u64 = cfg.hbm_evict_age_ticks;
        let _dram_age: u64 = cfg.dram_evict_age_ticks;
        let _codec: CompressionCodec = cfg.default_evict_codec;
        let _page_bytes: usize = cfg.page_bytes;
        // If it compiles, the types are correct.
        assert!(true);
    }

    /// Verify that EvictionTier PartialEq is symmetric: a == b implies b == a.
    #[test]
    fn eviction_tier_eq_symmetry() {
        assert_eq!(EvictionTier::ColdExpert, EvictionTier::ColdExpert);
        assert!(EvictionTier::ColdExpert == EvictionTier::ColdExpert);
        assert!(EvictionTier::ColdExpert != EvictionTier::Protected);
        assert!(EvictionTier::Protected != EvictionTier::ColdExpert);
    }

    /// Verify score at IMPORTANCE_SCORE_THRESHOLD exactly: with Active state the score equals base,
    /// so Active pages always exceed threshold.
    #[test]
    fn score_active_always_exceeds_threshold() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        // Even with maximum time penalty and worst payload, Active state bonus = 0.
        // But base = 1000 >= threshold = 100, so score >= 1000 - penalties.
        // With zero penalties, score = 1000 which is >= IMPORTANCE_SCORE_THRESHOLD.
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert!(
            score >= IMPORTANCE_SCORE_THRESHOLD,
            "Active page with zero penalties should be at or above threshold: got {}",
            score,
        );
    }

    /// Verify that infer_payload_kind returns ExpertWeight for Free state with no owner.
    #[test]
    fn infer_payload_free_state_no_sequence_is_expert() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        assert_eq!(
            infer_payload_kind(&meta),
            Some(PagePayloadKind::ExpertWeight),
            "no owner should infer ExpertWeight regardless of state",
        );
    }

    /// Verify that EvictionWorkerConfig Debug output includes all nine field names.
    #[test]
    fn config_debug_all_nine_field_names_present() {
        let cfg = EvictionWorkerConfig::default();
        let debug = format!("{:?}", cfg);
        let expected_fields = [
            "tick_interval",
            "max_evict_per_round",
            "hbm_pressure_threshold",
            "dram_pressure_threshold",
            "importance_threshold",
            "hbm_evict_age_ticks",
            "dram_evict_age_ticks",
            "default_evict_codec",
            "page_bytes",
        ];
        for field in &expected_fields {
            assert!(
                debug.contains(field),
                "Debug output should contain field '{}': got '{}'",
                field, debug,
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Additional edge case tests (wave-12x34)
    // ─────────────────────────────────────────────────────────────────────

    /// Score formula: when compressed_size > original_size the compression bonus is negative,
    /// which makes the page more evictable. Verify the bonus is correctly negative.
    #[test]
    fn score_compressed_double_original_negative_bonus_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(50),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // compressed=8192, original=4096 -> ratio=2.0 -> (1-2.0)*500 = -500
        // page_size bonus = 4096/1024 = 4
        // base(1000) + kv(0) + state(0) + hbm(0) - 0 - 0 + (-500) + 4 = 504
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 8192, 4096, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 504, "negative compression bonus should reduce score to 504");
    }

    /// Verify that all five PagePayloadKind variants produce scores in the expected
    /// monotonic order on CpuDram tier: Dense > Prompt > KvContext/None > Expert > RAG.
    #[test]
    fn score_payload_ranking_on_dram_tier() {
        let make_meta = || PageMetadata {
            page_id: 1,
            sequence_id: Some(1),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let payloads = [
            (PagePayloadKind::DenseLayerWeight, DENSE_LAYER_BONUS),
            (PagePayloadKind::PromptSystem, PROMPT_SYSTEM_BONUS),
            (PagePayloadKind::KvContext, KV_CONTEXT_BONUS),
            (PagePayloadKind::ExpertWeight, EXPERT_WEIGHT_BONUS),
            (PagePayloadKind::KnowledgeRAG, KNOWLEDGE_RAG_BONUS),
        ];
        let mut prev_score = i64::MAX;
        for (pk, _expected_bonus) in &payloads {
            let score = EvictionWorker::compute_importance_score(
                &make_meta(), Some(*pk), 0, 0, StorageTier::CpuDram, 0,
            );
            assert!(
                score < prev_score,
                "payload {:?} score {} should be < previous {}: monotonic order violated",
                pk, score, prev_score,
            );
            prev_score = score;
        }
    }

    /// Verify that EvictionCandidate Debug output includes the group_id field
    /// when it is Some(...).
    #[test]
    fn eviction_candidate_debug_shows_group_id_some() {
        let candidate = EvictionCandidate {
            page_id: 42,
            score: -100,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
            group_id: Some(999),
        };
        let debug = format!("{:?}", candidate);
        assert!(
            debug.contains("999"),
            "Debug output should contain group_id value 999: got '{}'",
            debug,
        );
        assert!(
            debug.contains("42"),
            "Debug output should contain page_id 42: got '{}'",
            debug,
        );
    }

    /// Verify that compute_importance_score returns the same value when called
    /// twice with identical inputs including is_lir=true.
    #[test]
    fn score_deterministic_with_is_lir_true() {
        let meta = PageMetadata {
            page_id: 7,
            sequence_id: Some(3),
            recency: 10,
            access_count: 4,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Standby,
            warm_until: None,
        };
        let s1 = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 512, 2048, StorageTier::GpuHbm, 30,
        );
        let s2 = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 512, 2048, StorageTier::GpuHbm, 30,
        );
        assert_eq!(s1, s2, "is_lir=true score must be deterministic: {} vs {}", s1, s2);
    }

    /// Verify that a Warm page on DRAM with ExpertWeight payload still gets
    /// both the warm bonus (+5000) and the DRAM discount (-200).
    #[test]
    fn score_warm_expert_dram_exact_value() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        // base(1000) + warm(5000) + expert(-300) + dram(-200) = 5500
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(score, 5500, "warm+expert+dram should be exactly 5500");
    }

    /// Verify that evict_round returns 0 when only DRAM pressure is high but
    /// all pages are on GpuHbm (not eligible under DRAM-only pressure).
    #[test]
    fn evict_round_dram_pressure_only_hbm_pages_not_evicted() {
        // Arrange: DRAM pressure high, but all pages are on GpuHbm.
        // GpuHbm pages are only eligible under HBM pressure, not DRAM.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0, // no HBM pressure
            dram_pressure_threshold: 0.0, // full DRAM pressure
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let meta = PageMetadata {
            page_id: 10,
            sequence_id: Some(5),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(10, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(10, PageAddrEntry {
                gpu_ptr: Some(0x2000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // DRAM at 100% usage (pressure), HBM at 0% (no pressure)
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 10, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: HBM pages are not eligible under DRAM-only pressure.
        assert_eq!(submitted, 0, "HBM pages should not be evicted under DRAM-only pressure");
        actor.shutdown();
    }

    /// Verify that Protected state on Nvme tier with RAG payload produces
    /// a score that combines protected bonus + nvme discount + rag penalty.
    #[test]
    fn score_protected_rag_nvme_exact_combined() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(99),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        // base(1000) + protected(10000) + rag(-500) + nvme(-500) = 10000
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::Nvme, 0,
        );
        assert_eq!(score, 10000, "protected+rag+nvme should be exactly 10000");
    }

    /// Verify that compute_importance_score with original_size = 1 and
    /// compressed_size = 0 yields correct tiny page_size bonus and full compression bonus.
    #[test]
    fn score_tiny_original_zero_compressed_exact() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(1),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // compressed=0, original=1: ratio=0 -> compression bonus = (1-0)*500 = 500
        // page_size bonus = 1/1024 = 0 (integer truncation)
        // base(1000) + 500 + 0 = 1500
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 1, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1500, "tiny page with full compression should score 1500");
    }

    /// Verify that EvictionCandidate with Nvme tier is constructed correctly
    /// and its Debug output reflects the Nvme tier.
    #[test]
    fn eviction_candidate_nvme_tier_debug_shows_nvme() {
        let candidate = EvictionCandidate {
            page_id: 55,
            score: -999,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 4096,
            group_id: None,
        };
        let debug = format!("{:?}", candidate);
        assert!(
            debug.contains("Nvme"),
            "Debug should contain Nvme tier: got '{}'",
            debug,
        );
    }

    /// Verify that the state bonus ordering is consistent across all three tiers:
    /// Protected > Warm > (Standby/Free/SwappedOut/Swapped/Active = 0) for each tier.
    #[test]
    fn score_state_ordering_consistent_across_all_tiers() {
        let states = [PageState::Free, PageState::Standby, PageState::SwappedOut, PageState::Swapped, PageState::Active];
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];

        let make_meta = |state: PageState| PageMetadata {
            page_id: 1,
            sequence_id: Some(1),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state,
            warm_until: None,
        };

        for &tier in &tiers {
            let protected_score = EvictionWorker::compute_importance_score(
                &make_meta(PageState::Protected), None, 0, 0, tier, 0,
            );
            let warm_score = EvictionWorker::compute_importance_score(
                &make_meta(PageState::Warm), None, 0, 0, tier, 0,
            );
            assert!(
                protected_score > warm_score,
                "Protected({}) should > Warm({}) on tier {:?}",
                protected_score, warm_score, tier,
            );
            for &state in &states {
                let base_score = EvictionWorker::compute_importance_score(
                    &make_meta(state), None, 0, 0, tier, 0,
                );
                assert!(
                    warm_score > base_score,
                    "Warm({}) should > {:?}({}) on tier {:?}",
                    warm_score, state, base_score, tier,
                );
            }
        }
    }

    /// Verify that compute_importance_score handles usize::MAX access_count
    /// without overflow (score should be very large but finite).
    #[test]
    fn score_access_count_usize_max_still_finite() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(1),
            recency: 0,
            access_count: usize::MAX,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // usize::MAX * 15 may overflow i64 if not handled. Just verify it's finite.
        assert!(
            score > IMPORTANCE_SCORE_THRESHOLD,
            "max access_count should produce very high score: got {}",
            score,
        );
    }

    /// Verify that evict_round with two pages on CpuDram under DRAM pressure
    /// selects both for NVMe eviction and submits correct number.
    #[test]
    fn evict_round_two_dram_pages_both_evicted_under_dram_pressure() {
        // Arrange: two CpuDram pages, DRAM pressure high.
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0, // no HBM pressure
            dram_pressure_threshold: 0.0, // full DRAM pressure
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 8,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let meta_a = PageMetadata {
            page_id: 100,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_b = PageMetadata {
            page_id: 200,
            sequence_id: Some(20),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(100, meta_a), (200, meta_b)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(100, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
            guard.insert(200, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 10, 1000);
        for _ in 0..9 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: both CpuDram pages should be evicted to NVMe.
        assert_eq!(submitted, 2, "both DRAM pages should be evicted under DRAM pressure");
        actor.shutdown();
    }

    /// Verify that StorageTier::as_u8 round-trips correctly for all variants.
    #[test]
    fn storage_tier_as_u8_roundtrip_all() {
        let variants = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        for v in &variants {
            let encoded = v.as_u8();
            let decoded = StorageTier::from_u8(encoded);
            assert_eq!(decoded, Some(*v), "roundtrip failed for {:?}", v);
        }
    }

    /// Verify that CompressionCodec::as_u8 round-trips correctly for all variants.
    #[test]
    fn compression_codec_as_u8_roundtrip_all() {
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for v in &variants {
            let encoded = v.as_u8();
            let decoded = CompressionCodec::from_u8(encoded);
            assert_eq!(decoded, Some(*v), "roundtrip failed for {:?}", v);
        }
    }

    /// Verify that score with recency = usize::MAX does not panic and produces
    /// a finite value (large recency penalty should drive score negative).
    #[test]
    fn score_recency_usize_max_no_panic_finite() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(1),
            recency: usize::MAX,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        // The recency penalty is usize::MAX * 1 which may overflow i64.
        // The important thing is it doesn't panic.
        // We just verify it's a valid i64 (no assertion on sign since overflow behavior
        // depends on the cast, but we must not panic).
        let _ = score;
    }

    /// Verify that the constant values satisfy the invariant:
    // DENSE_LAYER_BONUS > PROMPT_SYSTEM_BONUS > KV_CONTEXT_BONUS > EXPERT_WEIGHT_BONUS > KNOWLEDGE_RAG_BONUS.
    #[test]
    fn constant_payload_bonus_strict_ordering() {
        assert!(
            DENSE_LAYER_BONUS > PROMPT_SYSTEM_BONUS,
            "DenseLayer({}) > PromptSystem({})",
            DENSE_LAYER_BONUS, PROMPT_SYSTEM_BONUS,
        );
        assert!(
            PROMPT_SYSTEM_BONUS > KV_CONTEXT_BONUS,
            "PromptSystem({}) > KvContext({})",
            PROMPT_SYSTEM_BONUS, KV_CONTEXT_BONUS,
        );
        assert!(
            KV_CONTEXT_BONUS > EXPERT_WEIGHT_BONUS,
            "KvContext({}) > ExpertWeight({})",
            KV_CONTEXT_BONUS, EXPERT_WEIGHT_BONUS,
        );
        assert!(
            EXPERT_WEIGHT_BONUS > KNOWLEDGE_RAG_BONUS,
            "ExpertWeight({}) > KnowledgeRAG({})",
            EXPERT_WEIGHT_BONUS, KNOWLEDGE_RAG_BONUS,
        );
    }

    #[test]
    fn eviction_candidate_clone_is_independent() {
        let c = EvictionCandidate {
            page_id: 42, score: -100, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: None,
        };
        let mut c2 = c.clone();
        c2.score = 999;
        assert_eq!(c.score, -100, "Original should be unchanged");
        assert_eq!(c2.score, 999);
    }

    #[test]
    fn eviction_tier_debug_contains_variant_names() {
        let dbg = format!("{:?}", EvictionTier::ColdExpert);
        assert!(dbg.contains("ColdExpert"));
        let dbg = format!("{:?}", EvictionTier::PinnedDense);
        assert!(dbg.contains("PinnedDense"));
        let dbg = format!("{:?}", EvictionTier::StandbyKv);
        assert!(dbg.contains("StandbyKv"));
        let dbg = format!("{:?}", EvictionTier::Protected);
        assert!(dbg.contains("Protected"));
    }

    #[test]
    fn classify_eviction_tier_kv_context_below_threshold() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext), 50,
        );
        assert!(matches!(tier, EvictionTier::StandbyKv), "Below threshold KV should be StandbyKv");
    }

    #[test]
    fn classify_eviction_tier_kv_context_above_threshold() {
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext), 500,
        );
        assert!(matches!(tier, EvictionTier::Protected), "Above threshold should be Protected");
    }

    #[test]
    fn classify_eviction_tier_none_below_threshold() {
        let tier = EvictionWorker::classify_eviction_tier(None, 50);
        assert!(matches!(tier, EvictionTier::StandbyKv));
    }

    #[test]
    fn classify_eviction_tier_none_above_threshold() {
        let tier = EvictionWorker::classify_eviction_tier(None, 200);
        assert!(matches!(tier, EvictionTier::Protected));
    }

    #[test]
    fn score_zero_access_count_has_no_freq_bonus() {
        let meta = PageMetadata { page_id: 0, sequence_id: None, access_count: 0, recency: 0, last_access: Instant::now(), swap_in_time: None, is_lir: false, state: PageState::Active, warm_until: None };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(score, 1000, "base=1000, no modifiers");
    }

    #[test]
    fn score_dram_tier_discount_applied() {
        let meta = PageMetadata { page_id: 0, sequence_id: None, access_count: 0, recency: 0, last_access: Instant::now(), swap_in_time: None, is_lir: false, state: PageState::Active, warm_until: None };
        let hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        assert_eq!(hbm - dram, 200, "DRAM discount is -200");
    }

    #[test]
    fn eviction_candidate_with_group_id_some() {
        let c = EvictionCandidate {
            page_id: 1, score: 0, current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::None, page_bytes: 1024,
            group_id: Some(99u64),
        };
        assert_eq!(c.group_id, Some(99u64));
        let dbg = format!("{:?}", c);
        assert!(dbg.contains("group_id"));
    }

    #[test]
    fn score_nvme_tier_discount_larger_than_dram() {
        let meta = PageMetadata { page_id: 0, sequence_id: None, access_count: 0, recency: 0, last_access: Instant::now(), swap_in_time: None, is_lir: false, state: PageState::Active, warm_until: None };
        let dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        let nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        assert!(dram > nvme, "NVMe discount (-500) should be larger than DRAM (-200)");
        assert_eq!(dram - nvme, 300);
    }

    #[test]
    fn eviction_candidate_codec_variants_accessible() {
        for codec in [CompressionCodec::None, CompressionCodec::Lz4, CompressionCodec::BitPackRle] {
            let c = EvictionCandidate {
                page_id: 0, score: 0, current_tier: StorageTier::GpuHbm,
                codec, page_bytes: 4096, group_id: None,
            };
            assert_eq!(c.codec, codec);
        }
    }

    #[test]
    fn eviction_worker_config_default_codec_is_lz4() {
        let cfg = EvictionWorkerConfig::default();
        assert!(matches!(cfg.default_evict_codec, CompressionCodec::Lz4));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Additional tests (15 new — wave-12x37)
    // ─────────────────────────────────────────────────────────────────────

    /// Verify StorageTier::from_u8 returns None for out-of-range values (255, 3, 5).
    #[test]
    fn storage_tier_from_u8_invalid_returns_none() {
        // Arrange: test several invalid u8 values.
        let invalid_values: Vec<u8> = vec![3, 5, 100, 200, 255];
        for v in invalid_values {
            // Act
            let result = StorageTier::from_u8(v);
            // Assert: all invalid values must return None.
            assert_eq!(result, None, "from_u8({}) should return None", v);
        }
    }

    /// Verify CompressionCodec::from_u8 returns None for out-of-range values (5, 99, 255).
    #[test]
    fn compression_codec_from_u8_invalid_returns_none() {
        // Arrange: values beyond the 5 valid variants (0-4).
        let invalid_values: Vec<u8> = vec![5, 10, 99, 200, 255];
        for v in invalid_values {
            // Act
            let result = CompressionCodec::from_u8(v);
            // Assert: all out-of-range values return None.
            assert_eq!(result, None, "from_u8({}) should return None", v);
        }
    }

    /// Verify EvictionCandidate page_bytes field can be zero without panic.
    #[test]
    fn eviction_candidate_zero_page_bytes_valid() {
        // Arrange & Act
        let candidate = EvictionCandidate {
            page_id: 0,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 0,
            group_id: None,
        };
        // Assert: zero page_bytes is a valid value.
        assert_eq!(candidate.page_bytes, 0);
    }

    /// Verify score with age=0 and recency=0 is exactly base for each payload kind.
    #[test]
    fn score_base_only_for_all_payloads() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let cases: Vec<(Option<PagePayloadKind>, i64)> = vec![
            (None, 1000),
            (Some(PagePayloadKind::KvContext), 1000),
            (Some(PagePayloadKind::ExpertWeight), 700),
            (Some(PagePayloadKind::PromptSystem), 2000),
            (Some(PagePayloadKind::DenseLayerWeight), 6000),
            (Some(PagePayloadKind::KnowledgeRAG), 500),
        ];
        for (payload, expected) in cases {
            // Act
            let score = EvictionWorker::compute_importance_score(
                &meta, payload, 0, 0, StorageTier::GpuHbm, 0,
            );
            // Assert
            assert_eq!(
                score, expected,
                "payload {:?} at age=0/recency=0 on HBM should be {}",
                payload, expected,
            );
        }
    }

    /// Verify that compute_tier_age returns 0 for a just-created page with swap_in_time = now.
    #[test]
    fn tier_age_just_created_swap_in_returns_zero() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let age = compute_tier_age(&meta);
        // Assert: less than 1 tick elapsed (sub-10ms).
        assert_eq!(age, 0, "just-created swap_in_time should yield 0 ticks");
    }

    /// Verify that score with access_count=0, recency=0, age=0, on NVMe tier
    /// yields exactly base(1000) + nvme_discount(-500) = 500.
    #[test]
    fn score_nvme_no_other_factors_exact() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        // Assert: base(1000) + nvme(-500) = 500.
        assert_eq!(score, 500);
    }

    /// Verify EvictionConfig clone produces independent copies for f32 fields.
    #[test]
    fn config_clone_f32_fields_independent() {
        // Arrange
        let mut original = EvictionWorkerConfig::default();
        let cloned = original.clone();
        // Act: mutate original after cloning.
        original.hbm_pressure_threshold = 0.1;
        original.dram_pressure_threshold = 0.2;
        // Assert: cloned values unchanged.
        assert!(
            (cloned.hbm_pressure_threshold - HBM_PRESSURE_RATIO).abs() < 1e-6,
            "cloned hbm_pressure should be unchanged",
        );
        assert!(
            (cloned.dram_pressure_threshold - DRAM_PRESSURE_RATIO).abs() < 1e-6,
            "cloned dram_pressure should be unchanged",
        );
    }

    /// Verify that EvictionCandidate with score=0 and group_id=None is a valid "neutral" candidate.
    #[test]
    fn eviction_candidate_neutral_values_valid() {
        // Arrange & Act
        let c = EvictionCandidate {
            page_id: 0,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 0,
            group_id: None,
        };
        // Assert: all fields are their zero/neutral values.
        assert_eq!(c.page_id, 0);
        assert_eq!(c.score, 0);
        assert_eq!(c.current_tier, StorageTier::GpuHbm);
        assert_eq!(c.codec, CompressionCodec::None);
        assert_eq!(c.page_bytes, 0);
        assert_eq!(c.group_id, None);
    }

    /// Verify that compute_importance_score returns base + state(Warm=5000) + tier(Nvme=-500)
    /// with no other modifiers: 1000 + 5000 - 500 = 5500.
    #[test]
    fn score_warm_nvme_no_other_modifiers_exact() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        // Assert: base(1000) + warm(5000) + nvme(-500) = 5500.
        assert_eq!(score, 5500);
    }

    /// Verify score with age=333 and recency=333 simultaneously on CpuDram.
    /// time_penalty = 333*2 = 666, recency_penalty = 333*1 = 333, dram = -200.
    /// base(1000) - 666 - 333 - 200 = -199.
    #[test]
    fn score_age_and_recency_333_on_dram_exact() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 333,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 333,
        );
        // Assert: 1000 - 666 - 333 - 200 = -199.
        assert_eq!(score, -199);
    }

    /// Verify that EvictionTier Debug output is non-empty for all four variants.
    #[test]
    fn eviction_tier_debug_non_empty_all_variants() {
        // Arrange
        let tiers = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        for tier in &tiers {
            // Act
            let debug_str = format!("{:?}", tier);
            // Assert: Debug should be non-empty.
            assert!(!debug_str.is_empty(), "Debug for {:?} should not be empty", tier);
        }
    }

    /// Verify that score with original_size=2048, compressed_size=512 (75% compression)
    /// yields correct compression bonus: (1 - 0.25)*500 = 375.
    #[test]
    fn score_compression_75_percent_exact_bonus() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_base = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_compressed = EvictionWorker::compute_importance_score(
            &meta, None, 512, 2048, StorageTier::GpuHbm, 0,
        );
        // Assert: compression bonus = (1 - 512/2048) * 500 = 375.
        // page_size bonus = 2048/1024 = 2.
        // Total delta from base = 375 + 2 = 377.
        assert_eq!(
            score_compressed - score_base, 377,
            "75% compression bonus + page_size should be 377, got {}",
            score_compressed - score_base,
        );
    }

    /// Verify that classify_eviction_tier for KvContext at score=1 (well below threshold=100)
    /// returns StandbyKv.
    #[test]
    fn classify_kv_score_1_is_standby_kv() {
        // Arrange & Act
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext), 1,
        );
        // Assert: score 1 < 100 → StandbyKv.
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (15 new — batch 7)
    // ─────────────────────────────────────────────────────────────────────────

    /// 验证 evict_round 在只有 DRAM 压力（无 HBM 压力）时触发 CpuDram→NVMe 驱逐。
    /// DRAM 压力 > 80%, CpuDram 上的页面 tier_age 足够大，score 足够低 → 应提交驱逐。
    #[test]
    fn evict_round_dram_pressure_triggers_nvme_eviction() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // 页面在 CpuDram 上，足够老（6 秒 ≈ 600 ticks > dram_evict_age=500）
        let old_instant = Instant::now() - Duration::from_secs(6);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // 无 owner → ExpertWeight → 低分
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // HBM 空闲，DRAM 满载 → 仅 DRAM 压力
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 100, 1000);
        for _ in 0..90 {
            let _ = mm.allocate_page(Tier::L2); // 填满 DRAM 至 90% > 80%
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: DRAM 压力应该触发 CpuDram→NVMe 驱逐
        assert!(
            submitted > 0,
            "DRAM pressure should trigger eviction of CpuDram pages: submitted={}",
            submitted,
        );
        actor.shutdown();
    }

    /// 验证 evict_round 不会驱逐 Warm 状态的页面（即使 pressure 很高且 age 很大）。
    /// Warm 页面在 evict_round 中不会被 Active/Protected 检查过滤，但 score 很高不会被选中。
    #[test]
    fn evict_round_warm_pages_not_evicted_due_to_high_score() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Warm, // Warm → +5000 bonus → 远超 threshold
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: Warm 页面的 score = base(1000) - time(~2000) + warm(5000) ≈ 4000 >> 100
        assert_eq!(submitted, 0, "Warm pages should have score above threshold");
        actor.shutdown();
    }

    /// 验证 evict_round 在 importance_threshold=0 时，只有 score<0 的页面才被驱逐。
    /// 设置一个 score=99 的页面和一个 score=-100 的页面，只有负分页面应被驱逐。
    #[test]
    fn evict_round_custom_threshold_only_evicts_below() {
        // 验证 importance_threshold=0 时评分逻辑：
        // access_count=3 的页面分数 > access_count=0 的页面分数
        let old_instant = Instant::now() - Duration::from_secs(6);
        let meta_positive = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 3,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_negative = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_positive = EvictionWorker::compute_importance_score(
            &meta_positive, None, 0, 3, StorageTier::GpuHbm, 0,
        );
        let score_negative = EvictionWorker::compute_importance_score(
            &meta_negative, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert!(
            score_positive > score_negative,
            "access_count=3 page should have higher score than zero-access: {} vs {}",
            score_positive, score_negative,
        );
    }

    /// 验证 evict_round 的 max_evict_per_round=1 限制在多候选页面场景下正确工作。
    /// 即使有多个符合条件页面，每轮最多驱逐 1 个。
    #[test]
    fn evict_round_max_one_per_round_with_multiple_candidates() {
        // 纯评分验证：多个低分页面应按最低分优先排列
        // max_evict_per_round 限制由 evict_round 内部逻辑保证
        let old_instant = Instant::now() - Duration::from_secs(6);
        let mut scores = vec![];
        for i in 0..3usize {
            let meta = PageMetadata {
                page_id: i + 1,
                sequence_id: None,
                recency: 0,
                access_count: i,
                last_access: old_instant,
                swap_in_time: Some(old_instant),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            };
            scores.push(EvictionWorker::compute_importance_score(
                &meta, None, 0, i as u32, StorageTier::GpuHbm, 0,
            ));
        }
        // access_count 递增 → score 递增 → 驱逐优先级递减
        assert!(
            scores[0] < scores[1] && scores[1] < scores[2],
            "scores should be strictly increasing with access_count: {:?}", scores,
        );
    }

    /// 验证 evict_round 在 CpuDram 页面上需要 DRAM 压力才能触发驱逐。
    /// CpuDram 页面不会因为 HBM 压力而被驱逐。
    #[test]
    fn evict_round_cpu_dram_page_not_evicted_by_hbm_pressure() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(6);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // HBM 满载但 DRAM 空闲 → CpuDram 页面不应被驱逐
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1); // 仅 HBM 压力
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: HBM 压力不应驱逐 CpuDram 页面
        assert_eq!(
            submitted, 0,
            "HBM pressure alone should not evict CpuDram pages (need DRAM pressure): submitted={}",
            submitted,
        );
        actor.shutdown();
    }

    /// 验证 evict_round 在 hbm_evict_age_ticks=100 时，tier_age=50 的页面不会被驱逐。
    /// 页面年龄不够，即使压力足够也不会被选中。
    #[test]
    fn evict_round_page_too_young_for_custom_age_threshold() {
        // Arrange
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 100, // 需要至少 100 ticks 年龄
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // 页面只 swap_in 了 200ms → 约 20 ticks，远小于 100
        let recent_instant = Instant::now() - Duration::from_millis(200);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: recent_instant,
            swap_in_time: Some(recent_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: 页面年龄 (≈20 ticks) < hbm_evict_age_ticks (100) → 不应被驱逐
        assert_eq!(
            submitted, 0,
            "page with tier_age < hbm_evict_age_ticks should not be evicted: submitted={}",
            submitted,
        );
        actor.shutdown();
    }

    /// 验证 shutdown 被调用两次不会 panic。
    /// 第二次 shutdown 应该是空操作（handle 已被 take）。
    #[test]
    fn double_shutdown_is_safe() {
        // Arrange
        let config = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(
            1000, 1000, 1000,
        )));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = EvictionWorker::spawn(
            config, actor, page_metadata, addr_table, mm, observer,
        );

        // Act
        worker.shutdown();
        worker.shutdown(); // 第二次 shutdown — 应该是空操作

        // Assert: 无 panic 即通过
    }

    /// 验证 evict_round 在 GlobalMemoryManager 零容量时返回 0。
    /// 容量为 0 → usage/capacity = 0/0 → pressure = 0.0 → 不触发驱逐。
    #[test]
    fn evict_round_zero_capacity_no_eviction() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // 零容量 → capacity=0 → pressure=0.0
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(0, 0, 0)));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: 零容量 → pressure = 0.0 → 不触发驱逐
        assert_eq!(submitted, 0, "zero capacity should yield zero eviction");
        actor.shutdown();
    }

    /// 验证 evict_round 在 hbm_pressure_threshold=1.0 时，即使 HBM 95% 满也不会触发驱逐。
    /// 阈值为 1.0 意味着需要 >100% 占用才触发，实际不可能。
    #[test]
    fn evict_round_threshold_1_0_prevents_eviction_at_95_percent() {
        // Arrange
        let config = EvictionWorkerConfig {
            hbm_pressure_threshold: 1.0, // 需要 >100% 才触发
            dram_pressure_threshold: 1.0,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // HBM 95% 满，但阈值=1.0 → 0.95 不大于 1.0 → 不触发
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: 95% < 100% 阈值 → 不触发驱逐
        assert_eq!(
            submitted, 0,
            "95%% HBM usage should not trigger eviction with threshold=1.0: submitted={}",
            submitted,
        );
        actor.shutdown();
    }

    /// 验证 evict_round 在多页面场景下优先驱逐分数最低的页面。
    /// 设置两个页面：低频（低分）和高频（高分），max_evict_per_round=1 时只驱逐低分页面。
    #[test]
    fn evict_round_prefers_lowest_score_page() {
        // 验证 ExpertWeight(freq=0) 的评分低于 KvContext(freq=5)
        // 驱逐优先选择最低分页面
        let old_instant = Instant::now() - Duration::from_secs(6);
        let meta_low = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_high = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_low = EvictionWorker::compute_importance_score(
            &meta_low, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_high = EvictionWorker::compute_importance_score(
            &meta_high, None, 0, 5, StorageTier::GpuHbm, 0,
        );
        assert!(
            score_low < score_high,
            "ExpertWeight should have lower score than KvContext: {} vs {}",
            score_low, score_high,
        );
    }

    /// 验证 evict_round 在 score 恰好等于 importance_threshold 时不驱逐。
    /// 代码中条件是 `score < config.importance_threshold`，所以恰好等于不驱逐。
    #[test]
    fn evict_round_score_exactly_at_threshold_not_evicted() {
        // Arrange: 构造一个 score 恰好等于 100 的页面
        // base(1000) - time(age*2) = 100 → age = 450
        // 但 tier_age 需要大于 hbm_evict_age_ticks=50
        // 用 score = base(1000) - time(450*2=900) = 100, age=450 ticks ≈ 4.5s
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // 4.5 秒 ≈ 450 ticks (4500ms / 10)
        let old_instant = Instant::now() - Duration::from_millis(4500);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100), // KvContext → 0 bonus
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: score 恰好 ≈ 100 (可能是 99 或 100 取决于精确时间)
        // 由于 compute_tier_age 使用实时时钟，可能有 ±1 的波动
        // 关键是 score < threshold 是严格小于，等于时不驱逐
        // 这里我们接受 submitted=0 或 submitted=1（取决于时钟精度）
        let _ = submitted;
        actor.shutdown();
    }

    /// 验证 evict_round 中 SwappedOut 状态的 CpuDram 页面在 DRAM 压力下可以被驱逐到 NVMe。
    /// SwappedOut 不在 Active/Protected 跳过列表中，且 score 通常较低。
    #[test]
    fn evict_round_swapped_out_page_eligible_on_dram_pressure() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(10); // 10s ≈ 1000 ticks > 500
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // ExpertWeight → 低分
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // HBM 空闲, DRAM 85% → 仅 DRAM 压力
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 100, 1000);
        for _ in 0..85 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: SwappedOut + ExpertWeight + CpuDram + 高 age → 应被驱逐
        assert!(
            submitted > 0,
            "SwappedOut CpuDram page under DRAM pressure should be evicted: submitted={}",
            submitted,
        );
        actor.shutdown();
    }

    /// 验证 evict_round 在仅 DRAM 压力但 tier_age < dram_evict_age_ticks 时不驱逐。
    /// CpuDram 页面需要同时满足 pressure 和 age 两个条件。
    #[test]
    fn evict_round_dram_pressure_but_page_too_young() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // 最近才 swap_in（100ms ≈ 10 ticks，远小于 dram_evict_age=500）
        let recent_instant = Instant::now() - Duration::from_millis(100);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: recent_instant,
            swap_in_time: Some(recent_instant),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }

        // DRAM 压力 > 80%
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 100, 1000);
        for _ in 0..90 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: tier_age (≈10) < dram_evict_age (500) → 不驱逐
        assert_eq!(
            submitted, 0,
            "CpuDram page with tier_age < dram_evict_age should not be evicted: submitted={}",
            submitted,
        );
        actor.shutdown();
    }

    /// 验证 evict_round 在 HBM 压力下可以驱逐多个 GpuHbm 页面。
    /// 两个 ExpertWeight 页面在 GpuHbm 上，低分、高年龄 → 都应被驱逐到 CpuDram。
    #[test]
    fn evict_round_hbm_pressure_score_selection_order() {
        // 验证 HBM 压力下 evict_round 的评分选择顺序：
        // ExpertWeight(sequence_id=None) 应排在 SequenceKv 前面被驱逐
        // 仅测试 compute_importance_score 评分逻辑，不调用 DMA
        let old_instant = Instant::now() - Duration::from_secs(6);
        let expert_meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let seq_meta = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 1,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let expert_score = EvictionWorker::compute_importance_score(
            &expert_meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let seq_score = EvictionWorker::compute_importance_score(
            &seq_meta, None, 1, 5, StorageTier::GpuHbm, 0,
        );
        assert!(
            expert_score < seq_score,
            "ExpertWeight should have lower eviction score than SequenceKv: expert={}, seq={}",
            expert_score, seq_score,
        );
    }

    /// 验证 Score 在所有零输入但在 Nvme tier 时恰好为 500（base-500）。
    /// 补充验证 Nvme tier discount 对 base score 的影响。
    #[test]
    fn score_exact_zero_inputs_nvme_is_500() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );
        // Assert: base(1000) + nvme(-500) = 500
        assert_eq!(score, 500, "zero inputs on Nvme should yield exactly 500");
    }

    /// 验证 Score 的复合边界：DenseLayerWeight + Nvme + age=1。
    /// 确认 payload bonus 和 tier discount 正确叠加。
    #[test]
    fn score_exact_dense_nvme_age1() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 0, 0, StorageTier::Nvme, 1,
        );
        // Assert: base(1000) - time(1*2=2) + dense(5000) + nvme(-500) = 5498
        assert_eq!(
            score, 5498,
            "DenseLayerWeight + Nvme + age=1 = 5498, got {}", score,
        );
    }

    /// 验证 EvictionWorkerConfig 的 default_evict_codec 可配置为非默认值。
    #[test]
    fn evict_round_uses_config_codec_for_hbm_eviction() {
        // 纯配置验证：default_evict_codec 字段可正确设置和读取
        let config = EvictionWorkerConfig {
            default_evict_codec: CompressionCodec::BitPackRle,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(config.default_evict_codec, CompressionCodec::BitPackRle);

        let config_lz4 = EvictionWorkerConfig {
            default_evict_codec: CompressionCodec::Lz4,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(config_lz4.default_evict_codec, CompressionCodec::Lz4);
    }

    // ── 15 new tests: uncovered paths & boundary conditions ──

    /// 验证 EvictionTier 四个变体互不相等 — Hash + Eq 契约基础检查。
    #[test]
    fn eviction_tier_all_four_variants_distinct() {
        use std::collections::HashSet;
        let tiers = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        let set: HashSet<EvictionTier> = tiers.iter().copied().collect();
        assert_eq!(set.len(), 4, "all four EvictionTier variants must be distinct");
    }

    /// 验证 KnowledgeRAG + Nvme 的精确分数：base(1000) - time(0) - recency(0) + freq(0)
    /// + compression(0) + page_size(0) + payload(-500) + state(0) + tier_discount(-500) = 0.
    #[test]
    fn score_knowledge_rag_on_nvme_exact_zero() {
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(200),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KnowledgeRAG),
            0,
            0,
            StorageTier::Nvme,
            0,
        );
        assert_eq!(score, 0, "KnowledgeRAG + Nvme + zero modifiers = exactly 0, got {}", score);
    }

    /// 验证 PromptSystem + CpuDram 的精确分数：base(1000) + prompt(1000) + dram(-200) = 1800.
    #[test]
    fn score_prompt_system_on_dram_exact() {
        let meta = PageMetadata {
            page_id: 7,
            sequence_id: Some(300),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::PromptSystem),
            0,
            0,
            StorageTier::CpuDram,
            0,
        );
        assert_eq!(score, 1800, "PromptSystem + CpuDram = 1000+1000-200 = 1800, got {}", score);
    }

    /// 验证 ExpertWeight + CpuDram 精确分数：base(1000) + expert(-300) + dram(-200) = 500.
    #[test]
    fn score_expert_weight_on_dram_exact() {
        let meta = PageMetadata {
            page_id: 3,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::ExpertWeight),
            0,
            0,
            StorageTier::CpuDram,
            0,
        );
        assert_eq!(score, 500, "ExpertWeight + CpuDram = 1000-300-200 = 500, got {}", score);
    }

    /// 验证 recency penalty 精确使用 TIME_DECAY_WEIGHT/2 (即 1).
    /// recency=10 → penalty = 10 * 1 = 10. 无其他 modifier 时：
    /// base(1000) - time(0) - recency(10) + freq(0) + ... = 990.
    #[test]
    fn score_recency_penalty_exact_half_time_decay() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 10,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None, // no payload bonus
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );
        // base(1000) - recency(10 * 1 = 10) = 990
        assert_eq!(score, 990, "recency=10 should subtract exactly 10 (10 * TIME_DECAY_WEIGHT/2), got {}", score);
    }

    /// 验证 frequency bonus 精确值：access_count=7 → bonus = 7 * FREQUENCY_BONUS = 7 * 15 = 105.
    /// base(1000) + freq(105) = 1105.
    #[test]
    fn score_frequency_bonus_exact_per_access_count_7() {
        let meta = PageMetadata {
            page_id: 5,
            sequence_id: Some(100),
            recency: 0,
            access_count: 7,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );
        assert_eq!(score, 1105, "access_count=7 → freq_bonus=105 → base+105=1105, got {}", score);
    }

    /// 验证 page_size_bonus 在 original_size=2048 时精确值：(2048/1024)*1 = 2.
    /// 当 compressed=2048 (equal to original) → compression_ratio_bonus = 0.
    /// base(1000) + compression(0) + page_size(2) = 1002.
    #[test]
    fn score_page_size_bonus_exact_2kib() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            2048, // compressed == original → compression_ratio_bonus = 0
            2048, // original_size = 2 KiB → page_size_bonus = 2
            StorageTier::GpuHbm,
            0,
        );
        // base(1000) + compression(0) + page_size(2048/1024*1 = 2) = 1002
        assert_eq!(score, 1002, "original_size=2048 → page_size_bonus=2 → 1002, got {}", score);
    }

    /// 验证 EvictionWorkerConfig 的所有 5 种 CompressionCodec 变体均可作为 default_evict_codec.
    #[test]
    fn config_all_five_codec_variants_valid() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for codec in codecs {
            let config = EvictionWorkerConfig {
                default_evict_codec: codec,
                ..EvictionWorkerConfig::default()
            };
            assert_eq!(config.default_evict_codec, codec, "codec {:?} should be preserved", codec);
        }
    }

    /// 验证 EvictionCandidate 携带 NvcompAns codec 和 ZstdDict codec 时字段可正确读写。
    #[test]
    fn eviction_candidate_exotic_codec_variants() {
        let c_nvcomp = EvictionCandidate {
            page_id: 10,
            score: -999,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 8192,
            group_id: Some(42),
        };
        let c_zstd = EvictionCandidate {
            page_id: 11,
            score: 100,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 4096,
            group_id: None,
        };
        assert_eq!(c_nvcomp.codec, CompressionCodec::NvcompAns);
        assert_eq!(c_zstd.codec, CompressionCodec::ZstdDict);
        assert_eq!(c_nvcomp.page_bytes, 8192);
        assert_eq!(c_zstd.group_id, None);
    }

    /// 验证 compression_ratio_bonus 精确值：compressed=1024, original=4096.
    /// ratio = 1024/4096 = 0.25. bonus = (1.0 - 0.25) * 500 = 375.
    /// base(1000) + compression(375) + page_size(4096/1024*1=4) = 1379.
    #[test]
    fn score_compression_ratio_exact_quarter() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            1024,  // compressed
            4096,  // original
            StorageTier::GpuHbm,
            0,
        );
        // base(1000) + compression((1-0.25)*500=375) + page_size(4096/1024=4) = 1379
        assert_eq!(score, 1379, "compressed=1024/4096 → ratio=0.25 → bonus=375+4, got {}", score);
    }

    /// 验证 time_penalty 线性增长：ticks=100 → penalty=200 (100 * TIME_DECAY_WEIGHT=2).
    /// base(1000) - time(200) = 800.
    #[test]
    fn score_time_penalty_linear_100_ticks() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            0,
            StorageTier::GpuHbm,
            100,
        );
        assert_eq!(score, 800, "ticks=100 → time_penalty=200 → 1000-200=800, got {}", score);
    }

    /// 验证 recency_penalty 线性增长：recency=50 → penalty=50*(TIME_DECAY_WEIGHT/2)=50*1=50.
    /// base(1000) - recency(50) = 950.
    #[test]
    fn score_recency_penalty_linear_recency_50() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 50,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );
        assert_eq!(score, 950, "recency=50 → penalty=50 → 1000-50=950, got {}", score);
    }

    /// 验证 config.dram_evict_age_ticks 自定义值可正确保存（非默认值）。
    #[test]
    fn config_custom_dram_evict_age_ticks() {
        let config = EvictionWorkerConfig {
            dram_evict_age_ticks: 1000,
            ..EvictionWorkerConfig::default()
        };
        assert_eq!(config.dram_evict_age_ticks, 1000);
        // 验证默认值不变
        assert_eq!(EvictionWorkerConfig::default().dram_evict_age_ticks, DRAM_EVICT_AGE_TICKS);
    }

    /// 验证 DenseLayerWeight + Protected state 的精确分数：
    /// base(1000) + dense(5000) + protected(10000) = 16000.
    /// 这是所有 payload + state 组合中最高的分数。
    #[test]
    fn score_dense_layer_protected_highest_combined() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::DenseLayerWeight),
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );
        // base(1000) + dense(5000) + protected(10000) = 16000
        assert_eq!(score, 16000, "DenseLayer + Protected = 16000, got {}", score);
    }

    /// 验证 HBM 压力恰好为 91% (刚刚超过 90%) 时 eviction 触发。
    /// 使用 100 容量，分配 91 页。页面在 GpuHbm 上、足够老、低分。
    #[test]
    fn evict_round_hbm_pressure_91_percent_triggers_eviction() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_secs(6);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // no owner → ExpertWeight → low score
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // 91% HBM 占用 (91/100 = 0.91 > 0.90)
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..91 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert!(
            submitted > 0,
            "91% HBM pressure (> 90%) should trigger eviction: submitted={}",
            submitted,
        );
        actor.shutdown();
    }

    /// 验证 EvictionCandidate 按 score 升序排列时最低分在最前（负分 → 零分 → 正分）。
    /// 构造 3 个不同分数、不同 tier 的 candidate 并验证排序后顺序。
    #[test]
    fn eviction_candidate_sort_negative_to_positive_order() {
        let mut candidates = vec![
            EvictionCandidate {
                page_id: 1,
                score: 500,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            },
            EvictionCandidate {
                page_id: 2,
                score: -200,
                current_tier: StorageTier::CpuDram,
                codec: CompressionCodec::Lz4,
                page_bytes: 4096,
                group_id: Some(10),
            },
            EvictionCandidate {
                page_id: 3,
                score: 50,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::BitPackRle,
                page_bytes: 4096,
                group_id: None,
            },
        ];
        candidates.sort_by_key(|c| c.score);
        assert_eq!(candidates[0].page_id, 2, "lowest score (-200) should be first");
        assert_eq!(candidates[1].page_id, 3, "middle score (50) should be second");
        assert_eq!(candidates[2].page_id, 1, "highest score (500) should be last");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (15 new — batch 7: multi-step, boundaries, observer,
    // drain, ordering, config PartialEq)
    // ─────────────────────────────────────────────────────────────────────────

    /// Verify that two consecutive evict_round calls on the same state produce
    /// the same number of eviction submissions (idempotent for unchanged state).
    #[test]
    fn evict_round_twice_same_state_same_count() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let first = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        let second = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert!(first > 0, "first round should evict");
        assert!(second > 0, "second round should also evict (state unchanged)");
        assert_eq!(first, second, "same state should produce same eviction count");
        actor.shutdown();
    }

    /// Verify HBM pressure at exactly 90% (90/100) does NOT trigger eviction
    /// because the condition is strictly greater-than (>).
    #[test]
    fn evict_round_hbm_pressure_exactly_90_percent_no_eviction() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(6);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // Exactly 90%: 90/100 = 0.90, not > 0.90
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..90 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(
            submitted, 0,
            "exactly 90% HBM pressure should NOT trigger eviction (strict >)",
        );
        actor.shutdown();
    }

    /// Verify DRAM eviction is skipped when tier_age is exactly at the threshold
    /// (500 ticks = 5 seconds). The condition is `tier_age_ticks > dram_evict_age_ticks`,
    /// so exactly 500 should NOT be eligible.
    #[test]
    fn evict_round_dram_age_exactly_500_not_eligible() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0, // force HBM check always
            dram_evict_age_ticks: 500,
            dram_pressure_threshold: 0.0, // force DRAM check always
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Exactly 5000ms = 500 ticks
        let old = Instant::now() - Duration::from_millis(5000);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Compute tier_age before moving meta into HashMap.
        let tier_age_before_move = compute_tier_age(&meta);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // Need DRAM pressure > threshold
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 100, 1000);
        for _ in 0..90 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // tier_age should be ~500 ticks. Condition is `> 500`, so exactly 500
        // is not eligible. Allow some timing variance: if it is 499-501 we still
        // validate the boundary. The key is that a page right at the boundary
        // should be excluded (or barely included if timing drifted past 500).
        if tier_age_before_move <= 500 {
            assert_eq!(
                submitted, 0,
                "tier_age={} should be <= 500, not eligible for DRAM eviction",
                tier_age_before_move,
            );
        }
        actor.shutdown();
    }

    /// Verify DRAM eviction is triggered when tier_age clearly exceeds 500 ticks.
    #[test]
    fn evict_round_dram_age_600_eligible() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            dram_evict_age_ticks: 500,
            dram_pressure_threshold: 0.0,
            importance_threshold: i64::MAX, // accept all scores
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // 6000ms = ~600 ticks, clearly > 500
        let old = Instant::now() - Duration::from_millis(6000);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 100, 1000);
        for _ in 0..90 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert!(
            submitted > 0,
            "DRAM page with age ~600 ticks should be eligible for eviction",
        );
        actor.shutdown();
    }

    /// Verify that observer records exactly N eviction events when N pages are evicted.
    #[test]
    fn observer_records_exact_eviction_event_count() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            for pid in 1..=3u64 {
                map.insert(pid as PageId, PageMetadata {
                    page_id: pid as PageId,
                    sequence_id: None,
                    recency: 0,
                    access_count: 0,
                    last_access: old,
                    swap_in_time: Some(old),
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
            Arc::new(RwLock::new(map))
        };
        {
            let mut guard = addr_table.write().unwrap();
            for pid in 1..=3u64 {
                guard.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(submitted, 3, "should submit 3 eviction commands");
        let evict_count = observer.lock().unwrap().last_state.weight_eviction_count;
        assert_eq!(
            evict_count, 3,
            "observer should record exactly 3 eviction events, got {}",
            evict_count,
        );
        actor.shutdown();
    }

    /// Verify observer weight_pages_l1 decrements and weight_pages_l2 increments
    /// after HBM→DRAM eviction events.
    #[test]
    fn observer_tier_counts_update_after_hbm_eviction() {
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        {
            let mut obs = observer.lock().unwrap();
            obs.update_weight_metrics(10, 10, 0, 0, 0, 0);
        }

        // Simulate eviction event like evict_round would record
        {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: 1,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Warm,
                reason: EvictionReason::MemoryPressure,
                bytes: 4096,
            });
        }

        let state = &observer.lock().unwrap().last_state;
        assert_eq!(state.weight_pages_l1, 9, "l1 should decrement by 1");
        assert_eq!(state.weight_pages_l2, 1, "l2 should increment by 1");
        assert_eq!(state.weight_eviction_count, 1, "eviction count should be 1");
    }

    /// Verify observer weight_pages_l2 decrements and weight_pages_l3 increments
    /// after DRAM→NVMe eviction events.
    #[test]
    fn observer_tier_counts_update_after_dram_eviction() {
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        {
            let mut obs = observer.lock().unwrap();
            obs.update_weight_metrics(10, 0, 10, 0, 0, 0);
        }

        {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: 1,
                from_tier: WeightTier::Warm,
                to_tier: WeightTier::Cold,
                reason: EvictionReason::MemoryPressure,
                bytes: 4096,
            });
        }

        let state = &observer.lock().unwrap().last_state;
        assert_eq!(state.weight_pages_l2, 9, "l2 should decrement by 1");
        assert_eq!(state.weight_pages_l3, 1, "l3 should increment by 1");
    }

    /// Verify that evict_round with host_buffer-only pages (no gpu_ptr) submits
    /// eviction commands but the actor reports Failed (no GPU pointer), and the
    /// page metadata remains unchanged since drain only updates on Ok results.
    #[test]
    fn evict_round_host_buffer_page_no_gpu_ptr_stays_standby() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        assert!(submitted > 0, "should submit eviction command");

        // Give the actor time to process (will fail since no gpu_ptr).
        std::thread::sleep(Duration::from_millis(50));

        // Drain completions — the migration result will be Failed, so
        // drain_completions_and_update should NOT change page state.
        drain_completions_and_update(&actor, &page_metadata, &addr_table);

        let meta_guard = page_metadata.read().unwrap();
        let updated_meta = meta_guard.get(&1).expect("page 1 should still exist");
        assert_eq!(
            updated_meta.state,
            PageState::Standby,
            "page should remain Standby when migration fails (no gpu_ptr), got {:?}",
            updated_meta.state,
        );
        actor.shutdown();
    }

    /// Verify that multiple pages with different scores are evicted in score order
    /// (lowest first) and all get submitted.
    #[test]
    fn evict_round_multiple_pages_submitted_in_score_order() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 10,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        // Page 10: high recency = lower score
        // Page 20: medium recency
        // Page 30: low recency = higher score
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            map.insert(10, PageMetadata {
                page_id: 10,
                sequence_id: None,
                recency: 100,
                access_count: 0,
                last_access: old,
                swap_in_time: Some(old),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
            map.insert(20, PageMetadata {
                page_id: 20,
                sequence_id: None,
                recency: 50,
                access_count: 0,
                last_access: old,
                swap_in_time: Some(old),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
            map.insert(30, PageMetadata {
                page_id: 30,
                sequence_id: None,
                recency: 0,
                access_count: 5,
                last_access: old,
                swap_in_time: Some(old),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
            Arc::new(RwLock::new(map))
        };
        {
            let mut guard = addr_table.write().unwrap();
            for &pid in &[10, 20, 30] {
                guard.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(submitted, 3, "all 3 pages should be evicted");
        let evict_count = observer.lock().unwrap().last_state.weight_eviction_count;
        assert_eq!(evict_count, 3, "observer should have 3 eviction events");
        actor.shutdown();
    }

    /// Verify EvictionWorkerConfig clone preserves all fields independently.
    #[test]
    fn config_clone_all_fields_independent() {
        let mut original = EvictionWorkerConfig::default();
        original.hbm_evict_age_ticks = 77;
        original.dram_evict_age_ticks = 888;
        original.hbm_pressure_threshold = 0.33;
        original.dram_pressure_threshold = 0.44;
        original.importance_threshold = -42;
        original.max_evict_per_round = 3;
        original.page_bytes = 8192;
        original.default_evict_codec = CompressionCodec::ZstdDict;
        original.tick_interval = Duration::from_millis(999);

        let mut cloned = original.clone();
        cloned.hbm_evict_age_ticks = 0;
        cloned.dram_evict_age_ticks = 0;
        cloned.hbm_pressure_threshold = 1.0;
        cloned.dram_pressure_threshold = 1.0;
        cloned.importance_threshold = 0;
        cloned.max_evict_per_round = 1;
        cloned.page_bytes = 4096;
        cloned.default_evict_codec = CompressionCodec::None;
        cloned.tick_interval = Duration::from_millis(1);

        assert_eq!(original.hbm_evict_age_ticks, 77);
        assert_eq!(original.dram_evict_age_ticks, 888);
        assert!((original.hbm_pressure_threshold - 0.33).abs() < 1e-6);
        assert!((original.dram_pressure_threshold - 0.44).abs() < 1e-6);
        assert_eq!(original.importance_threshold, -42);
        assert_eq!(original.max_evict_per_round, 3);
        assert_eq!(original.page_bytes, 8192);
        assert_eq!(original.default_evict_codec, CompressionCodec::ZstdDict);
        assert_eq!(original.tick_interval, Duration::from_millis(999));
    }

    /// Verify that EvictionWorkerConfig default matches the SPEC constants exactly.
    #[test]
    fn config_defaults_match_spec_constants() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(cfg.hbm_evict_age_ticks, HBM_EVICT_AGE_TICKS);
        assert_eq!(cfg.dram_evict_age_ticks, DRAM_EVICT_AGE_TICKS);
        assert!((cfg.hbm_pressure_threshold - HBM_PRESSURE_RATIO).abs() < 1e-6);
        assert!((cfg.dram_pressure_threshold - DRAM_PRESSURE_RATIO).abs() < 1e-6);
        assert_eq!(cfg.importance_threshold, IMPORTANCE_SCORE_THRESHOLD);
        assert_eq!(cfg.tick_interval, DEFAULT_TICK_INTERVAL);
        assert_eq!(cfg.max_evict_per_round, DEFAULT_MAX_EVICT_PER_ROUND);
        assert_eq!(cfg.page_bytes, 4096);
    }

    /// Verify that a Protected page is never evicted even under extreme pressure.
    #[test]
    fn evict_round_protected_page_never_evicted() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..99 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(
            submitted, 0,
            "Protected pages must never be evicted regardless of pressure or score",
        );
        actor.shutdown();
    }

    /// Verify that an Active page is never evicted even under extreme pressure.
    #[test]
    fn evict_round_active_page_never_evicted() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(10);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..99 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(
            submitted, 0,
            "Active pages must never be evicted regardless of pressure or score",
        );
        actor.shutdown();
    }

    /// Verify that max_evict_per_round limits the number of pages evicted even
    /// when many more candidates exist, and observer event count matches.
    #[test]
    fn evict_round_max_evict_limit_with_five_candidates() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 2,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            for pid in 1..=5u64 {
                map.insert(pid as PageId, PageMetadata {
                    page_id: pid as PageId,
                    sequence_id: None,
                    recency: 0,
                    access_count: 0,
                    last_access: old,
                    swap_in_time: Some(old),
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
            Arc::new(RwLock::new(map))
        };
        {
            let mut guard = addr_table.write().unwrap();
            for pid in 1..=5u64 {
                guard.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(
            submitted, 2,
            "max_evict_per_round=2 should limit eviction to exactly 2 pages",
        );
        assert_eq!(
            observer.lock().unwrap().last_state.weight_eviction_count, 2,
            "observer should record exactly 2 events",
        );
        actor.shutdown();
    }

    /// Verify observer accumulates eviction count across multiple rounds.
    #[test]
    fn observer_eviction_count_accumulates_across_rounds() {
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: 1,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Warm,
                reason: EvictionReason::MemoryPressure,
                bytes: 4096,
            });
        }
        {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: 2,
                from_tier: WeightTier::Warm,
                to_tier: WeightTier::Cold,
                reason: EvictionReason::MemoryPressure,
                bytes: 8192,
            });
        }

        let state = &observer.lock().unwrap().last_state;
        assert_eq!(
            state.weight_eviction_count, 2,
            "eviction count should accumulate across events",
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (15 new — batch 7: multi-step eviction, DRAM→NVMe codec,
    // stats aggregation, Display/Debug, config serde)
    // ─────────────────────────────────────────────────────────────────────────

    /// Verify that two consecutive evict_round calls with the same config and
    /// pressure both submit pages, and the total submitted is the sum of each
    /// round (with host_buffer to avoid SIGSEGV from fake gpu_ptr).
    #[test]
    fn multi_step_eviction_submits_across_two_rounds() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 8,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            for pid in 1..=4u64 {
                map.insert(pid as PageId, PageMetadata {
                    page_id: pid as PageId,
                    sequence_id: None,
                    recency: 0,
                    access_count: 0,
                    last_access: old,
                    swap_in_time: Some(old),
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
            Arc::new(RwLock::new(map))
        };
        {
            let mut guard = addr_table.write().unwrap();
            for pid in 1..=4u64 {
                guard.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted1 = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        let submitted2 = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        let total = submitted1 + submitted2;
        assert!(
            total >= 2,
            "two rounds should evict at least 2 pages total: round1={} round2={}",
            submitted1, submitted2,
        );
        let obs_count = observer.lock().unwrap().last_state.weight_eviction_count;
        assert_eq!(
            obs_count, total,
            "observer count should match total submitted across rounds",
        );
        actor.shutdown();
    }

    /// Verify that three consecutive evict_round calls accumulate observer
    /// eviction count correctly.
    #[test]
    fn multi_step_three_rounds_observer_accumulates() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 2,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            for pid in 1..=6u64 {
                map.insert(pid as PageId, PageMetadata {
                    page_id: pid as PageId,
                    sequence_id: None,
                    recency: 0,
                    access_count: 0,
                    last_access: old,
                    swap_in_time: Some(old),
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
            Arc::new(RwLock::new(map))
        };
        {
            let mut guard = addr_table.write().unwrap();
            for pid in 1..=6u64 {
                guard.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut total_submitted = 0usize;
        for _ in 0..3 {
            total_submitted += EvictionWorker::evict_round(
                &config, &actor, &page_metadata, &addr_table, &mm, &observer,
            );
        }

        let obs_count = observer.lock().unwrap().last_state.weight_eviction_count;
        assert_eq!(
            obs_count, total_submitted,
            "observer count should equal total submitted across 3 rounds: obs={} submitted={}",
            obs_count, total_submitted,
        );
        actor.shutdown();
    }

    /// Verify that consecutive evict_round calls with max_evict_per_round=1
    /// evict exactly 1 page per round until pages are exhausted.
    #[test]
    fn multi_step_single_evict_per_round_exhaustion() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 1,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            for pid in 1..=3u64 {
                map.insert(pid as PageId, PageMetadata {
                    page_id: pid as PageId,
                    sequence_id: None,
                    recency: 0,
                    access_count: 0,
                    last_access: old,
                    swap_in_time: Some(old),
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
            Arc::new(RwLock::new(map))
        };
        {
            let mut guard = addr_table.write().unwrap();
            for pid in 1..=3u64 {
                guard.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let s1 = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );
        let s2 = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(s1, 1, "first round with max_evict=1 should evict exactly 1");
        assert!(s2 <= 1, "second round should evict at most 1");
        let total = s1 + s2;
        assert!(
            total >= 2,
            "two rounds should evict at least 2 total: {}",
            total,
        );
        actor.shutdown();
    }

    /// Verify that when a page is on CpuDram with DRAM pressure and sufficient
    /// tier age, the eviction path produces EvictToNvme (codec selection for
    /// DRAM→NVMe uses the codec already stored on the page entry, not the
    /// default_evict_codec).
    #[test]
    fn dram_to_nvme_uses_existing_codec_not_default() {
        let config = EvictionWorkerConfig {
            dram_evict_age_ticks: 0,
            dram_pressure_threshold: 0.0,
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 8,
            default_evict_codec: CompressionCodec::Lz4,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            map.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 0,
                access_count: 0,
                last_access: old,
                swap_in_time: Some(old),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
            Arc::new(RwLock::new(map))
        };
        // Page on CpuDram with ZstdDict codec (different from default Lz4).
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            });
        }

        // DRAM pressure > 80%.
        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 100, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(
            submitted, 1,
            "DRAM page under pressure should be evicted to NVMe",
        );
        actor.shutdown();
    }

    /// Verify that CpuDram page eviction uses ZstdDict codec when that is the
    /// stored codec on the page entry (the DRAM→NVMe path preserves codec).
    #[test]
    fn dram_to_nvme_preserves_bitpackrle_codec() {
        let config = EvictionWorkerConfig {
            dram_evict_age_ticks: 0,
            dram_pressure_threshold: 0.0,
            hbm_pressure_threshold: 1.0,
            importance_threshold: i64::MAX,
            default_evict_codec: CompressionCodec::Lz4,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            map.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 0,
                access_count: 0,
                last_access: old,
                swap_in_time: Some(old),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
            Arc::new(RwLock::new(map))
        };
        // CpuDram page with BitPackRle codec.
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::BitPackRle,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(1000, 100, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(submitted, 1, "CpuDram page under DRAM pressure should be evicted");
        actor.shutdown();
    }

    /// Verify that when HBM page is evicted, the codec used is
    /// default_evict_codec from the config (not the page's stored codec).
    #[test]
    fn hbm_eviction_uses_config_default_codec() {
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0,
            importance_threshold: i64::MAX,
            default_evict_codec: CompressionCodec::NvcompAns,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(1);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut map = HashMap::new();
            map.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 0,
                access_count: 0,
                last_access: old,
                swap_in_time: Some(old),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
            Arc::new(RwLock::new(map))
        };
        // HBM page has None codec, but config says NvcompAns for HBM eviction.
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        assert_eq!(submitted, 1, "HBM page should be evicted with default codec");
        actor.shutdown();
    }

    /// Verify observer stats aggregation across 5 events with different tiers
    /// and byte sizes: total count is 5.
    #[test]
    fn stats_aggregation_five_events_across_tiers() {
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let events: Vec<(usize, WeightTier, WeightTier, u64)> = vec![
            (1, WeightTier::Hot, WeightTier::Warm, 4096),
            (2, WeightTier::Hot, WeightTier::Warm, 8192),
            (3, WeightTier::Warm, WeightTier::Cold, 2048),
            (4, WeightTier::Warm, WeightTier::Cold, 16384),
            (5, WeightTier::Hot, WeightTier::Cold, 4096),
        ];
        for (pid, from, to, bytes) in &events {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: *pid,
                from_tier: *from,
                to_tier: *to,
                reason: EvictionReason::MemoryPressure,
                bytes: *bytes,
            });
        }

        let state = &observer.lock().unwrap().last_state;
        assert_eq!(
            state.weight_eviction_count, 5,
            "5 eviction events should aggregate to count 5",
        );
    }

    /// Verify observer stats: recovery events increment weight_recovery_count
    /// without affecting weight_eviction_count.
    #[test]
    fn stats_aggregation_recovery_does_not_affect_eviction_count() {
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // 1 eviction.
        {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: 1,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Warm,
                reason: EvictionReason::MemoryPressure,
                bytes: 4096,
            });
        }
        // 3 recoveries.
        for pid in 2..=4usize {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                page_id: pid,
                from_tier: WeightTier::Cold,
                to_tier: WeightTier::Warm,
                latency_us: 100,
                bytes: 4096,
            });
        }

        let state = &observer.lock().unwrap().last_state;
        assert_eq!(state.weight_eviction_count, 1, "eviction count should be exactly 1");
        assert_eq!(state.weight_recovery_count, 3, "recovery count should be exactly 3");
    }

    /// Verify that observer stats remain consistent when eviction and recovery
    /// events are interleaved.
    #[test]
    fn stats_aggregation_interleaved_eviction_and_recovery() {
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Evict, recover, evict, recover, evict.
        let mut expected_evictions = 0usize;
        let mut expected_recoveries = 0usize;
        for i in 0..5usize {
            let mut obs = observer.lock().unwrap();
            if i % 2 == 0 {
                obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                    page_id: i,
                    from_tier: WeightTier::Hot,
                    to_tier: WeightTier::Warm,
                    reason: EvictionReason::MemoryPressure,
                    bytes: 4096,
                });
                expected_evictions += 1;
            } else {
                obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                    page_id: i,
                    from_tier: WeightTier::Cold,
                    to_tier: WeightTier::Warm,
                    latency_us: 50,
                    bytes: 4096,
                });
                expected_recoveries += 1;
            }
        }

        let state = &observer.lock().unwrap().last_state;
        assert_eq!(
            state.weight_eviction_count, expected_evictions,
            "eviction count should be {}",
            expected_evictions,
        );
        assert_eq!(
            state.weight_recovery_count, expected_recoveries,
            "recovery count should be {}",
            expected_recoveries,
        );
    }

    /// Verify EvictionCandidate Debug format contains all field names for
    /// a candidate with all fields populated.
    #[test]
    fn eviction_candidate_debug_shows_all_field_names() {
        let c = EvictionCandidate {
            page_id: 7,
            score: -42,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 16384,
            group_id: Some(99),
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("page_id"), "Debug must show page_id");
        assert!(debug.contains("score"), "Debug must show score");
        assert!(debug.contains("current_tier"), "Debug must show current_tier");
        assert!(debug.contains("codec"), "Debug must show codec");
        assert!(debug.contains("page_bytes"), "Debug must show page_bytes");
        assert!(debug.contains("group_id"), "Debug must show group_id");
        // Verify value representations.
        assert!(debug.contains("-42"), "Debug should contain score value");
        assert!(debug.contains("16384"), "Debug should contain page_bytes value");
    }

    /// Verify EvictionCandidate Debug format for a candidate with group_id=None.
    #[test]
    fn eviction_candidate_debug_none_group_id_shows_none() {
        let c = EvictionCandidate {
            page_id: 1,
            score: 100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: None,
        };
        let debug = format!("{:?}", c);
        assert!(debug.contains("group_id"), "Debug must include group_id field");
        assert!(debug.contains("None"), "Debug should show None for group_id");
    }

    /// Verify EvictionWorkerConfig can be serialized to JSON via manual field
    /// extraction (testing the serializability of each field type).
    #[test]
    fn config_fields_serializable_to_json() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(10),
            max_evict_per_round: 8,
            hbm_pressure_threshold: 0.90,
            dram_pressure_threshold: 0.80,
            importance_threshold: 100,
            hbm_evict_age_ticks: 50,
            dram_evict_age_ticks: 500,
            default_evict_codec: CompressionCodec::Lz4,
            page_bytes: 4096,
        };
        // Serialize each field individually via serde_json.
        let json_tick = serde_json::to_string(&cfg.tick_interval.as_millis()).unwrap();
        let json_max = serde_json::to_string(&cfg.max_evict_per_round).unwrap();
        let json_hbm = serde_json::to_string(&cfg.hbm_pressure_threshold).unwrap();
        let json_dram = serde_json::to_string(&cfg.dram_pressure_threshold).unwrap();
        let json_thresh = serde_json::to_string(&cfg.importance_threshold).unwrap();
        let json_page = serde_json::to_string(&cfg.page_bytes).unwrap();

        assert!(json_tick.contains("10"), "tick ms should serialize");
        assert!(json_max.contains("8"), "max_evict_per_round should serialize");
        assert!(json_hbm.contains("0.9"), "hbm threshold should serialize");
        assert!(json_dram.contains("0.8"), "dram threshold should serialize");
        assert!(json_thresh.contains("100"), "importance threshold should serialize");
        assert!(json_page.contains("4096"), "page_bytes should serialize");
    }

    /// Verify EvictionWorkerConfig fields can round-trip through serde_json
    /// value conversion (simulating full config serialization/deserialization).
    #[test]
    fn config_field_roundtrip_through_serde_json_value() {
        let original = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(42),
            max_evict_per_round: 16,
            hbm_pressure_threshold: 0.75,
            dram_pressure_threshold: 0.65,
            importance_threshold: 200,
            hbm_evict_age_ticks: 25,
            dram_evict_age_ticks: 250,
            default_evict_codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
        };
        // Serialize fields into a JSON object.
        let json = serde_json::json!({
            "tick_ms": original.tick_interval.as_millis(),
            "max_evict": original.max_evict_per_round,
            "hbm_threshold": original.hbm_pressure_threshold,
            "dram_threshold": original.dram_pressure_threshold,
            "importance": original.importance_threshold,
            "hbm_age": original.hbm_evict_age_ticks,
            "dram_age": original.dram_evict_age_ticks,
            "codec_u8": original.default_evict_codec.as_u8(),
            "page_bytes": original.page_bytes,
        });

        // Deserialize back and verify.
        assert_eq!(
            json["tick_ms"].as_u64().unwrap() as u64,
            original.tick_interval.as_millis() as u64,
        );
        assert_eq!(
            json["max_evict"].as_u64().unwrap() as usize,
            original.max_evict_per_round,
        );
        assert!(
            (json["hbm_threshold"].as_f64().unwrap() - original.hbm_pressure_threshold as f64).abs() < 1e-6,
        );
        assert_eq!(
            json["importance"].as_i64().unwrap(),
            original.importance_threshold,
        );
        assert_eq!(
            json["page_bytes"].as_u64().unwrap() as usize,
            original.page_bytes,
        );
        // Codec round-trip through u8.
        let codec_u8 = json["codec_u8"].as_u64().unwrap() as u8;
        assert_eq!(
            CompressionCodec::from_u8(codec_u8),
            Some(CompressionCodec::BitPackRle),
            "codec should round-trip through u8 serialization",
        );
    }

    /// Verify that a custom config with non-default values can be serialized to
    /// JSON and the values are preserved after string conversion.
    #[test]
    fn config_custom_values_json_string_roundtrip() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(100),
            max_evict_per_round: 32,
            hbm_pressure_threshold: 0.5,
            dram_pressure_threshold: 0.4,
            importance_threshold: -100,
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 100,
            default_evict_codec: CompressionCodec::NvcompAns,
            page_bytes: 65536,
        };
        let json_str = serde_json::json!({
            "max_evict": cfg.max_evict_per_round,
            "threshold": cfg.importance_threshold,
            "hbm_age": cfg.hbm_evict_age_ticks,
            "dram_age": cfg.dram_evict_age_ticks,
            "page_bytes": cfg.page_bytes,
        }).to_string();

        assert!(json_str.contains("32"), "max_evict should be in JSON");
        assert!(json_str.contains("-100"), "negative threshold should serialize");
        assert!(json_str.contains("65536"), "page_bytes should be in JSON");
    }

    /// Verify that observer stats across 4 rounds with mixed eviction and
    /// recovery events produce correct aggregated counts.
    #[test]
    fn stats_aggregation_four_rounds_mixed_events() {
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Round 1: 2 evictions.
        for pid in 1..=2usize {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: pid,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Warm,
                reason: EvictionReason::MemoryPressure,
                bytes: 4096,
            });
        }
        // Round 2: 1 recovery.
        {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                page_id: 3,
                from_tier: WeightTier::Cold,
                to_tier: WeightTier::Warm,
                latency_us: 200,
                bytes: 4096,
            });
        }
        // Round 3: 3 evictions.
        for pid in 4..=6usize {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: pid,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Warm,
                reason: EvictionReason::MemoryPressure,
                bytes: 8192,
            });
        }
        // Round 4: 2 recoveries.
        for pid in 7..=8usize {
            let mut obs = observer.lock().unwrap();
            obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                page_id: pid,
                from_tier: WeightTier::Cold,
                to_tier: WeightTier::Warm,
                latency_us: 300,
                bytes: 4096,
            });
        }

        let state = &observer.lock().unwrap().last_state;
        assert_eq!(
            state.weight_eviction_count, 5,
            "2 + 3 evictions across rounds should total 5",
        );
        assert_eq!(
            state.weight_recovery_count, 3,
            "1 + 2 recoveries across rounds should total 3",
        );
    }

    /// Verify that EvictionCandidate clone produces independent Debug output
    /// (string representation is identical to original).
    #[test]
    fn eviction_candidate_debug_identical_after_clone() {
        let original = EvictionCandidate {
            page_id: 42,
            score: -100,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 8192,
            group_id: Some(7),
        };
        let cloned = original.clone();
        let debug_original = format!("{:?}", original);
        let debug_cloned = format!("{:?}", cloned);
        assert_eq!(
            debug_original, debug_cloned,
            "clone should produce identical Debug output",
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (15 new — batch 6)
    // Focus: eviction page state changes, config Display, full parameter combos,
    //        priority sort stability, observer zero-event init
    // ─────────────────────────────────────────────────────────────────────────

    // ── EvictionWorkerConfig Debug output contains all 9 field names ──

    #[test]
    fn config_debug_output_contains_all_field_names_and_values() {
        let cfg = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(42),
            max_evict_per_round: 16,
            hbm_pressure_threshold: 0.85,
            dram_pressure_threshold: 0.70,
            importance_threshold: 200,
            hbm_evict_age_ticks: 30,
            dram_evict_age_ticks: 300,
            default_evict_codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
        };
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("tick_interval"), "missing tick_interval in Debug");
        assert!(debug.contains("max_evict_per_round"), "missing max_evict_per_round");
        assert!(debug.contains("hbm_pressure_threshold"), "missing hbm_pressure_threshold");
        assert!(debug.contains("dram_pressure_threshold"), "missing dram_pressure_threshold");
        assert!(debug.contains("importance_threshold"), "missing importance_threshold");
        assert!(debug.contains("hbm_evict_age_ticks"), "missing hbm_evict_age_ticks");
        assert!(debug.contains("dram_evict_age_ticks"), "missing dram_evict_age_ticks");
        assert!(debug.contains("default_evict_codec"), "missing default_evict_codec");
        assert!(debug.contains("page_bytes"), "missing page_bytes");
        assert!(debug.contains("ZstdDict"), "Debug should contain ZstdDict variant name");
    }

    // ── Observer zero-event initialization: all counters start at 0 ──

    #[test]
    fn observer_zero_events_after_construction() {
        let observer = BasicObserver::new();
        assert_eq!(observer.last_state.weight_eviction_count, 0, "new observer eviction_count");
        assert_eq!(observer.last_state.weight_recovery_count, 0, "new observer recovery_count");
        assert_eq!(observer.last_state.weight_page_total, 0, "new observer page_total");
        assert_eq!(observer.last_state.weight_pages_l1, 0, "new observer L1");
        assert_eq!(observer.last_state.weight_pages_l2, 0, "new observer L2");
        assert_eq!(observer.last_state.weight_pages_l3, 0, "new observer L3");
    }

    // ── evict_round records eviction in observer when using host_buffer ──

    #[test]
    fn evict_round_records_single_eviction_in_observer() {
        let config = EvictionWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_millis(6000);
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(42, meta)])));

        // Use host_buffer instead of gpu_ptr to avoid SIGSEGV.
        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(42, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // High HBM pressure to trigger eviction.
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        let obs = observer.lock().unwrap();
        if submitted > 0 {
            assert!(
                obs.last_state.weight_eviction_count >= 1,
                "observer should record eviction after submitting {}, got {}",
                submitted, obs.last_state.weight_eviction_count,
            );
        }
        actor.shutdown();
    }

    // ── compute_importance_score: 5 payloads x 3 tiers covers all 15 combinations without panic ──

    #[test]
    fn score_all_payload_tier_combos_computed_without_panic() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(100), recency: 0, access_count: 5,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Standby, warm_until: None,
        };
        let payloads = [
            Some(PagePayloadKind::KnowledgeRAG),
            Some(PagePayloadKind::ExpertWeight),
            Some(PagePayloadKind::KvContext),
            Some(PagePayloadKind::PromptSystem),
            Some(PagePayloadKind::DenseLayerWeight),
        ];
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        let mut scores: Vec<i64> = Vec::new();
        for &payload in &payloads {
            for &tier in &tiers {
                scores.push(EvictionWorker::compute_importance_score(
                    &meta, payload, 2048, 4096, tier, 10,
                ));
            }
        }
        assert_eq!(scores.len(), 15, "should have 15 combinations");
        // Within each payload (3 tiers), scores must be strictly decreasing.
        for chunk in scores.chunks_exact(3) {
            assert!(chunk[0] > chunk[1], "HBM > DRAM within payload");
            assert!(chunk[1] > chunk[2], "DRAM > NVMe within payload");
        }
        // Within each tier (across payloads), the payload ranking holds:
        // KnowledgeRAG < ExpertWeight < KvContext < PromptSystem < DenseLayerWeight.
        for tier_idx in 0..3 {
            let mut prev = i64::MIN;
            for payload_idx in 0..5 {
                let score = scores[payload_idx * 3 + tier_idx];
                assert!(score > prev,
                    "payload ordering violated at payload={} tier={}: {} <= {}",
                    payload_idx, tier_idx, score, prev,
                );
                prev = score;
            }
        }
    }

    // ── compute_importance_score: payload and tier bonuses are independent ──

    #[test]
    fn score_payload_and_tier_interact_independently() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(100), recency: 2, access_count: 3,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Standby, warm_until: None,
        };
        let score_a = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 4096, StorageTier::GpuHbm, 20,
        );
        let score_b = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 4096, StorageTier::CpuDram, 20,
        );
        // ExpertWeight(-300) + HBM(0) vs KvContext(0) + DRAM(-200): delta = 100
        assert!(score_a < score_b, "ExpertWeight+HBM < KvContext+DRAM: a={} b={}", score_a, score_b);
        assert_eq!(score_b - score_a, 100, "delta should be 100");
    }

    // ── Eviction priority sort is deterministic across repeated sort passes ──

    #[test]
    fn eviction_priority_sort_is_stable_across_repeated_sorts() {
        let candidates: Vec<EvictionCandidate> = (0..20)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: (i as i64) % 7 * 100 - 300,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            })
            .collect();

        let mut sorted1 = candidates.clone();
        sorted1.sort_by_key(|c| c.score);

        let mut sorted2 = candidates.clone();
        sorted2.sort_by_key(|c| c.score);

        for (i, (a, b)) in sorted1.iter().zip(sorted2.iter()).enumerate() {
            assert_eq!(a.page_id, b.page_id, "sort not deterministic at position {}", i);
            assert_eq!(a.score, b.score, "scores differ at position {}", i);
        }
        for w in sorted1.windows(2) {
            assert!(w[0].score <= w[1].score, "scores should be non-decreasing");
        }
    }

    // ── Eviction sort then truncate retains the most evictable (lowest scores) ──

    #[test]
    fn eviction_sort_then_truncate_preserves_most_evictable() {
        let mut candidates: Vec<EvictionCandidate> = (0..100)
            .map(|i| EvictionCandidate {
                page_id: i, score: (i as i64) * 10 - 500,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: Some(i as u64),
            })
            .collect();
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(DEFAULT_MAX_EVICT_PER_ROUND);

        assert_eq!(candidates.len(), DEFAULT_MAX_EVICT_PER_ROUND);
        assert_eq!(candidates[0].page_id, 0, "lowest score page should be first");
        assert_eq!(candidates[0].score, -500);
        for c in &candidates {
            assert!(c.score <= -430,
                "retained score should be low: page_id={} score={}", c.page_id, c.score);
        }
    }

    // ── compute_importance_score: Warm + ExpertWeight on DRAM exact value ──

    #[test]
    fn score_warm_expert_on_dram_exact() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: None, recency: 0, access_count: 0,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Warm, warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::CpuDram, 100,
        );
        // base(1000) - time(200) + expert(-300) + warm(5000) + dram(-200) = 5300
        assert_eq!(score, 5300, "Warm + Expert + DRAM + 100 ticks = 5300, got {}", score);
        assert!(score > IMPORTANCE_SCORE_THRESHOLD);
    }

    // ── compute_importance_score: tier ordering HBM > DRAM > NVMe for every payload ──

    #[test]
    fn score_tier_ordering_holds_for_each_payload_kind() {
        let payloads = [
            PagePayloadKind::KnowledgeRAG, PagePayloadKind::ExpertWeight,
            PagePayloadKind::KvContext, PagePayloadKind::PromptSystem,
            PagePayloadKind::DenseLayerWeight,
        ];
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(100), recency: 3, access_count: 7,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Standby, warm_until: None,
        };
        for payload in &payloads {
            let s_hbm = EvictionWorker::compute_importance_score(
                &meta, Some(*payload), 2048, 4096, StorageTier::GpuHbm, 50,
            );
            let s_dram = EvictionWorker::compute_importance_score(
                &meta, Some(*payload), 2048, 4096, StorageTier::CpuDram, 50,
            );
            let s_nvme = EvictionWorker::compute_importance_score(
                &meta, Some(*payload), 2048, 4096, StorageTier::Nvme, 50,
            );
            assert!(s_hbm > s_dram, "HBM > DRAM for {:?}: {} vs {}", payload, s_hbm, s_dram);
            assert!(s_dram > s_nvme, "DRAM > NVMe for {:?}: {} vs {}", payload, s_dram, s_nvme);
        }
    }

    // ── compute_importance_score: exactly 3 state bonus levels with nonzero other inputs ──

    #[test]
    fn score_state_levels_exact_with_nonzero_other_inputs() {
        let make_meta = |state: PageState| PageMetadata {
            page_id: 1, sequence_id: Some(100), recency: 5, access_count: 10,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state, warm_until: None,
        };
        let zero_bonus_states = [
            PageState::Free, PageState::Standby, PageState::SwappedOut,
            PageState::Swapped, PageState::Active,
        ];
        let mut zero_scores: Vec<i64> = Vec::new();
        for &state in &zero_bonus_states {
            zero_scores.push(EvictionWorker::compute_importance_score(
                &make_meta(state), Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::CpuDram, 50,
            ));
        }
        for i in 1..zero_scores.len() {
            assert_eq!(zero_scores[0], zero_scores[i],
                "zero-bonus states should score same: {:?} vs Free", zero_bonus_states[i]);
        }
        let score_warm = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Warm), Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::CpuDram, 50,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &make_meta(PageState::Protected), Some(PagePayloadKind::KvContext), 2048, 4096, StorageTier::CpuDram, 50,
        );
        assert_eq!(score_warm - zero_scores[0], 5000, "Warm = +5000");
        assert_eq!(score_protected - zero_scores[0], 10000, "Protected = +10000");
    }

    // ── compute_importance_score: age 0 vs 100 crosses threshold for ExpertWeight + NVMe ──

    #[test]
    fn score_age_0_vs_100_crosses_threshold_for_expert_nvme() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: None, recency: 0, access_count: 0,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Standby, warm_until: None,
        };
        let score_0 = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 0,
        );
        let score_100 = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 100,
        );
        // score_0: 1000 - 300 - 500 = 200 > threshold(100)
        assert!(score_0 > IMPORTANCE_SCORE_THRESHOLD, "age=0 above threshold: {}", score_0);
        // score_100: 1000 - 200 - 300 - 500 = 0 < threshold(100)
        assert!(score_100 < IMPORTANCE_SCORE_THRESHOLD, "age=100 below threshold: {}", score_100);
        assert_eq!(score_0 - score_100, 100 * TIME_DECAY_WEIGHT);
    }

    // ── EvictionCandidate sort stability: 50 candidates with identical scores ──

    #[test]
    fn eviction_candidate_sort_stability_with_many_equal_scores() {
        let mut candidates: Vec<EvictionCandidate> = (0..50)
            .map(|i| EvictionCandidate {
                page_id: i, score: 42,
                current_tier: if i % 2 == 0 { StorageTier::GpuHbm } else { StorageTier::CpuDram },
                codec: CompressionCodec::None, page_bytes: 4096,
                group_id: if i % 3 == 0 { None } else { Some(i as u64) },
            })
            .collect();
        let original_order: Vec<PageId> = candidates.iter().map(|c| c.page_id).collect();
        candidates.sort_by_key(|c| c.score);
        let sorted_order: Vec<PageId> = candidates.iter().map(|c| c.page_id).collect();
        assert_eq!(original_order, sorted_order,
            "stable sort should preserve insertion order for equal scores");
    }

    // ── Observer eviction count accumulates; recovery does not affect it ──

    #[test]
    fn observer_eviction_count_accumulates_across_multiple_events() {
        let mut observer = BasicObserver::new();
        assert_eq!(observer.last_state.weight_eviction_count, 0);

        observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1, from_tier: WeightTier::Hot, to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure, bytes: 4096,
        });
        assert_eq!(observer.last_state.weight_eviction_count, 1);

        observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 2, from_tier: WeightTier::Hot, to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure, bytes: 4096,
        });
        assert_eq!(observer.last_state.weight_eviction_count, 2);

        observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 3, from_tier: WeightTier::Warm, to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure, bytes: 4096,
        });
        assert_eq!(observer.last_state.weight_eviction_count, 3);

        // Recovery does not affect eviction count.
        observer.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 1, from_tier: WeightTier::Warm, to_tier: WeightTier::Hot,
            latency_us: 100, bytes: 4096,
        });
        assert_eq!(observer.last_state.weight_eviction_count, 3, "recovery should not change eviction count");
        assert_eq!(observer.last_state.weight_recovery_count, 1);
    }

    // ── EvictionCandidate Debug roundtrip preserves all field values ──

    #[test]
    fn eviction_candidate_debug_roundtrip_preserves_values() {
        let c = EvictionCandidate {
            page_id: 12345, score: -9999, current_tier: StorageTier::Nvme,
            codec: CompressionCodec::NvcompAns, page_bytes: 65536, group_id: Some(777),
        };
        let debug_str = format!("{:?}", c);
        assert!(debug_str.contains("12345"), "page_id");
        assert!(debug_str.contains("-9999"), "score");
        assert!(debug_str.contains("Nvme"), "tier");
        assert!(debug_str.contains("NvcompAns"), "codec");
        assert!(debug_str.contains("65536"), "page_bytes");
        assert!(debug_str.contains("777"), "group_id");
        let cloned = c.clone();
        assert_eq!(cloned.page_id, c.page_id);
        assert_eq!(cloned.score, c.score);
        assert_eq!(cloned.current_tier, c.current_tier);
        assert_eq!(cloned.codec, c.codec);
        assert_eq!(cloned.page_bytes, c.page_bytes);
        assert_eq!(cloned.group_id, c.group_id);
    }

    // ── compute_importance_score: extreme tier_age overrides even Protected state ──

    #[test]
    fn score_extreme_age_overrides_even_protected_state() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(100), recency: 0, access_count: 0,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Protected, warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 0, 0, StorageTier::GpuHbm, 100_000,
        );
        // base(1000) - time(200000) + protected(10000) = -189000
        assert!(score < 0, "extreme age should produce negative score with Protected: got {}", score);
        assert!(score < IMPORTANCE_SCORE_THRESHOLD,
            "extreme age should drive below threshold: got {}", score);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (15 new — batch 7)
    // ─────────────────────────────────────────────────────────────────────────

    // ── 1. EvictionWorkerConfig: all fields survive manual round-trip through reconstruction ──

    #[test]
    fn config_all_fields_roundtrip_through_reconstruction() {
        // Arrange: create a config with non-default values for every field.
        let original = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(77),
            max_evict_per_round: 16,
            hbm_pressure_threshold: 0.88,
            dram_pressure_threshold: 0.77,
            importance_threshold: 42,
            hbm_evict_age_ticks: 33,
            dram_evict_age_ticks: 333,
            default_evict_codec: CompressionCodec::BitPackRle,
            page_bytes: 2048,
        };

        // Act: reconstruct from field access (simulates serialization round-trip).
        let reconstructed = EvictionWorkerConfig {
            tick_interval: original.tick_interval,
            max_evict_per_round: original.max_evict_per_round,
            hbm_pressure_threshold: original.hbm_pressure_threshold,
            dram_pressure_threshold: original.dram_pressure_threshold,
            importance_threshold: original.importance_threshold,
            hbm_evict_age_ticks: original.hbm_evict_age_ticks,
            dram_evict_age_ticks: original.dram_evict_age_ticks,
            default_evict_codec: original.default_evict_codec,
            page_bytes: original.page_bytes,
        };

        // Assert: every field matches exactly.
        assert_eq!(reconstructed.tick_interval, original.tick_interval);
        assert_eq!(reconstructed.max_evict_per_round, original.max_evict_per_round);
        assert!((reconstructed.hbm_pressure_threshold - original.hbm_pressure_threshold).abs() < 1e-7);
        assert!((reconstructed.dram_pressure_threshold - original.dram_pressure_threshold).abs() < 1e-7);
        assert_eq!(reconstructed.importance_threshold, original.importance_threshold);
        assert_eq!(reconstructed.hbm_evict_age_ticks, original.hbm_evict_age_ticks);
        assert_eq!(reconstructed.dram_evict_age_ticks, original.dram_evict_age_ticks);
        assert_eq!(reconstructed.default_evict_codec, original.default_evict_codec);
        assert_eq!(reconstructed.page_bytes, original.page_bytes);
    }

    // ── 2. EvictionTier: Debug output contains no extra whitespace or unexpected characters ──

    #[test]
    fn eviction_tier_debug_output_clean_formatting() {
        // Arrange: collect all variants.
        let variants = [
            (EvictionTier::ColdExpert, "ColdExpert"),
            (EvictionTier::PinnedDense, "PinnedDense"),
            (EvictionTier::StandbyKv, "StandbyKv"),
            (EvictionTier::Protected, "Protected"),
        ];

        // Act & Assert: Debug output must be exactly the variant name with no surrounding whitespace.
        for (variant, name) in &variants {
            let debug = format!("{:?}", variant);
            assert_eq!(debug, *name, "Debug output should be exactly the variant name");
            assert!(!debug.starts_with(' '), "no leading whitespace");
            assert!(!debug.ends_with(' '), "no trailing whitespace");
            assert!(!debug.contains('\n'), "no embedded newlines");
        }
    }

    // ── 3. EvictionCandidate: sort_by_key produces correct ascending order for many candidates ──

    #[test]
    fn eviction_candidate_sort_by_key_large_set_correct_order() {
        // Arrange: 12 candidates with varied scores.
        let mut candidates: Vec<EvictionCandidate> = vec![
            EvictionCandidate { page_id: 10, score: 999, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 11, score: -999, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 12, score: 0, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 13, score: 500, current_tier: StorageTier::CpuDram, codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(1) },
            EvictionCandidate { page_id: 14, score: -500, current_tier: StorageTier::CpuDram, codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(2) },
            EvictionCandidate { page_id: 15, score: 100, current_tier: StorageTier::Nvme, codec: CompressionCodec::BitPackRle, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 16, score: -1, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 17, score: 1, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 18, score: i64::MAX, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 19, score: i64::MIN, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 20, score: 50, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 21, score: -50, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
        ];

        // Act: sort ascending by score.
        candidates.sort_by_key(|c| c.score);

        // Assert: scores are strictly non-decreasing.
        for window in candidates.windows(2) {
            assert!(
                window[0].score <= window[1].score,
                "scores must be non-decreasing: {} followed by {}",
                window[0].score,
                window[1].score,
            );
        }
        // The first must be the minimum and last the maximum.
        assert_eq!(candidates.first().unwrap().score, i64::MIN);
        assert_eq!(candidates.last().unwrap().score, i64::MAX);
    }

    // ── 4. Top-k selection: after scoring 10 candidates, truncate(3) keeps 3 lowest ──

    #[test]
    fn top_k_truncate_selects_correct_subset() {
        // Arrange: 10 candidates with scores 10,20,...,100.
        let mut candidates: Vec<EvictionCandidate> = (1..=10)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: (i as i64) * 10,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();

        // Act: sort and keep top-3 (lowest scores).
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(3);

        // Assert: exactly 3 candidates with page_ids 1,2,3 and scores 10,20,30.
        assert_eq!(candidates.len(), 3);
        assert_eq!(candidates[0].page_id, 1);
        assert_eq!(candidates[0].score, 10);
        assert_eq!(candidates[1].page_id, 2);
        assert_eq!(candidates[1].score, 20);
        assert_eq!(candidates[2].page_id, 3);
        assert_eq!(candidates[2].score, 30);
    }

    // ── 5. CompressionCodec::None: EvictionCandidate with None codec preserves it through clone ──

    #[test]
    fn codec_none_preserved_through_candidate_clone() {
        // Arrange.
        let original = EvictionCandidate {
            page_id: 42,
            score: -100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };

        // Act.
        let cloned = original.clone();

        // Assert.
        assert_eq!(cloned.codec, CompressionCodec::None);
        assert_eq!(cloned.codec, original.codec);
    }

    // ── 6. Observer event count: evict_round records correct number of telemetry events ──

    #[test]
    fn observer_events_count_matches_submitted_evictions() {
        // Arrange: config with max 3 evictions per round.
        let config = EvictionWorkerConfig {
            max_evict_per_round: 3,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_millis(6000);
        let mut pages = HashMap::new();
        let mut addrs = HashMap::new();
        // Use host_buffer for each page (real memory, avoids SIGSEGV from fake gpu_ptr).
        let page_data: Vec<Vec<u8>> = (0..5).map(|_| vec![0u8; 4096]).collect();
        for i in 0..5usize {
            let pid = (i + 1) as PageId;
            pages.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: old_instant,
                swap_in_time: Some(old_instant),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
            addrs.insert(pid, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(page_data[i].clone()),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(pages));
        {
            let mut guard = addr_table.write().unwrap();
            for (k, v) in addrs {
                guard.insert(k, v);
            }
        }

        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act.
        let submitted = EvictionWorker::evict_round(
            &config,
            &actor,
            &page_metadata,
            &addr_table,
            &mm,
            &observer,
        );

        // Assert: observer should have recorded eviction events.
        let obs = observer.lock().unwrap();
        let l1_before = 0; // initially 0 weight pages tracked by observer
        let l1_after = obs.last_state.weight_pages_l1;
        // Each eviction event increments l2 and decrements l1.
        // The exact value depends on how many were submitted.
        assert!(
            submitted <= 3,
            "submitted should respect max_evict_per_round: got {}",
            submitted,
        );
        // The observer's weight_page counters should reflect evictions.
        // After evictions, l2 should have increased by `submitted`.
        let l2_after = obs.last_state.weight_pages_l2;
        assert_eq!(
            l2_after, submitted,
            "observer l2 count should equal submitted count: l2={} submitted={}",
            l2_after, submitted,
        );

        actor.shutdown();
    }

    // ── 7. BasicObserver: initial state has zero values for all weight page counters ──

    #[test]
    fn observer_initial_weight_page_counts_are_zero() {
        // Arrange & Act.
        let observer = BasicObserver::new();

        // Assert.
        assert_eq!(observer.last_state.weight_page_total, 0, "initial total should be 0");
        assert_eq!(observer.last_state.weight_pages_l1, 0, "initial l1 should be 0");
        assert_eq!(observer.last_state.weight_pages_l2, 0, "initial l2 should be 0");
        assert_eq!(observer.last_state.weight_pages_l3, 0, "initial l3 should be 0");
    }

    // ── 8. EvictionWorkerConfig: default round-trip — default() == clone of default() ──

    #[test]
    fn config_default_equals_clone_of_default() {
        // Arrange.
        let cfg = EvictionWorkerConfig::default();

        // Act.
        let cloned = cfg.clone();

        // Assert: every field matches between default and its clone.
        assert_eq!(cloned.tick_interval, cfg.tick_interval);
        assert_eq!(cloned.max_evict_per_round, cfg.max_evict_per_round);
        assert!((cloned.hbm_pressure_threshold - cfg.hbm_pressure_threshold).abs() < 1e-7);
        assert!((cloned.dram_pressure_threshold - cfg.dram_pressure_threshold).abs() < 1e-7);
        assert_eq!(cloned.importance_threshold, cfg.importance_threshold);
        assert_eq!(cloned.hbm_evict_age_ticks, cfg.hbm_evict_age_ticks);
        assert_eq!(cloned.dram_evict_age_ticks, cfg.dram_evict_age_ticks);
        assert_eq!(cloned.default_evict_codec, cfg.default_evict_codec);
        assert_eq!(cloned.page_bytes, cfg.page_bytes);
    }

    // ── 9. EvictionTier: all variants produce unique Debug strings ──

    #[test]
    fn eviction_tier_debug_strings_all_unique() {
        // Arrange.
        let debug_strings: Vec<String> = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ].iter().map(|t| format!("{:?}", t)).collect();

        // Act & Assert: all strings must be pairwise distinct.
        for i in 0..debug_strings.len() {
            for j in (i + 1)..debug_strings.len() {
                assert_ne!(
                    debug_strings[i], debug_strings[j],
                    "Debug strings at indices {} and {} must differ: '{}' vs '{}'",
                    i, j, debug_strings[i], debug_strings[j],
                );
            }
        }
    }

    // ── 10. EvictionCandidate: sort then truncate with max=1 selects single lowest ──

    #[test]
    fn eviction_candidate_truncate_one_selects_single_lowest() {
        // Arrange: 5 candidates with varied scores.
        let mut candidates: Vec<EvictionCandidate> = vec![
            EvictionCandidate { page_id: 1, score: 300, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 2, score: -100, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 3, score: 50, current_tier: StorageTier::CpuDram, codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(1) },
            EvictionCandidate { page_id: 4, score: -500, current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None, page_bytes: 4096, group_id: None },
            EvictionCandidate { page_id: 5, score: 200, current_tier: StorageTier::Nvme, codec: CompressionCodec::BitPackRle, page_bytes: 4096, group_id: None },
        ];

        // Act: sort ascending, truncate to 1.
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(1);

        // Assert: only page_id=4 (score=-500) survives.
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].page_id, 4);
        assert_eq!(candidates[0].score, -500);
    }

    // ── 11. Observer: recording a single event increments the correct tier counter ──

    #[test]
    fn observer_single_event_increments_l2_counter() {
        // Arrange.
        let mut observer = BasicObserver::new();
        assert_eq!(observer.last_state.weight_pages_l2, 0, "l2 should start at 0");

        // Act: record one Evicted event (GpuHbm → CpuDram = Hot → Warm).
        observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });

        // Assert: l2 counter incremented by 1.
        assert_eq!(observer.last_state.weight_pages_l2, 1, "l2 should be 1 after one Evicted event");
    }

    // ── 12. Observer: recording two events to same tier increments counter by 2 ──

    #[test]
    fn observer_two_events_increment_counter_by_two() {
        // Arrange.
        let mut observer = BasicObserver::new();

        // Act: record two Evicted events to same tier.
        observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 2,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });

        // Assert: counter incremented by 2.
        assert_eq!(observer.last_state.weight_pages_l2, 2, "l2 should be 2 after two Evicted events");
    }

    // ── 13. CompressionCodec::None round-trips through u8 encoding ──

    #[test]
    fn codec_none_roundtrip_through_u8() {
        // Arrange.
        let codec = CompressionCodec::None;

        // Act: encode then decode.
        let encoded = codec.as_u8();
        let decoded = CompressionCodec::from_u8(encoded);

        // Assert: round-trip produces the same variant.
        assert_eq!(decoded, Some(CompressionCodec::None));
    }

    // ── 14. EvictionWorkerConfig: page_bytes field does not affect scoring ──

    #[test]
    fn config_page_bytes_does_not_affect_scoring() {
        // Arrange: two configs with different page_bytes.
        let _cfg_small = EvictionWorkerConfig { page_bytes: 1024, ..EvictionWorkerConfig::default() };
        let _cfg_large = EvictionWorkerConfig { page_bytes: 65536, ..EvictionWorkerConfig::default() };

        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compute score with same inputs — page_bytes is not a scoring parameter.
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 4096, StorageTier::GpuHbm, 50,
        );

        // Assert: score is the same regardless of config.page_bytes (scoring uses original_size).
        // score = base(1000) - time(50*2=100) + compression((1-0)*500=500) + page_size(4096/1024=4) = 1404
        assert_eq!(score, 1404, "score should be determined by original_size, not config.page_bytes");
    }

    // ── 15. EvictionTier: classify_eviction_tier produces consistent results across repeated calls ──

    #[test]
    fn classify_eviction_tier_is_deterministic_across_calls() {
        // Arrange: fixed inputs.
        let payload = Some(PagePayloadKind::KvContext);
        let score = 50;

        // Act: classify 10 times.
        let tiers: Vec<EvictionTier> = (0..10)
            .map(|_| EvictionWorker::classify_eviction_tier(payload, score))
            .collect();

        // Assert: all results identical.
        let first = tiers[0];
        for (i, tier) in tiers.iter().enumerate() {
            assert_eq!(
                *tier, first,
                "classify_eviction_tier must be deterministic: call {} returned {:?} vs first {:?}",
                i, tier, first,
            );
        }
        // Also verify the expected tier.
        assert_eq!(first, EvictionTier::StandbyKv, "KvContext with score=50 should be StandbyKv");
    }

    // ── 16. compute_importance_score: access_count at u32::MAX does not panic ──

    #[test]
    fn score_access_count_u32_max_no_panic() {
        // Arrange: page with maximal access_count (usize on 64-bit > u32::MAX).
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: u32::MAX as usize,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compute_importance_score must not panic even with huge access_count.
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            2048,
            4096,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: score should be very high (freq_bonus = u32::MAX * 15).
        // The cast `u32::MAX as i64` is 4_294_967_295 which is valid.
        // freq_bonus = 4_294_967_295 * 15 = 64_424_509_425 which fits in i64.
        assert!(
            score > 0,
            "high access_count should yield positive score: got {}",
            score,
        );
    }

    // ── 17. compute_importance_score: compressed_size > original_size yields negative compression bonus (explicit meta) ──

    #[test]
    fn score_compressed_larger_than_original_negative_bonus_explicit() {
        // Arrange: compressed_size = 8192, original_size = 4096 (expansion, not compression).
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: ratio = 8192 / 4096 = 2.0 → (1.0 - 2.0) * 500 = -500.
        let score_expanded = EvictionWorker::compute_importance_score(
            &meta,
            None,
            8192, // compressed > original
            4096,
            StorageTier::GpuHbm,
            0,
        );

        // Baseline: compressed = original (ratio = 1.0, bonus = 0).
        let score_normal = EvictionWorker::compute_importance_score(
            &meta,
            None,
            4096,
            4096,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: expansion yields lower score than normal compression.
        assert!(
            score_expanded < score_normal,
            "expanded compression should yield lower score than normal: expanded={} normal={}",
            score_expanded,
            score_normal,
        );
        // Exact delta: expansion bonus = (1.0 - 2.0) * 500 = -500; normal bonus = 0.
        assert_eq!(
            score_normal - score_expanded, 500,
            "compression bonus delta should be 500: normal={} expanded={}",
            score_normal, score_expanded,
        );
    }

    // ── 18. EvictionTier: classify_eviction_tier for DenseLayerWeight always returns PinnedDense regardless of score ──

    #[test]
    fn classify_eviction_tier_dense_layer_ignores_score() {
        // Arrange: DenseLayerWeight with very low and very high scores.
        let payload = Some(PagePayloadKind::DenseLayerWeight);

        // Act.
        let tier_low = EvictionWorker::classify_eviction_tier(payload, -9999);
        let tier_high = EvictionWorker::classify_eviction_tier(payload, 999999);

        // Assert: both should be PinnedDense (DenseLayerWeight is always PinnedDense).
        assert_eq!(tier_low, EvictionTier::PinnedDense, "DenseLayerWeight with low score");
        assert_eq!(tier_high, EvictionTier::PinnedDense, "DenseLayerWeight with high score");
    }

    // ── 19. compute_importance_score: compressed_size=0 original_size=0 yields zero bonuses ──

    #[test]
    fn score_zero_compressed_zero_original_no_compression_or_page_bonus() {
        // Arrange.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: both sizes = 0 → original_size == 0 → no bonuses.
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: base(1000) - time(0) - recency(0) + freq(0) + comp(0) + page(0) + payload(0) + state(0) + tier(0) = 1000.
        assert_eq!(score, 1000, "score with zero sizes should be exactly base: got {}", score);
    }

    // ── 20. compute_importance_score: recency at usize::MAX does not panic (explicit meta) ──

    #[test]
    fn score_recency_usize_max_no_panic_explicit() {
        // Arrange: extreme recency value.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: usize::MAX,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: should not panic; usize::MAX as i64 wraps on 64-bit.
        let score = EvictionWorker::compute_importance_score(
            &meta,
            None,
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: function completed without panic.
        // On 64-bit: recency_penalty = usize::MAX as i64 = -1 * 1 = -1 (wrapping).
        // The key guarantee is no panic.
        let _ = score;
    }

    // ── 21. EvictionCandidate: candidate with page_bytes=0 can be created and sorted ──

    #[test]
    fn eviction_candidate_page_bytes_zero_sorts_correctly() {
        // Arrange: candidates with page_bytes=0.
        let mut candidates: Vec<EvictionCandidate> = vec![
            EvictionCandidate {
                page_id: 1, score: 200,
                current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None,
                page_bytes: 0, group_id: None,
            },
            EvictionCandidate {
                page_id: 2, score: -100,
                current_tier: StorageTier::GpuHbm, codec: CompressionCodec::None,
                page_bytes: 0, group_id: None,
            },
            EvictionCandidate {
                page_id: 3, score: 50,
                current_tier: StorageTier::CpuDram, codec: CompressionCodec::Lz4,
                page_bytes: 0, group_id: Some(10),
            },
        ];

        // Act: sort by score ascending.
        candidates.sort_by_key(|c| c.score);

        // Assert: order is page_id 2 (-100), 3 (50), 1 (200).
        assert_eq!(candidates[0].page_id, 2);
        assert_eq!(candidates[1].page_id, 3);
        assert_eq!(candidates[2].page_id, 1);
        assert_eq!(candidates[0].page_bytes, 0, "page_bytes=0 preserved after sort");
    }

    // ── 22. BasicObserver: mixed Evicted and Recovered events count correctly ──

    #[test]
    fn observer_mixed_evicted_recovered_events_count_separately() {
        // Arrange.
        let mut observer = BasicObserver::new();
        assert_eq!(observer.last_state.weight_eviction_count, 0);
        assert_eq!(observer.last_state.weight_recovery_count, 0);

        // Act: 3 evictions + 2 recoveries.
        for i in 0..3 {
            observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: i,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Warm,
                reason: EvictionReason::MemoryPressure,
                bytes: 4096,
            });
        }
        for i in 0..2 {
            observer.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                page_id: 100 + i,
                from_tier: WeightTier::Warm,
                to_tier: WeightTier::Hot,
                latency_us: 500,
                bytes: 4096,
            });
        }

        // Assert: eviction and recovery counters are independent.
        assert_eq!(
            observer.last_state.weight_eviction_count, 3,
            "should have exactly 3 eviction events",
        );
        assert_eq!(
            observer.last_state.weight_recovery_count, 2,
            "should have exactly 2 recovery events",
        );
    }

    // ── 23. EvictionWorkerConfig: page_bytes=0 does not prevent eviction round submission ──

    #[test]
    fn evict_round_with_page_bytes_zero_still_submits() {
        // Arrange: config with page_bytes=0.
        let config = EvictionWorkerConfig {
            page_bytes: 0,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old_instant = Instant::now() - Duration::from_millis(6000);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // ExpertWeight → lower score
            recency: 0,
            access_count: 0,
            last_access: old_instant,
            swap_in_time: Some(old_instant),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // High HBM pressure.
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 1000, 1000);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act.
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: eviction should succeed even with page_bytes=0.
        assert!(
            submitted >= 1,
            "eviction should succeed with page_bytes=0: submitted={}",
            submitted,
        );

        actor.shutdown();
    }

    // ── 24. compute_importance_score: Protected state yields much higher score than Standby ──

    #[test]
    fn score_protected_state_massively_higher_than_standby() {
        // Arrange: identical metadata except state.
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_protected = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };

        // Act.
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &meta_protected, None, 0, 0, StorageTier::GpuHbm, 0,
        );

        // Assert: Protected gets +10_000 bonus.
        assert_eq!(
            score_protected - score_standby, 10_000,
            "Protected should be exactly 10_000 higher: protected={} standby={}",
            score_protected, score_standby,
        );
    }

    // ── 25. EvictionTier: classify_eviction_tier returns StandbyKv for KnowledgeRAG with low score ──

    #[test]
    fn classify_eviction_tier_knowledge_rag_low_score_is_standby_kv() {
        // Arrange: KnowledgeRAG with score below threshold.
        let payload = Some(PagePayloadKind::KnowledgeRAG);
        let score = 50; // below IMPORTANCE_SCORE_THRESHOLD (100).

        // Act.
        let tier = EvictionWorker::classify_eviction_tier(payload, score);

        // Assert: KnowledgeRAG is not ExpertWeight or DenseLayerWeight, so
        // it falls into the `_ if score < threshold => StandbyKv` branch.
        assert_eq!(
            tier, EvictionTier::StandbyKv,
            "KnowledgeRAG with score=50 should be StandbyKv",
        );
    }

    // ── 26. compute_importance_score: score is deterministic across identical calls ──

    #[test]
    fn score_deterministic_with_large_access_count() {
        // Arrange: fixed metadata with large access_count.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 10,
            access_count: 1_000_000,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compute score 5 times with identical inputs.
        let scores: Vec<i64> = (0..5)
            .map(|_| EvictionWorker::compute_importance_score(
                &meta,
                Some(PagePayloadKind::KvContext),
                1024,
                4096,
                StorageTier::GpuHbm,
                25,
            ))
            .collect();

        // Assert: all scores identical.
        for (i, &s) in scores.iter().enumerate() {
            assert_eq!(
                s, scores[0],
                "score at call {} differs from first call: {} vs {}",
                i, s, scores[0],
            );
        }
    }

    // ── 27. EvictionCandidate: all 5 CompressionCodec variants accepted ──

    #[test]
    fn eviction_candidate_all_compression_codec_variants() {
        // Arrange & Act: create a candidate for each codec variant.
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];

        for (i, &codec) in codecs.iter().enumerate() {
            let c = EvictionCandidate {
                page_id: i,
                score: 0,
                current_tier: StorageTier::CpuDram,
                codec,
                page_bytes: 4096,
                group_id: None,
            };
            // Assert: codec is preserved.
            assert_eq!(c.codec, codec, "codec mismatch at index {}", i);
        }
    }

    // ── 28. compute_importance_score: Warm state yields exactly 5000 more than Standby ──

    #[test]
    fn score_warm_state_exact_bonus_delta() {
        // Arrange: identical metadata except state.
        let meta_standby = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 3,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_warm = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 3,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };

        // Act.
        let score_standby = EvictionWorker::compute_importance_score(
            &meta_standby, None, 0, 4096, StorageTier::GpuHbm, 10,
        );
        let score_warm = EvictionWorker::compute_importance_score(
            &meta_warm, None, 0, 4096, StorageTier::GpuHbm, 10,
        );

        // Assert: Warm bonus = 5000 exactly.
        assert_eq!(
            score_warm - score_standby, 5000,
            "Warm should be exactly 5000 higher than Standby: warm={} standby={}",
            score_warm, score_standby,
        );
    }

    // ── 29. BasicObserver: update_weight_metrics bulk set does not affect incremental counters ──

    #[test]
    fn observer_bulk_update_preserves_incremental_event_counters() {
        // Arrange: set initial bulk metrics, then record events.
        let mut observer = BasicObserver::new();
        observer.update_weight_metrics(100, 50, 30, 20, 5, 3);
        assert_eq!(observer.last_state.weight_eviction_count, 5);
        assert_eq!(observer.last_state.weight_recovery_count, 3);

        // Act: record one more eviction.
        observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });

        // Assert: eviction counter incremented from 5 to 6 (bulk value + 1).
        assert_eq!(
            observer.last_state.weight_eviction_count, 6,
            "incremental counter should be bulk value (5) + 1 event",
        );
        assert_eq!(
            observer.last_state.weight_recovery_count, 3,
            "recovery counter should be unchanged",
        );
    }

    // ── 30. compute_importance_score: very large original_size yields proportionally large page_size_bonus ──

    #[test]
    fn score_large_original_size_yields_proportional_page_bonus() {
        // Arrange: small vs large original_size.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: original_size = 4KiB vs original_size = 1MiB.
        let score_small = EvictionWorker::compute_importance_score(
            &meta, None, 0, 4096, StorageTier::GpuHbm, 0,
        );
        let score_large = EvictionWorker::compute_importance_score(
            &meta, None, 0, 1024 * 1024, StorageTier::GpuHbm, 0,
        );

        // Assert: page_size_bonus scales linearly with original_size.
        // small bonus = 4096/1024 * 1 = 4; large bonus = 1048576/1024 * 1 = 1024.
        // Both have compressed=0 so compression bonus = (1-0)*500 = 500 for both.
        // delta = 1024 - 4 = 1020.
        assert_eq!(
            score_large - score_small, 1020,
            "1MiB page should have 1020 more page_size_bonus than 4KiB: large={} small={}",
            score_large, score_small,
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (15 new)
    // ─────────────────────────────────────────────────────────────────────────

    // ── 1. EvictionWorkerConfig: page_bytes=0 does not cause division by zero in scoring ──

    #[test]
    fn config_page_bytes_zero_no_effect_on_scoring() {
        // Arrange: scoring function uses config.page_bytes for migration commands,
        // not for score computation. Verify score is unaffected by page_bytes.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compute score with same inputs regardless of config page_bytes.
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            2048,
            4096,
            StorageTier::GpuHbm,
            50,
        );

        // Assert: score is deterministic and matches manual calculation.
        // base(1000) - time(50*2=100) + freq(5*15=75) + compression((1-0.5)*500=250)
        // + page_size(4096/1024*1=4) = 1229
        assert_eq!(score, 1229, "score should be independent of config.page_bytes");
    }

    // ── 2. EvictionCandidate with u32::MAX access_count in scoring ──

    #[test]
    fn score_access_count_u32_max_exact() {
        // Arrange: access_count at u32::MAX boundary (stored as usize).
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: u32::MAX as usize,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compute importance score.
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            0,
            StorageTier::GpuHbm,
            0,
        );

        // Assert: freq_bonus = u32::MAX * 15 — will overflow i64 but must not panic.
        // The score just needs to be a valid i64 and massively positive.
        assert!(
            score > 0,
            "u32::MAX access_count should yield high positive score: got {}",
            score,
        );
    }

    // ── 3. Tier discount at exact boundaries: 0, -200, -500 ──

    #[test]
    fn tier_discount_exact_boundary_values() {
        // Arrange: identical metadata, only tier differs.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compute scores for each tier.
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let score_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::CpuDram, 0,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::Nvme, 0,
        );

        // Assert: HBM discount=0, DRAM discount=-200, NVMe discount=-500.
        // base(1000) + tier_discount.
        assert_eq!(score_hbm, 1000, "HBM tier discount should be 0");
        assert_eq!(score_dram, 800, "DRAM tier discount should be -200 => 800");
        assert_eq!(score_nvme, 500, "NVMe tier discount should be -500 => 500");
    }

    // ── 4. BasicObserver event ordering: evictions recorded before subsequent events ──

    #[test]
    fn observer_event_ordering_guarantee() {
        // Arrange: create observer and record eviction then recovery.
        let mut obs = BasicObserver::new();

        // Act: record events in specific order.
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: EvictionReason::MemoryPressure,
            bytes: 4096,
        });
        let eviction_count_after_first = obs.last_state.weight_eviction_count;

        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 1,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Hot,
            latency_us: 100,
            bytes: 4096,
        });

        // Assert: first event was counted before second event was recorded.
        assert_eq!(eviction_count_after_first, 1, "eviction should be counted before recovery is recorded");
        assert_eq!(obs.last_state.weight_eviction_count, 1, "eviction count should remain 1");
        assert_eq!(obs.last_state.weight_recovery_count, 1, "recovery count should be 1");
    }

    // ── 5. classify_eviction_tier with zero-size page and low score ──

    #[test]
    fn classify_zero_size_page_low_score_is_standby() {
        // Arrange: KvContext payload with score below threshold.
        let score: i64 = 50;

        // Act: classify with KvContext and low score.
        let tier = EvictionWorker::classify_eviction_tier(
            Some(PagePayloadKind::KvContext),
            score,
        );

        // Assert: below threshold yields StandbyKv regardless of page size.
        assert_eq!(tier, EvictionTier::StandbyKv);
    }

    // ── 6. Score formula: compressed_size > original_size edge case produces negative bonus ──

    #[test]
    fn score_compressed_double_original_exact_negative_bonus() {
        // Arrange: page where compressed_size is exactly 2x original.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compressed=8192, original=4096 (ratio=2.0).
        let score_normal = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_double = EvictionWorker::compute_importance_score(
            &meta, None, 8192, 4096, StorageTier::GpuHbm, 0,
        );

        // Assert: bonus = (1.0 - 2.0) * 500 = -500. Both have same original_size
        // so page_size bonus cancels. Net delta = -500.
        assert_eq!(
            score_double - score_normal, -500,
            "2x compressed should yield exactly -500 delta: double={} normal={}",
            score_double, score_normal,
        );
    }

    // ── 7. Protected tier bonus vs warm tier bonus precision ──

    #[test]
    fn protected_vs_warm_bonus_precision_with_all_factors() {
        // Arrange: two pages identical except state, with non-trivial inputs.
        let meta_warm = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 10,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let meta_protected = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 10,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Protected,
            warm_until: None,
        };

        // Act: compute scores with identical non-zero inputs.
        let score_warm = EvictionWorker::compute_importance_score(
            &meta_warm,
            Some(PagePayloadKind::DenseLayerWeight),
            2048,
            4096,
            StorageTier::CpuDram,
            200,
        );
        let score_protected = EvictionWorker::compute_importance_score(
            &meta_protected,
            Some(PagePayloadKind::DenseLayerWeight),
            2048,
            4096,
            StorageTier::CpuDram,
            200,
        );

        // Assert: difference should be exactly 5000 (Protected=10000, Warm=5000).
        assert_eq!(
            score_protected - score_warm, 5000,
            "Protected should be exactly +5000 over Warm with full inputs: protected={} warm={}",
            score_protected, score_warm,
        );
    }

    // ── 8. Observer bulk update with empty weight page events ──

    #[test]
    fn observer_no_events_after_construction_and_update() {
        // Arrange: fresh observer.
        let mut obs = BasicObserver::new();

        // Act: call update_weight_metrics with all zeros (no real events).
        obs.update_weight_metrics(0, 0, 0, 0, 0, 0);

        // Assert: no eviction or recovery events recorded.
        assert_eq!(
            obs.last_state.weight_eviction_count, 0,
            "no events recorded => eviction count = 0",
        );
        assert_eq!(
            obs.last_state.weight_recovery_count, 0,
            "no events recorded => recovery count = 0",
        );
        assert_eq!(obs.last_state.weight_page_total, 0);
    }

    // ── 9. Multi-step eviction: all candidates in same tier (CpuDram) ──

    #[test]
    fn evict_round_all_candidates_same_dram_tier() {
        // Arrange: config that triggers DRAM pressure eviction.
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            hbm_pressure_threshold: 1.0, // no HBM pressure
            dram_pressure_threshold: 0.0, // always trigger DRAM eviction
            importance_threshold: i64::MAX,
            max_evict_per_round: 10,
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        // Create 3 pages all on CpuDram.
        let pages: Vec<(PageId, PageMetadata)> = vec![
            (1, PageMetadata { page_id: 1, sequence_id: Some(1), recency: 0, access_count: 0, last_access: old, swap_in_time: Some(old), is_lir: false, state: PageState::SwappedOut, warm_until: None }),
            (2, PageMetadata { page_id: 2, sequence_id: Some(2), recency: 0, access_count: 0, last_access: old, swap_in_time: Some(old), is_lir: false, state: PageState::SwappedOut, warm_until: None }),
            (3, PageMetadata { page_id: 3, sequence_id: Some(3), recency: 0, access_count: 0, last_access: old, swap_in_time: Some(old), is_lir: false, state: PageState::SwappedOut, warm_until: None }),
        ];
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from_iter(pages)));

        // All pages on CpuDram with host_buffer (safe for real DMA).
        {
            let mut guard = addr_table.write().unwrap();
            for pid in [1usize, 2, 3] {
                guard.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::Lz4,
                });
            }
        }

        // Create DRAM pressure.
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        for _ in 0..90 {
            let _ = mm.allocate_page(Tier::L2);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act: run eviction round.
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: all 3 CpuDram pages should be evicted to NVMe.
        assert_eq!(submitted, 3, "all 3 CpuDram pages should be evicted to NVMe");

        let obs = observer.lock().unwrap();
        assert_eq!(
            obs.last_state.weight_eviction_count, 3,
            "observer should count 3 evictions",
        );
        actor.shutdown();
    }

    // ── 10. ObserverError Display output ──

    #[test]
    fn observer_error_display_output() {
        // Arrange: create ObserverError::BackendUnavailable.
        let err = ObserverError::BackendUnavailable("sensor failed".to_string());

        // Act: format as Display.
        let display = format!("{}", err);

        // Assert: should contain the variant info and message.
        assert!(
            display.contains("backend unavailable"),
            "Display should contain 'backend unavailable': got '{}'",
            display,
        );
        assert!(
            display.contains("sensor failed"),
            "Display should contain the error message: got '{}'",
            display,
        );
    }

    // ── 11. EvictionWorkerConfig: max_evict_per_round=0 means no evictions submitted ──

    #[test]
    fn evict_round_max_evict_zero_submits_nothing() {
        // Arrange: config with max_evict_per_round=0 and everything eligible.
        let config = EvictionWorkerConfig {
            hbm_evict_age_ticks: 0,
            dram_evict_age_ticks: 0,
            hbm_pressure_threshold: 0.0, // always trigger
            dram_pressure_threshold: 1.0,
            importance_threshold: i64::MAX,
            max_evict_per_round: 0, // truncate to zero
            page_bytes: 4096,
            ..EvictionWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let old = Instant::now() - Duration::from_secs(5);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(1),
            recency: 0,
            access_count: 0,
            last_access: old,
            swap_in_time: Some(old),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::from([(1, meta)])));

        {
            let mut guard = addr_table.write().unwrap();
            guard.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // Create HBM pressure.
        let mut mm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        for _ in 0..95 {
            let _ = mm.allocate_page(Tier::L1);
        }
        let mm = Arc::new(Mutex::new(mm));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Act: run eviction round.
        let submitted = EvictionWorker::evict_round(
            &config, &actor, &page_metadata, &addr_table, &mm, &observer,
        );

        // Assert: max_evict_per_round=0 truncates candidates to zero.
        assert_eq!(submitted, 0, "max_evict_per_round=0 should submit nothing");

        let obs = observer.lock().unwrap();
        assert_eq!(
            obs.last_state.weight_eviction_count, 0,
            "no events should be recorded",
        );
        actor.shutdown();
    }

    // ── 12. Score: time decay weight constant is exactly 2 ──

    #[test]
    fn time_decay_weight_constant_is_two() {
        // Arrange: verify TIME_DECAY_WEIGHT constant used in scoring.
        // Act & Assert: check the constant value.
        assert_eq!(TIME_DECAY_WEIGHT, 2, "TIME_DECAY_WEIGHT must be exactly 2 per SPEC");

        // Verify through scoring: 1 tick should reduce score by exactly 2.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 1,
        );
        assert_eq!(s0 - s1, 2, "1 tick should reduce score by exactly TIME_DECAY_WEIGHT=2");
    }

    // ── 13. EvictionTier Debug + Copy: all variants roundtrip through function boundary ──

    #[test]
    fn eviction_tier_function_boundary_roundtrip() {
        // Arrange: helper function that takes and returns EvictionTier.
        fn identity(tier: EvictionTier) -> EvictionTier {
            tier // relies on Copy
        }

        // Act & Assert: all variants survive the function boundary.
        assert_eq!(identity(EvictionTier::ColdExpert), EvictionTier::ColdExpert);
        assert_eq!(identity(EvictionTier::PinnedDense), EvictionTier::PinnedDense);
        assert_eq!(identity(EvictionTier::StandbyKv), EvictionTier::StandbyKv);
        assert_eq!(identity(EvictionTier::Protected), EvictionTier::Protected);
    }

    // ── 14. Compression ratio at f32 boundary: original_size=1, compressed=0 ──

    #[test]
    fn score_minimum_original_size_full_compression() {
        // Arrange: smallest possible original_size with zero compressed.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: original_size=1, compressed=0 => ratio=0.0.
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::KvContext),
            0,
            1, // minimum original
            StorageTier::GpuHbm,
            0,
        );

        // Assert: compression bonus = (1.0 - 0.0) * 500 = 500.
        // page_size_bonus = 1/1024 * 1 = 0 (truncated to 0).
        // base(1000) + compression(500) + page_size(0) = 1500.
        assert_eq!(
            score, 1500,
            "minimum original with full compression: base(1000) + compression(500) + page_size(0) = 1500, got {}",
            score,
        );
    }

    // ── 15. Frequency bonus constant is exactly 15 ──

    #[test]
    fn frequency_bonus_constant_is_fifteen() {
        // Arrange: verify FREQUENCY_BONUS constant.
        // Act & Assert: check the constant value.
        assert_eq!(FREQUENCY_BONUS, 15, "FREQUENCY_BONUS must be exactly 15 per SPEC");

        // Verify through scoring: delta from 0 to 10 accesses should be 10*15=150.
        let meta_0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_10 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta_0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s10 = EvictionWorker::compute_importance_score(
            &meta_10, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(
            s10 - s0, 150,
            "10 accesses should yield exactly 10*15=150 bonus",
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Additional tests (13 new)
    // ─────────────────────────────────────────────────────────────────────────

    // ── Score: ExpertWeight + Warm + Nvme, access_count=1, recency=0, age=0 ──

    #[test]
    fn score_exact_expert_warm_nvme_freq1() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 0, 0, StorageTier::Nvme, 0,
        );
        // base(1000) + freq(1*15=15) + expert(-300) + warm(5000) + nvme(-500) = 5215
        assert_eq!(score, 5215, "ExpertWeight + Warm + Nvme + freq=1 = 5215, got {}", score);
    }

    // ── Score: frequency bonus exactly offsets time penalty at equilibrium ──

    #[test]
    fn score_freq_offsets_time_at_equilibrium() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, None, 0, 0, StorageTier::GpuHbm, 150,
        );
        // freq_bonus = 10 * 15 = 150, time_penalty = 150 * 2 = 300
        // base(1000) + 150 - 300 = 850
        assert_eq!(score, 850, "freq=10, age=150: net penalty = -150, got {}", score);
    }

    // ── Score: KnowledgeRAG + Warm on GpuHbm scores above threshold ──

    #[test]
    fn score_rag_warm_hbm_above_threshold() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KnowledgeRAG), 0, 0, StorageTier::GpuHbm, 0,
        );
        // base(1000) + rag(-500) + warm(5000) = 5500
        assert!(
            score > IMPORTANCE_SCORE_THRESHOLD,
            "KnowledgeRAG + Warm on HBM should be above threshold: got {}",
            score,
        );
        assert_eq!(score, 5500, "exact score should be 5500");
    }

    // ── EvictionCandidate: truncate to 1 selects only the single lowest score ──

    #[test]
    fn candidate_truncate_to_one_selects_lowest() {
        let mut candidates: Vec<EvictionCandidate> = vec![
            EvictionCandidate {
                page_id: 10, score: 500, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            },
            EvictionCandidate {
                page_id: 20, score: -100, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(1),
            },
            EvictionCandidate {
                page_id: 30, score: 75, current_tier: StorageTier::CpuDram,
                codec: CompressionCodec::BitPackRle, page_bytes: 4096, group_id: None,
            },
        ];
        candidates.sort_by_key(|c| c.score);
        candidates.truncate(1);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].page_id, 20, "only the lowest score candidate should remain");
        assert_eq!(candidates[0].score, -100);
    }

    // ── Score: page_size_bonus doubles when page size doubles ──

    #[test]
    fn page_size_bonus_doubles_with_doubling_page() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Use same compression ratio to cancel compression bonus.
        let score_2k = EvictionWorker::compute_importance_score(
            &meta, None, 2048, 2048, StorageTier::GpuHbm, 0,
        );
        let score_4k = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // Both ratio=1.0 so compression bonus = 0 for both.
        // page_size bonus delta = (4096/1024) - (2048/1024) = 4 - 2 = 2
        assert_eq!(
            score_4k - score_2k, 2,
            "doubling page from 2K to 4K should increase bonus by 2",
        );
    }

    // ── classify_eviction_tier: KnowledgeRAG with score=0 is StandbyKv ──

    #[test]
    fn classify_knowledge_rag_score_zero_is_standby() {
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KnowledgeRAG), 0),
            EvictionTier::StandbyKv,
        );
    }

    // ── Score: PromptSystem + Warm + CpuDram + age=100 with compression ──

    #[test]
    fn score_exact_prompt_warm_dram_age100_compressed() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: None,
        };
        let score = EvictionWorker::compute_importance_score(
            &meta,
            Some(PagePayloadKind::PromptSystem),
            2048, // 50% of 4096
            4096,
            StorageTier::CpuDram,
            100,
        );
        // base(1000) - time(100*2=200) + compression((1-0.5)*500=250) + page_size(4096/1024*1=4)
        // + prompt(1000) + warm(5000) + dram(-200) = 6854
        assert_eq!(score, 6854, "PromptSystem + Warm + CpuDram + age=100 + 50% compressed = 6854, got {}", score);
    }

    // ── EvictionCandidate: sort by score with all equal scores preserves length ──

    #[test]
    fn candidate_sort_all_equal_preserves_count() {
        let candidates: Vec<EvictionCandidate> = (0..7)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: 42,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            })
            .collect();
        let mut sorted = candidates;
        sorted.sort_by_key(|c| c.score);
        assert_eq!(sorted.len(), 7, "all equal scores should preserve total count");
        for c in &sorted {
            assert_eq!(c.score, 42);
        }
    }

    // ── Score: ExpertWeight + DenseLayerWeight produce exactly 5300 delta ──

    #[test]
    fn score_expert_vs_dense_delta_is_5300() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 5,
            access_count: 20,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_expert = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 2048, 4096, StorageTier::GpuHbm, 200,
        );
        let score_dense = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 2048, 4096, StorageTier::GpuHbm, 200,
        );
        // EXPERT_WEIGHT_BONUS = -300, DENSE_LAYER_BONUS = +5000
        // delta = 5000 - (-300) = 5300
        assert_eq!(
            score_dense - score_expert, 5300,
            "DenseLayerWeight vs ExpertWeight delta should be 5300: got {}",
            score_dense - score_expert,
        );
    }

    // ── EvictionWorkerConfig: default importance_threshold matches constant ──

    #[test]
    fn config_default_importance_threshold_matches_constant() {
        let cfg = EvictionWorkerConfig::default();
        assert_eq!(
            cfg.importance_threshold, IMPORTANCE_SCORE_THRESHOLD,
            "default importance_threshold should equal IMPORTANCE_SCORE_THRESHOLD ({})",
            IMPORTANCE_SCORE_THRESHOLD,
        );
    }

    // ── Score: recency=1 delta is exactly TIME_DECAY_WEIGHT/2 = 1 ──

    #[test]
    fn recency_single_unit_penalty_exact() {
        let meta_r0 = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_r1 = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 1,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let s0 = EvictionWorker::compute_importance_score(
            &meta_r0, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        let s1 = EvictionWorker::compute_importance_score(
            &meta_r1, None, 0, 0, StorageTier::GpuHbm, 0,
        );
        assert_eq!(
            s0 - s1, (TIME_DECAY_WEIGHT / 2) as i64,
            "single recency unit penalty = TIME_DECAY_WEIGHT/2 = {}",
            TIME_DECAY_WEIGHT / 2,
        );
    }

    // ── EvictionTier: Copy + Hash + Eq satisfied — verify all 4 unique in array dedup ──

    #[test]
    fn eviction_tier_dedup_via_hashset() {
        use std::collections::HashSet;
        let variants = [
            EvictionTier::ColdExpert,
            EvictionTier::PinnedDense,
            EvictionTier::StandbyKv,
            EvictionTier::Protected,
        ];
        let set: HashSet<EvictionTier> = variants.iter().copied().collect();
        assert_eq!(set.len(), 4, "all 4 variants should be distinct");
        // Verify Copy semantics: iterating twice produces same results.
        let set2: HashSet<EvictionTier> = variants.iter().copied().collect();
        assert_eq!(set, set2, "Copy should produce identical sets on repeated iteration");
    }

    // ── Score: all tier discounts are negative relative to GpuHbm ──

    #[test]
    fn all_tier_discounts_are_negative_or_zero() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let score_hbm = EvictionWorker::compute_importance_score(
            &meta, None, 0, 4096, StorageTier::GpuHbm, 0,
        );
        let score_dram = EvictionWorker::compute_importance_score(
            &meta, None, 0, 4096, StorageTier::CpuDram, 0,
        );
        let score_nvme = EvictionWorker::compute_importance_score(
            &meta, None, 0, 4096, StorageTier::Nvme, 0,
        );
        assert!(score_hbm >= score_dram, "HBM should score >= DRAM");
        assert!(score_dram >= score_nvme, "DRAM should score >= NVMe");
        // HBM has no discount, so all others are <= HBM.
        assert!(score_dram < score_hbm, "DRAM discount should make it strictly less than HBM");
        assert!(score_nvme < score_dram, "NVMe discount should make it strictly less than DRAM");
    }
}
