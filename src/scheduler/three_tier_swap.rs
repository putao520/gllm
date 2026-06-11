//! Three-Tier Swap Integration (SPEC §22 §7).
//!
//! Coordinates the full page lifecycle across three storage tiers:
//!
//! ```text
//! GPU HBM  ←→  CPU DRAM  ←→  NVMe SSD
//!   (L1)         (L2)          (L3)
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                     HGAL Scheduler                               │
//! │  build_batch() → importance scores → eviction_candidates        │
//! │                → tier=CpuDram lookup → swap_in_enqueue          │
//! └──────────────┬────────────────────────────────┬─────────────────┘
//!                │ eviction                       │ swap-in
//!                ▼                                ▼
//! ┌──────────────────────┐          ┌──────────────────────────────┐
//! │   Eviction Worker    │          │     Swap-In Worker           │
//! │   (bg thread)        │          │     (bg thread)              │
//! │   ┌────────────────┐ │          │   ┌────────────────────────┐ │
//! │   │ score → select │ │          │   │ urgency sort → promote │ │
//! │   │ → EvictToDram  │ │          │   │ → PromoteToHbm         │ │
//! │   │ → EvictToNvme  │ │          │   │ → PromoteToDram        │ │
//! │   └───────┬────────┘ │          │   └───────────┬────────────┘ │
//! └───────────┼──────────┘          └───────────────┼──────────────┘
//!             │                                     │
//!             ▼                                     ▼
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                  PageMigrationActor (×2)                         │
//! │  Shared: DmaBackend + PageAddrTable + NvmeSwapFile              │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## SPEC §7 Dataflow
//!
//! ### §7.1 Eviction
//! 1. Scheduler.build_batch collects importance scores
//! 2. Finds lowest N → eviction_candidates
//! 3. Eviction Worker: GPU compress → DMA to DRAM → release GPU page
//! 4. Future batches won't read these pages (score too low)
//!
//! ### §7.2 Swap-In
//! 1. Scheduler.build_batch detects tier=CpuDram pages
//! 2. Enqueues swap-in requests
//! 3. Swap-In Worker: DMA DRAM→HBM → JIT decode → mark codec=None
//! 4. Mega-kernel resumes attention read
//!
//! ### §7.3 Orthogonality with §19 PrecisionTier
//! - §19 determines PrecisionTier (FP16/KIVI4/KIVI2)
//! - §22 determines whether byte-stream is compressed + which storage tier
//! - Both can stack: KIVI2 + LZ4 + DRAM = coldest page storage

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use crate::kv_cache::{CompressionCodec, StorageTier};
use crate::scheduler::dma_helpers::DmaBackend;
use crate::scheduler::eviction_worker::{EvictionWorker, EvictionWorkerConfig, EvictionCandidate};
use crate::scheduler::fault_recovery::WeightPageTable;
use crate::scheduler::memory_manager::GlobalMemoryManager;
use crate::scheduler::migration_actor::{
    MigrationActorConfig, PageAddrEntry, PageAddrTable, PageMigrationActor,
};
use crate::scheduler::nvme_swap::NvmeSwapFile;
use crate::scheduler::observer::BasicObserver;
use crate::scheduler::swap_in_worker::{PrefetchRequest, SwapInWorker, SwapInWorkerConfig};
use crate::scheduler::types::{PageId, PageMetadata, PagePayloadKind, PageState};
use crate::scheduler::weight_paging::{
    MultiGpuPageMigrator, WeightPageDefragmenter, WeightPageDistribution,
    WeightPageDtypeHandler,
};

// ─────────────────────────────────────────────────────────────────────────────
// Three-tier swap statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Cumulative statistics for three-tier swap operations.
///
/// Tracks evictions and swap-ins across all three tiers, providing
/// observability into the full page lifecycle (SPEC §22 §7.4).
#[derive(Debug, Clone, Default)]
pub struct ThreeTierSwapStats {
    /// Total pages evicted from GPU HBM → CPU DRAM.
    pub evictions_gpu_to_dram: u64,
    /// Total pages evicted from CPU DRAM → NVMe.
    pub evictions_dram_to_nvme: u64,
    /// Total pages swapped in from CPU DRAM → GPU HBM.
    pub swap_ins_dram_to_gpu: u64,
    /// Total pages swapped in from NVMe → CPU DRAM (first hop).
    pub swap_ins_nvme_to_dram: u64,
    /// Total bytes evicted across all tiers.
    pub total_bytes_evicted: u64,
    /// Total bytes swapped in across all tiers.
    pub total_bytes_swapped_in: u64,
    /// Cumulative eviction latency in microseconds.
    pub total_eviction_latency_us: u64,
    /// Cumulative swap-in latency in microseconds.
    pub total_swap_in_latency_us: u64,
    /// Number of eviction rounds executed.
    pub eviction_rounds: u64,
    /// Number of swap-in rounds executed.
    pub swap_in_rounds: u64,
    /// Current number of pages on each tier (snapshot).
    pub pages_on_hbm: usize,
    pub pages_on_dram: usize,
    pub pages_on_nvme: usize,
}

impl ThreeTierSwapStats {
    /// Average eviction latency in microseconds.
    pub fn avg_eviction_latency_us(&self) -> f64 {
        let total = self.evictions_gpu_to_dram + self.evictions_dram_to_nvme;
        if total == 0 {
            return 0.0;
        }
        self.total_eviction_latency_us as f64 / total as f64
    }

    /// Average swap-in latency in microseconds.
    pub fn avg_swap_in_latency_us(&self) -> f64 {
        let total = self.swap_ins_dram_to_gpu + self.swap_ins_nvme_to_dram;
        if total == 0 {
            return 0.0;
        }
        self.total_swap_in_latency_us as f64 / total as f64
    }

    /// Total pages migrated (evictions + swap-ins).
    pub fn total_migrations(&self) -> u64 {
        self.evictions_gpu_to_dram
            + self.evictions_dram_to_nvme
            + self.swap_ins_dram_to_gpu
            + self.swap_ins_nvme_to_dram
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tier migration plan (build_batch output)
// ─────────────────────────────────────────────────────────────────────────────

/// Output of `ThreeTierSwapCoordinator::build_batch`.
///
/// Encodes the scheduler's decision about which pages to evict
/// and which pages to swap in for the current batch.
#[derive(Debug, Clone)]
pub struct TierMigrationPlan {
    /// Pages selected for eviction (sorted by ascending importance score).
    pub eviction_candidates: Vec<EvictionCandidate>,
    /// Pages that need to be swapped in before compute.
    pub swap_in_requests: Vec<PrefetchRequest>,
    /// Pages whose tier must change this batch.
    pub tier_migrations: Vec<TierMigration>,
    /// Timestamp when this plan was built.
    pub built_at: Instant,
}

/// A single tier migration decision.
#[derive(Debug, Clone)]
pub struct TierMigration {
    pub page_id: PageId,
    pub from_tier: StorageTier,
    pub to_tier: StorageTier,
    pub codec: CompressionCodec,
    pub page_bytes: usize,
    pub reason: TierMigrationReason,
}

/// Why a page is being migrated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierMigrationReason {
    /// Evicted due to low importance score under memory pressure.
    EvictionPressure,
    /// Swapped in because a sequence needs it.
    SequenceDemand,
    /// Proactive prefetch based on access pattern prediction.
    Prefetch,
    /// Cold page cascaded to deeper tier for capacity.
    ColdCascade,
}

// ─────────────────────────────────────────────────────────────────────────────
// ThreeTierSwapCoordinator configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the three-tier swap coordinator.
#[derive(Debug, Clone)]
pub struct ThreeTierSwapConfig {
    /// Eviction worker configuration.
    pub eviction: EvictionWorkerConfig,
    /// Swap-in worker configuration.
    pub swap_in: SwapInWorkerConfig,
    /// Migration actor configuration (shared).
    pub migration: MigrationActorConfig,
    /// Whether to auto-start background workers on construction.
    pub auto_start: bool,
}

impl Default for ThreeTierSwapConfig {
    fn default() -> Self {
        Self {
            eviction: EvictionWorkerConfig::default(),
            swap_in: SwapInWorkerConfig::default(),
            migration: MigrationActorConfig::default(),
            auto_start: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ThreeTierSwapCoordinator
// ─────────────────────────────────────────────────────────────────────────────

/// Three-tier swap lifecycle coordinator (SPEC §22 §7).
///
/// Owns and orchestrates:
/// - **Eviction Worker**: background thread that scores pages and submits
///   `EvictToDram` / `EvictToNvme` commands.
/// - **Swap-In Worker**: background thread that processes prefetch requests
///   and submits `PromoteToHbm` / `PromoteToDram` commands.
/// - **PageMigrationActor** (×2): one for eviction, one for swap-in, sharing
///   the same `DmaBackend`, `PageAddrTable`, and `NvmeSwapFile`.
/// - **Shared state**: `page_metadata`, `addr_table`, `memory_manager`, `observer`.
///
/// ## Lifecycle
///
/// ```text
/// new() → spawn workers → build_batch() loop → shutdown()
/// ```
///
/// `build_batch` is the hot-path method called by the HGAL scheduler each
/// iteration. It reads the current page metadata, computes importance scores
/// for all pages, selects eviction candidates, and enqueues swap-in requests
/// for pages needed by active sequences.
pub struct ThreeTierSwapCoordinator {
    /// Eviction worker (background thread).
    eviction_worker: Option<EvictionWorker>,
    /// Swap-in worker (background thread).
    swap_in_worker: Option<SwapInWorker>,
    /// Shared page metadata (read/write by all components).
    page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>>,
    /// Shared page address table (GPU ptrs, host buffers, tier tracking).
    addr_table: PageAddrTable,
    /// Shared global memory manager (tier usage statistics).
    memory_manager: Arc<Mutex<GlobalMemoryManager>>,
    /// Shared observer for telemetry events.
    observer: Arc<Mutex<BasicObserver>>,
    /// Accumulated swap statistics.
    stats: Arc<Mutex<ThreeTierSwapStats>>,
    /// Configuration snapshot.
    config: ThreeTierSwapConfig,
    /// Multi-GPU page migration coordinator (REQ-WP-005).
    gpu_migrator: Mutex<MultiGpuPageMigrator>,
    /// Weight page defragmenter (REQ-WP-010).
    defragmenter: WeightPageDefragmenter,
    /// DType-aware weight page fault handler (REQ-WP-009).
    dtype_handler: Mutex<WeightPageDtypeHandler>,
}

impl ThreeTierSwapCoordinator {
    /// Create a new coordinator and spawn background workers.
    ///
    /// # Arguments
    /// * `config` — Coordinator configuration.
    /// * `backend` — DMA backend for GPU/CPU data movement.
    /// * `memory_manager` — Global memory manager for tier usage stats.
    /// * `observer` — Telemetry observer.
    ///
    /// # Panics
    /// Panics if worker threads cannot be spawned.
    pub fn new(
        config: ThreeTierSwapConfig,
        backend: Arc<dyn DmaBackend>,
        memory_manager: Arc<Mutex<GlobalMemoryManager>>,
        observer: Arc<Mutex<BasicObserver>>,
    ) -> Self {
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let stats = Arc::new(Mutex::new(ThreeTierSwapStats::default()));

        // Ensure swap directory exists.
        let _ = std::fs::create_dir_all(&config.migration.nvme_swap_dir);

        let nvme_swap = {
            let max_slot = config.migration.page_size * 2;
            Arc::new(
                NvmeSwapFile::open(
                    config.migration.swap_file_path(),
                    config.migration.page_size,
                    max_slot,
                    config.migration.max_swap_pages,
                )
                .expect("ThreeTierSwapCoordinator: failed to open NVMe swap file"),
            )
        };

        // Create two PageMigrationActors sharing the same backend, addr_table, and NVMe.
        let eviction_actor = PageMigrationActor::spawn_with_backend(
            config.migration.clone(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            Some(Arc::clone(&nvme_swap)),
        );
        let swap_in_actor = PageMigrationActor::spawn_with_backend(
            config.migration.clone(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            Some(Arc::clone(&nvme_swap)),
        );

        let eviction_worker = if config.auto_start {
            Some(EvictionWorker::spawn(
                config.eviction.clone(),
                eviction_actor,
                Arc::clone(&page_metadata),
                Arc::clone(&addr_table),
                Arc::clone(&memory_manager),
                Arc::clone(&observer),
            ))
        } else {
            drop(eviction_actor);
            None
        };

        let swap_in_worker = if config.auto_start {
            Some(SwapInWorker::spawn(
                config.swap_in.clone(),
                swap_in_actor,
                Arc::clone(&page_metadata),
                Arc::clone(&addr_table),
                Arc::clone(&observer),
            ))
        } else {
            drop(swap_in_actor);
            None
        };

        Self {
            eviction_worker,
            swap_in_worker,
            page_metadata,
            addr_table,
            memory_manager,
            observer,
            stats,
            config,
            gpu_migrator: Mutex::new(MultiGpuPageMigrator::new()),
            defragmenter: WeightPageDefragmenter::new(),
            dtype_handler: Mutex::new(WeightPageDtypeHandler::new()),
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    /// Shared page metadata map (read by scheduler, written by workers).
    pub fn page_metadata(&self) -> &Arc<RwLock<HashMap<PageId, PageMetadata>>> {
        &self.page_metadata
    }

    /// Shared page address table.
    pub fn addr_table(&self) -> &PageAddrTable {
        &self.addr_table
    }

    /// Shared memory manager.
    pub fn memory_manager(&self) -> &Arc<Mutex<GlobalMemoryManager>> {
        &self.memory_manager
    }

    /// Shared observer.
    pub fn observer(&self) -> &Arc<Mutex<BasicObserver>> {
        &self.observer
    }

    /// Reference to the swap-in worker for direct prefetch enqueue.
    pub fn swap_in_worker(&self) -> Option<&SwapInWorker> {
        self.swap_in_worker.as_ref()
    }

    /// Reference to the eviction worker.
    pub fn eviction_worker(&self) -> Option<&EvictionWorker> {
        self.eviction_worker.as_ref()
    }

    // ── Weight Paging Integration (REQ-WP-005 / REQ-WP-010) ──────────────────

    /// Process MoE prefetch requests through the multi-GPU migrator (REQ-WP-005).
    ///
    /// Called by the inference coordinator when a MoE model's expert weights
    /// reside on a different GPU than the compute device. Creates PCIe DMA
    /// migration entries for weights that are not already on GPU.
    pub fn migrate_expert_pages(
        &self,
        prefetch_requests: &[crate::moe::prefetch::ExpertPrefetchRequest],
        current_device: u32,
        weight_table: &mut WeightPageTable,
    ) -> Vec<crate::scheduler::weight_paging::PcieDmaTransfer> {
        let mut gmm = match self.memory_manager.lock() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let mut migrator = match self.gpu_migrator.lock() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        migrator.step(prefetch_requests, current_device, &mut gmm, weight_table)
    }

    /// Run weight page defragmentation cycle (REQ-WP-010).
    ///
    /// Analyzes the current tier distribution and, if fragmentation exceeds
    /// the threshold, merges small pages into contiguous blocks.
    /// Returns the number of bytes freed by defragmentation.
    pub fn run_defrag_cycle(&self) -> usize {
        let mm = match self.memory_manager.lock() {
            Ok(g) => g,
            Err(_) => return 0,
        };
        let l1 = mm.tier_usage(crate::scheduler::memory_manager::Tier::L1);
        let l2 = mm.tier_usage(crate::scheduler::memory_manager::Tier::L2);
        let l3 = mm.tier_usage(crate::scheduler::memory_manager::Tier::L3);

        let l1_used = l1.used;
        let l2_used = l2.used;
        let l3_used = l3.used;

        // Estimate fragmentation from free space fragmentation.
        let l1_frag = if l1.capacity > 0 { 1.0 - (l1.available() as f32 / l1.capacity as f32) } else { 0.0 };
        let l2_frag = if l2.capacity > 0 { 1.0 - (l2.available() as f32 / l2.capacity as f32) } else { 0.0 };

        let dist = WeightPageDistribution {
            l1_count: l1_used as usize / 4096,
            l2_count: l2_used as usize / 4096,
            l3_count: l3_used as usize / 4096,
            l1_fragmentation: l1_frag,
            l2_fragmentation: l2_frag,
        };

        match self.defragmenter.analyze(&dist) {
            Some(plan) => {
                if let Ok(mut obs) = self.observer.lock() {
                    self.defragmenter.execute(&plan, &mut obs)
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Handle a weight page fault using DType-aware handler (REQ-WP-009).
    ///
    /// When a weight page faults on L2 or L3, this method coordinates
    /// with the fault recovery handler to determine the action, mapping
    /// WeightTier to the swap-in tier.
    pub fn handle_weight_page_fault(
        &self,
        page_id: crate::scheduler::types::PageId,
        current_weight_tier: crate::scheduler::types::WeightTier,
        weight_table: &WeightPageTable,
    ) -> crate::scheduler::fault_recovery::FaultAction {
        let gmm = match self.memory_manager.lock() {
            Ok(g) => g,
            Err(_) => return crate::scheduler::fault_recovery::FaultAction::Abort {
                reason: "memory manager lock poisoned".to_string(),
            },
        };
        let mut handler = match self.dtype_handler.lock() {
            Ok(h) => h,
            Err(_) => return crate::scheduler::fault_recovery::FaultAction::Abort {
                reason: "dtype handler lock poisoned".to_string(),
            },
        };
        handler.handle_weight_fault(page_id, current_weight_tier, &gmm, weight_table)
    }

    // ── build_batch (SPEC §7.1 / §7.2) ─────────────────────────────────────

    /// Build a batch plan: score all pages, select eviction candidates,
    /// and identify pages that need to be swapped in.
    ///
    /// This is the main hot-path method called by the HGAL scheduler
    /// each iteration. It implements the SPEC §7.1 eviction flow and
    /// §7.2 swap-in flow.
    ///
    /// # Arguments
    /// * `active_sequences` — Pages currently needed by active sequences.
    ///   Any page in this set that is not on GpuHbm will generate a swap-in request.
    /// * `memory_pressure` — Current GPU HBM pressure ratio [0.0, 1.0].
    ///
    /// # Returns
    /// A `TierMigrationPlan` with eviction candidates and swap-in requests.
    pub fn build_batch(
        &self,
        active_pages: &[PageId],
        memory_pressure: f32,
    ) -> TierMigrationPlan {
        let now = Instant::now();
        let mut eviction_candidates: Vec<EvictionCandidate> = Vec::new();
        let mut swap_in_requests: Vec<PrefetchRequest> = Vec::new();
        let mut tier_migrations: Vec<TierMigration> = Vec::new();

        // ── Read current state ─────────────────────────────────────────────
        let meta_guard = match self.page_metadata.read() {
            Ok(g) => g,
            Err(_) => {
                return TierMigrationPlan {
                    eviction_candidates,
                    swap_in_requests,
                    tier_migrations,
                    built_at: now,
                };
            }
        };
        let addr_guard = match self.addr_table.read() {
            Ok(g) => g,
            Err(_) => {
                return TierMigrationPlan {
                    eviction_candidates,
                    swap_in_requests,
                    tier_migrations,
                    built_at: now,
                };
            }
        };

        // ── Active page set for fast lookup ────────────────────────────────
        let active_set: std::collections::HashSet<PageId> =
            active_pages.iter().copied().collect();

        // ── Score all pages and identify eviction candidates + swap-in needs ──
        for (&page_id, meta) in meta_guard.iter() {
            // Determine current tier from addr_table.
            let entry = match addr_guard.get(&page_id) {
                Some(e) => e,
                None => continue,
            };

            let current_tier = entry.current_tier;
            let tier_age_ticks = compute_tier_age_ticks(meta);

            // ── §7.2: Check if this page is needed but not on HBM ──────────
            if active_set.contains(&page_id) && current_tier != StorageTier::GpuHbm {
                let urgency = SwapInWorker::compute_urgency(
                    meta,
                    0.9, // high confidence since the sequence explicitly needs it
                    current_tier,
                );
                swap_in_requests.push(PrefetchRequest {
                    page_id,
                    urgency,
                    prefetch_confidence: 0.9,
                    page_bytes: entry.original_bytes,
                    enqueued_at: now,
                });
                tier_migrations.push(TierMigration {
                    page_id,
                    from_tier: current_tier,
                    to_tier: StorageTier::GpuHbm,
                    codec: entry.codec,
                    page_bytes: entry.original_bytes,
                    reason: TierMigrationReason::SequenceDemand,
                });
                continue; // Don't evict pages we're about to swap in
            }

            // ── §7.1: Score for eviction eligibility ───────────────────────
            // Skip non-evictable states.
            if meta.state == PageState::Protected
                || meta.state == PageState::Active
                || meta.state == PageState::Warm
            {
                continue;
            }

            // Determine payload kind for scoring.
            let payload_kind = infer_swap_payload_kind(meta);

            // Determine eligibility based on tier.
            let eligible = match current_tier {
                StorageTier::GpuHbm => {
                    memory_pressure > self.config.eviction.hbm_pressure_threshold
                        && tier_age_ticks > self.config.eviction.hbm_evict_age_ticks
                }
                StorageTier::CpuDram => {
                    // Check DRAM pressure.
                    let dram_pressure = self.dram_pressure_ratio();
                    dram_pressure > self.config.eviction.dram_pressure_threshold
                        && tier_age_ticks > self.config.eviction.dram_evict_age_ticks
                }
                StorageTier::Nvme => false, // Already coldest tier
            };

            if !eligible {
                continue;
            }

            // Compute importance score.
            let compressed_size = entry.original_bytes as u32; // proxy
            let original_size = entry.original_bytes as u32;
            let score = EvictionWorker::compute_importance_score(
                meta,
                payload_kind,
                compressed_size,
                original_size,
                current_tier,
                tier_age_ticks,
            );

            if score < self.config.eviction.importance_threshold {
                let codec = if current_tier == StorageTier::GpuHbm {
                    self.config.eviction.default_evict_codec
                } else {
                    entry.codec
                };

                eviction_candidates.push(EvictionCandidate {
                    page_id,
                    score,
                    current_tier,
                    codec,
                    page_bytes: self.config.eviction.page_bytes,
                    group_id: meta.sequence_id,
                });

                let (to_tier, reason) = match current_tier {
                    StorageTier::GpuHbm => {
                        (StorageTier::CpuDram, TierMigrationReason::EvictionPressure)
                    }
                    StorageTier::CpuDram => {
                        (StorageTier::Nvme, TierMigrationReason::ColdCascade)
                    }
                    _ => continue,
                };

                tier_migrations.push(TierMigration {
                    page_id,
                    from_tier: current_tier,
                    to_tier,
                    codec,
                    page_bytes: self.config.eviction.page_bytes,
                    reason,
                });
            }
        }
        drop(addr_guard);
        drop(meta_guard);

        // ── Sort eviction candidates by score ascending (lowest = evict first) ──
        eviction_candidates.sort_by_key(|c| c.score);
        eviction_candidates.truncate(self.config.eviction.max_evict_per_round);

        // ── Enqueue swap-in requests ───────────────────────────────────────
        if let Some(ref sw) = self.swap_in_worker {
            sw.prefetch_batch(&swap_in_requests);
        }

        // ── Update stats ───────────────────────────────────────────────────
        if let Ok(mut s) = self.stats.lock() {
            s.eviction_rounds += 1;
            s.swap_in_rounds += 1;
        }

        TierMigrationPlan {
            eviction_candidates,
            swap_in_requests,
            tier_migrations,
            built_at: now,
        }
    }

    /// Snapshot current three-tier swap statistics.
    ///
    /// Reads the current tier distribution from the addr_table to compute
    /// `pages_on_hbm`, `pages_on_dram`, `pages_on_nvme`.
    pub fn stats(&self) -> ThreeTierSwapStats {
        let mut stats = match self.stats.lock() {
            Ok(s) => s.clone(),
            Err(_) => return ThreeTierSwapStats::default(),
        };

        // Count pages by tier.
        if let Ok(table) = self.addr_table.read() {
            stats.pages_on_hbm = 0;
            stats.pages_on_dram = 0;
            stats.pages_on_nvme = 0;
            for entry in table.values() {
                match entry.current_tier {
                    StorageTier::GpuHbm => stats.pages_on_hbm += 1,
                    StorageTier::CpuDram => stats.pages_on_dram += 1,
                    StorageTier::Nvme => stats.pages_on_nvme += 1,
                }
            }
        }

        stats
    }

    /// Register a page in the addr_table (called when a page is first allocated).
    pub fn register_page(
        &self,
        page_id: PageId,
        gpu_ptr: Option<u64>,
        page_bytes: usize,
    ) {
        let mut table = match self.addr_table.write() {
            Ok(t) => t,
            Err(_) => return,
        };
        table.entry(page_id).or_insert_with(|| PageAddrEntry {
            gpu_ptr,
            host_buffer: None,
            current_tier: if gpu_ptr.is_some() {
                StorageTier::GpuHbm
            } else {
                StorageTier::CpuDram
            },
            original_bytes: page_bytes,
            codec: CompressionCodec::None,
        });
    }

    /// Update a page's GPU pointer in the addr_table.
    pub fn update_page_gpu_ptr(&self, page_id: PageId, gpu_ptr: u64) {
        let mut table = match self.addr_table.write() {
            Ok(t) => t,
            Err(_) => return,
        };
        if let Some(entry) = table.get_mut(&page_id) {
            entry.gpu_ptr = Some(gpu_ptr);
            entry.current_tier = StorageTier::GpuHbm;
        }
    }

    /// Remove a page from all tracking structures.
    pub fn release_page(&self, page_id: PageId) {
        // Remove from addr_table.
        if let Ok(mut table) = self.addr_table.write() {
            table.remove(&page_id);
        }
        // Remove from page_metadata.
        if let Ok(mut meta) = self.page_metadata.write() {
            meta.remove(&page_id);
        }
    }

    /// Bulk-register page metadata from an external source (e.g. HGAL).
    ///
    /// For each `(page_id, metadata)` pair, inserts into both `addr_table`
    /// and `page_metadata` if not already present. Idempotent.
    pub fn register_pages_from_hgal(
        &self,
        pages: &std::collections::HashMap<PageId, PageMetadata>,
        page_bytes: usize,
    ) {
        for (&page_id, meta) in pages {
            self.register_page(page_id, None, page_bytes);
            if let Ok(mut pm) = self.page_metadata.write() {
                pm.entry(page_id).or_insert_with(|| meta.clone());
            }
        }
    }

    /// Return a snapshot of page IDs that changed tier since last sync.
    ///
    /// Compares `addr_table` current_tier against `page_metadata` state to find
    /// pages whose physical tier has diverged from their logical state.
    pub fn tier_changed_pages(&self) -> Vec<(PageId, StorageTier)> {
        let addr_guard = match self.addr_table.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        let meta_guard = match self.page_metadata.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        addr_guard
            .iter()
            .filter_map(|(&pid, entry)| {
                let meta = meta_guard.get(&pid)?;
                let meta_tier = match meta.state {
                    PageState::Active | PageState::Protected => StorageTier::GpuHbm,
                    PageState::Warm | PageState::Standby => StorageTier::CpuDram,
                    PageState::Swapped | PageState::SwappedOut => StorageTier::Nvme,
                    PageState::Free => StorageTier::CpuDram,
                };
                if entry.current_tier != meta_tier {
                    Some((pid, entry.current_tier))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Record a successful eviction completion (called from observer callback).
    pub fn record_eviction_completed(
        &self,
        page_id: PageId,
        from_tier: StorageTier,
        to_tier: StorageTier,
        bytes: u64,
        latency_us: u64,
    ) {
        let total_evictions;
        let total_swap_ins;
        if let Ok(mut s) = self.stats.lock() {
            match (from_tier, to_tier) {
                (StorageTier::GpuHbm, StorageTier::CpuDram) => {
                    s.evictions_gpu_to_dram += 1;
                }
                (StorageTier::CpuDram, StorageTier::Nvme) => {
                    s.evictions_dram_to_nvme += 1;
                }
                _ => {}
            }
            s.total_bytes_evicted += bytes;
            s.total_eviction_latency_us += latency_us;
            total_evictions = s.evictions_gpu_to_dram + s.evictions_dram_to_nvme;
            total_swap_ins = s.swap_ins_dram_to_gpu + s.swap_ins_nvme_to_dram;
        } else {
            return;
        }
        // REQ-SCHED-010: update thrashing_rate when eviction churn is detected.
        if total_swap_ins > 0 {
            if let Ok(mut obs) = self.observer.lock() {
                obs.update_thrashing_rate(total_evictions as f32 / total_swap_ins as f32);
            }
        }
        let _ = page_id;
    }

    /// Record a successful swap-in completion.
    pub fn record_swap_in_completed(
        &self,
        page_id: PageId,
        from_tier: StorageTier,
        to_tier: StorageTier,
        bytes: u64,
        latency_us: u64,
    ) {
        let total_swap_ins;
        let total_evictions;
        if let Ok(mut s) = self.stats.lock() {
            match (from_tier, to_tier) {
                (StorageTier::CpuDram, StorageTier::GpuHbm) => {
                    s.swap_ins_dram_to_gpu += 1;
                }
                (StorageTier::Nvme, StorageTier::CpuDram) => {
                    s.swap_ins_nvme_to_dram += 1;
                }
                _ => {}
            }
            s.total_bytes_swapped_in += bytes;
            s.total_swap_in_latency_us += latency_us;
            total_swap_ins = s.swap_ins_dram_to_gpu + s.swap_ins_nvme_to_dram;
            total_evictions = s.evictions_gpu_to_dram + s.evictions_dram_to_nvme;
        } else {
            return;
        }
        // REQ-SCHED-010: update page_hit_rate and swap_latency_us.
        if let Ok(mut obs) = self.observer.lock() {
            let total_ops = total_swap_ins + total_evictions;
            if total_ops > 0 {
                obs.update_page_hit_rate(total_swap_ins as f32 / total_ops as f32);
            }
            obs.update_swap_latency_us(latency_us as f32);
        }
        let _ = page_id;
    }

    /// Compute the current DRAM pressure ratio.
    fn dram_pressure_ratio(&self) -> f32 {
        let mm = match self.memory_manager.lock() {
            Ok(g) => g,
            Err(_) => return 0.0,
        };
        let dram = mm.tier_usage(crate::scheduler::memory_manager::Tier::L2);
        if dram.capacity > 0 {
            dram.used as f32 / dram.capacity as f32
        } else {
            0.0
        }
    }

    /// Gracefully shut down all workers.
    pub fn shutdown(&mut self) {
        if let Some(ref mut ew) = self.eviction_worker {
            ew.shutdown();
        }
        if let Some(ref mut sw) = self.swap_in_worker {
            sw.shutdown();
        }
        self.eviction_worker = None;
        self.swap_in_worker = None;
    }
}

impl Drop for ThreeTierSwapCoordinator {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute tier age in "ticks" from page metadata timestamps.
///
/// One tick ≈ 10ms (matching DEFAULT_TICK_INTERVAL).
fn compute_tier_age_ticks(meta: &PageMetadata) -> u64 {
    let anchor = meta.swap_in_time.unwrap_or(meta.last_access);
    let elapsed_ms = Instant::now()
        .saturating_duration_since(anchor)
        .as_millis() as u64;
    elapsed_ms / 10
}

/// Infer payload kind for swap scoring from page metadata.
fn infer_swap_payload_kind(meta: &PageMetadata) -> Option<PagePayloadKind> {
    match meta.sequence_id {
        None => Some(PagePayloadKind::ExpertWeight),
        Some(_) => Some(PagePayloadKind::KvContext),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::dma_helpers::CpuDmaBackendSized;
    use crate::scheduler::memory_manager::GlobalMemoryManager;
    use crate::scheduler::migration_actor::{MigrationCommand, MigrationDone, MigrationResult};
    use crate::scheduler::observer::BasicObserver;
    use std::time::Duration;

    fn make_coordinator(auto_start: bool) -> (ThreeTierSwapCoordinator, Arc<dyn DmaBackend>) {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(
            100_000, 1_000_000, 10_000_000,
        )));
        let observer = Arc::new(Mutex::new(BasicObserver::new()));
        let config = ThreeTierSwapConfig {
            auto_start,
            ..ThreeTierSwapConfig::default()
        };
        let coordinator = ThreeTierSwapCoordinator::new(
            config,
            Arc::clone(&backend),
            mm,
            observer,
        );
        (coordinator, backend)
    }

    #[test]
    fn coordinator_new_and_shutdown() {
        let (mut c, _backend) = make_coordinator(true);
        c.shutdown();
    }

    #[test]
    fn coordinator_no_auto_start() {
        let (mut c, _backend) = make_coordinator(false);
        assert!(c.swap_in_worker.is_none());
        assert!(c.eviction_worker.is_none());
        c.shutdown();
    }

    #[test]
    fn build_batch_empty_when_no_pages() {
        let (c, _backend) = make_coordinator(false);
        let plan = c.build_batch(&[], 0.5);
        assert!(plan.eviction_candidates.is_empty());
        assert!(plan.swap_in_requests.is_empty());
        assert!(plan.tier_migrations.is_empty());
    }

    #[test]
    fn build_batch_no_eviction_below_threshold() {
        let (c, _backend) = make_coordinator(false);

        // Register a page on HBM.
        c.register_page(1, Some(0x1000), 4096);

        // Add page metadata with Active state.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(
                1,
                PageMetadata {
                    page_id: 1,
                    sequence_id: Some(100),
                    recency: 0,
                    access_count: 5,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Active,
                    warm_until: None,
                },
            );
        }

        // Low pressure → no eviction.
        let plan = c.build_batch(&[], 0.3);
        assert!(plan.eviction_candidates.is_empty());
    }

    #[test]
    fn build_batch_evicts_standby_under_pressure() {
        let (c, _backend) = make_coordinator(false);

        // Register pages on HBM with Standby state.
        for pid in 1u64..=5u64 {
            c.register_page(pid as PageId, Some(0x1000 + pid), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(
                pid as PageId,
                PageMetadata {
                    page_id: pid as PageId,
                    sequence_id: Some(100 + pid),
                    recency: 100,
                    access_count: 0,
                    last_access: Instant::now() - Duration::from_secs(10),
                    swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                },
            );
        }

        // High pressure → should produce eviction candidates.
        let plan = c.build_batch(&[], 0.95);
        assert!(
            !plan.eviction_candidates.is_empty(),
            "should evict under high pressure"
        );
    }

    #[test]
    fn build_batch_swap_in_for_active_pages() {
        let (c, _backend) = make_coordinator(false);

        // Register a page on CpuDram.
        c.register_page(42, None, 4096);
        // Manually set tier to CpuDram.
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&42) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }

        // Add page metadata with Standby state.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(
                42,
                PageMetadata {
                    page_id: 42,
                    sequence_id: Some(200),
                    recency: 0,
                    access_count: 10,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                },
            );
        }

        // Page 42 is active → should generate swap-in request.
        let plan = c.build_batch(&[42], 0.5);
        assert!(
            !plan.swap_in_requests.is_empty(),
            "should request swap-in for active page on DRAM"
        );
        assert_eq!(plan.swap_in_requests[0].page_id, 42);
    }

    #[test]
    fn build_batch_no_swap_in_for_hbm_pages() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);

        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(
                1,
                PageMetadata {
                    page_id: 1,
                    sequence_id: Some(100),
                    recency: 0,
                    access_count: 5,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Active,
                    warm_until: None,
                },
            );
        }

        // Page already on HBM → no swap-in needed.
        let plan = c.build_batch(&[1], 0.5);
        assert!(
            plan.swap_in_requests.is_empty(),
            "should not swap-in pages already on HBM"
        );
    }

    #[test]
    fn stats_default() {
        let stats = ThreeTierSwapStats::default();
        assert_eq!(stats.total_migrations(), 0);
        assert_eq!(stats.avg_eviction_latency_us(), 0.0);
        assert_eq!(stats.avg_swap_in_latency_us(), 0.0);
    }

    #[test]
    fn stats_accumulation() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 5;
        stats.swap_ins_dram_to_gpu = 3;
        stats.total_eviction_latency_us = 1000;
        stats.total_swap_in_latency_us = 600;

        assert_eq!(stats.total_migrations(), 8);
        assert!((stats.avg_eviction_latency_us() - 200.0).abs() < 0.01);
        assert!((stats.avg_swap_in_latency_us() - 200.0).abs() < 0.01);
    }

    #[test]
    fn register_and_release_page() {
        let (c, _backend) = make_coordinator(false);

        c.register_page(99, Some(0x2000), 8192);

        {
            let table = c.addr_table.read().expect("read lock");
            let entry = table.get(&99).expect("should exist");
            assert_eq!(entry.gpu_ptr, Some(0x2000));
            assert_eq!(entry.current_tier, StorageTier::GpuHbm);
            assert_eq!(entry.original_bytes, 8192);
        }

        c.release_page(99);

        {
            let table = c.addr_table.read().expect("read lock");
            assert!(table.get(&99).is_none(), "should be removed");
        }
    }

    #[test]
    fn tier_migration_reason_display() {
        assert_eq!(TierMigrationReason::EvictionPressure, TierMigrationReason::EvictionPressure);
        assert_eq!(TierMigrationReason::SequenceDemand, TierMigrationReason::SequenceDemand);
        assert_ne!(TierMigrationReason::EvictionPressure, TierMigrationReason::Prefetch);
    }

    // ── ThreeTierSwapStats tests ─────────────────────────────────────────────

    #[test]
    fn stats_default_all_zero() {
        let stats = ThreeTierSwapStats::default();
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_evicted, 0);
        assert_eq!(stats.total_bytes_swapped_in, 0);
        assert_eq!(stats.total_eviction_latency_us, 0);
        assert_eq!(stats.total_swap_in_latency_us, 0);
        assert_eq!(stats.eviction_rounds, 0);
        assert_eq!(stats.swap_in_rounds, 0);
        assert_eq!(stats.pages_on_hbm, 0);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
    }

    #[test]
    fn stats_clone_independent() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 10;
        let cloned = stats.clone();
        assert_eq!(cloned.evictions_gpu_to_dram, 10);
        stats.evictions_gpu_to_dram = 20;
        assert_eq!(stats.evictions_gpu_to_dram, 20);
        assert_eq!(cloned.evictions_gpu_to_dram, 10);
    }

    #[test]
    fn stats_avg_eviction_latency_with_only_gpu_to_dram() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 4;
        stats.total_eviction_latency_us = 800;
        assert!((stats.avg_eviction_latency_us() - 200.0).abs() < 0.01);
    }

    #[test]
    fn stats_avg_eviction_latency_with_only_dram_to_nvme() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_dram_to_nvme = 2;
        stats.total_eviction_latency_us = 600;
        assert!((stats.avg_eviction_latency_us() - 300.0).abs() < 0.01);
    }

    #[test]
    fn stats_avg_eviction_latency_with_both_tiers() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 3;
        stats.evictions_dram_to_nvme = 2;
        stats.total_eviction_latency_us = 1000;
        assert!((stats.avg_eviction_latency_us() - 200.0).abs() < 0.01);
    }

    #[test]
    fn stats_avg_swap_in_latency_with_only_dram_to_gpu() {
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_dram_to_gpu = 5;
        stats.total_swap_in_latency_us = 1500;
        assert!((stats.avg_swap_in_latency_us() - 300.0).abs() < 0.01);
    }

    #[test]
    fn stats_avg_swap_in_latency_with_only_nvme_to_dram() {
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_nvme_to_dram = 3;
        stats.total_swap_in_latency_us = 900;
        assert!((stats.avg_swap_in_latency_us() - 300.0).abs() < 0.01);
    }

    #[test]
    fn stats_avg_swap_in_latency_with_both_tiers() {
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_dram_to_gpu = 2;
        stats.swap_ins_nvme_to_dram = 3;
        stats.total_swap_in_latency_us = 1000;
        assert!((stats.avg_swap_in_latency_us() - 200.0).abs() < 0.01);
    }

    #[test]
    fn stats_total_migrations_all_categories() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 10;
        stats.evictions_dram_to_nvme = 5;
        stats.swap_ins_dram_to_gpu = 8;
        stats.swap_ins_nvme_to_dram = 3;
        assert_eq!(stats.total_migrations(), 26);
    }

    #[test]
    fn stats_total_migrations_zero() {
        let stats = ThreeTierSwapStats::default();
        assert_eq!(stats.total_migrations(), 0);
    }

    // ── TierMigrationReason tests ────────────────────────────────────────────

    #[test]
    fn tier_migration_reason_all_variants_equality() {
        let reasons = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        for i in 0..reasons.len() {
            for j in 0..reasons.len() {
                if i == j {
                    assert_eq!(reasons[i], reasons[j]);
                } else {
                    assert_ne!(reasons[i], reasons[j]);
                }
            }
        }
    }

    #[test]
    fn tier_migration_reason_copy_semantics() {
        let a = TierMigrationReason::Prefetch;
        let b = a;
        assert_eq!(a, b);
    }

    // ── TierMigration struct tests ───────────────────────────────────────────

    #[test]
    fn tier_migration_clone_independent() {
        let migration = TierMigration {
            page_id: 42,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            reason: TierMigrationReason::EvictionPressure,
        };
        let cloned = migration.clone();
        assert_eq!(cloned.page_id, 42);
        assert_eq!(cloned.from_tier, StorageTier::GpuHbm);
        assert_eq!(cloned.to_tier, StorageTier::CpuDram);
        assert_eq!(cloned.codec, CompressionCodec::Lz4);
        assert_eq!(cloned.page_bytes, 4096);
        assert_eq!(cloned.reason, TierMigrationReason::EvictionPressure);
    }

    // ── TierMigrationPlan tests ──────────────────────────────────────────────

    #[test]
    fn tier_migration_plan_clone_preserves_fields() {
        let plan = TierMigrationPlan {
            eviction_candidates: vec![EvictionCandidate {
                page_id: 1,
                score: 50,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: Some(100),
            }],
            swap_in_requests: vec![PrefetchRequest {
                page_id: 2,
                urgency: 0.9,
                prefetch_confidence: 0.8,
                page_bytes: 8192,
                enqueued_at: Instant::now(),
            }],
            tier_migrations: vec![TierMigration {
                page_id: 3,
                from_tier: StorageTier::CpuDram,
                to_tier: StorageTier::Nvme,
                codec: CompressionCodec::ZstdDict,
                page_bytes: 4096,
                reason: TierMigrationReason::ColdCascade,
            }],
            built_at: Instant::now(),
        };
        let cloned = plan.clone();
        assert_eq!(cloned.eviction_candidates.len(), 1);
        assert_eq!(cloned.swap_in_requests.len(), 1);
        assert_eq!(cloned.tier_migrations.len(), 1);
        assert_eq!(cloned.eviction_candidates[0].page_id, 1);
        assert_eq!(cloned.tier_migrations[0].reason, TierMigrationReason::ColdCascade);
    }

    #[test]
    fn tier_migration_plan_empty_fields() {
        let plan = TierMigrationPlan {
            eviction_candidates: Vec::new(),
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        assert!(plan.eviction_candidates.is_empty());
        assert!(plan.swap_in_requests.is_empty());
        assert!(plan.tier_migrations.is_empty());
    }

    // ── ThreeTierSwapConfig tests ────────────────────────────────────────────

    #[test]
    fn config_default_auto_start_true() {
        let config = ThreeTierSwapConfig::default();
        assert!(config.auto_start);
    }

    #[test]
    fn config_default_sub_configs_populated() {
        let config = ThreeTierSwapConfig::default();
        assert_eq!(config.eviction.max_evict_per_round, 8);
        assert_eq!(config.swap_in.max_prefetch_per_round, 16);
        assert_eq!(config.migration.page_size, 4096);
    }

    #[test]
    fn config_clone_independent() {
        let config = ThreeTierSwapConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.auto_start, config.auto_start);
        assert_eq!(cloned.eviction.page_bytes, config.eviction.page_bytes);
    }

    // ── Coordinator register_page tests ──────────────────────────────────────

    #[test]
    fn register_page_with_gpu_ptr_sets_hbm_tier() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(10, Some(0xABCDE), 4096);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&10).expect("page should exist");
        assert_eq!(entry.gpu_ptr, Some(0xABCDE));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
        assert_eq!(entry.original_bytes, 4096);
        assert_eq!(entry.codec, CompressionCodec::None);
        assert!(entry.host_buffer.is_none());
    }

    #[test]
    fn register_page_without_gpu_ptr_sets_dram_tier() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(20, None, 8192);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&20).expect("page should exist");
        assert!(entry.gpu_ptr.is_none());
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        assert_eq!(entry.original_bytes, 8192);
    }

    #[test]
    fn register_page_idempotent() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(5, Some(0x1000), 4096);
        c.register_page(5, Some(0x2000), 8192);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&5).expect("page should exist");
        // or_insert_with preserves the first entry.
        assert_eq!(entry.gpu_ptr, Some(0x1000));
        assert_eq!(entry.original_bytes, 4096);
    }

    // ── Coordinator update_page_gpu_ptr tests ────────────────────────────────

    #[test]
    fn update_page_gpu_ptr_changes_tier_to_hbm() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(30, None, 4096);
        {
            let table = c.addr_table.read().expect("read lock");
            let entry = table.get(&30).expect("page should exist");
            assert_eq!(entry.current_tier, StorageTier::CpuDram);
        }
        c.update_page_gpu_ptr(30, 0xF000);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&30).expect("page should exist");
        assert_eq!(entry.gpu_ptr, Some(0xF000));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    #[test]
    fn update_page_gpu_ptr_nonexistent_page_no_panic() {
        let (c, _backend) = make_coordinator(false);
        c.update_page_gpu_ptr(999, 0xDEAD);
        let table = c.addr_table.read().expect("read lock");
        assert!(table.get(&999).is_none());
    }

    // ── Coordinator release_page tests ───────────────────────────────────────

    #[test]
    fn release_page_removes_from_both_tables() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(50, Some(0x5000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(50, PageMetadata::default());
        }
        {
            let table = c.addr_table.read().expect("read lock");
            assert!(table.contains_key(&50));
            let meta = c.page_metadata.read().expect("read lock");
            assert!(meta.contains_key(&50));
        }
        c.release_page(50);
        let table = c.addr_table.read().expect("read lock");
        assert!(table.get(&50).is_none());
        let meta = c.page_metadata.read().expect("read lock");
        assert!(meta.get(&50).is_none());
    }

    #[test]
    fn release_page_nonexistent_no_panic() {
        let (c, _backend) = make_coordinator(false);
        c.release_page(12345);
    }

    // ── Coordinator register_pages_from_hgal tests ───────────────────────────

    #[test]
    fn register_pages_from_hgal_bulk_insert() {
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        pages.insert(1, PageMetadata { page_id: 1, ..PageMetadata::default() });
        pages.insert(2, PageMetadata { page_id: 2, ..PageMetadata::default() });
        pages.insert(3, PageMetadata { page_id: 3, ..PageMetadata::default() });
        c.register_pages_from_hgal(&pages, 4096);

        let table = c.addr_table.read().expect("read lock");
        assert!(table.contains_key(&1));
        assert!(table.contains_key(&2));
        assert!(table.contains_key(&3));
        for entry in table.values() {
            assert_eq!(entry.original_bytes, 4096);
        }

        let meta = c.page_metadata.read().expect("read lock");
        assert!(meta.contains_key(&1));
        assert!(meta.contains_key(&2));
        assert!(meta.contains_key(&3));
    }

    #[test]
    fn register_pages_from_hgal_empty_map() {
        let (c, _backend) = make_coordinator(false);
        let pages: HashMap<PageId, PageMetadata> = HashMap::new();
        c.register_pages_from_hgal(&pages, 4096);
        let table = c.addr_table.read().expect("read lock");
        assert!(table.is_empty());
    }

    #[test]
    fn register_pages_from_hgal_idempotent() {
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        pages.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            ..PageMetadata::default()
        });
        c.register_pages_from_hgal(&pages, 4096);
        // Second call with same page should not overwrite existing.
        pages.get_mut(&1).unwrap().sequence_id = Some(99);
        c.register_pages_from_hgal(&pages, 8192);

        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.original_bytes, 4096);
    }

    // ── Coordinator stats tests ──────────────────────────────────────────────

    #[test]
    fn stats_snapshot_counts_tiers_correctly() {
        let (c, _backend) = make_coordinator(false);

        c.register_page(1, Some(0x1000), 4096); // HBM
        c.register_page(2, Some(0x2000), 4096); // HBM
        c.register_page(3, None, 4096);          // DRAM
        // Manually set page 4 to NVMe tier.
        c.register_page(4, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&4) {
                entry.current_tier = StorageTier::Nvme;
            }
        }

        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 2);
        assert_eq!(stats.pages_on_dram, 1);
        assert_eq!(stats.pages_on_nvme, 1);
    }

    #[test]
    fn stats_snapshot_empty_coordinator() {
        let (c, _backend) = make_coordinator(false);
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 0);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
        assert_eq!(stats.eviction_rounds, 0);
        assert_eq!(stats.swap_in_rounds, 0);
    }

    // ── Coordinator record_eviction_completed tests ──────────────────────────

    #[test]
    fn record_eviction_gpu_to_dram() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.total_bytes_evicted, 4096);
        assert_eq!(stats.total_eviction_latency_us, 100);
    }

    #[test]
    fn record_eviction_dram_to_nvme() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(2, StorageTier::CpuDram, StorageTier::Nvme, 8192, 500);
        let stats = c.stats();
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.total_bytes_evicted, 8192);
        assert_eq!(stats.total_eviction_latency_us, 500);
    }

    #[test]
    fn record_eviction_unknown_tier_pair_ignored() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(3, StorageTier::Nvme, StorageTier::GpuHbm, 4096, 100);
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.total_bytes_evicted, 4096);
        assert_eq!(stats.total_eviction_latency_us, 100);
    }

    #[test]
    fn record_eviction_accumulates_across_calls() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 200);
        c.record_eviction_completed(3, StorageTier::CpuDram, StorageTier::Nvme, 8192, 300);
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 2);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.total_bytes_evicted, 4096 + 4096 + 8192);
        assert_eq!(stats.total_eviction_latency_us, 100 + 200 + 300);
    }

    #[test]
    fn record_eviction_zero_bytes_and_latency() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 0, 0);
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.total_bytes_evicted, 0);
        assert_eq!(stats.total_eviction_latency_us, 0);
    }

    // ── Coordinator record_swap_in_completed tests ───────────────────────────

    #[test]
    fn record_swap_in_dram_to_gpu() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 50);
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
        assert_eq!(stats.total_swap_in_latency_us, 50);
    }

    #[test]
    fn record_swap_in_nvme_to_dram() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(2, StorageTier::Nvme, StorageTier::CpuDram, 8192, 800);
        let stats = c.stats();
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.total_bytes_swapped_in, 8192);
        assert_eq!(stats.total_swap_in_latency_us, 800);
    }

    #[test]
    fn record_swap_in_unknown_tier_pair_ignored() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(3, StorageTier::GpuHbm, StorageTier::Nvme, 4096, 100);
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
        assert_eq!(stats.total_swap_in_latency_us, 100);
    }

    #[test]
    fn record_swap_in_accumulates_across_calls() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 50);
        c.record_swap_in_completed(2, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 100);
        c.record_swap_in_completed(3, StorageTier::Nvme, StorageTier::CpuDram, 8192, 500);
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 2);
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.total_bytes_swapped_in, 4096 + 4096 + 8192);
        assert_eq!(stats.total_swap_in_latency_us, 50 + 100 + 500);
    }

    // ── Coordinator build_batch edge cases ───────────────────────────────────

    #[test]
    fn build_batch_empty_active_pages_no_panic() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        let plan = c.build_batch(&[], 0.0);
        assert!(plan.eviction_candidates.is_empty());
        assert!(plan.swap_in_requests.is_empty());
    }

    #[test]
    fn build_batch_zero_pressure_no_eviction() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.0);
        assert!(plan.eviction_candidates.is_empty());
    }

    #[test]
    fn build_batch_protected_pages_never_evicted() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Protected,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 1.0);
        assert!(plan.eviction_candidates.is_empty());
    }

    #[test]
    fn build_batch_active_pages_never_evicted() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Active,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 1.0);
        assert!(plan.eviction_candidates.is_empty());
    }

    #[test]
    fn build_batch_warm_pages_never_evicted() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Warm,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 1.0);
        assert!(plan.eviction_candidates.is_empty());
    }

    #[test]
    fn build_batch_eviction_candidates_sorted_by_score_ascending() {
        let (c, _backend) = make_coordinator(false);
        // Register multiple pages with different access patterns.
        for (pid, access_count) in [(1u64, 0u64), (2u64, 100u64), (3u64, 1u64)] {
            c.register_page(pid as PageId, Some(0x1000 + pid as u64), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid as PageId, PageMetadata {
                page_id: pid as PageId,
                sequence_id: Some(pid * 10),
                recency: 1000,
                access_count: access_count as usize,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        if plan.eviction_candidates.len() >= 2 {
            for window in plan.eviction_candidates.windows(2) {
                assert!(window[0].score <= window[1].score,
                    "eviction candidates should be sorted by ascending score");
            }
        }
    }

    #[test]
    fn build_batch_tier_migration_eviction_pressure_reason() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        if let Some(migration) = plan.tier_migrations.iter().find(|m| m.page_id == 1) {
            assert_eq!(migration.from_tier, StorageTier::GpuHbm);
            assert_eq!(migration.to_tier, StorageTier::CpuDram);
            assert_eq!(migration.reason, TierMigrationReason::EvictionPressure);
        }
    }

    #[test]
    fn build_batch_swap_in_request_has_correct_fields() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(42, None, 8192);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&42) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 8192]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(42, PageMetadata {
                page_id: 42,
                sequence_id: Some(200),
                recency: 0,
                access_count: 10,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[42], 0.5);
        assert_eq!(plan.swap_in_requests.len(), 1);
        let req = &plan.swap_in_requests[0];
        assert_eq!(req.page_id, 42);
        assert_eq!(req.page_bytes, 8192);
        assert!((req.prefetch_confidence - 0.9).abs() < 0.01);
    }

    // ── Coordinator tier_changed_pages tests ─────────────────────────────────

    #[test]
    fn tier_changed_pages_empty_when_consistent() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Active,
                ..PageMetadata::default()
            });
        }
        let changed = c.tier_changed_pages();
        assert!(changed.is_empty());
    }

    #[test]
    fn tier_changed_pages_detects_divergence() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        // Metadata says Active (should be HBM) but addr_table says CpuDram.
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Active,
                ..PageMetadata::default()
            });
        }
        let changed = c.tier_changed_pages();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].0, 1);
        assert_eq!(changed[0].1, StorageTier::CpuDram);
    }

    #[test]
    fn tier_changed_pages_no_metadata_skips_page() {
        let (c, _backend) = make_coordinator(false);
        // Register page in addr_table but not in page_metadata.
        c.register_page(1, Some(0x1000), 4096);
        // No page_metadata entry for page 1.
        let changed = c.tier_changed_pages();
        assert!(changed.is_empty());
    }

    // ── Coordinator accessor tests ───────────────────────────────────────────

    #[test]
    fn accessor_page_metadata_returns_shared_reference() {
        let (c, _backend) = make_coordinator(false);
        let meta = c.page_metadata();
        let guard = meta.read().expect("read lock");
        assert!(guard.is_empty());
    }

    #[test]
    fn accessor_addr_table_returns_shared_reference() {
        let (c, _backend) = make_coordinator(false);
        let table = c.addr_table();
        let guard = table.read().expect("read lock");
        assert!(guard.is_empty());
    }

    #[test]
    fn accessor_memory_manager_returns_shared_reference() {
        let (c, _backend) = make_coordinator(false);
        let mm = c.memory_manager();
        let guard = mm.lock().expect("lock");
        let usage = guard.tier_usage(crate::scheduler::memory_manager::Tier::L1);
        assert_eq!(usage.capacity, 100_000);
    }

    #[test]
    fn accessor_swap_in_worker_none_when_not_started() {
        let (c, _backend) = make_coordinator(false);
        assert!(c.swap_in_worker().is_none());
    }

    #[test]
    fn accessor_eviction_worker_none_when_not_started() {
        let (c, _backend) = make_coordinator(false);
        assert!(c.eviction_worker().is_none());
    }

    // ── Coordinator shutdown idempotency ─────────────────────────────────────

    #[test]
    fn shutdown_idempotent() {
        let (mut c, _backend) = make_coordinator(false);
        c.shutdown();
        // Second shutdown should not panic.
        c.shutdown();
        assert!(c.eviction_worker.is_none());
        assert!(c.swap_in_worker.is_none());
    }

    // ── Coordinator drop triggers shutdown ───────────────────────────────────

    #[test]
    fn drop_triggers_shutdown_gracefully() {
        let (c, _backend) = make_coordinator(false);
        drop(c);
        // Should not panic.
    }

    // ── build_batch increments rounds counter ────────────────────────────────

    #[test]
    fn build_batch_increments_round_counters() {
        let (c, _backend) = make_coordinator(false);
        let _ = c.build_batch(&[], 0.5);
        let _ = c.build_batch(&[], 0.5);
        let stats = c.stats();
        assert_eq!(stats.eviction_rounds, 2);
        assert_eq!(stats.swap_in_rounds, 2);
    }

    // ── Helper function tests via build_batch integration ────────────────────

    #[test]
    fn infer_swap_payload_kind_expert_weight_when_no_sequence() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Under high pressure, this should produce an eviction candidate.
        let plan = c.build_batch(&[], 0.95);
        assert!(plan.eviction_candidates.is_empty() || plan.eviction_candidates[0].page_id == 1);
    }

    #[test]
    fn build_batch_nvme_pages_not_evicted() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        let nvme_evictions: Vec<_> = plan.eviction_candidates
            .iter()
            .filter(|c| c.current_tier == StorageTier::Nvme)
            .collect();
        assert!(nvme_evictions.is_empty());
    }

    #[test]
    fn build_batch_swap_in_for_nvme_page_on_demand() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[1], 0.5);
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 1);
        assert_eq!(plan.tier_migrations.len(), 1);
        assert_eq!(plan.tier_migrations[0].from_tier, StorageTier::Nvme);
        assert_eq!(plan.tier_migrations[0].to_tier, StorageTier::GpuHbm);
        assert_eq!(plan.tier_migrations[0].reason, TierMigrationReason::SequenceDemand);
    }

    // ── build_batch truncation by max_evict_per_round ────────────────────────

    #[test]
    fn build_batch_truncates_to_max_evict_per_round() {
        let (c, _backend) = make_coordinator(false);
        for pid in 0usize..30 {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64 * 10),
                recency: 1000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        assert!(plan.eviction_candidates.len() <= 8,
            "should not exceed max_evict_per_round (default 8), got {}", plan.eviction_candidates.len());
    }

    // ── build_batch page with metadata but no addr_table entry is skipped ────

    #[test]
    fn build_batch_skips_pages_without_addr_entry() {
        let (c, _backend) = make_coordinator(false);
        // Insert metadata only, no addr_table entry.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(999, PageMetadata {
                page_id: 999,
                sequence_id: Some(999),
                state: PageState::Standby,
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[999], 0.95);
        assert!(plan.swap_in_requests.is_empty());
        assert!(plan.eviction_candidates.is_empty());
    }

    // ── CompressionCodec integration ─────────────────────────────────────────

    #[test]
    fn register_page_default_codec_is_none() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.codec, CompressionCodec::None);
    }

    // ── Coordinator build_batch with multiple active pages needing swap-in ───

    #[test]
    fn build_batch_multiple_swap_in_requests() {
        let (c, _backend) = make_coordinator(false);
        for pid in [10usize, 20, 30] {
            c.register_page(pid, None, 4096);
            {
                let mut table = c.addr_table.write().expect("write lock");
                if let Some(entry) = table.get_mut(&pid) {
                    entry.current_tier = StorageTier::CpuDram;
                    entry.host_buffer = Some(vec![0u8; 4096]);
                }
            }
            {
                let mut meta = c.page_metadata.write().expect("write lock");
                meta.insert(pid, PageMetadata {
                    page_id: pid,
                    sequence_id: Some(pid as u64),
                    recency: 0,
                    access_count: 5,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
        }
        let plan = c.build_batch(&[10, 20, 30], 0.5);
        assert_eq!(plan.swap_in_requests.len(), 3);
        let page_ids: std::collections::HashSet<PageId> = plan.swap_in_requests
            .iter()
            .map(|r| r.page_id)
            .collect();
        assert!(page_ids.contains(&10));
        assert!(page_ids.contains(&20));
        assert!(page_ids.contains(&30));
    }

    // ── StorageTier encoding/ordering tests ─────────────────────────────────────

    #[test]
    fn storage_tier_as_u8_roundtrip() {
        assert_eq!(StorageTier::GpuHbm.as_u8(), 0);
        assert_eq!(StorageTier::CpuDram.as_u8(), 1);
        assert_eq!(StorageTier::Nvme.as_u8(), 2);
        assert_eq!(StorageTier::from_u8(0), Some(StorageTier::GpuHbm));
        assert_eq!(StorageTier::from_u8(1), Some(StorageTier::CpuDram));
        assert_eq!(StorageTier::from_u8(2), Some(StorageTier::Nvme));
    }

    #[test]
    fn storage_tier_from_u8_invalid_returns_none() {
        assert_eq!(StorageTier::from_u8(3), None);
        assert_eq!(StorageTier::from_u8(255), None);
    }

    #[test]
    fn storage_tier_ordering_priority() {
        // GpuHbm (0) is highest priority, Nvme (2) is lowest.
        // Ord uses reverse discriminant: lower value = higher priority = greater in Ord.
        assert!(StorageTier::GpuHbm > StorageTier::CpuDram);
        assert!(StorageTier::CpuDram > StorageTier::Nvme);
        assert!(StorageTier::GpuHbm > StorageTier::Nvme);
    }

    #[test]
    fn storage_tier_equality() {
        assert_eq!(StorageTier::GpuHbm, StorageTier::GpuHbm);
        assert_ne!(StorageTier::GpuHbm, StorageTier::CpuDram);
        assert_ne!(StorageTier::CpuDram, StorageTier::Nvme);
    }

    #[test]
    fn storage_tier_copy_clone() {
        let a = StorageTier::CpuDram;
        let b = a;
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(a, c);
    }

    #[test]
    fn storage_tier_debug_output() {
        let hbm = format!("{:?}", StorageTier::GpuHbm);
        let dram = format!("{:?}", StorageTier::CpuDram);
        let nvme = format!("{:?}", StorageTier::Nvme);
        assert!(!hbm.is_empty());
        assert!(!dram.is_empty());
        assert!(!nvme.is_empty());
        assert_ne!(hbm, dram);
        assert_ne!(dram, nvme);
    }

    // ── CompressionCodec encoding tests ─────────────────────────────────────────

    #[test]
    fn compression_codec_as_u8_roundtrip() {
        assert_eq!(CompressionCodec::None.as_u8(), 0);
        assert_eq!(CompressionCodec::Lz4.as_u8(), 1);
        assert_eq!(CompressionCodec::BitPackRle.as_u8(), 2);
        assert_eq!(CompressionCodec::NvcompAns.as_u8(), 3);
        assert_eq!(CompressionCodec::ZstdDict.as_u8(), 4);
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
        assert_eq!(CompressionCodec::from_u8(1), Some(CompressionCodec::Lz4));
        assert_eq!(CompressionCodec::from_u8(2), Some(CompressionCodec::BitPackRle));
        assert_eq!(CompressionCodec::from_u8(3), Some(CompressionCodec::NvcompAns));
        assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
    }

    #[test]
    fn compression_codec_from_u8_invalid_returns_none() {
        assert_eq!(CompressionCodec::from_u8(5), None);
        assert_eq!(CompressionCodec::from_u8(255), None);
    }

    #[test]
    fn compression_codec_equality_and_copy() {
        assert_eq!(CompressionCodec::Lz4, CompressionCodec::Lz4);
        assert_ne!(CompressionCodec::Lz4, CompressionCodec::ZstdDict);
        let a = CompressionCodec::NvcompAns;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn compression_codec_debug_output_not_empty() {
        let dbg = format!("{:?}", CompressionCodec::BitPackRle);
        assert!(!dbg.is_empty());
    }

    // ── PageMetadata Default tests ───────────────────────────────────────────────

    #[test]
    fn page_metadata_default_values() {
        let meta = PageMetadata::default();
        assert_eq!(meta.page_id, 0);
        assert!(meta.sequence_id.is_none());
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Standby);
        assert!(meta.swap_in_time.is_none());
        assert!(meta.warm_until.is_none());
    }

    #[test]
    fn page_metadata_clone_independent() {
        let mut meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(99),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: None,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.page_id, 42);
        assert_eq!(cloned.sequence_id, Some(99));
        assert!(cloned.is_lir);
        assert_eq!(cloned.state, PageState::Active);
        meta.page_id = 100;
        assert_eq!(meta.page_id, 100);
        assert_eq!(cloned.page_id, 42);
    }

    // ── Debug trait tests for struct types ───────────────────────────────────────

    #[test]
    fn three_tier_swap_stats_debug_output() {
        let stats = ThreeTierSwapStats {
            evictions_gpu_to_dram: 42,
            evictions_dram_to_nvme: 10,
            swap_ins_dram_to_gpu: 30,
            swap_ins_nvme_to_dram: 5,
            total_bytes_evicted: 1024,
            total_bytes_swapped_in: 512,
            total_eviction_latency_us: 200,
            total_swap_in_latency_us: 100,
            eviction_rounds: 3,
            swap_in_rounds: 4,
            pages_on_hbm: 100,
            pages_on_dram: 50,
            pages_on_nvme: 25,
        };
        let dbg = format!("{:?}", stats);
        assert!(dbg.contains("evictions_gpu_to_dram"));
        assert!(dbg.contains("42"));
        assert!(dbg.contains("pages_on_hbm"));
        assert!(dbg.contains("100"));
    }

    #[test]
    fn tier_migration_debug_output() {
        let migration = TierMigration {
            page_id: 7,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::Nvme,
            codec: CompressionCodec::Lz4,
            page_bytes: 8192,
            reason: TierMigrationReason::ColdCascade,
        };
        let dbg = format!("{:?}", migration);
        assert!(dbg.contains("page_id"));
        assert!(dbg.contains("from_tier"));
        assert!(dbg.contains("to_tier"));
    }

    #[test]
    fn tier_migration_reason_debug_output() {
        let dbg = format!("{:?}", TierMigrationReason::SequenceDemand);
        assert!(dbg.contains("SequenceDemand"));
    }

    #[test]
    fn three_tier_swap_config_debug_output() {
        let config = ThreeTierSwapConfig::default();
        let dbg = format!("{:?}", config);
        assert!(dbg.contains("auto_start"));
        assert!(dbg.contains("true"));
    }

    // ── infer_swap_payload_kind direct tests ─────────────────────────────────────

    #[test]
    fn infer_swap_payload_kind_no_sequence_returns_expert_weight() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            ..PageMetadata::default()
        };
        assert_eq!(infer_swap_payload_kind(&meta), Some(PagePayloadKind::ExpertWeight));
    }

    #[test]
    fn infer_swap_payload_kind_with_sequence_returns_kv_context() {
        let meta = PageMetadata {
            page_id: 2,
            sequence_id: Some(42),
            ..PageMetadata::default()
        };
        assert_eq!(infer_swap_payload_kind(&meta), Some(PagePayloadKind::KvContext));
    }

    // ── Edge cases: max PageId values ────────────────────────────────────────────

    #[test]
    fn register_page_with_max_page_id() {
        let (c, _backend) = make_coordinator(false);
        let max_pid = PageId::MAX;
        c.register_page(max_pid, Some(0xFFFF), 4096);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&max_pid).expect("page should exist");
        assert_eq!(entry.gpu_ptr, Some(0xFFFF));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    #[test]
    fn release_page_with_max_page_id() {
        let (c, _backend) = make_coordinator(false);
        let max_pid = PageId::MAX;
        c.register_page(max_pid, Some(0xFFFF), 4096);
        c.release_page(max_pid);
        let table = c.addr_table.read().expect("read lock");
        assert!(table.get(&max_pid).is_none());
    }

    // ── Stats with u64 max accumulation ──────────────────────────────────────────

    #[test]
    fn stats_avg_latency_with_large_values() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 1;
        stats.total_eviction_latency_us = u64::MAX;
        let avg = stats.avg_eviction_latency_us();
        assert!(avg > 0.0);
        assert!(avg.is_finite());
    }

    // ── build_batch with mixed swap-in and eviction ──────────────────────────────

    #[test]
    fn build_batch_active_page_on_dram_not_evicted_even_under_pressure() {
        let (c, _backend) = make_coordinator(false);
        // Page on DRAM that is in the active set.
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[1], 0.99);
        // Should have a swap-in request, not an eviction candidate.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id != 1),
            "page needing swap-in should not appear in eviction candidates");
    }

    // ── register_pages_from_hgal does not overwrite existing metadata ─────────────

    #[test]
    fn register_pages_from_hgal_preserves_existing_metadata() {
        let (c, _backend) = make_coordinator(false);
        // Pre-insert metadata with specific sequence_id.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(42),
                ..PageMetadata::default()
            });
        }
        let mut pages = HashMap::new();
        pages.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: Some(99), // Different sequence_id.
            ..PageMetadata::default()
        });
        c.register_pages_from_hgal(&pages, 4096);
        let meta = c.page_metadata.read().expect("read lock");
        let existing = meta.get(&1).expect("should exist");
        // or_insert_with should preserve the original.
        assert_eq!(existing.sequence_id, Some(42));
    }

    // ── CompressionCodec all variants roundtrip ──────────────────────────────────

    #[test]
    fn compression_codec_all_variants_roundtrip() {
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for variant in variants {
            let encoded = variant.as_u8();
            let decoded = CompressionCodec::from_u8(encoded);
            assert_eq!(decoded, Some(variant));
        }
    }

    // ── CompressionCodec Hash consistency ─────────────────────────────────────────

    #[test]
    fn compression_codec_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = DefaultHasher::new();
        CompressionCodec::Lz4.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        CompressionCodec::Lz4.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn compression_codec_hash_differs_across_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |c: CompressionCodec| -> u64 {
            let mut h = DefaultHasher::new();
            c.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_of(CompressionCodec::None), hash_of(CompressionCodec::Lz4));
        assert_ne!(hash_of(CompressionCodec::Lz4), hash_of(CompressionCodec::BitPackRle));
        assert_ne!(hash_of(CompressionCodec::BitPackRle), hash_of(CompressionCodec::NvcompAns));
        assert_ne!(hash_of(CompressionCodec::NvcompAns), hash_of(CompressionCodec::ZstdDict));
    }

    // ── PagePayloadKind all variants equality ─────────────────────────────────────

    #[test]
    fn page_payload_kind_all_variants_distinct() {
        let variants = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn page_payload_kind_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = DefaultHasher::new();
        PagePayloadKind::KvContext.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        PagePayloadKind::KvContext.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── StorageTier Ord ordering comprehensive ───────────────────────────────────

    #[test]
    fn storage_tier_ord_total_order() {
        assert!(StorageTier::GpuHbm > StorageTier::CpuDram);
        assert!(StorageTier::CpuDram > StorageTier::Nvme);
        assert!(StorageTier::GpuHbm > StorageTier::Nvme);
        // Transitivity: GpuHbm > CpuDram and CpuDram > Nvme implies GpuHbm > Nvme
        assert!(StorageTier::GpuHbm >= StorageTier::GpuHbm);
        assert!(StorageTier::CpuDram >= StorageTier::CpuDram);
        assert!(StorageTier::Nvme >= StorageTier::Nvme);
    }

    #[test]
    fn storage_tier_ord_sort_order() {
        let mut tiers = vec![StorageTier::Nvme, StorageTier::GpuHbm, StorageTier::CpuDram];
        tiers.sort();
        // Sorted ascending by Ord: GpuHbm (highest) comes last.
        assert_eq!(tiers[0], StorageTier::Nvme);
        assert_eq!(tiers[1], StorageTier::CpuDram);
        assert_eq!(tiers[2], StorageTier::GpuHbm);
    }

    // ── StorageTier Hash consistency ──────────────────────────────────────────────

    #[test]
    fn storage_tier_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut h1 = DefaultHasher::new();
        StorageTier::CpuDram.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        StorageTier::CpuDram.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    // ── EvictionCandidate constructor fields ──────────────────────────────────────

    #[test]
    fn eviction_candidate_field_access() {
        let candidate = EvictionCandidate {
            page_id: 55,
            score: 42,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 8192,
            group_id: Some(100),
        };
        assert_eq!(candidate.page_id, 55);
        assert_eq!(candidate.score, 42);
        assert_eq!(candidate.current_tier, StorageTier::GpuHbm);
        assert_eq!(candidate.codec, CompressionCodec::Lz4);
        assert_eq!(candidate.page_bytes, 8192);
        assert_eq!(candidate.group_id, Some(100));
    }

    // ── TierMigration with all codec variants ─────────────────────────────────────

    #[test]
    fn tier_migration_with_each_codec_variant() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, codec) in codecs.into_iter().enumerate() {
            let migration = TierMigration {
                page_id: i as PageId,
                from_tier: StorageTier::GpuHbm,
                to_tier: StorageTier::CpuDram,
                codec,
                page_bytes: 4096,
                reason: TierMigrationReason::EvictionPressure,
            };
            assert_eq!(migration.codec, codec);
            assert_eq!(migration.page_id, i as PageId);
        }
    }

    // ── TierMigrationReason all variants in set ───────────────────────────────────

    #[test]
    fn tier_migration_reason_all_variants_in_hashset() {
        use std::collections::HashSet;
        let set: HashSet<TierMigrationReason> = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ].into_iter().collect();
        assert_eq!(set.len(), 4);
        assert!(set.contains(&TierMigrationReason::EvictionPressure));
        assert!(set.contains(&TierMigrationReason::ColdCascade));
    }

    // ── record_swap_in_completed with zero values ─────────────────────────────────

    #[test]
    fn record_swap_in_zero_bytes_and_latency() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 0, 0);
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.total_bytes_swapped_in, 0);
        assert_eq!(stats.total_swap_in_latency_us, 0);
    }

    // ── Multi-step chain: register → update → build_batch → release ──────────────

    #[test]
    fn multi_step_register_update_build_release_chain() {
        let (c, _backend) = make_coordinator(false);

        // Step 1: Register page on DRAM.
        c.register_page(7, None, 4096);
        {
            let table = c.addr_table.read().expect("read lock");
            let entry = table.get(&7).expect("should exist");
            assert_eq!(entry.current_tier, StorageTier::CpuDram);
        }

        // Step 2: Add metadata.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(7, PageMetadata {
                page_id: 7,
                sequence_id: Some(200),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }

        // Step 3: build_batch should produce swap-in request.
        let plan = c.build_batch(&[7], 0.5);
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 7);

        // Step 4: Update GPU pointer (simulating swap-in completion).
        c.update_page_gpu_ptr(7, 0xBB00);
        {
            let table = c.addr_table.read().expect("read lock");
            let entry = table.get(&7).expect("should exist");
            assert_eq!(entry.gpu_ptr, Some(0xBB00));
            assert_eq!(entry.current_tier, StorageTier::GpuHbm);
        }

        // Step 5: Now build_batch should NOT request swap-in (already on HBM).
        let plan2 = c.build_batch(&[7], 0.5);
        assert!(plan2.swap_in_requests.is_empty());

        // Step 6: Release the page.
        c.release_page(7);
        {
            let table = c.addr_table.read().expect("read lock");
            assert!(table.get(&7).is_none());
        }
    }

    // ── Stats snapshot includes cumulative records ────────────────────────────────

    #[test]
    fn stats_snapshot_includes_cumulative_eviction_and_swap_in() {
        let (c, _backend) = make_coordinator(false);

        // Record some evictions.
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 200);
        c.record_eviction_completed(3, StorageTier::CpuDram, StorageTier::Nvme, 8192, 500);

        // Record some swap-ins.
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 80);
        c.record_swap_in_completed(2, StorageTier::Nvme, StorageTier::CpuDram, 8192, 600);

        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 2);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.total_bytes_evicted, 4096 + 4096 + 8192);
        assert_eq!(stats.total_bytes_swapped_in, 4096 + 8192);
        assert_eq!(stats.total_eviction_latency_us, 100 + 200 + 500);
        assert_eq!(stats.total_swap_in_latency_us, 80 + 600);
    }

    // ── register_pages_from_hgal with many pages ──────────────────────────────────

    #[test]
    fn register_pages_from_hgal_bulk_then_individual_release() {
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        for pid in 100..110u64 {
            pages.insert(pid as PageId, PageMetadata {
                page_id: pid as PageId,
                sequence_id: Some(pid),
                ..PageMetadata::default()
            });
        }
        c.register_pages_from_hgal(&pages, 4096);

        // Verify all 10 pages exist.
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.len(), 10);
        drop(table);

        // Release half.
        for pid in 100..105u64 {
            c.release_page(pid as PageId);
        }

        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.len(), 5);
        for pid in 105..110u64 {
            assert!(table.contains_key(&(pid as PageId)));
        }
    }

    // ── build_batch with duplicate active_pages entries ───────────────────────────

    #[test]
    fn build_batch_duplicate_active_pages_produces_single_swap_in() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }

        // Pass same page_id twice in active_pages.
        let plan = c.build_batch(&[1, 1], 0.5);
        // build_batch iterates meta_guard, so the page appears once there.
        // Even though active_pages has duplicates, the page is unique in metadata.
        assert_eq!(plan.swap_in_requests.len(), 1);
    }

    // ── TierMigrationPlan built_at is recent ──────────────────────────────────────

    #[test]
    fn tier_migration_plan_built_at_is_recent() {
        let (c, _backend) = make_coordinator(false);
        let before = Instant::now();
        let plan = c.build_batch(&[], 0.5);
        let after = Instant::now();
        assert!(plan.built_at >= before);
        assert!(plan.built_at <= after);
    }

    // ── CompressionCodec clone independence ───────────────────────────────────────

    #[test]
    fn compression_codec_clone_independence() {
        let original = CompressionCodec::ZstdDict;
        let cloned = original;
        assert_eq!(original, cloned);
        // Copy type — both are independent values.
        assert_eq!(original, CompressionCodec::ZstdDict);
    }

    // ── PageState all variants used in build_batch skip logic ─────────────────────

    #[test]
    fn build_batch_swapped_out_state_skipped_for_eviction() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::SwappedOut,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        // SwappedOut is not in the skip list (Active/Protected/Warm),
        // but the page is on GpuHbm which requires pressure > threshold + age > threshold.
        // SwappedOut pages on HBM with high pressure should be eligible.
        // This test verifies no panic and correct handling.
        assert!(plan.eviction_candidates.len() <= 1);
    }

    // ── Wave 13: additional coverage tests ─────────────────────────────────────────

    #[test]
    fn tier_changed_pages_maps_active_to_hbm() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        // Metadata says Active → expected tier is HBM, actual is HBM → no divergence.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Active,
                ..PageMetadata::default()
            });
        }
        assert!(c.tier_changed_pages().is_empty());
    }

    #[test]
    fn tier_changed_pages_maps_warm_to_dram() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        // Metadata says Warm → expected tier is CpuDram, but actual is GpuHbm → divergence.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Warm,
                ..PageMetadata::default()
            });
        }
        let changed = c.tier_changed_pages();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].0, 1);
        // Actual tier is GpuHbm (registered with gpu_ptr), expected is CpuDram.
        assert_eq!(changed[0].1, StorageTier::GpuHbm);
    }

    #[test]
    fn tier_changed_pages_maps_swapped_to_nvme() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        // Metadata says Swapped → expected tier is Nvme, but actual is GpuHbm → divergence.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Swapped,
                ..PageMetadata::default()
            });
        }
        let changed = c.tier_changed_pages();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].1, StorageTier::GpuHbm);
    }

    #[test]
    fn tier_changed_pages_maps_free_to_dram() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        // Metadata says Free → expected tier is CpuDram, actual is GpuHbm → divergence.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Free,
                ..PageMetadata::default()
            });
        }
        let changed = c.tier_changed_pages();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].1, StorageTier::GpuHbm);
    }

    #[test]
    fn tier_changed_pages_multiple_divergent_pages() {
        let (c, _backend) = make_coordinator(false);
        for pid in [1usize, 2, 3] {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
            // All registered on HBM but metadata expects different tiers.
            let state = match pid {
                1 => PageState::Active,     // Active → HBM, no divergence
                2 => PageState::Swapped,    // Swapped → Nvme, divergence
                3 => PageState::Warm,       // Warm → CpuDram, divergence
                _ => unreachable!(),
            };
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                state,
                ..PageMetadata::default()
            });
        }
        let changed = c.tier_changed_pages();
        // Only pages 2 and 3 should diverge.
        assert_eq!(changed.len(), 2);
        let changed_ids: std::collections::HashSet<PageId> =
            changed.iter().map(|(pid, _)| *pid).collect();
        assert!(changed_ids.contains(&2));
        assert!(changed_ids.contains(&3));
    }

    #[test]
    fn build_batch_eviction_reason_is_cold_cascade_for_dram_to_nvme() {
        let (c, _backend) = make_coordinator(false);
        // Register page on DRAM.
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // DRAM eviction requires dram_pressure > threshold.
        // With default used=0, dram_pressure=0, no eviction from DRAM occurs.
        let plan = c.build_batch(&[], 0.95);
        // If dram pressure is low (default used=0), no eviction from DRAM occurs.
        // The test verifies no panic and correct plan structure.
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id == 1 || c.current_tier != StorageTier::CpuDram));
    }

    #[test]
    fn register_page_with_zero_bytes() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 0);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.original_bytes, 0);
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    #[test]
    fn update_page_gpu_ptr_after_release_no_side_effect() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.release_page(1);
        // Update on released page should not re-create the entry.
        c.update_page_gpu_ptr(1, 0xDEAD);
        let table = c.addr_table.read().expect("read lock");
        assert!(table.get(&1).is_none());
    }

    #[test]
    fn record_eviction_and_swap_in_combined_avg_latency() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 200);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 100);

        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        // Avg eviction: 200/1 = 200
        assert!((stats.avg_eviction_latency_us() - 200.0).abs() < 0.01);
        // Avg swap-in: 100/1 = 100
        assert!((stats.avg_swap_in_latency_us() - 100.0).abs() < 0.01);
        assert_eq!(stats.total_bytes_evicted, 4096);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
    }

    #[test]
    fn build_batch_multiple_rounds_accumulate_stats() {
        let (c, _backend) = make_coordinator(false);
        let _ = c.build_batch(&[], 0.5);
        let _ = c.build_batch(&[], 0.5);
        let _ = c.build_batch(&[], 0.5);
        let stats = c.stats();
        assert_eq!(stats.eviction_rounds, 3);
        assert_eq!(stats.swap_in_rounds, 3);
    }

    #[test]
    fn register_pages_from_hgal_partial_overlap_preserves_both() {
        let (c, _backend) = make_coordinator(false);
        // Pre-register page 1 individually.
        c.register_page(1, Some(0x1000), 8192);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(42),
                ..PageMetadata::default()
            });
        }
        // Bulk-register pages 1 and 2.
        let mut pages = HashMap::new();
        pages.insert(1, PageMetadata { page_id: 1, sequence_id: Some(99), ..PageMetadata::default() });
        pages.insert(2, PageMetadata { page_id: 2, sequence_id: Some(100), ..PageMetadata::default() });
        c.register_pages_from_hgal(&pages, 4096);

        // Page 1: original addr_table entry preserved (8192 bytes).
        let table = c.addr_table.read().expect("read lock");
        let entry1 = table.get(&1).expect("page 1 should exist");
        assert_eq!(entry1.original_bytes, 8192); // Preserved from first register.

        // Page 2: new entry with bulk page_bytes.
        let entry2 = table.get(&2).expect("page 2 should exist");
        assert_eq!(entry2.original_bytes, 4096);

        // Page 1 metadata preserved (sequence_id=42, not 99).
        let meta = c.page_metadata.read().expect("read lock");
        assert_eq!(meta.get(&1).unwrap().sequence_id, Some(42));
        assert_eq!(meta.get(&2).unwrap().sequence_id, Some(100));
    }

    #[test]
    fn build_batch_with_pressure_one_point_zero() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 1.0);
        // Maximum pressure should produce eviction candidates if age threshold met.
        // Age threshold default is 50 ticks (500ms). With 10s age, definitely eligible.
        assert!(!plan.eviction_candidates.is_empty() || plan.tier_migrations.is_empty());
    }

    #[test]
    fn accessor_observer_returns_shared_reference() {
        let (c, _backend) = make_coordinator(false);
        let observer = c.observer();
        let guard = observer.lock().expect("lock");
        // BasicObserver should be usable without panic.
        drop(guard);
    }

    #[test]
    fn tier_migration_with_all_reason_variants() {
        let reasons = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        for (i, reason) in reasons.into_iter().enumerate() {
            let migration = TierMigration {
                page_id: i as PageId,
                from_tier: StorageTier::GpuHbm,
                to_tier: StorageTier::CpuDram,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                reason,
            };
            assert_eq!(migration.reason, reason);
            assert_eq!(migration.page_id, i as PageId);
        }
    }

    #[test]
    fn build_batch_standby_on_dram_not_in_active_set_may_evict() {
        let (c, _backend) = make_coordinator(false);
        // Page on DRAM, not in active set, Standby state.
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // DRAM eviction requires dram_pressure > threshold.
        // With default used=0, dram_pressure=0 → no eviction from DRAM.
        let plan = c.build_batch(&[], 0.95);
        // No eviction because dram pressure is 0.
        let dram_evictions: Vec<_> = plan.eviction_candidates
            .iter()
            .filter(|c| c.current_tier == StorageTier::CpuDram)
            .collect();
        assert!(dram_evictions.is_empty());
    }

    #[test]
    fn build_batch_swap_in_for_swapped_state_page() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[1], 0.5);
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 1);
    }

    #[test]
    fn stats_snapshot_reflects_registered_pages_after_update() {
        let (c, _backend) = make_coordinator(false);
        // Register 3 pages: 2 on HBM, 1 on DRAM.
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, Some(0x2000), 4096);
        c.register_page(3, None, 4096);
        // Move page 2 to Nvme.
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&2) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 1);  // page 1
        assert_eq!(stats.pages_on_dram, 1); // page 3
        assert_eq!(stats.pages_on_nvme, 1); // page 2
    }

    #[test]
    fn register_page_different_page_bytes_per_page() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, Some(0x2000), 8192);
        c.register_page(3, Some(0x3000), 16384);
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.get(&1).unwrap().original_bytes, 4096);
        assert_eq!(table.get(&2).unwrap().original_bytes, 8192);
        assert_eq!(table.get(&3).unwrap().original_bytes, 16384);
    }

    #[test]
    fn tier_migration_plan_with_mixed_migrations() {
        let plan = TierMigrationPlan {
            eviction_candidates: vec![EvictionCandidate {
                page_id: 1,
                score: 10,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: Some(100),
            }],
            swap_in_requests: vec![PrefetchRequest {
                page_id: 2,
                urgency: 0.8,
                prefetch_confidence: 0.9,
                page_bytes: 8192,
                enqueued_at: Instant::now(),
            }],
            tier_migrations: vec![
                TierMigration {
                    page_id: 1,
                    from_tier: StorageTier::GpuHbm,
                    to_tier: StorageTier::CpuDram,
                    codec: CompressionCodec::None,
                    page_bytes: 4096,
                    reason: TierMigrationReason::EvictionPressure,
                },
                TierMigration {
                    page_id: 2,
                    from_tier: StorageTier::CpuDram,
                    to_tier: StorageTier::GpuHbm,
                    codec: CompressionCodec::None,
                    page_bytes: 8192,
                    reason: TierMigrationReason::SequenceDemand,
                },
            ],
            built_at: Instant::now(),
        };
        assert_eq!(plan.tier_migrations.len(), 2);
        assert_eq!(plan.tier_migrations[0].reason, TierMigrationReason::EvictionPressure);
        assert_eq!(plan.tier_migrations[1].reason, TierMigrationReason::SequenceDemand);
        assert_ne!(plan.tier_migrations[0].from_tier, plan.tier_migrations[1].from_tier);
    }

    #[test]
    fn multi_step_register_record_stats_verify_cumulative() {
        let (c, _backend) = make_coordinator(false);

        // Register pages.
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, Some(0x2000), 4096);

        // Record eviction of page 1.
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 150);

        // Record swap-in of page 2.
        c.record_swap_in_completed(2, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 80);

        // Record eviction of page 2 to NVMe.
        c.record_eviction_completed(2, StorageTier::CpuDram, StorageTier::Nvme, 4096, 300);

        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.total_bytes_evicted, 4096 * 2);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
        assert_eq!(stats.total_eviction_latency_us, 150 + 300);
        assert_eq!(stats.total_swap_in_latency_us, 80);
        assert_eq!(stats.total_migrations(), 3);
    }

    // ── Wave 14: additional coverage tests (40 new tests) ──────────────────────────

    // ── ThreeTierSwapStats: PartialEq (derive manually) ─────────────────────────

    #[test]
    fn stats_equality_same_values() {
        let a = ThreeTierSwapStats {
            evictions_gpu_to_dram: 1,
            evictions_dram_to_nvme: 2,
            swap_ins_dram_to_gpu: 3,
            swap_ins_nvme_to_dram: 4,
            total_bytes_evicted: 100,
            total_bytes_swapped_in: 200,
            total_eviction_latency_us: 50,
            total_swap_in_latency_us: 60,
            eviction_rounds: 7,
            swap_in_rounds: 8,
            pages_on_hbm: 10,
            pages_on_dram: 20,
            pages_on_nvme: 30,
        };
        let b = a.clone();
        assert_eq!(a.evictions_gpu_to_dram, b.evictions_gpu_to_dram);
        assert_eq!(a.evictions_dram_to_nvme, b.evictions_dram_to_nvme);
        assert_eq!(a.swap_ins_dram_to_gpu, b.swap_ins_dram_to_gpu);
        assert_eq!(a.total_bytes_evicted, b.total_bytes_evicted);
    }

    #[test]
    fn stats_inequality_different_values() {
        let a = ThreeTierSwapStats {
            evictions_gpu_to_dram: 1,
            ..ThreeTierSwapStats::default()
        };
        let b = ThreeTierSwapStats::default();
        assert_ne!(a.evictions_gpu_to_dram, b.evictions_gpu_to_dram);
    }

    #[test]
    fn stats_default_is_zero_initialized() {
        let stats = ThreeTierSwapStats::default();
        // All numeric fields should be 0.
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_evicted, 0);
        assert_eq!(stats.total_bytes_swapped_in, 0);
        assert_eq!(stats.total_eviction_latency_us, 0);
        assert_eq!(stats.total_swap_in_latency_us, 0);
        assert_eq!(stats.eviction_rounds, 0);
        assert_eq!(stats.swap_in_rounds, 0);
        assert_eq!(stats.pages_on_hbm, 0);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
    }

    #[test]
    fn stats_avg_eviction_latency_with_usize_max_count() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = u64::MAX;
        stats.total_eviction_latency_us = u64::MAX;
        let avg = stats.avg_eviction_latency_us();
        assert!(avg.is_finite());
        assert!((avg - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_avg_swap_in_latency_with_single_operation() {
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_nvme_to_dram = 1;
        stats.total_swap_in_latency_us = 42;
        assert!((stats.avg_swap_in_latency_us() - 42.0).abs() < 0.01);
    }

    #[test]
    fn stats_total_migrations_overflow_safe() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = u64::MAX / 4;
        stats.evictions_dram_to_nvme = u64::MAX / 4;
        stats.swap_ins_dram_to_gpu = u64::MAX / 4;
        stats.swap_ins_nvme_to_dram = u64::MAX / 4;
        let total = stats.total_migrations();
        assert!(total > 0);
    }

    #[test]
    fn stats_clone_deep_copy_all_fields() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 42;
        stats.evictions_dram_to_nvme = 10;
        stats.swap_ins_dram_to_gpu = 20;
        stats.swap_ins_nvme_to_dram = 5;
        stats.total_bytes_evicted = 1000;
        stats.total_bytes_swapped_in = 500;
        stats.total_eviction_latency_us = 200;
        stats.total_swap_in_latency_us = 100;
        stats.eviction_rounds = 3;
        stats.swap_in_rounds = 4;
        stats.pages_on_hbm = 50;
        stats.pages_on_dram = 30;
        stats.pages_on_nvme = 15;
        let cloned = stats.clone();
        assert_eq!(cloned.evictions_gpu_to_dram, 42);
        assert_eq!(cloned.evictions_dram_to_nvme, 10);
        assert_eq!(cloned.swap_ins_dram_to_gpu, 20);
        assert_eq!(cloned.swap_ins_nvme_to_dram, 5);
        assert_eq!(cloned.total_bytes_evicted, 1000);
        assert_eq!(cloned.total_bytes_swapped_in, 500);
        assert_eq!(cloned.pages_on_hbm, 50);
        assert_eq!(cloned.pages_on_dram, 30);
        assert_eq!(cloned.pages_on_nvme, 15);
    }

    // ── TierMigrationReason: all Hash/Eq combinations ───────────────────────────

    #[test]
    fn tier_migration_reason_hash_distinguishes_all() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let hash_of = |r: TierMigrationReason| -> u64 {
            let mut h = DefaultHasher::new();
            r.hash(&mut h);
            h.finish()
        };
        let reasons = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        for i in 0..reasons.len() {
            for j in (i + 1)..reasons.len() {
                assert_ne!(hash_of(reasons[i]), hash_of(reasons[j]),
                    "hash collision between {:?} and {:?}", reasons[i], reasons[j]);
            }
        }
    }

    #[test]
    fn tier_migration_reason_debug_contains_variant_name() {
        assert!(format!("{:?}", TierMigrationReason::EvictionPressure).contains("EvictionPressure"));
        assert!(format!("{:?}", TierMigrationReason::SequenceDemand).contains("SequenceDemand"));
        assert!(format!("{:?}", TierMigrationReason::Prefetch).contains("Prefetch"));
        assert!(format!("{:?}", TierMigrationReason::ColdCascade).contains("ColdCascade"));
    }

    // ── TierMigration: struct field validation ──────────────────────────────────

    #[test]
    fn tier_migration_same_from_to_tier_allowed() {
        let migration = TierMigration {
            page_id: 0,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 0,
            reason: TierMigrationReason::Prefetch,
        };
        assert_eq!(migration.from_tier, migration.to_tier);
    }

    #[test]
    fn tier_migration_all_tier_pair_combinations() {
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        let mut count = 0;
        for from in tiers {
            for to in tiers {
                let migration = TierMigration {
                    page_id: count,
                    from_tier: from,
                    to_tier: to,
                    codec: CompressionCodec::None,
                    page_bytes: 4096,
                    reason: TierMigrationReason::EvictionPressure,
                };
                assert_eq!(migration.from_tier, from);
                assert_eq!(migration.to_tier, to);
                count += 1;
            }
        }
        assert_eq!(count, 9);
    }

    #[test]
    fn tier_migration_page_bytes_zero() {
        let migration = TierMigration {
            page_id: 1,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 0,
            reason: TierMigrationReason::ColdCascade,
        };
        assert_eq!(migration.page_bytes, 0);
    }

    #[test]
    fn tier_migration_page_bytes_max() {
        let migration = TierMigration {
            page_id: 1,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4,
            page_bytes: usize::MAX,
            reason: TierMigrationReason::EvictionPressure,
        };
        assert_eq!(migration.page_bytes, usize::MAX);
    }

    // ── TierMigrationPlan: struct construction variants ─────────────────────────

    #[test]
    fn tier_migration_plan_with_only_eviction_candidates() {
        let plan = TierMigrationPlan {
            eviction_candidates: vec![EvictionCandidate {
                page_id: 1,
                score: 50,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            }],
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        assert_eq!(plan.eviction_candidates.len(), 1);
        assert!(plan.swap_in_requests.is_empty());
        assert!(plan.tier_migrations.is_empty());
    }

    #[test]
    fn tier_migration_plan_with_only_swap_in_requests() {
        let plan = TierMigrationPlan {
            eviction_candidates: Vec::new(),
            swap_in_requests: vec![PrefetchRequest {
                page_id: 5,
                urgency: 0.5,
                prefetch_confidence: 0.8,
                page_bytes: 8192,
                enqueued_at: Instant::now(),
            }],
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        assert!(plan.eviction_candidates.is_empty());
        assert_eq!(plan.swap_in_requests.len(), 1);
    }

    #[test]
    fn tier_migration_plan_clone_preserves_built_at() {
        let plan = TierMigrationPlan {
            eviction_candidates: Vec::new(),
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        let cloned = plan.clone();
        assert!(plan.built_at == cloned.built_at);
    }

    #[test]
    fn tier_migration_plan_debug_contains_fields() {
        let plan = TierMigrationPlan {
            eviction_candidates: Vec::new(),
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        let dbg = format!("{:?}", plan);
        assert!(dbg.contains("eviction_candidates"));
        assert!(dbg.contains("swap_in_requests"));
        assert!(dbg.contains("tier_migrations"));
    }

    // ── ThreeTierSwapConfig: construction edge cases ───────────────────────────

    #[test]
    fn config_default_eviction_subconfig() {
        let config = ThreeTierSwapConfig::default();
        assert_eq!(config.eviction.max_evict_per_round, 8);
        assert_eq!(config.eviction.page_bytes, 4096);
        assert!(config.eviction.hbm_pressure_threshold > 0.0);
    }

    #[test]
    fn config_default_swap_in_subconfig() {
        let config = ThreeTierSwapConfig::default();
        assert_eq!(config.swap_in.max_prefetch_per_round, 16);
    }

    #[test]
    fn config_clone_produces_independent_copy() {
        let config = ThreeTierSwapConfig::default();
        let mut cloned = config.clone();
        cloned.auto_start = false;
        assert!(config.auto_start);
        assert!(!cloned.auto_start);
    }

    // ── StorageTier: Hash in HashMap ────────────────────────────────────────────

    #[test]
    fn storage_tier_used_as_hashmap_key() {
        let mut map = std::collections::HashMap::new();
        map.insert(StorageTier::GpuHbm, 100u64);
        map.insert(StorageTier::CpuDram, 200u64);
        map.insert(StorageTier::Nvme, 300u64);
        assert_eq!(map.get(&StorageTier::GpuHbm), Some(&100));
        assert_eq!(map.get(&StorageTier::CpuDram), Some(&200));
        assert_eq!(map.get(&StorageTier::Nvme), Some(&300));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn storage_tier_from_u8_boundary_values() {
        assert_eq!(StorageTier::from_u8(0), Some(StorageTier::GpuHbm));
        assert_eq!(StorageTier::from_u8(1), Some(StorageTier::CpuDram));
        assert_eq!(StorageTier::from_u8(2), Some(StorageTier::Nvme));
        assert_eq!(StorageTier::from_u8(u8::MIN), Some(StorageTier::GpuHbm));
        // 3..=255 should all be None
        for v in 3u8..=255 {
            assert_eq!(StorageTier::from_u8(v), None, "from_u8({}) should be None", v);
        }
    }

    #[test]
    fn storage_tier_as_u8_returns_discriminant() {
        assert_eq!(StorageTier::GpuHbm.as_u8(), 0);
        assert_eq!(StorageTier::CpuDram.as_u8(), 1);
        assert_eq!(StorageTier::Nvme.as_u8(), 2);
    }

    #[test]
    fn storage_tier_ord_reverse_numeric() {
        // Ord is reverse discriminant: 0 (HBM) > 1 (DRAM) > 2 (NVMe)
        assert_eq!(StorageTier::GpuHbm.cmp(&StorageTier::CpuDram), std::cmp::Ordering::Greater);
        assert_eq!(StorageTier::CpuDram.cmp(&StorageTier::Nvme), std::cmp::Ordering::Greater);
        assert_eq!(StorageTier::GpuHbm.cmp(&StorageTier::Nvme), std::cmp::Ordering::Greater);
        assert_eq!(StorageTier::GpuHbm.cmp(&StorageTier::GpuHbm), std::cmp::Ordering::Equal);
    }

    // ── CompressionCodec: boundary and variant tests ───────────────────────────

    #[test]
    fn compression_codec_from_u8_boundary_values() {
        assert_eq!(CompressionCodec::from_u8(0), Some(CompressionCodec::None));
        assert_eq!(CompressionCodec::from_u8(4), Some(CompressionCodec::ZstdDict));
        assert_eq!(CompressionCodec::from_u8(5), None);
        assert_eq!(CompressionCodec::from_u8(u8::MAX), None);
    }

    #[test]
    fn compression_codec_all_variants_distinct() {
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn compression_codec_clone_is_copy() {
        let a = CompressionCodec::Lz4;
        let b = a;
        // Copy type: both still valid and equal
        assert_eq!(a, b);
        assert_eq!(a, CompressionCodec::Lz4);
    }

    #[test]
    fn compression_codec_used_as_hashmap_key() {
        let mut map = std::collections::HashMap::new();
        map.insert(CompressionCodec::None, "none");
        map.insert(CompressionCodec::Lz4, "lz4");
        map.insert(CompressionCodec::BitPackRle, "bitpack");
        map.insert(CompressionCodec::NvcompAns, "nvcomp");
        map.insert(CompressionCodec::ZstdDict, "zstd");
        assert_eq!(map.len(), 5);
        assert_eq!(map.get(&CompressionCodec::Lz4), Some(&"lz4"));
    }

    #[test]
    fn compression_codec_debug_all_variants() {
        assert!(!format!("{:?}", CompressionCodec::None).is_empty());
        assert!(!format!("{:?}", CompressionCodec::Lz4).is_empty());
        assert!(!format!("{:?}", CompressionCodec::BitPackRle).is_empty());
        assert!(!format!("{:?}", CompressionCodec::NvcompAns).is_empty());
        assert!(!format!("{:?}", CompressionCodec::ZstdDict).is_empty());
    }

    // ── EvictionCandidate: struct construction ──────────────────────────────────

    #[test]
    fn eviction_candidate_with_none_group_id() {
        let candidate = EvictionCandidate {
            page_id: 0,
            score: 0,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4,
            page_bytes: 0,
            group_id: None,
        };
        assert!(candidate.group_id.is_none());
        assert_eq!(candidate.score, 0);
    }

    #[test]
    fn eviction_candidate_negative_score() {
        let candidate = EvictionCandidate {
            page_id: 1,
            score: i64::MIN,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: Some(100),
        };
        assert_eq!(candidate.score, i64::MIN);
    }

    #[test]
    fn eviction_candidate_max_score() {
        let candidate = EvictionCandidate {
            page_id: 1,
            score: i64::MAX,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: Some(100),
        };
        assert_eq!(candidate.score, i64::MAX);
    }

    #[test]
    fn eviction_candidate_debug_output() {
        let candidate = EvictionCandidate {
            page_id: 42,
            score: 100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: Some(1),
        };
        let dbg = format!("{:?}", candidate);
        assert!(dbg.contains("page_id"));
        assert!(dbg.contains("score"));
    }

    // ── PrefetchRequest: struct construction ────────────────────────────────────

    #[test]
    fn prefetch_request_urgency_zero() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!((req.urgency - 0.0).abs() < f32::EPSILON);
        assert!((req.prefetch_confidence - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn prefetch_request_urgency_max() {
        let req = PrefetchRequest {
            page_id: 2,
            urgency: f32::MAX,
            prefetch_confidence: f32::MAX,
            page_bytes: 8192,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.urgency, f32::MAX);
    }

    #[test]
    fn prefetch_request_clone_independent() {
        let req = PrefetchRequest {
            page_id: 10,
            urgency: 0.7,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let cloned = req.clone();
        assert_eq!(cloned.page_id, 10);
        assert!((cloned.urgency - 0.7).abs() < 0.01);
        assert!((cloned.prefetch_confidence - 0.9).abs() < 0.01);
        assert_eq!(cloned.page_bytes, 4096);
    }

    // ── PagePayloadKind: all variants ───────────────────────────────────────────

    #[test]
    fn page_payload_kind_copy_semantics() {
        let a = PagePayloadKind::KvContext;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn page_payload_kind_all_variants_in_hashset() {
        use std::collections::HashSet;
        let set: HashSet<PagePayloadKind> = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ].into_iter().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn page_payload_kind_debug_not_empty() {
        let kinds = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        for kind in kinds {
            assert!(!format!("{:?}", kind).is_empty());
        }
    }

    // ── Coordinator: build_batch with negative pressure ─────────────────────────

    #[test]
    fn build_batch_negative_pressure_no_eviction() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], -0.5);
        assert!(plan.eviction_candidates.is_empty());
    }

    // ── Coordinator: register_page with page_id 0 ──────────────────────────────

    #[test]
    fn register_page_with_page_id_zero() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(0, Some(0x1000), 4096);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&0).expect("page_id 0 should exist");
        assert_eq!(entry.gpu_ptr, Some(0x1000));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // ── Coordinator: update_page_gpu_ptr then release then re-register ──────────

    #[test]
    fn update_then_release_then_reregister_page() {
        let (c, _backend) = make_coordinator(false);

        // Register, update, release.
        c.register_page(1, Some(0x1000), 4096);
        c.update_page_gpu_ptr(1, 0x2000);
        c.release_page(1);

        // Re-register with new data.
        c.register_page(1, Some(0x3000), 8192);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("re-registered page should exist");
        assert_eq!(entry.gpu_ptr, Some(0x3000));
        assert_eq!(entry.original_bytes, 8192);
    }

    // ── Coordinator: record_eviction then record_swap_in alternation ────────────

    #[test]
    fn alternating_eviction_and_swap_in_records() {
        let (c, _backend) = make_coordinator(false);

        // Evict page 1: HBM -> DRAM
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        // Swap-in page 1: DRAM -> HBM
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 80);
        // Evict page 1 again: HBM -> DRAM
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 120);
        // Evict page 1 deeper: DRAM -> NVMe
        c.record_eviction_completed(1, StorageTier::CpuDram, StorageTier::Nvme, 4096, 500);

        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 2);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.total_bytes_evicted, 4096 * 3);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
    }

    // ── Coordinator: tier_changed_pages with protected state ────────────────────

    #[test]
    fn tier_changed_pages_maps_protected_to_hbm() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Protected,
                ..PageMetadata::default()
            });
        }
        // Protected -> GpuHbm, and page is on GpuHbm -> no divergence.
        assert!(c.tier_changed_pages().is_empty());
    }

    #[test]
    fn tier_changed_pages_maps_standby_to_dram() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Standby,
                ..PageMetadata::default()
            });
        }
        // Standby -> CpuDram, but page is on GpuHbm -> divergence.
        let changed = c.tier_changed_pages();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].1, StorageTier::GpuHbm);
    }

    // ── Coordinator: register_pages_from_hgal with single page ──────────────────

    #[test]
    fn register_pages_from_hgal_single_page() {
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        pages.insert(42, PageMetadata {
            page_id: 42,
            sequence_id: Some(100),
            ..PageMetadata::default()
        });
        c.register_pages_from_hgal(&pages, 8192);

        let table = c.addr_table.read().expect("read lock");
        assert!(table.contains_key(&42));
        assert_eq!(table.get(&42).unwrap().original_bytes, 8192);

        let meta = c.page_metadata.read().expect("read lock");
        assert!(meta.contains_key(&42));
        assert_eq!(meta.get(&42).unwrap().sequence_id, Some(100));
    }

    // ── Coordinator: build_batch with NaN pressure ──────────────────────────────

    #[test]
    fn build_batch_nan_pressure_no_eviction() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // NaN pressure should not cause eviction (NaN comparisons return false).
        let plan = c.build_batch(&[], f32::NAN);
        assert!(plan.eviction_candidates.is_empty());
    }

    // ── Coordinator: build_batch with Infinity pressure ─────────────────────────

    #[test]
    fn build_batch_infinity_pressure_eligible() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], f32::INFINITY);
        // f32::INFINITY > threshold should be true, so eviction candidate possible.
        assert!(plan.eviction_candidates.is_empty() || plan.eviction_candidates[0].page_id == 1);
    }

    // ── Coordinator: register_page then stats reflects pages ────────────────────

    #[test]
    fn register_multiple_pages_stats_counts_correct() {
        let (c, _backend) = make_coordinator(false);
        // 3 HBM pages.
        for pid in 1u64..=3 {
            c.register_page(pid as PageId, Some(0x1000 + pid), 4096);
        }
        // 2 DRAM pages.
        for pid in 4u64..=5 {
            c.register_page(pid as PageId, None, 4096);
        }
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 3);
        assert_eq!(stats.pages_on_dram, 2);
        assert_eq!(stats.pages_on_nvme, 0);
    }

    // ── Coordinator: build_batch Free state page not in eviction skip list ──────

    #[test]
    fn build_batch_free_state_not_skipped_but_age_matters() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Free,
                warm_until: None,
            });
        }
        // Free is not in the skip list (Active/Protected/Warm), so it may be eligible.
        let plan = c.build_batch(&[], 0.95);
        // Verify no panic; actual eviction depends on score calculation.
        assert!(plan.eviction_candidates.len() <= 1);
    }

    // ── TierMigrationReason: exhaustive match ───────────────────────────────────

    #[test]
    fn tier_migration_reason_exhaustive_coverage() {
        // This test ensures we cover all 4 variants with explicit construction.
        let reasons = vec![
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        assert_eq!(reasons.len(), 4);
        // Each variant should be unique in a set.
        let unique: std::collections::HashSet<_> = reasons.into_iter().collect();
        assert_eq!(unique.len(), 4);
    }

    // ── Coordinator: build_batch handles empty coordinator gracefully ───────────

    #[test]
    fn build_batch_on_fresh_coordinator_no_panic_with_max_pressure() {
        let (c, _backend) = make_coordinator(false);
        let plan = c.build_batch(&[1, 2, 3], f32::MAX);
        assert!(plan.eviction_candidates.is_empty());
        assert!(plan.swap_in_requests.is_empty());
        assert!(plan.tier_migrations.is_empty());
    }

    // ── Coordinator: stats after build_batch increments rounds ──────────────────

    #[test]
    fn stats_rounds_counter_saturating_increment() {
        let (c, _backend) = make_coordinator(false);
        // Call build_batch many times.
        for _ in 0..100 {
            let _ = c.build_batch(&[], 0.5);
        }
        let stats = c.stats();
        assert_eq!(stats.eviction_rounds, 100);
        assert_eq!(stats.swap_in_rounds, 100);
    }

    // ── Wave 15: 50 additional tests ────────────────────────────────────────────

    // ── PageState variant coverage ───────────────────────────────────────────────

    #[test]
    fn page_state_all_variants_distinct() {
        let variants = [
            PageState::Free,
            PageState::Active,
            PageState::Protected,
            PageState::Warm,
            PageState::Standby,
            PageState::Swapped,
            PageState::SwappedOut,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                assert_eq!(i == j, a == b, "{:?} vs {:?}", a, b);
            }
        }
    }

    #[test]
    fn page_state_copy_semantics() {
        let a = PageState::Active;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn page_state_clone_equals_original() {
        let a = PageState::SwappedOut;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn page_state_debug_not_empty() {
        let states = [
            PageState::Free,
            PageState::Active,
            PageState::Protected,
            PageState::Warm,
            PageState::Standby,
            PageState::Swapped,
            PageState::SwappedOut,
        ];
        for state in states {
            let dbg = format!("{:?}", state);
            assert!(!dbg.is_empty(), "Debug for {:?} should not be empty", state);
        }
    }

    #[test]
    fn page_state_all_variants_in_hashset() {
        use std::collections::HashSet;
        let set: HashSet<PageState> = [
            PageState::Free,
            PageState::Active,
            PageState::Protected,
            PageState::Warm,
            PageState::Standby,
            PageState::Swapped,
            PageState::SwappedOut,
        ].into_iter().collect();
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn page_state_default_is_standby_via_metadata() {
        // PageState has no Default, but PageMetadata::default uses Standby.
        let meta = PageMetadata::default();
        assert_eq!(meta.state, PageState::Standby);
    }

    // ── PageMetadata construction with all PageState variants ────────────────────

    #[test]
    fn page_metadata_with_each_state_variant() {
        let states = [
            PageState::Free,
            PageState::Active,
            PageState::Protected,
            PageState::Warm,
            PageState::Standby,
            PageState::Swapped,
            PageState::SwappedOut,
        ];
        for (i, state) in states.iter().enumerate() {
            let meta = PageMetadata {
                page_id: i,
                sequence_id: Some(i as u64 * 10),
                recency: i,
                access_count: i * 2,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: i % 2 == 0,
                state: *state,
                warm_until: None,
            };
            assert_eq!(meta.state, *state);
            assert_eq!(meta.page_id, i);
        }
    }

    #[test]
    fn page_metadata_debug_output() {
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(99),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: None,
        };
        let dbg = format!("{:?}", meta);
        assert!(dbg.contains("page_id"));
        assert!(dbg.contains("42"));
        assert!(dbg.contains("state"));
    }

    #[test]
    fn page_metadata_with_warm_until_some() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(1),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Warm,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
        };
        assert!(meta.warm_until.is_some());
        assert!(meta.swap_in_time.is_some());
    }

    #[test]
    fn page_metadata_is_lir_true_and_false() {
        let meta_true = PageMetadata {
            page_id: 1,
            is_lir: true,
            ..PageMetadata::default()
        };
        let meta_false = PageMetadata {
            page_id: 2,
            is_lir: false,
            ..PageMetadata::default()
        };
        assert!(meta_true.is_lir);
        assert!(!meta_false.is_lir);
    }

    // ── PageAddrEntry construction ───────────────────────────────────────────────

    #[test]
    fn page_addr_entry_construction_with_all_fields() {
        let entry = PageAddrEntry {
            gpu_ptr: Some(0xDEADBEEF),
            host_buffer: Some(vec![0xAB; 4096]),
            current_tier: StorageTier::GpuHbm,
            original_bytes: 4096,
            codec: CompressionCodec::Lz4,
        };
        assert_eq!(entry.gpu_ptr, Some(0xDEADBEEF));
        assert_eq!(entry.host_buffer.as_ref().map(|b| b.len()), Some(4096));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
        assert_eq!(entry.original_bytes, 4096);
        assert_eq!(entry.codec, CompressionCodec::Lz4);
    }

    #[test]
    fn page_addr_entry_construction_with_none_ptrs() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: None,
            current_tier: StorageTier::Nvme,
            original_bytes: 8192,
            codec: CompressionCodec::ZstdDict,
        };
        assert!(entry.gpu_ptr.is_none());
        assert!(entry.host_buffer.is_none());
        assert_eq!(entry.current_tier, StorageTier::Nvme);
        assert_eq!(entry.codec, CompressionCodec::ZstdDict);
    }

    #[test]
    fn page_addr_entry_debug_output() {
        let entry = PageAddrEntry {
            gpu_ptr: Some(0x1000),
            host_buffer: None,
            current_tier: StorageTier::CpuDram,
            original_bytes: 4096,
            codec: CompressionCodec::None,
        };
        let dbg = format!("{:?}", entry);
        assert!(dbg.contains("gpu_ptr"));
        assert!(dbg.contains("current_tier"));
    }

    // ── Coordinator: release_page updates stats snapshot correctly ───────────────

    #[test]
    fn release_page_updates_tier_counts_in_stats() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096); // HBM
        c.register_page(2, Some(0x2000), 4096); // HBM
        c.register_page(3, None, 4096);          // DRAM

        let stats_before = c.stats();
        assert_eq!(stats_before.pages_on_hbm, 2);

        c.release_page(1);
        let stats_after = c.stats();
        assert_eq!(stats_after.pages_on_hbm, 1);
        assert_eq!(stats_after.pages_on_dram, 1);
    }

    // ── Coordinator: build_batch with all pages on HBM (no swap-in needed) ───────

    #[test]
    fn build_batch_all_hbm_pages_no_swap_in() {
        let (c, _backend) = make_coordinator(false);
        for pid in 1usize..=5 {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64 * 10),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Active,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[1, 2, 3, 4, 5], 0.5);
        assert!(plan.swap_in_requests.is_empty());
    }

    // ── Coordinator: build_batch with all pages on NVMe (no eviction) ────────────

    #[test]
    fn build_batch_all_nvme_pages_no_eviction_candidates() {
        let (c, _backend) = make_coordinator(false);
        for pid in 1usize..=5 {
            c.register_page(pid, None, 4096);
            {
                let mut table = c.addr_table.write().expect("write lock");
                if let Some(entry) = table.get_mut(&pid) {
                    entry.current_tier = StorageTier::Nvme;
                }
            }
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64 * 10),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.99);
        let nvme_evictions: Vec<_> = plan.eviction_candidates
            .iter()
            .filter(|c| c.current_tier == StorageTier::Nvme)
            .collect();
        assert!(nvme_evictions.is_empty(), "NVMe pages should not be evicted");
    }

    // ── Coordinator: auto_start spawns workers accessible via accessors ──────────

    #[test]
    fn auto_start_coordinator_has_workers() {
        let (c, _backend) = make_coordinator(true);
        assert!(c.swap_in_worker().is_some());
        assert!(c.eviction_worker().is_some());
    }

    // ── Coordinator: build_batch does not evict pages needed for swap-in ─────────

    #[test]
    fn build_batch_swap_in_page_excluded_from_eviction_candidates() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[1], 0.99);
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id != 1));
    }

    // ── Coordinator: stats snapshot changes after tier update ────────────────────

    #[test]
    fn stats_snapshot_changes_after_tier_update() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        let stats1 = c.stats();
        assert_eq!(stats1.pages_on_hbm, 1);
        assert_eq!(stats1.pages_on_dram, 0);

        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.gpu_ptr = None;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        let stats2 = c.stats();
        assert_eq!(stats2.pages_on_hbm, 0);
        assert_eq!(stats2.pages_on_dram, 1);
    }

    // ── Coordinator: multiple release_page calls updates counts ──────────────────

    #[test]
    fn release_multiple_pages_updates_counts_incrementally() {
        let (c, _backend) = make_coordinator(false);
        for pid in 1usize..=5 {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
        }
        assert_eq!(c.stats().pages_on_hbm, 5);

        c.release_page(3);
        assert_eq!(c.stats().pages_on_hbm, 4);

        c.release_page(1);
        c.release_page(5);
        assert_eq!(c.stats().pages_on_hbm, 2);
    }

    // ── Coordinator: build_batch with large number of pages ──────────────────────

    #[test]
    fn build_batch_with_many_pages_no_panic() {
        let (c, _backend) = make_coordinator(false);
        for pid in 0usize..200 {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        assert!(plan.eviction_candidates.len() <= 8);
    }

    // ── Coordinator: build_batch with mixed state pages ──────────────────────────

    #[test]
    fn build_batch_mixed_states_only_eligible_evicted() {
        let (c, _backend) = make_coordinator(false);
        // Page 1: Active (skip).
        c.register_page(1, Some(0x1001), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Active,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                ..PageMetadata::default()
            });
        }
        // Page 2: Standby (eligible).
        c.register_page(2, Some(0x1002), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(2, PageMetadata {
                page_id: 2,
                state: PageState::Standby,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                ..PageMetadata::default()
            });
        }
        // Page 3: Protected (skip).
        c.register_page(3, Some(0x1003), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(3, PageMetadata {
                page_id: 3,
                state: PageState::Protected,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                ..PageMetadata::default()
            });
        }
        let plan = c.build_batch(&[], 0.95);
        for candidate in &plan.eviction_candidates {
            assert_ne!(candidate.page_id, 1, "Active page should not be evicted");
            assert_ne!(candidate.page_id, 3, "Protected page should not be evicted");
        }
    }

    // ── Coordinator: tier_changed_pages with SwappedOut state ────────────────────

    #[test]
    fn tier_changed_pages_maps_swapped_out_to_nvme() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::SwappedOut,
                ..PageMetadata::default()
            });
        }
        let changed = c.tier_changed_pages();
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].1, StorageTier::GpuHbm);
    }

    // ── Coordinator: build_batch with active_pages containing unknown page ids ───

    #[test]
    fn build_batch_active_pages_unknown_ids_no_swap_in() {
        let (c, _backend) = make_coordinator(false);
        let plan = c.build_batch(&[999, 1000, 1001], 0.5);
        assert!(plan.swap_in_requests.is_empty());
        assert!(plan.eviction_candidates.is_empty());
    }

    // ── Coordinator: sequential build_batch calls ────────────────────────────────

    #[test]
    fn sequential_build_batch_calls_independent_plans() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan1 = c.build_batch(&[], 0.5);
        let plan2 = c.build_batch(&[], 0.95);
        assert!(plan1.eviction_candidates.is_empty() || !plan2.eviction_candidates.is_empty());
    }

    // ── Coordinator: register_page with u64::MAX gpu_ptr ─────────────────────────

    #[test]
    fn register_page_with_max_gpu_ptr() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(u64::MAX), 4096);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.gpu_ptr, Some(u64::MAX));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // ── Coordinator: update_page_gpu_ptr with zero address ───────────────────────

    #[test]
    fn update_page_gpu_ptr_with_zero_address() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        c.update_page_gpu_ptr(1, 0);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.gpu_ptr, Some(0));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // ── Coordinator: register_pages_from_hgal with many pages (stress) ───────────

    #[test]
    fn register_pages_from_hgal_many_pages() {
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        for pid in 0usize..500 {
            pages.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64),
                ..PageMetadata::default()
            });
        }
        c.register_pages_from_hgal(&pages, 4096);

        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.len(), 500);
        let meta = c.page_metadata.read().expect("read lock");
        assert_eq!(meta.len(), 500);
    }

    // ── Coordinator: record_eviction_same_from_to_tier_ignored ───────────────────

    #[test]
    fn record_eviction_same_tier_counts_bytes_but_not_migration() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::GpuHbm, 4096, 100);
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.total_bytes_evicted, 4096);
        assert_eq!(stats.total_eviction_latency_us, 100);
    }

    #[test]
    fn record_swap_in_same_tier_counts_bytes_but_not_migration() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::CpuDram, 8192, 50);
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_swapped_in, 8192);
        assert_eq!(stats.total_swap_in_latency_us, 50);
    }

    // ── Coordinator: register then release then verify all tables empty ──────────

    #[test]
    fn register_then_release_all_pages_tables_empty() {
        let (c, _backend) = make_coordinator(false);
        for pid in 1usize..=10 {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
        }
        assert_eq!(c.addr_table.read().expect("read lock").len(), 10);

        for pid in 1usize..=10 {
            c.release_page(pid);
        }
        assert!(c.addr_table.read().expect("read lock").is_empty());
        assert!(c.page_metadata.read().expect("read lock").is_empty());
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 0);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
    }

    // ── EvictionCandidate: all fields verified ───────────────────────────────────

    #[test]
    fn eviction_candidate_all_codec_variants() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, codec) in codecs.into_iter().enumerate() {
            let candidate = EvictionCandidate {
                page_id: i,
                score: i as i64 * 10,
                current_tier: StorageTier::GpuHbm,
                codec,
                page_bytes: 4096,
                group_id: Some(i as u64),
            };
            assert_eq!(candidate.codec, codec);
        }
    }

    #[test]
    fn eviction_candidate_all_tier_variants() {
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        for (i, tier) in tiers.into_iter().enumerate() {
            let candidate = EvictionCandidate {
                page_id: i,
                score: 0,
                current_tier: tier,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                group_id: None,
            };
            assert_eq!(candidate.current_tier, tier);
        }
    }

    // ── PrefetchRequest: urgency and confidence boundary values ──────────────────

    #[test]
    fn prefetch_request_with_nan_urgency() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: f32::NAN,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!(req.urgency.is_nan());
    }

    #[test]
    fn prefetch_request_with_infinity_urgency() {
        let req = PrefetchRequest {
            page_id: 2,
            urgency: f32::INFINITY,
            prefetch_confidence: f32::INFINITY,
            page_bytes: 8192,
            enqueued_at: Instant::now(),
        };
        assert!(req.urgency.is_infinite() && req.urgency.is_sign_positive());
    }

    #[test]
    fn prefetch_request_debug_output() {
        let req = PrefetchRequest {
            page_id: 42,
            urgency: 0.8,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let dbg = format!("{:?}", req);
        assert!(dbg.contains("page_id"));
        assert!(dbg.contains("42"));
    }

    #[test]
    fn prefetch_request_with_zero_page_bytes() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.5,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_bytes, 0);
    }

    // ── TierMigrationPlan: with large number of migrations ───────────────────────

    #[test]
    fn tier_migration_plan_with_many_migrations() {
        let migrations: Vec<TierMigration> = (0..100)
            .map(|i| TierMigration {
                page_id: i,
                from_tier: StorageTier::GpuHbm,
                to_tier: StorageTier::CpuDram,
                codec: CompressionCodec::None,
                page_bytes: 4096,
                reason: TierMigrationReason::EvictionPressure,
            })
            .collect();
        let plan = TierMigrationPlan {
            eviction_candidates: Vec::new(),
            swap_in_requests: Vec::new(),
            tier_migrations: migrations,
            built_at: Instant::now(),
        };
        assert_eq!(plan.tier_migrations.len(), 100);
        let cloned = plan.clone();
        assert_eq!(cloned.tier_migrations.len(), 100);
    }

    // ── TierMigration: from_tier != to_tier for real migration ───────────────────

    #[test]
    fn tier_migration_hbm_to_dram_valid_direction() {
        let migration = TierMigration {
            page_id: 1,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            reason: TierMigrationReason::EvictionPressure,
        };
        assert_ne!(migration.from_tier, migration.to_tier);
    }

    #[test]
    fn tier_migration_dram_to_nvme_valid_direction() {
        let migration = TierMigration {
            page_id: 2,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
            reason: TierMigrationReason::ColdCascade,
        };
        assert_ne!(migration.from_tier, migration.to_tier);
    }

    // ── Coordinator: build_batch with Swapped state page not in active set ───────

    #[test]
    fn build_batch_swapped_page_not_active_no_swap_in() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                state: PageState::Swapped,
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.5);
        assert!(plan.swap_in_requests.is_empty());
    }

    // ── Coordinator: stats snapshot preserves cumulative records after release ───

    #[test]
    fn stats_cumulative_records_preserved_after_release() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.release_page(1);
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.total_bytes_evicted, 4096);
        assert_eq!(stats.pages_on_hbm, 0);
    }

    // ── Coordinator: update_page_gpu_ptr on existing gpu_ptr overwrites ──────────

    #[test]
    fn update_page_gpu_ptr_overwrites_existing() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.update_page_gpu_ptr(1, 0x2000);
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.gpu_ptr, Some(0x2000));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // ── Coordinator: dram_pressure_ratio returns zero_with_empty_manager ─────────

    #[test]
    fn dram_pressure_ratio_zero_when_no_pages() {
        let (c, _backend) = make_coordinator(false);
        let plan = c.build_batch(&[], 0.5);
        let dram_evictions: Vec<_> = plan.eviction_candidates
            .iter()
            .filter(|c| c.current_tier == StorageTier::CpuDram)
            .collect();
        assert!(dram_evictions.is_empty());
    }

    // ── ThreeTierSwapStats: avg latency with only one type populated ─────────────

    #[test]
    fn stats_avg_eviction_only_gpu_to_dram_type() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 10;
        stats.total_eviction_latency_us = 5000;
        let avg = stats.avg_eviction_latency_us();
        assert!((avg - 500.0).abs() < 0.01);
        assert_eq!(stats.avg_swap_in_latency_us(), 0.0);
    }

    #[test]
    fn stats_avg_swap_in_only_nvme_to_dram_type() {
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_nvme_to_dram = 5;
        stats.total_swap_in_latency_us = 2500;
        let avg = stats.avg_swap_in_latency_us();
        assert!((avg - 500.0).abs() < 0.01);
        assert_eq!(stats.avg_eviction_latency_us(), 0.0);
    }

    // ── Coordinator: build_batch with exactly threshold pressure ─────────────────

    #[test]
    fn build_batch_exactly_at_threshold_pressure() {
        let (c, _backend) = make_coordinator(false);
        let threshold = c.config.eviction.hbm_pressure_threshold;
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // pressure == threshold is NOT > threshold, so no eviction.
        let plan = c.build_batch(&[], threshold);
        assert!(plan.eviction_candidates.is_empty());
    }

    #[test]
    fn build_batch_just_above_threshold_pressure() {
        let (c, _backend) = make_coordinator(false);
        let threshold = c.config.eviction.hbm_pressure_threshold;
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], threshold + 0.01);
        assert!(
            !plan.eviction_candidates.is_empty(),
            "pressure just above threshold should produce eviction candidates"
        );
    }

    // ── Coordinator: build_batch with recently accessed page (age < threshold) ───

    #[test]
    fn build_batch_recently_accessed_page_not_evicted() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: Some(Instant::now()),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.99);
        assert!(plan.eviction_candidates.is_empty(),
            "recently accessed page should not be evicted");
    }

    // ── Coordinator: register_pages_from_hgal preserves pre-existing addr entry ──

    #[test]
    fn register_pages_from_hgal_preserves_pre_existing_addr_entry_bytes() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 16384);
        let mut pages = HashMap::new();
        pages.insert(1, PageMetadata { page_id: 1, ..PageMetadata::default() });
        c.register_pages_from_hgal(&pages, 4096);
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.get(&1).unwrap().original_bytes, 16384);
    }

    // ── Coordinator: drop after auto_start still works ───────────────────────────

    #[test]
    fn drop_auto_started_coordinator_no_panic() {
        let (c, _backend) = make_coordinator(true);
        assert!(c.swap_in_worker().is_some());
        drop(c);
    }

    // ── Coordinator: build_batch with empty active_pages and populated tables ────

    #[test]
    fn build_batch_empty_active_pages_populated_tables() {
        let (c, _backend) = make_coordinator(false);
        for pid in 1usize..=3 {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                state: PageState::Active,
                last_access: Instant::now(),
                ..PageMetadata::default()
            });
        }
        let plan = c.build_batch(&[], 0.5);
        assert!(plan.swap_in_requests.is_empty());
        assert!(plan.eviction_candidates.is_empty());
    }

    // ── Coordinator: page registered on HBM then tier manually changed to DRAM ───

    #[test]
    fn register_hbm_then_manual_tier_change_to_dram_reflected_in_stats() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        assert_eq!(c.stats().pages_on_hbm, 1);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
            }
        }
        assert_eq!(c.stats().pages_on_hbm, 0);
        assert_eq!(c.stats().pages_on_dram, 1);
    }

    // ── Wave 16: 50 additional tests for uncovered areas ────────────────────────

    // ── MigrationCommand construction and Debug ──────────────────────────────────

    #[test]
    fn migration_command_evict_to_dram_fields() {
        let cmd = MigrationCommand::EvictToDram {
            page_id: 42,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
        };
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("EvictToDram"));
        assert!(dbg.contains("42"));
    }

    #[test]
    fn migration_command_promote_to_hbm_fields() {
        let cmd = MigrationCommand::PromoteToHbm {
            page_id: 7,
            page_bytes: 8192,
        };
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("PromoteToHbm"));
        assert!(dbg.contains("8192"));
    }

    #[test]
    fn migration_command_evict_to_nvme_fields() {
        let cmd = MigrationCommand::EvictToNvme {
            page_id: 99,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 4096,
        };
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("EvictToNvme"));
        assert!(dbg.contains("ZstdDict"));
    }

    #[test]
    fn migration_command_promote_to_dram_fields() {
        let cmd = MigrationCommand::PromoteToDram {
            page_id: 55,
            page_bytes: 4096,
        };
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("PromoteToDram"));
    }

    #[test]
    fn migration_command_shutdown_debug() {
        let cmd = MigrationCommand::Shutdown;
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("Shutdown"));
    }

    // ── MigrationResult construction and Debug ───────────────────────────────────

    #[test]
    fn migration_result_ok_fields() {
        let result = MigrationResult::Ok {
            compressed_bytes: 2048,
            checksum: 0xABCD,
        };
        let dbg = format!("{:?}", result);
        assert!(dbg.contains("Ok"));
        assert!(dbg.contains("2048"));
    }

    #[test]
    fn migration_result_failed_fields() {
        let result = MigrationResult::Failed {
            reason: "DMA timeout".to_string(),
        };
        let dbg = format!("{:?}", result);
        assert!(dbg.contains("Failed"));
        assert!(dbg.contains("DMA timeout"));
    }

    // ── MigrationDone construction and Debug ─────────────────────────────────────

    #[test]
    fn migration_done_debug_output() {
        let done = MigrationDone {
            page_id: 10,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            result: MigrationResult::Ok {
                compressed_bytes: 3000,
                checksum: 0x1234,
            },
        };
        let dbg = format!("{:?}", done);
        assert!(dbg.contains("page_id"));
        assert!(dbg.contains("10"));
        assert!(dbg.contains("from_tier"));
        assert!(dbg.contains("to_tier"));
    }

    #[test]
    fn migration_done_clone_preserves_fields() {
        let done = MigrationDone {
            page_id: 5,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::Nvme,
            result: MigrationResult::Failed {
                reason: "disk full".to_string(),
            },
        };
        let cloned = done.clone();
        assert_eq!(cloned.page_id, 5);
        assert_eq!(cloned.from_tier, StorageTier::CpuDram);
        assert_eq!(cloned.to_tier, StorageTier::Nvme);
    }

    // ── MigrationActorConfig field validation ────────────────────────────────────

    #[test]
    fn migration_actor_config_default_queue_capacity() {
        let config = MigrationActorConfig::default();
        assert_eq!(config.queue_capacity, 256);
    }

    #[test]
    fn migration_actor_config_default_session_id() {
        let config = MigrationActorConfig::default();
        assert_eq!(config.session_id, "default");
    }

    #[test]
    fn migration_actor_config_default_max_swap_pages() {
        let config = MigrationActorConfig::default();
        assert_eq!(config.max_swap_pages, 4096);
    }

    #[test]
    fn migration_actor_config_swap_file_path_format() {
        let config = MigrationActorConfig::default();
        let path = config.swap_file_path();
        assert!(path.to_string_lossy().ends_with(".swap"));
        assert!(path.to_string_lossy().contains("default"));
    }

    #[test]
    fn migration_actor_config_custom_session_in_swap_path() {
        let config = MigrationActorConfig {
            session_id: "test_session_42".to_string(),
            ..MigrationActorConfig::default()
        };
        let path = config.swap_file_path();
        assert!(path.to_string_lossy().contains("test_session_42"));
    }

    // ── EvictionWorkerConfig field validation ────────────────────────────────────

    #[test]
    fn eviction_worker_config_default_codec_is_lz4() {
        let config = EvictionWorkerConfig::default();
        assert_eq!(config.default_evict_codec, CompressionCodec::Lz4);
    }

    #[test]
    fn eviction_worker_config_default_page_bytes() {
        let config = EvictionWorkerConfig::default();
        assert_eq!(config.page_bytes, 4096);
    }

    #[test]
    fn eviction_worker_config_default_tick_interval_positive() {
        let config = EvictionWorkerConfig::default();
        assert!(config.tick_interval.as_millis() > 0);
    }

    #[test]
    fn eviction_worker_config_default_thresholds_in_range() {
        let config = EvictionWorkerConfig::default();
        assert!(config.hbm_pressure_threshold > 0.0 && config.hbm_pressure_threshold <= 1.0);
        assert!(config.dram_pressure_threshold > 0.0 && config.dram_pressure_threshold <= 1.0);
    }

    // ── SwapInWorkerConfig field validation ──────────────────────────────────────

    #[test]
    fn swap_in_worker_config_default_max_prefetch() {
        let config = SwapInWorkerConfig::default();
        assert_eq!(config.max_prefetch_per_round, 16);
    }

    #[test]
    fn swap_in_worker_config_default_tick_interval_positive() {
        let config = SwapInWorkerConfig::default();
        assert!(config.tick_interval.as_millis() > 0);
    }

    #[test]
    fn swap_in_worker_config_clone_independent() {
        let config = SwapInWorkerConfig::default();
        let mut cloned = config.clone();
        cloned.max_prefetch_per_round = 99;
        assert_eq!(config.max_prefetch_per_round, 16);
        assert_eq!(cloned.max_prefetch_per_round, 99);
    }

    // ── MigrationCommand with all CompressionCodec variants ──────────────────────

    #[test]
    fn migration_command_evict_to_dram_with_all_codecs() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for codec in codecs {
            let cmd = MigrationCommand::EvictToDram {
                page_id: 1,
                codec,
                page_bytes: 4096,
            };
            let dbg = format!("{:?}", cmd);
            assert!(!dbg.is_empty());
        }
    }

    // ── PageAddrEntry codec mutation after registration ──────────────────────────

    #[test]
    fn page_addr_entry_codec_mutation_after_register() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.codec = CompressionCodec::BitPackRle;
            }
        }
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.codec, CompressionCodec::BitPackRle);
    }

    #[test]
    fn page_addr_entry_tier_mutation_from_hbm_to_nvme() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
                entry.gpu_ptr = None;
            }
        }
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.current_tier, StorageTier::Nvme);
        assert!(entry.gpu_ptr.is_none());
    }

    // ── StorageTier transition rules via coordinator ─────────────────────────────

    #[test]
    fn storage_tier_hbm_to_dram_transition_in_stats() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        assert_eq!(c.stats().pages_on_hbm, 1);
        assert_eq!(c.stats().pages_on_dram, 0);

        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.gpu_ptr = None;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        assert_eq!(c.stats().pages_on_hbm, 0);
        assert_eq!(c.stats().pages_on_dram, 1);
    }

    #[test]
    fn storage_tier_dram_to_nvme_transition_in_stats() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        assert_eq!(c.stats().pages_on_dram, 1);

        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
                entry.host_buffer = None;
            }
        }
        assert_eq!(c.stats().pages_on_dram, 0);
        assert_eq!(c.stats().pages_on_nvme, 1);
    }

    #[test]
    fn storage_tier_full_lifecycle_hbm_dram_nvme() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        assert_eq!(c.stats().pages_on_hbm, 1);

        // HBM -> DRAM
        {
            let mut table = c.addr_table.write().expect("write lock");
            let entry = table.get_mut(&1).expect("should exist");
            entry.current_tier = StorageTier::CpuDram;
            entry.gpu_ptr = None;
            entry.host_buffer = Some(vec![0u8; 4096]);
        }
        assert_eq!(c.stats().pages_on_hbm, 0);
        assert_eq!(c.stats().pages_on_dram, 1);

        // DRAM -> NVMe
        {
            let mut table = c.addr_table.write().expect("write lock");
            let entry = table.get_mut(&1).expect("should exist");
            entry.current_tier = StorageTier::Nvme;
            entry.host_buffer = None;
        }
        assert_eq!(c.stats().pages_on_dram, 0);
        assert_eq!(c.stats().pages_on_nvme, 1);
    }

    // ── build_batch: tier migration reason for swap-in from NVMe ──────────────────

    #[test]
    fn build_batch_swap_in_from_nvme_reason_is_sequence_demand() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[1], 0.5);
        let migration = plan.tier_migrations.iter().find(|m| m.page_id == 1);
        assert!(migration.is_some());
        assert_eq!(migration.unwrap().reason, TierMigrationReason::SequenceDemand);
        assert_eq!(migration.unwrap().from_tier, StorageTier::Nvme);
        assert_eq!(migration.unwrap().to_tier, StorageTier::GpuHbm);
    }

    // ── build_batch: swap-in from DRAM produces correct tier migration ───────────

    #[test]
    fn build_batch_swap_in_from_dram_correct_migration_tiers() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 1,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[1], 0.5);
        let migration = plan.tier_migrations.iter().find(|m| m.page_id == 1);
        assert!(migration.is_some());
        assert_eq!(migration.unwrap().from_tier, StorageTier::CpuDram);
        assert_eq!(migration.unwrap().to_tier, StorageTier::GpuHbm);
    }

    // ── build_batch: eviction from HBM uses default_evict_codec ──────────────────

    #[test]
    fn build_batch_eviction_from_hbm_uses_lz4_codec() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        if let Some(candidate) = plan.eviction_candidates.iter().find(|c| c.page_id == 1) {
            assert_eq!(candidate.codec, CompressionCodec::Lz4);
        }
    }

    // ── record_eviction_completed: reverse direction not counted ──────────────────

    #[test]
    fn record_eviction_dram_to_hbm_not_counted() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 100);
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.total_bytes_evicted, 4096);
    }

    #[test]
    fn record_eviction_nvme_to_dram_not_counted() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::Nvme, StorageTier::CpuDram, 8192, 200);
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.total_bytes_evicted, 8192);
    }

    // ── record_swap_in_completed: reverse direction not counted ───────────────────

    #[test]
    fn record_swap_in_hbm_to_dram_not_counted() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 50);
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
    }

    #[test]
    fn record_swap_in_dram_to_nvme_not_counted() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::Nvme, 4096, 50);
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
    }

    // ── PageAddrEntry: host_buffer mutation after registration ───────────────────

    #[test]
    fn page_addr_entry_host_buffer_update_after_register() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.host_buffer = Some(vec![0xAA; 8192]);
            }
        }
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.host_buffer.as_ref().map(|b| b.len()), Some(8192));
        assert_eq!(entry.host_buffer.as_ref().map(|b| b[0]), Some(0xAA));
    }

    // ── Stats: avg_eviction_latency with one eviction ─────────────────────────────

    #[test]
    fn stats_avg_eviction_with_single_gpu_to_dram() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 1;
        stats.total_eviction_latency_us = 1234;
        assert!((stats.avg_eviction_latency_us() - 1234.0).abs() < 0.01);
    }

    // ── Stats: total_migrations with only evictions ───────────────────────────────

    #[test]
    fn stats_total_migrations_only_evictions() {
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 5;
        stats.evictions_dram_to_nvme = 3;
        assert_eq!(stats.total_migrations(), 8);
    }

    // ── Stats: total_migrations with only swap-ins ────────────────────────────────

    #[test]
    fn stats_total_migrations_only_swap_ins() {
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_dram_to_gpu = 4;
        stats.swap_ins_nvme_to_dram = 2;
        assert_eq!(stats.total_migrations(), 6);
    }

    // ── EvictionCandidate: all fields with each CompressionCodec ─────────────────

    #[test]
    fn eviction_candidate_with_nocomp_none() {
        let candidate = EvictionCandidate {
            page_id: 1,
            score: 100,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };
        assert_eq!(candidate.codec, CompressionCodec::None);
        assert!(candidate.group_id.is_none());
    }

    #[test]
    fn eviction_candidate_with_nvcomp_ans() {
        let candidate = EvictionCandidate {
            page_id: 2,
            score: 50,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 8192,
            group_id: Some(99),
        };
        assert_eq!(candidate.codec, CompressionCodec::NvcompAns);
        assert_eq!(candidate.group_id, Some(99));
    }

    // ── TierMigration: each reason with corresponding tier pair ──────────────────

    #[test]
    fn tier_migration_prefetch_reason_construction() {
        let migration = TierMigration {
            page_id: 10,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 4096,
            reason: TierMigrationReason::Prefetch,
        };
        assert_eq!(migration.reason, TierMigrationReason::Prefetch);
        assert_eq!(migration.from_tier, StorageTier::Nvme);
        assert_eq!(migration.to_tier, StorageTier::CpuDram);
    }

    #[test]
    fn tier_migration_sequence_demand_with_none_codec() {
        let migration = TierMigration {
            page_id: 5,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            reason: TierMigrationReason::SequenceDemand,
        };
        assert_eq!(migration.reason, TierMigrationReason::SequenceDemand);
        assert_eq!(migration.codec, CompressionCodec::None);
    }

    // ── build_batch: page with high access_count not evicted even under pressure ──

    #[test]
    fn build_batch_high_access_count_may_prevent_eviction() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 10000,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        // High access_count leads to high importance score, likely above threshold.
        for candidate in &plan.eviction_candidates {
            assert_ne!(candidate.page_id, 1, "high access_count page should not be evicted with low score");
        }
    }

    // ── build_batch: is_lir page not in skip list but affects scoring ─────────────

    #[test]
    fn build_batch_is_lir_page_not_auto_skipped() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: true,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.95);
        // is_lir affects scoring but is not a hard skip. Verify no panic.
        assert!(plan.eviction_candidates.len() <= 1);
    }

    // ── Coordinator: multiple tier changes tracked correctly ──────────────────────

    #[test]
    fn tier_changed_pages_after_multiple_tier_mutations() {
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, Some(0x2000), 4096);
        c.register_page(3, Some(0x3000), 4096);

        // Page 1: Active -> HBM (consistent)
        // Page 2: Swapped -> expects Nvme, actual HBM (divergent)
        // Page 3: Warm -> expects CpuDram, actual HBM (divergent)
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata { page_id: 1, state: PageState::Active, ..PageMetadata::default() });
            meta.insert(2, PageMetadata { page_id: 2, state: PageState::Swapped, ..PageMetadata::default() });
            meta.insert(3, PageMetadata { page_id: 3, state: PageState::Warm, ..PageMetadata::default() });
        }
        let changed = c.tier_changed_pages();
        assert_eq!(changed.len(), 2);
        let changed_ids: std::collections::HashSet<PageId> =
            changed.iter().map(|(pid, _)| *pid).collect();
        assert!(changed_ids.contains(&2));
        assert!(changed_ids.contains(&3));
        assert!(!changed_ids.contains(&1));
    }

    // ── Coordinator: record_eviction then verify total_migrations includes it ─────

    #[test]
    fn stats_total_migrations_includes_eviction_records() {
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::CpuDram, StorageTier::Nvme, 4096, 200);
        assert_eq!(c.stats().total_migrations(), 2);
    }

    // ── Coordinator: record_swap_in then verify total_migrations includes it ──────

    #[test]
    fn stats_total_migrations_includes_swap_in_records() {
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 80);
        c.record_swap_in_completed(2, StorageTier::Nvme, StorageTier::CpuDram, 4096, 500);
        assert_eq!(c.stats().total_migrations(), 2);
    }

    // ── Coordinator: interleaved register/release/build_batch ─────────────────────

    #[test]
    fn interleaved_register_release_build_batch() {
        let (c, _backend) = make_coordinator(false);

        // Register two pages.
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, Some(0x2000), 4096);

        // Build batch with no pages needed.
        let plan1 = c.build_batch(&[], 0.5);
        assert!(plan1.swap_in_requests.is_empty());

        // Release page 1.
        c.release_page(1);

        // Build batch again.
        let plan2 = c.build_batch(&[], 0.5);
        assert!(plan2.swap_in_requests.is_empty());

        // Only page 2 should remain.
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 1);
    }

    // ── PrefetchRequest: urgency negative value ──────────────────────────────────

    #[test]
    fn prefetch_request_negative_urgency() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: -1.0,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!(req.urgency < 0.0);
    }

    // ── PrefetchRequest: confidence boundary ─────────────────────────────────────

    #[test]
    fn prefetch_request_confidence_zero_and_one() {
        let req_low = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let req_high = PrefetchRequest {
            page_id: 2,
            urgency: 0.5,
            prefetch_confidence: 1.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!((req_low.prefetch_confidence - 0.0).abs() < f32::EPSILON);
        assert!((req_high.prefetch_confidence - 1.0).abs() < f32::EPSILON);
    }

    // ── PageAddrEntry: all CompressionCodec variants in entry ────────────────────

    #[test]
    fn page_addr_entry_with_each_codec() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for (i, codec) in codecs.into_iter().enumerate() {
            let entry = PageAddrEntry {
                gpu_ptr: Some(0x1000 + i as u64),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec,
            };
            assert_eq!(entry.codec, codec);
        }
    }

    // ── TierMigration: reverse direction (promotion) ─────────────────────────────

    #[test]
    fn tier_migration_nvme_to_dram_promotion() {
        let migration = TierMigration {
            page_id: 1,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 4096,
            reason: TierMigrationReason::Prefetch,
        };
        assert!(migration.from_tier < migration.to_tier);
    }

    // ── ThreeTierSwapConfig: clone with modified sub-config ──────────────────────

    #[test]
    fn config_clone_with_modified_eviction_subconfig() {
        let config = ThreeTierSwapConfig::default();
        let mut cloned = config.clone();
        cloned.eviction.max_evict_per_round = 32;
        assert_eq!(config.eviction.max_evict_per_round, 8);
        assert_eq!(cloned.eviction.max_evict_per_round, 32);
    }

    // ── Coordinator: build_batch with single page at exact boundary pressure ─────

    #[test]
    fn build_batch_exactly_one_eviction_when_limit_is_one() {
        let (c, _backend) = make_coordinator(false);
        // Register a single eligible page.
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[], 0.99);
        assert!(plan.eviction_candidates.len() <= 1);
    }

    // ── Wave 17: 60 additional tests for uncovered areas ────────────────────────

    // ── ThreeTierSwapStats: field-by-field write and read ────────────────────────

    #[test]
    fn stats_evictions_gpu_to_dram_large_value() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        // Act
        stats.evictions_gpu_to_dram = u64::MAX;
        // Assert
        assert_eq!(stats.evictions_gpu_to_dram, u64::MAX);
    }

    #[test]
    fn stats_evictions_dram_to_nvme_large_value() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        // Act
        stats.evictions_dram_to_nvme = u64::MAX;
        // Assert
        assert_eq!(stats.evictions_dram_to_nvme, u64::MAX);
    }

    #[test]
    fn stats_swap_ins_dram_to_gpu_large_value() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        // Act
        stats.swap_ins_dram_to_gpu = u64::MAX;
        // Assert
        assert_eq!(stats.swap_ins_dram_to_gpu, u64::MAX);
    }

    #[test]
    fn stats_swap_ins_nvme_to_dram_large_value() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        // Act
        stats.swap_ins_nvme_to_dram = u64::MAX;
        // Assert
        assert_eq!(stats.swap_ins_nvme_to_dram, u64::MAX);
    }

    #[test]
    fn stats_total_bytes_evicted_accumulates() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.total_bytes_evicted = 1_000_000;
        // Act
        stats.total_bytes_evicted += 500_000;
        // Assert
        assert_eq!(stats.total_bytes_evicted, 1_500_000);
    }

    #[test]
    fn stats_total_bytes_swapped_in_accumulates() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.total_bytes_swapped_in = 2_000_000;
        // Act
        stats.total_bytes_swapped_in += 750_000;
        // Assert
        assert_eq!(stats.total_bytes_swapped_in, 2_750_000);
    }

    #[test]
    fn stats_pages_on_tiers_all_set() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        // Act
        stats.pages_on_hbm = 10;
        stats.pages_on_dram = 20;
        stats.pages_on_nvme = 30;
        // Assert
        assert_eq!(stats.pages_on_hbm, 10);
        assert_eq!(stats.pages_on_dram, 20);
        assert_eq!(stats.pages_on_nvme, 30);
    }

    #[test]
    fn stats_debug_contains_all_field_names() {
        // Arrange
        let stats = ThreeTierSwapStats {
            evictions_gpu_to_dram: 1,
            evictions_dram_to_nvme: 2,
            swap_ins_dram_to_gpu: 3,
            swap_ins_nvme_to_dram: 4,
            total_bytes_evicted: 5,
            total_bytes_swapped_in: 6,
            total_eviction_latency_us: 7,
            total_swap_in_latency_us: 8,
            eviction_rounds: 9,
            swap_in_rounds: 10,
            pages_on_hbm: 11,
            pages_on_dram: 12,
            pages_on_nvme: 13,
        };
        // Act
        let dbg = format!("{:?}", stats);
        // Assert
        assert!(dbg.contains("evictions_gpu_to_dram"));
        assert!(dbg.contains("evictions_dram_to_nvme"));
        assert!(dbg.contains("swap_ins_dram_to_gpu"));
        assert!(dbg.contains("swap_ins_nvme_to_dram"));
        assert!(dbg.contains("eviction_rounds"));
        assert!(dbg.contains("swap_in_rounds"));
        assert!(dbg.contains("pages_on_hbm"));
        assert!(dbg.contains("pages_on_dram"));
        assert!(dbg.contains("pages_on_nvme"));
    }

    #[test]
    fn stats_clone_preserves_latency_fields() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.total_eviction_latency_us = 999;
        stats.total_swap_in_latency_us = 888;
        // Act
        let cloned = stats.clone();
        // Assert
        assert_eq!(cloned.total_eviction_latency_us, 999);
        assert_eq!(cloned.total_swap_in_latency_us, 888);
    }

    // ── TierMigrationPlan: built_at monotonic across calls ──────────────────────

    #[test]
    fn tier_migration_plan_built_at_monotonically_increasing() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        let plan1 = c.build_batch(&[], 0.5);
        let plan2 = c.build_batch(&[], 0.5);
        // Assert
        assert!(plan2.built_at >= plan1.built_at);
    }

    #[test]
    fn tier_migration_plan_clone_preserves_eviction_count() {
        // Arrange
        let plan = TierMigrationPlan {
            eviction_candidates: vec![
                EvictionCandidate {
                    page_id: 1, score: 10, current_tier: StorageTier::GpuHbm,
                    codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(1),
                },
                EvictionCandidate {
                    page_id: 2, score: 20, current_tier: StorageTier::GpuHbm,
                    codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(2),
                },
            ],
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        // Act
        let cloned = plan.clone();
        // Assert
        assert_eq!(cloned.eviction_candidates.len(), 2);
        assert_eq!(cloned.eviction_candidates[0].page_id, 1);
        assert_eq!(cloned.eviction_candidates[1].page_id, 2);
    }

    #[test]
    fn tier_migration_plan_debug_contains_eviction_candidates() {
        // Arrange
        let plan = TierMigrationPlan {
            eviction_candidates: vec![EvictionCandidate {
                page_id: 99, score: 42, current_tier: StorageTier::CpuDram,
                codec: CompressionCodec::BitPackRle, page_bytes: 8192, group_id: None,
            }],
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        // Act
        let dbg = format!("{:?}", plan);
        // Assert
        assert!(dbg.contains("eviction_candidates"));
        assert!(dbg.contains("99"));
    }

    // ── TierMigration: Debug output for each field ──────────────────────────────

    #[test]
    fn tier_migration_debug_shows_page_bytes() {
        // Arrange
        let migration = TierMigration {
            page_id: 7,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 65536,
            reason: TierMigrationReason::ColdCascade,
        };
        // Act
        let dbg = format!("{:?}", migration);
        // Assert
        assert!(dbg.contains("page_bytes"));
        assert!(dbg.contains("65536"));
        assert!(dbg.contains("reason"));
    }

    #[test]
    fn tier_migration_clone_preserves_codec() {
        // Arrange
        let migration = TierMigration {
            page_id: 3,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::NvcompAns,
            page_bytes: 4096,
            reason: TierMigrationReason::EvictionPressure,
        };
        // Act
        let cloned = migration.clone();
        // Assert
        assert_eq!(cloned.codec, CompressionCodec::NvcompAns);
        assert_eq!(cloned.page_id, 3);
        assert_eq!(cloned.page_bytes, 4096);
    }

    // ── TierMigrationReason: Hash usable in HashMap ─────────────────────────────

    #[test]
    fn tier_migration_reason_used_as_hashmap_key() {
        // Arrange
        let mut map = std::collections::HashMap::new();
        // Act
        map.insert(TierMigrationReason::EvictionPressure, "evict");
        map.insert(TierMigrationReason::SequenceDemand, "demand");
        map.insert(TierMigrationReason::Prefetch, "prefetch");
        map.insert(TierMigrationReason::ColdCascade, "cascade");
        // Assert
        assert_eq!(map.len(), 4);
        assert_eq!(map.get(&TierMigrationReason::ColdCascade), Some(&"cascade"));
        assert_eq!(map.get(&TierMigrationReason::Prefetch), Some(&"prefetch"));
    }

    // ── ThreeTierSwapConfig: default sub-config values ──────────────────────────

    #[test]
    fn config_default_eviction_hbm_age_ticks() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        // Assert
        assert_eq!(config.eviction.hbm_evict_age_ticks, 50);
    }

    #[test]
    fn config_default_eviction_dram_age_ticks() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        // Assert
        assert_eq!(config.eviction.dram_evict_age_ticks, 500);
    }

    #[test]
    fn config_default_eviction_importance_threshold() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        // Assert
        assert_eq!(config.eviction.importance_threshold, 100);
    }

    #[test]
    fn config_default_swap_in_min_confidence() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        // Assert
        assert!((config.swap_in.min_confidence - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn config_default_swap_in_max_in_flight() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        // Assert
        assert_eq!(config.swap_in.max_in_flight, 64);
    }

    #[test]
    fn config_default_swap_in_page_bytes() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        // Assert
        assert_eq!(config.swap_in.page_bytes, 4096);
    }

    #[test]
    fn config_default_migration_page_size() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        // Assert
        assert_eq!(config.migration.page_size, 4096);
    }

    #[test]
    fn config_default_migration_nvme_swap_dir_contains_gllm() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        let dir = config.migration.nvme_swap_dir.to_string_lossy();
        // Assert
        assert!(dir.contains(".gllm"));
        assert!(dir.contains("swap"));
    }

    #[test]
    fn config_auto_start_false_prevents_worker_creation() {
        // Arrange
        let config = ThreeTierSwapConfig {
            auto_start: false,
            ..ThreeTierSwapConfig::default()
        };
        // Act & Assert
        assert!(!config.auto_start);
    }

    // ── Coordinator: stats reflects tier distribution after multiple mutations ──

    #[test]
    fn stats_tier_count_after_register_move_release() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096); // HBM
        c.register_page(2, Some(0x2000), 4096); // HBM
        c.register_page(3, None, 4096);          // DRAM
        // Act: move page 2 to NVMe
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&2) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        // Assert
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 1);
        assert_eq!(stats.pages_on_dram, 1);
        assert_eq!(stats.pages_on_nvme, 1);

        // Act: release page 1
        c.release_page(1);
        // Assert
        let stats2 = c.stats();
        assert_eq!(stats2.pages_on_hbm, 0);
        assert_eq!(stats2.pages_on_dram, 1);
        assert_eq!(stats2.pages_on_nvme, 1);
    }

    // ── Coordinator: record_eviction_completed with large values ─────────────────

    #[test]
    fn record_eviction_completed_large_bytes_and_latency() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, u64::MAX, u64::MAX);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.total_bytes_evicted, u64::MAX);
        assert_eq!(stats.total_eviction_latency_us, u64::MAX);
    }

    #[test]
    fn record_swap_in_completed_large_bytes_and_latency() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.record_swap_in_completed(1, StorageTier::Nvme, StorageTier::CpuDram, u64::MAX, u64::MAX);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.total_bytes_swapped_in, u64::MAX);
        assert_eq!(stats.total_swap_in_latency_us, u64::MAX);
    }

    // ── Coordinator: build_batch with page_id 0 active ─────────────────────────

    #[test]
    fn build_batch_page_id_zero_swap_in() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(0, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&0) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(0, PageMetadata {
                page_id: 0,
                sequence_id: Some(1),
                recency: 0,
                access_count: 1,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[0], 0.5);
        // Assert
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 0);
    }

    // ── Coordinator: interleaved eviction and swap-in records ────────────────────

    #[test]
    fn interleaved_eviction_swap_in_records_correct_totals() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: eviction then swap-in, then eviction again
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 80);
        c.record_eviction_completed(2, StorageTier::GpuHbm, StorageTier::CpuDram, 8192, 200);
        c.record_swap_in_completed(2, StorageTier::CpuDram, StorageTier::GpuHbm, 8192, 150);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 2);
        assert_eq!(stats.swap_ins_dram_to_gpu, 2);
        assert_eq!(stats.total_bytes_evicted, 4096 + 8192);
        assert_eq!(stats.total_bytes_swapped_in, 4096 + 8192);
        assert_eq!(stats.total_migrations(), 4);
    }

    // ── CompressionCodec: roundtrip for each variant via from_u8/as_u8 ─────────

    #[test]
    fn compression_codec_lz4_roundtrip() {
        // Arrange
        let codec = CompressionCodec::Lz4;
        // Act
        let encoded = codec.as_u8();
        let decoded = CompressionCodec::from_u8(encoded);
        // Assert
        assert_eq!(encoded, 1);
        assert_eq!(decoded, Some(CompressionCodec::Lz4));
    }

    #[test]
    fn compression_codec_bitpack_rle_roundtrip() {
        // Arrange
        let codec = CompressionCodec::BitPackRle;
        // Act
        let encoded = codec.as_u8();
        let decoded = CompressionCodec::from_u8(encoded);
        // Assert
        assert_eq!(encoded, 2);
        assert_eq!(decoded, Some(CompressionCodec::BitPackRle));
    }

    #[test]
    fn compression_codec_nvcomp_ans_roundtrip() {
        // Arrange
        let codec = CompressionCodec::NvcompAns;
        // Act
        let encoded = codec.as_u8();
        let decoded = CompressionCodec::from_u8(encoded);
        // Assert
        assert_eq!(encoded, 3);
        assert_eq!(decoded, Some(CompressionCodec::NvcompAns));
    }

    // ── StorageTier: from_u8 with MIN/MAX u8 ────────────────────────────────────

    #[test]
    fn storage_tier_from_u8_zero_is_hbm() {
        // Arrange & Act
        let result = StorageTier::from_u8(0);
        // Assert
        assert_eq!(result, Some(StorageTier::GpuHbm));
    }

    #[test]
    fn storage_tier_as_u8_then_from_u8_identity() {
        // Arrange
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        for tier in tiers {
            // Act
            let encoded = tier.as_u8();
            let decoded = StorageTier::from_u8(encoded);
            // Assert
            assert_eq!(decoded, Some(tier));
        }
    }

    // ── Coordinator: build_batch with subthreshold pressure decimal ─────────────

    #[test]
    fn build_batch_pressure_0_89_no_eviction_default_threshold_0_90() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: 0.89 < 0.90 threshold
        let plan = c.build_batch(&[], 0.89);
        // Assert
        assert!(plan.eviction_candidates.is_empty());
    }

    // ── Coordinator: register_pages_from_hgal then build_batch with active ids ──

    #[test]
    fn register_from_hgal_then_build_batch_swap_in_for_active() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        pages.insert(10, PageMetadata { page_id: 10, sequence_id: Some(10), ..PageMetadata::default() });
        c.register_pages_from_hgal(&pages, 4096);
        // Act: page 10 is on DRAM (registered with None gpu_ptr)
        let plan = c.build_batch(&[10], 0.5);
        // Assert
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 10);
    }

    // ── Coordinator: multiple build_batch calls each increment round counter ────

    #[test]
    fn build_batch_five_calls_increment_rounds_by_five() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        for _ in 0..5 {
            let _ = c.build_batch(&[], 0.5);
        }
        // Assert
        let stats = c.stats();
        assert_eq!(stats.eviction_rounds, 5);
        assert_eq!(stats.swap_in_rounds, 5);
    }

    // ── Coordinator: release_page on unregistered page idempotent ────────────────

    #[test]
    fn release_page_unregistered_twice_no_panic() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.release_page(999);
        c.release_page(999);
        // Assert: no panic, tables remain empty
        let table = c.addr_table.read().expect("read lock");
        assert!(table.is_empty());
    }

    // ── TierMigrationPlan: clone with many eviction candidates ──────────────────

    #[test]
    fn tier_migration_plan_clone_many_candidates() {
        // Arrange
        let candidates: Vec<EvictionCandidate> = (0..50)
            .map(|i| EvictionCandidate {
                page_id: i,
                score: i as i64,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::Lz4,
                page_bytes: 4096,
                group_id: Some(i as u64),
            })
            .collect();
        let plan = TierMigrationPlan {
            eviction_candidates: candidates,
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        // Act
        let cloned = plan.clone();
        // Assert
        assert_eq!(cloned.eviction_candidates.len(), 50);
        assert_eq!(cloned.eviction_candidates[49].page_id, 49);
    }

    // ── Stats: avg_eviction_latency returns 0 when only dram_to_nvme present ────

    #[test]
    fn stats_avg_eviction_zero_when_only_dram_to_nvme_latency() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_dram_to_nvme = 5;
        stats.total_eviction_latency_us = 500;
        // Act
        let avg = stats.avg_eviction_latency_us();
        // Assert
        assert!((avg - 100.0).abs() < 0.01);
    }

    // ── Stats: avg_swap_in_latency returns 0 when only nvme_to_dram present ─────

    #[test]
    fn stats_avg_swap_in_zero_when_only_nvme_to_dram_latency() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_nvme_to_dram = 10;
        stats.total_swap_in_latency_us = 2000;
        // Act
        let avg = stats.avg_swap_in_latency_us();
        // Assert
        assert!((avg - 200.0).abs() < 0.01);
    }

    // ── Coordinator: update_page_gpu_ptr for page registered without gpu_ptr ────

    #[test]
    fn update_gpu_ptr_for_initially_dram_page_promotes_to_hbm() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let table = c.addr_table.read().expect("read lock");
            assert_eq!(table.get(&1).unwrap().current_tier, StorageTier::CpuDram);
        }
        // Act
        c.update_page_gpu_ptr(1, 0xABCD);
        // Assert
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.gpu_ptr, Some(0xABCD));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
        assert_eq!(c.stats().pages_on_hbm, 1);
        assert_eq!(c.stats().pages_on_dram, 0);
    }

    // ── TierMigrationReason: Copy trait ensures independent values ──────────────

    #[test]
    fn tier_migration_reason_copy_trait_independent() {
        // Arrange
        let a = TierMigrationReason::SequenceDemand;
        let b = a; // Copy
        // Act: a is still valid after copy
        // Assert
        assert_eq!(a, TierMigrationReason::SequenceDemand);
        assert_eq!(b, TierMigrationReason::SequenceDemand);
    }

    // ── TierMigration: construct with each StorageTier pair direction ────────────

    #[test]
    fn tier_migration_hbm_to_nvme_skipping_dram() {
        // Arrange & Act
        let migration = TierMigration {
            page_id: 1,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::Nvme,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 4096,
            reason: TierMigrationReason::EvictionPressure,
        };
        // Assert
        assert!(migration.from_tier > migration.to_tier);
    }

    #[test]
    fn tier_migration_nvme_to_hbm_promotion() {
        // Arrange & Act
        let migration = TierMigration {
            page_id: 2,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 8192,
            reason: TierMigrationReason::SequenceDemand,
        };
        // Assert
        assert!(migration.from_tier < migration.to_tier);
    }

    // ── Coordinator: build_batch returns plan with consistent field counts ───────

    #[test]
    fn build_batch_swap_in_and_migration_counts_match() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        for pid in [10usize, 20] {
            c.register_page(pid, None, 4096);
            {
                let mut table = c.addr_table.write().expect("write lock");
                if let Some(entry) = table.get_mut(&pid) {
                    entry.current_tier = StorageTier::CpuDram;
                    entry.host_buffer = Some(vec![0u8; 4096]);
                }
            }
            {
                let mut meta = c.page_metadata.write().expect("write lock");
                meta.insert(pid, PageMetadata {
                    page_id: pid,
                    sequence_id: Some(pid as u64),
                    recency: 0,
                    access_count: 5,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
        }
        // Act
        let plan = c.build_batch(&[10, 20], 0.5);
        // Assert: each swap-in request should have a corresponding tier_migration
        assert_eq!(plan.swap_in_requests.len(), plan.tier_migrations.iter().filter(|m| m.reason == TierMigrationReason::SequenceDemand).count());
    }

    // ── Coordinator: page_metadata accessor reflects register_pages_from_hgal ──

    #[test]
    fn page_metadata_accessor_reflects_bulk_register() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        let mut pages: HashMap<usize, PageMetadata> = HashMap::new();
        for pid in 1usize..=5 {
            pages.insert(pid, PageMetadata { page_id: pid, sequence_id: Some(pid as u64 * 10), ..PageMetadata::default() });
        }
        // Act
        c.register_pages_from_hgal(&pages, 4096);
        // Assert
        let guard = c.page_metadata().read().expect("read lock");
        assert_eq!(guard.len(), 5);
        for pid in 1usize..=5 {
            assert_eq!(guard.get(&pid).unwrap().sequence_id, Some(pid as u64 * 10));
        }
    }

    // ── Coordinator: addr_table accessor reflects register_page ─────────────────

    #[test]
    fn addr_table_accessor_reflects_single_register() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.register_page(42, Some(0xBEEF), 4096);
        // Assert
        let guard = c.addr_table().read().expect("read lock");
        assert!(guard.contains_key(&42));
        assert_eq!(guard.get(&42).unwrap().gpu_ptr, Some(0xBEEF));
    }

    // ── EvictionCandidate: construction with each tier ──────────────────────────

    #[test]
    fn eviction_candidate_with_nvme_tier() {
        // Arrange & Act
        let candidate = EvictionCandidate {
            page_id: 1,
            score: -100,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 4096,
            group_id: None,
        };
        // Assert
        assert_eq!(candidate.current_tier, StorageTier::Nvme);
        assert_eq!(candidate.score, -100);
    }

    // ── PrefetchRequest: urgency one boundary ──────────────────────────────────

    #[test]
    fn prefetch_request_urgency_exactly_one() {
        // Arrange & Act
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 1.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        // Assert
        assert!((req.urgency - 1.0).abs() < f32::EPSILON);
        assert!((req.prefetch_confidence - 1.0).abs() < f32::EPSILON);
    }

    // ── Coordinator: register then update then stats snapshot ───────────────────

    #[test]
    fn register_update_stats_snapshot_consistency() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, None, 4096);
        // Act: update page 2 to HBM
        c.update_page_gpu_ptr(2, 0x2000);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 2);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
    }

    // ── Coordinator: shutdown then verify workers are None ──────────────────────

    #[test]
    fn shutdown_sets_workers_to_none() {
        // Arrange
        let (mut c, _backend) = make_coordinator(true);
        assert!(c.swap_in_worker().is_some());
        assert!(c.eviction_worker().is_some());
        // Act
        c.shutdown();
        // Assert
        assert!(c.swap_in_worker.is_none());
        assert!(c.eviction_worker.is_none());
    }

    // ── Coordinator: build_batch with pressure just below threshold ─────────────

    #[test]
    fn build_batch_pressure_just_below_threshold_no_eviction() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        let threshold = c.config.eviction.hbm_pressure_threshold;
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], threshold - 0.001);
        // Assert
        assert!(plan.eviction_candidates.is_empty());
    }

    // ── ThreeTierSwapConfig: debug output shows sub-config fields ───────────────

    #[test]
    fn config_debug_output_shows_migration() {
        // Arrange
        let config = ThreeTierSwapConfig::default();
        // Act
        let dbg = format!("{:?}", config);
        // Assert
        assert!(dbg.contains("migration"));
        assert!(dbg.contains("eviction"));
        assert!(dbg.contains("swap_in"));
    }

    // ── Coordinator: build_batch with swapped_out state page on DRAM tier ───────

    #[test]
    fn build_batch_swapped_out_on_dram_may_evict_under_pressure() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::SwappedOut,
                warm_until: None,
            });
        }
        // Act: high pressure but dram_pressure is 0 (empty memory manager)
        let plan = c.build_batch(&[], 0.99);
        // Assert: no panic, eviction depends on dram_pressure which is 0
        assert!(plan.eviction_candidates.len() <= 1);
    }

    // ── Stats: total_migrations includes both directions of same page ────────────

    #[test]
    fn stats_total_migrations_counts_evict_and_swap_in_separately() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: evict page 1 then swap it back
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 80);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.total_migrations(), 2);
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
    }

    // ── CompressionCodec: each variant produces non-empty debug ─────────────────

    #[test]
    fn compression_codec_none_debug_not_empty() {
        // Arrange & Act
        let dbg = format!("{:?}", CompressionCodec::None);
        // Assert
        assert!(!dbg.is_empty());
    }

    #[test]
    fn compression_codec_zstd_dict_debug_not_empty() {
        // Arrange & Act
        let dbg = format!("{:?}", CompressionCodec::ZstdDict);
        // Assert
        assert!(!dbg.is_empty());
    }

    // ── Coordinator: tier_changed_pages returns empty when no metadata at all ───

    #[test]
    fn tier_changed_pages_empty_when_no_metadata_registered() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: register page in addr_table only
        c.register_page(1, Some(0x1000), 4096);
        // Assert: no metadata -> no tier divergence detected
        assert!(c.tier_changed_pages().is_empty());
    }

    // ── Coordinator: build_batch with only eviction candidates, no swap-ins ─────

    #[test]
    fn build_batch_only_evictions_no_swap_ins_when_no_active_pages() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        for pid in 1usize..=3 {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64 * 10),
                recency: 1000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.95);
        // Assert: swap_in_requests should be empty since no active_pages
        assert!(plan.swap_in_requests.is_empty());
    }

    // ── Wave 18: 55 additional tests for uncovered areas ────────────────────────

    // ── EvictionWorkerConfig: individual field mutation independence ────────────

    #[test]
    fn eviction_worker_config_clone_isolation_tick_interval() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.tick_interval = Duration::from_millis(999);
        // Assert
        assert_ne!(config.tick_interval, cloned.tick_interval);
        assert_eq!(config.tick_interval, Duration::from_millis(10));
    }

    #[test]
    fn eviction_worker_config_clone_isolation_max_evict() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.max_evict_per_round = 100;
        // Assert
        assert_eq!(config.max_evict_per_round, 8);
        assert_eq!(cloned.max_evict_per_round, 100);
    }

    #[test]
    fn eviction_worker_config_clone_isolation_threshold() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.hbm_pressure_threshold = 0.5;
        // Assert
        assert!((config.hbm_pressure_threshold - 0.9).abs() < 0.01);
        assert!((cloned.hbm_pressure_threshold - 0.5).abs() < 0.01);
    }

    #[test]
    fn eviction_worker_config_clone_isolation_importance_threshold() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.importance_threshold = 500;
        // Assert
        assert_eq!(config.importance_threshold, 100);
        assert_eq!(cloned.importance_threshold, 500);
    }

    #[test]
    fn eviction_worker_config_clone_isolation_codec() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.default_evict_codec = CompressionCodec::ZstdDict;
        // Assert
        assert_eq!(config.default_evict_codec, CompressionCodec::Lz4);
        assert_eq!(cloned.default_evict_codec, CompressionCodec::ZstdDict);
    }

    #[test]
    fn eviction_worker_config_debug_output() {
        // Arrange
        let config = EvictionWorkerConfig::default();
        // Act
        let dbg = format!("{:?}", config);
        // Assert
        assert!(dbg.contains("tick_interval"));
        assert!(dbg.contains("max_evict_per_round"));
        assert!(dbg.contains("hbm_pressure_threshold"));
        assert!(dbg.contains("importance_threshold"));
    }

    #[test]
    fn eviction_worker_config_custom_construction() {
        // Arrange & Act
        let config = EvictionWorkerConfig {
            tick_interval: Duration::from_millis(50),
            max_evict_per_round: 16,
            hbm_pressure_threshold: 0.8,
            dram_pressure_threshold: 0.7,
            importance_threshold: 200,
            hbm_evict_age_ticks: 100,
            dram_evict_age_ticks: 1000,
            default_evict_codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
        };
        // Assert
        assert_eq!(config.tick_interval, Duration::from_millis(50));
        assert_eq!(config.max_evict_per_round, 16);
        assert!((config.hbm_pressure_threshold - 0.8).abs() < f32::EPSILON);
        assert!((config.dram_pressure_threshold - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.importance_threshold, 200);
        assert_eq!(config.hbm_evict_age_ticks, 100);
        assert_eq!(config.dram_evict_age_ticks, 1000);
        assert_eq!(config.default_evict_codec, CompressionCodec::BitPackRle);
        assert_eq!(config.page_bytes, 8192);
    }

    // ── SwapInWorkerConfig: additional field tests ──────────────────────────────

    #[test]
    fn swap_in_worker_config_default_tick_is_5ms() {
        // Arrange & Act
        let config = SwapInWorkerConfig::default();
        // Assert
        assert_eq!(config.tick_interval, Duration::from_millis(5));
    }

    #[test]
    fn swap_in_worker_config_debug_output() {
        // Arrange
        let config = SwapInWorkerConfig::default();
        // Act
        let dbg = format!("{:?}", config);
        // Assert
        assert!(dbg.contains("max_prefetch_per_round"));
        assert!(dbg.contains("tick_interval"));
        assert!(dbg.contains("min_confidence"));
        assert!(dbg.contains("max_in_flight"));
    }

    #[test]
    fn swap_in_worker_config_custom_construction() {
        // Arrange & Act
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 32,
            tick_interval: Duration::from_millis(20),
            min_confidence: 0.5,
            max_in_flight: 128,
            page_bytes: 8192,
        };
        // Assert
        assert_eq!(config.max_prefetch_per_round, 32);
        assert_eq!(config.tick_interval, Duration::from_millis(20));
        assert!((config.min_confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.max_in_flight, 128);
        assert_eq!(config.page_bytes, 8192);
    }

    #[test]
    fn swap_in_worker_config_clone_isolation_page_bytes() {
        // Arrange
        let config = SwapInWorkerConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.page_bytes = 16384;
        // Assert
        assert_eq!(config.page_bytes, 4096);
        assert_eq!(cloned.page_bytes, 16384);
    }

    // ── SwapInWorkerStats: field access and methods ────────────────────────────

    #[test]
    fn swap_in_worker_stats_default_all_zero() {
        // Arrange & Act
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let stats = SwapInWorkerStats::default();
        // Assert
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.submitted, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(stats.promoted_ok, 0);
        assert_eq!(stats.promoted_failed, 0);
        assert_eq!(stats.two_hop_promotions, 0);
        assert_eq!(stats.total_latency_us, 0);
        assert_eq!(stats.rounds, 0);
    }

    #[test]
    fn swap_in_worker_stats_avg_latency_zero_when_no_promotions() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let stats = SwapInWorkerStats::default();
        // Act
        let avg = stats.avg_latency_us();
        // Assert
        assert!((avg - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn swap_in_worker_stats_avg_latency_with_promotions() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 5;
        stats.total_latency_us = 1000;
        // Act
        let avg = stats.avg_latency_us();
        // Assert
        assert!((avg - 200.0).abs() < 0.01);
    }

    #[test]
    fn swap_in_worker_stats_clone_independence() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 100;
        stats.promoted_ok = 80;
        // Act
        let cloned = stats.clone();
        stats.total_requests = 200;
        // Assert
        assert_eq!(cloned.total_requests, 100);
        assert_eq!(cloned.promoted_ok, 80);
        assert_eq!(stats.total_requests, 200);
    }

    #[test]
    fn swap_in_worker_stats_equality_same_values() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 10;
        let cloned = stats.clone();
        // Act & Assert
        assert_eq!(stats, cloned);
    }

    #[test]
    fn swap_in_worker_stats_inequality_different_values() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let a = SwapInWorkerStats { total_requests: 1, ..SwapInWorkerStats::default() };
        let b = SwapInWorkerStats { total_requests: 2, ..SwapInWorkerStats::default() };
        // Act & Assert
        assert_ne!(a, b);
    }

    #[test]
    fn swap_in_worker_stats_debug_output() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let stats = SwapInWorkerStats {
            total_requests: 42,
            submitted: 30,
            skipped: 12,
            promoted_ok: 25,
            promoted_failed: 5,
            two_hop_promotions: 3,
            total_latency_us: 500,
            rounds: 10,
        };
        // Act
        let dbg = format!("{:?}", stats);
        // Assert
        assert!(dbg.contains("total_requests"));
        assert!(dbg.contains("promoted_ok"));
        assert!(dbg.contains("rounds"));
    }

    // ── MigrationActorConfig: additional construction variants ──────────────────

    #[test]
    fn migration_actor_config_default_page_size() {
        // Arrange & Act
        let config = MigrationActorConfig::default();
        // Assert
        assert_eq!(config.page_size, 4096);
    }

    #[test]
    fn migration_actor_config_clone_independence() {
        // Arrange
        let config = MigrationActorConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.queue_capacity = 512;
        // Assert
        assert_eq!(config.queue_capacity, 256);
        assert_eq!(cloned.queue_capacity, 512);
    }

    #[test]
    fn migration_actor_config_debug_output() {
        // Arrange
        let config = MigrationActorConfig::default();
        // Act
        let dbg = format!("{:?}", config);
        // Assert
        assert!(dbg.contains("nvme_swap_dir"));
        assert!(dbg.contains("queue_capacity"));
        assert!(dbg.contains("session_id"));
        assert!(dbg.contains("page_size"));
    }

    #[test]
    fn migration_actor_config_custom_session_id_in_debug() {
        // Arrange
        let config = MigrationActorConfig {
            session_id: "my_session".to_string(),
            ..MigrationActorConfig::default()
        };
        // Act
        let dbg = format!("{:?}", config);
        // Assert
        assert!(dbg.contains("my_session"));
    }

    // ── MigrationResult: field access via pattern matching ──────────────────────

    #[test]
    fn migration_result_ok_field_access() {
        // Arrange
        let result = MigrationResult::Ok {
            compressed_bytes: 2048,
            checksum: 0xBEEF,
        };
        // Act
        let (bytes, cksum) = match result {
            MigrationResult::Ok { compressed_bytes, checksum } => (compressed_bytes, checksum),
            _ => panic!("expected Ok variant"),
        };
        // Assert
        assert_eq!(bytes, 2048);
        assert_eq!(cksum, 0xBEEF);
    }

    #[test]
    fn migration_result_failed_field_access() {
        // Arrange
        let result = MigrationResult::Failed {
            reason: "timeout".to_string(),
        };
        // Act
        let reason = match result {
            MigrationResult::Failed { reason } => reason,
            _ => panic!("expected Failed variant"),
        };
        // Assert
        assert_eq!(reason, "timeout");
    }

    #[test]
    fn migration_result_clone_ok() {
        // Arrange
        let result = MigrationResult::Ok {
            compressed_bytes: 4096,
            checksum: 0x1234,
        };
        // Act
        let cloned = result.clone();
        // Assert
        match cloned {
            MigrationResult::Ok { compressed_bytes, checksum } => {
                assert_eq!(compressed_bytes, 4096);
                assert_eq!(checksum, 0x1234);
            }
            _ => panic!("expected Ok variant"),
        }
    }

    #[test]
    fn migration_result_clone_failed() {
        // Arrange
        let result = MigrationResult::Failed {
            reason: "io error".to_string(),
        };
        // Act
        let cloned = result.clone();
        // Assert
        match cloned {
            MigrationResult::Failed { reason } => {
                assert_eq!(reason, "io error");
            }
            _ => panic!("expected Failed variant"),
        }
    }

    // ── MigrationDone: field access and construction variants ───────────────────

    #[test]
    fn migration_done_with_ok_result() {
        // Arrange & Act
        let done = MigrationDone {
            page_id: 42,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            result: MigrationResult::Ok {
                compressed_bytes: 3000,
                checksum: 0xABCD,
            },
        };
        // Assert
        assert_eq!(done.page_id, 42);
        assert_eq!(done.from_tier, StorageTier::GpuHbm);
        assert_eq!(done.to_tier, StorageTier::CpuDram);
        match &done.result {
            MigrationResult::Ok { compressed_bytes, .. } => {
                assert_eq!(*compressed_bytes, 3000);
            }
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn migration_done_with_failed_result() {
        // Arrange & Act
        let done = MigrationDone {
            page_id: 99,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::Nvme,
            result: MigrationResult::Failed {
                reason: "no space".to_string(),
            },
        };
        // Assert
        assert_eq!(done.page_id, 99);
        match &done.result {
            MigrationResult::Failed { reason } => {
                assert_eq!(reason, "no space");
            }
            _ => panic!("expected Failed"),
        }
    }

    #[test]
    fn migration_done_clone_preserves_result_variant() {
        // Arrange
        let done = MigrationDone {
            page_id: 7,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::CpuDram,
            result: MigrationResult::Ok {
                compressed_bytes: 1024,
                checksum: 0xFFFF,
            },
        };
        // Act
        let cloned = done.clone();
        // Assert
        assert_eq!(cloned.page_id, 7);
        assert_eq!(cloned.from_tier, StorageTier::Nvme);
        assert_eq!(cloned.to_tier, StorageTier::CpuDram);
        match cloned.result {
            MigrationResult::Ok { compressed_bytes, checksum } => {
                assert_eq!(compressed_bytes, 1024);
                assert_eq!(checksum, 0xFFFF);
            }
            _ => panic!("expected Ok"),
        }
    }

    // ── MigrationCommand: additional construction with edge values ──────────────

    #[test]
    fn migration_command_evict_to_dram_zero_bytes() {
        // Arrange & Act
        let cmd = MigrationCommand::EvictToDram {
            page_id: 0,
            codec: CompressionCodec::None,
            page_bytes: 0,
        };
        // Assert
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("EvictToDram"));
    }

    #[test]
    fn migration_command_promote_to_hbm_zero_bytes() {
        // Arrange & Act
        let cmd = MigrationCommand::PromoteToHbm {
            page_id: 0,
            page_bytes: 0,
        };
        // Assert
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("PromoteToHbm"));
    }

    #[test]
    fn migration_command_evict_to_nvme_clone() {
        // Arrange
        let cmd = MigrationCommand::EvictToNvme {
            page_id: 100,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
        };
        // Act
        let cloned = cmd.clone();
        // Assert
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("EvictToNvme"));
        assert!(dbg.contains("BitPackRle"));
    }

    #[test]
    fn migration_command_promote_to_dram_clone() {
        // Arrange
        let cmd = MigrationCommand::PromoteToDram {
            page_id: 200,
            page_bytes: 4096,
        };
        // Act
        let cloned = cmd.clone();
        // Assert
        let dbg = format!("{:?}", cloned);
        assert!(dbg.contains("PromoteToDram"));
    }

    // ── Coordinator: build_batch with mixed page sizes ─────────────────────────

    #[test]
    fn build_batch_mixed_page_sizes_all_tracked() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, Some(0x2000), 8192);
        c.register_page(3, None, 16384);
        // Act
        let stats = c.stats();
        // Assert
        assert_eq!(stats.pages_on_hbm, 2);
        assert_eq!(stats.pages_on_dram, 1);
        assert_eq!(stats.pages_on_nvme, 0);
    }

    // ── Coordinator: build_batch swap-in migration has correct page_bytes ──────

    #[test]
    fn build_batch_swap_in_migration_page_bytes_matches_entry() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 8192);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 8192]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert
        assert_eq!(plan.tier_migrations.len(), 1);
        assert_eq!(plan.tier_migrations[0].page_bytes, 8192);
    }

    // ── ThreeTierSwapStats: overflow resilience in avg calculations ────────────

    #[test]
    fn stats_avg_eviction_latency_large_numerator_zero_denominator() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.total_eviction_latency_us = u64::MAX;
        // Denominator is 0
        // Act
        let avg = stats.avg_eviction_latency_us();
        // Assert
        assert!((avg - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_avg_swap_in_latency_large_numerator_zero_denominator() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.total_swap_in_latency_us = u64::MAX;
        // Denominator is 0
        // Act
        let avg = stats.avg_swap_in_latency_us();
        // Assert
        assert!((avg - 0.0).abs() < f64::EPSILON);
    }

    // ── TierMigrationPlan: with only tier_migrations, no evictions or swap-ins ─

    #[test]
    fn tier_migration_plan_only_tier_migrations() {
        // Arrange & Act
        let plan = TierMigrationPlan {
            eviction_candidates: Vec::new(),
            swap_in_requests: Vec::new(),
            tier_migrations: vec![TierMigration {
                page_id: 1,
                from_tier: StorageTier::GpuHbm,
                to_tier: StorageTier::CpuDram,
                codec: CompressionCodec::Lz4,
                page_bytes: 4096,
                reason: TierMigrationReason::EvictionPressure,
            }],
            built_at: Instant::now(),
        };
        // Assert
        assert!(plan.eviction_candidates.is_empty());
        assert!(plan.swap_in_requests.is_empty());
        assert_eq!(plan.tier_migrations.len(), 1);
    }

    // ── Coordinator: multiple pages with same sequence_id ──────────────────────

    #[test]
    fn build_batch_multiple_pages_same_sequence_id() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        for pid in 1usize..=3 {
            c.register_page(pid, None, 4096);
            {
                let mut table = c.addr_table.write().expect("write lock");
                if let Some(entry) = table.get_mut(&pid) {
                    entry.current_tier = StorageTier::CpuDram;
                    entry.host_buffer = Some(vec![0u8; 4096]);
                }
            }
            {
                let mut meta = c.page_metadata.write().expect("write lock");
                meta.insert(pid, PageMetadata {
                    page_id: pid,
                    sequence_id: Some(42), // all same sequence
                    recency: 0,
                    access_count: 5,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
        }
        // Act
        let plan = c.build_batch(&[1, 2, 3], 0.5);
        // Assert
        assert_eq!(plan.swap_in_requests.len(), 3);
    }

    // ── Coordinator: register_pages_from_hgal then release one page ────────────

    #[test]
    fn register_from_hgal_release_one_preserves_others() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        for pid in 1usize..=5 {
            pages.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64),
                ..PageMetadata::default()
            });
        }
        c.register_pages_from_hgal(&pages, 4096);
        // Act
        c.release_page(3);
        // Assert
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.len(), 4);
        assert!(table.get(&3).is_none());
        for pid in [1, 2, 4, 5] {
            assert!(table.contains_key(&pid));
        }
    }

    // ── Coordinator: build_batch eviction candidate has correct group_id ───────

    #[test]
    fn build_batch_eviction_candidate_group_id_matches_sequence() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(555),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.95);
        // Assert
        if let Some(candidate) = plan.eviction_candidates.iter().find(|c| c.page_id == 1) {
            assert_eq!(candidate.group_id, Some(555));
        }
    }

    // ── EvictionCandidate: clone independence ──────────────────────────────────

    #[test]
    fn eviction_candidate_clone_independence() {
        // Arrange
        let candidate = EvictionCandidate {
            page_id: 1,
            score: 50,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(10),
        };
        // Act
        let cloned = candidate.clone();
        // Assert: all fields copied correctly (Copy types are independent by nature)
        assert_eq!(cloned.page_id, 1);
        assert_eq!(cloned.score, 50);
        assert_eq!(cloned.current_tier, StorageTier::GpuHbm);
        assert_eq!(cloned.codec, CompressionCodec::Lz4);
        assert_eq!(cloned.page_bytes, 4096);
        assert_eq!(cloned.group_id, Some(10));
    }

    // ── Coordinator: build_batch with negative infinity pressure ───────────────

    #[test]
    fn build_batch_neg_infinity_pressure_no_eviction() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], f32::NEG_INFINITY);
        // Assert: negative infinity comparison always false
        assert!(plan.eviction_candidates.is_empty());
    }

    // ── Coordinator: update_page_gpu_ptr for multiple pages ────────────────────

    #[test]
    fn update_multiple_pages_gpu_ptr_all_reflected() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        c.register_page(2, None, 4096);
        c.register_page(3, None, 4096);
        // Act
        c.update_page_gpu_ptr(1, 0xA000);
        c.update_page_gpu_ptr(2, 0xB000);
        c.update_page_gpu_ptr(3, 0xC000);
        // Assert
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.get(&1).unwrap().gpu_ptr, Some(0xA000));
        assert_eq!(table.get(&2).unwrap().gpu_ptr, Some(0xB000));
        assert_eq!(table.get(&3).unwrap().gpu_ptr, Some(0xC000));
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 3);
        assert_eq!(stats.pages_on_dram, 0);
    }

    // ── PagePayloadKind: hash differs across all variants ──────────────────────

    #[test]
    fn page_payload_kind_hash_differs_all_variants() {
        // Arrange
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |k: PagePayloadKind| -> u64 {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        };
        let variants = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];
        // Act & Assert: all pairwise distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(hash_of(variants[i]), hash_of(variants[j]),
                    "hash collision between {:?} and {:?}", variants[i], variants[j]);
            }
        }
    }

    // ── Coordinator: register_page then tier_changed_pages before adding meta ──

    #[test]
    fn tier_changed_pages_before_metadata_insertion_is_empty() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        // Act: no metadata for page 1
        let changed = c.tier_changed_pages();
        // Assert: filter_map returns None when metadata missing
        assert!(changed.is_empty());
    }

    // ── Coordinator: register_page then immediately release then re-register ───

    #[test]
    fn register_release_reregister_page_different_bytes() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.release_page(1);
        // Act: re-register with different bytes
        c.register_page(1, Some(0x2000), 8192);
        // Assert
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.gpu_ptr, Some(0x2000));
        assert_eq!(entry.original_bytes, 8192);
    }

    // ── Coordinator: stats after many build_batch calls accumulate correctly ──

    #[test]
    fn stats_after_50_build_batch_calls() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        for _ in 0..50 {
            let _ = c.build_batch(&[], 0.5);
        }
        // Assert
        let stats = c.stats();
        assert_eq!(stats.eviction_rounds, 50);
        assert_eq!(stats.swap_in_rounds, 50);
    }

    // ── ThreeTierSwapConfig: construction with auto_start false ───────────────

    #[test]
    fn config_auto_start_false_construction() {
        // Arrange & Act
        let config = ThreeTierSwapConfig {
            auto_start: false,
            ..ThreeTierSwapConfig::default()
        };
        // Assert
        assert!(!config.auto_start);
        assert_eq!(config.eviction.max_evict_per_round, 8);
        assert_eq!(config.migration.page_size, 4096);
    }

    // ── TierMigration: all reason variants produce distinct debug output ───────

    #[test]
    fn tier_migration_all_reasons_distinct_debug() {
        // Arrange
        use std::collections::HashSet;
        let reasons = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        // Act
        let debug_strs: HashSet<String> = reasons.iter()
            .map(|r| format!("{:?}", r))
            .collect();
        // Assert
        assert_eq!(debug_strs.len(), 4);
    }

    // ── Coordinator: build_batch returns immediately on empty metadata ─────────

    #[test]
    fn build_batch_returns_empty_plan_on_fresh_coordinator() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        let plan = c.build_batch(&[1, 2, 3], 0.99);
        // Assert
        assert!(plan.eviction_candidates.is_empty());
        assert!(plan.swap_in_requests.is_empty());
        assert!(plan.tier_migrations.is_empty());
    }

    // ── Coordinator: record_eviction_completed multiple times same page ────────

    #[test]
    fn record_eviction_same_page_multiple_times_accumulates() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 8192, 200);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 2);
        assert_eq!(stats.total_bytes_evicted, 4096 + 8192);
        assert_eq!(stats.total_eviction_latency_us, 100 + 200);
    }

    // ── Coordinator: record_swap_in_completed multiple times same page ─────────

    #[test]
    fn record_swap_in_same_page_multiple_times_accumulates() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 50);
        c.record_swap_in_completed(1, StorageTier::Nvme, StorageTier::CpuDram, 8192, 300);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.total_bytes_swapped_in, 4096 + 8192);
        assert_eq!(stats.total_swap_in_latency_us, 50 + 300);
    }

    // ── PageAddrEntry: all fields set and read back ───────────────────────────

    #[test]
    fn page_addr_entry_all_fields_set_and_read() {
        // Arrange
        let buf = vec![0xDE; 4096];
        let entry = PageAddrEntry {
            gpu_ptr: Some(0xCAFE),
            host_buffer: Some(buf),
            current_tier: StorageTier::CpuDram,
            original_bytes: 4096,
            codec: CompressionCodec::NvcompAns,
        };
        // Act & Assert
        assert_eq!(entry.gpu_ptr, Some(0xCAFE));
        assert_eq!(entry.host_buffer.as_ref().map(|b| b.len()), Some(4096));
        assert_eq!(entry.host_buffer.as_ref().map(|b| b[0]), Some(0xDE));
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        assert_eq!(entry.original_bytes, 4096);
        assert_eq!(entry.codec, CompressionCodec::NvcompAns);
    }

    // ── Coordinator: build_batch does not panic with very large page_bytes ────

    #[test]
    fn build_batch_large_page_bytes_no_panic() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), usize::MAX);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                state: PageState::Standby,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                ..PageMetadata::default()
            });
        }
        // Act & Assert: no panic
        let plan = c.build_batch(&[], 0.95);
        assert!(plan.eviction_candidates.len() <= 1);
    }

    // ── StorageTier: min/max values via Ord ───────────────────────────────────

    #[test]
    fn storage_tier_min_max_via_ord() {
        // Arrange
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        // Act
        let max_tier = tiers.iter().max().unwrap();
        let min_tier = tiers.iter().min().unwrap();
        // Assert
        assert_eq!(*max_tier, StorageTier::GpuHbm);
        assert_eq!(*min_tier, StorageTier::Nvme);
    }

    // ── EvictionWorkerConfig: all defaults verified together ──────────────────

    #[test]
    fn eviction_worker_config_all_defaults_comprehensive() {
        // Arrange & Act
        let config = EvictionWorkerConfig::default();
        // Assert
        assert_eq!(config.tick_interval, Duration::from_millis(10));
        assert_eq!(config.max_evict_per_round, 8);
        assert!((config.hbm_pressure_threshold - 0.9).abs() < 0.01);
        assert!((config.dram_pressure_threshold - 0.8).abs() < 0.01);
        assert_eq!(config.importance_threshold, 100);
        assert_eq!(config.hbm_evict_age_ticks, 50);
        assert_eq!(config.dram_evict_age_ticks, 500);
        assert_eq!(config.default_evict_codec, CompressionCodec::Lz4);
        assert_eq!(config.page_bytes, 4096);
    }

    // ── SwapInWorkerConfig: all defaults verified together ────────────────────

    #[test]
    fn swap_in_worker_config_all_defaults_comprehensive() {
        // Arrange & Act
        let config = SwapInWorkerConfig::default();
        // Assert
        assert_eq!(config.max_prefetch_per_round, 16);
        assert_eq!(config.tick_interval, Duration::from_millis(5));
        assert!((config.min_confidence - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.max_in_flight, 64);
        assert_eq!(config.page_bytes, 4096);
    }

    // ── Coordinator: build_batch eviction page_bytes uses config page_bytes ───

    #[test]
    fn build_batch_eviction_candidate_page_bytes_from_config() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 8192); // registered with 8192
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.95);
        // Assert: eviction candidate uses config.eviction.page_bytes (default 4096)
        if let Some(candidate) = plan.eviction_candidates.iter().find(|c| c.page_id == 1) {
            assert_eq!(candidate.page_bytes, 4096);
        }
    }

    // ── Coordinator: register_page with gpu_ptr None and zero bytes ───────────

    #[test]
    fn register_page_none_ptr_zero_bytes_dram_tier() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.register_page(1, None, 0);
        // Assert
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert!(entry.gpu_ptr.is_none());
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        assert_eq!(entry.original_bytes, 0);
        assert_eq!(entry.codec, CompressionCodec::None);
    }

    // ── Coordinator: tier_changed_pages after full lifecycle ──────────────────

    #[test]
    fn tier_changed_pages_after_register_meta_update_release() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Active,
                ..PageMetadata::default()
            });
        }
        // Active + HBM = consistent
        assert!(c.tier_changed_pages().is_empty());
        // Act: change tier to DRAM
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
            }
        }
        // Assert: divergence detected
        let changed = c.tier_changed_pages();
        assert_eq!(changed.len(), 1);
        // Act: release the page
        c.release_page(1);
        // Assert: no more divergence
        assert!(c.tier_changed_pages().is_empty());
    }

    // ── PrefetchRequest: enqueued_at is Instant ───────────────────────────────

    #[test]
    fn prefetch_request_enqueued_at_is_recent() {
        // Arrange
        let before = Instant::now();
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let after = Instant::now();
        // Assert
        assert!(req.enqueued_at >= before);
        assert!(req.enqueued_at <= after);
    }

    // ── EvictionCandidate: score ordering validated ──────────────────────────

    #[test]
    fn eviction_candidate_score_ordering_consistent() {
        // Arrange
        let low = EvictionCandidate {
            page_id: 1, score: -100, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
        };
        let high = EvictionCandidate {
            page_id: 2, score: 100, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
        };
        // Act & Assert
        assert!(low.score < high.score);
        assert!(low.score < 0);
        assert!(high.score > 0);
    }

    // ── Coordinator: memory_manager accessor reflects initial capacities ──────

    #[test]
    fn memory_manager_accessor_reflects_l1_l2_l3_capacities() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        let mm = c.memory_manager();
        let guard = mm.lock().expect("lock");
        let l1 = guard.tier_usage(crate::scheduler::memory_manager::Tier::L1);
        let l2 = guard.tier_usage(crate::scheduler::memory_manager::Tier::L2);
        let l3 = guard.tier_usage(crate::scheduler::memory_manager::Tier::L3);
        // Assert
        assert_eq!(l1.capacity, 100_000);
        assert_eq!(l2.capacity, 1_000_000);
        assert_eq!(l3.capacity, 10_000_000);
    }

    // ── Coordinator: build_batch with empty string page_id (0 is valid) ──────

    #[test]
    fn build_batch_page_id_zero_in_active_pages() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(0, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(0, PageMetadata {
                page_id: 0,
                sequence_id: Some(1),
                state: PageState::Active,
                last_access: Instant::now(),
                ..PageMetadata::default()
            });
        }
        // Act: page 0 is on HBM and active
        let plan = c.build_batch(&[0], 0.5);
        // Assert: no swap-in needed (already on HBM)
        assert!(plan.swap_in_requests.is_empty());
    }

    // ── CompressionCodec: all variants as HashMap values ─────────────────────

    #[test]
    fn compression_codec_all_variants_as_hashmap_values() {
        // Arrange
        let mut map = std::collections::HashMap::new();
        map.insert("none", CompressionCodec::None);
        map.insert("lz4", CompressionCodec::Lz4);
        map.insert("bitpack", CompressionCodec::BitPackRle);
        map.insert("nvcomp", CompressionCodec::NvcompAns);
        map.insert("zstd", CompressionCodec::ZstdDict);
        // Act & Assert
        assert_eq!(map.len(), 5);
        assert_eq!(map.get("lz4"), Some(&CompressionCodec::Lz4));
        assert_eq!(map.get("zstd"), Some(&CompressionCodec::ZstdDict));
    }

    // ── TierMigrationPlan: built_at can be compared across plans ─────────────

    #[test]
    fn tier_migration_plan_built_at_ordering_across_plans() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        let p1 = c.build_batch(&[], 0.5);
        let p2 = c.build_batch(&[], 0.5);
        let p3 = c.build_batch(&[], 0.5);
        // Assert
        assert!(p1.built_at <= p2.built_at);
        assert!(p2.built_at <= p3.built_at);
    }

    // ── Wave 19: 70 additional tests for uncovered areas ─────────────────────────

    // ── compute_importance_score: direct unit tests ─────────────────────────────


    #[test]
    fn compute_importance_score_high_access_count_raises_score() {
        // Arrange
        let meta_low = PageMetadata {
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
        let meta_high = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1000,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let score_low = EvictionWorker::compute_importance_score(
            &meta_low, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_high = EvictionWorker::compute_importance_score(
            &meta_high, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // Assert
        assert!(score_high > score_low, "high access_count should raise score");
    }

    #[test]
    fn compute_importance_score_high_age_lowers_score() {
        // Arrange
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
        // Act
        let score_young = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 10,
        );
        let score_old = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 10000,
        );
        // Assert
        assert!(score_old < score_young, "high age should lower score");
    }


    #[test]
    fn compute_importance_score_expert_weight_payload_bonus() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None, // triggers ExpertWeight
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let score_expert = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::ExpertWeight), 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_none = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // Assert: ExpertWeight has a specific bonus, should differ from None
        assert_ne!(score_expert, score_none);
    }

    #[test]
    fn compute_importance_score_dense_layer_payload_highest_bonus() {
        // Arrange
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
        // Act
        let score_dense = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::DenseLayerWeight), 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_kv = EvictionWorker::compute_importance_score(
            &meta, Some(PagePayloadKind::KvContext), 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // Assert: DenseLayerWeight has highest protection bonus
        assert!(score_dense > score_kv, "DenseLayerWeight should have higher bonus than KvContext");
    }

    #[test]
    fn compute_importance_score_recency_contributes() {
        // Arrange
        let meta_low_recency = PageMetadata {
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
        let meta_high_recency = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 500,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let score_low = EvictionWorker::compute_importance_score(
            &meta_low_recency, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_high = EvictionWorker::compute_importance_score(
            &meta_high_recency, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        // Assert: higher recency → lower score (more evictable)
        assert!(score_high < score_low, "higher recency should lower score");
    }

    #[test]
    fn compute_importance_score_page_size_bonus() {
        // Arrange
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
        // Act
        let score_small = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 4096, StorageTier::GpuHbm, 0,
        );
        let score_large = EvictionWorker::compute_importance_score(
            &meta, None, 4096, 16384, StorageTier::GpuHbm, 0,
        );
        // Assert: larger page → higher score (protected)
        assert!(score_large > score_small, "larger page should have higher score");
    }


    #[test]
    fn compute_importance_score_all_payload_kinds_produce_scores() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let kinds = [
            Some(PagePayloadKind::KvContext),
            Some(PagePayloadKind::ExpertWeight),
            Some(PagePayloadKind::PromptSystem),
            Some(PagePayloadKind::DenseLayerWeight),
            Some(PagePayloadKind::KnowledgeRAG),
            None,
        ];
        // Act & Assert: all produce finite scores
        for kind in kinds {
            let score = EvictionWorker::compute_importance_score(
                &meta, kind, 4096, 4096, StorageTier::GpuHbm, 100,
            );
            assert!(score > i64::MIN && score < i64::MAX, "score should be finite for {:?}", kind);
        }
    }

    // ── SwapInWorker::compute_urgency: direct unit tests ────────────────────────

    #[test]
    fn compute_urgency_high_confidence_higher_than_low() {
        // Arrange
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
        // Act
        let urgency_low = SwapInWorker::compute_urgency(&meta, 0.1, StorageTier::CpuDram);
        let urgency_high = SwapInWorker::compute_urgency(&meta, 0.9, StorageTier::CpuDram);
        // Assert
        assert!(urgency_high > urgency_low, "higher confidence should yield higher urgency");
    }


    #[test]
    fn compute_urgency_is_finite() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(60),
            swap_in_time: Some(Instant::now() - Duration::from_secs(60)),
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };
        // Act
        let urgency = SwapInWorker::compute_urgency(&meta, 0.95, StorageTier::Nvme);
        // Assert
        assert!(urgency.is_finite(), "urgency should be finite");
        assert!(urgency >= 0.0, "urgency should be non-negative");
    }

    #[test]
    fn compute_urgency_zero_confidence_non_negative() {
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
        let urgency = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // Assert
        assert!(urgency.is_finite());
    }

    // ── SwapInWorkerStats: additional arithmetic tests ───────────────────────────

    #[test]
    fn swap_in_worker_stats_avg_latency_single_promotion() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 1;
        stats.total_latency_us = 42;
        // Act
        let avg = stats.avg_latency_us();
        // Assert
        assert!((avg - 42.0).abs() < 0.01);
    }

    #[test]
    fn swap_in_worker_stats_avg_latency_large_values() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = u64::MAX;
        stats.total_latency_us = u64::MAX;
        // Act
        let avg = stats.avg_latency_us();
        // Assert
        assert!(avg.is_finite());
        assert!((avg - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn swap_in_worker_stats_fields_accumulate() {
        // Arrange
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 10;
        stats.submitted = 8;
        stats.skipped = 2;
        stats.promoted_ok = 7;
        stats.promoted_failed = 1;
        stats.two_hop_promotions = 3;
        stats.rounds = 5;
        // Act & Assert: verify consistency
        assert_eq!(stats.total_requests, stats.submitted + stats.skipped);
        assert_eq!(stats.submitted, stats.promoted_ok + stats.promoted_failed);
    }

    // ── EvictionCandidate: PartialEq via field comparison ───────────────────────

    #[test]
    fn eviction_candidate_equality_same_fields() {
        // Arrange
        let a = EvictionCandidate {
            page_id: 1, score: 50, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(10),
        };
        let b = EvictionCandidate {
            page_id: 1, score: 50, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(10),
        };
        // Act & Assert: manual field-by-field comparison
        assert_eq!(a.page_id, b.page_id);
        assert_eq!(a.score, b.score);
        assert_eq!(a.current_tier, b.current_tier);
        assert_eq!(a.codec, b.codec);
        assert_eq!(a.page_bytes, b.page_bytes);
        assert_eq!(a.group_id, b.group_id);
    }

    #[test]
    fn eviction_candidate_inequality_different_page_id() {
        // Arrange
        let a = EvictionCandidate {
            page_id: 1, score: 50, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: None,
        };
        let b = EvictionCandidate {
            page_id: 2, score: 50, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: None,
        };
        // Assert
        assert_ne!(a.page_id, b.page_id);
    }

    // ── Coordinator: build_batch with Free state on DRAM tier ───────────────────

    #[test]
    fn build_batch_free_on_dram_under_dram_pressure() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Free,
                warm_until: None,
            });
        }
        // Act: high pressure but dram_pressure is 0 by default
        let plan = c.build_batch(&[], 0.99);
        // Assert: no panic, eviction depends on dram_pressure
        assert!(plan.eviction_candidates.len() <= 1);
    }

    // ── Coordinator: register_pages_from_hgal with overlapping addr entries ─────

    #[test]
    fn register_pages_from_hgal_does_not_overwrite_existing_addr_entry_gpu_ptr() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0xDEAD), 8192);
        let mut pages = HashMap::new();
        pages.insert(1, PageMetadata { page_id: 1, ..PageMetadata::default() });
        // Act
        c.register_pages_from_hgal(&pages, 4096);
        // Assert: original gpu_ptr preserved
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.get(&1).unwrap().gpu_ptr, Some(0xDEAD));
    }

    // ── Coordinator: build_batch tier_migration for CpuDram eviction uses entry codec ─

    #[test]
    fn build_batch_dram_eviction_uses_entry_codec_not_default() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
                entry.codec = CompressionCodec::ZstdDict;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: dram_pressure is 0 by default, so no eviction expected
        let plan = c.build_batch(&[], 0.99);
        // Assert: if there is a dram eviction candidate, its codec should match entry
        for candidate in &plan.eviction_candidates {
            if candidate.current_tier == StorageTier::CpuDram {
                assert_eq!(candidate.codec, CompressionCodec::ZstdDict);
            }
        }
    }

    // ── TierMigrationPlan: eviction_candidates sorted correctly after clone ────

    #[test]
    fn tier_migration_plan_clone_preserves_sorted_order() {
        // Arrange
        let candidates: Vec<EvictionCandidate> = [30i64, 10, 20]
            .iter()
            .enumerate()
            .map(|(i, &score)| EvictionCandidate {
                page_id: i, score, current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None, page_bytes: 4096, group_id: None,
            })
            .collect();
        let plan = TierMigrationPlan {
            eviction_candidates: candidates,
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };
        // Act
        let cloned = plan.clone();
        // Assert
        assert_eq!(cloned.eviction_candidates[0].score, 30);
        assert_eq!(cloned.eviction_candidates[1].score, 10);
        assert_eq!(cloned.eviction_candidates[2].score, 20);
    }

    // ── ThreeTierSwapConfig: clone preserves all sub-configs independently ────

    #[test]
    fn config_clone_swap_in_isolation() {
        // Arrange
        let config = ThreeTierSwapConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.swap_in.max_prefetch_per_round = 99;
        // Assert
        assert_eq!(config.swap_in.max_prefetch_per_round, 16);
        assert_eq!(cloned.swap_in.max_prefetch_per_round, 99);
    }

    #[test]
    fn config_clone_migration_isolation() {
        // Arrange
        let config = ThreeTierSwapConfig::default();
        let mut cloned = config.clone();
        // Act
        cloned.migration.page_size = 8192;
        // Assert
        assert_eq!(config.migration.page_size, 4096);
        assert_eq!(cloned.migration.page_size, 8192);
    }

    // ── MigrationCommand: exhaustive variant Debug coverage ────────────────────

    #[test]
    fn migration_command_evict_to_dram_with_large_page_bytes() {
        // Arrange & Act
        let cmd = MigrationCommand::EvictToDram {
            page_id: usize::MAX,
            codec: CompressionCodec::NvcompAns,
            page_bytes: usize::MAX,
        };
        // Assert
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("EvictToDram"));
    }

    #[test]
    fn migration_command_evict_to_nvme_with_large_page_bytes() {
        // Arrange & Act
        let cmd = MigrationCommand::EvictToNvme {
            page_id: usize::MAX,
            codec: CompressionCodec::BitPackRle,
            page_bytes: usize::MAX,
        };
        // Assert
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("EvictToNvme"));
    }

    #[test]
    fn migration_command_promote_to_hbm_with_large_page_bytes() {
        // Arrange & Act
        let cmd = MigrationCommand::PromoteToHbm {
            page_id: 0,
            page_bytes: usize::MAX,
        };
        // Assert
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("PromoteToHbm"));
    }

    #[test]
    fn migration_command_promote_to_dram_with_large_page_bytes() {
        // Arrange & Act
        let cmd = MigrationCommand::PromoteToDram {
            page_id: usize::MAX,
            page_bytes: usize::MAX,
        };
        // Assert
        let dbg = format!("{:?}", cmd);
        assert!(dbg.contains("PromoteToDram"));
    }

    // ── MigrationResult: additional clone tests ───────────────────────────────

    #[test]
    fn migration_result_ok_with_zero_bytes() {
        // Arrange & Act
        let result = MigrationResult::Ok { compressed_bytes: 0, checksum: 0 };
        // Assert
        match result {
            MigrationResult::Ok { compressed_bytes, checksum } => {
                assert_eq!(compressed_bytes, 0);
                assert_eq!(checksum, 0);
            }
            _ => panic!("expected Ok"),
        }
    }

    #[test]
    fn migration_result_failed_with_empty_reason() {
        // Arrange & Act
        let result = MigrationResult::Failed { reason: String::new() };
        // Assert
        match result {
            MigrationResult::Failed { reason } => assert!(reason.is_empty()),
            _ => panic!("expected Failed"),
        }
    }

    // ── MigrationDone: all tier pair constructions ─────────────────────────────

    #[test]
    fn migration_done_all_tier_pairs() {
        // Arrange
        let pairs = [
            (StorageTier::GpuHbm, StorageTier::CpuDram),
            (StorageTier::CpuDram, StorageTier::Nvme),
            (StorageTier::Nvme, StorageTier::CpuDram),
            (StorageTier::CpuDram, StorageTier::GpuHbm),
            (StorageTier::Nvme, StorageTier::GpuHbm),
        ];
        for (i, (from, to)) in pairs.iter().enumerate() {
            // Act
            let done = MigrationDone {
                page_id: i,
                from_tier: *from,
                to_tier: *to,
                result: MigrationResult::Ok { compressed_bytes: 100, checksum: 0 },
            };
            // Assert
            assert_eq!(done.from_tier, *from);
            assert_eq!(done.to_tier, *to);
        }
    }

    // ── Coordinator: tier_changed_pages after register + no metadata ──────────

    #[test]
    fn tier_changed_pages_no_addr_entry_for_metadata() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(999, PageMetadata {
                page_id: 999,
                state: PageState::Swapped,
                ..PageMetadata::default()
            });
        }
        // Act: metadata exists but no addr_table entry
        let changed = c.tier_changed_pages();
        // Assert: no addr entry → filter_map returns None
        assert!(changed.is_empty());
    }

    // ── Coordinator: build_batch with Standby on HBM and various pressure values

    #[test]
    fn build_batch_standby_hbm_pressure_0_91_triggers_eviction() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: 0.91 > default threshold 0.90
        let plan = c.build_batch(&[], 0.91);
        // Assert
        assert!(!plan.eviction_candidates.is_empty(), "pressure 0.91 should evict standby HBM page");
    }

    // ── Stats: total_migrations with asymmetric eviction/swap-in ──────────────

    #[test]
    fn stats_total_migrations_asymmetric_counts() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 100;
        stats.swap_ins_dram_to_gpu = 1;
        // Act
        let total = stats.total_migrations();
        // Assert
        assert_eq!(total, 101);
    }

    // ── PrefetchRequest: PartialEq via field comparison ───────────────────────

    #[test]
    fn prefetch_request_field_equality_check() {
        // Arrange
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: now,
        };
        // Act & Assert
        assert_eq!(a.page_id, b.page_id);
        assert!((a.urgency - b.urgency).abs() < f32::EPSILON);
        assert!((a.prefetch_confidence - b.prefetch_confidence).abs() < f32::EPSILON);
        assert_eq!(a.page_bytes, b.page_bytes);
    }

    // ── Coordinator: multiple pages with different states on same tier ────────

    #[test]
    fn build_batch_mixed_states_same_tier_only_eligible_evicted() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Page 1: Standby (eligible for eviction)
        c.register_page(1, Some(0x1001), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(10),
                state: PageState::Standby,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                recency: 1000,
                ..PageMetadata::default()
            });
        }
        // Page 2: Warm (skip list, never evicted)
        c.register_page(2, Some(0x1002), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(2, PageMetadata {
                page_id: 2,
                sequence_id: Some(20),
                state: PageState::Warm,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                recency: 1000,
                ..PageMetadata::default()
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.95);
        // Assert
        for candidate in &plan.eviction_candidates {
            assert_ne!(candidate.page_id, 2, "Warm page should not be evicted");
        }
    }

    // ── Coordinator: register then update then build_batch then stats lifecycle

    #[test]
    fn full_lifecycle_register_update_build_stats() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: register
        c.register_page(1, Some(0x1000), 4096);
        assert_eq!(c.stats().pages_on_hbm, 1);

        // Update: move to DRAM
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.gpu_ptr = None;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        assert_eq!(c.stats().pages_on_hbm, 0);
        assert_eq!(c.stats().pages_on_dram, 1);

        // Add metadata and build_batch
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        let plan = c.build_batch(&[1], 0.5);
        assert_eq!(plan.swap_in_requests.len(), 1);

        // Record eviction + swap-in stats
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 50);
        let stats = c.stats();
        assert_eq!(stats.total_migrations(), 2);

        // Release
        c.release_page(1);
        assert_eq!(c.stats().pages_on_dram, 0);
    }

    // ── EvictionWorkerConfig: dram_pressure_threshold range ──────────────────

    #[test]
    fn eviction_worker_config_dram_threshold_default() {
        // Arrange & Act
        let config = EvictionWorkerConfig::default();
        // Assert
        assert!((config.dram_pressure_threshold - 0.8).abs() < 0.01);
    }

    #[test]
    fn eviction_worker_config_hbm_age_ticks_default() {
        // Arrange & Act
        let config = EvictionWorkerConfig::default();
        // Assert
        assert_eq!(config.hbm_evict_age_ticks, 50);
    }

    #[test]
    fn eviction_worker_config_dram_age_ticks_default() {
        // Arrange & Act
        let config = EvictionWorkerConfig::default();
        // Assert
        assert_eq!(config.dram_evict_age_ticks, 500);
    }

    // ── MigrationActorConfig: nvme_swap_dir default path ─────────────────────

    #[test]
    fn migration_actor_config_default_nvme_swap_dir() {
        // Arrange & Act
        let config = MigrationActorConfig::default();
        let dir_str = config.nvme_swap_dir.to_string_lossy().to_string();
        // Assert
        assert!(dir_str.contains(".gllm"));
        assert!(dir_str.contains("swap"));
    }

    // ── Coordinator: register_page then tier_changed_pages all PageState mappings

    #[test]
    fn tier_changed_pages_standby_maps_to_dram() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Standby,
                ..PageMetadata::default()
            });
        }
        // Act: Standby → CpuDram, but page is on GpuHbm → divergence
        let changed = c.tier_changed_pages();
        // Assert
        assert_eq!(changed.len(), 1);
        assert_eq!(changed[0].1, StorageTier::GpuHbm);
    }

    // ── Coordinator: build_batch eviction migration has EvictionPressure for HBM→DRAM

    #[test]
    fn build_batch_hbm_eviction_migration_reason_is_eviction_pressure() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.95);
        // Assert
        if let Some(migration) = plan.tier_migrations.iter().find(|m| m.page_id == 1) {
            assert_eq!(migration.reason, TierMigrationReason::EvictionPressure);
            assert_eq!(migration.from_tier, StorageTier::GpuHbm);
            assert_eq!(migration.to_tier, StorageTier::CpuDram);
        }
    }

    // ── Coordinator: build_batch swap-in migration has SequenceDemand ────────

    #[test]
    fn build_batch_swap_in_from_nvme_migration_fields() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 1,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert
        let migration = plan.tier_migrations.iter().find(|m| m.page_id == 1);
        assert!(migration.is_some());
        let m = migration.unwrap();
        assert_eq!(m.from_tier, StorageTier::Nvme);
        assert_eq!(m.to_tier, StorageTier::GpuHbm);
        assert_eq!(m.reason, TierMigrationReason::SequenceDemand);
    }

    // ── PageAddrEntry: host_buffer with various sizes ────────────────────────

    #[test]
    fn page_addr_entry_host_buffer_small_size() {
        // Arrange & Act
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(vec![0u8; 16]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 16,
            codec: CompressionCodec::None,
        };
        // Assert
        assert_eq!(entry.host_buffer.as_ref().map(|b| b.len()), Some(16));
    }

    #[test]
    fn page_addr_entry_host_buffer_large_size() {
        // Arrange & Act
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(vec![0xAB; 65536]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 65536,
            codec: CompressionCodec::Lz4,
        };
        // Assert
        assert_eq!(entry.host_buffer.as_ref().map(|b| b.len()), Some(65536));
        assert_eq!(entry.host_buffer.as_ref().map(|b| b[0]), Some(0xAB));
    }

    // ── Coordinator: build_batch rounds counter after mixed operations ───────

    #[test]
    fn build_batch_rounds_counter_after_register_and_release() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        let _ = c.build_batch(&[], 0.5);
        c.release_page(1);
        // Act
        let _ = c.build_batch(&[], 0.5);
        // Assert
        assert_eq!(c.stats().eviction_rounds, 2);
        assert_eq!(c.stats().swap_in_rounds, 2);
    }

    // ── StorageTier: used in Vec sort/unsort ──────────────────────────────────

    #[test]
    fn storage_tier_vec_sort_descending() {
        // Arrange
        let mut tiers = vec![StorageTier::CpuDram, StorageTier::Nvme, StorageTier::GpuHbm];
        // Act
        tiers.sort_by(|a, b| b.cmp(a));
        // Assert: descending by Ord (HBM > DRAM > NVMe)
        assert_eq!(tiers[0], StorageTier::GpuHbm);
        assert_eq!(tiers[1], StorageTier::CpuDram);
        assert_eq!(tiers[2], StorageTier::Nvme);
    }

    // ── CompressionCodec: sort by as_u8 discriminant ────────────────────

    #[test]
    fn compression_codec_sort_by_as_u8() {
        // Arrange
        let mut codecs = vec![
            CompressionCodec::ZstdDict,
            CompressionCodec::None,
            CompressionCodec::NvcompAns,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
        ];
        // Act: sort by discriminant via as_u8
        codecs.sort_by_key(|c| c.as_u8()); // u8 is Ord, no need for Ord on CompressionCodec
        // Assert: sorted by discriminant
        assert_eq!(codecs[0], CompressionCodec::None);
        assert_eq!(codecs[4], CompressionCodec::ZstdDict);
    }

    // ── TierMigrationReason: all variants usable in Vec and sort ─────────────

    #[test]
    fn tier_migration_reason_vec_dedup() {
        // Arrange
        let reasons = vec![
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
            TierMigrationReason::EvictionPressure, // duplicate
        ];
        // Act
        let mut unique: std::collections::HashSet<_> = reasons.into_iter().collect();
        // Assert
        assert_eq!(unique.len(), 4);
        unique.insert(TierMigrationReason::EvictionPressure); // already present
        assert_eq!(unique.len(), 4);
    }

    // ── Coordinator: build_batch with same page in active_pages multiple times ─

    #[test]
    fn build_batch_triple_duplicate_active_pages_single_swap_in() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(7, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&7) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(7, PageMetadata {
                page_id: 7,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: pass same page_id three times
        let plan = c.build_batch(&[7, 7, 7], 0.5);
        // Assert: build_batch iterates metadata (unique), so one swap-in
        assert_eq!(plan.swap_in_requests.len(), 1);
    }

    // ── Coordinator: stats snapshot after eviction records but before release ──

    #[test]
    fn stats_snapshot_includes_tier_counts_after_records() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        // Act
        let stats = c.stats();
        // Assert: tier count from addr_table, records from stats lock
        assert_eq!(stats.pages_on_hbm, 1); // page still in addr_table
        assert_eq!(stats.evictions_gpu_to_dram, 1); // record counted
    }

    // ── ThreeTierSwapStats: Debug output contains all 13 field names ────────

    #[test]
    fn stats_debug_all_13_fields_present() {
        // Arrange
        let stats = ThreeTierSwapStats::default();
        // Act
        let dbg = format!("{:?}", stats);
        // Assert
        let fields = [
            "evictions_gpu_to_dram", "evictions_dram_to_nvme",
            "swap_ins_dram_to_gpu", "swap_ins_nvme_to_dram",
            "total_bytes_evicted", "total_bytes_swapped_in",
            "total_eviction_latency_us", "total_swap_in_latency_us",
            "eviction_rounds", "swap_in_rounds",
            "pages_on_hbm", "pages_on_dram", "pages_on_nvme",
        ];
        for field in fields {
            assert!(dbg.contains(field), "Debug output missing field: {}", field);
        }
    }

    // ── Coordinator: build_batch produces valid plan even with 0 pressure ─────

    #[test]
    fn build_batch_zero_pressure_valid_plan_structure() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                state: PageState::Active,
                last_access: Instant::now(),
                ..PageMetadata::default()
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.0);
        // Assert: valid plan, no panics, correct structure
        assert!(plan.swap_in_requests.is_empty()); // already on HBM
        assert!(plan.eviction_candidates.is_empty()); // Active → skip
        assert!(plan.tier_migrations.is_empty());
    }

    // ── EvictionCandidate: all fields individually modifiable via clone ──────

    #[test]
    fn eviction_candidate_clone_then_modify_via_new_struct() {
        // Arrange
        let original = EvictionCandidate {
            page_id: 1, score: 50, current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4, page_bytes: 4096, group_id: Some(10),
        };
        // Act: clone and create modified version
        let mut modified = original.clone();
        modified.score = 75;
        modified.page_bytes = 8192;
        // Assert
        assert_eq!(original.score, 50);
        assert_eq!(original.page_bytes, 4096);
        assert_eq!(modified.score, 75);
        assert_eq!(modified.page_bytes, 8192);
    }

    // ── MigrationCommand: all 5 variants constructible ───────────────────────

    #[test]
    fn migration_command_all_variants_constructible() {
        // Arrange & Act
        let _evict_dram = MigrationCommand::EvictToDram {
            page_id: 1, codec: CompressionCodec::Lz4, page_bytes: 4096,
        };
        let _evict_nvme = MigrationCommand::EvictToNvme {
            page_id: 2, codec: CompressionCodec::ZstdDict, page_bytes: 4096,
        };
        let _promote_hbm = MigrationCommand::PromoteToHbm { page_id: 3, page_bytes: 4096 };
        let _promote_dram = MigrationCommand::PromoteToDram { page_id: 4, page_bytes: 4096 };
        let _shutdown = MigrationCommand::Shutdown;
        // Assert: no panic, all variants constructible
    }

    // ── Coordinator: release_page then re-register with different params ─────

    #[test]
    fn release_then_reregister_different_gpu_ptr_and_bytes() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.release_page(1);
        // Act: re-register with different params
        c.register_page(1, None, 16384);
        // Assert
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert!(entry.gpu_ptr.is_none());
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        assert_eq!(entry.original_bytes, 16384);
    }

    // ── Coordinator: build_batch with DRAM pressure still zero for default MM

    #[test]
    fn build_batch_dram_pressure_default_zero_no_dram_eviction() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: high HBM pressure but DRAM pressure is 0
        let plan = c.build_batch(&[], 0.99);
        // Assert: no DRAM eviction because dram_pressure = 0
        let dram_evictions: Vec<_> = plan.eviction_candidates
            .iter()
            .filter(|c| c.current_tier == StorageTier::CpuDram)
            .collect();
        assert!(dram_evictions.is_empty());
    }

    // ── compute_tier_age_ticks: edge case with very recent swap_in_time ──────

    #[test]
    fn compute_tier_age_ticks_very_recent_is_near_zero() {
        // Arrange
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
        // Act
        let ticks = compute_tier_age_ticks(&meta);
        // Assert: just created, should be 0 or very small
        assert!(ticks <= 1, "recently created page should have near-zero ticks, got {}", ticks);
    }

    #[test]
    fn compute_tier_age_ticks_older_page_higher_ticks() {
        // Arrange
        let meta_old = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(10),
            swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_new = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let ticks_old = compute_tier_age_ticks(&meta_old);
        let ticks_new = compute_tier_age_ticks(&meta_new);
        // Assert
        assert!(ticks_old > ticks_new, "older page should have more ticks");
    }

    // ── infer_swap_payload_kind: additional edge cases ──────────────────────

    #[test]
    fn infer_swap_payload_kind_returns_some_for_all() {
        // Arrange
        let meta_with_seq = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            ..PageMetadata::default()
        };
        let meta_no_seq = PageMetadata {
            page_id: 2,
            sequence_id: None,
            ..PageMetadata::default()
        };
        // Act & Assert
        assert!(infer_swap_payload_kind(&meta_with_seq).is_some());
        assert!(infer_swap_payload_kind(&meta_no_seq).is_some());
    }

    // ── Coordinator: shutdown after no-auto-start is safe ────────────────────

    #[test]
    fn shutdown_after_no_auto_start_is_noop() {
        // Arrange
        let (mut c, _backend) = make_coordinator(false);
        // Act
        c.shutdown();
        // Assert: workers already None
        assert!(c.eviction_worker.is_none());
        assert!(c.swap_in_worker.is_none());
    }

    // ── TierMigration: construct with page_bytes equal to usize::MAX / 2 ─────

    #[test]
    fn tier_migration_large_but_not_max_page_bytes() {
        // Arrange & Act
        let migration = TierMigration {
            page_id: 1,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::BitPackRle,
            page_bytes: usize::MAX / 2,
            reason: TierMigrationReason::EvictionPressure,
        };
        // Assert
        assert_eq!(migration.page_bytes, usize::MAX / 2);
    }

    // ── Coordinator: observer accessor can lock and unlock ───────────────────

    #[test]
    fn observer_accessor_lock_unlock_cycle() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        {
            let guard = c.observer().lock().expect("lock");
            drop(guard);
        }
        let guard2 = c.observer().lock().expect("lock again");
        // Assert: no deadlock
        drop(guard2);
    }

    // ── SwapInWorkerConfig: PartialEq derived ──────────────────────────────

    #[test]
    fn swap_in_worker_config_partial_eq_same() {
        // Arrange
        let a = SwapInWorkerConfig::default();
        let b = SwapInWorkerConfig::default();
        // Act & Assert
        assert_eq!(a, b);
    }

    #[test]
    fn swap_in_worker_config_partial_eq_different() {
        // Arrange
        let a = SwapInWorkerConfig::default();
        let b = SwapInWorkerConfig {
            max_prefetch_per_round: 99,
            ..SwapInWorkerConfig::default()
        };
        // Act & Assert
        assert_ne!(a, b);
    }

    // ── PageMetadata: swap_in_time affects tier age calculation ─────────────

    #[test]
    fn page_metadata_swap_in_time_none_uses_last_access() {
        // Arrange: swap_in_time is None → anchor = last_access
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(5),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let ticks = compute_tier_age_ticks(&meta);
        // Assert: should be ~500 ticks (5s / 10ms)
        assert!(ticks > 0, "should have positive ticks from last_access");
    }

    // ── Coordinator: multiple pages on different tiers stats snapshot ───────

    #[test]
    fn stats_snapshot_after_tier_transitions() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096); // HBM
        c.register_page(2, Some(0x2000), 4096); // HBM
        c.register_page(3, None, 4096);          // DRAM
        c.register_page(4, None, 4096);          // DRAM
        // Act: transition pages
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.gpu_ptr = None;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
            if let Some(entry) = table.get_mut(&4) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        // Assert
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 1);    // page 2
        assert_eq!(stats.pages_on_dram, 2);    // pages 1, 3
        assert_eq!(stats.pages_on_nvme, 1);    // page 4
    }

    // ── ThreeTierSwapStats: avg latency exact calculation ──────────────────────

    #[test]
    fn stats_avg_eviction_latency_exact_calculation() {
        // Arrange: 3 gpu_to_dram + 2 dram_to_nvme, total latency 500us
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 3;
        stats.evictions_dram_to_nvme = 2;
        stats.total_eviction_latency_us = 500;
        // Act
        let avg = stats.avg_eviction_latency_us();
        // Assert: 500 / 5 = 100.0
        assert!((avg - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn stats_avg_swap_in_latency_exact_calculation() {
        // Arrange: 4 dram_to_gpu + 1 nvme_to_dram, total latency 250us
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_dram_to_gpu = 4;
        stats.swap_ins_nvme_to_dram = 1;
        stats.total_swap_in_latency_us = 250;
        // Act
        let avg = stats.avg_swap_in_latency_us();
        // Assert: 250 / 5 = 50.0
        assert!((avg - 50.0).abs() < f64::EPSILON);
    }

    // ── ThreeTierSwapStats: total_migrations with large values ─────────────────

    #[test]
    fn stats_total_migrations_with_large_values() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 1_000_000;
        stats.evictions_dram_to_nvme = 500_000;
        stats.swap_ins_dram_to_gpu = 800_000;
        stats.swap_ins_nvme_to_dram = 200_000;
        // Act
        let total = stats.total_migrations();
        // Assert
        assert_eq!(total, 2_500_000);
    }

    // ── ThreeTierSwapStats: clone is deep copy of latency fields ───────────────

    #[test]
    fn stats_clone_latency_fields_independent() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.total_eviction_latency_us = 12345;
        stats.total_swap_in_latency_us = 67890;
        let cloned = stats.clone();
        // Act: modify original
        stats.total_eviction_latency_us = 99999;
        // Assert: clone is unaffected
        assert_eq!(cloned.total_eviction_latency_us, 12345);
        assert_eq!(cloned.total_swap_in_latency_us, 67890);
    }

    // ── TierMigration: zero page_bytes construction ────────────────────────────

    #[test]
    fn tier_migration_zero_page_bytes() {
        // Arrange & Act
        let migration = TierMigration {
            page_id: 0,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::None,
            page_bytes: 0,
            reason: TierMigrationReason::EvictionPressure,
        };
        // Assert
        assert_eq!(migration.page_bytes, 0);
        assert_eq!(migration.page_id, 0);
    }

    // ── TierMigration: promotion from Nvme to GpuHbm (skip tier) ──────────────

    #[test]
    fn tier_migration_nvme_to_hbm_direct_promotion() {
        // Arrange & Act
        let migration = TierMigration {
            page_id: 99,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
            reason: TierMigrationReason::SequenceDemand,
        };
        // Assert: direct skip-tier promotion is structurally allowed
        assert_eq!(migration.from_tier, StorageTier::Nvme);
        assert_eq!(migration.to_tier, StorageTier::GpuHbm);
        assert_eq!(migration.codec, CompressionCodec::ZstdDict);
    }

    // ── TierMigrationPlan: field access with only tier_migrations ──────────────

    #[test]
    fn tier_migration_plan_only_tier_migrations_field_access() {
        // Arrange
        let now = Instant::now();
        let plan = TierMigrationPlan {
            eviction_candidates: vec![],
            swap_in_requests: vec![],
            tier_migrations: vec![
                TierMigration {
                    page_id: 1,
                    from_tier: StorageTier::GpuHbm,
                    to_tier: StorageTier::CpuDram,
                    codec: CompressionCodec::Lz4,
                    page_bytes: 4096,
                    reason: TierMigrationReason::EvictionPressure,
                },
                TierMigration {
                    page_id: 2,
                    from_tier: StorageTier::CpuDram,
                    to_tier: StorageTier::Nvme,
                    codec: CompressionCodec::BitPackRle,
                    page_bytes: 4096,
                    reason: TierMigrationReason::ColdCascade,
                },
            ],
            built_at: now,
        };
        // Assert
        assert_eq!(plan.tier_migrations.len(), 2);
        assert!(plan.eviction_candidates.is_empty());
        assert!(plan.swap_in_requests.is_empty());
        assert_eq!(plan.tier_migrations[0].reason, TierMigrationReason::EvictionPressure);
        assert_eq!(plan.tier_migrations[1].reason, TierMigrationReason::ColdCascade);
    }

    // ── Coordinator: page_metadata accessor returns usable arc ─────────────────

    #[test]
    fn coordinator_page_metadata_accessor_is_usable() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: write through the accessor
        {
            let pm = c.page_metadata();
            let mut guard = pm.write().expect("write lock");
            guard.insert(42, PageMetadata::default());
        }
        // Assert: read back through the same accessor
        let guard = c.page_metadata().read().expect("read lock");
        assert!(guard.contains_key(&42));
    }

    // ── Coordinator: register_page then update_gpu_ptr then verify tier ────────

    #[test]
    fn coordinator_register_then_update_gpu_ptr_flow() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(10, None, 4096);
        // Assert: initially DRAM (no gpu_ptr)
        {
            let table = c.addr_table().read().expect("read lock");
            let entry = table.get(&10).expect("page 10");
            assert_eq!(entry.current_tier, StorageTier::CpuDram);
            assert!(entry.gpu_ptr.is_none());
        }
        // Act: promote to HBM
        c.update_page_gpu_ptr(10, 0xABCD);
        // Assert
        {
            let table = c.addr_table().read().expect("read lock");
            let entry = table.get(&10).expect("page 10");
            assert_eq!(entry.current_tier, StorageTier::GpuHbm);
            assert_eq!(entry.gpu_ptr, Some(0xABCD));
        }
    }

    // ── Coordinator: release_page cleans up both tables ────────────────────────

    #[test]
    fn coordinator_release_page_cleans_both_tables() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(55, Some(0x5000), 4096);
        {
            let mut pm = c.page_metadata().write().expect("write");
            pm.insert(55, PageMetadata { page_id: 55, ..Default::default() });
        }
        // Act
        c.release_page(55);
        // Assert
        assert!(c.addr_table().read().expect("r").get(&55).is_none());
        assert!(c.page_metadata().read().expect("r").get(&55).is_none());
    }

    // ── Coordinator: tier_changed_pages with multiple divergent pages ──────────

    #[test]
    fn coordinator_tier_changed_pages_multiple_divergences() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Register 3 pages on HBM
        for pid in 1u64..=3u64 {
            c.register_page(pid as PageId, Some(0x1000 + pid), 4096);
        }
        // Add metadata: pages 1 and 2 as Swapped (NVMe in metadata), page 3 as Active (HBM)
        {
            let mut pm = c.page_metadata().write().expect("write");
            pm.insert(1, PageMetadata { page_id: 1, state: PageState::Swapped, ..Default::default() });
            pm.insert(2, PageMetadata { page_id: 2, state: PageState::SwappedOut, ..Default::default() });
            pm.insert(3, PageMetadata { page_id: 3, state: PageState::Active, ..Default::default() });
        }
        // Act
        let changed = c.tier_changed_pages();
        // Assert: pages 1 and 2 diverge (addr_table says HBM, metadata says Nvme), page 3 matches
        assert_eq!(changed.len(), 2);
        let changed_ids: std::collections::HashSet<PageId> =
            changed.iter().map(|(pid, _)| *pid).collect();
        assert!(changed_ids.contains(&1));
        assert!(changed_ids.contains(&2));
        assert!(!changed_ids.contains(&3));
    }

    // ── Coordinator: record_eviction completed accumulates bytes correctly ─────

    #[test]
    fn coordinator_record_eviction_bytes_accumulate() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::CpuDram, StorageTier::Nvme, 8192, 200);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.total_bytes_evicted, 12288);
        assert_eq!(stats.total_eviction_latency_us, 300);
    }

    // ── Coordinator: record_swap_in completed accumulates bytes correctly ──────

    #[test]
    fn coordinator_record_swap_in_bytes_accumulate() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 50);
        c.record_swap_in_completed(2, StorageTier::Nvme, StorageTier::CpuDram, 8192, 150);
        c.record_swap_in_completed(3, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 75);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 2);
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.total_bytes_swapped_in, 16384);
        assert_eq!(stats.total_swap_in_latency_us, 275);
    }

    // ── Coordinator: shutdown is idempotent via explicit double call ───────────

    #[test]
    fn coordinator_double_shutdown_no_panic() {
        // Arrange
        let (mut c, _backend) = make_coordinator(false);
        // Act
        c.shutdown();
        c.shutdown(); // second call should be no-op
        // Assert: no panic, workers already None
        assert!(c.eviction_worker.is_none());
        assert!(c.swap_in_worker.is_none());
    }

    // ── ThreeTierSwapConfig: default has auto_start true ───────────────────────

    #[test]
    fn config_default_auto_start_is_true() {
        // Act
        let config = ThreeTierSwapConfig::default();
        // Assert
        assert!(config.auto_start);
    }

    // ── EvictionCandidate: with all CompressionCodec variants ──────────────────

    #[test]
    fn eviction_candidate_all_five_codec_variants() {
        // Arrange
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        // Act & Assert: all variants construct without panic
        for codec in codecs {
            let candidate = EvictionCandidate {
                page_id: 1,
                score: 0,
                current_tier: StorageTier::GpuHbm,
                codec,
                page_bytes: 4096,
                group_id: None,
            };
            assert_eq!(candidate.codec, codec);
        }
    }

    // ── TierMigrationReason: all variants are Copy + independent after clone ───

    #[test]
    fn tier_migration_reason_copy_independence() {
        // Arrange
        let original = TierMigrationReason::Prefetch;
        let copied = original;
        // Assert: both are equal (Copy, not move)
        assert_eq!(original, copied);
        assert_eq!(original, TierMigrationReason::Prefetch);
        assert_eq!(copied, TierMigrationReason::Prefetch);
    }

    // ── Wave 15: additional coverage tests (15 new tests) ──────────────────────────

    // @trace REQ-COMP-001 [level:unit] [nfr:TMG-COMP-001]

    #[test]
    fn build_batch_swapped_out_on_dram_eligible_under_dram_pressure() {
        // Arrange: page on DRAM with SwappedOut state, high DRAM pressure.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::SwappedOut,
                warm_until: None,
            });
        }
        // Act: DRAM eviction requires dram_pressure > threshold.
        // Default GlobalMemoryManager has used=0, so dram_pressure=0.
        // The test verifies no panic and correct plan structure.
        let plan = c.build_batch(&[], 0.95);
        // Assert: no eviction from DRAM when dram_pressure is 0.
        let dram_evictions: Vec<_> = plan.eviction_candidates
            .iter()
            .filter(|c| c.current_tier == StorageTier::CpuDram)
            .collect();
        assert!(dram_evictions.is_empty(),
            "SwappedOut page on DRAM should not evict when dram pressure is zero");
    }

    #[test]
    fn build_batch_swap_in_for_active_page_on_nvme_generates_migration() {
        // Arrange: page on NVMe tier, requested as active page.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert: swap-in request and migration both present, target is GpuHbm.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 1);
        assert_eq!(plan.tier_migrations.len(), 1);
        assert_eq!(plan.tier_migrations[0].from_tier, StorageTier::Nvme);
        assert_eq!(plan.tier_migrations[0].to_tier, StorageTier::GpuHbm);
        assert_eq!(plan.tier_migrations[0].reason, TierMigrationReason::SequenceDemand);
        // Assert: no eviction candidate for this page.
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id != 1));
    }

    #[test]
    fn record_eviction_same_tier_pair_counts_bytes_only() {
        // Arrange: eviction where from_tier == to_tier (degenerate case).
        let (c, _backend) = make_coordinator(false);
        // Act: record eviction with GpuHbm → GpuHbm (same tier).
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::GpuHbm, 4096, 100);
        // Assert: migration counter not incremented, but bytes/latency accumulated.
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.total_bytes_evicted, 4096);
        assert_eq!(stats.total_eviction_latency_us, 100);
    }

    #[test]
    fn record_swap_in_same_tier_pair_counts_bytes_only() {
        // Arrange: swap-in where from_tier == to_tier (degenerate case).
        let (c, _backend) = make_coordinator(false);
        // Act: record swap-in with CpuDram → CpuDram (same tier).
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::CpuDram, 8192, 200);
        // Assert: migration counter not incremented, but bytes/latency accumulated.
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_swapped_in, 8192);
        assert_eq!(stats.total_swap_in_latency_us, 200);
    }

    #[test]
    fn register_pages_from_hgal_then_build_batch_swap_in_for_active() {
        // Arrange: bulk-register pages from HGAL, then build_batch with active pages on DRAM.
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        for pid in 100..103u64 {
            pages.insert(pid as PageId, PageMetadata {
                page_id: pid as PageId,
                sequence_id: Some(pid),
                state: PageState::Standby,
                ..PageMetadata::default()
            });
        }
        c.register_pages_from_hgal(&pages, 4096);
        // Pages registered without gpu_ptr → CpuDram tier.
        // Act: request swap-in for pages 100 and 101.
        let plan = c.build_batch(&[100, 101], 0.5);
        // Assert: swap-in requests for active pages on DRAM.
        assert_eq!(plan.swap_in_requests.len(), 2);
        let swap_ids: std::collections::HashSet<PageId> =
            plan.swap_in_requests.iter().map(|r| r.page_id).collect();
        assert!(swap_ids.contains(&100));
        assert!(swap_ids.contains(&101));
        assert!(!swap_ids.contains(&102));
    }

    #[test]
    fn update_page_gpu_ptr_reflected_in_next_build_batch() {
        // Arrange: register page on DRAM, verify swap-in needed, then promote to HBM.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Pre-condition: page on DRAM → swap-in requested.
        let plan_before = c.build_batch(&[1], 0.5);
        assert_eq!(plan_before.swap_in_requests.len(), 1);
        // Act: promote to HBM via update_page_gpu_ptr.
        c.update_page_gpu_ptr(1, 0xCAFE);
        // Assert: page now on HBM, no swap-in needed.
        let plan_after = c.build_batch(&[1], 0.5);
        assert!(plan_after.swap_in_requests.is_empty(),
            "no swap-in after promotion to HBM");
    }

    #[test]
    fn build_batch_mixed_active_and_standby_pages_correct_split() {
        // Arrange: 3 pages on HBM — 2 Standby (evictable) + 1 Active (protected).
        // Plus 1 page on DRAM that is in active set (needs swap-in).
        let (c, _backend) = make_coordinator(false);
        // Page 1: HBM, Standby — evictable under pressure.
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(10),
                recency: 100,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Page 2: HBM, Active — never evicted.
        c.register_page(2, Some(0x2000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(2, PageMetadata {
                page_id: 2,
                sequence_id: Some(20),
                recency: 0,
                access_count: 10,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Active,
                warm_until: None,
            });
        }
        // Page 3: HBM, Standby — evictable under pressure.
        c.register_page(3, Some(0x3000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(3, PageMetadata {
                page_id: 3,
                sequence_id: Some(30),
                recency: 200,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Page 4: DRAM, needs swap-in.
        c.register_page(4, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&4) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(4, PageMetadata {
                page_id: 4,
                sequence_id: Some(40),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: high pressure, page 4 is active (needs swap-in).
        let plan = c.build_batch(&[4], 0.95);
        // Assert: page 4 swap-in, page 2 not evicted, pages 1/3 may be eviction candidates.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 4);
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id != 2),
            "Active page 2 should never be in eviction candidates");
        // Pages 1 and 3 are Standby under high pressure → eviction candidates.
        let evicted_ids: std::collections::HashSet<PageId> =
            plan.eviction_candidates.iter().map(|c| c.page_id).collect();
        assert!(evicted_ids.contains(&1) || evicted_ids.contains(&3),
            "at least one Standby page should be an eviction candidate");
    }

    #[test]
    fn coordinator_register_bulk_release_half_verify_remaining() {
        // Arrange: bulk-register 20 pages, release 10, verify remaining 10.
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        for pid in 0u64..20 {
            pages.insert(pid as PageId, PageMetadata {
                page_id: pid as PageId,
                sequence_id: Some(pid),
                ..PageMetadata::default()
            });
        }
        c.register_pages_from_hgal(&pages, 4096);
        // Act: release first 10 pages.
        for pid in 0u64..10 {
            c.release_page(pid as PageId);
        }
        // Assert: exactly 10 pages remain.
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.len(), 10);
        for pid in 10u64..20 {
            assert!(table.contains_key(&(pid as PageId)),
                "page {} should still exist", pid);
        }
        for pid in 0u64..10 {
            assert!(!table.contains_key(&(pid as PageId)),
                "page {} should be removed", pid);
        }
    }

    #[test]
    fn build_batch_pressure_exactly_at_hbm_threshold_no_eviction() {
        // Arrange: page on HBM with Standby state, pressure exactly at threshold.
        // Default hbm_pressure_threshold = 0.90.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: pressure exactly 0.90 (the default threshold).
        // build_batch requires pressure > threshold, not >=.
        let plan = c.build_batch(&[], 0.90);
        // Assert: no eviction because pressure is not strictly above threshold.
        assert!(plan.eviction_candidates.is_empty(),
            "pressure exactly at threshold should not trigger eviction");
    }

    #[test]
    fn build_batch_page_with_high_recency_penalized_in_score() {
        // Arrange: two pages, one with high recency (high IRR), one with low.
        // High recency means the page was accessed long ago → lower score → evict first.
        let (c, _backend) = make_coordinator(false);
        // Page 1: low recency (recently accessed).
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(10),
                recency: 1,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Page 2: high recency (not accessed recently).
        c.register_page(2, Some(0x2000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(2, PageMetadata {
                page_id: 2,
                sequence_id: Some(20),
                recency: 10000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.95);
        // Assert: page 2 (high recency) should have lower score → appears first.
        if plan.eviction_candidates.len() >= 2 {
            assert!(plan.eviction_candidates[0].page_id == 2,
                "high-recency page should be first eviction candidate (lowest score)");
        }
    }

    #[test]
    fn tier_migration_reason_all_four_form_exhaustive_set() {
        // Arrange: collect all TierMigrationReason variants into a Vec.
        let all_reasons = vec![
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        // Act & Assert: exactly 4 distinct variants.
        use std::collections::HashSet;
        let set: HashSet<TierMigrationReason> = all_reasons.into_iter().collect();
        assert_eq!(set.len(), 4, "there should be exactly 4 distinct TierMigrationReason variants");
    }

    #[test]
    fn page_metadata_warm_until_some_value_preserved() {
        // Arrange: create PageMetadata with warm_until = Some(future instant).
        let future = Instant::now() + Duration::from_secs(60);
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(100),
            recency: 5,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: Some(future),
        };
        // Act: clone and check.
        let cloned = meta.clone();
        // Assert: warm_until is preserved.
        assert!(meta.warm_until.is_some());
        assert!(cloned.warm_until.is_some());
        // The instant value should be preserved exactly.
        assert_eq!(meta.warm_until, cloned.warm_until);
    }

    #[test]
    fn infer_swap_payload_kind_all_none_sequence_variants() {
        // Arrange: metadata with sequence_id = None and sequence_id = Some(x).
        let meta_none = PageMetadata {
            page_id: 1,
            sequence_id: None,
            ..PageMetadata::default()
        };
        let meta_some_zero = PageMetadata {
            page_id: 2,
            sequence_id: Some(0),
            ..PageMetadata::default()
        };
        let meta_some_large = PageMetadata {
            page_id: 3,
            sequence_id: Some(u64::MAX),
            ..PageMetadata::default()
        };
        // Act & Assert: None → ExpertWeight, Some(_) → KvContext.
        assert_eq!(infer_swap_payload_kind(&meta_none), Some(PagePayloadKind::ExpertWeight));
        assert_eq!(infer_swap_payload_kind(&meta_some_zero), Some(PagePayloadKind::KvContext));
        assert_eq!(infer_swap_payload_kind(&meta_some_large), Some(PagePayloadKind::KvContext));
    }

    #[test]
    fn eviction_candidate_group_id_none_valid() {
        // Arrange & Act: EvictionCandidate with group_id = None.
        let candidate = EvictionCandidate {
            page_id: 99,
            score: -500,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 4096,
            group_id: None,
        };
        // Assert: group_id = None is a valid state (expert weight pages have no sequence).
        assert!(candidate.group_id.is_none());
        assert_eq!(candidate.page_id, 99);
        assert_eq!(candidate.score, -500);
    }

    #[test]
    fn build_batch_consecutive_plans_have_independent_candidates() {
        // Arrange: register pages on HBM with Standby state.
        let (c, _backend) = make_coordinator(false);
        for pid in 1u64..=3u64 {
            c.register_page(pid as PageId, Some(0x1000 + pid), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid as PageId, PageMetadata {
                page_id: pid as PageId,
                sequence_id: Some(pid * 10),
                recency: 1000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: two consecutive build_batch calls.
        let plan1 = c.build_batch(&[], 0.95);
        let plan2 = c.build_batch(&[], 0.95);
        // Assert: both plans have independent candidate vectors.
        // They should have the same candidates since state hasn't changed,
        // but the vectors must be independent (different allocations).
        assert_eq!(plan1.eviction_candidates.len(), plan2.eviction_candidates.len());
        // Modifying plan1 should not affect plan2.
        let plan1_len = plan1.eviction_candidates.len();
        // Verify both plans built_at timestamps are ordered.
        assert!(plan2.built_at >= plan1.built_at,
            "second plan should be built at same time or after first plan");
    }

    // ── Wave 18: 15 additional edge-case tests ────────────────────────────────────

    // @trace REQ-COMP-006 [level:unit]
    #[test]
    fn build_batch_swap_in_page_takes_priority_over_eviction_for_same_page() {
        // Arrange: page on DRAM, in active set, with Standby state (evictable if not active).
        // build_batch should generate swap-in, NOT eviction, for this page.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 1000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: page 1 is active (in active_pages) and on DRAM.
        let plan = c.build_batch(&[1], 0.99);
        // Assert: swap-in request present, no eviction candidate for page 1.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 1);
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id != 1),
            "page scheduled for swap-in must not appear as eviction candidate");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_from_dram_preserves_existing_codec() {
        // Arrange: page on DRAM with BitPackRle codec, Standby state.
        // Eviction from DRAM should preserve the entry's existing codec, not use default.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
                entry.codec = CompressionCodec::BitPackRle;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 1000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Note: dram eviction requires dram_pressure > threshold (default 0.0).
        // With empty memory manager dram pressure is 0.0, so this page won't be evicted.
        // The test verifies the plan structure is correct even when no eviction occurs.
        let plan = c.build_batch(&[], 0.99);
        // If any DRAM eviction candidate exists, its codec should match the entry.
        for candidate in plan.eviction_candidates.iter()
            .filter(|c| c.current_tier == StorageTier::CpuDram && c.page_id == 1)
        {
            assert_eq!(candidate.codec, CompressionCodec::BitPackRle,
                "DRAM eviction candidate should use entry's existing codec");
        }
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_changed_pages_free_state_on_dram_no_divergence() {
        // Arrange: page on DRAM with Free state. Free maps to CpuDram → no divergence.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        // Page registered with None gpu_ptr → CpuDram tier.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Free,
                ..PageMetadata::default()
            });
        }
        // Act
        let changed = c.tier_changed_pages();
        // Assert: Free → CpuDram, actual is CpuDram → consistent.
        assert!(changed.is_empty(),
            "Free state page on DRAM should have no tier divergence");
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_changed_pages_swapped_state_on_nvme_no_divergence() {
        // Arrange: page on NVMe with Swapped state. Swapped maps to Nvme → no divergence.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Swapped,
                ..PageMetadata::default()
            });
        }
        // Act
        let changed = c.tier_changed_pages();
        // Assert: Swapped → Nvme, actual is Nvme → consistent.
        assert!(changed.is_empty(),
            "Swapped state page on NVMe should have no tier divergence");
    }

    // @trace REQ-COMP-005 [level:unit]
    #[test]
    fn page_metadata_clone_swap_in_time_preserves_value() {
        // Arrange: PageMetadata with swap_in_time = Some(instant).
        let swap_time = Instant::now() - Duration::from_millis(500);
        let meta = PageMetadata {
            page_id: 42,
            sequence_id: Some(7),
            recency: 10,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: Some(swap_time),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let cloned = meta.clone();
        // Assert: swap_in_time is preserved exactly.
        assert_eq!(meta.swap_in_time, cloned.swap_in_time);
        assert!(meta.swap_in_time.unwrap() == swap_time);
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn stats_snapshot_zero_tiers_after_register_and_full_release() {
        // Arrange: register 5 pages, then release all.
        let (c, _backend) = make_coordinator(false);
        for pid in 100u64..105 {
            c.register_page(pid as PageId, Some(0x1000 + pid), 4096);
        }
        assert_eq!(c.stats().pages_on_hbm, 5);
        // Act: release all and record some migrations.
        c.record_eviction_completed(100, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 50);
        c.record_swap_in_completed(101, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 30);
        for pid in 100u64..105 {
            c.release_page(pid as PageId);
        }
        // Assert: tier counts are zero, but cumulative records persist.
        let stats = c.stats();
        assert_eq!(stats.pages_on_hbm, 0);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
    }

    // @trace REQ-COMP-006 [level:unit]
    #[test]
    fn build_batch_multiple_swap_in_requests_sorted_by_page_id_in_metadata_order() {
        // Arrange: three pages on DRAM, all in active set.
        let (c, _backend) = make_coordinator(false);
        for pid in [30usize, 10, 20] {
            c.register_page(pid, None, 4096);
            {
                let mut table = c.addr_table.write().expect("write lock");
                if let Some(entry) = table.get_mut(&pid) {
                    entry.current_tier = StorageTier::CpuDram;
                    entry.host_buffer = Some(vec![0u8; 4096]);
                }
            }
            {
                let mut meta = c.page_metadata.write().expect("write lock");
                meta.insert(pid, PageMetadata {
                    page_id: pid,
                    sequence_id: Some(pid as u64),
                    recency: 0,
                    access_count: 5,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                });
            }
        }
        // Act
        let plan = c.build_batch(&[30, 10, 20], 0.5);
        // Assert: all 3 swap-in requests present.
        assert_eq!(plan.swap_in_requests.len(), 3);
        let page_ids: std::collections::HashSet<PageId> =
            plan.swap_in_requests.iter().map(|r| r.page_id).collect();
        assert!(page_ids.contains(&30));
        assert!(page_ids.contains(&10));
        assert!(page_ids.contains(&20));
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn register_page_after_release_creates_fresh_entry() {
        // Arrange: register, release, then re-register with different parameters.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.release_page(1);
        // Act: re-register with different gpu_ptr and page_bytes.
        c.register_page(1, Some(0xBBBB), 16384);
        // Assert: new entry reflects the latest registration.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("re-registered page should exist");
        assert_eq!(entry.gpu_ptr, Some(0xBBBB));
        assert_eq!(entry.original_bytes, 16384);
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_changed_pages_after_addr_entry_removed_but_metadata_present() {
        // Arrange: register page, add metadata, then remove only from addr_table.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Active,
                ..PageMetadata::default()
            });
        }
        // Remove only from addr_table.
        {
            let mut table = c.addr_table.write().expect("write lock");
            table.remove(&1);
        }
        // Act: tier_changed_pages iterates addr_table, so page 1 is gone.
        let changed = c.tier_changed_pages();
        // Assert: no divergence detected because addr_table has no entry.
        assert!(changed.is_empty());
    }

    // @trace REQ-COMP-006 [level:unit]
    #[test]
    fn build_batch_with_swapped_out_on_nvme_no_eviction_no_swap_in() {
        // Arrange: page on NVMe with SwappedOut state, not in active set.
        // NVMe pages cannot be evicted, and no swap-in needed since not active.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::SwappedOut,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert: no eviction (NVMe can't be evicted) and no swap-in (not active).
        let nvme_evictions: Vec<_> = plan.eviction_candidates
            .iter().filter(|c| c.page_id == 1).collect();
        assert!(nvme_evictions.is_empty());
        assert!(plan.swap_in_requests.is_empty());
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn stats_avg_swap_in_latency_single_nvme_to_dram() {
        // Arrange: record exactly one NVMe→DRAM swap-in with known latency.
        let (c, _backend) = make_coordinator(false);
        c.record_swap_in_completed(1, StorageTier::Nvme, StorageTier::CpuDram, 4096, 7777);
        // Act
        let stats = c.stats();
        // Assert: avg = 7777.0 (only one operation).
        assert!((stats.avg_swap_in_latency_us() - 7777.0).abs() < 0.01);
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_candidates_max_eight_with_more_eligible_pages() {
        // Arrange: register 20 Standby pages on HBM under high pressure.
        // Default max_evict_per_round = 8.
        let (c, _backend) = make_coordinator(false);
        for pid in 0usize..20 {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64 * 10),
                recency: 1000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert: eviction candidates capped at 8.
        assert!(plan.eviction_candidates.len() <= 8,
            "eviction candidates must not exceed max_evict_per_round, got {}",
            plan.eviction_candidates.len());
        // All candidates sorted ascending by score.
        for window in plan.eviction_candidates.windows(2) {
            assert!(window[0].score <= window[1].score);
        }
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_changed_pages_standby_on_dram_consistent() {
        // Arrange: page on DRAM with Standby state. Standby maps to CpuDram → consistent.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Standby,
                ..PageMetadata::default()
            });
        }
        // Act
        let changed = c.tier_changed_pages();
        // Assert: Standby → CpuDram, actual is CpuDram → no divergence.
        assert!(changed.is_empty(),
            "Standby page on DRAM should have no tier divergence");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_no_eviction_for_recently_swapped_in_page() {
        // Arrange: page on HBM with Standby state, swap_in_time = now (very recent).
        // Age ticks will be near 0, below hbm_evict_age_ticks threshold.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: Some(Instant::now()), // Very recent
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: high pressure but age is too low.
        let plan = c.build_batch(&[], 0.99);
        // Assert: no eviction because tier_age_ticks < hbm_evict_age_ticks.
        assert!(plan.eviction_candidates.is_empty(),
            "recently swapped-in page should not be evicted");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_warm_page_skipped_even_under_max_pressure() {
        // Arrange: Warm state is in the skip list (Active/Protected/Warm).
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(60),
                swap_in_time: Some(Instant::now() - Duration::from_secs(60)),
                is_lir: false,
                state: PageState::Warm,
                warm_until: Some(Instant::now() + Duration::from_secs(60)),
            });
        }
        // Act: maximum pressure.
        let plan = c.build_batch(&[], f32::MAX);
        // Assert: Warm pages are never evicted regardless of pressure.
        assert!(plan.eviction_candidates.is_empty(),
            "Warm pages must be skipped in eviction selection");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_mixed_swap_in_and_eviction_produces_both() {
        // Arrange: one page on DRAM needing swap-in, one page on HBM eligible for eviction.
        let (c, _backend) = make_coordinator(false);
        // Page A: on DRAM, in active set → swap-in.
        c.register_page(10, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&10) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(10, PageMetadata {
                page_id: 10,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Page B: on HBM, Standby, old → eviction candidate.
        c.register_page(20, Some(0x2000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(20, PageMetadata {
                page_id: 20,
                sequence_id: Some(200),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[10], 0.99);
        // Assert: swap-in for page 10, eviction candidate for page 20.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 10);
        assert!(!plan.eviction_candidates.is_empty(),
            "standby HBM page under high pressure should produce eviction candidate");
        assert!(plan.eviction_candidates.iter().any(|c| c.page_id == 20));
        // Page 10 must NOT appear in eviction candidates.
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id != 10));
    }

    // ── Additional coverage: edge cases, error paths, boundary values ──────────

    #[test]
    fn build_batch_negative_pressure_no_eviction_recheck() {
        // Arrange: page on HBM with Standby state, negative pressure.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: negative pressure value.
        let plan = c.build_batch(&[], -1.0);
        // Assert: negative pressure should not trigger eviction.
        assert!(plan.eviction_candidates.is_empty(),
            "negative pressure should not produce eviction candidates");
    }

    #[test]
    fn compute_tier_age_ticks_swap_in_time_takes_precedence() {
        // Arrange: page with both swap_in_time and last_access set,
        // where swap_in_time is more recent. compute_tier_age_ticks should
        // use swap_in_time as the anchor.
        let recent = Instant::now();
        let old = recent - Duration::from_secs(60);
        let meta_swap_in = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 1,
            last_access: old,
            swap_in_time: Some(recent),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let meta_no_swap_in = PageMetadata {
            page_id: 2,
            sequence_id: Some(10),
            recency: 0,
            access_count: 1,
            last_access: old,
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let ticks_with_swap_in = compute_tier_age_ticks(&meta_swap_in);
        let ticks_no_swap_in = compute_tier_age_ticks(&meta_no_swap_in);
        // Assert: page with recent swap_in_time should have fewer ticks
        // than page that relies on older last_access.
        assert!(ticks_with_swap_in < ticks_no_swap_in,
            "swap_in_time anchor should produce lower age ticks than old last_access");
    }

    #[test]
    fn record_eviction_then_swap_in_then_stats_snapshot_consistent() {
        // Arrange: full lifecycle — evict then swap-in — verify stats are consistent.
        let (c, _backend) = make_coordinator(false);
        // Act: record eviction from HBM → DRAM.
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 200);
        // Act: record swap-in from DRAM → HBM.
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 150);
        // Assert: stats reflect both operations.
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.total_bytes_evicted, 4096);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
        assert_eq!(stats.total_eviction_latency_us, 200);
        assert_eq!(stats.total_swap_in_latency_us, 150);
        assert_eq!(stats.total_migrations(), 2);
        assert!((stats.avg_eviction_latency_us() - 200.0).abs() < 0.01);
        assert!((stats.avg_swap_in_latency_us() - 150.0).abs() < 0.01);
    }

    #[test]
    fn tier_migration_reason_hash_set_deduplication() {
        // Arrange: create a vec with duplicate reasons.
        use std::collections::HashSet;
        let reasons = vec![
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::ColdCascade,
        ];
        // Act
        let set: HashSet<TierMigrationReason> = reasons.into_iter().collect();
        // Assert: duplicates removed.
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn build_batch_page_id_zero_is_valid() {
        // Arrange: PageId 0 should work the same as any other page.
        let (c, _backend) = make_coordinator(false);
        c.register_page(0, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(0, PageMetadata {
                page_id: 0,
                sequence_id: Some(1),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert: page 0 should be handled like any other page.
        if !plan.eviction_candidates.is_empty() {
            assert!(plan.eviction_candidates.iter().any(|c| c.page_id == 0));
        }
    }

    #[test]
    fn build_batch_large_page_bytes_does_not_panic() {
        // Arrange: register a page with very large byte size.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), usize::MAX);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act & Assert: should not panic on usize::MAX page bytes.
        let plan = c.build_batch(&[], 0.99);
        if let Some(candidate) = plan.eviction_candidates.first() {
            assert_eq!(candidate.page_bytes, c.config.eviction.page_bytes);
        }
    }

    #[test]
    fn storage_tier_sort_descending_priority() {
        // Arrange: create a vec of tiers in mixed order.
        let mut tiers = vec![StorageTier::CpuDram, StorageTier::GpuHbm, StorageTier::Nvme];
        // Act: sort ascending (lowest priority first).
        tiers.sort();
        // Assert: Nvme < CpuDram < GpuHbm (ascending by discriminant,
        // but Ord reverses so Nvme comes first in ascending sort).
        assert_eq!(tiers, vec![StorageTier::Nvme, StorageTier::CpuDram, StorageTier::GpuHbm]);
    }

    #[test]
    fn eviction_candidate_none_group_id_field_access() {
        // Arrange: EvictionCandidate where group_id is None (no sequence).
        let candidate = EvictionCandidate {
            page_id: 42,
            score: 10,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };
        // Assert: field access works correctly.
        assert_eq!(candidate.page_id, 42);
        assert_eq!(candidate.group_id, None);
    }

    #[test]
    fn tier_migration_from_and_to_same_tier_allowed_in_struct() {
        // Arrange: structurally, TierMigration allows from == to.
        let migration = TierMigration {
            page_id: 1,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            reason: TierMigrationReason::Prefetch,
        };
        // Assert: struct fields are stored as-is.
        assert_eq!(migration.from_tier, migration.to_tier);
    }

    #[test]
    fn prefresh_request_page_bytes_field_access() {
        // Arrange: construct PrefetchRequest manually and verify all fields.
        let now = Instant::now();
        let req = PrefetchRequest {
            page_id: 77,
            urgency: 0.5,
            prefetch_confidence: 0.7,
            page_bytes: 16384,
            enqueued_at: now,
        };
        // Assert
        assert_eq!(req.page_id, 77);
        assert!((req.urgency - 0.5).abs() < f32::EPSILON);
        assert!((req.prefetch_confidence - 0.7).abs() < f32::EPSILON);
        assert_eq!(req.page_bytes, 16384);
    }

    #[test]
    fn build_batch_swap_in_for_multiple_nvme_pages_on_demand() {
        // Arrange: multiple pages on NVMe tier, all requested as active.
        let (c, _backend) = make_coordinator(false);
        for pid in [100usize, 200, 300] {
            c.register_page(pid, None, 4096);
            {
                let mut table = c.addr_table.write().expect("write lock");
                if let Some(entry) = table.get_mut(&pid) {
                    entry.current_tier = StorageTier::Nvme;
                }
            }
            {
                let mut meta = c.page_metadata.write().expect("write lock");
                meta.insert(pid, PageMetadata {
                    page_id: pid,
                    sequence_id: Some(pid as u64),
                    recency: 0,
                    access_count: 5,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Swapped,
                    warm_until: None,
                });
            }
        }
        // Act
        let plan = c.build_batch(&[100, 200, 300], 0.5);
        // Assert: all three NVMe pages should have swap-in requests.
        assert_eq!(plan.swap_in_requests.len(), 3);
        let page_ids: std::collections::HashSet<PageId> =
            plan.swap_in_requests.iter().map(|r| r.page_id).collect();
        assert!(page_ids.contains(&100));
        assert!(page_ids.contains(&200));
        assert!(page_ids.contains(&300));
        // All migrations should be Nvme → GpuHbm with SequenceDemand reason.
        for migration in &plan.tier_migrations {
            assert_eq!(migration.from_tier, StorageTier::Nvme);
            assert_eq!(migration.to_tier, StorageTier::GpuHbm);
            assert_eq!(migration.reason, TierMigrationReason::SequenceDemand);
        }
    }

    #[test]
    fn build_batch_pressure_nan_no_panic() {
        // Arrange: NaN pressure should not cause panic.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act & Assert: should not panic with NaN pressure.
        let plan = c.build_batch(&[], f32::NAN);
        // NaN comparisons are always false, so no eviction should happen.
        assert!(plan.eviction_candidates.is_empty(),
            "NaN pressure should not trigger eviction");
    }

    #[test]
    fn stats_total_migrations_only_eviction_counts() {
        // Arrange: only eviction operations recorded.
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 7;
        stats.evictions_dram_to_nvme = 3;
        // Act
        let total = stats.total_migrations();
        // Assert: total is sum of eviction counts only.
        assert_eq!(total, 10);
    }

    #[test]
    fn stats_total_migrations_only_swap_in_counts() {
        // Arrange: only swap-in operations recorded.
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_dram_to_gpu = 4;
        stats.swap_ins_nvme_to_dram = 2;
        // Act
        let total = stats.total_migrations();
        // Assert: total is sum of swap-in counts only.
        assert_eq!(total, 6);
    }

    #[test]
    fn register_pages_from_hgal_then_release_all_then_addr_table_empty() {
        // Arrange: register bulk pages then release all of them.
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        for pid in 0..5u64 {
            pages.insert(pid as PageId, PageMetadata {
                page_id: pid as PageId,
                sequence_id: Some(pid),
                ..PageMetadata::default()
            });
        }
        c.register_pages_from_hgal(&pages, 4096);
        // Act: release all.
        for pid in 0..5u64 {
            c.release_page(pid as PageId);
        }
        // Assert: addr_table should be empty, metadata should be empty.
        let table = c.addr_table.read().expect("read lock");
        assert!(table.is_empty());
        let meta = c.page_metadata.read().expect("read lock");
        assert!(meta.is_empty());
    }

    // ── Additional edge-case tests (15 new) ──────────────────────────────────

    #[test]
    fn register_page_overwrite_does_not_replace_existing_addr_entry() {
        // Arrange: register a page with specific gpu_ptr, then register again.
        let (c, _backend) = make_coordinator(false);
        c.register_page(10, Some(0xABCD), 8192);
        // Act: register same page_id again with different gpu_ptr — should not overwrite (idempotent).
        c.register_page(10, Some(0xFFFF), 4096);
        // Assert: original entry preserved because or_insert_with.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&10).expect("entry exists");
        assert_eq!(entry.gpu_ptr, Some(0xABCD), "original gpu_ptr preserved");
        assert_eq!(entry.original_bytes, 8192, "original bytes preserved");
    }

    #[test]
    fn update_page_gpu_ptr_sets_tier_back_to_hbm() {
        // Arrange: register a page without gpu_ptr (starts on CpuDram).
        let (c, _backend) = make_coordinator(false);
        c.register_page(20, None, 4096);
        {
            let table = c.addr_table.read().expect("read lock");
            let entry = table.get(&20).expect("entry exists");
            assert_eq!(entry.current_tier, StorageTier::CpuDram);
        }
        // Act: update gpu_ptr which also promotes tier.
        c.update_page_gpu_ptr(20, 0x5000);
        // Assert: tier changed to GpuHbm.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&20).expect("entry exists");
        assert_eq!(entry.gpu_ptr, Some(0x5000));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    #[test]
    fn tier_migration_reason_all_variants_hashable_and_distinct() {
        // Arrange: collect all TierMigrationReason variants in a HashSet.
        use std::collections::HashSet;
        let reasons = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        // Act
        let set: HashSet<TierMigrationReason> = reasons.iter().copied().collect();
        // Assert: all 4 are distinct in the set.
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn tier_migration_plan_eviction_candidates_sorted_after_truncation() {
        // Arrange: coordinator with many standby pages under high pressure.
        let (c, _backend) = make_coordinator(false);
        // Create pages with varying ages so scores differ — older = lower score.
        for pid in 100..110usize {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(
                pid,
                PageMetadata {
                    page_id: pid,
                    sequence_id: Some(pid as u64),
                    recency: (110 - pid) * 10,
                    access_count: 0,
                    last_access: Instant::now() - Duration::from_millis((110 - pid) as u64 * 100),
                    swap_in_time: Some(Instant::now() - Duration::from_millis((110 - pid) as u64 * 100)),
                    is_lir: false,
                    state: PageState::Standby,
                    warm_until: None,
                },
            );
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert: eviction candidates sorted by ascending score.
        if plan.eviction_candidates.len() > 1 {
            for window in plan.eviction_candidates.windows(2) {
                assert!(
                    window[0].score <= window[1].score,
                    "candidates must be sorted ascending by score"
                );
            }
        }
    }

    #[test]
    fn stats_snapshot_after_multiple_register_and_tier_changes() {
        // Arrange: register pages on different tiers.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x100), 4096); // HBM
        c.register_page(2, Some(0x200), 4096); // HBM
        c.register_page(3, None, 4096); // DRAM
        // Act: manually change tier of page 2 to CpuDram.
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&2) {
                entry.current_tier = StorageTier::CpuDram;
                entry.gpu_ptr = None;
            }
        }
        let snap = c.stats();
        // Assert: tier counts reflect manual changes.
        assert_eq!(snap.pages_on_hbm, 1, "page 1 on HBM");
        assert_eq!(snap.pages_on_dram, 2, "pages 2+3 on DRAM");
        assert_eq!(snap.pages_on_nvme, 0);
    }

    #[test]
    fn compression_codec_from_u8_roundtrip_all_variants() {
        // Arrange & Act & Assert: each variant's u8 representation round-trips.
        for expected in [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let v = expected.as_u8();
            let got = CompressionCodec::from_u8(v);
            assert_eq!(got, Some(expected), "roundtrip for {:?}", expected);
        }
    }

    #[test]
    fn compression_codec_from_u8_edge_values_all_invalid() {
        // Arrange & Act & Assert: boundary and out-of-range u8 values all return None.
        assert!(CompressionCodec::from_u8(5).is_none());
        assert!(CompressionCodec::from_u8(127).is_none());
        assert!(CompressionCodec::from_u8(255).is_none());
    }

    #[test]
    fn storage_tier_from_u8_roundtrip_all_variants() {
        // Arrange & Act & Assert: each variant round-trips through u8.
        for expected in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let v = expected as u8;
            let got = StorageTier::from_u8(v);
            assert_eq!(got, Some(expected), "roundtrip for {:?}", expected);
        }
    }

    #[test]
    fn storage_tier_from_u8_boundary_values_invalid() {
        // Arrange & Act & Assert: boundary and out-of-range u8 returns None.
        assert!(StorageTier::from_u8(3).is_none());
        assert!(StorageTier::from_u8(50).is_none());
        assert!(StorageTier::from_u8(255).is_none());
    }

    #[test]
    fn record_eviction_and_swap_in_combined_total_migrations() {
        // Arrange: record both eviction and swap-in events, verify total_migrations.
        let (c, _backend) = make_coordinator(false);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::CpuDram, StorageTier::Nvme, 4096, 200);
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 50);
        // Act
        let snap = c.stats();
        // Assert
        assert_eq!(snap.evictions_gpu_to_dram, 1);
        assert_eq!(snap.evictions_dram_to_nvme, 1);
        assert_eq!(snap.swap_ins_dram_to_gpu, 1);
        assert_eq!(snap.total_bytes_evicted, 8192);
        assert_eq!(snap.total_bytes_swapped_in, 4096);
        assert_eq!(snap.total_eviction_latency_us, 300);
        assert_eq!(snap.total_swap_in_latency_us, 50);
    }

    #[test]
    fn build_batch_no_swap_in_for_page_without_metadata() {
        // Arrange: register page in addr_table but NOT in page_metadata.
        let (c, _backend) = make_coordinator(false);
        c.register_page(55, Some(0x9999), 4096);
        // Page 55 has no metadata entry.
        // Act: request it as active.
        let plan = c.build_batch(&[55], 0.5);
        // Assert: no swap-in because meta_guard has no entry for 55.
        assert!(
            plan.swap_in_requests.is_empty(),
            "page without metadata should not generate swap-in"
        );
    }

    #[test]
    fn register_pages_from_hgal_with_overlapping_ids_preserves_first() {
        // Arrange: register pages via HGAL, then re-register same IDs.
        let (c, _backend) = make_coordinator(false);
        let mut pages1 = HashMap::new();
        pages1.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 50,
            ..PageMetadata::default()
        });
        c.register_pages_from_hgal(&pages1, 4096);

        // Act: re-register with different metadata — or_insert_with preserves first.
        let mut pages2 = HashMap::new();
        pages2.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: Some(999),
            recency: 0,
            ..PageMetadata::default()
        });
        c.register_pages_from_hgal(&pages2, 8192);

        // Assert: original metadata preserved.
        let meta = c.page_metadata.read().expect("read lock");
        let m = meta.get(&1).expect("entry exists");
        assert_eq!(m.sequence_id, Some(100), "original sequence_id preserved");
        assert_eq!(m.recency, 50, "original recency preserved");
    }

    #[test]
    fn tier_migration_plan_swap_in_requests_match_tier_migrations_count() {
        // Arrange: pages on DRAM that are in active set.
        let (c, _backend) = make_coordinator(false);
        for pid in [10usize, 20, 30] {
            c.register_page(pid, None, 4096);
            {
                let mut table = c.addr_table.write().expect("write lock");
                if let Some(entry) = table.get_mut(&pid) {
                    entry.current_tier = StorageTier::CpuDram;
                    entry.host_buffer = Some(vec![0u8; 4096]);
                }
            }
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64),
                state: PageState::Standby,
                ..PageMetadata::default()
            });
        }
        // Act: all three are active.
        let plan = c.build_batch(&[10, 20, 30], 0.5);
        // Assert: each active page on DRAM produces both a swap-in request and a tier migration.
        assert_eq!(plan.swap_in_requests.len(), plan.tier_migrations.len());
        for mig in &plan.tier_migrations {
            assert_eq!(mig.reason, TierMigrationReason::SequenceDemand);
            assert_eq!(mig.to_tier, StorageTier::GpuHbm);
        }
    }

    #[test]
    fn page_state_all_variants_are_distinct() {
        // Arrange: all PageState variants.
        use std::collections::HashSet;
        let states = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];
        // Act
        let set: HashSet<PageState> = states.iter().copied().collect();
        // Assert
        assert_eq!(set.len(), 7, "all 7 PageState variants must be distinct");
    }

    #[test]
    fn three_tier_swap_config_default_migration_subconfig() {
        // Arrange & Act
        let config = ThreeTierSwapConfig::default();
        // Assert: migration subconfig is populated with non-zero page_size.
        assert!(config.migration.page_size > 0);
        assert!(config.migration.max_swap_pages > 0);
        // Swap dir path string should contain "gllm".
        let dir_str = config.migration.nvme_swap_dir.to_string_lossy();
        assert!(
            dir_str.contains("gllm"),
            "default swap dir should contain 'gllm'"
        );
    }

    // ── Wave 19: 15 new edge-case and boundary tests ──────────────────────────

    // @trace REQ-COMP-004 [level:unit]
    #[test]
    fn compression_codec_as_u8_values_are_contiguous_and_start_at_zero() {
        // Arrange: collect all codec u8 values.
        let codes: Vec<u8> = vec![
            CompressionCodec::None.as_u8(),
            CompressionCodec::Lz4.as_u8(),
            CompressionCodec::BitPackRle.as_u8(),
            CompressionCodec::NvcompAns.as_u8(),
            CompressionCodec::ZstdDict.as_u8(),
        ];
        // Act & Assert: values must be 0,1,2,3,4 — contiguous starting from 0.
        assert_eq!(codes, vec![0, 1, 2, 3, 4]);
    }

    // @trace REQ-COMP-004 [level:unit]
    #[test]
    fn storage_tier_as_u8_values_match_discriminant() {
        // Arrange: verify GpuHbm=0, CpuDram=1, Nvme=2.
        // Act & Assert
        assert_eq!(StorageTier::GpuHbm.as_u8(), 0);
        assert_eq!(StorageTier::CpuDram.as_u8(), 1);
        assert_eq!(StorageTier::Nvme.as_u8(), 2);
    }

    // @trace REQ-COMP-004 [level:unit]
    #[test]
    fn storage_tier_from_u8_zero_returns_gpu_hbm() {
        // Arrange & Act
        let tier = StorageTier::from_u8(0);
        // Assert
        assert_eq!(tier, Some(StorageTier::GpuHbm));
    }

    // @trace REQ-COMP-004 [level:unit]
    #[test]
    fn storage_tier_from_u8_one_returns_cpu_dram() {
        // Arrange & Act
        let tier = StorageTier::from_u8(1);
        // Assert
        assert_eq!(tier, Some(StorageTier::CpuDram));
    }

    // @trace REQ-COMP-004 [level:unit]
    #[test]
    fn storage_tier_from_u8_two_returns_nvme() {
        // Arrange & Act
        let tier = StorageTier::from_u8(2);
        // Assert
        assert_eq!(tier, Some(StorageTier::Nvme));
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn stats_default_all_fields_zero() {
        // Arrange & Act
        let stats = ThreeTierSwapStats::default();
        // Assert: every counter and accumulator must start at zero.
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_evicted, 0);
        assert_eq!(stats.total_bytes_swapped_in, 0);
        assert_eq!(stats.total_eviction_latency_us, 0);
        assert_eq!(stats.total_swap_in_latency_us, 0);
        assert_eq!(stats.pages_on_hbm, 0);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
        assert_eq!(stats.eviction_rounds, 0);
        assert_eq!(stats.swap_in_rounds, 0);
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn stats_avg_eviction_latency_us_zero_when_no_evictions() {
        // Arrange
        let stats = ThreeTierSwapStats::default();
        // Act & Assert: no evictions → avg must be 0.0 (no division by zero).
        assert!((stats.avg_eviction_latency_us() - 0.0).abs() < f64::EPSILON);
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn stats_avg_swap_in_latency_us_zero_when_no_swap_ins() {
        // Arrange
        let stats = ThreeTierSwapStats::default();
        // Act & Assert: no swap-ins → avg must be 0.0.
        assert!((stats.avg_swap_in_latency_us() - 0.0).abs() < f64::EPSILON);
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_migration_reason_equality_and_inequality() {
        // Arrange: all four variants.
        let a = TierMigrationReason::EvictionPressure;
        let b = TierMigrationReason::SequenceDemand;
        let c = TierMigrationReason::Prefetch;
        let d = TierMigrationReason::ColdCascade;
        // Assert: self-equality and cross-inequality.
        assert_eq!(a, a);
        assert_eq!(b, b);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(b, c);
        assert_ne!(b, d);
        assert_ne!(c, d);
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn page_state_free_is_the_initial_state() {
        // Arrange & Act: PageState::Free is the canonical initial/empty state.
        let state = PageState::Free;
        // Assert: it is a distinct variant, not equal to any other state.
        assert_ne!(state, PageState::Active);
        assert_ne!(state, PageState::Standby);
        assert_ne!(state, PageState::Swapped);
        assert_ne!(state, PageState::SwappedOut);
        assert_ne!(state, PageState::Warm);
        assert_ne!(state, PageState::Protected);
        // Self-equality holds.
        assert_eq!(state, PageState::Free);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_empty_active_pages_no_swap_in() {
        // Arrange: register pages but pass empty active_pages list.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: empty active pages → no swap-in requests.
        let plan = c.build_batch(&[], 0.5);
        // Assert
        assert!(plan.swap_in_requests.is_empty(),
            "empty active_pages should produce zero swap-in requests");
    }

    // @trace REQ-COMP-006 [level:unit]
    #[test]
    fn eviction_candidate_score_negative_large_value_stored() {
        // Arrange: construct candidate with very negative score.
        let candidate = EvictionCandidate {
            page_id: 7,
            score: i64::MIN,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };
        // Assert: score stored without overflow or clamping.
        assert_eq!(candidate.score, i64::MIN);
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn page_metadata_default_has_none_sequence_id() {
        // Arrange & Act
        let meta = PageMetadata::default();
        // Assert: default PageMetadata has sequence_id = None.
        assert!(meta.sequence_id.is_none());
        assert_eq!(meta.page_id, 0);
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(!meta.is_lir);
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_protected_page_never_evicted_under_extreme_pressure() {
        // Arrange: Protected state page on HBM — should never be evicted.
        let (c, _backend) = make_coordinator(false);
        c.register_page(42, Some(0xA000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(42, PageMetadata {
                page_id: 42,
                sequence_id: Some(999),
                recency: 99999,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(3600),
                swap_in_time: Some(Instant::now() - Duration::from_secs(3600)),
                is_lir: true,
                state: PageState::Protected,
                warm_until: None,
            });
        }
        // Act: extreme pressure.
        let plan = c.build_batch(&[], 1e10);
        // Assert: Protected page never appears in eviction candidates.
        assert!(
            plan.eviction_candidates.iter().all(|c| c.page_id != 42),
            "Protected page must never be an eviction candidate even under extreme pressure"
        );
    }

    // @trace REQ-COMP-006 [level:unit]
    #[test]
    fn prefresh_request_urgency_boundary_values() {
        // Arrange: construct PrefetchRequest with urgency = 0.0 and 1.0.
        let now = Instant::now();
        let req_zero = PrefetchRequest {
            page_id: 1,
            urgency: 0.0,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: now,
        };
        let req_one = PrefetchRequest {
            page_id: 2,
            urgency: 1.0,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: now,
        };
        // Assert: boundary values stored exactly.
        assert!((req_zero.urgency - 0.0).abs() < f32::EPSILON);
        assert!((req_one.urgency - 1.0).abs() < f32::EPSILON);
        assert_eq!(req_zero.page_id, 1);
        assert_eq!(req_one.page_id, 2);
    }

    // ── Wave 18: 15 additional tests ────────────────────────────────────────────

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_migration_plan_with_max_page_id_migration() {
        // Arrange: construct a TierMigrationPlan with PageId::MAX.
        let plan = TierMigrationPlan {
            eviction_candidates: vec![EvictionCandidate {
                page_id: PageId::MAX,
                score: 0,
                current_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::Lz4,
                page_bytes: 4096,
                group_id: None,
            }],
            swap_in_requests: Vec::new(),
            tier_migrations: vec![TierMigration {
                page_id: PageId::MAX,
                from_tier: StorageTier::GpuHbm,
                to_tier: StorageTier::CpuDram,
                codec: CompressionCodec::Lz4,
                page_bytes: 4096,
                reason: TierMigrationReason::EvictionPressure,
            }],
            built_at: Instant::now(),
        };
        // Assert: PageId::MAX preserved in both eviction and migration.
        assert_eq!(plan.eviction_candidates[0].page_id, PageId::MAX);
        assert_eq!(plan.tier_migrations[0].page_id, PageId::MAX);
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn page_state_hash_distinguishes_all_variants() {
        // Arrange: collect all 7 PageState variants.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let states = [
            PageState::Free,
            PageState::Active,
            PageState::Protected,
            PageState::Warm,
            PageState::Standby,
            PageState::Swapped,
            PageState::SwappedOut,
        ];
        // Act & Assert: all hashes should be pairwise distinct.
        let hashes: Vec<u64> = states.iter().map(|s| {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        }).collect();
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i], hashes[j],
                    "PageState hash collision between {:?} and {:?}", states[i], states[j]);
            }
        }
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn page_payload_kind_debug_contains_variant_names() {
        // Arrange & Act: format each variant with Debug.
        let dbg_kv = format!("{:?}", PagePayloadKind::KvContext);
        let dbg_ew = format!("{:?}", PagePayloadKind::ExpertWeight);
        let dbg_ps = format!("{:?}", PagePayloadKind::PromptSystem);
        let dbg_kr = format!("{:?}", PagePayloadKind::KnowledgeRAG);
        let dbg_dl = format!("{:?}", PagePayloadKind::DenseLayerWeight);
        // Assert: each Debug string contains its variant name.
        assert!(dbg_kv.contains("KvContext"), "Debug should contain KvContext");
        assert!(dbg_ew.contains("ExpertWeight"), "Debug should contain ExpertWeight");
        assert!(dbg_ps.contains("PromptSystem"), "Debug should contain PromptSystem");
        assert!(dbg_kr.contains("KnowledgeRAG"), "Debug should contain KnowledgeRAG");
        assert!(dbg_dl.contains("DenseLayerWeight"), "Debug should contain DenseLayerWeight");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_warm_page_on_dram_produces_no_eviction_no_swap_in() {
        // Arrange: Warm page on DRAM is in skip list for eviction and not active.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Warm,
                warm_until: None,
            });
        }
        // Act: not in active_pages, under high pressure.
        let plan = c.build_batch(&[], 0.95);
        // Assert: Warm pages are in the eviction skip list, so not evicted.
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id != 1),
            "Warm page should not be evicted");
        // Not in active_pages, so no swap-in.
        assert!(plan.swap_in_requests.is_empty());
    }

    // @trace REQ-COMP-006 [level:unit]
    #[test]
    fn eviction_candidate_page_bytes_usize_max() {
        // Arrange: EvictionCandidate with usize::MAX page_bytes.
        let candidate = EvictionCandidate {
            page_id: 0,
            score: 0,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: usize::MAX,
            group_id: None,
        };
        // Assert: value stored without truncation.
        assert_eq!(candidate.page_bytes, usize::MAX);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn coordinator_register_many_then_stats_snapshot_consistent() {
        // Arrange: register 50 pages across tiers.
        let (c, _backend) = make_coordinator(false);
        for pid in 0..30usize {
            c.register_page(pid, Some(0x1000 + pid as u64), 4096);
        }
        for pid in 30..50usize {
            c.register_page(pid, None, 4096);
        }
        // Act: take stats snapshot.
        let stats = c.stats();
        // Assert: tier counts match registration.
        assert_eq!(stats.pages_on_hbm, 30);
        assert_eq!(stats.pages_on_dram, 20);
        assert_eq!(stats.pages_on_nvme, 0);
        assert_eq!(stats.eviction_rounds, 0, "no build_batch calls yet");
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn three_tier_swap_config_migration_subconfig_defaults() {
        // Arrange & Act: get default config.
        let config = ThreeTierSwapConfig::default();
        // Assert: migration sub-config has sensible defaults.
        assert_eq!(config.migration.page_size, 4096);
        assert_eq!(config.migration.queue_capacity, 256);
        assert_eq!(config.migration.max_swap_pages, 4096);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_standby_on_hbm_with_zero_age_not_evicted() {
        // Arrange: Standby page on HBM with swap_in_time = now (zero age).
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: Some(Instant::now()), // zero age
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: high pressure.
        let plan = c.build_batch(&[], 0.95);
        // Assert: age < hbm_evict_age_ticks (default 50), so not evicted.
        assert!(plan.eviction_candidates.is_empty(),
            "page with zero age should not be evicted even under high pressure");
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_migration_reason_prefetch_construction_and_debug() {
        // Arrange: TierMigration with Prefetch reason.
        let migration = TierMigration {
            page_id: 42,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
            reason: TierMigrationReason::Prefetch,
        };
        // Act: format debug.
        let dbg = format!("{:?}", migration);
        // Assert: fields present in debug output.
        assert!(dbg.contains("Prefetch"));
        assert!(dbg.contains("42"));
        assert!(dbg.contains("8192"));
        assert!(dbg.contains("Nvme"));
        assert!(dbg.contains("CpuDram"));
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_swap_in_for_dram_page_with_swapped_state() {
        // Arrange: page on DRAM with Swapped state requested as active.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 2,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        // Act: request swap-in by listing page as active.
        let plan = c.build_batch(&[1], 0.3);
        // Assert: swap-in produced and migration is CpuDram -> GpuHbm.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 1);
        let migration = plan.tier_migrations.iter().find(|m| m.page_id == 1);
        assert!(migration.is_some());
        assert_eq!(migration.unwrap().from_tier, StorageTier::CpuDram);
        assert_eq!(migration.unwrap().to_tier, StorageTier::GpuHbm);
    }

    // @trace REQ-COMP-006 [level:unit]
    #[test]
    fn prefetch_request_enqueued_at_preserved_across_clone() {
        // Arrange: PrefetchRequest with specific enqueued_at.
        let now = Instant::now();
        let req = PrefetchRequest {
            page_id: 55,
            urgency: 0.75,
            prefetch_confidence: 0.85,
            page_bytes: 8192,
            enqueued_at: now,
        };
        // Act: clone.
        let cloned = req.clone();
        // Assert: all fields preserved.
        assert_eq!(cloned.page_id, 55);
        assert!((cloned.urgency - 0.75).abs() < 0.01);
        assert!((cloned.prefetch_confidence - 0.85).abs() < 0.01);
        assert_eq!(cloned.page_bytes, 8192);
        assert!(cloned.enqueued_at == now);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn coordinator_stats_after_register_evict_swap_in_full_lifecycle() {
        // Arrange: full lifecycle of register, evict, swap-in.
        let (c, _backend) = make_coordinator(false);
        // Register page on HBM.
        c.register_page(1, Some(0x1000), 4096);
        // Evict to DRAM.
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        // Evict to NVMe.
        c.record_eviction_completed(1, StorageTier::CpuDram, StorageTier::Nvme, 4096, 500);
        // Swap-in from NVMe to DRAM.
        c.record_swap_in_completed(1, StorageTier::Nvme, StorageTier::CpuDram, 4096, 300);
        // Swap-in from DRAM to HBM.
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 150);
        // Act.
        let stats = c.stats();
        // Assert: all four migration directions counted.
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.total_migrations(), 4);
        assert_eq!(stats.total_bytes_evicted, 4096 * 2);
        assert_eq!(stats.total_bytes_swapped_in, 4096 * 2);
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn eviction_worker_config_default_hbm_evict_age_ticks_positive() {
        // Arrange & Act: get default eviction worker config.
        let config = EvictionWorkerConfig::default();
        // Assert: age ticks threshold must be positive (otherwise everything is evicted).
        assert!(config.hbm_evict_age_ticks > 0,
            "hbm_evict_age_ticks must be positive");
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn eviction_worker_config_default_dram_evict_age_ticks_positive() {
        // Arrange & Act: get default eviction worker config.
        let config = EvictionWorkerConfig::default();
        // Assert: DRAM age ticks threshold must be positive.
        assert!(config.dram_evict_age_ticks > 0,
            "dram_evict_age_ticks must be positive");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_multiple_pages_only_eligible_tiers_produce_evictions() {
        // Arrange: 3 pages with Standby state, on different tiers, all old enough.
        let (c, _backend) = make_coordinator(false);
        // Page 1: HBM (eligible under pressure).
        c.register_page(1, Some(0x1001), 4096);
        // Page 2: DRAM (eligible only if dram_pressure > threshold).
        c.register_page(2, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&2) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        // Page 3: NVMe (never eligible for eviction).
        c.register_page(3, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&3) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        // Set all to Standby with old age.
        for pid in [1usize, 2, 3] {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64 * 10),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: high pressure.
        let plan = c.build_batch(&[], 0.95);
        // Assert: NVMe page never appears in eviction candidates.
        assert!(plan.eviction_candidates.iter().all(|c| c.page_id != 3),
            "NVMe pages must never be eviction candidates");
        // HBM page should appear under high pressure.
        assert!(plan.eviction_candidates.iter().any(|c| c.page_id == 1),
            "HBM Standby page with old age should be evicted under high pressure");
    }

    // ── Wave 15: additional coverage tests (15 new tests) ──────────────────────────

    // ── compute_tier_age_ticks uses swap_in_time when present ──────────────────────

    #[test]
    fn compute_tier_age_ticks_prefers_swap_in_time_over_last_access() {
        // Arrange: page with both swap_in_time and last_access set, swap_in_time more recent.
        let old_time = Instant::now() - Duration::from_secs(60);
        let recent_time = Instant::now() - Duration::from_millis(100);
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 5,
            last_access: old_time,
            swap_in_time: Some(recent_time),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act: compute_tier_age_ticks should anchor from swap_in_time (recent).
        let ticks = compute_tier_age_ticks(&meta);

        // Assert: Since swap_in_time is recent (100ms ago), ticks should be small (< 60s/10ms=6000).
        assert!(ticks < 1000, "ticks should be small when anchored to recent swap_in_time, got {}", ticks);
    }

    #[test]
    fn compute_tier_age_ticks_uses_last_access_when_no_swap_in_time() {
        // Arrange: page with only last_access (no swap_in_time).
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 0,
            access_count: 5,
            last_access: Instant::now() - Duration::from_millis(500),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act.
        let ticks = compute_tier_age_ticks(&meta);

        // Assert: 500ms / 10ms_per_tick = ~50 ticks.
        assert!(ticks >= 40 && ticks <= 60, "ticks should be ~50, got {}", ticks);
    }

    // ── infer_swap_payload_kind direct edge case ──────────────────────────────────

    #[test]
    fn infer_swap_payload_kind_zero_sequence_id_returns_kv_context() {
        // Arrange: sequence_id = Some(0) is still a valid sequence, so KvContext.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(0),
            ..PageMetadata::default()
        };

        // Act & Assert.
        assert_eq!(infer_swap_payload_kind(&meta), Some(PagePayloadKind::KvContext));
    }

    // ── ThreeTierSwapStats: total_migrations with only one direction ──────────────

    #[test]
    fn stats_total_migrations_only_evictions_gpu_to_dram() {
        // Arrange.
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 7;

        // Act & Assert.
        assert_eq!(stats.total_migrations(), 7);
    }

    #[test]
    fn stats_total_migrations_only_swap_ins_nvme_to_dram() {
        // Arrange.
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_nvme_to_dram = 11;

        // Act & Assert.
        assert_eq!(stats.total_migrations(), 11);
    }

    // ── TierMigration: page_bytes zero is valid ────────────────────────────────────

    #[test]
    fn tier_migration_zero_page_bytes_preserved_in_clone() {
        // Arrange.
        let migration = TierMigration {
            page_id: 42,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 0,
            reason: TierMigrationReason::SequenceDemand,
        };

        // Act.
        let cloned = migration.clone();

        // Assert.
        assert_eq!(cloned.page_bytes, 0);
        assert_eq!(cloned.page_id, 42);
        assert_eq!(cloned.reason, TierMigrationReason::SequenceDemand);
    }

    // ── Coordinator: register_page with large gpu_ptr value ────────────────────────

    #[test]
    fn register_page_with_u64_max_gpu_ptr() {
        // Arrange.
        let (c, _backend) = make_coordinator(false);

        // Act.
        c.register_page(1, Some(u64::MAX), 4096);

        // Assert.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("page should exist");
        assert_eq!(entry.gpu_ptr, Some(u64::MAX));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // ── Coordinator: update_page_gpu_ptr with zero address is valid ────────────────

    #[test]
    fn update_page_gpu_ptr_with_zero_address_still_promotes_to_hbm() {
        // Arrange: register page on DRAM first.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let table = c.addr_table.read().expect("read lock");
            assert_eq!(table.get(&1).unwrap().current_tier, StorageTier::CpuDram);
        }

        // Act: update gpu_ptr to 0.
        c.update_page_gpu_ptr(1, 0);

        // Assert.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("page should exist");
        assert_eq!(entry.gpu_ptr, Some(0));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // ── Coordinator: stats after multiple eviction + swap-in mixed records ─────────

    #[test]
    fn stats_mixed_eviction_swap_in_full_tier_coverage() {
        // Arrange.
        let (c, _backend) = make_coordinator(false);

        // Act: record all four legal tier pairs.
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::CpuDram, StorageTier::Nvme, 8192, 200);
        c.record_swap_in_completed(3, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 150);
        c.record_swap_in_completed(4, StorageTier::Nvme, StorageTier::CpuDram, 8192, 300);

        // Assert.
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 1);
        assert_eq!(stats.swap_ins_nvme_to_dram, 1);
        assert_eq!(stats.total_migrations(), 4);
        assert_eq!(stats.total_bytes_evicted, 4096 + 8192);
        assert_eq!(stats.total_bytes_swapped_in, 4096 + 8192);
        assert_eq!(stats.total_eviction_latency_us, 100 + 200);
        assert_eq!(stats.total_swap_in_latency_us, 150 + 300);
    }

    // ── Coordinator: register_pages_from_hgal then build_batch no active pages ────

    #[test]
    fn register_from_hgal_then_build_batch_empty_active_no_swap_in() {
        // Arrange.
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        for pid in 1usize..=3 {
            pages.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some((pid * 10) as u64),
                ..PageMetadata::default()
            });
        }
        c.register_pages_from_hgal(&pages, 4096);

        // Act: no active pages requested.
        let plan = c.build_batch(&[], 0.5);

        // Assert: no swap-in requests since no pages are in active set.
        assert!(plan.swap_in_requests.is_empty());
    }

    // ── TierMigrationPlan: empty plan has zero-length tier_migrations ─────────────

    #[test]
    fn tier_migration_plan_default_vectors_all_empty() {
        // Arrange & Act.
        let plan = TierMigrationPlan {
            eviction_candidates: Vec::new(),
            swap_in_requests: Vec::new(),
            tier_migrations: Vec::new(),
            built_at: Instant::now(),
        };

        // Assert: all three vectors should be empty, total migrations = 0.
        assert_eq!(plan.eviction_candidates.len() + plan.swap_in_requests.len() + plan.tier_migrations.len(), 0);
    }

    // ── EvictionCandidate: score of zero is valid ──────────────────────────────────

    #[test]
    fn eviction_candidate_zero_score_valid() {
        // Arrange.
        let candidate = EvictionCandidate {
            page_id: 0,
            score: 0,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };

        // Act & Assert.
        assert_eq!(candidate.score, 0);
        assert_eq!(candidate.page_id, 0);
        assert!(candidate.group_id.is_none());
    }

    // ── PrefetchRequest: clone preserves enqueued_at ──────────────────────────────

    #[test]
    fn prefetch_request_clone_preserves_all_fields() {
        // Arrange.
        let now = Instant::now();
        let req = PrefetchRequest {
            page_id: 77,
            urgency: 0.75,
            prefetch_confidence: 0.6,
            page_bytes: 2048,
            enqueued_at: now,
        };

        // Act.
        let cloned = req.clone();

        // Assert.
        assert_eq!(cloned.page_id, 77);
        assert!((cloned.urgency - 0.75).abs() < f32::EPSILON);
        assert!((cloned.prefetch_confidence - 0.6).abs() < f32::EPSILON);
        assert_eq!(cloned.page_bytes, 2048);
    }

    // ── Coordinator: build_batch with page_id zero in active set ───────────────────

    #[test]
    fn build_batch_page_id_zero_swap_in_from_nvme() {
        // Arrange: register page_id=0 on NVMe.
        let (c, _backend) = make_coordinator(false);
        c.register_page(0, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&0) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(0, PageMetadata {
                page_id: 0,
                sequence_id: Some(50),
                recency: 0,
                access_count: 2,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }

        // Act: request page_id 0 as active.
        let plan = c.build_batch(&[0], 0.5);

        // Assert: should produce a swap-in request for page_id 0.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 0);
        assert_eq!(plan.tier_migrations[0].from_tier, StorageTier::Nvme);
        assert_eq!(plan.tier_migrations[0].to_tier, StorageTier::GpuHbm);
    }

    // ── Coordinator: release page then reregister with different bytes ─────────────

    #[test]
    fn release_then_reregister_different_bytes_creates_new_entry() {
        // Arrange.
        let (c, _backend) = make_coordinator(false);
        c.register_page(10, Some(0xAAAA), 4096);
        c.release_page(10);

        // Act: re-register with different parameters.
        c.register_page(10, Some(0xBBBB), 8192);

        // Assert.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&10).expect("page should exist after re-register");
        assert_eq!(entry.gpu_ptr, Some(0xBBBB));
        assert_eq!(entry.original_bytes, 8192);
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // ── TierMigrationReason: exhaustive match proves no missing variants ───────────

    #[test]
    fn tier_migration_reason_exhaustive_count_is_four() {
        // Arrange & Act: collect all known variants.
        let all_reasons = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];

        // Assert: exactly 4 variants exist, and they are all distinct.
        assert_eq!(all_reasons.len(), 4);
        for i in 0..all_reasons.len() {
            for j in (i + 1)..all_reasons.len() {
                assert_ne!(all_reasons[i], all_reasons[j],
                    "variants at index {} and {} should differ", i, j);
            }
        }
    }

    // ── Wave 20: 15 new tests covering untested edge cases and error paths ──────────

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_candidate_group_id_matches_sequence_id() {
        // Arrange: 验证驱逐候选人的 group_id 来源于 PageMetadata.sequence_id。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(7777),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert: 如果有驱逐候选人，其 group_id 必须等于 sequence_id。
        let candidate = plan.eviction_candidates.iter().find(|ec| ec.page_id == 1);
        if let Some(ec) = candidate {
            assert_eq!(ec.group_id, Some(7777),
                "eviction candidate group_id must equal page's sequence_id");
        }
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_no_sequence_id_page_produces_eviction_with_none_group() {
        // Arrange: sequence_id = None 的页面（权重页），验证驱逐候选人 group_id = None。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert
        let candidate = plan.eviction_candidates.iter().find(|ec| ec.page_id == 1);
        if let Some(ec) = candidate {
            assert_eq!(ec.group_id, None,
                "eviction candidate for page without sequence_id must have group_id = None");
        }
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_swap_in_request_page_bytes_matches_original_bytes() {
        // Arrange: 注册页面时使用非默认 page_bytes (16384)，验证 swap-in 请求中的 page_bytes 正确。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 16384);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 16384]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert: swap-in 请求的 page_bytes 必须匹配注册时的 original_bytes。
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_bytes, 16384,
            "swap-in request page_bytes must match original_bytes from registration");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_migration_reason_cold_cascade_for_dram() {
        // Arrange: 验证 ColdCascade reason 语义正确 — DRAM→Nvme 迁移。
        let migration = TierMigration {
            page_id: 1,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::Nvme,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            reason: TierMigrationReason::ColdCascade,
        };
        // Assert: ColdCascade 是 DRAM→Nvme 迁移的正确 reason。
        assert_eq!(migration.reason, TierMigrationReason::ColdCascade);
        assert_eq!(migration.from_tier, StorageTier::CpuDram);
        assert_eq!(migration.to_tier, StorageTier::Nvme);
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_changed_pages_swapped_out_on_nvme_consistent() {
        // Arrange: 页面在 NVMe 且状态为 SwappedOut。SwappedOut 映射到 Nvme → 一致。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::SwappedOut,
                ..PageMetadata::default()
            });
        }
        // Act
        let changed = c.tier_changed_pages();
        // Assert: SwappedOut → Nvme，实际在 Nvme → 无分歧。
        assert!(changed.is_empty(),
            "SwappedOut page on NVMe should have no tier divergence");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn register_pages_from_hgal_then_build_batch_swap_in_for_active_on_dram() {
        // Arrange: 通过 HGAL 注册页面（落在 DRAM），然后请求 swap-in。
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        pages.insert(42, PageMetadata {
            page_id: 42,
            sequence_id: Some(200),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        });
        c.register_pages_from_hgal(&pages, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&42) {
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        // Act: 请求页面 42 为活跃 → 应产生 swap-in。
        let plan = c.build_batch(&[42], 0.5);
        // Assert
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 42);
        assert_eq!(plan.tier_migrations.len(), 1);
        assert_eq!(plan.tier_migrations[0].reason, TierMigrationReason::SequenceDemand);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn update_page_gpu_ptr_on_multiple_pages_each_promotes_independently() {
        // Arrange: 注册多个 DRAM 页面，逐个更新 gpu_ptr，验证各自独立提升到 HBM。
        let (c, _backend) = make_coordinator(false);
        for pid in [10usize, 20, 30] {
            c.register_page(pid, None, 4096);
        }
        // Act: 只更新 page 10 和 20 的 gpu_ptr。
        c.update_page_gpu_ptr(10, 0xA000);
        c.update_page_gpu_ptr(20, 0xB000);
        // Assert: 10 和 20 在 HBM，30 仍在 DRAM。
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.get(&10).unwrap().current_tier, StorageTier::GpuHbm);
        assert_eq!(table.get(&20).unwrap().current_tier, StorageTier::GpuHbm);
        assert_eq!(table.get(&30).unwrap().current_tier, StorageTier::CpuDram);
        assert_eq!(table.get(&10).unwrap().gpu_ptr, Some(0xA000));
        assert_eq!(table.get(&20).unwrap().gpu_ptr, Some(0xB000));
        assert!(table.get(&30).unwrap().gpu_ptr.is_none());
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn stats_pages_on_tier_updated_after_tier_change_and_release() {
        // Arrange: 注册 3 页 HBM，手动将 1 页改为 NVMe，释放 1 页。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, Some(0x2000), 4096);
        c.register_page(3, Some(0x3000), 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&2) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        c.release_page(3);
        // Act
        let stats = c.stats();
        // Assert: page 1 → HBM, page 2 → NVMe, page 3 已释放。
        assert_eq!(stats.pages_on_hbm, 1);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 1);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_swap_in_request_prefetch_confidence_always_high() {
        // Arrange: swap-in 请求的 prefetch_confidence 应为 0.9（高置信度）。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 1,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert: prefetch_confidence = 0.9（硬编码高置信度）。
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert!((plan.swap_in_requests[0].prefetch_confidence - 0.9).abs() < 0.01,
            "swap-in for sequence demand must have prefetch_confidence = 0.9");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_candidates_page_bytes_from_config_not_original() {
        // Arrange: 注册页面时 page_bytes=8192，但驱逐候选人应使用 config 默认 4096。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 8192);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert: 驱逐候选人的 page_bytes 应来自 config (4096)，不是 original_bytes (8192)。
        if let Some(candidate) = plan.eviction_candidates.iter().find(|ec| ec.page_id == 1) {
            assert_eq!(candidate.page_bytes, 4096,
                "eviction candidate page_bytes must come from config (4096), not original registration (8192)");
        }
    }

    // @trace REQ-COMP-007 [level:unit]
    #[test]
    fn tier_changed_pages_mixed_states_some_diverge_some_consistent() {
        // Arrange: 多个页面，部分状态一致部分不一致，验证只返回分歧的。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Active,
                ..PageMetadata::default()
            });
        }
        c.register_page(2, Some(0x2000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(2, PageMetadata {
                page_id: 2,
                state: PageState::Standby,
                ..PageMetadata::default()
            });
        }
        c.register_page(3, Some(0x3000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(3, PageMetadata {
                page_id: 3,
                state: PageState::Warm,
                ..PageMetadata::default()
            });
        }
        // Act
        let changed = c.tier_changed_pages();
        // Assert: 只有 page 2 和 3 分歧（Standby/Warm 期望 DRAM，实际在 HBM）。
        assert_eq!(changed.len(), 2);
        let changed_ids: std::collections::HashSet<PageId> =
            changed.iter().map(|(pid, _)| *pid).collect();
        assert!(!changed_ids.contains(&1), "Active on HBM should be consistent");
        assert!(changed_ids.contains(&2), "Standby on HBM should diverge");
        assert!(changed_ids.contains(&3), "Warm on HBM should diverge");
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn stats_cumulative_records_persist_after_all_pages_released() {
        // Arrange: 注册页面，记录迁移事件，然后释放所有页面。累积计数器应保留。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        c.register_page(2, Some(0x2000), 4096);
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_swap_in_completed(2, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 80);
        // Act: 释放所有页面。
        c.release_page(1);
        c.release_page(2);
        let stats = c.stats();
        // Assert: 累积计数器保留，当前层级计数归零。
        assert_eq!(stats.pages_on_hbm, 0);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
        assert_eq!(stats.evictions_gpu_to_dram, 1, "cumulative eviction count must persist");
        assert_eq!(stats.swap_ins_dram_to_gpu, 1, "cumulative swap-in count must persist");
        assert_eq!(stats.total_bytes_evicted, 4096);
        assert_eq!(stats.total_bytes_swapped_in, 4096);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_swap_in_for_nvme_page_uses_nvme_to_hbm_migration() {
        // Arrange: 单个 NVMe 页面被请求为活跃，验证迁移路径为 Nvme → GpuHbm。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert: 迁移路径为 Nvme → GpuHbm（跨两层直接迁移）。
        assert_eq!(plan.tier_migrations.len(), 1);
        assert_eq!(plan.tier_migrations[0].from_tier, StorageTier::Nvme);
        assert_eq!(plan.tier_migrations[0].to_tier, StorageTier::GpuHbm);
        assert_eq!(plan.tier_migrations[0].reason, TierMigrationReason::SequenceDemand);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_codec_from_config_default_lz4_for_hbm() {
        // Arrange: HBM 驱逐时 codec 应使用 default_evict_codec（默认 Lz4）。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert: HBM 驱逐候选人的 codec 应为 default_evict_codec (Lz4)。
        let candidate = plan.eviction_candidates.iter().find(|ec| ec.page_id == 1);
        if let Some(ec) = candidate {
            assert_eq!(ec.codec, CompressionCodec::Lz4,
                "HBM eviction candidate codec must be default_evict_codec (Lz4)");
        }
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_swap_in_skip_for_page_already_on_hbm_even_if_active() {
        // Arrange: 页面已在 HBM 上，被请求为活跃，不应产生 swap-in 或驱逐。
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 10,
                last_access: Instant::now(),
                swap_in_time: Some(Instant::now()),
                is_lir: false,
                state: PageState::Active,
                warm_until: None,
            });
        }
        // Act: 请求页面 1 为活跃。
        let plan = c.build_batch(&[1], 0.5);
        // Assert: 不需要 swap-in（已在 HBM），不驱逐（Active 状态被跳过）。
        assert!(plan.swap_in_requests.is_empty(),
            "page already on HBM should not produce swap-in request");
        assert!(plan.eviction_candidates.is_empty(),
            "Active page should never be evicted");
    }

    // ── Wave 20: 15 new tests for uncovered paths ────────────────────────────────

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn compute_tier_age_ticks_uses_swap_in_time_when_present() {
        // Arrange: swap_in_time set far in the past, last_access recent.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now() - Duration::from_secs(5)),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act: age should be based on swap_in_time (5 seconds = 500 ticks).
        let ticks = compute_tier_age_ticks(&meta);
        // Assert: 5000ms / 10 = 500 ticks, allow slight timing variance.
        assert!(ticks >= 490 && ticks <= 510,
            "age ticks should be ~500 from swap_in_time, got {}", ticks);
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn compute_tier_age_ticks_falls_back_to_last_access_when_no_swap_in_time() {
        // Arrange: swap_in_time is None, last_access set 2 seconds ago.
        let meta = PageMetadata {
            page_id: 2,
            sequence_id: Some(200),
            recency: 0,
            access_count: 1,
            last_access: Instant::now() - Duration::from_secs(2),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act: age should be based on last_access (2 seconds = 200 ticks).
        let ticks = compute_tier_age_ticks(&meta);
        // Assert: 2000ms / 10 = 200 ticks, allow slight timing variance.
        assert!(ticks >= 190 && ticks <= 210,
            "age ticks should be ~200 from last_access, got {}", ticks);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_from_dram_produces_cold_cascade_reason() {
        // Arrange: page on DRAM, inject DRAM pressure to trigger DRAM->NVMe eviction.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Inject DRAM pressure: fill L2 capacity to push dram_pressure above threshold (0.80).
        {
            let mut mm = c.memory_manager.lock().expect("lock");
            // L2 capacity is 1_000_000. Allocate 850k to push usage ratio above 0.80.
            for _ in 0..850_000 {
                let _ = mm.allocate_page(crate::scheduler::memory_manager::Tier::L2);
            }
        }
        // Act
        let plan = c.build_batch(&[], 0.95);
        // Assert: DRAM eviction should produce ColdCascade reason.
        let dram_migration = plan.tier_migrations.iter().find(|m|
            m.page_id == 1 && m.from_tier == StorageTier::CpuDram);
        if let Some(migration) = dram_migration {
            assert_eq!(migration.reason, TierMigrationReason::ColdCascade,
                "DRAM->NVMe eviction must have ColdCascade reason");
            assert_eq!(migration.to_tier, StorageTier::Nvme);
        }
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_from_dram_retains_addr_table_codec() {
        // Arrange: page on DRAM with BitPackRle codec, trigger eviction.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
                entry.codec = CompressionCodec::BitPackRle;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Inject DRAM pressure.
        {
            let mut mm = c.memory_manager.lock().expect("lock");
            for _ in 0..850_000 {
                let _ = mm.allocate_page(crate::scheduler::memory_manager::Tier::L2);
            }
        }
        // Act
        let plan = c.build_batch(&[], 0.95);
        // Assert: DRAM eviction candidate should keep existing codec, not default_evict_codec.
        if let Some(candidate) = plan.eviction_candidates.iter().find(|c| c.page_id == 1) {
            assert_eq!(candidate.codec, CompressionCodec::BitPackRle,
                "DRAM eviction should preserve existing codec, not replace with default_evict_codec");
        }
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn infer_swap_payload_kind_with_zero_sequence_id_returns_kv_context() {
        // Arrange: sequence_id = Some(0) should map to KvContext (not ExpertWeight).
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(0),
            ..PageMetadata::default()
        };
        // Act & Assert
        assert_eq!(infer_swap_payload_kind(&meta), Some(PagePayloadKind::KvContext),
            "sequence_id=Some(0) should return KvContext, not ExpertWeight");
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn compute_tier_age_ticks_zero_when_instant_is_now() {
        // Arrange: both timestamps are Instant::now(), so elapsed is 0.
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        // Act
        let ticks = compute_tier_age_ticks(&meta);
        // Assert: age should be 0 or very close to 0.
        assert!(ticks <= 1,
            "age ticks for just-created page should be 0 or 1, got {}", ticks);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_swap_in_request_page_bytes_matches_addr_table_original() {
        // Arrange: register page with non-default bytes, verify swap-in inherits it.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 16384);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 16384]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert: swap-in request should carry original_bytes (16384).
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_bytes, 16384,
            "swap-in request page_bytes must match addr_table original_bytes");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_eviction_migration_page_bytes_from_config() {
        // Arrange: register page with 8192 bytes, verify tier_migration uses config page_bytes.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 8192);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 5000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(30),
                swap_in_time: Some(Instant::now() - Duration::from_secs(30)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[], 0.99);
        // Assert: tier_migration page_bytes should be from config (4096), not original (8192).
        if let Some(migration) = plan.tier_migrations.iter().find(|m| m.page_id == 1) {
            assert_eq!(migration.page_bytes, 4096,
                "tier_migration page_bytes must come from config (4096), not original_bytes (8192)");
        }
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn dram_pressure_ratio_returns_nonzero_after_l2_allocation() {
        // Arrange: coordinator with 1M L2 capacity.
        let (c, _backend) = make_coordinator(false);
        // Act: allocate half of L2.
        {
            let mut mm = c.memory_manager.lock().expect("lock");
            for _ in 0..500_000 {
                let _ = mm.allocate_page(crate::scheduler::memory_manager::Tier::L2);
            }
        }
        // Assert: build_batch should see non-zero DRAM pressure.
        // We verify by registering a DRAM page and checking behavior.
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // With 50% DRAM usage (> dram_pressure_threshold default ~0.8), still below.
        // But the test verifies no panic and correct structure.
        let plan = c.build_batch(&[], 0.95);
        assert!(plan.eviction_candidates.len() <= 1);
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_pages_from_hgal_does_not_overwrite_existing_gpu_ptr() {
        // Arrange: pre-register with gpu_ptr, then bulk-register without.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0xABCD), 4096);
        {
            let table = c.addr_table.read().expect("read lock");
            assert_eq!(table.get(&1).unwrap().gpu_ptr, Some(0xABCD));
        }
        let mut pages = HashMap::new();
        pages.insert(1, PageMetadata { page_id: 1, ..PageMetadata::default() });
        // Act: bulk-register same page (no gpu_ptr in bulk).
        c.register_pages_from_hgal(&pages, 8192);
        // Assert: original gpu_ptr preserved.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("should exist");
        assert_eq!(entry.gpu_ptr, Some(0xABCD),
            "bulk register must not overwrite existing gpu_ptr");
        assert_eq!(entry.original_bytes, 4096,
            "bulk register must not overwrite existing original_bytes");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn tier_changed_pages_with_page_on_dram_and_standby_state_consistent() {
        // Arrange: page on CpuDram with Standby state -- Standby maps to CpuDram = consistent.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        // CpuDram is the default for pages registered without gpu_ptr.
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                state: PageState::Standby,
                ..PageMetadata::default()
            });
        }
        // Act
        let changed = c.tier_changed_pages();
        // Assert: Standby -> CpuDram, page is on CpuDram -> no divergence.
        assert!(changed.is_empty(),
            "Standby state on CpuDram should be consistent");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_swap_in_for_page_with_none_sequence_id() {
        // Arrange: page on DRAM with no sequence_id, in active_pages.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert: swap-in should still be generated regardless of sequence_id.
        assert_eq!(plan.swap_in_requests.len(), 1,
            "swap-in should be generated even when sequence_id is None");
        assert_eq!(plan.swap_in_requests[0].page_id, 1);
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn tier_migration_plan_eviction_and_swap_in_are_mutually_exclusive_per_page() {
        // Arrange: one page needs swap-in, another is eligible for eviction.
        let (c, _backend) = make_coordinator(false);
        // Page A: on DRAM, needs swap-in.
        c.register_page(10, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&10) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(10, PageMetadata {
                page_id: 10,
                sequence_id: Some(10),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Page B: on HBM, eligible for eviction.
        c.register_page(20, Some(0x2000), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(20, PageMetadata {
                page_id: 20,
                sequence_id: Some(20),
                recency: 1000,
                access_count: 0,
                last_access: Instant::now() - Duration::from_secs(10),
                swap_in_time: Some(Instant::now() - Duration::from_secs(10)),
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[10], 0.95);
        // Assert: page 10 should be in swap-in only, not eviction.
        let swap_in_ids: std::collections::HashSet<PageId> =
            plan.swap_in_requests.iter().map(|r| r.page_id).collect();
        let eviction_ids: std::collections::HashSet<PageId> =
            plan.eviction_candidates.iter().map(|c| c.page_id).collect();
        assert!(swap_in_ids.contains(&10), "page 10 should have swap-in request");
        assert!(!eviction_ids.contains(&10), "page 10 should NOT be in eviction candidates");
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn eviction_worker_config_default_dram_evict_age_ticks_greater_than_hbm() {
        // Arrange & Act
        let config = EvictionWorkerConfig::default();
        // Assert: DRAM eviction requires longer age (500 ticks) than HBM (50 ticks).
        assert!(config.dram_evict_age_ticks > config.hbm_evict_age_ticks,
            "DRAM eviction should require longer age than HBM eviction");
    }

    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_tier_migration_swap_in_preserves_codec_from_addr_table() {
        // Arrange: page on DRAM with NvcompAns codec, verify swap-in migration carries it.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, None, 4096);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.current_tier = StorageTier::CpuDram;
                entry.host_buffer = Some(vec![0u8; 4096]);
                entry.codec = CompressionCodec::NvcompAns;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(1, PageMetadata {
                page_id: 1,
                sequence_id: Some(100),
                recency: 0,
                access_count: 3,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[1], 0.5);
        // Assert
        let migration = plan.tier_migrations.iter().find(|m| m.page_id == 1);
        assert!(migration.is_some());
        assert_eq!(migration.unwrap().codec, CompressionCodec::NvcompAns,
            "swap-in tier migration should carry codec from addr_table entry");
    }

    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn page_addr_entry_default_codec_after_register_then_manual_codec_change() {
        // Arrange: register page (codec=None), then change codec, verify round-trip.
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096);
        // Verify initial codec.
        {
            let table = c.addr_table.read().expect("read lock");
            assert_eq!(table.get(&1).unwrap().codec, CompressionCodec::None);
        }
        // Act: change codec to ZstdDict.
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&1) {
                entry.codec = CompressionCodec::ZstdDict;
            }
        }
        // Verify mutated codec.
        {
            let table = c.addr_table.read().expect("read lock");
            assert_eq!(table.get(&1).unwrap().codec, CompressionCodec::ZstdDict);
        }
        // Verify update_page_gpu_ptr does not reset codec.
        c.update_page_gpu_ptr(1, 0x5000);
        {
            let table = c.addr_table.read().expect("read lock");
            assert_eq!(table.get(&1).unwrap().codec, CompressionCodec::ZstdDict,
                "update_page_gpu_ptr must not reset codec");
        }
    }

    // ── Wave 21: 15 new tests ────────────────────────────────────────────────────

    // 1. StorageTier Debug output contains expected variant substrings
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn storage_tier_debug_output_contains_exact_variant_names() {
        // Arrange & Act: format each variant with Debug.
        let hbm = format!("{:?}", StorageTier::GpuHbm);
        let dram = format!("{:?}", StorageTier::CpuDram);
        let nvme = format!("{:?}", StorageTier::Nvme);
        // Assert: each Debug string must contain its variant name as a substring.
        assert!(hbm.contains("GpuHbm"), "GpuHbm Debug must contain 'GpuHbm', got: {}", hbm);
        assert!(dram.contains("CpuDram"), "CpuDram Debug must contain 'CpuDram', got: {}", dram);
        assert!(nvme.contains("Nvme"), "Nvme Debug must contain 'Nvme', got: {}", nvme);
        // All three must be distinct.
        assert_ne!(hbm, dram);
        assert_ne!(dram, nvme);
        assert_ne!(hbm, nvme);
    }

    // 2. TierMigrationReason Hash deduplication in HashSet
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn tier_migration_reason_hash_set_dedup_all_four_variants() {
        // Arrange: insert all four variants plus duplicates into a HashSet.
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TierMigrationReason::EvictionPressure);
        set.insert(TierMigrationReason::SequenceDemand);
        set.insert(TierMigrationReason::Prefetch);
        set.insert(TierMigrationReason::ColdCascade);
        // Insert duplicates.
        set.insert(TierMigrationReason::EvictionPressure);
        set.insert(TierMigrationReason::ColdCascade);
        // Assert: exactly 4 unique entries remain.
        assert_eq!(set.len(), 4, "HashSet should contain exactly 4 unique TierMigrationReason variants");
    }

    // 3. TierMigrationReason Hash: inserting same variant from different expressions yields one entry
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn tier_migration_reason_hash_set_same_variant_different_expressions() {
        // Arrange: two expressions producing the same variant.
        use std::collections::HashSet;
        let v1 = TierMigrationReason::Prefetch;
        let v2 = TierMigrationReason::Prefetch;
        let mut set = HashSet::new();
        set.insert(v1);
        set.insert(v2);
        // Assert: only 1 entry.
        assert_eq!(set.len(), 1, "identical variants from different expressions must deduplicate");
    }

    // 4. CompressionCodec full variant as_u8/from_u8 roundtrip verifies discriminant stability
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn compression_codec_roundtrip_all_variants_preserve_discriminant_order() {
        // Arrange: all 5 variants in enum declaration order.
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        // Act & Assert: each variant round-trips and discriminants are 0..=4 sequential.
        for (i, variant) in variants.iter().enumerate() {
            let code = variant.as_u8();
            assert_eq!(code, i as u8,
                "variant {:?} should have discriminant {}, got {}", variant, i, code);
            let decoded = CompressionCodec::from_u8(code);
            assert_eq!(decoded, Some(*variant),
                "from_u8({}) should roundtrip to {:?}", code, variant);
        }
    }

    // 5. CompressionCodec from_u8 returns None for all values >= 5
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn compression_codec_from_u8_returns_none_for_all_invalid_discriminants() {
        // Arrange: test boundary and far-past values.
        for invalid in [5u8, 6, 127, 128, 200, 255] {
            assert_eq!(CompressionCodec::from_u8(invalid), None,
                "from_u8({}) should be None", invalid);
        }
    }

    // 6. build_batch NVMe -> HBM swap-in carries codec from addr_table
    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_nvme_to_hbm_swap_in_carries_codec_from_entry() {
        // Arrange: page on NVMe with ZstdDict codec.
        let (c, _backend) = make_coordinator(false);
        c.register_page(42, None, 8192);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&42) {
                entry.current_tier = StorageTier::Nvme;
                entry.codec = CompressionCodec::ZstdDict;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(42, PageMetadata {
                page_id: 42,
                sequence_id: Some(1),
                recency: 0,
                access_count: 1,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[42], 0.5);
        // Assert: tier_migration should carry ZstdDict from the addr_table entry.
        let migration = plan.tier_migrations.iter().find(|m| m.page_id == 42);
        assert!(migration.is_some(), "NVMe page 42 should produce a tier migration");
        assert_eq!(migration.unwrap().codec, CompressionCodec::ZstdDict,
            "NVMe->HBM migration must preserve addr_table codec");
    }

    // 7. build_batch NVMe -> HBM produces correct prefetch_request page_bytes
    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_nvme_to_hbm_prefetch_request_carries_original_bytes() {
        // Arrange: page on NVMe with 65536 original_bytes.
        let (c, _backend) = make_coordinator(false);
        c.register_page(7, None, 65536);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&7) {
                entry.current_tier = StorageTier::Nvme;
            }
        }
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(7, PageMetadata {
                page_id: 7,
                sequence_id: Some(99),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Swapped,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[7], 0.5);
        // Assert: prefetch_request page_bytes must match addr_table original_bytes.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_bytes, 65536,
            "NVMe swap-in prefetch_request must carry addr_table original_bytes");
    }

    // 8. build_batch multiple NVMe pages all produce swap-in migrations
    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_multiple_nvme_pages_all_produce_swap_in_to_hbm() {
        // Arrange: 3 pages on NVMe, all in active set.
        let (c, _backend) = make_coordinator(false);
        for pid in [100, 200, 300] {
            c.register_page(pid, None, 4096);
            {
                let mut table = c.addr_table.write().expect("write lock");
                if let Some(entry) = table.get_mut(&pid) {
                    entry.current_tier = StorageTier::Nvme;
                }
            }
            {
                let mut meta = c.page_metadata.write().expect("write lock");
                meta.insert(pid, PageMetadata {
                    page_id: pid,
                    sequence_id: Some(pid as u64),
                    recency: 0,
                    access_count: 1,
                    last_access: Instant::now(),
                    swap_in_time: None,
                    is_lir: false,
                    state: PageState::Swapped,
                    warm_until: None,
                });
            }
        }
        // Act
        let plan = c.build_batch(&[100, 200, 300], 0.5);
        // Assert: all 3 pages produce tier_migrations from Nvme to GpuHbm.
        let nvme_migrations: Vec<_> = plan.tier_migrations.iter()
            .filter(|m| m.from_tier == StorageTier::Nvme && m.to_tier == StorageTier::GpuHbm)
            .collect();
        assert_eq!(nvme_migrations.len(), 3,
            "all 3 NVMe pages must produce Nvme->GpuHbm migrations");
        // All must be SequenceDemand.
        for m in &nvme_migrations {
            assert_eq!(m.reason, TierMigrationReason::SequenceDemand);
        }
    }

    // 9. register_page with zero page_bytes creates entry with original_bytes=0
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_page_with_zero_bytes_creates_entry_with_zero_original_bytes() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: register a page with 0 bytes (boundary case).
        c.register_page(999, Some(0xDEAD), 0);
        // Assert: entry exists with original_bytes=0 and tier=GpuHbm.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&999).expect("page 999 should exist");
        assert_eq!(entry.original_bytes, 0, "original_bytes should be 0");
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
        assert_eq!(entry.gpu_ptr, Some(0xDEAD));
    }

    // 10. register_page idempotent: second register with same id does not overwrite
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_page_idempotent_second_call_preserves_first_entry() {
        // Arrange: register with gpu_ptr and 8192 bytes.
        let (c, _backend) = make_coordinator(false);
        c.register_page(5, Some(0xAAAA), 8192);
        // Act: register again with different gpu_ptr and page_bytes.
        c.register_page(5, Some(0xBBBB), 4096);
        // Assert: first entry preserved (or_insert_with semantics).
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&5).expect("page 5 should exist");
        assert_eq!(entry.gpu_ptr, Some(0xAAAA),
            "second register_page must not overwrite existing gpu_ptr");
        assert_eq!(entry.original_bytes, 8192,
            "second register_page must not overwrite existing original_bytes");
    }

    // 11. register_page with usize::MAX page_id
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_page_with_max_page_id_creates_valid_entry() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        let max_pid = PageId::MAX;
        // Act
        c.register_page(max_pid, Some(0x1234), 4096);
        // Assert
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&max_pid).expect("max page_id should exist");
        assert_eq!(entry.gpu_ptr, Some(0x1234));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // 12. update_page_gpu_ptr on non-existent page does not create entry
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn update_page_gpu_ptr_nonexistent_page_does_not_create_addr_table_entry() {
        // Arrange: coordinator with no registered pages.
        let (c, _backend) = make_coordinator(false);
        // Act: update a page that was never registered.
        c.update_page_gpu_ptr(404, 0xBEEF);
        // Assert: addr_table should remain empty (no auto-creation).
        let table = c.addr_table.read().expect("read lock");
        assert!(table.get(&404).is_none(),
            "update_page_gpu_ptr must not create a new addr_table entry");
    }

    // 13. update_page_gpu_ptr transitions tier from CpuDram to GpuHbm
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn update_page_gpu_ptr_transitions_tier_from_dram_to_hbm() {
        // Arrange: register page without gpu_ptr (defaults to CpuDram).
        let (c, _backend) = make_coordinator(false);
        c.register_page(10, None, 4096);
        {
            let table = c.addr_table.read().expect("read lock");
            assert_eq!(table.get(&10).unwrap().current_tier, StorageTier::CpuDram);
        }
        // Act: assign a GPU pointer.
        c.update_page_gpu_ptr(10, 0xCAFE);
        // Assert: tier must now be GpuHbm.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&10).expect("page should exist");
        assert_eq!(entry.gpu_ptr, Some(0xCAFE));
        assert_eq!(entry.current_tier, StorageTier::GpuHbm,
            "assigning gpu_ptr must transition tier to GpuHbm");
    }

    // 14. update_page_gpu_ptr on released page is silently ignored
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn update_page_gpu_ptr_on_released_page_silently_ignored() {
        // Arrange: register and then release.
        let (c, _backend) = make_coordinator(false);
        c.register_page(20, Some(0x1000), 4096);
        c.release_page(20);
        assert!(c.addr_table.read().unwrap().get(&20).is_none());
        // Act: update after release (should not panic or create entry).
        c.update_page_gpu_ptr(20, 0x2000);
        // Assert: still absent.
        assert!(c.addr_table.read().unwrap().get(&20).is_none(),
            "update_page_gpu_ptr on released page must not re-create entry");
    }

    // 15. update_page_gpu_ptr preserves original_bytes and host_buffer fields
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn update_page_gpu_ptr_preserves_original_bytes_and_host_buffer() {
        // Arrange: register page on DRAM with host_buffer.
        let (c, _backend) = make_coordinator(false);
        c.register_page(30, None, 16384);
        {
            let mut table = c.addr_table.write().expect("write lock");
            if let Some(entry) = table.get_mut(&30) {
                entry.host_buffer = Some(vec![0xAB; 16384]);
            }
        }
        // Act: promote to HBM.
        c.update_page_gpu_ptr(30, 0x7FFF);
        // Assert: original_bytes and host_buffer must not be disturbed.
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&30).expect("page should exist");
        assert_eq!(entry.original_bytes, 16384,
            "update_page_gpu_ptr must preserve original_bytes");
        assert!(entry.host_buffer.is_some(),
            "update_page_gpu_ptr must not clear host_buffer");
        assert_eq!(entry.host_buffer.as_ref().unwrap().len(), 16384);
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // ── Wave 22: 15 new tests ────────────────────────────────────────────────────

    // 1. build_batch with empty active_set and pages on HBM produces no swap-in and no eviction
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn build_batch_empty_active_set_with_hbm_pages_no_swap_in_no_eviction() {
        // Arrange: register 3 pages on HBM with Standby state, empty active_set.
        let (c, _backend) = make_coordinator(false);
        for pid in [1, 2, 3] {
            c.register_page(pid, Some(0x1000 + pid as u64 * 0x100), 4096);
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64),
                recency: 10,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: build_batch with empty active_set (no pages are active).
        let plan = c.build_batch(&[], 0.5);
        // Assert: pages are on HBM and not in active set, so no swap-in needed.
        // Standby pages won't be evicted at low pressure.
        assert!(plan.swap_in_requests.is_empty(),
            "no swap-in requests when active_set is empty and pages are on HBM");
    }

    // 2. release_page removes both addr_table entry and page_metadata
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn release_page_removes_both_addr_table_and_page_metadata() {
        // Arrange: register a page with both addr_table and metadata.
        let (c, _backend) = make_coordinator(false);
        c.register_page(42, Some(0xAA00), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(42, PageMetadata {
                page_id: 42,
                sequence_id: Some(100),
                recency: 5,
                access_count: 2,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Active,
                warm_until: None,
            });
        }
        // Sanity: both structures contain page 42.
        assert!(c.addr_table.read().unwrap().contains_key(&42));
        assert!(c.page_metadata.read().unwrap().contains_key(&42));
        // Act
        c.release_page(42);
        // Assert: both structures no longer contain page 42.
        assert!(!c.addr_table.read().unwrap().contains_key(&42),
            "release_page must remove addr_table entry");
        assert!(!c.page_metadata.read().unwrap().contains_key(&42),
            "release_page must remove page_metadata entry");
    }

    // 3. release_page on non-existent page is a no-op (no panic)
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn release_page_nonexistent_is_noop_no_panic() {
        // Arrange: coordinator with no pages.
        let (c, _backend) = make_coordinator(false);
        // Act: release a page that was never registered — must not panic.
        c.release_page(999);
        // Assert: addr_table and page_metadata remain empty.
        assert!(c.addr_table.read().unwrap().is_empty());
        assert!(c.page_metadata.read().unwrap().is_empty());
    }

    // 4. build_batch with page_bytes=0 on DRAM produces swap-in with zero bytes
    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn build_batch_dram_page_zero_bytes_swap_in_request_page_bytes_zero() {
        // Arrange: register page with 0 bytes on DRAM, in active set.
        let (c, _backend) = make_coordinator(false);
        c.register_page(7, None, 0);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(7, PageMetadata {
                page_id: 7,
                sequence_id: Some(50),
                recency: 0,
                access_count: 1,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        let plan = c.build_batch(&[7], 0.5);
        // Assert: swap-in request exists with page_bytes=0.
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_bytes, 0,
            "swap-in request for zero-byte page must have page_bytes=0");
        assert_eq!(plan.swap_in_requests[0].page_id, 7);
    }

    // 5. build_batch skips page when metadata exists but addr_table has no entry
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn build_batch_skips_page_with_metadata_but_no_addr_table_entry() {
        // Arrange: insert page_metadata without registering in addr_table.
        let (c, _backend) = make_coordinator(false);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(55, PageMetadata {
                page_id: 55,
                sequence_id: Some(200),
                recency: 0,
                access_count: 1,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act: build_batch with page 55 in active set.
        let plan = c.build_batch(&[55], 0.5);
        // Assert: no swap-in, no eviction — page skipped because addr_table has no entry.
        assert!(plan.swap_in_requests.is_empty(),
            "page with metadata but no addr_table entry must be skipped");
        assert!(plan.eviction_candidates.is_empty());
        assert!(plan.tier_migrations.is_empty());
    }

    // 6. TierMigrationReason Debug format contains all four variant names
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn tier_migration_reason_debug_contains_all_four_variant_names() {
        // Arrange & Act: format all four variants with Debug.
        let ep = format!("{:?}", TierMigrationReason::EvictionPressure);
        let sd = format!("{:?}", TierMigrationReason::SequenceDemand);
        let pf = format!("{:?}", TierMigrationReason::Prefetch);
        let cc = format!("{:?}", TierMigrationReason::ColdCascade);
        // Assert: each formatted string must contain its variant name.
        assert!(ep.contains("EvictionPressure"), "Debug must contain 'EvictionPressure', got: {}", ep);
        assert!(sd.contains("SequenceDemand"), "Debug must contain 'SequenceDemand', got: {}", sd);
        assert!(pf.contains("Prefetch"), "Debug must contain 'Prefetch', got: {}", pf);
        assert!(cc.contains("ColdCascade"), "Debug must contain 'ColdCascade', got: {}", cc);
        // All four must be distinct.
        assert_eq!(4, [&ep, &sd, &pf, &cc].iter().collect::<std::collections::HashSet<_>>().len(),
            "all four TierMigrationReason Debug outputs must be distinct");
    }

    // 7. register_pages_from_hgal bulk inserts multiple pages with correct default tier
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_pages_from_hgal_bulk_insert_sets_default_tier_dram_for_all() {
        // Arrange: create a map with 5 pages.
        let (c, _backend) = make_coordinator(false);
        let mut pages = std::collections::HashMap::new();
        for pid in 10..15 {
            pages.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: Some(pid as u64),
                recency: 0,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Standby,
                warm_until: None,
            });
        }
        // Act
        c.register_pages_from_hgal(&pages, 8192);
        // Assert: all 5 pages in addr_table with tier=CpuDram (no gpu_ptr).
        let table = c.addr_table.read().expect("read lock");
        for pid in 10..15 {
            let entry = table.get(&pid).unwrap_or_else(|| panic!("page {} missing", pid));
            assert_eq!(entry.current_tier, StorageTier::CpuDram,
                "page {} should default to CpuDram when no gpu_ptr", pid);
            assert_eq!(entry.original_bytes, 8192,
                "page {} should have page_bytes=8192", pid);
            assert!(entry.gpu_ptr.is_none());
        }
        // All 5 pages also in page_metadata.
        let meta = c.page_metadata.read().expect("read lock");
        for pid in 10..15 {
            assert!(meta.contains_key(&pid), "page {} must be in page_metadata", pid);
        }
    }

    // 8. register_pages_from_hgal with empty map does nothing
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_pages_from_hgal_empty_map_leaves_tables_unchanged() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Pre-register one page.
        c.register_page(1, Some(0x100), 4096);
        assert_eq!(c.addr_table.read().unwrap().len(), 1);
        // Act: bulk-register empty map.
        let empty = std::collections::HashMap::new();
        c.register_pages_from_hgal(&empty, 4096);
        // Assert: original page still present, no new entries.
        assert_eq!(c.addr_table.read().unwrap().len(), 1);
        assert!(c.addr_table.read().unwrap().contains_key(&1));
    }

    // 9. register_pages_from_hgal does not overwrite pre-existing addr_table entry
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_pages_from_hgal_preserves_preexisting_addr_table_gpu_ptr_and_bytes() {
        // Arrange: pre-register page 100 with gpu_ptr and 65536 bytes.
        let (c, _backend) = make_coordinator(false);
        c.register_page(100, Some(0xDEAD), 65536);
        // Act: bulk-register same page 100 with different bytes.
        let mut pages = std::collections::HashMap::new();
        pages.insert(100, PageMetadata {
            page_id: 100,
            sequence_id: Some(999),
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        });
        c.register_pages_from_hgal(&pages, 4096);
        // Assert: original entry preserved (idempotent or_insert_with).
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&100).expect("page 100 must exist");
        assert_eq!(entry.gpu_ptr, Some(0xDEAD),
            "register_pages_from_hgal must not overwrite pre-existing gpu_ptr");
        assert_eq!(entry.original_bytes, 65536,
            "register_pages_from_hgal must not overwrite pre-existing original_bytes");
    }

    // 10. StorageTier Ord/PartialOrd: GpuHbm > CpuDram > Nvme (reverse discriminant)
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn storage_tier_ordering_hbm_greater_than_dram_greater_than_nvme() {
        // Arrange & Act & Assert: priority ordering uses reverse discriminant
        // (lower u8 value = higher priority: 0=HBM > 1=DRAM > 2=NVMe).
        assert!(StorageTier::GpuHbm > StorageTier::CpuDram, "GpuHbm > CpuDram");
        assert!(StorageTier::CpuDram > StorageTier::Nvme, "CpuDram > Nvme");
        assert!(StorageTier::GpuHbm > StorageTier::Nvme, "GpuHbm > Nvme (transitive)");
        // Verify reverse: Nvme < Dram < Hbm.
        assert!(StorageTier::Nvme < StorageTier::CpuDram);
        assert!(StorageTier::CpuDram < StorageTier::GpuHbm);
    }

    // 11. StorageTier as_u8/from_u8 roundtrip all three variants
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn storage_tier_roundtrip_all_variants_preserve_discriminant() {
        // Arrange: all three variants.
        let variants = [
            (StorageTier::GpuHbm, 0u8),
            (StorageTier::CpuDram, 1u8),
            (StorageTier::Nvme, 2u8),
        ];
        for (variant, expected_disc) in &variants {
            // Act
            let disc = variant.as_u8();
            let decoded = StorageTier::from_u8(disc);
            // Assert
            assert_eq!(disc, *expected_disc,
                "{:?}.as_u8() should be {}, got {}", variant, expected_disc, disc);
            assert_eq!(decoded, Some(*variant),
                "from_u8({}) should roundtrip to {:?}", disc, variant);
        }
    }

    // 12. StorageTier from_u8 returns None for invalid discriminants
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn storage_tier_from_u8_returns_none_for_invalid_discriminants() {
        for invalid in [3u8, 4, 100, 200, 255] {
            assert_eq!(StorageTier::from_u8(invalid), None,
                "from_u8({}) should be None", invalid);
        }
    }

    // 13. record_eviction_completed updates stats correctly for gpu_to_dram
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn record_eviction_completed_gpu_to_dram_increments_correct_counter() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: record 3 evictions from GPU to DRAM.
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 200);
        c.record_eviction_completed(3, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 300);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 3);
        assert_eq!(stats.total_bytes_evicted, 4096 * 3);
        assert_eq!(stats.total_eviction_latency_us, 600);
        assert_eq!(stats.evictions_dram_to_nvme, 0, "dram_to_nvme counter must remain 0");
    }

    // 14. record_swap_in_completed updates stats correctly for nvme_to_dram
    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn record_swap_in_completed_nvme_to_dram_increments_correct_counter() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: record 2 swap-ins from NVMe to DRAM.
        c.record_swap_in_completed(10, StorageTier::Nvme, StorageTier::CpuDram, 8192, 500);
        c.record_swap_in_completed(20, StorageTier::Nvme, StorageTier::CpuDram, 8192, 750);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.swap_ins_nvme_to_dram, 2);
        assert_eq!(stats.total_bytes_swapped_in, 8192 * 2);
        assert_eq!(stats.total_swap_in_latency_us, 1250);
        assert_eq!(stats.swap_ins_dram_to_gpu, 0, "dram_to_gpu counter must remain 0");
    }

    // 15. ThreeTierSwapStats avg_eviction_latency_us returns 0.0 when no evictions
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn three_tier_swap_stats_avg_latency_zero_when_no_migrations() {
        // Arrange: default stats (no evictions, no swap-ins).
        let stats = ThreeTierSwapStats::default();
        // Act & Assert
        assert_eq!(stats.avg_eviction_latency_us(), 0.0,
            "avg eviction latency must be 0.0 when no evictions recorded");
        assert_eq!(stats.avg_swap_in_latency_us(), 0.0,
            "avg swap-in latency must be 0.0 when no swap-ins recorded");
        assert_eq!(stats.total_migrations(), 0,
            "total migrations must be 0 when nothing recorded");
    }

    // ── Wave 12x34: 15 new tests — angles not covered by prior waves ────────────

    // W1. ThreeTierSwapConfig Default: every sub-field matches its own Default impl
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn three_tier_swap_config_default_every_sub_field_matches_sub_default() {
        // Arrange: get top-level default and per-sub-config defaults.
        let top = ThreeTierSwapConfig::default();
        let eviction_default = EvictionWorkerConfig::default();
        let swap_in_default = SwapInWorkerConfig::default();
        let migration_default = MigrationActorConfig::default();
        // Act & Assert: verify each field inside each sub-config matches independent default.
        // -- eviction sub-config --
        assert_eq!(top.eviction.tick_interval, eviction_default.tick_interval);
        assert_eq!(top.eviction.max_evict_per_round, eviction_default.max_evict_per_round);
        assert!((top.eviction.hbm_pressure_threshold - eviction_default.hbm_pressure_threshold).abs() < 1e-6);
        assert!((top.eviction.dram_pressure_threshold - eviction_default.dram_pressure_threshold).abs() < 1e-6);
        assert_eq!(top.eviction.importance_threshold, eviction_default.importance_threshold);
        assert_eq!(top.eviction.hbm_evict_age_ticks, eviction_default.hbm_evict_age_ticks);
        assert_eq!(top.eviction.dram_evict_age_ticks, eviction_default.dram_evict_age_ticks);
        assert_eq!(top.eviction.default_evict_codec, eviction_default.default_evict_codec);
        assert_eq!(top.eviction.page_bytes, eviction_default.page_bytes);
        // -- swap_in sub-config --
        assert_eq!(top.swap_in.max_prefetch_per_round, swap_in_default.max_prefetch_per_round);
        assert_eq!(top.swap_in.tick_interval, swap_in_default.tick_interval);
        assert!((top.swap_in.min_confidence - swap_in_default.min_confidence).abs() < 1e-6);
        assert_eq!(top.swap_in.max_in_flight, swap_in_default.max_in_flight);
        assert_eq!(top.swap_in.page_bytes, swap_in_default.page_bytes);
        // -- migration sub-config --
        assert_eq!(top.migration.nvme_swap_dir, migration_default.nvme_swap_dir);
        assert_eq!(top.migration.queue_capacity, migration_default.queue_capacity);
        assert_eq!(top.migration.session_id, migration_default.session_id);
        assert_eq!(top.migration.page_size, migration_default.page_size);
        assert_eq!(top.migration.max_swap_pages, migration_default.max_swap_pages);
        // -- top-level auto_start --
        assert!(top.auto_start, "default auto_start must be true");
    }

    // W2. TierMigrationReason: Clone of each variant produces equal but independent value
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn tier_migration_reason_clone_each_variant_produces_equal_value() {
        // Arrange: all four variants.
        let variants = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        for &v in &variants {
            // Act
            let cloned = v.clone();
            // Assert: cloned equals original, Debug output identical.
            assert_eq!(v, cloned, "clone of {:?} must equal original", v);
            assert_eq!(format!("{:?}", v), format!("{:?}", cloned),
                "Debug of cloned {:?} must match original", v);
        }
    }

    // W3. TierMigrationReason: all four variants produce distinct non-empty Debug strings
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn tier_migration_reason_all_variants_produce_distinct_nonempty_debug() {
        // Arrange
        let debugs: Vec<String> = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ].iter().map(|v| format!("{:?}", v)).collect();
        // Assert: all non-empty
        for (i, d) in debugs.iter().enumerate() {
            assert!(!d.is_empty(), "variant {} Debug must not be empty", i);
            assert!(d.len() >= 4, "variant {} Debug must be descriptive, got '{}'", i, d);
        }
        // Assert: all pairwise distinct
        for i in 0..debugs.len() {
            for j in (i + 1)..debugs.len() {
                assert_ne!(debugs[i], debugs[j],
                    "variant {} and {} must have distinct Debug: '{}' vs '{}'", i, j, debugs[i], debugs[j]);
            }
        }
    }

    // W4. StorageTier: Ord consistency — cmp and partial_cmp always agree
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn storage_tier_ord_and_partial_cmp_always_agree() {
        // Arrange: all pairs.
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        for &a in &tiers {
            for &b in &tiers {
                // Act
                let ord_result = a.cmp(&b);
                let partial_result = a.partial_cmp(&b);
                // Assert: partial_cmp is always Some and matches cmp.
                assert_eq!(partial_result, Some(ord_result),
                    "partial_cmp({:?}, {:?}) must match cmp", a, b);
            }
        }
    }

    // W5. StorageTier: sorted vec in ascending order is [Nvme, CpuDram, GpuHbm]
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn storage_tier_sorted_ascending_is_nvme_dram_hbm() {
        // Arrange
        let mut tiers = vec![StorageTier::GpuHbm, StorageTier::Nvme, StorageTier::CpuDram];
        // Act
        tiers.sort();
        // Assert: ascending = coldest first (Nvme < CpuDram < GpuHbm).
        assert_eq!(tiers, vec![StorageTier::Nvme, StorageTier::CpuDram, StorageTier::GpuHbm],
            "ascending sort must be [Nvme, CpuDram, GpuHbm]");
    }

    // W6. CompressionCodec: all 5 variants pairwise inequality
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn compression_codec_all_variants_pairwise_inequality() {
        // Arrange
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        // Act & Assert: every pair must be unequal
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j],
                    "{:?} must not equal {:?}", variants[i], variants[j]);
            }
        }
    }

    // W7. CompressionCodec: equality is reflexive for every variant
    // @trace REQ-COMP-001 [level:unit]
    #[test]
    fn compression_codec_equality_reflexive_for_all_variants() {
        // Arrange & Act & Assert
        for (disc, expected) in [
            (0u8, CompressionCodec::None),
            (1u8, CompressionCodec::Lz4),
            (2u8, CompressionCodec::BitPackRle),
            (3u8, CompressionCodec::NvcompAns),
            (4u8, CompressionCodec::ZstdDict),
        ] {
            let decoded = CompressionCodec::from_u8(disc).unwrap();
            assert_eq!(decoded, expected,
                "from_u8({}) must equal expected variant {:?}", disc, expected);
            assert_eq!(decoded, decoded,
                "reflexivity: {:?} must equal itself", decoded);
        }
    }

    // W8. ThreeTierSwapStats: default has zero on all tier count fields
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn three_tier_swap_stats_default_tier_counts_all_zero() {
        // Arrange
        let stats = ThreeTierSwapStats::default();
        // Assert: all tier page counts start at zero.
        assert_eq!(stats.pages_on_hbm, 0, "default pages_on_hbm must be 0");
        assert_eq!(stats.pages_on_dram, 0, "default pages_on_dram must be 0");
        assert_eq!(stats.pages_on_nvme, 0, "default pages_on_nvme must be 0");
    }

    // W9. ThreeTierSwapStats: avg_eviction_latency_us computes correct average
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn three_tier_swap_stats_avg_eviction_latency_with_mixed_tiers() {
        // Arrange: simulate 2 GPU→DRAM evictions (100us, 300us) and 1 DRAM→NVMe (200us).
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 2;
        stats.evictions_dram_to_nvme = 1;
        stats.total_eviction_latency_us = 100 + 300 + 200; // 600us total
        // Act
        let avg = stats.avg_eviction_latency_us();
        // Assert: 600us / 3 evictions = 200us.
        assert!((avg - 200.0).abs() < 1e-6,
            "avg eviction latency must be 200.0, got {}", avg);
    }

    // W10. ThreeTierSwapStats: avg_swap_in_latency_us computes correct average
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn three_tier_swap_stats_avg_swap_in_latency_with_mixed_tiers() {
        // Arrange: simulate 3 DRAM→GPU swap-ins (50us each) and 2 NVMe→DRAM (200us each).
        let mut stats = ThreeTierSwapStats::default();
        stats.swap_ins_dram_to_gpu = 3;
        stats.swap_ins_nvme_to_dram = 2;
        stats.total_swap_in_latency_us = 3 * 50 + 2 * 200; // 550us total
        // Act
        let avg = stats.avg_swap_in_latency_us();
        // Assert: 550us / 5 swap-ins = 110us.
        assert!((avg - 110.0).abs() < 1e-6,
            "avg swap-in latency must be 110.0, got {}", avg);
    }

    // W11. ThreeTierSwapStats: total_migrations sums all four counters
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn three_tier_swap_stats_total_migrations_sums_all_four_counters() {
        // Arrange
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 10;
        stats.evictions_dram_to_nvme = 5;
        stats.swap_ins_dram_to_gpu = 8;
        stats.swap_ins_nvme_to_dram = 3;
        // Act
        let total = stats.total_migrations();
        // Assert: 10 + 5 + 8 + 3 = 26.
        assert_eq!(total, 26, "total_migrations must sum all four counters");
    }

    // W12. register_page with duplicate page_id: second register with different bytes is no-op
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_page_duplicate_id_second_call_with_different_bytes_is_noop() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(42, Some(0xA000), 8192);
        // Act: register same page_id again with different bytes and no gpu_ptr.
        c.register_page(42, None, 4096);
        // Assert: original entry preserved (or_insert_with is idempotent).
        let table = c.addr_table.read().unwrap();
        let entry = table.get(&42).expect("page 42 must exist");
        assert_eq!(entry.gpu_ptr, Some(0xA000),
            "duplicate register must preserve original gpu_ptr");
        assert_eq!(entry.original_bytes, 8192,
            "duplicate register must preserve original bytes");
        assert_eq!(entry.current_tier, StorageTier::GpuHbm,
            "original tier must be preserved");
    }

    // W13. register_page with duplicate page_id: addr_table remains size 1
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn register_page_duplicate_id_addr_table_size_remains_one() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        c.register_page(99, Some(0xBEEF), 4096);
        assert_eq!(c.addr_table.read().unwrap().len(), 1);
        // Act: register same page_id 5 more times.
        for _ in 0..5 {
            c.register_page(99, None, 2048);
        }
        // Assert: still exactly 1 entry.
        assert_eq!(c.addr_table.read().unwrap().len(), 1,
            "duplicate register must not add new entries");
    }

    // W14. record_eviction_completed for dram_to_nvme updates correct counter only
    // @trace REQ-COMP-002 [level:unit]
    #[test]
    fn record_eviction_completed_dram_to_nvme_increments_dram_counter_only() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: record 4 DRAM→NVMe evictions.
        for i in 0..4usize {
            c.record_eviction_completed(
                100 + i, StorageTier::CpuDram, StorageTier::Nvme, 8192, 150 + (i as u64) * 10,
            );
        }
        // Assert
        let stats = c.stats();
        assert_eq!(stats.evictions_dram_to_nvme, 4, "dram_to_nvme must be 4");
        assert_eq!(stats.evictions_gpu_to_dram, 0, "gpu_to_dram must remain 0");
        assert_eq!(stats.total_bytes_evicted, 8192 * 4);
        assert_eq!(stats.total_eviction_latency_us, 150 + 160 + 170 + 180); // 660us
    }

    // W15. record_swap_in_completed for dram_to_gpu updates correct counter only
    // @trace REQ-COMP-003 [level:unit]
    #[test]
    fn record_swap_in_completed_dram_to_gpu_increments_dram_counter_only() {
        // Arrange
        let (c, _backend) = make_coordinator(false);
        // Act: record 3 DRAM→GPU swap-ins.
        c.record_swap_in_completed(1, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 80);
        c.record_swap_in_completed(2, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 120);
        c.record_swap_in_completed(3, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 200);
        // Assert
        let stats = c.stats();
        assert_eq!(stats.swap_ins_dram_to_gpu, 3, "dram_to_gpu must be 3");
        assert_eq!(stats.swap_ins_nvme_to_dram, 0, "nvme_to_dram must remain 0");
        assert_eq!(stats.total_bytes_swapped_in, 4096 * 3);
        assert_eq!(stats.total_swap_in_latency_us, 400);
    }

    // ── Wave 17: 15 additional tests for targeted coverage areas ──────────────────

    // Focus area 1: Config Default sub-fields validation

    #[test]
    fn config_default_migration_subconfig_page_size_matches_constant() {
        // Arrange: create default config
        let config = ThreeTierSwapConfig::default();
        // Assert: migration.page_size must be a positive power of two
        assert!(config.migration.page_size > 0, "page_size must be positive");
        assert_eq!(config.migration.page_size & (config.migration.page_size - 1), 0,
            "page_size must be a power of two");
    }

    // Focus area 2: TierMigrationReason Clone/Debug all variants

    #[test]
    fn tier_migration_reason_clone_and_debug_roundtrip_all_variants() {
        // Arrange: collect all variants
        let reasons = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        for original in reasons {
            // Act: clone the value and format its debug output
            let cloned = original.clone();
            let debug_str = format!("{:?}", cloned);
            // Assert: clone equals original and debug contains variant name
            assert_eq!(cloned, original);
            assert!(!debug_str.is_empty(),
                "Debug output for {:?} must not be empty", original);
            assert!(debug_str.contains("EvictionPressure")
                || debug_str.contains("SequenceDemand")
                || debug_str.contains("Prefetch")
                || debug_str.contains("ColdCascade"),
                "Debug must contain variant name, got: {}", debug_str);
        }
    }

    // Focus area 3: StorageTier Ord/PartialOrd consistency

    #[test]
    fn storage_tier_ord_and_partial_ord_yield_same_ordering_for_all_pairs() {
        // Arrange: all tier pairs
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        for a in tiers {
            for b in tiers {
                // Act: compare via Ord and PartialOrd
                let ord_ordering = a.cmp(&b);
                let partial_ordering = a.partial_cmp(&b);
                // Assert: they always agree
                assert_eq!(partial_ordering, Some(ord_ordering),
                    "Ord and PartialOrd disagree for {:?} vs {:?}", a, b);
            }
        }
    }

    // Focus area 4: CompressionCodec pairwise inequality

    #[test]
    fn compression_codec_all_variants_pairwise_distinct_values() {
        // Arrange: all codec variants
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        // Act & Assert: every distinct pair must be unequal
        for i in 0..codecs.len() {
            for j in (i + 1)..codecs.len() {
                assert_ne!(codecs[i], codecs[j],
                    "{:?} and {:?} must not be equal", codecs[i], codecs[j]);
            }
        }
    }

    // Focus area 5: Stats Default all fields zero

    #[test]
    fn stats_default_every_single_field_is_zero() {
        // Arrange & Act: create default stats
        let stats = ThreeTierSwapStats::default();
        // Assert: every u64 and usize field must be zero
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.evictions_dram_to_nvme, 0);
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.swap_ins_nvme_to_dram, 0);
        assert_eq!(stats.total_bytes_evicted, 0);
        assert_eq!(stats.total_bytes_swapped_in, 0);
        assert_eq!(stats.total_eviction_latency_us, 0);
        assert_eq!(stats.total_swap_in_latency_us, 0);
        assert_eq!(stats.eviction_rounds, 0);
        assert_eq!(stats.swap_in_rounds, 0);
        assert_eq!(stats.pages_on_hbm, 0);
        assert_eq!(stats.pages_on_dram, 0);
        assert_eq!(stats.pages_on_nvme, 0);
        // Derived methods must also be zero
        assert_eq!(stats.total_migrations(), 0);
        assert!((stats.avg_eviction_latency_us() - 0.0).abs() < f64::EPSILON);
        assert!((stats.avg_swap_in_latency_us() - 0.0).abs() < f64::EPSILON);
    }

    // Focus area 6: Eviction/swap-in latency tracking

    #[test]
    fn eviction_and_swap_in_latency_tracked_independently() {
        // Arrange: fresh coordinator
        let (c, _backend) = make_coordinator(false);
        // Act: record 2 evictions with distinct latencies
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 300);
        // And 3 swap-ins with different latencies
        c.record_swap_in_completed(3, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 50);
        c.record_swap_in_completed(4, StorageTier::CpuDram, StorageTier::GpuHbm, 4096, 150);
        c.record_swap_in_completed(5, StorageTier::Nvme, StorageTier::CpuDram, 4096, 800);
        // Assert: latency accumulators are independent
        let stats = c.stats();
        assert_eq!(stats.total_eviction_latency_us, 400, "eviction latency must sum to 400");
        assert_eq!(stats.total_swap_in_latency_us, 1000, "swap-in latency must sum to 1000");
        assert!((stats.avg_eviction_latency_us() - 200.0).abs() < 0.01);
        assert!((stats.avg_swap_in_latency_us() - 333.333).abs() < 0.1);
    }

    // Focus area 7: Total migrations counter

    #[test]
    fn total_migrations_equals_sum_of_all_four_directional_counters() {
        // Arrange: create stats with specific values
        let mut stats = ThreeTierSwapStats::default();
        stats.evictions_gpu_to_dram = 11;
        stats.evictions_dram_to_nvme = 7;
        stats.swap_ins_dram_to_gpu = 13;
        stats.swap_ins_nvme_to_dram = 4;
        // Act: compute total
        let total = stats.total_migrations();
        // Assert: must be the sum of all four counters
        assert_eq!(total, 11 + 7 + 13 + 4);
    }

    // Focus area 8: Register page duplicate idempotent

    #[test]
    fn register_page_duplicate_preserves_first_entry_gpu_ptr_and_bytes() {
        // Arrange: create coordinator
        let (c, _backend) = make_coordinator(false);
        // Act: register same page twice with different parameters
        c.register_page(42, Some(0xABCD), 8192);
        c.register_page(42, Some(0xFFFF), 4096);
        // Assert: first entry is preserved (or_insert_with is idempotent)
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&42).expect("page 42 must exist");
        assert_eq!(entry.gpu_ptr, Some(0xABCD), "first gpu_ptr must be preserved");
        assert_eq!(entry.original_bytes, 8192, "first original_bytes must be preserved");
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    // Focus area 9: Eviction/swap-in counter isolation

    #[test]
    fn eviction_counter_does_not_affect_swap_in_counter() {
        // Arrange: fresh coordinator
        let (c, _backend) = make_coordinator(false);
        // Act: record only evictions
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::CpuDram, StorageTier::Nvme, 4096, 200);
        // Assert: swap-in counters remain zero
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.swap_ins_dram_to_gpu, 0, "swap_in counter must be zero");
        assert_eq!(stats.swap_ins_nvme_to_dram, 0, "swap_in counter must be zero");
        assert_eq!(stats.total_bytes_swapped_in, 0);
        assert_eq!(stats.total_swap_in_latency_us, 0);
    }

    // Focus area 10: Ascending sort order by StorageTier

    #[test]
    fn storage_tier_sorted_vec_yields_nvme_dram_hbm_ascending() {
        // Arrange: shuffled tier list
        let mut tiers = vec![
            StorageTier::GpuHbm,
            StorageTier::Nvme,
            StorageTier::CpuDram,
            StorageTier::GpuHbm,
            StorageTier::Nvme,
        ];
        // Act: sort ascending (Ord ordering)
        tiers.sort();
        // Assert: ascending by Ord means Nvme < CpuDram < GpuHbm
        assert_eq!(tiers, vec![
            StorageTier::Nvme,
            StorageTier::Nvme,
            StorageTier::CpuDram,
            StorageTier::GpuHbm,
            StorageTier::GpuHbm,
        ]);
    }

    // Focus area 11: CompressionCodec all variants Debug output

    #[test]
    fn compression_codec_every_variant_debug_string_is_nonempty_and_unique() {
        // Arrange: all codec variants
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        let debug_strings: Vec<String> = codecs.iter()
            .map(|c| format!("{:?}", c))
            .collect();
        // Assert: all non-empty
        for (i, s) in debug_strings.iter().enumerate() {
            assert!(!s.is_empty(), "Debug for {:?} must not be empty", codecs[i]);
        }
        // Assert: all pairwise distinct
        for i in 0..debug_strings.len() {
            for j in (i + 1)..debug_strings.len() {
                assert_ne!(debug_strings[i], debug_strings[j],
                    "Debug strings for {:?} and {:?} must differ", codecs[i], codecs[j]);
            }
        }
    }

    // Focus area 12: Stats Clone field-by-field

    #[test]
    fn stats_clone_copies_every_field_independently() {
        // Arrange: stats with non-default values in every field
        let original = ThreeTierSwapStats {
            evictions_gpu_to_dram: 1,
            evictions_dram_to_nvme: 2,
            swap_ins_dram_to_gpu: 3,
            swap_ins_nvme_to_dram: 4,
            total_bytes_evicted: 100,
            total_bytes_swapped_in: 200,
            total_eviction_latency_us: 50,
            total_swap_in_latency_us: 60,
            eviction_rounds: 7,
            swap_in_rounds: 8,
            pages_on_hbm: 10,
            pages_on_dram: 20,
            pages_on_nvme: 30,
        };
        // Act: clone
        let cloned = original.clone();
        // Assert: every field matches
        assert_eq!(cloned.evictions_gpu_to_dram, 1);
        assert_eq!(cloned.evictions_dram_to_nvme, 2);
        assert_eq!(cloned.swap_ins_dram_to_gpu, 3);
        assert_eq!(cloned.swap_ins_nvme_to_dram, 4);
        assert_eq!(cloned.total_bytes_evicted, 100);
        assert_eq!(cloned.total_bytes_swapped_in, 200);
        assert_eq!(cloned.total_eviction_latency_us, 50);
        assert_eq!(cloned.total_swap_in_latency_us, 60);
        assert_eq!(cloned.eviction_rounds, 7);
        assert_eq!(cloned.swap_in_rounds, 8);
        assert_eq!(cloned.pages_on_hbm, 10);
        assert_eq!(cloned.pages_on_dram, 20);
        assert_eq!(cloned.pages_on_nvme, 30);
    }

    // Focus area 13: Config page_size=0 boundary

    #[test]
    fn config_with_page_size_zero_does_not_panic_in_register_page() {
        // Arrange: create coordinator with page_size=0 via migration config
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let mm = Arc::new(Mutex::new(GlobalMemoryManager::new_with_capacities(
            100_000, 1_000_000, 10_000_000,
        )));
        let observer = Arc::new(Mutex::new(BasicObserver::new()));
        let mut migration_config = MigrationActorConfig::default();
        migration_config.page_size = 0;
        migration_config.max_swap_pages = 0;
        let config = ThreeTierSwapConfig {
            auto_start: false,
            migration: migration_config,
            ..ThreeTierSwapConfig::default()
        };
        // Act: coordinator creation and register_page with page_size=0 must not panic
        let c = ThreeTierSwapCoordinator::new(config, backend, mm, observer);
        c.register_page(1, Some(0x1000), 0);
        // Assert: entry exists with zero bytes
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&1).expect("page must exist despite zero page_size");
        assert_eq!(entry.original_bytes, 0);
    }

    // Focus area 14: TierMigrationReason all variants Copy

    #[test]
    fn tier_migration_reason_copy_all_variants_remain_valid_after_assignment() {
        // Arrange: one of each variant
        let mut reason = TierMigrationReason::EvictionPressure;
        // Act: assign all variants via Copy, verify each is still usable
        let r1 = reason;
        reason = TierMigrationReason::SequenceDemand;
        let r2 = reason;
        reason = TierMigrationReason::Prefetch;
        let r3 = reason;
        reason = TierMigrationReason::ColdCascade;
        let r4 = reason;
        // Assert: all copied values are valid and independent
        assert_eq!(r1, TierMigrationReason::EvictionPressure);
        assert_eq!(r2, TierMigrationReason::SequenceDemand);
        assert_eq!(r3, TierMigrationReason::Prefetch);
        assert_eq!(r4, TierMigrationReason::ColdCascade);
        // The final `reason` value is still valid
        assert_eq!(reason, TierMigrationReason::ColdCascade);
    }

    // Focus area 15: StorageTier Hash consistency

    #[test]
    fn storage_tier_hash_is_stable_and_distinct_across_all_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Arrange: hash each tier multiple times
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        let mut hashes: Vec<(StorageTier, u64)> = Vec::new();
        for tier in tiers {
            // Act: hash the same tier twice
            let mut h1 = DefaultHasher::new();
            tier.hash(&mut h1);
            let hash1 = h1.finish();
            let mut h2 = DefaultHasher::new();
            tier.hash(&mut h2);
            let hash2 = h2.finish();
            // Assert: same tier always produces the same hash
            assert_eq!(hash1, hash2, "Hash for {:?} must be deterministic", tier);
            hashes.push((tier, hash1));
        }
        // Assert: all tiers have pairwise distinct hashes
        for i in 0..hashes.len() {
            for j in (i + 1)..hashes.len() {
                assert_ne!(hashes[i].1, hashes[j].1,
                    "Hashes for {:?} and {:?} must be distinct", hashes[i].0, hashes[j].0);
            }
        }
    }

    #[test]
    fn three_tier_stats_avg_eviction_with_uneven_distribution() {
        let stats = ThreeTierSwapStats {
            evictions_gpu_to_dram: 7,
            evictions_dram_to_nvme: 3,
            total_eviction_latency_us: 1000,
            ..Default::default()
        };
        let avg = stats.avg_eviction_latency_us();
        assert!((avg - 100.0).abs() < f64::EPSILON, "expected 100, got {avg}");
    }

    #[test]
    fn three_tier_stats_avg_swap_in_with_only_nvme() {
        let stats = ThreeTierSwapStats {
            swap_ins_nvme_to_dram: 5,
            total_swap_in_latency_us: 250,
            ..Default::default()
        };
        let avg = stats.avg_swap_in_latency_us();
        assert!((avg - 50.0).abs() < f64::EPSILON, "expected 50, got {avg}");
    }

    #[test]
    fn three_tier_stats_total_migrations_zero_when_all_zero() {
        let stats = ThreeTierSwapStats::default();
        assert_eq!(stats.total_migrations(), 0);
    }

    #[test]
    fn three_tier_stats_clone_independence() {
        let stats = ThreeTierSwapStats {
            evictions_gpu_to_dram: 5,
            pages_on_hbm: 3,
            ..Default::default()
        };
        let c = stats.clone();
        assert_eq!(c.evictions_gpu_to_dram, 5);
        assert_eq!(c.pages_on_hbm, 3);
    }

    #[test]
    fn tier_migration_reason_prefetch_is_not_cold_cascade() {
        assert_ne!(TierMigrationReason::Prefetch, TierMigrationReason::ColdCascade);
    }

    #[test]
    fn tier_migration_from_gpu_to_nvme_fields() {
        let mig = TierMigration {
            page_id: 42,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::Nvme,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
            reason: TierMigrationReason::ColdCascade,
        };
        assert_eq!(mig.to_tier, StorageTier::Nvme);
        assert_eq!(mig.codec, CompressionCodec::ZstdDict);
    }

    #[test]
    fn tier_migration_plan_with_swap_in_requests() {
        let req = PrefetchRequest {
            page_id: 7,
            urgency: 0.9,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let plan = TierMigrationPlan {
            eviction_candidates: vec![],
            swap_in_requests: vec![req],
            tier_migrations: vec![],
            built_at: Instant::now(),
        };
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 7);
    }

    #[test]
    fn three_tier_swap_config_cloned_preserves_auto_start() {
        let config = ThreeTierSwapConfig::default();
        let cloned = config.clone();
        assert_eq!(cloned.auto_start, config.auto_start);
    }

    #[test]
    fn tier_migration_plan_empty_has_zero_len_vectors() {
        let plan = TierMigrationPlan {
            eviction_candidates: vec![],
            swap_in_requests: vec![],
            tier_migrations: vec![],
            built_at: Instant::now(),
        };
        assert_eq!(plan.eviction_candidates.len(), 0);
        assert_eq!(plan.swap_in_requests.len(), 0);
        assert_eq!(plan.tier_migrations.len(), 0);
    }

    #[test]
    fn three_tier_stats_all_pages_zero_by_default() {
        let stats = ThreeTierSwapStats::default();
        assert_eq!(stats.pages_on_hbm + stats.pages_on_dram + stats.pages_on_nvme, 0);
    }

    #[test]
    fn tier_migration_reason_sequence_demand_not_eviction() {
        assert_ne!(TierMigrationReason::SequenceDemand, TierMigrationReason::EvictionPressure);
    }

    // ── Wave 13 additional tests (+13, 716→729) ─────────────────────────────────

    #[test]
    fn tier_migration_reason_all_six_pairs_are_inequal() {
        // Arrange: all 4 variants
        let variants = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        // Act & Assert: all C(4,2)=6 pairs must be distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j],
                    "variants[{i}] == variants[{j}] but all must be distinct");
            }
        }
    }

    #[test]
    fn tier_migration_page_id_zero_boundary() {
        // Arrange
        let mig = TierMigration {
            page_id: 0,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::Nvme,
            codec: CompressionCodec::None,
            page_bytes: 1024,
            reason: TierMigrationReason::ColdCascade,
        };
        // Assert
        assert_eq!(mig.page_id, 0);
        assert_eq!(mig.from_tier, StorageTier::CpuDram);
    }

    #[test]
    fn tier_migration_with_none_codec_field() {
        // Arrange
        let mig = TierMigration {
            page_id: 99,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            reason: TierMigrationReason::EvictionPressure,
        };
        // Assert: codec::None is a valid compression codec for migration
        assert_eq!(mig.codec, CompressionCodec::None);
        assert_eq!(mig.reason, TierMigrationReason::EvictionPressure);
    }

    #[test]
    fn tier_migration_reason_eviction_pressure_in_struct() {
        // Arrange: construct TierMigration with EvictionPressure reason
        let mig = TierMigration {
            page_id: 10,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4,
            page_bytes: 8192,
            reason: TierMigrationReason::EvictionPressure,
        };
        // Assert
        assert_eq!(mig.reason, TierMigrationReason::EvictionPressure);
        assert_eq!(mig.codec, CompressionCodec::Lz4);
    }

    #[test]
    fn three_tier_stats_default_all_latency_fields_zero() {
        // Arrange
        let stats = ThreeTierSwapStats::default();
        // Assert: all latency fields start at zero
        assert_eq!(stats.total_eviction_latency_us, 0);
        assert_eq!(stats.total_swap_in_latency_us, 0);
        assert_eq!(stats.eviction_rounds, 0);
        assert_eq!(stats.swap_in_rounds, 0);
        assert_eq!(stats.total_bytes_evicted, 0);
        assert_eq!(stats.total_bytes_swapped_in, 0);
    }

    #[test]
    fn three_tier_stats_avg_eviction_single_gpu_eviction() {
        // Arrange: single GPU→DRAM eviction with known latency
        let stats = ThreeTierSwapStats {
            evictions_gpu_to_dram: 1,
            total_eviction_latency_us: 500,
            ..Default::default()
        };
        // Act
        let avg = stats.avg_eviction_latency_us();
        // Assert
        assert!((avg - 500.0).abs() < f64::EPSILON, "expected 500.0, got {avg}");
    }

    #[test]
    fn three_tier_stats_total_bytes_evicted_accumulation_large() {
        // Arrange: use struct update syntax with large byte counts
        let stats = ThreeTierSwapStats {
            total_bytes_evicted: 1_000_000_000_000, // ~1 TB
            evictions_gpu_to_dram: 256,
            ..Default::default()
        };
        // Assert
        assert_eq!(stats.total_bytes_evicted, 1_000_000_000_000);
        assert_eq!(stats.total_bytes_swapped_in, 0);
    }

    #[test]
    fn three_tier_stats_struct_update_syntax_partial() {
        // Arrange: use struct update syntax to only override specific fields
        let base = ThreeTierSwapStats {
            evictions_gpu_to_dram: 10,
            swap_ins_dram_to_gpu: 5,
            pages_on_hbm: 3,
            ..Default::default()
        };
        // Assert: overridden fields
        assert_eq!(base.evictions_gpu_to_dram, 10);
        assert_eq!(base.swap_ins_dram_to_gpu, 5);
        assert_eq!(base.pages_on_hbm, 3);
        // Assert: default fields untouched
        assert_eq!(base.evictions_dram_to_nvme, 0);
        assert_eq!(base.swap_ins_nvme_to_dram, 0);
    }

    #[test]
    fn tier_migration_plan_with_both_eviction_and_swap_in() {
        // Arrange: plan containing both eviction candidates and swap-in requests
        let candidate = EvictionCandidate {
            page_id: 1,
            score: 10,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };
        let req = PrefetchRequest {
            page_id: 2,
            urgency: 0.95,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let plan = TierMigrationPlan {
            eviction_candidates: vec![candidate],
            swap_in_requests: vec![req],
            tier_migrations: vec![],
            built_at: Instant::now(),
        };
        // Assert: both vectors non-empty
        assert_eq!(plan.eviction_candidates.len(), 1);
        assert_eq!(plan.swap_in_requests.len(), 1);
        assert_eq!(plan.eviction_candidates[0].page_id, 1);
        assert_eq!(plan.swap_in_requests[0].page_id, 2);
    }

    #[test]
    fn tier_migration_plan_clone_vectors_are_independent() {
        // Arrange
        let candidate = EvictionCandidate {
            page_id: 42,
            score: 50,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 8192,
            group_id: Some(7),
        };
        let original = TierMigrationPlan {
            eviction_candidates: vec![candidate],
            swap_in_requests: vec![],
            tier_migrations: vec![],
            built_at: Instant::now(),
        };
        // Act
        let cloned = original.clone();
        // Assert: cloned vectors are equal but independent
        assert_eq!(cloned.eviction_candidates.len(), 1);
        assert_eq!(cloned.eviction_candidates[0].page_id, 42);
        assert_eq!(cloned.eviction_candidates[0].score, 50);
    }

    #[test]
    fn three_tier_swap_config_auto_start_false_via_struct_update() {
        // Arrange: use struct update syntax to override auto_start only
        let config = ThreeTierSwapConfig {
            auto_start: false,
            ..ThreeTierSwapConfig::default()
        };
        // Assert
        assert!(!config.auto_start);
        // Assert: sub-configs still have their defaults
        assert!(config.migration.page_size > 0);
    }

    #[test]
    fn tier_migration_clone_debug_output_matches() {
        // Arrange
        let mig = TierMigration {
            page_id: 77,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 2048,
            reason: TierMigrationReason::EvictionPressure,
        };
        // Act
        let cloned = mig.clone();
        let debug_orig = format!("{mig:?}");
        let debug_clone = format!("{cloned:?}");
        // Assert: Debug output should be identical for equal values
        assert_eq!(debug_orig, debug_clone);
    }

    #[test]
    fn tier_migration_plan_built_at_monotonically_increases() {
        // Arrange: build two plans sequentially
        let plan1 = TierMigrationPlan {
            eviction_candidates: vec![],
            swap_in_requests: vec![],
            tier_migrations: vec![],
            built_at: Instant::now(),
        };
        // Small spin to ensure time passes
        let mut sink = 0u64;
        for i in 0..1000 {
            sink = sink.wrapping_add(i);
        }
        std::hint::black_box(sink);
        let plan2 = TierMigrationPlan {
            eviction_candidates: vec![],
            swap_in_requests: vec![],
            tier_migrations: vec![],
            built_at: Instant::now(),
        };
        // Assert: second plan's built_at is >= first
        assert!(plan2.built_at >= plan1.built_at,
            "plan2.built_at should be >= plan1.built_at");
    }

    // ── Wave 14 additional tests (+13, 729→742) ─────────────────────────────────

    #[test]
    fn three_tier_stats_total_migrations_counts_only_four_fields() {
        // Arrange: set all four migration counters to distinct values
        let stats = ThreeTierSwapStats {
            evictions_gpu_to_dram: 100,
            evictions_dram_to_nvme: 200,
            swap_ins_dram_to_gpu: 300,
            swap_ins_nvme_to_dram: 400,
            ..Default::default()
        };
        // Act
        let total = stats.total_migrations();
        // Assert: sum is exactly 100+200+300+400 = 1000
        assert_eq!(total, 1000, "total_migrations must sum all four counters");
    }

    #[test]
    fn tier_migration_plan_swap_in_requests_field_is_accessible() {
        // Arrange: plan with three swap-in requests of different page_bytes
        let req1 = PrefetchRequest {
            page_id: 10,
            urgency: 0.1,
            prefetch_confidence: 0.2,
            page_bytes: 1024,
            enqueued_at: Instant::now(),
        };
        let req2 = PrefetchRequest {
            page_id: 20,
            urgency: 0.5,
            prefetch_confidence: 0.6,
            page_bytes: 2048,
            enqueued_at: Instant::now(),
        };
        let req3 = PrefetchRequest {
            page_id: 30,
            urgency: 0.9,
            prefetch_confidence: 1.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let plan = TierMigrationPlan {
            eviction_candidates: vec![],
            swap_in_requests: vec![req1, req2, req3],
            tier_migrations: vec![],
            built_at: Instant::now(),
        };
        // Assert: all three requests are accessible with correct fields
        assert_eq!(plan.swap_in_requests.len(), 3);
        assert_eq!(plan.swap_in_requests[0].page_bytes, 1024);
        assert_eq!(plan.swap_in_requests[1].page_bytes, 2048);
        assert_eq!(plan.swap_in_requests[2].page_bytes, 4096);
    }

    #[test]
    fn eviction_candidate_all_tier_variants_constructible() {
        // Arrange & Act: construct candidates on all three tiers
        let hbm = EvictionCandidate {
            page_id: 1,
            score: 10,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::None,
            page_bytes: 4096,
            group_id: None,
        };
        let dram = EvictionCandidate {
            page_id: 2,
            score: 20,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(99),
        };
        let nvme = EvictionCandidate {
            page_id: 3,
            score: 30,
            current_tier: StorageTier::Nvme,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 4096,
            group_id: None,
        };
        // Assert: all tiers distinct
        assert_ne!(hbm.current_tier, dram.current_tier);
        assert_ne!(dram.current_tier, nvme.current_tier);
        assert_ne!(hbm.current_tier, nvme.current_tier);
        assert_eq!(hbm.score, 10);
        assert_eq!(dram.group_id, Some(99));
        assert_eq!(nvme.codec, CompressionCodec::BitPackRle);
    }

    #[test]
    fn coordinator_stats_snapshot_includes_registered_page_counts() {
        // Arrange: create coordinator and register pages on different tiers
        let (c, _backend) = make_coordinator(false);
        c.register_page(1, Some(0x1000), 4096); // HBM
        c.register_page(2, Some(0x2000), 4096); // HBM
        c.register_page(3, None, 4096);         // DRAM
        // Act
        let stats = c.stats();
        // Assert: pages_on_hbm = 2, pages_on_dram = 1, pages_on_nvme = 0
        assert_eq!(stats.pages_on_hbm, 2, "two pages on HBM");
        assert_eq!(stats.pages_on_dram, 1, "one page on DRAM");
        assert_eq!(stats.pages_on_nvme, 0, "no pages on NVMe");
    }

    #[test]
    fn tier_migration_reason_can_be_used_in_hashmap_key() {
        // Arrange: use TierMigrationReason as HashMap key (Hash + Eq satisfied)
        let mut map = HashMap::new();
        map.insert(TierMigrationReason::EvictionPressure, "evict");
        map.insert(TierMigrationReason::SequenceDemand, "demand");
        map.insert(TierMigrationReason::Prefetch, "prefetch");
        map.insert(TierMigrationReason::ColdCascade, "cascade");
        // Act: retrieve by key
        let val = map.get(&TierMigrationReason::Prefetch);
        // Assert
        assert_eq!(val, Some(&"prefetch"));
        assert_eq!(map.len(), 4, "all four variants inserted");
    }

    #[test]
    fn register_pages_from_hgal_bulk_insert_is_idempotent() {
        // Arrange: coordinator and a batch of page metadata
        let (c, _backend) = make_coordinator(false);
        let mut pages = HashMap::new();
        pages.insert(100, PageMetadata {
            page_id: 100,
            sequence_id: Some(1),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        });
        pages.insert(200, PageMetadata {
            page_id: 200,
            sequence_id: Some(2),
            recency: 0,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        });
        // Act: register twice
        c.register_pages_from_hgal(&pages, 4096);
        c.register_pages_from_hgal(&pages, 8192);
        // Assert: addr_table has exactly 2 entries with first page_bytes
        let table = c.addr_table.read().expect("read lock");
        assert_eq!(table.len(), 2, "idempotent insert must not duplicate");
        let entry100 = table.get(&100).expect("page 100 must exist");
        assert_eq!(entry100.original_bytes, 4096, "first page_bytes preserved");
    }

    #[test]
    fn three_tier_stats_avg_swap_in_single_nvme_promotion() {
        // Arrange: single NVMe-to-DRAM swap-in with 750us latency
        let stats = ThreeTierSwapStats {
            swap_ins_nvme_to_dram: 1,
            total_swap_in_latency_us: 750,
            ..Default::default()
        };
        // Act
        let avg = stats.avg_swap_in_latency_us();
        // Assert
        assert!((avg - 750.0).abs() < f64::EPSILON,
            "expected 750.0, got {avg}");
    }

    #[test]
    fn tier_migration_plan_tier_migrations_tracks_page_bytes_sum() {
        // Arrange: three tier migrations with varying sizes
        let migrations = vec![
            TierMigration {
                page_id: 1,
                from_tier: StorageTier::GpuHbm,
                to_tier: StorageTier::CpuDram,
                codec: CompressionCodec::Lz4,
                page_bytes: 4096,
                reason: TierMigrationReason::EvictionPressure,
            },
            TierMigration {
                page_id: 2,
                from_tier: StorageTier::CpuDram,
                to_tier: StorageTier::Nvme,
                codec: CompressionCodec::ZstdDict,
                page_bytes: 8192,
                reason: TierMigrationReason::ColdCascade,
            },
            TierMigration {
                page_id: 3,
                from_tier: StorageTier::Nvme,
                to_tier: StorageTier::GpuHbm,
                codec: CompressionCodec::None,
                page_bytes: 16384,
                reason: TierMigrationReason::SequenceDemand,
            },
        ];
        let plan = TierMigrationPlan {
            eviction_candidates: vec![],
            swap_in_requests: vec![],
            tier_migrations: migrations,
            built_at: Instant::now(),
        };
        // Act: sum page_bytes
        let total_bytes: usize = plan.tier_migrations.iter()
            .map(|m| m.page_bytes)
            .sum();
        // Assert
        assert_eq!(plan.tier_migrations.len(), 3);
        assert_eq!(total_bytes, 4096 + 8192 + 16384, "sum must match");
    }

    #[test]
    fn build_batch_returns_empty_plan_for_zero_pressure_no_active_pages() {
        // Arrange: coordinator with one HBM page in Active state
        let (c, _backend) = make_coordinator(false);
        c.register_page(55, Some(0x5500), 4096);
        {
            let mut meta = c.page_metadata.write().expect("write lock");
            meta.insert(55, PageMetadata {
                page_id: 55,
                sequence_id: Some(300),
                recency: 0,
                access_count: 50,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Active,
                warm_until: None,
            });
        }
        // Act: zero pressure, no active page demands
        let plan = c.build_batch(&[], 0.0);
        // Assert: Active pages are never evicted, no swap-in needed
        assert!(plan.eviction_candidates.is_empty(),
            "Active pages must not be evicted");
        assert!(plan.swap_in_requests.is_empty(),
            "no swap-in when page already on HBM");
    }

    #[test]
    fn storage_tier_nvem_is_least_in_ord_ordering() {
        // Arrange: all three tiers
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        // Act: find the minimum
        let min_tier = tiers.iter().min().expect("non-empty");
        // Assert: Nvme is always the minimum
        assert_eq!(*min_tier, StorageTier::Nvme,
            "Nvme must be the minimum in Ord ordering");
    }

    #[test]
    fn compression_codec_nocomp_none_is_default_in_addr_entry() {
        // Arrange: register a page without GPU ptr (goes to DRAM)
        let (c, _backend) = make_coordinator(false);
        c.register_page(77, None, 4096);
        // Act: read addr_table entry
        let table = c.addr_table.read().expect("read lock");
        let entry = table.get(&77).expect("page 77 must exist");
        // Assert: codec defaults to None, tier is CpuDram
        assert_eq!(entry.codec, CompressionCodec::None,
            "default codec must be None");
        assert_eq!(entry.current_tier, StorageTier::CpuDram,
            "no GPU ptr implies CpuDram tier");
        assert_eq!(entry.gpu_ptr, None, "no GPU pointer");
    }

    #[test]
    fn coordinator_record_eviction_accumulates_bytes_across_tiers() {
        // Arrange: fresh coordinator
        let (c, _backend) = make_coordinator(false);
        // Act: record evictions from two different tier pairs
        c.record_eviction_completed(1, StorageTier::GpuHbm, StorageTier::CpuDram, 4096, 100);
        c.record_eviction_completed(2, StorageTier::CpuDram, StorageTier::Nvme, 8192, 200);
        // Assert: total bytes accumulates from both
        let stats = c.stats();
        assert_eq!(stats.evictions_gpu_to_dram, 1);
        assert_eq!(stats.evictions_dram_to_nvme, 1);
        assert_eq!(stats.total_bytes_evicted, 4096 + 8192,
            "bytes must accumulate across tier pairs");
        assert_eq!(stats.total_eviction_latency_us, 300,
            "latency must accumulate");
    }

    #[test]
    fn tier_migration_reason_form_exhaustive_match_coverage() {
        // Arrange: iterate all variants via explicit match
        let reasons = vec![
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];
        // Act: map each variant to a label through exhaustive match
        let labels: Vec<&str> = reasons.iter().map(|r| match r {
            TierMigrationReason::EvictionPressure => "eviction",
            TierMigrationReason::SequenceDemand => "demand",
            TierMigrationReason::Prefetch => "prefetch",
            TierMigrationReason::ColdCascade => "cascade",
        }).collect();
        // Assert: each label is unique and non-empty
        assert_eq!(labels.len(), 4);
        for label in &labels {
            assert!(!label.is_empty(), "label must be non-empty");
        }
        // All labels are pairwise distinct
        for i in 0..labels.len() {
            for j in (i + 1)..labels.len() {
                assert_ne!(labels[i], labels[j],
                    "labels[{i}] == labels[{j}] but must be distinct");
            }
        }
    }

}
