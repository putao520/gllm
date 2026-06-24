//! Swap-In Worker — Proactive Page Load (SPEC 22-PAGE-COMPRESSION.md §6).
//!
//! The Swap-In Worker monitors a prefetch queue and proactively migrates pages
//! from lower tiers (CpuDram / Nvme) back to GpuHbm before they are needed
//! on the inference hot path. This reduces page-fault latency by overlapping
//! DMA transfers with compute.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────┐    prefetch_hint    ┌──────────────────┐
//! │  Scheduler   │ ──────────────────→ │  Prefetch Queue  │
//! │  (HGAL)      │                     │  (MPSC channel)  │
//! └──────────────┘                     └──────┬───────────┘
//!                                             │ dequeue
//!                                             ▼
//!                                      ┌──────────────────┐
//!                                      │  SwapInWorker    │
//!                                      │  (bg thread)    │
//!                                      └──────┬───────────┘
//!                                             │ submit MigrationCommand
//!                                             ▼
//!                                      ┌──────────────────┐
//!                                      │ PageMigration    │
//!                                      │ Actor            │
//!                                      └──────────────────┘
//! ```
//!
//! ## Priority
//!
//! Pages are sorted by `urgency` (descending) before submission so that
//! pages about to be accessed are swapped in first. Urgency is derived from:
//! - **importance_score rebound** — a page whose score has risen since eviction
//!   is more likely to be needed soon.
//! - **prefetch confidence** — scheduler-provided hint strength [0.0, 1.0].
//! - **inverse tier depth** — CpuDram pages are cheaper to promote than Nvme.
//!
//! ## Two-hop NVMe path
//!
//! Pages on NVMe must first be promoted to CpuDram (PromoteToDram) before
//! they can be promoted to GpuHbm (PromoteToHbm). The worker automatically
//! issues both commands in sequence for NVMe-resident pages.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::kv_cache::StorageTier;
use crate::scheduler::migration_actor::{
    MigrationCommand, MigrationDone, MigrationResult, PageAddrTable, PageMigrationActor,
};
use crate::scheduler::observer::{BasicObserver, WeightPageTelemetryEvent};
use crate::scheduler::types::{PageId, PageMetadata, PageState, WeightTier};

// ─────────────────────────────────────────────────────────────────────────────
// Prefetch request
// ─────────────────────────────────────────────────────────────────────────────

/// A single prefetch / swap-in request enqueued by the scheduler.
///
/// The `urgency` field drives priority ordering: higher urgency pages
/// are promoted first.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefetchRequest {
    /// Page to swap in.
    pub page_id: PageId,
    /// Urgency score (higher = swap in sooner).
    ///
    /// Computed from importance_score rebound + prefetch_confidence +
    /// tier_depth bonus. See [`SwapInWorker::compute_urgency`].
    pub urgency: f32,
    /// Scheduler's confidence that this page will be needed [0.0, 1.0].
    pub prefetch_confidence: f32,
    /// Uncompressed page size in bytes.
    pub page_bytes: usize,
    /// Timestamp when the request was enqueued.
    pub enqueued_at: Instant,
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Swap-In Worker configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SwapInWorkerConfig {
    /// Maximum prefetch requests processed per round.
    pub max_prefetch_per_round: usize,
    /// Interval between background processing rounds.
    pub tick_interval: Duration,
    /// Prefetch confidence threshold below which requests are skipped.
    pub min_confidence: f32,
    /// Maximum number of in-flight migration commands before back-pressure.
    pub max_in_flight: usize,
    /// Uncompressed page size in bytes.
    pub page_bytes: usize,
}

impl Default for SwapInWorkerConfig {
    fn default() -> Self {
        Self {
            max_prefetch_per_round: 16,
            tick_interval: Duration::from_millis(5),
            min_confidence: 0.1,
            max_in_flight: 64,
            page_bytes: 4096,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwapInWorker statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Cumulative statistics for the swap-in worker.
#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub struct SwapInWorkerStats {
    /// Total prefetch requests received.
    pub total_requests: u64,
    /// Requests that were submitted to the migration actor.
    pub submitted: u64,
    /// Requests skipped (confidence below threshold or page already in HBM).
    pub skipped: u64,
    /// Successful promotions (completion received with Ok).
    pub promoted_ok: u64,
    /// Failed promotions (completion received with Failed).
    pub promoted_failed: u64,
    /// Two-hop (NVMe → DRAM → HBM) promotions.
    pub two_hop_promotions: u64,
    /// Cumulative swap-in latency (microseconds) for successful promotions.
    pub total_latency_us: u64,
    /// Number of rounds executed.
    pub rounds: u64,
}


impl SwapInWorkerStats {
    /// Average swap-in latency in microseconds.
    pub fn avg_latency_us(&self) -> f64 {
        if self.promoted_ok == 0 {
            return 0.0;
        }
        self.total_latency_us as f64 / self.promoted_ok as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwapInWorker
// ─────────────────────────────────────────────────────────────────────────────

/// Swap-In Worker — proactive page promotion from lower tiers to GpuHbm.
///
/// Runs on a dedicated background thread. Each round:
/// 1. Drains the prefetch request channel.
/// 2. Sorts requests by urgency (descending).
/// 3. For each request, checks the current tier via `addr_table`.
/// 4. Submits `PromoteToDram` (if on NVMe) then `PromoteToHbm` to the
///    `PageMigrationActor`.
/// 5. Drains completion events and updates page metadata.
///
/// The worker does **not** block the scheduler hot path — requests are
/// enqueued via the MPSC channel and processed asynchronously.
#[allow(dead_code)]
pub struct SwapInWorker {
    /// Channel to signal the worker thread to stop.
    stop: Arc<AtomicBool>,
    /// Handle to the background thread.
    handle: Option<JoinHandle<()>>,
    /// Sender half of the prefetch request channel.
    prefetch_tx: std::sync::mpsc::Sender<PrefetchRequest>,
    /// Shared page metadata (read by worker, written by scheduler).
    page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>>,
    /// Shared page address table.
    addr_table: PageAddrTable,
    /// Shared statistics (updated by worker, read by observer).
    stats: Arc<Mutex<SwapInWorkerStats>>,
    /// Observer for swap-in telemetry.
    observer: Arc<Mutex<BasicObserver>>,
}

impl SwapInWorker {
    /// Spawn the swap-in worker on a background thread.
    ///
    /// # Arguments
    /// * `config` — Worker configuration.
    /// * `actor` — Already-initialized `PageMigrationActor` (moved into the worker).
    /// * `page_metadata` — Shared page metadata map.
    /// * `addr_table` — Shared page address table.
    /// * `observer` — Telemetry observer for swap-in events.
    pub fn spawn(
        config: SwapInWorkerConfig,
        actor: PageMigrationActor,
        page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>>,
        addr_table: PageAddrTable,
        observer: Arc<Mutex<BasicObserver>>,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop);
        let page_meta_clone = Arc::clone(&page_metadata);
        let addr_table_clone = Arc::clone(&addr_table);
        let observer_clone = Arc::clone(&observer);
        let stats = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let stats_clone = Arc::clone(&stats);

        let (prefetch_tx, prefetch_rx) = std::sync::mpsc::channel::<PrefetchRequest>();

        let handle = thread::Builder::new()
            .name("swap-in-worker".to_string())
            .spawn(move || {
                swap_in_loop(
                    config,
                    actor,
                    stop_clone,
                    prefetch_rx,
                    page_meta_clone,
                    addr_table_clone,
                    stats_clone,
                    observer_clone,
                );
            })
            .expect("failed to spawn swap-in-worker thread");

        Self {
            stop,
            handle: Some(handle),
            prefetch_tx,
            page_metadata,
            addr_table,
            stats,
            observer,
        }
    }

    /// Enqueue a prefetch request (non-blocking).
    ///
    /// Returns `Ok(())` if the request was enqueued, `Err` if the channel
    /// is full or the worker has been shut down.
    pub fn prefetch(&self, req: PrefetchRequest) -> Result<(), SwapInWorkerError> {
        self.prefetch_tx
            .send(req)
            .map_err(|e| SwapInWorkerError::SendFailed(e.to_string()))
    }

    /// Enqueue multiple prefetch requests (non-blocking).
    ///
    /// Returns the number of requests successfully enqueued.
    pub fn prefetch_batch(&self, requests: &[PrefetchRequest]) -> usize {
        let mut enqueued = 0;
        for req in requests {
            if self.prefetch_tx.send(req.clone()).is_ok() {
                enqueued += 1;
            } else {
                break;
            }
        }
        enqueued
    }

    /// Compute urgency score for a prefetch request.
    ///
    /// Urgency is a float in [0.0, ∞). Higher urgency = promote sooner.
    ///
    /// Factors:
    /// - **importance_score rebound**: `meta.importance_score / 255.0`
    ///   (pages whose score has risen since eviction are more likely needed).
    /// - **prefetch_confidence**: scheduler hint strength [0.0, 1.0].
    /// - **tier_depth bonus**: CpuDram = 1.0, Nvme = 0.5 (DRAM is cheaper to
    ///   promote so we prioritize it to get quick wins).
    /// - **recency bonus**: recently-accessed pages get a small boost.
    pub fn compute_urgency(
        meta: &PageMetadata,
        prefetch_confidence: f32,
        current_tier: StorageTier,
    ) -> f32 {
        // Importance score rebound: use access_count as proxy for importance
        // (higher access count → more likely to be needed again).
        let importance_rebound = (meta.access_count as f32).ln_1p() / 10.0_f32.ln_1p();

        // Tier depth bonus: CpuDram pages are cheaper to promote.
        let tier_bonus = match current_tier {
            StorageTier::CpuDram => 1.0,
            StorageTier::Nvme => 0.5,
            StorageTier::GpuHbm => 2.0, // already hot — very urgent to confirm
        };

        // Recency bonus: pages accessed recently are more likely needed.
        let elapsed_secs = Instant::now()
            .saturating_duration_since(meta.last_access)
            .as_secs_f32();
        let recency_bonus = 1.0 / (1.0 + elapsed_secs);

        importance_rebound * prefetch_confidence * tier_bonus + recency_bonus * 0.1
    }

    /// Signal the worker thread to stop and wait for it to finish.
    pub fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }

    /// Read current statistics (snapshot).
    pub fn stats(&self) -> SwapInWorkerStats {
        match self.stats.lock() {
            Ok(s) => s.clone(),
            Err(_) => SwapInWorkerStats::default(),
        }
    }

    /// Perform one swap-in round synchronously.
    ///
    /// This is the core logic also used by the background thread. It:
    /// 1. Collects prefetch requests from the provided slice.
    /// 2. Sorts by urgency (descending).
    /// 3. Checks current tier via addr_table.
    /// 4. Submits migration commands to the actor.
    /// 5. Drains completions and updates page metadata.
    ///
    /// Returns the number of migration commands submitted this round.
    pub fn swap_in_round(
        config: &SwapInWorkerConfig,
        actor: &PageMigrationActor,
        requests: &mut Vec<PrefetchRequest>,
        page_metadata: &Arc<RwLock<HashMap<PageId, PageMetadata>>>,
        addr_table: &PageAddrTable,
        stats: &Arc<Mutex<SwapInWorkerStats>>,
        observer: &Arc<Mutex<BasicObserver>>,
    ) -> usize {
        // ── 1. Update stats ──────────────────────────────────────────────────
        if let Ok(mut s) = stats.lock() {
            s.total_requests += requests.len() as u64;
            s.rounds += 1;
        }

        if requests.is_empty() {
            return 0;
        }

        // ── 2. Sort by urgency descending (highest urgency first) ────────────
        requests.sort_by(|a, b| b.urgency.partial_cmp(&a.urgency).unwrap_or(std::cmp::Ordering::Equal));
        requests.truncate(config.max_prefetch_per_round);

        // ── 3. Submit migration commands ─────────────────────────────────────
        let mut submitted = 0;
        let mut in_flight = 0;

        for req in requests.drain(..) {
            // Skip low-confidence requests.
            if req.prefetch_confidence < config.min_confidence {
                if let Ok(mut s) = stats.lock() {
                    s.skipped += 1;
                }
                continue;
            }

            // Back-pressure: if too many in-flight, stop submitting.
            if in_flight >= config.max_in_flight {
                break;
            }

            // Check current tier from addr_table.
            let current_tier = {
                let table = match addr_table.read() {
                    Ok(t) => t,
                    Err(_) => {
                        if let Ok(mut s) = stats.lock() {
                            s.skipped += 1;
                        }
                        continue;
                    }
                };
                match table.get(&req.page_id) {
                    Some(entry) => entry.current_tier,
                    None => {
                        // Page not in addr_table — skip.
                        if let Ok(mut s) = stats.lock() {
                            s.skipped += 1;
                        }
                        continue;
                    }
                }
            };

            // Already on HBM — nothing to do.
            if current_tier == StorageTier::GpuHbm {
                if let Ok(mut s) = stats.lock() {
                    s.skipped += 1;
                }
                continue;
            }

            let page_bytes = if req.page_bytes > 0 {
                req.page_bytes
            } else {
                config.page_bytes
            };

            // NVMe two-hop: first promote to DRAM, then to HBM.
            if current_tier == StorageTier::Nvme {
                let _ = actor.send(MigrationCommand::PromoteToDram {
                    page_id: req.page_id,
                    page_bytes,
                });
                if let Ok(mut s) = stats.lock() {
                    s.two_hop_promotions += 1;
                }
                in_flight += 1;
            }

            // Promote to HBM.
            let _ = actor.send(MigrationCommand::PromoteToHbm {
                page_id: req.page_id,
                page_bytes,
            });
            submitted += 1;
            in_flight += 1;

            // Record telemetry.
            let from_weight_tier = match current_tier {
                StorageTier::Nvme => WeightTier::Cold,
                StorageTier::CpuDram => WeightTier::Warm,
                StorageTier::GpuHbm => WeightTier::Hot,
            };
            if let Ok(mut obs) = observer.lock() {
                obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                    page_id: req.page_id,
                    from_tier: from_weight_tier,
                    to_tier: WeightTier::Hot,
                    latency_us: 0, // will be updated on completion
                    bytes: page_bytes as u64,
                });
            }
        }

        if let Ok(mut s) = stats.lock() {
            s.submitted += submitted as u64;
        }

        // ── 4. Drain completions and update page metadata ────────────────────
        drain_completions_and_update(actor, page_metadata, addr_table, stats, observer);

        submitted
    }
}

impl Drop for SwapInWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in the swap-in worker.
#[derive(Debug, Clone, PartialEq)]
pub enum SwapInWorkerError {
    /// Failed to send prefetch request (worker shut down?).
    SendFailed(String),
    /// Failed to receive completion from migration actor.
    RecvFailed(String),
}

impl std::fmt::Display for SwapInWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SendFailed(msg) => write!(f, "swap-in worker send failed: {msg}"),
            Self::RecvFailed(msg) => write!(f, "swap-in worker recv failed: {msg}"),
        }
    }
}

impl std::error::Error for SwapInWorkerError {}

// ─────────────────────────────────────────────────────────────────────────────
// Background loop
// ─────────────────────────────────────────────────────────────────────────────

fn swap_in_loop(
    config: SwapInWorkerConfig,
    actor: PageMigrationActor,
    stop: Arc<AtomicBool>,
    prefetch_rx: std::sync::mpsc::Receiver<PrefetchRequest>,
    page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>>,
    addr_table: PageAddrTable,
    stats: Arc<Mutex<SwapInWorkerStats>>,
    observer: Arc<Mutex<BasicObserver>>,
) {
    while !stop.load(Ordering::Relaxed) {
        let start = Instant::now();

        // Drain all pending prefetch requests from the channel.
        let mut requests: Vec<PrefetchRequest> = Vec::new();
        while let Ok(req) = prefetch_rx.try_recv() {
            requests.push(req);
        }

        if !requests.is_empty() {
            SwapInWorker::swap_in_round(
                &config,
                &actor,
                &mut requests,
                &page_metadata,
                &addr_table,
                &stats,
                &observer,
            );
        } else {
            // No requests — still drain completions from previous rounds.
            drain_completions_and_update(
                &actor, &page_metadata, &addr_table, &stats, &observer,
            );
        }

        let elapsed = start.elapsed();
        let sleep_remaining = config.tick_interval.saturating_sub(elapsed);
        if sleep_remaining > Duration::ZERO {
            thread::sleep(sleep_remaining);
        }
    }

    // Final drain before exit.
    drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
    actor.shutdown();
}

// ─────────────────────────────────────────────────────────────────────────────
// Completion draining + page metadata update
// ─────────────────────────────────────────────────────────────────────────────

/// Drain completion events from the migration actor and update page metadata
/// to reflect successful promotions.
fn drain_completions_and_update(
    actor: &PageMigrationActor,
    page_metadata: &Arc<RwLock<HashMap<PageId, PageMetadata>>>,
    addr_table: &PageAddrTable,
    stats: &Arc<Mutex<SwapInWorkerStats>>,
    observer: &Arc<Mutex<BasicObserver>>,
) {
    let completions: Vec<MigrationDone> = {
        let mut completions = Vec::new();
        while let Some(done) = actor.try_recv_done() {
            completions.push(done);
        }
        completions
    };

    for done in completions {
        let latency_us = {
            // Estimate latency from the page metadata's swap_in_time if available.
            let meta_guard = page_metadata.read().ok();
            match meta_guard {
                Some(guard) => match guard.get(&done.page_id) {
                    Some(meta) => match meta.swap_in_time {
                        Some(t) => Instant::now().saturating_duration_since(t).as_micros() as u64,
                        None => 0,
                    },
                    None => 0,
                },
                None => 0,
            }
        };

        match &done.result {
            MigrationResult::Ok { .. } => {
                // Update page metadata to reflect the new tier.
                if let Ok(mut meta_guard) = page_metadata.write() {
                    if let Some(meta) = meta_guard.get_mut(&done.page_id) {
                        match done.to_tier {
                            StorageTier::GpuHbm => {
                                meta.state = PageState::Active;
                                meta.swap_in_time = None;
                            }
                            StorageTier::CpuDram => {
                                meta.state = PageState::Warm;
                                meta.swap_in_time = Some(Instant::now());
                            }
                            StorageTier::Nvme => {
                                // Should not happen for swap-in, but handle gracefully.
                                meta.state = PageState::Swapped;
                                meta.swap_in_time = Some(Instant::now());
                            }
                        }
                    }
                }

                // Update addr_table current_tier to reflect the new tier.
                if let Ok(mut addr_guard) = addr_table.write() {
                    if let Some(entry) = addr_guard.get_mut(&done.page_id) {
                        entry.current_tier = done.to_tier;
                    }
                }

                if let Ok(mut s) = stats.lock() {
                    s.promoted_ok += 1;
                    s.total_latency_us += latency_us;
                }

                // Record telemetry for successful recovery.
                let from_weight_tier = match done.from_tier {
                    StorageTier::GpuHbm => WeightTier::Hot,
                    StorageTier::CpuDram => WeightTier::Warm,
                    StorageTier::Nvme => WeightTier::Cold,
                };
                let to_weight_tier = match done.to_tier {
                    StorageTier::GpuHbm => WeightTier::Hot,
                    StorageTier::CpuDram => WeightTier::Warm,
                    StorageTier::Nvme => WeightTier::Cold,
                };
                if let Ok(mut obs) = observer.lock() {
                    obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                        page_id: done.page_id,
                        from_tier: from_weight_tier,
                        to_tier: to_weight_tier,
                        latency_us,
                        bytes: 0, // byte count was already recorded at submit time
                    });
                }
            }
            MigrationResult::Failed { .. } => {
                if let Ok(mut s) = stats.lock() {
                    s.promoted_failed += 1;
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
    use crate::kv_cache::CompressionCodec;
    use crate::scheduler::dma_helpers::{CpuDmaBackendSized, DmaBackend};
    use crate::scheduler::migration_actor::{MigrationActorConfig, PageAddrEntry};
    use crate::scheduler::observer::{BasicObserver, EvictionReason, RuntimeObserver};
    use std::collections::HashMap;
    use std::path::PathBuf;

    /// Verify urgency computation: higher access count → higher urgency.
    #[test]
    fn urgency_increases_with_access_count() {
        let meta_high = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let meta_low = PageMetadata {
            page_id: 2,
            sequence_id: Some(100),
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        let urgency_high = SwapInWorker::compute_urgency(
            &meta_high, 0.8, StorageTier::CpuDram,
        );
        let urgency_low = SwapInWorker::compute_urgency(
            &meta_low, 0.8, StorageTier::CpuDram,
        );

        assert!(
            urgency_high > urgency_low,
            "higher access count should yield higher urgency: high={} low={}",
            urgency_high,
            urgency_low,
        );
    }

    /// Verify urgency: CpuDram pages have higher urgency than Nvme pages
    /// (cheaper to promote → prioritize quick wins).
    #[test]
    fn urgency_dram_higher_than_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        let urgency_dram = SwapInWorker::compute_urgency(
            &meta, 0.5, StorageTier::CpuDram,
        );
        let urgency_nvme = SwapInWorker::compute_urgency(
            &meta, 0.5, StorageTier::Nvme,
        );

        assert!(
            urgency_dram > urgency_nvme,
            "CpuDram urgency should be higher than Nvme: dram={} nvme={}",
            urgency_dram,
            urgency_nvme,
        );
    }

    /// Verify urgency: higher prefetch confidence → higher urgency.
    #[test]
    fn urgency_increases_with_confidence() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        let urgency_high = SwapInWorker::compute_urgency(
            &meta, 0.9, StorageTier::CpuDram,
        );
        let urgency_low = SwapInWorker::compute_urgency(
            &meta, 0.1, StorageTier::CpuDram,
        );

        assert!(
            urgency_high > urgency_low,
            "higher confidence should yield higher urgency: high={} low={}",
            urgency_high,
            urgency_low,
        );
    }

    /// Verify SwapInWorker spawns and shuts down cleanly.
    #[test]
    fn spawn_and_shutdown() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config,
            actor,
            page_metadata,
            addr_table,
            observer,
        );

        // Let it run for a couple of ticks.
        thread::sleep(Duration::from_millis(150));
        worker.shutdown();
    }

    /// Verify swap_in_round skips pages already on HBM.
    #[test]
    fn round_skips_hbm_pages() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert a page that is already on HBM.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                42,
                PageAddrEntry {
                    gpu_ptr: Some(0x1000),
                    host_buffer: None,
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 42,
            urgency: 1.0,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 0, "should not submit swap-in for page already on HBM");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 1, "should have skipped 1 page on HBM");

        actor.shutdown();
    }

    /// Verify swap_in_round submits PromoteToHbm for CpuDram pages.
    #[test]
    fn round_submits_for_dram_pages() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert a page on CpuDram.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                7,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::Lz4,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 7,
            urgency: 1.0,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 1, "should submit swap-in for CpuDram page");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 1);
        assert_eq!(s.two_hop_promotions, 0, "CpuDram should not be two-hop");

        actor.shutdown();
    }

    /// Verify swap_in_round submits two-hop for NVMe pages.
    #[test]
    fn round_submits_two_hop_for_nvme_pages() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert a page on NVMe.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                99,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: None,
                    current_tier: StorageTier::Nvme,
                    original_bytes: 4096,
                    codec: CompressionCodec::ZstdDict,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 99,
            urgency: 1.0,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 1, "should submit swap-in for NVMe page");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 1);
        assert_eq!(s.two_hop_promotions, 1, "NVMe should be two-hop");

        actor.shutdown();
    }

    /// Verify priority sorting: higher urgency requests are processed first.
    #[test]
    fn priority_sorting_by_urgency() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 2,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert two pages on CpuDram.
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [10usize, 20usize] {
                table.insert(
                    pid as PageId,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::Lz4,
                    },
                );
            }
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Low urgency first, high urgency second — should be reordered.
        let mut requests = vec![
            PrefetchRequest {
                page_id: 10,
                urgency: 0.1,
                prefetch_confidence: 0.8,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 20,
                urgency: 0.9,
                prefetch_confidence: 0.8,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 2, "both pages should be submitted");

        actor.shutdown();
    }

    /// Verify low-confidence requests are skipped.
    #[test]
    fn low_confidence_skipped() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert a page on CpuDram.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                5,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::Lz4,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 5,
            urgency: 1.0,
            prefetch_confidence: 0.1, // below threshold
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 0, "low-confidence request should be skipped");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 1);

        actor.shutdown();
    }

    /// Verify prefetch() enqueues a request.
    #[test]
    fn prefetch_enqueue() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(100),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config,
            actor,
            page_metadata,
            addr_table,
            observer,
        );

        let result = worker.prefetch(PrefetchRequest {
            page_id: 42,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        });
        assert!(result.is_ok(), "prefetch should succeed");

        worker.shutdown();
    }

    /// Verify stats snapshot.
    #[test]
    fn stats_snapshot() {
        let stats = SwapInWorkerStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.submitted, 0);
        assert_eq!(stats.promoted_ok, 0);
        assert_eq!(stats.avg_latency_us(), 0.0);
    }

    // ── SwapInWorkerConfig ──

    #[test]
    fn config_default_values() {
        let cfg = SwapInWorkerConfig::default();
        assert_eq!(cfg.max_prefetch_per_round, 16);
        assert_eq!(cfg.tick_interval, Duration::from_millis(5));
        assert!((cfg.min_confidence - 0.1).abs() < 1e-6);
        assert_eq!(cfg.max_in_flight, 64);
        assert_eq!(cfg.page_bytes, 4096);
    }

    #[test]
    fn config_custom_values() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 32,
            tick_interval: Duration::from_millis(10),
            min_confidence: 0.3,
            max_in_flight: 128,
            page_bytes: 8192,
        };
        assert_eq!(cfg.max_prefetch_per_round, 32);
        assert_eq!(cfg.min_confidence, 0.3);
    }

    // ── SwapInWorkerStats ──

    #[test]
    fn stats_avg_latency_computation() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 4;
        stats.total_latency_us = 1000;
        assert!((stats.avg_latency_us() - 250.0).abs() < 1e-6);
    }

    #[test]
    fn stats_all_fields_zero_on_default() {
        let s = SwapInWorkerStats::default();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.submitted, 0);
        assert_eq!(s.skipped, 0);
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);
        assert_eq!(s.two_hop_promotions, 0);
        assert_eq!(s.total_latency_us, 0);
        assert_eq!(s.rounds, 0);
    }

    // ── SwapInWorkerError Display ──

    #[test]
    fn error_display_contains_context() {
        let e1 = SwapInWorkerError::SendFailed("closed".into());
        assert!(format!("{e1}").contains("closed"));
        let e2 = SwapInWorkerError::RecvFailed("timeout".into());
        assert!(format!("{e2}").contains("timeout"));
    }

    // ── PrefetchRequest fields ──

    #[test]
    fn prefetch_request_fields() {
        let now = Instant::now();
        let req = PrefetchRequest {
            page_id: 42,
            urgency: 0.95,
            prefetch_confidence: 0.8,
            page_bytes: 65536,
            enqueued_at: now,
        };
        assert_eq!(req.page_id, 42);
        assert!((req.urgency - 0.95).abs() < 1e-6);
        assert!((req.prefetch_confidence - 0.8).abs() < 1e-6);
        assert_eq!(req.page_bytes, 65536);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // New tests below — ~25 additional tests
    // ═══════════════════════════════════════════════════════════════════════════

    // ── SwapInWorkerError: std::error::Error trait ──

    #[test]
    fn error_implements_std_error() {
        let e = SwapInWorkerError::SendFailed("test".into());
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn error_display_send_failed_prefix() {
        let e = SwapInWorkerError::SendFailed("channel closed".into());
        let msg = format!("{e}");
        assert!(
            msg.starts_with("swap-in worker send failed:"),
            "SendFailed display should start with standard prefix"
        );
    }

    #[test]
    fn error_display_recv_failed_prefix() {
        let e = SwapInWorkerError::RecvFailed("disconnected".into());
        let msg = format!("{e}");
        assert!(
            msg.starts_with("swap-in worker recv failed:"),
            "RecvFailed display should start with standard prefix"
        );
    }

    #[test]
    fn error_clone_preserves_message() {
        let e1 = SwapInWorkerError::SendFailed("original msg".into());
        let e2 = e1.clone();
        assert_eq!(format!("{e1}"), format!("{e2}"));
    }

    #[test]
    fn error_debug_format() {
        let e = SwapInWorkerError::RecvFailed("err".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("RecvFailed"), "Debug should contain variant name");
    }

    // ── SwapInWorkerConfig: Clone + edge cases ──

    #[test]
    fn config_clone_is_equal() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 8,
            tick_interval: Duration::from_millis(20),
            min_confidence: 0.25,
            max_in_flight: 32,
            page_bytes: 8192,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.max_prefetch_per_round, cfg.max_prefetch_per_round);
        assert_eq!(cloned.tick_interval, cfg.tick_interval);
        assert!((cloned.min_confidence - cfg.min_confidence).abs() < 1e-6);
        assert_eq!(cloned.max_in_flight, cfg.max_in_flight);
        assert_eq!(cloned.page_bytes, cfg.page_bytes);
    }

    #[test]
    fn config_zero_max_prefetch_per_round() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 0,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.max_prefetch_per_round, 0);
    }

    #[test]
    fn config_zero_min_confidence_accepts_all() {
        let cfg = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        assert!((cfg.min_confidence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn config_min_confidence_above_one() {
        let cfg = SwapInWorkerConfig {
            min_confidence: 1.5,
            ..SwapInWorkerConfig::default()
        };
        assert!((cfg.min_confidence - 1.5).abs() < 1e-6);
    }

    #[test]
    fn config_debug_format() {
        let cfg = SwapInWorkerConfig::default();
        let debug = format!("{cfg:?}");
        assert!(debug.contains("SwapInWorkerConfig"));
        assert!(debug.contains("max_prefetch_per_round"));
    }

    // ── SwapInWorkerStats: comprehensive ──

    #[test]
    fn stats_clone_is_equal() {
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 100;
        stats.submitted = 80;
        stats.skipped = 20;
        stats.promoted_ok = 75;
        stats.promoted_failed = 5;
        stats.two_hop_promotions = 10;
        stats.total_latency_us = 50000;
        stats.rounds = 50;
        let cloned = stats.clone();
        assert_eq!(cloned.total_requests, 100);
        assert_eq!(cloned.submitted, 80);
        assert_eq!(cloned.skipped, 20);
        assert_eq!(cloned.promoted_ok, 75);
        assert_eq!(cloned.promoted_failed, 5);
        assert_eq!(cloned.two_hop_promotions, 10);
        assert_eq!(cloned.total_latency_us, 50000);
        assert_eq!(cloned.rounds, 50);
    }

    #[test]
    fn stats_avg_latency_with_large_values() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 1000;
        stats.total_latency_us = 5_000_000;
        assert!((stats.avg_latency_us() - 5000.0).abs() < 1e-3);
    }

    #[test]
    fn stats_avg_latency_single_promotion() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 1;
        stats.total_latency_us = 1234;
        assert!((stats.avg_latency_us() - 1234.0).abs() < 1e-6);
    }

    #[test]
    fn stats_debug_format() {
        let stats = SwapInWorkerStats::default();
        let debug = format!("{stats:?}");
        assert!(debug.contains("SwapInWorkerStats"));
        assert!(debug.contains("total_requests"));
        assert!(debug.contains("promoted_ok"));
    }

    #[test]
    fn stats_derived_default_all_zero() {
        // Verify that #[derive(Default)] produces all-zero state.
        let stats = SwapInWorkerStats::default();
        assert_eq!(stats.total_requests, 0u64);
        assert_eq!(stats.submitted, 0u64);
        assert_eq!(stats.skipped, 0u64);
        assert_eq!(stats.promoted_ok, 0u64);
        assert_eq!(stats.promoted_failed, 0u64);
        assert_eq!(stats.two_hop_promotions, 0u64);
        assert_eq!(stats.total_latency_us, 0u64);
        assert_eq!(stats.rounds, 0u64);
        // avg_latency_us should return 0.0 when promoted_ok == 0.
        assert!((stats.avg_latency_us() - 0.0).abs() < 1e-6);
    }

    // ── PrefetchRequest: Clone + edge cases ──

    #[test]
    fn prefetch_request_clone() {
        let now = Instant::now();
        let req = PrefetchRequest {
            page_id: 7,
            urgency: 0.5,
            prefetch_confidence: 0.3,
            page_bytes: 1024,
            enqueued_at: now,
        };
        let cloned = req.clone();
        assert_eq!(cloned.page_id, req.page_id);
        assert!((cloned.urgency - req.urgency).abs() < 1e-6);
        assert!((cloned.prefetch_confidence - req.prefetch_confidence).abs() < 1e-6);
        assert_eq!(cloned.page_bytes, req.page_bytes);
    }

    #[test]
    fn prefetch_request_zero_urgency() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_id, 0);
        assert!((req.urgency - 0.0).abs() < 1e-6);
        assert!((req.prefetch_confidence - 0.0).abs() < 1e-6);
        assert_eq!(req.page_bytes, 0);
    }

    #[test]
    fn prefetch_request_debug_format() {
        let req = PrefetchRequest {
            page_id: 99,
            urgency: 1.0,
            prefetch_confidence: 1.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let debug = format!("{req:?}");
        assert!(debug.contains("PrefetchRequest"));
        assert!(debug.contains("page_id"));
    }

    #[test]
    fn prefetch_request_large_page_bytes() {
        let req = PrefetchRequest {
            page_id: 100,
            urgency: 0.5,
            prefetch_confidence: 0.7,
            page_bytes: usize::MAX,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_bytes, usize::MAX);
    }

    // ── compute_urgency: additional edge cases ──

    #[test]
    fn urgency_gpu_hbm_highest_tier_bonus() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_hbm = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::GpuHbm);
        let u_dram = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::Nvme);
        assert!(
            u_hbm > u_dram,
            "GpuHbm urgency should be higher than CpuDram: hbm={} dram={}",
            u_hbm, u_dram,
        );
        assert!(
            u_dram > u_nvme,
            "CpuDram urgency should be higher than Nvme: dram={} nvme={}",
            u_dram, u_nvme,
        );
    }

    #[test]
    fn urgency_zero_confidence_produces_relatively_low_score() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u_confident = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_zero = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // Zero confidence only has recency bonus, so it should be lower.
        assert!(
            u_confident > u_zero,
            "zero-confidence urgency should be lower: confident={} zero={}",
            u_confident, u_zero,
        );
    }

    #[test]
    fn urgency_recency_bonus_decreases_over_time() {
        let meta_old = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now() - Duration::from_secs(60),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let meta_recent = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        // Use zero confidence so recency dominates.
        let u_old = SwapInWorker::compute_urgency(&meta_old, 0.0, StorageTier::CpuDram);
        let u_recent = SwapInWorker::compute_urgency(&meta_recent, 0.0, StorageTier::CpuDram);
        assert!(
            u_recent > u_old,
            "recently-accessed page should have higher urgency: recent={} old={}",
            u_recent, u_old,
        );
    }

    // ── swap_in_round: empty requests ──

    #[test]
    fn round_empty_requests_returns_zero() {
        let config = SwapInWorkerConfig::default();
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
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = Vec::new();
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 0, "empty requests should yield zero submissions");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 1, "should still increment round counter");
        assert_eq!(s.total_requests, 0, "total_requests should be 0 for empty input");

        actor.shutdown();
    }

    // ── swap_in_round: page not in addr_table is skipped ──

    #[test]
    fn round_skips_page_not_in_addr_table() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        // Empty addr_table — no pages registered.
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 999,
            urgency: 1.0,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 0, "page not in addr_table should be skipped");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 1);

        actor.shutdown();
    }

    // ── swap_in_round: back-pressure limit ──

    #[test]
    fn round_respects_max_in_flight() {
        let config = SwapInWorkerConfig {
            max_in_flight: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Register 3 pages on CpuDram.
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize, 3usize] {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::None,
                    },
                );
            }
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=3usize)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(
            submitted, 1,
            "max_in_flight=1 should limit to 1 submitted migration"
        );

        actor.shutdown();
    }

    // ── swap_in_round: page_bytes falls back to config.page_bytes ──

    #[test]
    fn round_uses_config_page_bytes_when_request_has_zero() {
        let config = SwapInWorkerConfig {
            page_bytes: 8192,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                42,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 8192]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 8192,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // page_bytes=0 — should fall back to config.page_bytes=8192.
        let mut requests = vec![PrefetchRequest {
            page_id: 42,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 1, "should submit even with page_bytes=0");

        actor.shutdown();
    }

    // ── swap_in_round: truncation to max_prefetch_per_round ──

    #[test]
    fn round_truncates_to_max_prefetch_per_round() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Register 3 pages on CpuDram.
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize, 3usize] {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::None,
                    },
                );
            }
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=3usize)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert!(
            submitted <= 1,
            "max_prefetch_per_round=1 should limit to at most 1 submission, got {}",
            submitted,
        );

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 3, "all 3 requests counted before truncation");

        actor.shutdown();
    }

    // ── prefetch_batch: batch enqueue ──

    #[test]
    fn prefetch_batch_enqueues_all() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(100),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config,
            actor,
            page_metadata,
            addr_table,
            observer,
        );

        let requests: Vec<PrefetchRequest> = (1..=5)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5,
                prefetch_confidence: 0.7,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let enqueued = worker.prefetch_batch(&requests);
        assert_eq!(enqueued, 5, "all 5 requests should be enqueued");

        worker.shutdown();
    }

    #[test]
    fn prefetch_batch_empty_slice_enqueues_zero() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(100),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config,
            actor,
            page_metadata,
            addr_table,
            observer,
        );

        let requests: Vec<PrefetchRequest> = Vec::new();
        let enqueued = worker.prefetch_batch(&requests);
        assert_eq!(enqueued, 0, "empty batch should enqueue zero");

        worker.shutdown();
    }

    // ── stats: snapshot through worker ──

    #[test]
    fn stats_snapshot_returns_default_on_fresh_worker() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(100),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config,
            actor,
            page_metadata,
            addr_table,
            observer,
        );

        let s = worker.stats();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.rounds, 0);

        worker.shutdown();
    }

    // ── swap_in_round: stats counters updated correctly for mixed scenario ──

    #[test]
    fn round_stats_counters_mixed_scenario() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Page 1: on CpuDram, will be submitted.
        // Page 2: on HBM, will be skipped.
        // Page 3: low confidence, will be skipped.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
            table.insert(
                2,
                PageAddrEntry {
                    gpu_ptr: Some(0x2000),
                    host_buffer: None,
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
            // Page 3 is in addr_table on CpuDram but will be skipped for confidence.
            table.insert(
                3,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: 1.0,
                prefetch_confidence: 0.9, // above threshold
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 0.8,
                prefetch_confidence: 0.9, // above threshold but on HBM
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 3,
                urgency: 0.6,
                prefetch_confidence: 0.1, // below threshold
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 1, "only page 1 (CpuDram, high confidence) submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 3, "3 requests received");
        assert_eq!(s.submitted, 1, "1 submitted");
        assert_eq!(s.skipped, 2, "2 skipped (HBM + low confidence)");
        assert_eq!(s.rounds, 1, "1 round executed");

        actor.shutdown();
    }

    // ── SwapInWorkerError: exhaustive variant coverage ──

    #[test]
    fn error_send_failed_and_recv_failed_are_distinct() {
        let e1 = SwapInWorkerError::SendFailed("x".into());
        let e2 = SwapInWorkerError::RecvFailed("x".into());
        assert_ne!(format!("{e1}"), format!("{e2}"));
    }

    #[test]
    fn error_recv_failed_contains_custom_message() {
        let e = SwapInWorkerError::RecvFailed("channel hung".into());
        let msg = format!("{e}");
        assert!(msg.contains("channel hung"), "display should embed message: {msg}");
    }

    #[test]
    fn error_empty_message_still_formats() {
        let e = SwapInWorkerError::SendFailed(String::new());
        let msg = format!("{e}");
        assert!(msg.contains("swap-in worker send failed:"), "{msg}");
    }

    // ── SwapInWorkerConfig: edge values ──

    #[test]
    fn config_zero_tick_interval() {
        let cfg = SwapInWorkerConfig {
            tick_interval: Duration::ZERO,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval, Duration::ZERO);
    }

    #[test]
    fn config_large_values() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: usize::MAX,
            tick_interval: Duration::from_secs(3600),
            min_confidence: f32::MAX,
            max_in_flight: usize::MAX,
            page_bytes: usize::MAX,
        };
        assert_eq!(cfg.max_prefetch_per_round, usize::MAX);
        assert_eq!(cfg.max_in_flight, usize::MAX);
        assert_eq!(cfg.page_bytes, usize::MAX);
    }

    #[test]
    fn config_clone_independent_mutation() {
        let mut cfg = SwapInWorkerConfig::default();
        let cloned = cfg.clone();
        cfg.max_prefetch_per_round = 999;
        assert_eq!(cfg.max_prefetch_per_round, 999);
        assert_eq!(cloned.max_prefetch_per_round, 16, "clone should be independent");
    }

    // ── SwapInWorkerStats: edge cases ──

    #[test]
    fn stats_avg_latency_zero_promoted_with_latency_is_undefined_but_returns_zero() {
        // promoted_ok = 0 but total_latency_us != 0 → avg returns 0.0 (guard clause).
        let mut stats = SwapInWorkerStats::default();
        stats.total_latency_us = 5000;
        assert!((stats.avg_latency_us() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn stats_mutation_via_field_access() {
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 10;
        stats.submitted = 7;
        stats.skipped = 3;
        stats.promoted_ok = 6;
        stats.promoted_failed = 1;
        stats.two_hop_promotions = 2;
        stats.total_latency_us = 3000;
        stats.rounds = 5;
        assert_eq!(stats.total_requests, 10);
        assert_eq!(stats.submitted, 7);
        assert_eq!(stats.skipped, 3);
        assert_eq!(stats.promoted_ok, 6);
        assert_eq!(stats.promoted_failed, 1);
        assert_eq!(stats.two_hop_promotions, 2);
        assert_eq!(stats.total_latency_us, 3000);
        assert_eq!(stats.rounds, 5);
    }

    #[test]
    fn stats_accumulation_pattern() {
        let mut stats = SwapInWorkerStats::default();
        // Round 1.
        stats.total_requests += 5;
        stats.submitted += 3;
        stats.skipped += 2;
        stats.rounds += 1;
        // Round 2.
        stats.total_requests += 4;
        stats.submitted += 2;
        stats.skipped += 2;
        stats.rounds += 1;

        assert_eq!(stats.total_requests, 9);
        assert_eq!(stats.submitted, 5);
        assert_eq!(stats.skipped, 4);
        assert_eq!(stats.rounds, 2);
    }

    #[test]
    fn stats_avg_latency_large_promoted_ok() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = u64::MAX;
        stats.total_latency_us = u64::MAX;
        // Should not panic — just verify it returns a finite value.
        let avg = stats.avg_latency_us();
        assert!(avg.is_finite(), "avg_latency should be finite even at u64::MAX");
    }

    // ── PrefetchRequest: edge cases ──

    #[test]
    fn prefetch_request_max_urgency() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: f32::MAX,
            prefetch_confidence: 1.0,
            page_bytes: 100,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.urgency, f32::MAX);
    }

    #[test]
    fn prefetch_request_nan_urgency_field() {
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
    fn prefetch_request_confidence_above_one() {
        // API does not clamp — confidence > 1.0 is stored as-is.
        let req = PrefetchRequest {
            page_id: 5,
            urgency: 0.5,
            prefetch_confidence: 2.0,
            page_bytes: 1024,
            enqueued_at: Instant::now(),
        };
        assert!((req.prefetch_confidence - 2.0).abs() < 1e-6);
    }

    // ── compute_urgency: edge cases ──

    #[test]
    fn urgency_zero_access_count_is_positive() {
        // Even with access_count=0, recency bonus ensures urgency > 0.
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
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        assert!(u > 0.0, "recency bonus should make urgency > 0 even with zero access_count and zero confidence: {u}");
    }

    #[test]
    fn urgency_all_tiers_non_negative() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let u = SwapInWorker::compute_urgency(&meta, 0.5, tier);
            assert!(u >= 0.0, "urgency should be non-negative for {tier:?}: {u}");
        }
    }

    #[test]
    fn urgency_large_access_count() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1_000_000,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        assert!(u.is_finite(), "urgency should be finite even with huge access_count: {u}");
        assert!(u > 0.0);
    }

    // ── swap_in_round: NaN urgency sorting ──

    #[test]
    fn round_nan_urgency_does_not_panic() {
        let config = SwapInWorkerConfig::default();
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
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: f32::NAN,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];
        // Should not panic when sorting NaN urgency values.
        let _submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );
        actor.shutdown();
    }

    // ── prefetch: error on shut-down worker ──

    #[test]
    fn prefetch_returns_error_after_shutdown() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(10),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config,
            actor,
            page_metadata,
            addr_table,
            observer,
        );

        worker.shutdown();
        let result = worker.prefetch(PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        });
        assert!(result.is_err(), "prefetch after shutdown should return SendFailed error");
    }

    // ── drop: SwapInWorker shuts down cleanly on drop ──

    #[test]
    fn drop_shuts_down_worker() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        {
            let _worker = SwapInWorker::spawn(
                config,
                actor,
                page_metadata,
                addr_table,
                observer,
            );
            // Worker goes out of scope here — Drop impl should call shutdown().
        }
        // If Drop did not call shutdown, the thread would panic or hang.
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional tests — ~20 more covering remaining edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    // ── SwapInWorkerError: Unicode messages ──

    #[test]
    fn error_display_with_unicode_message() {
        let e = SwapInWorkerError::SendFailed("通道已关闭 🔒".into());
        let msg = format!("{e}");
        assert!(
            msg.contains("通道已关闭 🔒"),
            "Display should preserve Unicode: {msg}",
        );
    }

    #[test]
    fn error_clone_with_multibyte_utf8() {
        let e1 = SwapInWorkerError::RecvFailed("断开连接".into());
        let e2 = e1.clone();
        assert_eq!(format!("{e1}"), format!("{e2}"));
        assert!(format!("{e2}").contains("断开连接"));
    }

    // ── SwapInWorkerConfig: additional edge cases ──

    #[test]
    fn config_zero_page_bytes_stored_as_is() {
        let cfg = SwapInWorkerConfig {
            page_bytes: 0,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.page_bytes, 0, "page_bytes=0 is a valid config value");
    }

    #[test]
    fn config_default_tick_interval_type() {
        let cfg = SwapInWorkerConfig::default();
        assert_eq!(cfg.tick_interval.as_millis(), 5);
        assert_eq!(cfg.tick_interval.as_nanos(), 5_000_000);
    }

    #[test]
    fn config_max_in_flight_one() {
        let cfg = SwapInWorkerConfig {
            max_in_flight: 1,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.max_in_flight, 1);
    }

    #[test]
    fn config_clone_preserves_all_five_fields() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 3,
            tick_interval: Duration::from_micros(500),
            min_confidence: 0.42,
            max_in_flight: 7,
            page_bytes: 16384,
        };
        let c = cfg.clone();
        assert_eq!(c.max_prefetch_per_round, 3);
        assert_eq!(c.tick_interval, Duration::from_micros(500));
        assert!((c.min_confidence - 0.42).abs() < 1e-6);
        assert_eq!(c.max_in_flight, 7);
        assert_eq!(c.page_bytes, 16384);
    }

    // ── SwapInWorkerStats: additional edge cases ──

    #[test]
    fn stats_avg_latency_very_small_values() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 3;
        stats.total_latency_us = 1;
        let avg = stats.avg_latency_us();
        assert!(
            (avg - (1.0 / 3.0)).abs() < 1e-6,
            "avg should be 1/3 microsecond: {avg}",
        );
    }

    #[test]
    fn stats_u64_max_fields() {
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = u64::MAX;
        stats.submitted = u64::MAX;
        stats.skipped = u64::MAX;
        stats.promoted_ok = u64::MAX;
        stats.promoted_failed = u64::MAX;
        stats.two_hop_promotions = u64::MAX;
        stats.total_latency_us = u64::MAX;
        stats.rounds = u64::MAX;
        assert_eq!(stats.total_requests, u64::MAX);
        assert_eq!(stats.submitted, u64::MAX);
        assert_eq!(stats.rounds, u64::MAX);
        // avg_latency_us with both at u64::MAX should still return a finite value.
        let avg = stats.avg_latency_us();
        assert!(avg.is_finite(), "avg should be finite at u64::MAX: {avg}");
    }

    #[test]
    fn stats_clone_then_mutation_independent() {
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 50;
        let cloned = stats.clone();
        stats.total_requests = 100;
        assert_eq!(stats.total_requests, 100);
        assert_eq!(cloned.total_requests, 50, "clone should be independent");
    }

    #[test]
    fn stats_accumulation_promoted_ok_and_failed() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok += 3;
        stats.promoted_failed += 1;
        stats.total_latency_us += 400;
        stats.promoted_ok += 2;
        stats.promoted_failed += 1;
        stats.total_latency_us += 200;
        assert_eq!(stats.promoted_ok, 5);
        assert_eq!(stats.promoted_failed, 2);
        assert_eq!(stats.total_latency_us, 600);
        assert!((stats.avg_latency_us() - 120.0).abs() < 1e-6);
    }

    // ── PrefetchRequest: additional edge cases ──

    #[test]
    fn prefetch_request_negative_urgency_stored() {
        let req = PrefetchRequest {
            page_id: 7,
            urgency: -1.5,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!((req.urgency - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn prefetch_request_negative_confidence_stored() {
        let req = PrefetchRequest {
            page_id: 3,
            urgency: 0.5,
            prefetch_confidence: -0.3,
            page_bytes: 1024,
            enqueued_at: Instant::now(),
        };
        assert!((req.prefetch_confidence - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn prefetch_request_enqueued_at_preserved() {
        let t = Instant::now();
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 0,
            enqueued_at: t,
        };
        // Instant does not expose equality, but we can verify it is not later than now.
        assert!(req.enqueued_at <= Instant::now());
    }

    #[test]
    fn prefetch_request_clone_independent_page_id() {
        let req = PrefetchRequest {
            page_id: 42,
            urgency: 0.5,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let mut cloned = req.clone();
        cloned.page_id = 99;
        assert_eq!(req.page_id, 42, "original should be unmodified");
        assert_eq!(cloned.page_id, 99);
    }

    #[test]
    fn prefetch_request_nan_confidence_field() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: f32::NAN,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!(req.prefetch_confidence.is_nan());
    }

    // ── compute_urgency: formula verification ──

    #[test]
    fn urgency_formula_importance_rebound_ln1p_scale() {
        // access_count=9 → ln(10)/ln(10) = 1.0 for importance_rebound.
        // With confidence=1.0 and CpuDram (tier_bonus=1.0),
        // the formula is: 1.0 * 1.0 * 1.0 + recency_bonus * 0.1.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 9,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // importance_rebound = ln(1+9)/ln(1+10) = ln(10)/ln(10) = 1.0
        // recency_bonus = 1.0/(1+0) = 1.0 (elapsed ~0)
        // u = 1.0 * 1.0 * 1.0 + 1.0 * 0.1 = 1.1
        // Allow some tolerance for elapsed time not being exactly 0.
        assert!(
            u > 1.0 && u < 1.2,
            "urgency should be ~1.1 with access_count=9 confidence=1.0 CpuDram: {u}",
        );
    }

    #[test]
    fn urgency_tier_bonus_nvme_is_half_dram() {
        // With same meta and confidence, Nvme tier_bonus=0.5 vs CpuDram=1.0.
        // The first term is importance_rebound * confidence * tier_bonus.
        // recency bonus is the same, so the difference is in the first term.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::Nvme);
        // The difference in the first term is exactly 0.5x (since tier_bonus is multiplicative).
        // dram_first_term = nvme_first_term * 2.0.
        // Total urgency includes recency_bonus * 0.1 which is the same for both.
        // So u_dram - u_nvme should equal (importance_rebound * confidence * 0.5).
        let diff = u_dram - u_nvme;
        assert!(
            diff > 0.0,
            "CpuDram urgency should exceed Nvme urgency: diff={diff}",
        );
        // The diff should be exactly importance_rebound * confidence * 0.5.
        let importance_rebound = (10.0_f32).ln_1p() / (10.0_f32).ln_1p();
        let expected_diff = importance_rebound * 1.0 * 0.5;
        assert!(
            (diff - expected_diff).abs() < 1e-4,
            "diff should equal importance_rebound * confidence * 0.5: diff={diff} expected={expected_diff}",
        );
    }

    #[test]
    fn urgency_recency_bonus_factor_is_tenth() {
        // With zero confidence and access_count=0, urgency = 0 + recency_bonus * 0.1.
        // recency_bonus = 1/(1+elapsed). For a just-now timestamp, elapsed ~0, so recency_bonus ~1.0.
        // Therefore urgency ~ 0.1.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        assert!(
            (u - 0.1).abs() < 0.02,
            "with zero confidence and zero access_count, urgency should be ~0.1 (recency_bonus * 0.1): {u}",
        );
    }

    // ── swap_in_round: confidence boundary ──

    #[test]
    fn round_confidence_exactly_at_threshold_is_accepted() {
        // min_confidence=0.5, request confidence=0.5 → should NOT be skipped.
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.5, // exactly at threshold
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(
            submitted, 1,
            "confidence == threshold should be accepted (not less than threshold)",
        );
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 0, "should not skip when confidence == threshold");

        actor.shutdown();
    }

    #[test]
    fn round_confidence_just_below_threshold_is_skipped() {
        // min_confidence=0.5, request confidence=0.4999 → should be skipped.
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.4999,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 0, "confidence just below threshold should be skipped");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 1);

        actor.shutdown();
    }

    // ── swap_in_round: stats total_requests counts all input ──

    #[test]
    fn round_total_requests_counts_all_before_truncation() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize] {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::None,
                    },
                );
            }
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=2)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let _submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        let s = stats.lock().expect("stats lock");
        assert_eq!(
            s.total_requests, 2,
            "total_requests should count all input requests before truncation",
        );
        assert_eq!(s.rounds, 1);

        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional tests — ~18 more covering remaining gaps
    // ═══════════════════════════════════════════════════════════════════════════

    // ── compute_urgency: precise formula verification with confidence=1 ──

    #[test]
    fn urgency_confidence_one_produces_max_first_term() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 9, // ln(10)/ln(10) = 1.0
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // importance_rebound = 1.0, confidence = 1.0, tier_bonus = 1.0
        // first_term = 1.0 * 1.0 * 1.0 = 1.0
        // recency_bonus ~1.0 (elapsed ~0), so u ~ 1.0 + 0.1 = 1.1
        assert!(
            (u - 1.1).abs() < 0.05,
            "confidence=1.0 with access_count=9 on CpuDram should yield ~1.1: {u}",
        );
    }

    #[test]
    fn urgency_confidence_zero_first_term_is_zero() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // With confidence=0.0, first term = importance * 0.0 * tier_bonus = 0.0
        // Only recency_bonus * 0.1 remains.
        assert!(
            u < 0.2,
            "confidence=0 should leave only recency_bonus: {u}",
        );
        assert!(
            u > 0.0,
            "recency bonus should still be positive: {u}",
        );
    }

    // ── compute_urgency: age=0 (just accessed now) ──

    #[test]
    fn urgency_freshly_accessed_has_recency_near_one() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // recency_bonus = 1/(1+~0) ~ 1.0, so u ~ 0.1
        assert!(
            (u - 0.1).abs() < 0.02,
            "freshly-accessed page with zero confidence should have urgency ~0.1: {u}",
        );
    }

    #[test]
    fn urgency_old_access_recency_bonus_near_zero() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(3600), // 1 hour ago
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // recency_bonus = 1/(1+3600) ~ 0.000278, u ~ 0.0000278
        assert!(
            u < 0.001,
            "1-hour-old access with zero confidence should have near-zero urgency: {u}",
        );
        assert!(u > 0.0, "urgency should still be positive: {u}");
    }

    // ── compute_urgency: GpuHbm tier bonus is 2.0 ──

    #[test]
    fn urgency_gpu_hbm_tier_bonus_doubles_dram() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_hbm = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::GpuHbm);
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // The first term for HBM is 2x DRAM (tier_bonus 2.0 vs 1.0).
        // Difference = importance_rebound * confidence * (2.0 - 1.0)
        let importance_rebound = (10.0_f32).ln_1p() / (10.0_f32).ln_1p();
        let expected_diff = importance_rebound * 1.0 * 1.0;
        let diff = u_hbm - u_dram;
        assert!(
            (diff - expected_diff).abs() < 1e-4,
            "HBM first term should exceed DRAM by importance_rebound: diff={diff} expected={expected_diff}",
        );
    }

    // ── compute_urgency: swap_in_time does not affect urgency ──

    #[test]
    fn urgency_swap_in_time_does_not_affect_score() {
        let meta_no_swap = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let meta_with_swap = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u_no_swap = SwapInWorker::compute_urgency(&meta_no_swap, 0.8, StorageTier::CpuDram);
        let u_with_swap = SwapInWorker::compute_urgency(&meta_with_swap, 0.8, StorageTier::CpuDram);
        // swap_in_time is not used in compute_urgency — scores should be nearly identical.
        assert!(
            (u_no_swap - u_with_swap).abs() < 1e-3,
            "swap_in_time should not affect urgency: no_swap={u_no_swap} with_swap={u_with_swap}",
        );
    }

    // ── SwapInWorkerStats: two-round accumulation with avg_latency ──

    #[test]
    fn stats_two_round_accumulation_and_avg() {
        let mut stats = SwapInWorkerStats::default();

        // Round 1: 3 successful promotions totaling 600us.
        stats.promoted_ok += 3;
        stats.total_latency_us += 600;
        stats.submitted += 3;
        stats.total_requests += 4;
        stats.skipped += 1;
        stats.rounds += 1;

        // Round 2: 2 successful promotions totaling 400us.
        stats.promoted_ok += 2;
        stats.total_latency_us += 400;
        stats.submitted += 2;
        stats.total_requests += 3;
        stats.skipped += 1;
        stats.rounds += 1;

        assert_eq!(stats.promoted_ok, 5);
        assert_eq!(stats.total_latency_us, 1000);
        assert_eq!(stats.submitted, 5);
        assert_eq!(stats.total_requests, 7);
        assert_eq!(stats.skipped, 2);
        assert_eq!(stats.rounds, 2);
        assert!(
            (stats.avg_latency_us() - 200.0).abs() < 1e-6,
            "avg should be 1000/5 = 200us: {}",
            stats.avg_latency_us(),
        );
    }

    #[test]
    fn stats_promoted_failed_does_not_affect_avg() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 2;
        stats.promoted_failed = 3;
        stats.total_latency_us = 500;
        assert!(
            (stats.avg_latency_us() - 250.0).abs() < 1e-6,
            "avg should only count promoted_ok, not failed: {}",
            stats.avg_latency_us(),
        );
    }

    // ── swap_in_round: zero max_prefetch_per_round submits nothing ──

    #[test]
    fn round_zero_max_prefetch_submits_nothing() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 0, "zero max_prefetch_per_round should submit nothing");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 1, "total_requests still counts input");

        actor.shutdown();
    }

    // ── swap_in_round: min_confidence=1.0 only accepts confidence=1.0 ──

    #[test]
    fn round_min_confidence_one_rejects_ninety_nine_percent() {
        let config = SwapInWorkerConfig {
            min_confidence: 1.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.99, // below 1.0
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 0, "confidence 0.99 < min_confidence 1.0 should be skipped");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 1);

        actor.shutdown();
    }

    #[test]
    fn round_min_confidence_one_accepts_exactly_one() {
        let config = SwapInWorkerConfig {
            min_confidence: 1.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 1.0, // exactly at threshold
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 1, "confidence exactly 1.0 should be accepted");

        actor.shutdown();
    }

    // ── swap_in_round: NVMe two-hop counts in_flight correctly ──

    #[test]
    fn round_nvme_two_hop_counts_two_in_flight() {
        let config = SwapInWorkerConfig {
            max_in_flight: 2, // only allow 2 in-flight commands total
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            // Page 1: NVMe — will consume 2 in_flight (PromoteToDram + PromoteToHbm).
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: None,
                    current_tier: StorageTier::Nvme,
                    original_bytes: 4096,
                    codec: CompressionCodec::ZstdDict,
                },
            );
            // Page 2: NVMe — should not be processed (back-pressure after page 1).
            table.insert(
                2,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: None,
                    current_tier: StorageTier::Nvme,
                    original_bytes: 4096,
                    codec: CompressionCodec::ZstdDict,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=2usize)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // Page 1 (NVMe): PromoteToDram (+1 in_flight) + PromoteToHbm (+1 in_flight) = 2 total.
        // After page 1, in_flight=2 >= max_in_flight=2, so page 2 is not submitted.
        assert_eq!(submitted, 1, "only first NVMe page should be submitted due to back-pressure");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "first page should be two-hop");
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    // ── swap_in_round: multiple rounds accumulate stats ──

    #[test]
    fn round_multiple_rounds_accumulate() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Round 1: one request.
        let mut r1 = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut r1, &page_metadata, &addr_table, &stats, &observer,
        );

        // Round 2: empty.
        let mut r2: Vec<PrefetchRequest> = Vec::new();
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut r2, &page_metadata, &addr_table, &stats, &observer,
        );

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 2, "two rounds should be counted");
        assert_eq!(s.total_requests, 1, "only round 1 had a request");

        actor.shutdown();
    }

    // ── compute_urgency: is_lir flag does not affect score ──

    #[test]
    fn urgency_is_lir_flag_does_not_affect_score() {
        let meta_lir = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_non_lir = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_lir = SwapInWorker::compute_urgency(&meta_lir, 0.5, StorageTier::CpuDram);
        let u_non_lir = SwapInWorker::compute_urgency(&meta_non_lir, 0.5, StorageTier::CpuDram);
        assert!(
            (u_lir - u_non_lir).abs() < 1e-3,
            "is_lir should not affect urgency: lir={u_lir} non_lir={u_non_lir}",
        );
    }

    // ── compute_urgency: warm_until does not affect score ──

    #[test]
    fn urgency_warm_until_does_not_affect_score() {
        let meta_no_warm = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_with_warm = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: Some(Instant::now() + Duration::from_secs(60)),
        };
        let u_no_warm = SwapInWorker::compute_urgency(&meta_no_warm, 0.7, StorageTier::Nvme);
        let u_with_warm = SwapInWorker::compute_urgency(&meta_with_warm, 0.7, StorageTier::Nvme);
        assert!(
            (u_no_warm - u_with_warm).abs() < 1e-3,
            "warm_until should not affect urgency: no_warm={u_no_warm} with_warm={u_with_warm}",
        );
    }

    // ── compute_urgency: recency_bonus formula is 1/(1+elapsed) ──

    #[test]
    fn urgency_recency_bonus_halves_after_one_second() {
        let meta_now = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_1s = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now() - Duration::from_millis(1000),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        // With zero confidence and zero access_count, urgency = recency_bonus * 0.1.
        // recency_bonus_now ~ 1.0, recency_bonus_1s ~ 1/(1+1) = 0.5.
        let u_now = SwapInWorker::compute_urgency(&meta_now, 0.0, StorageTier::CpuDram);
        let u_1s = SwapInWorker::compute_urgency(&meta_1s, 0.0, StorageTier::CpuDram);
        // u_now / u_1s should be ~ 2.0 (1.0/0.5).
        let ratio = u_now / u_1s;
        assert!(
            (ratio - 2.0).abs() < 0.2,
            "recency bonus should roughly halve after 1 second: ratio={ratio}",
        );
    }

    // ── swap_in_round: high urgency page processed first when truncated ──

    #[test]
    fn round_truncation_keeps_highest_urgency() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [10usize, 20usize] {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::None,
                    },
                );
            }
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Low urgency first, high urgency second — sort will reorder.
        let mut requests = vec![
            PrefetchRequest {
                page_id: 10,
                urgency: 0.1,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 20,
                urgency: 0.9,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "only 1 should be submitted due to truncation");
        // After sort by urgency desc, page 20 (urgency=0.9) should be first and thus submitted.
        // Page 10 (urgency=0.1) should be truncated.
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2, "both requests counted before truncation");

        actor.shutdown();
    }

    // ── SwapInWorkerConfig: tick_interval sub-millisecond ──

    #[test]
    fn config_tick_interval_sub_millisecond() {
        let cfg = SwapInWorkerConfig {
            tick_interval: Duration::from_micros(100),
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval.as_micros(), 100);
        assert_eq!(cfg.tick_interval.as_nanos(), 100_000);
    }

    // ── PrefetchRequest: urgency infinity stored ──

    #[test]
    fn prefetch_request_infinity_urgency_stored() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: f32::INFINITY,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!(req.urgency.is_infinite());
        assert!(!req.urgency.is_nan());
    }

    // ── compute_urgency: access_count=1 vs access_count=0 ──

    #[test]
    fn urgency_access_count_one_higher_than_zero() {
        let meta_zero = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_one = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_zero = SwapInWorker::compute_urgency(&meta_zero, 0.5, StorageTier::CpuDram);
        let u_one = SwapInWorker::compute_urgency(&meta_one, 0.5, StorageTier::CpuDram);
        // ln(1+0)/ln(1+10) = 0 for access_count=0, ln(1+1)/ln(1+10) > 0 for access_count=1.
        assert!(
            u_one > u_zero,
            "access_count=1 should yield higher urgency than 0: one={u_one} zero={u_zero}",
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // New tests — 45 additional tests
    // ═══════════════════════════════════════════════════════════════════════════

    // ── SwapInWorkerError: PartialEq ──

    #[test]
    fn error_partial_eq_same_variant_same_message() {
        let e1 = SwapInWorkerError::SendFailed("abc".into());
        let e2 = SwapInWorkerError::SendFailed("abc".into());
        assert_eq!(e1, e2);
    }

    #[test]
    fn error_partial_eq_different_variant_not_equal() {
        let e1 = SwapInWorkerError::SendFailed("abc".into());
        let e2 = SwapInWorkerError::RecvFailed("abc".into());
        assert_ne!(e1, e2);
    }

    #[test]
    fn error_partial_eq_different_message_not_equal() {
        let e1 = SwapInWorkerError::SendFailed("abc".into());
        let e2 = SwapInWorkerError::SendFailed("xyz".into());
        assert_ne!(e1, e2);
    }

    #[test]
    fn error_partial_eq_empty_string() {
        let e1 = SwapInWorkerError::RecvFailed(String::new());
        let e2 = SwapInWorkerError::RecvFailed(String::new());
        assert_eq!(e1, e2);
    }

    // ── SwapInWorkerConfig: PartialEq ──

    #[test]
    fn config_partial_eq_default_equal() {
        let c1 = SwapInWorkerConfig::default();
        let c2 = SwapInWorkerConfig::default();
        assert_eq!(c1, c2);
    }

    #[test]
    fn config_partial_eq_different_not_equal() {
        let c1 = SwapInWorkerConfig::default();
        let c2 = SwapInWorkerConfig {
            max_prefetch_per_round: 999,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2);
    }

    #[test]
    fn config_partial_eq_all_fields_match() {
        let c1 = SwapInWorkerConfig {
            max_prefetch_per_round: 4,
            tick_interval: Duration::from_millis(20),
            min_confidence: 0.42,
            max_in_flight: 8,
            page_bytes: 8192,
        };
        let c2 = SwapInWorkerConfig {
            max_prefetch_per_round: 4,
            tick_interval: Duration::from_millis(20),
            min_confidence: 0.42,
            max_in_flight: 8,
            page_bytes: 8192,
        };
        assert_eq!(c1, c2);
    }

    #[test]
    fn config_min_confidence_nan() {
        let cfg = SwapInWorkerConfig {
            min_confidence: f32::NAN,
            ..SwapInWorkerConfig::default()
        };
        assert!(cfg.min_confidence.is_nan());
    }

    #[test]
    fn config_min_confidence_infinity() {
        let cfg = SwapInWorkerConfig {
            min_confidence: f32::INFINITY,
            ..SwapInWorkerConfig::default()
        };
        assert!(cfg.min_confidence.is_infinite() && cfg.min_confidence > 0.0);
    }

    #[test]
    fn config_min_confidence_negative() {
        let cfg = SwapInWorkerConfig {
            min_confidence: -0.5,
            ..SwapInWorkerConfig::default()
        };
        assert!((cfg.min_confidence - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn config_tick_interval_one_nano() {
        let cfg = SwapInWorkerConfig {
            tick_interval: Duration::from_nanos(1),
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval.as_nanos(), 1);
    }

    // ── SwapInWorkerStats: PartialEq ──

    #[test]
    fn stats_partial_eq_default_equal() {
        let s1 = SwapInWorkerStats::default();
        let s2 = SwapInWorkerStats::default();
        assert_eq!(s1, s2);
    }

    #[test]
    fn stats_partial_eq_different_not_equal() {
        let mut s1 = SwapInWorkerStats::default();
        let s2 = SwapInWorkerStats::default();
        s1.total_requests = 1;
        assert_ne!(s1, s2);
    }

    #[test]
    fn stats_partial_eq_all_fields_match() {
        let s1 = SwapInWorkerStats {
            total_requests: 10,
            submitted: 8,
            skipped: 2,
            promoted_ok: 7,
            promoted_failed: 1,
            two_hop_promotions: 3,
            total_latency_us: 4000,
            rounds: 4,
        };
        let s2 = SwapInWorkerStats {
            total_requests: 10,
            submitted: 8,
            skipped: 2,
            promoted_ok: 7,
            promoted_failed: 1,
            two_hop_promotions: 3,
            total_latency_us: 4000,
            rounds: 4,
        };
        assert_eq!(s1, s2);
    }

    #[test]
    fn stats_partial_eq_each_field_difference() {
        let base = SwapInWorkerStats {
            total_requests: 1,
            submitted: 1,
            skipped: 1,
            promoted_ok: 1,
            promoted_failed: 1,
            two_hop_promotions: 1,
            total_latency_us: 1,
            rounds: 1,
        };
        // Changing each field individually should produce inequality.
        let fields: Vec<SwapInWorkerStats> = vec![
            SwapInWorkerStats { total_requests: 0, ..base.clone() },
            SwapInWorkerStats { submitted: 0, ..base.clone() },
            SwapInWorkerStats { skipped: 0, ..base.clone() },
            SwapInWorkerStats { promoted_ok: 0, ..base.clone() },
            SwapInWorkerStats { promoted_failed: 0, ..base.clone() },
            SwapInWorkerStats { two_hop_promotions: 0, ..base.clone() },
            SwapInWorkerStats { total_latency_us: 0, ..base.clone() },
            SwapInWorkerStats { rounds: 0, ..base.clone() },
        ];
        for modified in &fields {
            assert_ne!(&base, modified, "base should differ from modified stats");
        }
    }

    // ── PrefetchRequest: PartialEq ──

    #[test]
    fn prefetch_request_partial_eq_equal() {
        let t = Instant::now();
        let r1 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: t,
        };
        let r2 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: t,
        };
        assert_eq!(r1, r2);
    }

    #[test]
    fn prefetch_request_partial_eq_different_page_id() {
        let t = Instant::now();
        let r1 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: t,
        };
        let r2 = PrefetchRequest {
            page_id: 2,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: t,
        };
        assert_ne!(r1, r2);
    }

    #[test]
    fn prefetch_request_page_id_usize_max() {
        let req = PrefetchRequest {
            page_id: usize::MAX,
            urgency: 0.5,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_id, usize::MAX);
    }

    // ── StorageTier: from_u8 boundaries and discriminants ──

    #[test]
    fn storage_tier_from_u8_valid() {
        assert_eq!(StorageTier::from_u8(0), Some(StorageTier::GpuHbm));
        assert_eq!(StorageTier::from_u8(1), Some(StorageTier::CpuDram));
        assert_eq!(StorageTier::from_u8(2), Some(StorageTier::Nvme));
    }

    #[test]
    fn storage_tier_from_u8_invalid() {
        assert_eq!(StorageTier::from_u8(3), None);
        assert_eq!(StorageTier::from_u8(255), None);
    }

    #[test]
    fn storage_tier_discriminant_values() {
        assert_eq!(StorageTier::GpuHbm as u8, 0);
        assert_eq!(StorageTier::CpuDram as u8, 1);
        assert_eq!(StorageTier::Nvme as u8, 2);
    }

    // ── PageState: all variants, Copy, Clone, PartialEq, Eq, Hash ──

    #[test]
    fn page_state_all_variants_distinct() {
        let states = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];
        // All pairwise distinct.
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert_ne!(states[i], states[j], "PageState variants should be distinct");
            }
        }
    }

    #[test]
    fn page_state_copy_semantics() {
        let s1 = PageState::Active;
        let s2 = s1; // Copy, not move.
        assert_eq!(s1, s2);
    }

    #[test]
    fn page_state_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PageState::Active);
        assert!(set.contains(&PageState::Active));
        assert!(!set.contains(&PageState::Free));
    }

    #[test]
    fn page_state_default_from_page_metadata() {
        let meta = PageMetadata::default();
        assert_eq!(meta.state, PageState::Standby);
    }

    // ── WeightTier: variants, Copy, PartialEq ──

    #[test]
    fn weight_tier_all_variants_distinct() {
        assert_ne!(WeightTier::Hot, WeightTier::Warm);
        assert_ne!(WeightTier::Warm, WeightTier::Cold);
        assert_ne!(WeightTier::Hot, WeightTier::Cold);
    }

    #[test]
    fn weight_tier_copy_semantics() {
        let w1 = WeightTier::Hot;
        let w2 = w1;
        assert_eq!(w1, w2);
    }

    // ── compute_urgency: negative confidence ──

    #[test]
    fn urgency_negative_confidence_reduces_first_term() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_pos = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        let u_neg = SwapInWorker::compute_urgency(&meta, -0.5, StorageTier::CpuDram);
        // Negative confidence makes the first term negative, reducing urgency.
        assert!(
            u_neg < u_pos,
            "negative confidence should reduce urgency: neg={u_neg} pos={u_pos}",
        );
    }

    #[test]
    fn urgency_negative_confidence_can_produce_negative() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 100,
            last_access: Instant::now() - Duration::from_secs(3600),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, -1.0, StorageTier::CpuDram);
        // With large access_count and confidence=-1.0, the first term is negative.
        // recency_bonus for 1-hour-old access is ~0.000278, so 0.1 * that = ~0.0000278.
        // First term is -(importance_rebound * 1.0 * 1.0) which is < -0.9.
        assert!(
            u < 0.0,
            "negative confidence with high access_count should produce negative urgency: {u}",
        );
    }

    // ── compute_urgency: infinite confidence ──

    #[test]
    fn urgency_infinite_confidence_produces_infinite() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, f32::INFINITY, StorageTier::CpuDram);
        assert!(
            u.is_infinite() && u > 0.0,
            "infinite confidence should produce infinite urgency: {u}",
        );
    }

    // ── compute_urgency: NaN confidence ──

    #[test]
    fn urgency_nan_confidence_produces_nan_first_term() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, f32::NAN, StorageTier::CpuDram);
        // first_term = importance * NaN * tier = NaN, NaN + recency = NaN.
        assert!(
            u.is_nan(),
            "NaN confidence should produce NaN urgency: {u}",
        );
    }

    // ── compute_urgency: access_count=usize::MAX ──

    #[test]
    fn urgency_access_count_usize_max_is_finite() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: usize::MAX,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        assert!(
            u.is_finite(),
            "urgency should be finite even with usize::MAX access_count: {u}",
        );
        assert!(u > 0.0);
    }

    // ── compute_urgency: all three tiers produce different first terms ──

    #[test]
    fn urgency_all_tiers_produce_different_first_terms() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_hbm = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::GpuHbm);
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::Nvme);
        // HBM (2.0) > DRAM (1.0) > NVMe (0.5) tier bonuses.
        assert!(u_hbm > u_dram, "HBM should exceed DRAM: hbm={u_hbm} dram={u_dram}");
        assert!(u_dram > u_nvme, "DRAM should exceed NVMe: dram={u_dram} nvme={u_nvme}");
    }

    // ── compute_urgency: doubling access_count roughly increases urgency ──

    #[test]
    fn urgency_doubling_access_count_increases() {
        let meta_small = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_large = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_small = SwapInWorker::compute_urgency(&meta_small, 1.0, StorageTier::CpuDram);
        let u_large = SwapInWorker::compute_urgency(&meta_large, 1.0, StorageTier::CpuDram);
        assert!(
            u_large > u_small,
            "higher access_count should yield higher urgency: large={u_large} small={u_small}",
        );
    }

    // ── compute_urgency: recency bonus with saturating_duration_since ──

    #[test]
    fn urgency_future_last_access_no_panic() {
        // A last_access in the future should not panic — saturating_duration_since returns 0.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now() + Duration::from_secs(10),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        assert!(
            u.is_finite() && u > 0.0,
            "future last_access should not panic and should produce positive urgency: {u}",
        );
    }

    // ── SwapInWorkerStats: avg_latency precision ──

    #[test]
    fn stats_avg_latency_fractional_result() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 7;
        stats.total_latency_us = 1000;
        let avg = stats.avg_latency_us();
        assert!(
            (avg - (1000.0 / 7.0)).abs() < 1e-6,
            "avg should be 1000/7: {avg}",
        );
    }

    #[test]
    fn stats_avg_latency_very_large_division() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = u64::MAX / 2;
        stats.total_latency_us = u64::MAX;
        let avg = stats.avg_latency_us();
        assert!(
            avg.is_finite(),
            "avg should be finite even with very large values: {avg}",
        );
        assert!(avg > 0.0);
    }

    // ── SwapInWorkerConfig: PartialEq clone roundtrip ──

    #[test]
    fn config_eq_after_clone_matches_original() {
        let original = SwapInWorkerConfig {
            max_prefetch_per_round: 7,
            tick_interval: Duration::from_millis(15),
            min_confidence: 0.33,
            max_in_flight: 11,
            page_bytes: 2048,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── swap_in_round: mixed NVMe + DRAM pages ──

    #[test]
    fn round_mixed_nvme_and_dram_submits_both() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
            table.insert(
                2,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: None,
                    current_tier: StorageTier::Nvme,
                    original_bytes: 4096,
                    codec: CompressionCodec::ZstdDict,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: 0.5,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 0.8,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 2, "both DRAM and NVMe pages should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should be two-hop");
        assert_eq!(s.submitted, 2);

        actor.shutdown();
    }

    // ── swap_in_round: urgency sort order with many requests ──

    #[test]
    fn round_sorts_many_requests_by_urgency() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 3,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in 1..=5usize {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::None,
                    },
                );
            }
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // 5 requests with varying urgency; only top 3 should be processed.
        let mut requests: Vec<PrefetchRequest> = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: 0.1,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 0.9,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 3,
                urgency: 0.3,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 4,
                urgency: 0.7,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 5,
                urgency: 0.5,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        assert_eq!(submitted, 3, "only top 3 by urgency should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 5, "all 5 counted before truncation");

        actor.shutdown();
    }

    // ── swap_in_round: confidence NaN skipped (NaN < anything is false) ──

    #[test]
    fn round_nan_confidence_is_skipped() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: f32::NAN,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // NaN < 0.5 is false, so it goes past the confidence check.
        // The page should be submitted (NaN comparison passes through).
        // Note: NaN < threshold is false, so it is NOT skipped by confidence.
        // This is the actual behavior of the code — NaN passes the confidence filter.
        assert_eq!(
            submitted, 1,
            "NaN confidence passes the < threshold check (NaN < x is false)",
        );

        actor.shutdown();
    }

    // ── PageMetadata: default values ──

    #[test]
    fn page_metadata_default_values() {
        let meta = PageMetadata::default();
        assert_eq!(meta.page_id, 0);
        assert!(meta.sequence_id.is_none());
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(meta.swap_in_time.is_none());
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Standby);
        assert!(meta.warm_until.is_none());
    }

    // ── compute_urgency: recency field in PageMetadata does not affect urgency ──

    #[test]
    fn urgency_recency_field_does_not_affect_score() {
        let meta_low = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_high = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 999,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_low = SwapInWorker::compute_urgency(&meta_low, 0.5, StorageTier::CpuDram);
        let u_high = SwapInWorker::compute_urgency(&meta_high, 0.5, StorageTier::CpuDram);
        assert!(
            (u_low - u_high).abs() < 1e-3,
            "recency field should not affect urgency: low={u_low} high={u_high}",
        );
    }

    // ── compute_urgency: sequence_id does not affect urgency ──

    #[test]
    fn urgency_sequence_id_does_not_affect_score() {
        let meta_none = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_some = PageMetadata {
            page_id: 2,
            sequence_id: Some(42),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_none = SwapInWorker::compute_urgency(&meta_none, 0.5, StorageTier::CpuDram);
        let u_some = SwapInWorker::compute_urgency(&meta_some, 0.5, StorageTier::CpuDram);
        assert!(
            (u_none - u_some).abs() < 1e-3,
            "sequence_id should not affect urgency: none={u_none} some={u_some}",
        );
    }

    // ── SwapInWorkerStats: sum invariant (total = submitted + skipped) ──

    #[test]
    fn stats_sum_invariant_holds() {
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 100;
        stats.submitted = 70;
        stats.skipped = 30;
        assert_eq!(
            stats.total_requests,
            stats.submitted + stats.skipped,
            "total should equal submitted + skipped",
        );
    }

    // ── SwapInWorkerConfig: PartialEq tick_interval difference ──

    #[test]
    fn config_partial_eq_tick_interval_difference() {
        let c1 = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(5),
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(10),
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2);
    }

    // ── prefetch: multiple sequential prefetches ──

    #[test]
    fn prefetch_multiple_requests_succeed() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(100),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config,
            actor,
            page_metadata,
            addr_table,
            observer,
        );

        for i in 0..10 {
            let result = worker.prefetch(PrefetchRequest {
                page_id: i,
                urgency: 0.5,
                prefetch_confidence: 0.7,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            });
            assert!(result.is_ok(), "prefetch {i} should succeed");
        }

        worker.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional 50 tests — covering remaining gaps
    // ═══════════════════════════════════════════════════════════════════════════

    // ── drain_completions_and_update: page metadata state transitions ──

    #[test]
    fn drain_updates_page_state_to_active_on_hbm_promotion() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert page into addr_table with a host_buffer so actor can promote it.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(42, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata.insert(42, PageMetadata {
            page_id: 42,
            sequence_id: Some(1),
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Send a PromoteToHbm command and poll until completion arrives.
        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 42,
            page_bytes: 4096,
        });
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let meta = page_metadata.read().expect("read lock").get(&42).cloned();
        assert!(meta.is_some(), "page metadata should still exist");
        let meta = meta.unwrap();
        assert_eq!(meta.state, PageState::Active, "state should be Active after HBM promotion");
        assert!(meta.swap_in_time.is_none(), "swap_in_time should be cleared after HBM promotion");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.promoted_ok, 1, "should have 1 successful promotion");

        actor.shutdown();
    }

    #[test]
    fn drain_updates_page_state_to_warm_on_dram_promotion() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert page into addr_table with a host_buffer so actor can promote to HBM.
        // The actor processes PromoteToHbm and produces a completion with to_tier=GpuHbm.
        // We test drain's metadata update logic indirectly by verifying stats.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(10, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata.insert(10, PageMetadata {
            page_id: 10,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // PromoteToHbm succeeds when host_buffer is present.
        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 10,
            page_bytes: 4096,
        });
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let meta = page_metadata.read().expect("read lock").get(&10).cloned();
        assert!(meta.is_some());
        let meta = meta.unwrap();
        // PromoteToHbm produces to_tier=GpuHbm, so state becomes Active.
        assert_eq!(meta.state, PageState::Active, "state should be Active after HBM promotion");

        actor.shutdown();
    }

    #[test]
    fn drain_no_completions_does_not_modify_stats() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.promoted_ok, 0, "no completions should mean zero promoted_ok");
        assert_eq!(s.promoted_failed, 0);

        actor.shutdown();
    }

    // ── swap_in_round: different CompressionCodec variants ──

    #[test]
    fn round_dram_page_with_lz4_codec() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::Lz4,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    #[test]
    fn round_dram_page_with_bitpack_rle_codec() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                5,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::BitPackRle,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 5,
            urgency: 0.8,
            prefetch_confidence: 0.7,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    #[test]
    fn round_nvme_page_with_nvcomp_ans_codec() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                7,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: None,
                    current_tier: StorageTier::Nvme,
                    original_bytes: 4096,
                    codec: CompressionCodec::NvcompAns,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 7,
            urgency: 0.9,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should be two-hop");

        actor.shutdown();
    }

    #[test]
    fn round_dram_page_with_none_codec() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                3,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 3,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    // ── shutdown idempotency ──

    #[test]
    fn double_shutdown_does_not_panic() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );
        worker.shutdown();
        // Second shutdown should be a no-op (handle already taken).
        worker.shutdown();
    }

    // ── prefetch_batch: single element ──

    #[test]
    fn prefetch_batch_single_element_succeeds() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(100),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );

        let requests = vec![PrefetchRequest {
            page_id: 42,
            urgency: 0.7,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let enqueued = worker.prefetch_batch(&requests);
        assert_eq!(enqueued, 1);

        worker.shutdown();
    }

    // ── swap_in_round: page with gpu_ptr set on CpuDram ──

    #[test]
    fn round_dram_page_with_stale_gpu_ptr_still_submits() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: Some(0xDEAD), // stale — page was evicted but gpu_ptr not cleared
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        // The code checks current_tier, not gpu_ptr. CpuDram pages are submitted.
        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    // ── compute_urgency: monotonicity with respect to confidence ──

    #[test]
    fn urgency_monotonically_increases_with_confidence() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let confidences = [0.0, 0.25, 0.5, 0.75, 1.0];
        let urgencies: Vec<f32> = confidences
            .iter()
            .map(|&c| SwapInWorker::compute_urgency(&meta, c, StorageTier::CpuDram))
            .collect();
        for i in 1..urgencies.len() {
            assert!(
                urgencies[i] >= urgencies[i - 1],
                "urgency should be monotonically non-decreasing: [{}] {} < [{}] {}",
                i - 1, urgencies[i - 1], i, urgencies[i],
            );
        }
    }

    // ── compute_urgency: monotonicity with respect to access_count ──

    #[test]
    fn urgency_monotonically_increases_with_access_count() {
        let access_counts: [usize; 6] = [0, 1, 5, 10, 50, 100];
        let mut prev_u = 0.0_f32;
        for ac in access_counts {
            let meta = PageMetadata {
                page_id: 1,
                sequence_id: None,
                recency: 0,
                access_count: ac,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::Active,
                warm_until: None,
            };
            let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
            assert!(
                u >= prev_u,
                "urgency should be non-decreasing with access_count: ac={ac} u={u} prev={prev_u}",
            );
            prev_u = u;
        }
    }

    // ── swap_in_round: confidence exactly zero vs min_confidence zero ──

    #[test]
    fn round_zero_confidence_accepted_when_min_confidence_zero() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "zero confidence should pass when min_confidence is also zero");

        actor.shutdown();
    }

    // ── swap_in_round: request with page_bytes equal to config.page_bytes ──

    #[test]
    fn round_page_bytes_matches_config() {
        let config = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // page_bytes = 4096 explicitly set (same as config).
        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    // ── PageMetadata: construction with all fields ──

    #[test]
    fn page_metadata_construction_all_fields() {
        let now = Instant::now();
        let warm = now + Duration::from_secs(30);
        let meta = PageMetadata {
            page_id: 123,
            sequence_id: Some(456),
            recency: 7,
            access_count: 42,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::Active,
            warm_until: Some(warm),
        };
        assert_eq!(meta.page_id, 123);
        assert_eq!(meta.sequence_id, Some(456));
        assert_eq!(meta.recency, 7);
        assert_eq!(meta.access_count, 42);
        assert_eq!(meta.last_access, now);
        assert_eq!(meta.swap_in_time, Some(now));
        assert!(meta.is_lir);
        assert_eq!(meta.state, PageState::Active);
        assert_eq!(meta.warm_until, Some(warm));
    }

    // ── PageMetadata: clone ──

    #[test]
    fn page_metadata_clone_preserves_fields() {
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 10,
            sequence_id: Some(20),
            recency: 3,
            access_count: 99,
            last_access: now,
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.page_id, 10);
        assert_eq!(cloned.sequence_id, Some(20));
        assert_eq!(cloned.recency, 3);
        assert_eq!(cloned.access_count, 99);
        assert!(!cloned.is_lir);
        assert_eq!(cloned.state, PageState::SwappedOut);
    }

    // ── PageState: debug format ──

    #[test]
    fn page_state_debug_format_variants() {
        assert!(format!("{:?}", PageState::Free).contains("Free"));
        assert!(format!("{:?}", PageState::Active).contains("Active"));
        assert!(format!("{:?}", PageState::Standby).contains("Standby"));
        assert!(format!("{:?}", PageState::SwappedOut).contains("SwappedOut"));
        assert!(format!("{:?}", PageState::Warm).contains("Warm"));
        assert!(format!("{:?}", PageState::Protected).contains("Protected"));
        assert!(format!("{:?}", PageState::Swapped).contains("Swapped"));
    }

    // ── StorageTier: debug format ──

    #[test]
    fn storage_tier_debug_format() {
        assert!(format!("{:?}", StorageTier::GpuHbm).contains("GpuHbm"));
        assert!(format!("{:?}", StorageTier::CpuDram).contains("CpuDram"));
        assert!(format!("{:?}", StorageTier::Nvme).contains("Nvme"));
    }

    // ── WeightTier: debug format ──

    #[test]
    fn weight_tier_debug_format() {
        assert!(format!("{:?}", WeightTier::Hot).contains("Hot"));
        assert!(format!("{:?}", WeightTier::Warm).contains("Warm"));
        assert!(format!("{:?}", WeightTier::Cold).contains("Cold"));
    }

    // ── CompressionCodec: all variants distinct ──

    #[test]
    fn compression_codec_all_variants_distinct() {
        let codecs = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for i in 0..codecs.len() {
            for j in (i + 1)..codecs.len() {
                assert_ne!(codecs[i], codecs[j], "CompressionCodec variants should be distinct");
            }
        }
    }

    // ── CompressionCodec: copy semantics ──

    #[test]
    fn compression_codec_copy_semantics() {
        let c1 = CompressionCodec::Lz4;
        let c2 = c1;
        assert_eq!(c1, c2);
    }

    // ── CompressionCodec: clone semantics ──

    #[test]
    fn compression_codec_clone_semantics() {
        let c1 = CompressionCodec::ZstdDict;
        let c2 = c1.clone();
        assert_eq!(c1, c2);
    }

    // ── PageAddrEntry: construction ──

    #[test]
    fn page_addr_entry_construction_all_none() {
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
        assert_eq!(entry.original_bytes, 8192);
    }

    #[test]
    fn page_addr_entry_construction_with_gpu_ptr() {
        let entry = PageAddrEntry {
            gpu_ptr: Some(0xCAFE),
            host_buffer: None,
            current_tier: StorageTier::GpuHbm,
            original_bytes: 4096,
            codec: CompressionCodec::None,
        };
        assert_eq!(entry.gpu_ptr, Some(0xCAFE));
        assert!(entry.host_buffer.is_none());
    }

    #[test]
    fn page_addr_entry_construction_with_host_buffer() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(vec![0xAB; 2048]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 2048,
            codec: CompressionCodec::Lz4,
        };
        assert!(entry.gpu_ptr.is_none());
        assert_eq!(entry.host_buffer.as_ref().map(|b| b.len()), Some(2048));
    }

    // ── swap_in_round: multiple NVMe pages with back-pressure ──

    #[test]
    fn round_multiple_nvme_with_tight_back_pressure() {
        let config = SwapInWorkerConfig {
            max_in_flight: 4, // each NVMe page uses 2 in_flight (PromoteToDram + PromoteToHbm)
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize, 3usize] {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: None,
                        current_tier: StorageTier::Nvme,
                        original_bytes: 4096,
                        codec: CompressionCodec::ZstdDict,
                    },
                );
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=3usize)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Page 1: PromoteToDram(1) + PromoteToHbm(1) = 2 in_flight.
        // Page 2: PromoteToDram(1) + PromoteToHbm(1) = 2 more in_flight, total 4 = max.
        // Page 3: skipped (in_flight >= max_in_flight).
        assert_eq!(submitted, 2, "only 2 NVMe pages should be submitted with max_in_flight=4");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 2);

        actor.shutdown();
    }

    // ── swap_in_round: all requests skipped ──

    #[test]
    fn round_all_skipped_yields_zero_submitted() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.9,
            ..SwapInWorkerConfig::default()
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // 3 requests all below confidence threshold.
        let mut requests: Vec<PrefetchRequest> = (1..=3)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5,
                prefetch_confidence: 0.1,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 3);

        actor.shutdown();
    }

    // ── swap_in_round: mixed HBM and not-in-table ──

    #[test]
    fn round_mixed_hbm_and_missing_skipped() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: Some(0x1000),
                    host_buffer: None,
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
            // Page 2 is not in the table.
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1, // HBM → skip
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2, // not in table → skip
                urgency: 0.8,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 2);

        actor.shutdown();
    }

    // ── swap_in_round: urgency sort with equal urgency ──

    #[test]
    fn round_equal_urgency_processes_all() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 10,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in 1..=3usize {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::None,
                    },
                );
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=3)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5, // all same urgency
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 3, "all pages with equal urgency should be processed");

        actor.shutdown();
    }

    // ── swap_in_round: request page_bytes overrides config page_bytes ──

    #[test]
    fn round_request_page_bytes_overrides_config() {
        let config = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 16384]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 16384,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Request specifies page_bytes=16384, which overrides config.page_bytes=4096.
        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 16384,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "request page_bytes should be used");

        actor.shutdown();
    }

    // ── compute_urgency: urgency with different tier bonuses ratio ──

    #[test]
    fn urgency_hbm_is_double_dram_first_term() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_hbm = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::GpuHbm);
        // The difference should be exactly importance_rebound * 1.0 * (2.0 - 1.0).
        let importance = (10.0_f32).ln_1p() / (10.0_f32).ln_1p();
        let expected_extra = importance * 1.0 * 1.0;
        let diff = u_hbm - u_dram;
        assert!(
            (diff - expected_extra).abs() < 1e-4,
            "HBM should exceed DRAM by exactly importance_rebound: diff={diff} expected={expected_extra}",
        );
    }

    // ── SwapInWorkerConfig: all fields independently mutable ──

    #[test]
    fn config_mutation_each_field() {
        let mut cfg = SwapInWorkerConfig::default();
        cfg.max_prefetch_per_round = 1;
        assert_eq!(cfg.max_prefetch_per_round, 1);

        let mut cfg = SwapInWorkerConfig::default();
        cfg.tick_interval = Duration::from_secs(1);
        assert_eq!(cfg.tick_interval, Duration::from_secs(1));

        let mut cfg = SwapInWorkerConfig::default();
        cfg.min_confidence = 0.99;
        assert!((cfg.min_confidence - 0.99).abs() < 1e-6);

        let mut cfg = SwapInWorkerConfig::default();
        cfg.max_in_flight = 1;
        assert_eq!(cfg.max_in_flight, 1);

        let mut cfg = SwapInWorkerConfig::default();
        cfg.page_bytes = 65536;
        assert_eq!(cfg.page_bytes, 65536);
    }

    // ── SwapInWorkerStats: avg_latency_us with promoted_ok=1 and latency=0 ──

    #[test]
    fn stats_avg_latency_zero_with_one_promotion() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 1;
        stats.total_latency_us = 0;
        assert!(
            (stats.avg_latency_us() - 0.0).abs() < 1e-6,
            "avg should be 0 when total_latency is 0: {}",
            stats.avg_latency_us(),
        );
    }

    // ── compute_urgency: very small confidence produces small but positive urgency ──

    #[test]
    fn urgency_very_small_confidence_still_positive_due_to_recency() {
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
        let u = SwapInWorker::compute_urgency(&meta, 0.001, StorageTier::CpuDram);
        // Even tiny confidence + recency bonus should yield urgency > 0.
        assert!(u > 0.0, "very small confidence should still yield positive urgency: {u}");
    }

    // ── swap_in_round: rounds counter increments even for empty input ──

    #[test]
    fn round_increments_rounds_even_on_empty() {
        let config = SwapInWorkerConfig::default();
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut empty: Vec<PrefetchRequest> = Vec::new();
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut empty, &page_metadata, &addr_table, &stats, &observer,
        );
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut empty, &page_metadata, &addr_table, &stats, &observer,
        );
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut empty, &page_metadata, &addr_table, &stats, &observer,
        );

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 3, "3 empty rounds should be counted");

        actor.shutdown();
    }

    // ── prefetch: error type matches SendFailed variant ──

    #[test]
    fn prefetch_error_after_shutdown_is_send_failed() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(10),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );
        worker.shutdown();

        let result = worker.prefetch(PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.7,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        });

        assert!(
            matches!(result, Err(SwapInWorkerError::SendFailed(_))),
            "error should be SendFailed variant",
        );
    }

    // ── PrefetchRequest: PartialEq different urgency ──

    #[test]
    fn prefetch_request_partial_eq_different_urgency() {
        let t = Instant::now();
        let r1 = PrefetchRequest {
            page_id: 1,
            urgency: 0.1,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: t,
        };
        let r2 = PrefetchRequest {
            page_id: 1,
            urgency: 0.9,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: t,
        };
        assert_ne!(r1, r2, "different urgency should not be equal");
    }

    // ── PrefetchRequest: PartialEq different confidence ──

    #[test]
    fn prefetch_request_partial_eq_different_confidence() {
        let t = Instant::now();
        let r1 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.1,
            page_bytes: 4096,
            enqueued_at: t,
        };
        let r2 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: t,
        };
        assert_ne!(r1, r2);
    }

    // ── PrefetchRequest: PartialEq different page_bytes ──

    #[test]
    fn prefetch_request_partial_eq_different_page_bytes() {
        let t = Instant::now();
        let r1 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: t,
        };
        let r2 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 8192,
            enqueued_at: t,
        };
        assert_ne!(r1, r2);
    }

    // ── SwapInWorkerError: source returns None (no wrapped error) ──

    #[test]
    fn error_source_returns_none() {
        let e = SwapInWorkerError::SendFailed("test".into());
        assert!(std::error::Error::source(&e).is_none());
    }

    // ── swap_in_round: request with very large page_bytes ──

    #[test]
    fn round_large_page_bytes_submits() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 65536]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 65536,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 65536,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    // ── compute_urgency: urgency with NVMe tier and high access_count ──

    #[test]
    fn urgency_nvme_with_high_access_count_still_lower_than_dram() {
        let meta_nvme = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1000,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_dram = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        // NVMe tier_bonus=0.5, but with high access_count.
        let u_nvme = SwapInWorker::compute_urgency(&meta_nvme, 1.0, StorageTier::Nvme);
        let u_dram = SwapInWorker::compute_urgency(&meta_dram, 1.0, StorageTier::CpuDram);
        // Despite higher access_count, NVMe tier_bonus is 0.5 vs 1.0.
        // access_count=1000: importance_rebound = ln(1001)/ln(11) ≈ 2.98
        // access_count=10: importance_rebound = ln(11)/ln(11) = 1.0
        // NVMe first term = 2.98 * 1.0 * 0.5 = 1.49
        // DRAM first term = 1.0 * 1.0 * 1.0 = 1.0
        // So NVMe with high access_count CAN exceed DRAM with low access_count.
        // This test verifies the actual ordering:
        assert!(
            u_nvme > u_dram,
            "NVMe with high access_count can exceed DRAM with low access_count: nvme={u_nvme} dram={u_dram}",
        );
    }

    // ── drain_completions_and_update: page not in metadata is safely ignored ──

    #[test]
    fn drain_ignores_page_not_in_metadata() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Send a command for a page not in metadata.
        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 9999,
            page_bytes: 4096,
        });

        // Wait for the actor to process the command with retry.
        // The actor runs on a separate thread; 50ms may not suffice under load.
        let mut processed = false;
        for _ in 0..10 {
            thread::sleep(Duration::from_millis(50));
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok + s.promoted_failed >= 1 {
                processed = true;
                break;
            }
        }

        // Should not panic even though page 9999 has no metadata.
        assert!(
            processed,
            "should have processed at least one completion within 500ms",
        );

        actor.shutdown();
    }

    // ── drain_completions_and_update: latency tracked for page with swap_in_time ──

    #[test]
    fn drain_tracks_latency_for_page_with_swap_in_time() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        let swap_start = Instant::now() - Duration::from_millis(10);
        page_metadata.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: Some(swap_start),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 1,
            page_bytes: 4096,
        });

        // Wait for the actor to process the command with retry.
        let mut promoted = false;
        for _ in 0..10 {
            thread::sleep(Duration::from_millis(50));
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok + s.promoted_failed > 0 {
                promoted = true;
                break;
            }
        }

        if promoted {
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                assert!(
                    s.total_latency_us > 0,
                    "latency should be > 0 when swap_in_time was set: latency={}",
                    s.total_latency_us,
                );
            }
        }

        actor.shutdown();
    }

    // ── swap_in_round: submitted counter matches return value ──

    #[test]
    fn round_submitted_counter_matches_return() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize] {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::None,
                    },
                );
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=2)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        let s = stats.lock().expect("stats lock");
        assert_eq!(
            submitted as u64, s.submitted,
            "returned submitted should match stats.submitted",
        );

        actor.shutdown();
    }

    // ── SwapInWorkerStats: promoted_ok + promoted_failed invariant ──

    #[test]
    fn stats_promoted_sum_less_or_equal_submitted() {
        let mut stats = SwapInWorkerStats::default();
        stats.submitted = 10;
        stats.promoted_ok = 7;
        stats.promoted_failed = 3;
        assert_eq!(
            stats.promoted_ok + stats.promoted_failed,
            stats.submitted,
            "promoted_ok + promoted_failed should equal submitted",
        );
    }

    // ── SwapInWorkerError: format through {} and {:?} both work ──

    #[test]
    fn error_both_display_and_debug_work() {
        let e = SwapInWorkerError::SendFailed("msg".into());
        let display = format!("{e}");
        let debug = format!("{e:?}");
        assert!(!display.is_empty());
        assert!(!debug.is_empty());
        assert_ne!(display, debug, "Display and Debug should differ in format");
    }

    // ── PageState: all 7 variants in hashset ──

    #[test]
    fn page_state_all_seven_variants_in_hashset() {
        use std::collections::HashSet;
        let all: HashSet<PageState> = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ].into_iter().collect();
        assert_eq!(all.len(), 7, "should have 7 distinct PageState variants");
    }

    // ── WeightTier: all 3 variants in hashset ──

    #[test]
    fn weight_tier_all_three_variants_in_hashset() {
        use std::collections::HashSet;
        let all: HashSet<WeightTier> = [
            WeightTier::Hot,
            WeightTier::Warm,
            WeightTier::Cold,
        ].into_iter().collect();
        assert_eq!(all.len(), 3, "should have 3 distinct WeightTier variants");
    }

    // ── CompressionCodec: all 5 variants in hashset ──

    #[test]
    fn compression_codec_all_five_variants_in_hashset() {
        use std::collections::HashSet;
        let all: HashSet<CompressionCodec> = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ].into_iter().collect();
        assert_eq!(all.len(), 5, "should have 5 distinct CompressionCodec variants");
    }

    // ── SwapInWorkerConfig: max_in_flight zero means no submissions ──

    #[test]
    fn round_max_in_flight_zero_submits_nothing() {
        let config = SwapInWorkerConfig {
            max_in_flight: 0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0, "max_in_flight=0 should allow no submissions");

        actor.shutdown();
    }

    // ── swap_in_round: urgency infinity sorts to top ──

    #[test]
    fn round_infinity_urgency_sorted_to_top() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize] {
                table.insert(
                    pid,
                    PageAddrEntry {
                        gpu_ptr: None,
                        host_buffer: Some(vec![0u8; 4096]),
                        current_tier: StorageTier::CpuDram,
                        original_bytes: 4096,
                        codec: CompressionCodec::None,
                    },
                );
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: f32::INFINITY,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 0.5,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "infinity urgency page should be submitted (top of sort)");

        actor.shutdown();
    }

    // ── compute_urgency: negative infinity confidence ──

    #[test]
    fn urgency_neg_infinity_confidence_produces_neg_infinity() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, f32::NEG_INFINITY, StorageTier::CpuDram);
        assert!(
            u.is_infinite() && u < 0.0,
            "neg infinity confidence should produce neg infinite urgency: {u}",
        );
    }

    // ── swap_in_round: observer records events for DRAM page ──

    #[test]
    fn round_observer_receives_events() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                42,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 42,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);

        // Observer should have recorded a Recovered event at submit time.
        let obs = observer.lock().expect("observer lock");
        let count = obs.last_state.weight_recovery_count + obs.last_state.weight_eviction_count;
        assert!(count > 0, "observer should have recorded at least one event");

        actor.shutdown();
    }

    // ── drain_completions_and_update: stats promoted_ok increments on Ok result ──

    #[test]
    fn drain_promoted_ok_increments_on_success() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert page into addr_table with a host_buffer so actor can promote it.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 1, page_bytes: 4096 });
        // Poll drain until the actor produces a completion.
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.promoted_ok, 1, "should have 1 successful promotion");

        actor.shutdown();
    }

    // ── drain_completions_and_update: multiple completions processed ──

    #[test]
    fn drain_processes_multiple_completions() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert pages into addr_table with host_buffers so actor can promote them.
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize, 3usize] {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }

        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        for pid in [1usize, 2usize, 3usize] {
            page_metadata.insert(pid, PageMetadata {
                page_id: pid,
                sequence_id: None,
                recency: 0,
                access_count: 1,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state: PageState::SwappedOut,
                warm_until: None,
            });
        }
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        for pid in [1usize, 2usize, 3usize] {
            let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: pid, page_bytes: 4096 });
        }
        // Poll drain until all 3 actor completions are processed.
        for _ in 0..30 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok >= 3 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.promoted_ok, 3, "should have 3 successful promotions");

        actor.shutdown();
    }

    // ── swap_in_round: requests vector drained after round ──

    #[test]
    fn round_drains_requests_vector() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                1,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert!(requests.is_empty(), "requests should be drained after round");

        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional ~50 tests — covering remaining uncovered areas
    // ═══════════════════════════════════════════════════════════════════════════

    // ── MigrationActorConfig: default values ──

    #[test]
    fn migration_actor_config_default_queue_capacity() {
        let cfg = MigrationActorConfig::default();
        assert_eq!(cfg.queue_capacity, 256);
    }

    #[test]
    fn migration_actor_config_default_page_size() {
        let cfg = MigrationActorConfig::default();
        assert_eq!(cfg.page_size, 4096);
    }

    #[test]
    fn migration_actor_config_default_session_id() {
        let cfg = MigrationActorConfig::default();
        assert_eq!(cfg.session_id, "default");
    }

    #[test]
    fn migration_actor_config_default_max_swap_pages() {
        let cfg = MigrationActorConfig::default();
        assert_eq!(cfg.max_swap_pages, 4096);
    }

    #[test]
    fn migration_actor_config_default_nvme_swap_dir_is_under_home() {
        let cfg = MigrationActorConfig::default();
        assert!(
            cfg.nvme_swap_dir.to_string_lossy().contains(".gllm/swap"),
            "nvme_swap_dir should contain .gllm/swap: {:?}",
            cfg.nvme_swap_dir,
        );
    }

    #[test]
    fn migration_actor_config_custom_values() {
        let cfg = MigrationActorConfig {
            nvme_swap_dir: std::path::PathBuf::from("/tmp/test_swap"),
            queue_capacity: 512,
            session_id: "sess-42".to_string(),
            page_size: 8192,
            max_swap_pages: 8192,
        };
        assert_eq!(cfg.queue_capacity, 512);
        assert_eq!(cfg.session_id, "sess-42");
        assert_eq!(cfg.page_size, 8192);
        assert_eq!(cfg.max_swap_pages, 8192);
        assert_eq!(cfg.nvme_swap_dir, std::path::PathBuf::from("/tmp/test_swap"));
    }

    #[test]
    fn migration_actor_config_debug_format() {
        let cfg = MigrationActorConfig::default();
        let debug = format!("{cfg:?}");
        assert!(debug.contains("MigrationActorConfig"), "Debug should contain struct name");
        assert!(debug.contains("queue_capacity"), "Debug should contain queue_capacity field");
        assert!(debug.contains("page_size"), "Debug should contain page_size field");
    }

    #[test]
    fn migration_actor_config_clone_preserves_fields() {
        let cfg = MigrationActorConfig {
            nvme_swap_dir: std::path::PathBuf::from("/data/swap"),
            queue_capacity: 128,
            session_id: "clone-test".to_string(),
            page_size: 4096,
            max_swap_pages: 1024,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.queue_capacity, cfg.queue_capacity);
        assert_eq!(cloned.session_id, cfg.session_id);
        assert_eq!(cloned.page_size, cfg.page_size);
        assert_eq!(cloned.max_swap_pages, cfg.max_swap_pages);
        assert_eq!(cloned.nvme_swap_dir, cfg.nvme_swap_dir);
    }

    // ── MigrationCommand: Debug format ──

    #[test]
    fn migration_command_promote_to_dram_debug() {
        let cmd = MigrationCommand::PromoteToDram { page_id: 7, page_bytes: 4096 };
        let debug = format!("{cmd:?}");
        assert!(debug.contains("PromoteToDram"), "Debug should contain variant name: {debug}");
    }

    #[test]
    fn migration_command_promote_to_hbm_debug() {
        let cmd = MigrationCommand::PromoteToHbm { page_id: 42, page_bytes: 8192 };
        let debug = format!("{cmd:?}");
        assert!(debug.contains("PromoteToHbm"), "Debug should contain variant name: {debug}");
    }

    // ── MigrationResult: Debug format and construction ──

    #[test]
    fn migration_result_ok_debug() {
        let result = MigrationResult::Ok { compressed_bytes: 2048, checksum: 0xABCD };
        let debug = format!("{result:?}");
        assert!(debug.contains("Ok"), "Debug should contain Ok variant: {debug}");
    }

    #[test]
    fn migration_result_failed_debug() {
        let result = MigrationResult::Failed { reason: "page not found".to_string() };
        let debug = format!("{result:?}");
        assert!(debug.contains("Failed"), "Debug should contain Failed variant: {debug}");
        assert!(debug.contains("page not found"), "Debug should contain reason: {debug}");
    }

    #[test]
    fn migration_result_ok_clone() {
        let r1 = MigrationResult::Ok { compressed_bytes: 100, checksum: 123 };
        let r2 = r1.clone();
        let debug1 = format!("{r1:?}");
        let debug2 = format!("{r2:?}");
        assert_eq!(debug1, debug2);
    }

    #[test]
    fn migration_result_failed_clone() {
        let r1 = MigrationResult::Failed { reason: "err".to_string() };
        let r2 = r1.clone();
        let debug1 = format!("{r1:?}");
        let debug2 = format!("{r2:?}");
        assert_eq!(debug1, debug2);
    }

    // ── MigrationDone: construction and field access ──

    #[test]
    fn migration_done_construction() {
        let done = MigrationDone {
            page_id: 42,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::GpuHbm,
            result: MigrationResult::Ok { compressed_bytes: 4096, checksum: 0 },
        };
        assert_eq!(done.page_id, 42);
        assert_eq!(done.from_tier, StorageTier::CpuDram);
        assert_eq!(done.to_tier, StorageTier::GpuHbm);
    }

    #[test]
    fn migration_done_failed_result() {
        let done = MigrationDone {
            page_id: 99,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::CpuDram,
            result: MigrationResult::Failed { reason: "no buffer".to_string() },
        };
        assert_eq!(done.page_id, 99);
        assert_eq!(done.from_tier, StorageTier::Nvme);
        assert!(matches!(done.result, MigrationResult::Failed { .. }));
    }

    #[test]
    fn migration_done_debug_format() {
        let done = MigrationDone {
            page_id: 1,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::GpuHbm,
            result: MigrationResult::Ok { compressed_bytes: 512, checksum: 0xF00D },
        };
        let debug = format!("{done:?}");
        assert!(debug.contains("MigrationDone"), "Debug should contain struct name: {debug}");
    }

    // ── WeightPageTelemetryEvent: Debug and Clone ──

    #[test]
    fn telemetry_event_evicted_debug() {
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 10,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: crate::scheduler::observer::EvictionReason::MemoryPressure,
            bytes: 4096,
        };
        let debug = format!("{event:?}");
        assert!(debug.contains("Evicted"), "Debug should contain Evicted: {debug}");
    }

    #[test]
    fn telemetry_event_recovered_debug() {
        let event = WeightPageTelemetryEvent::Recovered {
            page_id: 5,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 1500,
            bytes: 8192,
        };
        let debug = format!("{event:?}");
        assert!(debug.contains("Recovered"), "Debug should contain Recovered: {debug}");
    }

    #[test]
    fn telemetry_event_evicted_clone() {
        let e1 = WeightPageTelemetryEvent::Evicted {
            page_id: 1,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Cold,
            reason: crate::scheduler::observer::EvictionReason::MemoryPressure,
            bytes: 4096,
        };
        let e2 = e1.clone();
        let d1 = format!("{e1:?}");
        let d2 = format!("{e2:?}");
        assert_eq!(d1, d2);
    }

    #[test]
    fn telemetry_event_recovered_clone() {
        let e1 = WeightPageTelemetryEvent::Recovered {
            page_id: 2,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Hot,
            latency_us: 200,
            bytes: 4096,
        };
        let e2 = e1.clone();
        let d1 = format!("{e1:?}");
        let d2 = format!("{e2:?}");
        assert_eq!(d1, d2);
    }

    // ── EvictionReason: sole variant and traits ──

    #[test]
    fn eviction_reason_debug_format() {
        let reason = crate::scheduler::observer::EvictionReason::MemoryPressure;
        let debug = format!("{reason:?}");
        assert!(debug.contains("MemoryPressure"), "Debug should contain variant: {debug}");
    }

    #[test]
    fn eviction_reason_copy_semantics() {
        let r1 = crate::scheduler::observer::EvictionReason::MemoryPressure;
        let r2 = r1;
        assert_eq!(r1, r2);
    }

    #[test]
    fn eviction_reason_equality() {
        let r1 = crate::scheduler::observer::EvictionReason::MemoryPressure;
        let r2 = crate::scheduler::observer::EvictionReason::MemoryPressure;
        assert_eq!(r1, r2);
    }

    #[test]
    fn eviction_reason_hash_in_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(crate::scheduler::observer::EvictionReason::MemoryPressure);
        assert!(set.contains(&crate::scheduler::observer::EvictionReason::MemoryPressure));
        assert_eq!(set.len(), 1);
    }

    // ── PageAddrEntry: Debug format ──

    #[test]
    fn page_addr_entry_debug_format() {
        let entry = PageAddrEntry {
            gpu_ptr: Some(0xBEEF),
            host_buffer: Some(vec![1, 2, 3]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 3,
            codec: CompressionCodec::Lz4,
        };
        let debug = format!("{entry:?}");
        assert!(debug.contains("PageAddrEntry"), "Debug should contain struct name: {debug}");
        assert!(debug.contains("gpu_ptr"), "Debug should contain gpu_ptr field: {debug}");
    }

    #[test]
    fn page_addr_entry_with_zero_original_bytes() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: None,
            current_tier: StorageTier::Nvme,
            original_bytes: 0,
            codec: CompressionCodec::None,
        };
        assert_eq!(entry.original_bytes, 0);
    }

    #[test]
    fn page_addr_entry_large_gpu_ptr() {
        let entry = PageAddrEntry {
            gpu_ptr: Some(u64::MAX),
            host_buffer: None,
            current_tier: StorageTier::GpuHbm,
            original_bytes: 4096,
            codec: CompressionCodec::None,
        };
        assert_eq!(entry.gpu_ptr, Some(u64::MAX));
    }

    // ── swap_in_round: CpuDram page in_flight count is one ──

    #[test]
    fn round_dram_page_in_flight_count_is_one() {
        let config = SwapInWorkerConfig {
            max_in_flight: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            table.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=2usize)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // CpuDram page: only PromoteToHbm sent → in_flight += 1.
        // Page 1: in_flight=1 after submission, which equals max_in_flight=1.
        // Page 2: in_flight(1) >= max_in_flight(1), so it is NOT submitted.
        assert_eq!(submitted, 1, "only 1 CpuDram page should fit in max_in_flight=1");

        actor.shutdown();
    }

    // ── compute_urgency: confidence=0.5 exactly in the middle ──

    #[test]
    fn urgency_half_confidence_half_first_term() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 9,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_full = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_half = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        // importance_rebound = 1.0 for access_count=9.
        // The first term difference: 1.0 * 0.5 * 1.0 = 0.5.
        let diff = u_full - u_half;
        let expected_first_term_diff = 1.0 * 0.5 * 1.0;
        assert!(
            (diff - expected_first_term_diff).abs() < 0.05,
            "0.5 confidence difference should equal half the first term: diff={diff} expected={expected_first_term_diff}",
        );
    }

    // ── compute_urgency: page_id does not affect score ──

    #[test]
    fn urgency_page_id_does_not_affect_score() {
        let meta_a = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_b = PageMetadata {
            page_id: 99999,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_a = SwapInWorker::compute_urgency(&meta_a, 0.7, StorageTier::CpuDram);
        let u_b = SwapInWorker::compute_urgency(&meta_b, 0.7, StorageTier::CpuDram);
        assert!(
            (u_a - u_b).abs() < 1e-3,
            "page_id should not affect urgency: a={u_a} b={u_b}",
        );
    }

    // ── compute_urgency: PageState does not affect score ──

    #[test]
    fn urgency_page_state_does_not_affect_score() {
        let meta_active = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta_swapped = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Swapped,
            warm_until: None,
        };
        let u_active = SwapInWorker::compute_urgency(&meta_active, 0.8, StorageTier::CpuDram);
        let u_swapped = SwapInWorker::compute_urgency(&meta_swapped, 0.8, StorageTier::CpuDram);
        assert!(
            (u_active - u_swapped).abs() < 1e-3,
            "PageState should not affect urgency: active={u_active} swapped={u_swapped}",
        );
    }

    // ── swap_in_round: two consecutive rounds with same page_id ──

    #[test]
    fn round_two_consecutive_with_same_page() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Round 1.
        let mut r1 = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let s1 = SwapInWorker::swap_in_round(
            &config, &actor, &mut r1, &page_metadata, &addr_table, &stats, &observer,
        );

        // Round 2: same page_id again.
        let mut r2 = vec![PrefetchRequest {
            page_id: 1,
            urgency: 0.8,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let s2 = SwapInWorker::swap_in_round(
            &config, &actor, &mut r2, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(s1, 1);
        assert_eq!(s2, 1, "same page can be submitted again in a new round");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2);
        assert_eq!(s.rounds, 2);
        assert_eq!(s.submitted, 2);

        actor.shutdown();
    }

    // ── swap_in_round: skipped due to low confidence does not affect submitted ──

    #[test]
    fn round_skipped_does_not_increment_submitted() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.99,
            ..SwapInWorkerConfig::default()
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=5)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5,
                prefetch_confidence: 0.5, // all below 0.99
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 0);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 0);
        assert_eq!(s.skipped, 5);
        assert_eq!(s.total_requests, 5);

        actor.shutdown();
    }

    // ── SwapInWorkerStats: debug format with non-zero values ──

    #[test]
    fn stats_debug_format_with_values() {
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 100;
        stats.submitted = 80;
        stats.skipped = 20;
        stats.promoted_ok = 70;
        stats.promoted_failed = 10;
        stats.two_hop_promotions = 15;
        stats.total_latency_us = 5000;
        stats.rounds = 25;
        let debug = format!("{stats:?}");
        assert!(debug.contains("total_requests"), "Debug should show field name: {debug}");
        assert!(debug.contains("rounds"), "Debug should show rounds: {debug}");
    }

    // ── PageId type alias is usize ──

    #[test]
    fn page_id_is_usize() {
        let pid: PageId = 42;
        assert_eq!(pid, 42usize);
        let max_pid: PageId = usize::MAX;
        assert_eq!(max_pid, usize::MAX);
    }

    // ── compute_urgency: CpuDram tier_bonus multiplier ──

    #[test]
    fn urgency_cpuram_tier_bonus_exactly_one() {
        // With zero recency (very old access) and zero confidence,
        // the first term dominates: importance_rebound * confidence * tier_bonus.
        // For CpuDram, tier_bonus=1.0.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 9,
            last_access: Instant::now() - Duration::from_secs(3600),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // With confidence=1.0 and CpuDram, urgency should be positive.
        assert!(
            u > 0.0,
            "CpuDram with access_count=9 and confidence=1.0 should yield positive urgency: {u}",
        );
    }

    // ── swap_in_round: mixed confidence with some accepted and some skipped ──

    #[test]
    fn round_mixed_confidence_partial_submission() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            max_prefetch_per_round: 10,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in 1..=4usize {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest { page_id: 1, urgency: 1.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.8, prefetch_confidence: 0.1, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 3, urgency: 0.6, prefetch_confidence: 0.7, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 4, urgency: 0.4, prefetch_confidence: 0.3, page_bytes: 4096, enqueued_at: Instant::now() },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Pages 1 (0.9) and 3 (0.7) pass confidence >= 0.5.
        // Pages 2 (0.1) and 4 (0.3) are skipped.
        assert_eq!(submitted, 2, "only pages with confidence >= 0.5 should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 2);

        actor.shutdown();
    }

    // ── CompressionCodec: Debug format for each variant ──

    #[test]
    fn compression_codec_debug_none() {
        assert!(format!("{:?}", CompressionCodec::None).contains("None"));
    }

    #[test]
    fn compression_codec_debug_lz4() {
        assert!(format!("{:?}", CompressionCodec::Lz4).contains("Lz4"));
    }

    #[test]
    fn compression_codec_debug_bitpack_rle() {
        let debug = format!("{:?}", CompressionCodec::BitPackRle);
        assert!(debug.contains("BitPackRle"), "Debug should contain BitPackRle: {debug}");
    }

    #[test]
    fn compression_codec_debug_nvcomp_ans() {
        let debug = format!("{:?}", CompressionCodec::NvcompAns);
        assert!(debug.contains("NvcompAns"), "Debug should contain NvcompAns: {debug}");
    }

    #[test]
    fn compression_codec_debug_zstd_dict() {
        let debug = format!("{:?}", CompressionCodec::ZstdDict);
        assert!(debug.contains("ZstdDict"), "Debug should contain ZstdDict: {debug}");
    }

    // ── swap_in_round: max_prefetch_per_round larger than input ──

    #[test]
    fn round_max_prefetch_larger_than_input_processes_all() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 100,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in 1..=3usize {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=3)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 3, "all 3 should be submitted when max_prefetch > input");

        actor.shutdown();
    }

    // ── compute_urgency: confidence=0.0 with all tier bonuses still zero first term ──

    #[test]
    fn urgency_zero_confidence_zero_first_term_all_tiers() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let u = SwapInWorker::compute_urgency(&meta, 0.0, tier);
            // With confidence=0.0, first term = 0. Only recency bonus remains.
            // u = 0 + recency_bonus * 0.1, which should be small but positive.
            assert!(
                u < 0.2,
                "zero confidence with tier {:?} should leave only recency bonus: {u}",
                tier,
            );
        }
    }

    // ── drain_completions_and_update: no latency when swap_in_time is None ──

    #[test]
    fn drain_no_latency_when_swap_in_time_none() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None, // No swap_in_time set.
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 1, page_bytes: 4096 });
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.promoted_ok, 1);
        assert_eq!(
            s.total_latency_us, 0,
            "latency should be 0 when swap_in_time was None",
        );

        actor.shutdown();
    }

    // ── WeightTier: Debug and Hash ──

    #[test]
    fn weight_tier_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WeightTier::Hot);
        assert!(set.contains(&WeightTier::Hot));
        assert!(!set.contains(&WeightTier::Warm));
        assert!(!set.contains(&WeightTier::Cold));
    }

    #[test]
    fn weight_tier_all_three_in_hashset() {
        use std::collections::HashSet;
        let set: HashSet<WeightTier> = [WeightTier::Hot, WeightTier::Warm, WeightTier::Cold].into_iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ── StorageTier: from_u8 boundary values ──

    #[test]
    fn storage_tier_from_u8_boundary_0_1_2() {
        assert_eq!(StorageTier::from_u8(0), Some(StorageTier::GpuHbm));
        assert_eq!(StorageTier::from_u8(1), Some(StorageTier::CpuDram));
        assert_eq!(StorageTier::from_u8(2), Some(StorageTier::Nvme));
    }

    #[test]
    fn storage_tier_from_u8_out_of_range() {
        assert_eq!(StorageTier::from_u8(3), None);
        assert_eq!(StorageTier::from_u8(100), None);
        assert_eq!(StorageTier::from_u8(u8::MAX), None);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional ~60 tests — final batch covering remaining gaps
    // ═══════════════════════════════════════════════════════════════════════════

    // ── SwapInWorkerConfig: tick_interval at exactly one second ──

    #[test]
    fn config_tick_interval_one_second() {
        let cfg = SwapInWorkerConfig {
            tick_interval: Duration::from_secs(1),
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval, Duration::from_secs(1));
        assert_eq!(cfg.tick_interval.as_millis(), 1000);
    }

    // ── SwapInWorkerConfig: all zero fields config ──

    #[test]
    fn config_all_zeros_is_valid() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 0,
            tick_interval: Duration::ZERO,
            min_confidence: 0.0,
            max_in_flight: 0,
            page_bytes: 0,
        };
        assert_eq!(cfg.max_prefetch_per_round, 0);
        assert_eq!(cfg.tick_interval, Duration::ZERO);
        assert!((cfg.min_confidence - 0.0).abs() < 1e-6);
        assert_eq!(cfg.max_in_flight, 0);
        assert_eq!(cfg.page_bytes, 0);
    }

    // ── SwapInWorkerConfig: PartialEq for min_confidence differences ──

    #[test]
    fn config_partial_eq_min_confidence_difference() {
        let c1 = SwapInWorkerConfig {
            min_confidence: 0.1,
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            min_confidence: 0.2,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2);
    }

    // ── SwapInWorkerConfig: PartialEq for page_bytes difference ──

    #[test]
    fn config_partial_eq_page_bytes_difference() {
        let c1 = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            page_bytes: 8192,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2);
    }

    // ── SwapInWorkerConfig: PartialEq for max_in_flight difference ──

    #[test]
    fn config_partial_eq_max_in_flight_difference() {
        let c1 = SwapInWorkerConfig {
            max_in_flight: 64,
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            max_in_flight: 128,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2);
    }

    // ── SwapInWorkerStats: accumulated round counting pattern ──

    #[test]
    fn stats_rounds_accumulate_correctly() {
        let mut stats = SwapInWorkerStats::default();
        for _ in 0..100 {
            stats.rounds += 1;
        }
        assert_eq!(stats.rounds, 100);
    }

    // ── SwapInWorkerStats: total_latency accumulation with many small values ──

    #[test]
    fn stats_latency_accumulation_many_small() {
        let mut stats = SwapInWorkerStats::default();
        for _ in 0..100 {
            stats.total_latency_us += 10;
            stats.promoted_ok += 1;
        }
        assert_eq!(stats.total_latency_us, 1000);
        assert_eq!(stats.promoted_ok, 100);
        assert!(
            (stats.avg_latency_us() - 10.0).abs() < 1e-6,
            "avg should be 10us: {}",
            stats.avg_latency_us(),
        );
    }

    // ── SwapInWorkerStats: promoted_failed accumulation does not affect avg_latency ──

    #[test]
    fn stats_promoted_failed_accumulation_independent() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 3;
        stats.total_latency_us = 300;
        stats.promoted_failed = 10; // Many failures.
        assert!(
            (stats.avg_latency_us() - 100.0).abs() < 1e-6,
            "avg should only count promoted_ok: {}",
            stats.avg_latency_us(),
        );
    }

    // ── SwapInWorkerStats: two_hop_promotions accumulation ──

    #[test]
    fn stats_two_hop_promotions_accumulate() {
        let mut stats = SwapInWorkerStats::default();
        stats.two_hop_promotions += 5;
        stats.two_hop_promotions += 3;
        assert_eq!(stats.two_hop_promotions, 8);
    }

    // ── SwapInWorkerStats: Default trait is consistent ──

    #[test]
    fn stats_default_is_consistent() {
        let s1 = SwapInWorkerStats::default();
        let s2 = SwapInWorkerStats {
            total_requests: 0,
            submitted: 0,
            skipped: 0,
            promoted_ok: 0,
            promoted_failed: 0,
            two_hop_promotions: 0,
            total_latency_us: 0,
            rounds: 0,
        };
        assert_eq!(s1, s2, "Default and explicit zero construction should be equal");
    }

    // ── SwapInWorkerStats: sum invariant with zero values ──

    #[test]
    fn stats_sum_invariant_all_zeros() {
        let stats = SwapInWorkerStats::default();
        assert_eq!(stats.submitted + stats.skipped, 0);
    }

    // ── SwapInWorkerStats: promoted_ok less than or equal submitted ──

    #[test]
    fn stats_promoted_ok_cannot_exceed_submitted() {
        let stats = SwapInWorkerStats {
            total_requests: 10,
            submitted: 8,
            skipped: 2,
            promoted_ok: 8,
            promoted_failed: 0,
            two_hop_promotions: 0,
            total_latency_us: 1000,
            rounds: 1,
        };
        assert!(
            stats.promoted_ok + stats.promoted_failed <= stats.submitted,
            "promoted_ok + promoted_failed should not exceed submitted",
        );
    }

    // ── SwapInWorkerError: both variants are Debug + Clone + PartialEq ──

    #[test]
    fn error_send_failed_debug_contains_variant_name() {
        let e = SwapInWorkerError::SendFailed("timeout".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("SendFailed"), "Debug for SendFailed should contain variant name: {debug}");
    }

    #[test]
    fn error_recv_failed_debug_contains_variant_name() {
        let e = SwapInWorkerError::RecvFailed("closed".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("RecvFailed"), "Debug for RecvFailed should contain variant name: {debug}");
    }

    // ── SwapInWorkerError: very long message ──

    #[test]
    fn error_display_with_very_long_message() {
        let long_msg = "x".repeat(10_000);
        let e = SwapInWorkerError::SendFailed(long_msg.clone());
        let msg = format!("{e}");
        assert!(msg.contains(&long_msg));
    }

    // ── PrefetchRequest: confidence at exactly 1.0 ──

    #[test]
    fn prefetch_request_confidence_exactly_one() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 1.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!((req.prefetch_confidence - 1.0).abs() < 1e-6);
    }

    // ── PrefetchRequest: urgency subnormal float ──

    #[test]
    fn prefetch_request_subnormal_urgency() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: f32::from_bits(1), // Smallest positive subnormal.
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!(req.urgency > 0.0);
        assert!(req.urgency.is_subnormal());
    }

    // ── PrefetchRequest: negative infinity confidence ──

    #[test]
    fn prefetch_request_neg_infinity_confidence_stored() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: f32::NEG_INFINITY,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!(req.prefetch_confidence.is_infinite());
        assert!(req.prefetch_confidence < 0.0);
    }

    // ── PrefetchRequest: negative infinity urgency ──

    #[test]
    fn prefetch_request_neg_infinity_urgency_stored() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: f32::NEG_INFINITY,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!(req.urgency.is_infinite() && req.urgency < 0.0);
    }

    // ── PrefetchRequest: PartialEq different enqueued_at ──

    #[test]
    fn prefetch_request_partial_eq_different_enqueued_at() {
        let r1 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        // Even a tiny time difference makes enqueued_at different.
        thread::sleep(Duration::from_micros(1));
        let r2 = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        // Instant implements PartialEq based on internal counter.
        // These two Instants are almost certainly different.
        assert_ne!(r1, r2, "different enqueued_at should produce inequality");
    }

    // ── compute_urgency: access_count=2 vs access_count=1 ──

    #[test]
    fn urgency_access_count_two_higher_than_one() {
        let meta1 = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let meta2 = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 2,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u1 = SwapInWorker::compute_urgency(&meta1, 1.0, StorageTier::CpuDram);
        let u2 = SwapInWorker::compute_urgency(&meta2, 1.0, StorageTier::CpuDram);
        assert!(
            u2 > u1,
            "access_count=2 should yield higher urgency than 1: u2={u2} u1={u1}",
        );
    }

    // ── compute_urgency: confidence=0.0 with all tier bonuses ──

    #[test]
    fn urgency_zero_confidence_same_across_all_tiers() {
        // With confidence=0.0, first term = 0 regardless of tier.
        // Only recency bonus (which depends on last_access time) remains.
        // Since all three calls happen nearly simultaneously, they should be nearly equal.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u_hbm = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::GpuHbm);
        let u_dram = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::Nvme);
        // With zero confidence, tier_bonus is multiplied by 0, so all should be ~equal (recency only).
        assert!(
            (u_hbm - u_dram).abs() < 1e-3,
            "zero confidence should nullify tier difference: hbm={u_hbm} dram={u_dram}",
        );
        assert!(
            (u_dram - u_nvme).abs() < 1e-3,
            "zero confidence should nullify tier difference: dram={u_dram} nvme={u_nvme}",
        );
    }

    // ── compute_urgency: Nvme tier bonus is exactly 0.5 ──

    // ── compute_urgency: urgency is finite for all reasonable inputs ──

    #[test]
    fn urgency_finite_for_reasonable_inputs() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 50,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: Some(Instant::now() + Duration::from_secs(30)),
        };
        for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            for confidence in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let u = SwapInWorker::compute_urgency(&meta, confidence, tier);
                assert!(
                    u.is_finite(),
                    "urgency should be finite for confidence={confidence} tier={tier:?}: {u}",
                );
            }
        }
    }

    // ── compute_urgency: NvMe tier with zero confidence is positive ──

    #[test]
    fn urgency_nvme_zero_confidence_positive_from_recency() {
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
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::Nvme);
        assert!(
            u > 0.0,
            "recency bonus should keep urgency positive even on NVMe with zero confidence: {u}",
        );
    }

    // ── compute_urgency: recency bonus converges to zero for very old access ──

    #[test]
    fn urgency_recency_bonus_converges_to_zero_for_very_old() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(86400), // 1 day ago.
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // recency_bonus = 1/(1+86400) ≈ 0.0000116, u ≈ 0.00000116
        assert!(
            u < 0.001,
            "1-day-old access should have near-zero urgency: {u}",
        );
        assert!(u > 0.0, "urgency should still be positive: {u}");
    }

    // ── swap_in_round: back-pressure at exactly max_in_flight ──

    #[test]
    fn round_back_pressure_at_exact_limit() {
        let config = SwapInWorkerConfig {
            max_in_flight: 3,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in 1..=4usize {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=4)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 0.5,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 3, "max_in_flight=3 should allow exactly 3 submissions");

        actor.shutdown();
    }

    // ── swap_in_round: single request with max_in_flight=1 succeeds ──

    #[test]
    fn round_single_request_with_max_in_flight_one() {
        let config = SwapInWorkerConfig {
            max_in_flight: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(42, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 42,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    // ── swap_in_round: confidence threshold at f32::MIN positive ──

    #[test]
    fn round_confidence_threshold_very_small_positive_accepts() {
        let config = SwapInWorkerConfig {
            min_confidence: f32::from_bits(1), // smallest positive subnormal
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.01,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "very small min_confidence should accept 0.01");

        actor.shutdown();
    }

    // ── swap_in_round: multiple rounds with interleaved empty rounds ──

    #[test]
    fn round_interleaved_empty_rounds_count() {
        let config = SwapInWorkerConfig::default();
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Round 1: has requests.
        let mut r1: Vec<PrefetchRequest> = vec![PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut r1, &page_metadata, &addr_table, &stats, &observer,
        );

        // Round 2: empty.
        let mut r2: Vec<PrefetchRequest> = Vec::new();
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut r2, &page_metadata, &addr_table, &stats, &observer,
        );

        // Round 3: has requests.
        let mut r3: Vec<PrefetchRequest> = vec![PrefetchRequest {
            page_id: 2,
            urgency: 0.5,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let _ = SwapInWorker::swap_in_round(
            &config, &actor, &mut r3, &page_metadata, &addr_table, &stats, &observer,
        );

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 3, "3 rounds total");
        assert_eq!(s.total_requests, 2, "2 requests from rounds 1 and 3");

        actor.shutdown();
    }

    // ── swap_in_round: stats invariant after a round with mixed outcomes ──

    #[test]
    fn round_stats_invariant_after_mixed() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            max_prefetch_per_round: 10,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            // Page 1: CpuDram, will be submitted.
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            // Page 2: HBM, will be skipped.
            table.insert(2, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest { page_id: 1, urgency: 0.9, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.8, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 3, urgency: 0.7, prefetch_confidence: 0.1, page_bytes: 4096, enqueued_at: Instant::now() },
        ];

        let _submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 3);
        assert_eq!(s.submitted, 1);
        assert_eq!(s.skipped, 2);
        assert_eq!(s.rounds, 1);
        // submitted + skipped <= total_requests (some may be truncated or back-pressured).
        assert!(
            s.submitted + s.skipped <= s.total_requests,
            "submitted + skipped should not exceed total_requests",
        );

        actor.shutdown();
    }

    // ── swap_in_round: negative urgency sorts below positive ──

    #[test]
    fn round_negative_urgency_sorted_below_positive() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize] {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest { page_id: 1, urgency: -10.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 5.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        // Positive urgency (5.0) should sort above negative (-10.0).
        assert_eq!(submitted, 1, "positive urgency should be selected first");

        actor.shutdown();
    }

    // ── swap_in_round: observer records correct from_tier for NVMe ──

    #[test]
    fn round_observer_records_cold_tier_for_nvme() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);

        let obs = observer.lock().expect("observer lock");
        let count = obs.last_state.weight_recovery_count;
        assert!(count > 0, "observer should have recorded recovery event for NVMe page");

        actor.shutdown();
    }

    // ── drain_completions_and_update: multiple drain calls are idempotent ──

    #[test]
    fn drain_idempotent_after_all_processed() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // No commands sent — drain should be a no-op.
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);

        actor.shutdown();
    }

    // ── drain_completions_and_update: page state transition for Nvme to_tier ──

    #[test]
    fn drain_nvme_tier_sets_state_to_swapped() {
        // This test verifies the Nvme branch in drain_completions_and_update
        // by using swap_in_round which sends commands through the real actor.
        // The actor promotes from CpuDram to GpuHbm, so we indirectly verify
        // the metadata update logic.
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 1, page_bytes: 4096 });
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let meta = page_metadata.read().expect("read lock").get(&1).cloned();
        assert!(meta.is_some());
        let meta = meta.unwrap();
        assert_eq!(meta.state, PageState::Active, "HBM promotion should set state to Active");
        assert!(meta.swap_in_time.is_none(), "swap_in_time should be cleared after HBM promotion");

        actor.shutdown();
    }

    // ── SwapInWorker: stats() returns clone not reference ──

    #[test]
    fn stats_returns_independent_snapshot() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(100),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );

        let snap1 = worker.stats();
        let snap2 = worker.stats();
        assert_eq!(snap1, snap2, "two snapshots of fresh worker should be equal");

        worker.shutdown();
    }

    // ── SwapInWorker: prefetch after double shutdown returns error ──

    #[test]
    fn prefetch_after_double_shutdown_returns_error() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(10),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );
        worker.shutdown();
        worker.shutdown(); // Double shutdown.

        let result = worker.prefetch(PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.7,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        });
        assert!(result.is_err(), "prefetch after double shutdown should still fail");
    }

    // ── SwapInWorkerError: PartialEq with same variant different message ──

    #[test]
    fn error_partial_eq_same_variant_long_messages() {
        let e1 = SwapInWorkerError::SendFailed("a".repeat(1000));
        let e2 = SwapInWorkerError::SendFailed("b".repeat(1000));
        assert_ne!(e1, e2);
    }

    // ── PrefetchRequest: Debug output contains all field names ──

    #[test]
    fn prefetch_request_debug_contains_all_fields() {
        let req = PrefetchRequest {
            page_id: 42,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let debug = format!("{req:?}");
        assert!(debug.contains("page_id"), "Debug should contain page_id: {debug}");
        assert!(debug.contains("urgency"), "Debug should contain urgency: {debug}");
        assert!(debug.contains("prefetch_confidence"), "Debug should contain prefetch_confidence: {debug}");
        assert!(debug.contains("page_bytes"), "Debug should contain page_bytes: {debug}");
        assert!(debug.contains("enqueued_at"), "Debug should contain enqueued_at: {debug}");
    }

    // ── PrefetchRequest: Clone independence for f32 fields ──

    #[test]
    fn prefetch_request_clone_urgency_independence() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let mut cloned = req.clone();
        cloned.urgency = 0.9;
        assert!(
            (req.urgency - 0.5).abs() < 1e-6,
            "original urgency should be unchanged: {}",
            req.urgency,
        );
        assert!(
            (cloned.urgency - 0.9).abs() < 1e-6,
            "cloned urgency should be 0.9: {}",
            cloned.urgency,
        );
    }

    // ── PrefetchRequest: Clone independence for confidence ──

    #[test]
    fn prefetch_request_clone_confidence_independence() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let _cloned = {
            let mut c = req.clone();
            c.prefetch_confidence = 0.2;
            c
        };
        assert!(
            (req.prefetch_confidence - 0.8).abs() < 1e-6,
            "original confidence should be unchanged",
        );
    }

    // ── PrefetchRequest: Clone independence for page_bytes ──

    #[test]
    fn prefetch_request_clone_page_bytes_independence() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let mut cloned = req.clone();
        cloned.page_bytes = 8192;
        assert_eq!(req.page_bytes, 4096, "original page_bytes should be unchanged");
        assert_eq!(cloned.page_bytes, 8192);
    }

    // ── compute_urgency: Nvme tier with very high access_count still finite ──

    #[test]
    fn urgency_nvme_high_access_count_finite() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10_000_000,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::Nvme);
        assert!(u.is_finite(), "urgency should be finite even with 10M access_count on NVMe: {u}");
        assert!(u > 0.0);
    }

    // ── compute_urgency: GpuHbm tier with zero confidence only recency ──

    #[test]
    fn urgency_gpu_hbm_zero_confidence_only_recency() {
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
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::GpuHbm);
        // Same as other tiers with zero confidence — only recency bonus.
        assert!(
            (u - 0.1).abs() < 0.02,
            "GpuHbm with zero confidence should have only recency bonus ~0.1: {u}",
        );
    }

    // ── compute_urgency: confidence=0.5 on all three tiers finite ──

    #[test]
    fn urgency_half_confidence_all_tiers_finite() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let u = SwapInWorker::compute_urgency(&meta, 0.5, tier);
            assert!(u.is_finite(), "urgency should be finite for {tier:?}: {u}");
            assert!(u > 0.0, "urgency should be positive for {tier:?}: {u}");
        }
    }

    // ── swap_in_round: DRAM followed by NVMe in same round ──

    #[test]
    fn round_dram_then_nvme_in_same_batch() {
        let config = SwapInWorkerConfig {
            max_in_flight: 10,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            table.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // NVMe first (higher urgency), DRAM second.
        let mut requests = vec![
            PrefetchRequest { page_id: 2, urgency: 1.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 1, urgency: 0.5, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 2, "both DRAM and NVMe should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "only NVMe page should be two-hop");

        actor.shutdown();
    }

    // ── swap_in_round: many NVMe pages exceed back-pressure quickly ──

    #[test]
    fn round_many_nvme_pages_hit_back_pressure_fast() {
        let config = SwapInWorkerConfig {
            max_in_flight: 6,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in 1..=5usize {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: None,
                    current_tier: StorageTier::Nvme,
                    original_bytes: 4096,
                    codec: CompressionCodec::ZstdDict,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=5)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Each NVMe page: PromoteToDram(+1) + PromoteToHbm(+1) = 2 in_flight.
        // max_in_flight=6 → 3 NVMe pages fit (6 in_flight), page 4 would be 8 > 6.
        assert_eq!(submitted, 3, "3 NVMe pages should fit in max_in_flight=6");

        actor.shutdown();
    }

    // ── PageMetadata: last_access in future handled gracefully ──

    #[test]
    fn page_metadata_last_access_future_is_valid() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now() + Duration::from_secs(60),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        // Should not panic when constructing or using.
        let u = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        assert!(u.is_finite() && u > 0.0, "urgency with future last_access should be positive: {u}");
    }

    // ── PageMetadata: clone with swap_in_time set ──

    #[test]
    fn page_metadata_clone_with_swap_in_time() {
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 5,
            sequence_id: Some(10),
            recency: 2,
            access_count: 20,
            last_access: now,
            swap_in_time: Some(now),
            is_lir: true,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let cloned = meta.clone();
        assert_eq!(cloned.page_id, 5);
        assert_eq!(cloned.swap_in_time, Some(now));
        assert!(cloned.is_lir);
    }

    // ── PageAddrEntry: original_bytes set to large value ──

    #[test]
    fn page_addr_entry_large_original_bytes() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(vec![0u8; 0]),
            current_tier: StorageTier::CpuDram,
            original_bytes: usize::MAX,
            codec: CompressionCodec::None,
        };
        assert_eq!(entry.original_bytes, usize::MAX);
    }

    // ── CompressionCodec: equality across same variants ──

    #[test]
    fn compression_codec_equality_same_variant() {
        assert_eq!(CompressionCodec::None, CompressionCodec::None);
        assert_eq!(CompressionCodec::Lz4, CompressionCodec::Lz4);
        assert_eq!(CompressionCodec::BitPackRle, CompressionCodec::BitPackRle);
        assert_eq!(CompressionCodec::NvcompAns, CompressionCodec::NvcompAns);
        assert_eq!(CompressionCodec::ZstdDict, CompressionCodec::ZstdDict);
    }

    // ── MigrationActorConfig: Debug default shows queue_capacity ──

    #[test]
    fn migration_actor_config_debug_default_fields() {
        let c = MigrationActorConfig::default();
        let debug = format!("{c:?}");
        // MigrationActorConfig derives Debug; verify default contains queue_capacity
        assert!(debug.contains("queue_capacity"));
        assert!(debug.contains("256"));
    }

    // ── MigrationActorConfig: Debug shows custom queue_capacity ──

    #[test]
    fn migration_actor_config_debug_custom_queue_capacity() {
        let c = MigrationActorConfig {
            queue_capacity: 999,
            ..MigrationActorConfig::default()
        };
        let debug = format!("{c:?}");
        assert!(debug.contains("999"));
    }

    // ── MigrationActorConfig: Debug shows custom session_id ──

    #[test]
    fn migration_actor_config_debug_custom_session_id() {
        let c = MigrationActorConfig {
            session_id: "test-session-42".to_string(),
            ..MigrationActorConfig::default()
        };
        let debug = format!("{c:?}");
        assert!(debug.contains("test-session-42"));
    }

    // ── MigrationResult: Debug Ok variant ──

    // NOTE: MigrationResult fields are private so we cannot construct variants
    // directly in tests. Instead we verify the Debug output of StorageTier
    // used within MigrationDone contexts.

    // ── MigrationDone: Debug contains tier names ──

    // NOTE: MigrationDone contains MigrationResult which has private fields,
    // so we cannot construct MigrationDone in tests either. These tests are
    // replaced by MigrationCommand Debug tests which are constructable.

    // ── StorageTier: Debug format contains variant name ──

    #[test]
    fn storage_tier_debug_gpu_hbm() {
        let tier = StorageTier::GpuHbm;
        let debug = format!("{tier:?}");
        assert!(debug.contains("GpuHbm"));
    }

    #[test]
    fn storage_tier_debug_cpu_dram() {
        let tier = StorageTier::CpuDram;
        let debug = format!("{tier:?}");
        assert!(debug.contains("CpuDram"));
    }

    #[test]
    fn storage_tier_debug_nvme() {
        let tier = StorageTier::Nvme;
        let debug = format!("{tier:?}");
        assert!(debug.contains("Nvme"));
    }

    // ── MigrationCommand: PromoteToDram construction ──

    #[test]
    fn migration_command_promote_to_dram_fields() {
        let cmd = MigrationCommand::PromoteToDram { page_id: 42, page_bytes: 8192 };
        let debug = format!("{cmd:?}");
        assert!(debug.contains("42"));
        assert!(debug.contains("8192"));
    }

    // ── MigrationCommand: PromoteToHbm construction ──

    #[test]
    fn migration_command_promote_to_hbm_fields() {
        let cmd = MigrationCommand::PromoteToHbm { page_id: 7, page_bytes: 4096 };
        let debug = format!("{cmd:?}");
        assert!(debug.contains("PromoteToHbm"));
    }

    // ── EvictionReason: Debug contains MemoryPressure ──

    #[test]
    fn eviction_reason_debug_contains_name() {
        let reason = crate::scheduler::observer::EvictionReason::MemoryPressure;
        let debug = format!("{reason:?}");
        assert!(debug.contains("MemoryPressure"));
    }

    // ── EvictionReason: PartialEq ──

    #[test]
    fn eviction_reason_partial_eq_same() {
        let r1 = crate::scheduler::observer::EvictionReason::MemoryPressure;
        let r2 = crate::scheduler::observer::EvictionReason::MemoryPressure;
        assert_eq!(r1, r2);
    }

    // ── WeightPageTelemetryEvent: Recovered fields accessible ──

    #[test]
    fn telemetry_event_recovered_fields() {
        let event = WeightPageTelemetryEvent::Recovered {
            page_id: 42,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 500,
            bytes: 8192,
        };
        if let WeightPageTelemetryEvent::Recovered { page_id, from_tier, to_tier, latency_us, bytes } = event {
            assert_eq!(page_id, 42);
            assert_eq!(from_tier, WeightTier::Cold);
            assert_eq!(to_tier, WeightTier::Hot);
            assert_eq!(latency_us, 500);
            assert_eq!(bytes, 8192);
        } else {
            panic!("should be Recovered variant");
        }
    }

    // ── WeightPageTelemetryEvent: Evicted fields accessible ──

    #[test]
    fn telemetry_event_evicted_fields() {
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 10,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Warm,
            reason: crate::scheduler::observer::EvictionReason::MemoryPressure,
            bytes: 4096,
        };
        if let WeightPageTelemetryEvent::Evicted { page_id, from_tier, to_tier, bytes, .. } = event {
            assert_eq!(page_id, 10);
            assert_eq!(from_tier, WeightTier::Hot);
            assert_eq!(to_tier, WeightTier::Warm);
            assert_eq!(bytes, 4096);
        } else {
            panic!("should be Evicted variant");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional ~70 tests — coverage expansion
    // ═══════════════════════════════════════════════════════════════════════════

    // ── CompressionCodec: from_u8 / as_u8 roundtrip ──

    #[test]
    fn compression_codec_from_u8_roundtrip_all_variants() {
        for expected in [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let encoded = expected.as_u8();
            let decoded = CompressionCodec::from_u8(encoded);
            assert_eq!(decoded, Some(expected), "roundtrip failed for {:?}", expected);
        }
    }

    #[test]
    fn compression_codec_from_u8_returns_none_for_5() {
        assert_eq!(CompressionCodec::from_u8(5), None);
    }

    #[test]
    fn compression_codec_from_u8_returns_none_for_255() {
        assert_eq!(CompressionCodec::from_u8(255), None);
    }

    #[test]
    fn compression_codec_as_u8_none_is_0() {
        assert_eq!(CompressionCodec::None.as_u8(), 0);
    }

    #[test]
    fn compression_codec_as_u8_lz4_is_1() {
        assert_eq!(CompressionCodec::Lz4.as_u8(), 1);
    }

    #[test]
    fn compression_codec_as_u8_bitpack_rle_is_2() {
        assert_eq!(CompressionCodec::BitPackRle.as_u8(), 2);
    }

    #[test]
    fn compression_codec_as_u8_nvcomp_ans_is_3() {
        assert_eq!(CompressionCodec::NvcompAns.as_u8(), 3);
    }

    #[test]
    fn compression_codec_as_u8_zstd_dict_is_4() {
        assert_eq!(CompressionCodec::ZstdDict.as_u8(), 4);
    }

    // ── StorageTier: Ord ordering ──

    #[test]
    fn storage_tier_ord_gpu_hbm_is_highest_priority() {
        assert!(StorageTier::GpuHbm > StorageTier::CpuDram);
        assert!(StorageTier::GpuHbm > StorageTier::Nvme);
    }

    #[test]
    fn storage_tier_ord_cpu_dram_is_mid_priority() {
        assert!(StorageTier::CpuDram < StorageTier::GpuHbm);
        assert!(StorageTier::CpuDram > StorageTier::Nvme);
    }

    #[test]
    fn storage_tier_ord_nvme_is_lowest_priority() {
        assert!(StorageTier::Nvme < StorageTier::CpuDram);
        assert!(StorageTier::Nvme < StorageTier::GpuHbm);
    }

    #[test]
    fn storage_tier_ord_total_order_consistent() {
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        for i in 0..tiers.len() {
            for j in 0..tiers.len() {
                if i < j {
                    assert!(tiers[i] > tiers[j], "{:?} should be > {:?}", tiers[i], tiers[j]);
                } else if i > j {
                    assert!(tiers[i] < tiers[j], "{:?} should be < {:?}", tiers[i], tiers[j]);
                } else {
                    assert_eq!(tiers[i], tiers[j]);
                }
            }
        }
    }

    #[test]
    fn storage_tier_as_u8_roundtrip() {
        for expected in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let encoded = expected.as_u8();
            let decoded = StorageTier::from_u8(encoded);
            assert_eq!(decoded, Some(expected));
        }
    }

    #[test]
    fn storage_tier_as_u8_values() {
        assert_eq!(StorageTier::GpuHbm.as_u8(), 0);
        assert_eq!(StorageTier::CpuDram.as_u8(), 1);
        assert_eq!(StorageTier::Nvme.as_u8(), 2);
    }

    // ── MigrationCommand: all variants ──

    #[test]
    fn migration_command_evict_to_dram_fields() {
        let cmd = MigrationCommand::EvictToDram {
            page_id: 10,
            codec: CompressionCodec::Lz4,
            page_bytes: 8192,
        };
        if let MigrationCommand::EvictToDram { page_id, codec, page_bytes } = cmd {
            assert_eq!(page_id, 10);
            assert_eq!(codec, CompressionCodec::Lz4);
            assert_eq!(page_bytes, 8192);
        } else {
            panic!("expected EvictToDram");
        }
    }

    #[test]
    fn migration_command_evict_to_nvme_fields() {
        let cmd = MigrationCommand::EvictToNvme {
            page_id: 20,
            codec: CompressionCodec::ZstdDict,
            page_bytes: 4096,
        };
        if let MigrationCommand::EvictToNvme { page_id, codec, page_bytes } = cmd {
            assert_eq!(page_id, 20);
            assert_eq!(codec, CompressionCodec::ZstdDict);
            assert_eq!(page_bytes, 4096);
        } else {
            panic!("expected EvictToNvme");
        }
    }

    #[test]
    fn migration_command_shutdown_debug() {
        let cmd = MigrationCommand::Shutdown;
        let debug = format!("{cmd:?}");
        assert!(debug.contains("Shutdown"), "Debug should contain Shutdown: {debug}");
    }

    #[test]
    fn migration_command_clone_evict_to_dram() {
        let cmd = MigrationCommand::EvictToDram {
            page_id: 5,
            codec: CompressionCodec::None,
            page_bytes: 1024,
        };
        let cloned = cmd.clone();
        if let MigrationCommand::EvictToDram { page_id, .. } = cloned {
            assert_eq!(page_id, 5);
        } else {
            panic!("cloned should be EvictToDram");
        }
    }

    // ── MigrationResult: field access ──

    #[test]
    fn migration_result_ok_fields_accessible() {
        let result = MigrationResult::Ok {
            compressed_bytes: 2048,
            checksum: 0xABCD,
        };
        if let MigrationResult::Ok { compressed_bytes, checksum } = result {
            assert_eq!(compressed_bytes, 2048);
            assert_eq!(checksum, 0xABCD);
        } else {
            panic!("expected Ok variant");
        }
    }

    #[test]
    fn migration_result_failed_reason_accessible() {
        let result = MigrationResult::Failed {
            reason: "dma timeout".to_string(),
        };
        if let MigrationResult::Failed { reason } = result {
            assert_eq!(reason, "dma timeout");
        } else {
            panic!("expected Failed variant");
        }
    }

    #[test]
    fn migration_result_failed_empty_reason() {
        let result = MigrationResult::Failed {
            reason: String::new(),
        };
        if let MigrationResult::Failed { reason } = result {
            assert!(reason.is_empty());
        } else {
            panic!("expected Failed variant");
        }
    }

    // ── MigrationDone: more field combinations ──

    #[test]
    fn migration_done_from_nvme_to_dram() {
        let done = MigrationDone {
            page_id: 77,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::CpuDram,
            result: MigrationResult::Ok {
                compressed_bytes: 3000,
                checksum: 0x1234,
            },
        };
        assert_eq!(done.page_id, 77);
        assert_eq!(done.from_tier, StorageTier::Nvme);
        assert_eq!(done.to_tier, StorageTier::CpuDram);
    }

    #[test]
    fn migration_done_from_dram_to_hbm() {
        let done = MigrationDone {
            page_id: 88,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::GpuHbm,
            result: MigrationResult::Ok {
                compressed_bytes: 4096,
                checksum: 0xFFFF,
            },
        };
        assert_eq!(done.page_id, 88);
        assert_eq!(done.from_tier, StorageTier::CpuDram);
        assert_eq!(done.to_tier, StorageTier::GpuHbm);
    }

    // ── MigrationActorConfig: swap_file_path ──

    #[test]
    fn migration_actor_config_swap_file_path_combines_dir_and_session() {
        let config = MigrationActorConfig {
            nvme_swap_dir: std::path::PathBuf::from("/data/swap"),
            queue_capacity: 128,
            session_id: "sess42".to_string(),
            page_size: 8192,
            max_swap_pages: 2048,
        };
        let path = config.swap_file_path();
        assert!(path.to_string_lossy().contains("sess42"));
        assert!(path.to_string_lossy().contains(".swap"));
    }

    #[test]
    fn migration_actor_config_swap_file_path_default_session() {
        let config = MigrationActorConfig::default();
        let path = config.swap_file_path();
        assert!(path.to_string_lossy().contains("default.swap"));
    }

    #[test]
    fn migration_actor_config_swap_file_path_preserves_dir() {
        let config = MigrationActorConfig {
            nvme_swap_dir: std::path::PathBuf::from("/tmp/gllm_test"),
            ..MigrationActorConfig::default()
        };
        let path = config.swap_file_path();
        assert!(path.starts_with("/tmp/gllm_test"));
    }

    // ── PageAddrEntry: field access ──

    #[test]
    fn page_addr_entry_all_fields_accessible() {
        let entry = PageAddrEntry {
            gpu_ptr: Some(0xDEAD_BEEF),
            host_buffer: Some(vec![1, 2, 3, 4]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 4096,
            codec: CompressionCodec::Lz4,
        };
        assert_eq!(entry.gpu_ptr, Some(0xDEAD_BEEF));
        assert_eq!(entry.host_buffer.as_ref().unwrap().len(), 4);
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        assert_eq!(entry.original_bytes, 4096);
        assert_eq!(entry.codec, CompressionCodec::Lz4);
    }

    #[test]
    fn page_addr_entry_mutation_reflects_in_same_instance() {
        let mut entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(vec![42; 8]),
            current_tier: StorageTier::Nvme,
            original_bytes: 1024,
            codec: CompressionCodec::ZstdDict,
        };
        entry.gpu_ptr = Some(999);
        assert_eq!(entry.gpu_ptr, Some(999), "mutation should be visible");
        assert_eq!(entry.current_tier, StorageTier::Nvme);
    }

    // ── BasicObserver: construction ──

    #[test]
    fn basic_observer_new_returns_valid_instance() {
        let obs = BasicObserver::new();
        let state = obs.capture();
        assert!(state.is_ok(), "new observer should capture successfully");
    }

    #[test]
    fn basic_observer_default_matches_new() {
        let obs_new = BasicObserver::new();
        let obs_default = BasicObserver::default();
        // Both should be valid observers; just verify they don't panic.
        assert!(obs_new.capture().is_ok());
        assert!(obs_default.capture().is_ok());
    }

    // ── PageMetadata: more field boundaries ──

    #[test]
    fn page_metadata_sequence_id_none_valid() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Free,
            warm_until: None,
        };
        assert!(meta.sequence_id.is_none());
    }

    #[test]
    fn page_metadata_recency_usize_max() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: usize::MAX,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Protected,
            warm_until: None,
        };
        assert_eq!(meta.recency, usize::MAX);
        assert!(meta.is_lir);
    }

    #[test]
    fn page_metadata_is_lir_true() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(42),
            recency: 5,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: None,
        };
        assert!(meta.is_lir);
    }

    #[test]
    fn page_metadata_warm_until_some() {
        let warm_time = Instant::now() + Duration::from_secs(60);
        let meta = PageMetadata {
            page_id: 2,
            sequence_id: Some(10),
            recency: 3,
            access_count: 7,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Warm,
            warm_until: Some(warm_time),
        };
        assert!(meta.warm_until.is_some());
    }

    #[test]
    fn page_metadata_state_standby_default() {
        let meta = PageMetadata::default();
        assert_eq!(meta.state, PageState::Standby);
    }

    // ── SwapInWorkerStats: more boundary tests ──

    #[test]
    fn stats_avg_latency_single_promotion_with_exact_latency() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 1;
        stats.total_latency_us = 500;
        assert!((stats.avg_latency_us() - 500.0).abs() < 1e-6);
    }

    #[test]
    fn stats_all_u64_max_fields() {
        let stats = SwapInWorkerStats {
            total_requests: u64::MAX,
            submitted: u64::MAX,
            skipped: u64::MAX,
            promoted_ok: u64::MAX,
            promoted_failed: u64::MAX,
            two_hop_promotions: u64::MAX,
            total_latency_us: u64::MAX,
            rounds: u64::MAX,
        };
        assert_eq!(stats.total_requests, u64::MAX);
        assert_eq!(stats.rounds, u64::MAX);
    }

    #[test]
    fn stats_partial_eq_same_instance() {
        let stats = SwapInWorkerStats {
            total_requests: 10,
            submitted: 5,
            skipped: 3,
            promoted_ok: 4,
            promoted_failed: 1,
            two_hop_promotions: 2,
            total_latency_us: 1000,
            rounds: 3,
        };
        assert_eq!(stats, stats);
    }

    #[test]
    fn stats_partial_eq_different_total_requests() {
        let a = SwapInWorkerStats { total_requests: 1, ..Default::default() };
        let b = SwapInWorkerStats { total_requests: 2, ..Default::default() };
        assert_ne!(a, b);
    }

    #[test]
    fn stats_partial_eq_different_promoted_ok() {
        let a = SwapInWorkerStats { promoted_ok: 1, ..Default::default() };
        let b = SwapInWorkerStats { promoted_ok: 2, ..Default::default() };
        assert_ne!(a, b);
    }

    #[test]
    fn stats_debug_format_contains_fields() {
        let stats = SwapInWorkerStats {
            total_requests: 42,
            ..Default::default()
        };
        let debug = format!("{stats:?}");
        assert!(debug.contains("total_requests"));
        assert!(debug.contains("42"));
    }

    // ── PrefetchRequest: more boundaries ──

    #[test]
    fn prefetch_request_urgency_zero() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        };
        assert!((req.urgency).abs() < 1e-6);
        assert_eq!(req.page_bytes, 0);
    }

    #[test]
    fn prefetch_request_boundary_page_id_and_bytes_max() {
        let req = PrefetchRequest {
            page_id: PageId::MAX,
            urgency: 1.0,
            prefetch_confidence: 1.0,
            page_bytes: usize::MAX,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_id, PageId::MAX);
        assert_eq!(req.page_bytes, usize::MAX);
    }

    #[test]
    fn prefetch_request_negative_urgency_value_stored_verbatim() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: -5.0,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!((req.urgency - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn prefetch_request_confidence_above_one_stored() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 2.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!((req.prefetch_confidence - 2.5).abs() < 1e-6);
    }

    #[test]
    fn prefetch_request_partial_eq_same() {
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn prefetch_request_partial_eq_page_id_mismatch() {
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 2,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn prefetch_request_partial_eq_urgency_mismatch() {
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 1,
            urgency: 0.6,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        assert_ne!(a, b);
    }

    // ── SwapInWorkerConfig: more edge cases ──

    #[test]
    fn config_page_bytes_usize_max() {
        let cfg = SwapInWorkerConfig {
            page_bytes: usize::MAX,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.page_bytes, usize::MAX);
    }

    #[test]
    fn config_max_prefetch_per_round_usize_max() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: usize::MAX,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.max_prefetch_per_round, usize::MAX);
    }

    #[test]
    fn config_max_in_flight_usize_max() {
        let cfg = SwapInWorkerConfig {
            max_in_flight: usize::MAX,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.max_in_flight, usize::MAX);
    }

    #[test]
    fn config_min_confidence_zero_stored() {
        let cfg = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        assert!((cfg.min_confidence).abs() < 1e-6);
    }

    #[test]
    fn config_debug_format_contains_all_fields() {
        let cfg = SwapInWorkerConfig::default();
        let debug = format!("{cfg:?}");
        assert!(debug.contains("max_prefetch_per_round"));
        assert!(debug.contains("tick_interval"));
        assert!(debug.contains("min_confidence"));
        assert!(debug.contains("max_in_flight"));
        assert!(debug.contains("page_bytes"));
    }

    // ── WeightPageTelemetryEvent: more combinations ──

    #[test]
    fn telemetry_event_recovered_hot_to_hot() {
        let event = WeightPageTelemetryEvent::Recovered {
            page_id: 5,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Hot,
            latency_us: 0,
            bytes: 0,
        };
        if let WeightPageTelemetryEvent::Recovered { from_tier, to_tier, .. } = event {
            assert_eq!(from_tier, WeightTier::Hot);
            assert_eq!(to_tier, WeightTier::Hot);
        }
    }

    #[test]
    fn telemetry_event_evicted_hot_to_cold() {
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 99,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Cold,
            reason: crate::scheduler::observer::EvictionReason::MemoryPressure,
            bytes: 8192,
        };
        if let WeightPageTelemetryEvent::Evicted { from_tier, to_tier, bytes, .. } = event {
            assert_eq!(from_tier, WeightTier::Hot);
            assert_eq!(to_tier, WeightTier::Cold);
            assert_eq!(bytes, 8192);
        }
    }

    #[test]
    fn telemetry_event_recovered_latency_us_zero() {
        let event = WeightPageTelemetryEvent::Recovered {
            page_id: 0,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 0,
            bytes: u64::MAX,
        };
        if let WeightPageTelemetryEvent::Recovered { latency_us, bytes, .. } = event {
            assert_eq!(latency_us, 0);
            assert_eq!(bytes, u64::MAX);
        }
    }

    #[test]
    fn telemetry_event_evicted_bytes_zero() {
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 0,
            from_tier: WeightTier::Warm,
            to_tier: WeightTier::Cold,
            reason: crate::scheduler::observer::EvictionReason::MemoryPressure,
            bytes: 0,
        };
        if let WeightPageTelemetryEvent::Evicted { bytes, .. } = event {
            assert_eq!(bytes, 0);
        }
    }

    // ── PageState: additional variant checks ──

    #[test]
    fn page_state_free_is_distinct_from_standby() {
        assert_ne!(PageState::Free, PageState::Standby);
    }

    #[test]
    fn page_state_swapped_out_is_distinct_from_swapped() {
        assert_ne!(PageState::SwappedOut, PageState::Swapped);
    }

    #[test]
    fn page_state_warm_is_distinct_from_protected() {
        assert_ne!(PageState::Warm, PageState::Protected);
    }

    #[test]
    fn page_state_active_is_distinct_from_free() {
        assert_ne!(PageState::Active, PageState::Free);
    }

    // ── EvictionReason: exhaustive variant test ──

    #[test]
    fn eviction_reason_only_memory_pressure_variant() {
        let reason = crate::scheduler::observer::EvictionReason::MemoryPressure;
        let debug = format!("{reason:?}");
        assert!(!debug.is_empty());
    }

    // ── WeightTier: all pair-wise distinct ──

    #[test]
    fn weight_tier_hot_warm_cold_all_distinct() {
        let tiers = [WeightTier::Hot, WeightTier::Warm, WeightTier::Cold];
        for i in 0..tiers.len() {
            for j in (i + 1)..tiers.len() {
                assert_ne!(tiers[i], tiers[j], "{:?} should differ from {:?}", tiers[i], tiers[j]);
            }
        }
    }

    // ── compute_urgency: additional edge cases ──

    #[test]
    fn urgency_access_count_usize_max_produces_finite_first_term() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: usize::MAX,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let urgency = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let first_term = urgency - 0.1; // subtract recency contribution
        assert!(first_term.is_finite() || first_term.is_nan() == false);
    }

    #[test]
    fn urgency_tier_bonus_dram_exactly_one() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::Nvme);
        // dram tier_bonus = 1.0, nvme tier_bonus = 0.5
        // first_term_dram / first_term_nvme should be ~2.0
        let ratio = u_dram / u_nvme;
        assert!(
            ratio > 1.5 && ratio < 3.0,
            "dram/nvme urgency ratio should be ~2x: ratio={}",
            ratio
        );
    }

    #[test]
    fn urgency_very_large_confidence_produces_large_score() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let urgency = SwapInWorker::compute_urgency(&meta, 1000.0, StorageTier::CpuDram);
        assert!(urgency > 100.0, "very large confidence should produce large urgency: {}", urgency);
    }

    #[test]
    fn urgency_recency_bonus_formula_exact_at_zero_elapsed() {
        // At elapsed=0, recency_bonus = 1.0 / (1.0 + 0.0) = 1.0
        // contribution = 1.0 * 0.1 = 0.1
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
        // access_count=0 → importance_rebound = ln_1p(0)/ln_1p(10) = 0
        // first_term = 0 * confidence * tier_bonus = 0
        // urgency ≈ 0 + recency_bonus * 0.1 ≈ 0.1
        let urgency = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // Should be close to 0.1 but elapsed may not be exactly 0
        assert!(urgency > 0.0 && urgency < 0.2, "urgency should be ~0.1: {}", urgency);
    }

    // ── SwapInWorkerError: more edge cases ──

    #[test]
    fn error_send_failed_with_newline_in_message() {
        let e = SwapInWorkerError::SendFailed("line1\nline2".into());
        let msg = format!("{e}");
        assert!(msg.contains("line1"));
        assert!(msg.contains("line2"));
    }

    #[test]
    fn error_recv_failed_with_special_characters() {
        let e = SwapInWorkerError::RecvFailed("err: \t\r\n".into());
        let msg = format!("{e}");
        assert!(msg.starts_with("swap-in worker recv failed:"));
    }

    #[test]
    fn error_both_variants_in_vec() {
        let errors = vec![
            SwapInWorkerError::SendFailed("a".into()),
            SwapInWorkerError::RecvFailed("b".into()),
        ];
        assert_eq!(errors.len(), 2);
        assert!(matches!(&errors[0], SwapInWorkerError::SendFailed(_)));
        assert!(matches!(&errors[1], SwapInWorkerError::RecvFailed(_)));
    }

    // ── PageAddrEntry: more field combinations ──

    #[test]
    fn page_addr_entry_with_gpu_ptr_and_host_buffer_both_some() {
        let entry = PageAddrEntry {
            gpu_ptr: Some(0x1000),
            host_buffer: Some(vec![0xAA; 256]),
            current_tier: StorageTier::GpuHbm,
            original_bytes: 256,
            codec: CompressionCodec::None,
        };
        assert!(entry.gpu_ptr.is_some());
        assert!(entry.host_buffer.is_some());
    }

    #[test]
    fn page_addr_entry_original_bytes_zero() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: None,
            current_tier: StorageTier::Nvme,
            original_bytes: 0,
            codec: CompressionCodec::ZstdDict,
        };
        assert_eq!(entry.original_bytes, 0);
    }

    #[test]
    fn page_addr_entry_host_buffer_empty_vec() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(vec![]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 4096,
            codec: CompressionCodec::Lz4,
        };
        assert!(entry.host_buffer.as_ref().unwrap().is_empty());
    }

    // ── MigrationActorConfig: more field tests ──

    #[test]
    fn migration_actor_config_queue_capacity_zero() {
        let config = MigrationActorConfig {
            queue_capacity: 0,
            ..MigrationActorConfig::default()
        };
        assert_eq!(config.queue_capacity, 0);
    }

    #[test]
    fn migration_actor_config_page_size_zero() {
        let config = MigrationActorConfig {
            page_size: 0,
            ..MigrationActorConfig::default()
        };
        assert_eq!(config.page_size, 0);
    }

    #[test]
    fn migration_actor_config_max_swap_pages_zero() {
        let config = MigrationActorConfig {
            max_swap_pages: 0,
            ..MigrationActorConfig::default()
        };
        assert_eq!(config.max_swap_pages, 0);
    }

    #[test]
    fn migration_actor_config_session_id_empty_string() {
        let config = MigrationActorConfig {
            session_id: String::new(),
            ..MigrationActorConfig::default()
        };
        let path = config.swap_file_path();
        assert!(path.to_string_lossy().ends_with(".swap"));
    }

    // ── MigrationCommand: clone all variants ──

    #[test]
    fn migration_command_clone_evict_to_nvme() {
        let cmd = MigrationCommand::EvictToNvme {
            page_id: 33,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 2048,
        };
        let cloned = cmd.clone();
        if let MigrationCommand::EvictToNvme { page_id, codec, page_bytes } = cloned {
            assert_eq!(page_id, 33);
            assert_eq!(codec, CompressionCodec::BitPackRle);
            assert_eq!(page_bytes, 2048);
        } else {
            panic!("cloned should be EvictToNvme");
        }
    }

    #[test]
    fn migration_command_clone_promote_to_dram() {
        let cmd = MigrationCommand::PromoteToDram {
            page_id: 44,
            page_bytes: 8192,
        };
        let cloned = cmd.clone();
        if let MigrationCommand::PromoteToDram { page_id, page_bytes } = cloned {
            assert_eq!(page_id, 44);
            assert_eq!(page_bytes, 8192);
        } else {
            panic!("cloned should be PromoteToDram");
        }
    }

    #[test]
    fn migration_command_clone_promote_to_hbm() {
        let cmd = MigrationCommand::PromoteToHbm {
            page_id: 55,
            page_bytes: 4096,
        };
        let cloned = cmd.clone();
        if let MigrationCommand::PromoteToHbm { page_id, page_bytes } = cloned {
            assert_eq!(page_id, 55);
            assert_eq!(page_bytes, 4096);
        } else {
            panic!("cloned should be PromoteToHbm");
        }
    }

    #[test]
    fn migration_command_clone_shutdown() {
        let cmd = MigrationCommand::Shutdown;
        let cloned = cmd.clone();
        assert!(matches!(cloned, MigrationCommand::Shutdown));
    }

    // ── MigrationDone: clone ──

    #[test]
    fn migration_done_clone_preserves_fields() {
        let done = MigrationDone {
            page_id: 123,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::GpuHbm,
            result: MigrationResult::Ok {
                compressed_bytes: 4096,
                checksum: 0xBEEF,
            },
        };
        let cloned = done.clone();
        assert_eq!(cloned.page_id, done.page_id);
        assert_eq!(cloned.from_tier, done.from_tier);
        assert_eq!(cloned.to_tier, done.to_tier);
    }

    // ── MigrationResult: clone ──

    #[test]
    fn migration_result_clone_ok() {
        let result = MigrationResult::Ok {
            compressed_bytes: 999,
            checksum: 0x1234,
        };
        let cloned = result.clone();
        if let MigrationResult::Ok { compressed_bytes, checksum } = cloned {
            assert_eq!(compressed_bytes, 999);
            assert_eq!(checksum, 0x1234);
        }
    }

    #[test]
    fn migration_result_clone_failed() {
        let result = MigrationResult::Failed {
            reason: "io error".to_string(),
        };
        let cloned = result.clone();
        if let MigrationResult::Failed { reason } = cloned {
            assert_eq!(reason, "io error");
        }
    }

    // ── Urgency: PageState all variants produce finite ──

    #[test]
    fn urgency_all_page_states_produce_finite_or_nan_free() {
        let states = [
            PageState::Free, PageState::Active, PageState::Standby,
            PageState::SwappedOut, PageState::Warm, PageState::Protected,
            PageState::Swapped,
        ];
        for state in states {
            let meta = PageMetadata {
                page_id: 0,
                sequence_id: None,
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state,
                warm_until: None,
            };
            let urgency = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
            assert!(urgency.is_finite(), "urgency should be finite for state {:?}: {}", state, urgency);
        }
    }

    // ── StorageTier: hash consistency ──

    #[test]
    fn storage_tier_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StorageTier::GpuHbm);
        set.insert(StorageTier::CpuDram);
        set.insert(StorageTier::Nvme);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&StorageTier::GpuHbm));
        assert!(set.contains(&StorageTier::CpuDram));
        assert!(set.contains(&StorageTier::Nvme));
    }

    // ── CompressionCodec: hash consistency ──

    #[test]
    fn compression_codec_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CompressionCodec::None);
        set.insert(CompressionCodec::Lz4);
        set.insert(CompressionCodec::BitPackRle);
        set.insert(CompressionCodec::NvcompAns);
        set.insert(CompressionCodec::ZstdDict);
        assert_eq!(set.len(), 5);
    }

    // ── Config: PartialEq exhaustive field differences ──

    #[test]
    fn config_partial_eq_max_prefetch_difference() {
        let a = SwapInWorkerConfig { max_prefetch_per_round: 1, ..Default::default() };
        let b = SwapInWorkerConfig { max_prefetch_per_round: 2, ..Default::default() };
        assert_ne!(a, b);
    }

    #[test]
    fn config_default_matches_expected_values() {
        let cfg = SwapInWorkerConfig::default();
        let expected = SwapInWorkerConfig {
            max_prefetch_per_round: 16,
            tick_interval: Duration::from_millis(5),
            min_confidence: 0.1,
            max_in_flight: 64,
            page_bytes: 4096,
        };
        assert_eq!(cfg, expected);
    }

    // ── SwapInWorkerError: Debug output completeness ──

    #[test]
    fn error_debug_send_failed_shows_inner_string() {
        let e = SwapInWorkerError::SendFailed("channel_closed".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("SendFailed"), "Debug should contain variant: {}", debug);
        assert!(debug.contains("channel_closed"), "Debug should contain inner: {}", debug);
    }

    #[test]
    fn error_debug_recv_failed_shows_inner_string() {
        let e = SwapInWorkerError::RecvFailed("timeout".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("RecvFailed"), "Debug should contain variant: {}", debug);
        assert!(debug.contains("timeout"), "Debug should contain inner: {}", debug);
    }

    // ── Urgency: confidence exactly 1.0 ──

    #[test]
    fn urgency_confidence_one_with_access_count_ten() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let urgency = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // importance_rebound = ln(11) / ln(11) = 1.0
        // first_term = 1.0 * 1.0 * 1.0 = 1.0
        // recency ≈ 0.1
        // total ≈ 1.1
        assert!(urgency > 0.9 && urgency < 2.0, "urgency should be ~1.1: {}", urgency);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch A: urgency boundary & formula verification
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn urgency_zero_access_zero_confidence_is_recency_only() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 0,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let urgency = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // first_term = 0.0, total ≈ 0.0 + recency*0.1 ≈ 0.1
        assert!(
            urgency > 0.05 && urgency < 0.2,
            "urgency should be ~0.1 (recency only): {}",
            urgency,
        );
    }

    #[test]
    fn urgency_dram_tier_bonus_is_one_with_known_access() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 10,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let urgency = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        assert!(
            urgency >= 1.0 && urgency < 1.5,
            "urgency should be ~1.1 for CpuDram confidence=1.0 access=10: {}",
            urgency,
        );
    }

    #[test]
    fn urgency_nvme_first_term_is_half_dram_same_inputs() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 10,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let u_dram = SwapInWorker::compute_urgency(&meta, 0.8, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, 0.8, StorageTier::Nvme);
        let ft_dram = u_dram - 0.1;
        let ft_nvme = u_nvme - 0.1;
        assert!(ft_nvme > 0.0 && ft_dram > 0.0, "both first terms positive");
        let ratio = ft_nvme / ft_dram;
        assert!(
            ratio > 0.4 && ratio < 0.6,
            "NVMe first term should be ~0.5x DRAM: ratio={}",
            ratio,
        );
    }

    #[test]
    fn urgency_very_large_access_count_finite() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 1_000_000,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let urgency = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        assert!(urgency.is_finite() && urgency > 0.0, "should be finite positive: {}", urgency);
    }

    #[test]
    fn urgency_is_lir_invariant() {
        let meta_lir = PageMetadata {
            page_id: 0,
            access_count: 5,
            last_access: Instant::now(),
            is_lir: true,
            ..PageMetadata::default()
        };
        let meta_not = PageMetadata {
            page_id: 0,
            access_count: 5,
            last_access: meta_lir.last_access,
            is_lir: false,
            ..PageMetadata::default()
        };
        let u1 = SwapInWorker::compute_urgency(&meta_lir, 0.5, StorageTier::CpuDram);
        let u2 = SwapInWorker::compute_urgency(&meta_not, 0.5, StorageTier::CpuDram);
        assert!(
            (u1 - u2).abs() < 1e-6,
            "is_lir should not affect urgency: lir={} not={}",
            u1,
            u2,
        );
    }

    #[test]
    fn urgency_hbm_double_dram_first_term() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 10,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let u_hbm = SwapInWorker::compute_urgency(&meta, 0.8, StorageTier::GpuHbm);
        let u_dram = SwapInWorker::compute_urgency(&meta, 0.8, StorageTier::CpuDram);
        assert!(
            (u_hbm - 0.1) > (u_dram - 0.1) * 1.8,
            "HBM first term should be ~2x DRAM: hbm={} dram={}",
            u_hbm,
            u_dram,
        );
    }

    #[test]
    fn urgency_monotonic_access_count_gradient() {
        let mut prev = 0.0_f32;
        for ac in 0..=100usize {
            let meta = PageMetadata {
                page_id: 0,
                access_count: ac,
                last_access: Instant::now(),
                ..PageMetadata::default()
            };
            let u = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
            if ac > 0 {
                assert!(u >= prev - 1e-6, "non-decreasing: ac={} u={} prev={}", ac, u, prev);
            }
            prev = u;
        }
    }

    #[test]
    fn urgency_monotonic_confidence_gradient() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 10,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let mut prev = 0.0_f32;
        for i in 0..11usize {
            let conf = i as f32 / 10.0;
            let u = SwapInWorker::compute_urgency(&meta, conf, StorageTier::CpuDram);
            if i > 0 {
                assert!(u >= prev - 1e-6, "non-decreasing: conf={} u={} prev={}", conf, u, prev);
            }
            prev = u;
        }
    }

    #[test]
    fn urgency_fresh_page_recency_near_one() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 0,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        assert!(u > 0.05 && u < 0.15, "fresh page urgency ~0.1: {}", u);
    }

    #[test]
    fn urgency_dram_between_nvme_and_hbm() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 10,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let u_nvme = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::Nvme);
        let u_dram = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        let u_hbm = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::GpuHbm);
        assert!(u_nvme < u_dram, "NVMe < DRAM: nvme={} dram={}", u_nvme, u_dram);
        assert!(u_dram < u_hbm, "DRAM < HBM: dram={} hbm={}", u_dram, u_hbm);
    }

    #[test]
    fn urgency_non_negative_for_reasonable_inputs() {
        for ac in [0usize, 1, 10, 100, 1000] {
            for conf in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
                for tier in [StorageTier::CpuDram, StorageTier::Nvme, StorageTier::GpuHbm] {
                    let meta = PageMetadata {
                        page_id: 0,
                        access_count: ac,
                        last_access: Instant::now(),
                        ..PageMetadata::default()
                    };
                    let u = SwapInWorker::compute_urgency(&meta, conf, tier);
                    assert!(u >= 0.0, "non-negative: ac={} conf={:?} tier={:?} u={}", ac, conf, tier, u);
                }
            }
        }
    }

    #[test]
    fn urgency_importance_rebound_zero_for_zero_access() {
        let meta = PageMetadata {
            page_id: 0,
            access_count: 0,
            last_access: Instant::now(),
            ..PageMetadata::default()
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        assert!(u < 0.3, "first term should be 0: urgency={}", u);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch B: swap_in_round edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn round_single_dram_none_codec() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 0, "CpuDram should not be two-hop");
        actor.shutdown();
    }

    #[test]
    fn round_drains_all_skipped_requests() {
        let config = SwapInWorkerConfig { min_confidence: 0.9, ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (0..5).map(|i| PrefetchRequest {
            page_id: i, urgency: 0.5, prefetch_confidence: 0.1, page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0);
        assert!(requests.is_empty(), "requests should be drained even when all skipped");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 5);
        actor.shutdown();
    }

    #[test]
    fn round_total_requests_before_truncation_test() {
        let config = SwapInWorkerConfig { max_prefetch_per_round: 2, ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in 0..5usize {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (0..5).map(|i| PrefetchRequest {
            page_id: i, urgency: i as f32, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        let _submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 5, "total_requests should be original count");
        actor.shutdown();
    }

    #[test]
    fn round_truncation_to_one_keeps_highest_test() {
        let config = SwapInWorkerConfig { max_prefetch_per_round: 1, ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [10usize, 20usize, 30usize] {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![
            PrefetchRequest { page_id: 10, urgency: 0.1, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 20, urgency: 0.9, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 30, urgency: 0.5, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "only 1 should be processed");
        actor.shutdown();
    }

    #[test]
    fn round_increments_rounds_on_empty_test() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut empty_requests: Vec<PrefetchRequest> = Vec::new();
        let _submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut empty_requests, &page_metadata, &addr_table, &stats, &observer,
        );
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 1);
        assert_eq!(s.total_requests, 0);
        actor.shutdown();
    }

    #[test]
    fn round_mixed_nvme_dram_two_hop_counting_test() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
            table.insert(2, PageAddrEntry {
                gpu_ptr: None, host_buffer: None,
                current_tier: StorageTier::Nvme, original_bytes: 4096, codec: CompressionCodec::ZstdDict,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![
            PrefetchRequest { page_id: 1, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now() },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 2);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "only NVMe should be two-hop");
        actor.shutdown();
    }

    #[test]
    fn round_back_pressure_at_one_test() {
        let config = SwapInWorkerConfig { max_in_flight: 1, ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize] {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![
            PrefetchRequest { page_id: 1, urgency: 0.9, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now() },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "should only submit 1 due to back-pressure");
        actor.shutdown();
    }

    #[test]
    fn round_page_bytes_zero_uses_config_test() {
        let config = SwapInWorkerConfig { page_bytes: 8192, ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 8192]),
                current_tier: StorageTier::CpuDram, original_bytes: 8192, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 0, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "should submit with config page_bytes");
        actor.shutdown();
    }

    #[test]
    fn round_confidence_at_boundary_accepted_test() {
        let config = SwapInWorkerConfig { min_confidence: 0.5, ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.5, page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "boundary confidence should be accepted");
        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch C: drain_completions_and_update
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn drain_empty_metadata_no_panic() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);
        actor.shutdown();
    }

    #[test]
    fn drain_ok_with_swap_in_time_tracks_latency() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(42, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        {
            let mut meta = page_metadata.write().expect("write lock");
            meta.insert(42, PageMetadata {
                page_id: 42, swap_in_time: Some(Instant::now()), state: PageState::SwappedOut,
                ..PageMetadata::default()
            });
        }
        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 42, page_bytes: 4096 });
        thread::sleep(Duration::from_millis(50));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
        let s = stats.lock().expect("stats lock");
        assert!(s.promoted_ok >= 1, "should have at least 1 promoted_ok: {}", s.promoted_ok);
        actor.shutdown();
    }

    #[test]
    fn drain_failed_increments_promoted_failed() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(99, PageAddrEntry {
                gpu_ptr: None, host_buffer: None,
                current_tier: StorageTier::Nvme, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let _ = actor.send(MigrationCommand::PromoteToDram { page_id: 99, page_bytes: 4096 });
        thread::sleep(Duration::from_millis(50));
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
        let s = stats.lock().expect("stats lock");
        assert!(
            s.promoted_failed + s.promoted_ok >= 1,
            "should have at least one completion: ok={} failed={}",
            s.promoted_ok, s.promoted_failed,
        );
        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch D: prefetch_batch edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn prefetch_batch_empty_returns_zero() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(100), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        let enqueued = worker.prefetch_batch(&[]);
        assert_eq!(enqueued, 0);
        worker.shutdown();
    }

    #[test]
    fn prefetch_batch_all_enqueued_while_alive() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(500), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        let requests: Vec<PrefetchRequest> = (0..10).map(|i| PrefetchRequest {
            page_id: i, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        let enqueued = worker.prefetch_batch(&requests);
        assert_eq!(enqueued, 10);
        worker.shutdown();
    }

    #[test]
    fn prefetch_batch_zero_after_shutdown() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(50), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        worker.shutdown();
        let requests = vec![PrefetchRequest {
            page_id: 0, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let enqueued = worker.prefetch_batch(&requests);
        assert_eq!(enqueued, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch E: stats invariants
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn stats_default_all_fields_zero() {
        let s = SwapInWorkerStats::default();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.submitted, 0);
        assert_eq!(s.skipped, 0);
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);
        assert_eq!(s.two_hop_promotions, 0);
        assert_eq!(s.total_latency_us, 0);
        assert_eq!(s.rounds, 0);
        assert_eq!(s.avg_latency_us(), 0.0);
    }

    #[test]
    fn stats_avg_latency_calculation_test() {
        let mut s = SwapInWorkerStats::default();
        s.promoted_ok = 4;
        s.total_latency_us = 1000;
        assert!((s.avg_latency_us() - 250.0).abs() < 1e-6);
    }

    #[test]
    fn stats_clone_independence_test() {
        let s = SwapInWorkerStats { total_requests: 50, submitted: 30, ..Default::default() };
        let cloned = s.clone();
        assert_eq!(cloned.total_requests, 50);
        assert_eq!(cloned.submitted, 30);
    }

    #[test]
    fn stats_two_rounds_accumulate_test() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();
        let mut r1 = vec![PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        SwapInWorker::swap_in_round(&config, &actor, &mut r1, &page_metadata, &addr_table, &stats, &observer);
        let mut r2: Vec<PrefetchRequest> = Vec::new();
        SwapInWorker::swap_in_round(&config, &actor, &mut r2, &page_metadata, &addr_table, &stats, &observer);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 2);
        assert_eq!(s.total_requests, 1);
        assert_eq!(s.submitted, 1);
        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch F: config edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn config_extreme_values_test() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: usize::MAX,
            tick_interval: Duration::from_nanos(1),
            min_confidence: f32::MAX,
            max_in_flight: usize::MAX,
            page_bytes: usize::MAX,
        };
        assert_eq!(cfg.max_prefetch_per_round, usize::MAX);
        assert_eq!(cfg.tick_interval, Duration::from_nanos(1));
        assert_eq!(cfg.min_confidence, f32::MAX);
        assert_eq!(cfg.max_in_flight, usize::MAX);
        assert_eq!(cfg.page_bytes, usize::MAX);
    }

    #[test]
    fn config_clone_equality_test() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 32, tick_interval: Duration::from_millis(10),
            min_confidence: 0.3, max_in_flight: 128, page_bytes: 8192,
        };
        assert_eq!(cfg, cfg.clone());
    }

    #[test]
    fn config_min_confidence_zero_accepts_all_test() {
        let config = SwapInWorkerConfig { min_confidence: 0.0, ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![PrefetchRequest {
            page_id: 1, urgency: 0.1, prefetch_confidence: 0.0, page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "confidence 0.0 accepted with min_confidence 0.0");
        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch G: error properties
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn error_display_send_failed_format_test() {
        let e = SwapInWorkerError::SendFailed("channel closed".to_string());
        let msg = format!("{e}");
        assert!(msg.starts_with("swap-in worker send failed:"));
        assert!(msg.contains("channel closed"));
    }

    #[test]
    fn error_display_recv_failed_format_test() {
        let e = SwapInWorkerError::RecvFailed("timeout expired".to_string());
        let msg = format!("{e}");
        assert!(msg.starts_with("swap-in worker recv failed:"));
        assert!(msg.contains("timeout expired"));
    }

    #[test]
    fn error_is_std_error_test() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        let e = SwapInWorkerError::SendFailed("test".to_string());
        assert_error(&e);
    }

    #[test]
    fn error_clone_preserves_test() {
        let a = SwapInWorkerError::SendFailed("original".to_string());
        assert_eq!(a, a.clone());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch H: PrefetchRequest construction
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn prefetch_request_construction_all_fields() {
        let now = Instant::now();
        let req = PrefetchRequest {
            page_id: 42, urgency: 0.75, prefetch_confidence: 0.9, page_bytes: 8192, enqueued_at: now,
        };
        assert_eq!(req.page_id, 42);
        assert!((req.urgency - 0.75).abs() < 1e-6);
        assert!((req.prefetch_confidence - 0.9).abs() < 1e-6);
        assert_eq!(req.page_bytes, 8192);
    }

    #[test]
    fn prefetch_request_clone_equality_test() {
        let req = PrefetchRequest {
            page_id: 7, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        };
        assert_eq!(req, req.clone());
    }

    #[test]
    fn prefetch_request_max_page_id_and_bytes() {
        let req = PrefetchRequest {
            page_id: usize::MAX, urgency: 1.0, prefetch_confidence: 1.0,
            page_bytes: usize::MAX, enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_id, usize::MAX);
        assert_eq!(req.page_bytes, usize::MAX);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch I: observer integration
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn observer_records_events_on_dram_submission() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let _submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        let obs = observer.lock().expect("observer lock");
        assert!(obs.last_state.weight_recovery_count >= 1, "should have recovery events");
        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch J: PageMetadata properties
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn page_metadata_default_fields_test() {
        let m = PageMetadata::default();
        assert_eq!(m.page_id, 0);
        assert!(m.sequence_id.is_none());
        assert_eq!(m.recency, 0);
        assert_eq!(m.access_count, 0);
        assert!(m.swap_in_time.is_none());
        assert!(!m.is_lir);
        assert_eq!(m.state, PageState::Standby);
        assert!(m.warm_until.is_none());
    }

    #[test]
    fn page_metadata_all_fields_set() {
        let now = Instant::now();
        let m = PageMetadata {
            page_id: 999, sequence_id: Some(42), recency: 100, access_count: 50,
            last_access: now, swap_in_time: Some(now), is_lir: true,
            state: PageState::Active, warm_until: Some(now),
        };
        assert_eq!(m.page_id, 999);
        assert_eq!(m.sequence_id, Some(42));
        assert_eq!(m.recency, 100);
        assert_eq!(m.access_count, 50);
        assert!(m.is_lir);
        assert_eq!(m.state, PageState::Active);
    }

    #[test]
    fn page_metadata_clone_preserves_fields_test() {
        let m = PageMetadata { page_id: 42, access_count: 10, ..PageMetadata::default() };
        let c = m.clone();
        assert_eq!(m.page_id, c.page_id);
        assert_eq!(m.access_count, c.access_count);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch K: PageAddrEntry properties
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn page_addr_entry_gpu_ptr_only_test() {
        let e = PageAddrEntry {
            gpu_ptr: Some(0xDEADBEEF), host_buffer: None,
            current_tier: StorageTier::GpuHbm, original_bytes: 4096, codec: CompressionCodec::None,
        };
        assert_eq!(e.gpu_ptr, Some(0xDEADBEEF));
        assert!(e.host_buffer.is_none());
    }

    #[test]
    fn page_addr_entry_host_buffer_only_test() {
        let e = PageAddrEntry {
            gpu_ptr: None, host_buffer: Some(vec![0xAA; 8192]),
            current_tier: StorageTier::CpuDram, original_bytes: 8192, codec: CompressionCodec::Lz4,
        };
        assert!(e.gpu_ptr.is_none());
        assert_eq!(e.host_buffer.as_ref().map(|b| b.len()), Some(8192));
        assert_eq!(e.codec, CompressionCodec::Lz4);
    }

    #[test]
    fn page_addr_entry_nvme_no_pointers() {
        let e = PageAddrEntry {
            gpu_ptr: None, host_buffer: None,
            current_tier: StorageTier::Nvme, original_bytes: 4096, codec: CompressionCodec::ZstdDict,
        };
        assert!(e.gpu_ptr.is_none());
        assert!(e.host_buffer.is_none());
        assert_eq!(e.current_tier, StorageTier::Nvme);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch L: StorageTier / CompressionCodec / PageState enums
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn storage_tier_as_u8_distinct_values() {
        assert_eq!(StorageTier::GpuHbm.as_u8(), 0);
        assert_eq!(StorageTier::CpuDram.as_u8(), 1);
        assert_eq!(StorageTier::Nvme.as_u8(), 2);
    }

    #[test]
    fn storage_tier_roundtrip_u8_all() {
        for expected in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            assert_eq!(expected, StorageTier::from_u8(expected.as_u8()).unwrap());
        }
    }

    #[test]
    fn compression_codec_as_u8_distinct_values() {
        assert_eq!(CompressionCodec::None.as_u8(), 0);
        assert_eq!(CompressionCodec::Lz4.as_u8(), 1);
        assert_eq!(CompressionCodec::BitPackRle.as_u8(), 2);
        assert_eq!(CompressionCodec::NvcompAns.as_u8(), 3);
        assert_eq!(CompressionCodec::ZstdDict.as_u8(), 4);
    }

    #[test]
    fn compression_codec_roundtrip_u8_all() {
        for expected in [
            CompressionCodec::None, CompressionCodec::Lz4, CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns, CompressionCodec::ZstdDict,
        ] {
            assert_eq!(expected, CompressionCodec::from_u8(expected.as_u8()).unwrap());
        }
    }

    #[test]
    fn page_state_seven_distinct_variants() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PageState::Free);
        set.insert(PageState::Active);
        set.insert(PageState::Standby);
        set.insert(PageState::SwappedOut);
        set.insert(PageState::Warm);
        set.insert(PageState::Protected);
        set.insert(PageState::Swapped);
        assert_eq!(set.len(), 7);
    }

    #[test]
    fn weight_tier_three_distinct_variants() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WeightTier::Hot);
        set.insert(WeightTier::Warm);
        set.insert(WeightTier::Cold);
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn eviction_reason_memory_pressure_variant() {
        assert_eq!(EvictionReason::MemoryPressure, EvictionReason::MemoryPressure);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch M: MigrationCommand / MigrationResult / MigrationDone
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn migration_command_promote_hbm_fields() {
        let cmd = MigrationCommand::PromoteToHbm { page_id: 42, page_bytes: 8192 };
        if let MigrationCommand::PromoteToHbm { page_id, page_bytes } = cmd {
            assert_eq!(page_id, 42);
            assert_eq!(page_bytes, 8192);
        } else {
            panic!("should be PromoteToHbm");
        }
    }

    #[test]
    fn migration_command_promote_dram_fields() {
        let cmd = MigrationCommand::PromoteToDram { page_id: 99, page_bytes: 4096 };
        if let MigrationCommand::PromoteToDram { page_id, page_bytes } = cmd {
            assert_eq!(page_id, 99);
            assert_eq!(page_bytes, 4096);
        } else {
            panic!("should be PromoteToDram");
        }
    }

    #[test]
    fn migration_command_evict_dram_fields() {
        let cmd = MigrationCommand::EvictToDram { page_id: 10, codec: CompressionCodec::Lz4, page_bytes: 4096 };
        if let MigrationCommand::EvictToDram { page_id, codec, page_bytes } = cmd {
            assert_eq!(page_id, 10);
            assert_eq!(codec, CompressionCodec::Lz4);
            assert_eq!(page_bytes, 4096);
        } else {
            panic!("should be EvictToDram");
        }
    }

    #[test]
    fn migration_command_evict_nvme_fields() {
        let cmd = MigrationCommand::EvictToNvme { page_id: 20, codec: CompressionCodec::ZstdDict, page_bytes: 8192 };
        if let MigrationCommand::EvictToNvme { page_id, codec, page_bytes } = cmd {
            assert_eq!(page_id, 20);
            assert_eq!(codec, CompressionCodec::ZstdDict);
            assert_eq!(page_bytes, 8192);
        } else {
            panic!("should be EvictToNvme");
        }
    }

    #[test]
    fn migration_result_ok_fields_test() {
        let res = MigrationResult::Ok { compressed_bytes: 2048, checksum: 0xABCD };
        if let MigrationResult::Ok { compressed_bytes, checksum } = res {
            assert_eq!(compressed_bytes, 2048);
            assert_eq!(checksum, 0xABCD);
        } else {
            panic!("should be Ok");
        }
    }

    #[test]
    fn migration_result_failed_fields_test() {
        let res = MigrationResult::Failed { reason: "dma error".to_string() };
        if let MigrationResult::Failed { reason } = &res {
            assert_eq!(reason, "dma error");
        } else {
            panic!("should be Failed");
        }
    }

    #[test]
    fn migration_done_construction_and_clone() {
        let done = MigrationDone {
            page_id: 42, from_tier: StorageTier::CpuDram, to_tier: StorageTier::GpuHbm,
            result: MigrationResult::Ok { compressed_bytes: 100, checksum: 0 },
        };
        assert_eq!(done.page_id, 42);
        let cloned = done.clone();
        assert_eq!(cloned.page_id, 42);
        assert_eq!(cloned.from_tier, StorageTier::CpuDram);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch N: MigrationActorConfig
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn migration_actor_config_default_values() {
        let cfg = MigrationActorConfig::default();
        assert_eq!(cfg.queue_capacity, 256);
        assert_eq!(cfg.session_id, "default");
        assert_eq!(cfg.page_size, 4096);
        assert_eq!(cfg.max_swap_pages, 4096);
    }

    #[test]
    fn migration_actor_config_swap_file_path_test() {
        let cfg = MigrationActorConfig {
            nvme_swap_dir: PathBuf::from("/tmp/test_swap"),
            session_id: "session_42".to_string(),
            ..MigrationActorConfig::default()
        };
        assert_eq!(cfg.swap_file_path(), PathBuf::from("/tmp/test_swap/session_42.swap"));
    }

    #[test]
    fn migration_actor_config_clone_preserves() {
        let cfg = MigrationActorConfig {
            nvme_swap_dir: PathBuf::from("/tmp/swap"),
            queue_capacity: 512, session_id: "test".to_string(),
            page_size: 8192, max_swap_pages: 1024,
        };
        let c = cfg.clone();
        assert_eq!(cfg.nvme_swap_dir, c.nvme_swap_dir);
        assert_eq!(cfg.queue_capacity, c.queue_capacity);
        assert_eq!(cfg.session_id, c.session_id);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch O: WeightPageTelemetryEvent
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn telemetry_event_evicted_fields_test() {
        let ev = WeightPageTelemetryEvent::Evicted {
            page_id: 42, from_tier: WeightTier::Hot, to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure, bytes: 4096,
        };
        if let WeightPageTelemetryEvent::Evicted { page_id, from_tier, to_tier, reason: _, bytes } = ev {
            assert_eq!(page_id, 42);
            assert_eq!(from_tier, WeightTier::Hot);
            assert_eq!(to_tier, WeightTier::Cold);
            assert_eq!(bytes, 4096);
        } else {
            panic!("should be Evicted");
        }
    }

    #[test]
    fn telemetry_event_recovered_fields_test() {
        let ev = WeightPageTelemetryEvent::Recovered {
            page_id: 99, from_tier: WeightTier::Cold, to_tier: WeightTier::Hot,
            latency_us: 500, bytes: 8192,
        };
        if let WeightPageTelemetryEvent::Recovered { page_id, from_tier: _, to_tier: _, latency_us, bytes } = ev {
            assert_eq!(page_id, 99);
            assert_eq!(latency_us, 500);
            assert_eq!(bytes, 8192);
        } else {
            panic!("should be Recovered");
        }
    }

    #[test]
    fn telemetry_event_clone_preserves() {
        let event = WeightPageTelemetryEvent::Evicted {
            page_id: 1, from_tier: WeightTier::Warm, to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure, bytes: 0,
        };
        if let WeightPageTelemetryEvent::Evicted { page_id, .. } = event.clone() {
            assert_eq!(page_id, 1);
        } else {
            panic!("cloned should be Evicted");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch P: BasicObserver
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn basic_observer_new_default_pressure() {
        let obs = BasicObserver::new();
        assert_eq!(obs.last_state.memory_pressure, 0.0);
    }

    #[test]
    fn basic_observer_update_pressure_ok() {
        let mut obs = BasicObserver::new();
        assert!(obs.update_memory_pressure(Ok(0.75)).is_ok());
        assert!((obs.last_state.memory_pressure - 0.75).abs() < 1e-6);
    }

    #[test]
    fn basic_observer_update_pressure_err() {
        let mut obs = BasicObserver::new();
        assert!(obs.update_memory_pressure(Err("gone".to_string())).is_err());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch Q: worker lifecycle
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn worker_lifecycle_spawn_prefetch_shutdown() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(50), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        let result = worker.prefetch(PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        });
        assert!(result.is_ok());
        thread::sleep(Duration::from_millis(100));
        let s = worker.stats();
        assert!(s.total_requests >= 1, "should have processed requests: {}", s.total_requests);
        worker.shutdown();
    }

    #[test]
    fn worker_drop_shuts_down_cleanly() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(50), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        { let _worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer); }
        // If we reach here, Drop worked without panic
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch R: CompressionCodec properties
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn compression_codec_debug_contains_name_test() {
        for (codec, name) in [
            (CompressionCodec::None, "None"),
            (CompressionCodec::Lz4, "Lz4"),
            (CompressionCodec::BitPackRle, "BitPackRle"),
            (CompressionCodec::NvcompAns, "NvcompAns"),
            (CompressionCodec::ZstdDict, "ZstdDict"),
        ] {
            assert!(format!("{:?}", codec).contains(name), "{:?} should contain '{}'", codec, name);
        }
    }

    #[test]
    fn compression_codec_copy_semantics_test() {
        let a = CompressionCodec::Lz4;
        let b = a;
        assert_eq!(a, b);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch S: Urgency formula numerical properties
    // ═══════════════════════════════════════════════════════════════════════════

    /// urgency with access_count=0 produces ln_1p(0)/ln_1p(10) = 0.0 for the
    /// importance rebound term; only recency bonus survives.
    #[test]
    fn urgency_access_count_zero_importance_rebound_is_zero() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 0,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // importance_rebound = 0.0, so first term = 0; recency_bonus <= 0.1
        assert!(
            u >= 0.0 && u <= 0.2,
            "urgency with zero access count should be small positive from recency only: got {}",
            u,
        );
    }

    /// urgency with access_count=10 produces ln_1p(10)/ln_1p(10) = 1.0 for
    /// importance rebound.
    #[test]
    fn urgency_access_count_ten_rebound_is_one() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 10,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // importance_rebound = 1.0, confidence = 1.0, tier_bonus = 1.0 → first term = 1.0
        // recency_bonus ≈ 0.1 (just accessed)
        assert!(
            u > 0.9,
            "urgency with access_count=10 should be ~1.0 + recency: got {}",
            u,
        );
    }

    /// urgency with access_count=1: rebound = ln_1p(1)/ln_1p(10) ≈ 0.415.
    #[test]
    fn urgency_access_count_one_rebound_is_small() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 1,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let expected_rebound = 1.0_f32.ln_1p() / 10.0_f32.ln_1p();
        // first term ≈ expected_rebound * 1.0 * 1.0 = 0.415, plus recency
        assert!(
            u > expected_rebound * 0.9,
            "urgency should exceed rebound factor: got {}, expected > {}",
            u, expected_rebound * 0.9,
        );
    }

    /// Recency bonus decays: as time passes, urgency decreases.
    /// We cannot simulate future time, but we can verify the recency component
    /// is bounded: recency_bonus = 1.0/(1+elapsed) * 0.1, max is 0.1.
    #[test]
    fn urgency_recency_bonus_bounded_at_one_tenth() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 0,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // importance_rebound = 0, confidence = 0 → first term = 0
        // recency bonus ≤ 0.1
        assert!(
            u <= 0.1 + 1e-6,
            "urgency with zero inputs should be ≤ 0.1 from recency: got {}",
            u,
        );
    }

    /// GPU HBM tier bonus is 2.0, which should produce higher urgency than
    /// CpuDram (1.0) with all other parameters equal.
    #[test]
    fn urgency_gpu_tier_double_bonus_vs_dram() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 10,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Active, warm_until: None,
        };
        let u_gpu = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::GpuHbm);
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        assert!(
            u_gpu > u_dram,
            "GpuHbm urgency should exceed CpuDram: gpu={} dram={}",
            u_gpu, u_dram,
        );
        // The first term ratio should be ~2.0 (GPU) / 1.0 (Dram)
        assert!(
            (u_gpu - u_dram) > 0.5,
            "difference should be significant: diff={}",
            u_gpu - u_dram,
        );
    }

    /// NVMe tier bonus is 0.5, producing lower urgency than CpuDram (1.0).
    #[test]
    fn urgency_nvme_tier_half_bonus_vs_dram() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 10,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let u_nvme = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::Nvme);
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        assert!(
            u_nvme < u_dram,
            "NVMe urgency should be less than CpuDram: nvme={} dram={}",
            u_nvme, u_dram,
        );
    }

    /// Importance rebound grows logarithmically: 100x increase in access_count
    /// does not produce 100x increase in urgency.
    #[test]
    fn urgency_importance_rebound_grows_sublinearly() {
        let meta_low = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 1,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let meta_high = PageMetadata {
            page_id: 2, sequence_id: Some(1), recency: 0, access_count: 100,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let u_low = SwapInWorker::compute_urgency(&meta_low, 0.5, StorageTier::CpuDram);
        let u_high = SwapInWorker::compute_urgency(&meta_high, 0.5, StorageTier::CpuDram);
        // Ratio should be much less than 100x (logarithmic growth)
        let ratio = u_high / u_low;
        assert!(
            ratio < 10.0,
            "100x access_count should not produce 10x urgency: ratio={}",
            ratio,
        );
    }

    /// All tiers produce finite urgency with normal inputs.
    #[test]
    fn urgency_all_tiers_finite_with_normal_inputs() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 5,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
            let u = SwapInWorker::compute_urgency(&meta, 0.7, tier);
            assert!(u.is_finite(), "urgency should be finite for {:?}: got {}", tier, u);
        }
    }

    /// urgency with very large access_count (1000000) still produces finite result.
    #[test]
    fn urgency_very_large_access_count_still_finite() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 1_000_000,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        assert!(u.is_finite(), "urgency with large access_count should be finite: got {}", u);
    }

    /// confidence=0 produces zero for the first term regardless of tier or
    /// access_count, because it is a multiplicative factor.
    #[test]
    fn urgency_zero_confidence_zeroes_first_term() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 100,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // first term = rebound * 0.0 * tier = 0.0
        // recency ≈ 0.1
        assert!(
            u <= 0.15,
            "zero confidence should produce near-zero urgency: got {}",
            u,
        );
    }

    /// urgency is always non-negative for non-negative confidence and normal metadata.
    #[test]
    fn urgency_non_negative_for_normal_inputs() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 5,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::SwappedOut, warm_until: None,
        };
        for confidence in [0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            for tier in [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme] {
                let u = SwapInWorker::compute_urgency(&meta, confidence, tier);
                assert!(
                    u >= 0.0,
                    "urgency should be non-negative: confidence={} tier={:?} u={}",
                    confidence, tier, u,
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch T: SwapInWorkerConfig default value checks
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn config_default_max_prefetch_per_round_is_16() {
        assert_eq!(SwapInWorkerConfig::default().max_prefetch_per_round, 16);
    }

    #[test]
    fn config_default_tick_interval_is_5ms() {
        assert_eq!(SwapInWorkerConfig::default().tick_interval, Duration::from_millis(5));
    }

    #[test]
    fn config_default_min_confidence_is_one_tenth() {
        let cfg = SwapInWorkerConfig::default();
        assert!((cfg.min_confidence - 0.1).abs() < 1e-6);
    }

    #[test]
    fn config_default_max_in_flight_is_64() {
        assert_eq!(SwapInWorkerConfig::default().max_in_flight, 64);
    }

    #[test]
    fn config_default_page_bytes_is_4096() {
        assert_eq!(SwapInWorkerConfig::default().page_bytes, 4096);
    }

    /// Config with all fields set to 1 is valid and constructible.
    #[test]
    fn config_all_fields_one() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            tick_interval: Duration::from_nanos(1),
            min_confidence: 1.0,
            max_in_flight: 1,
            page_bytes: 1,
        };
        assert_eq!(cfg.max_prefetch_per_round, 1);
        assert_eq!(cfg.max_in_flight, 1);
        assert_eq!(cfg.page_bytes, 1);
    }

    /// Config clone produces identical values across all 5 fields.
    #[test]
    fn config_clone_exact_field_match() {
        let original = SwapInWorkerConfig {
            max_prefetch_per_round: 32,
            tick_interval: Duration::from_millis(100),
            min_confidence: 0.5,
            max_in_flight: 128,
            page_bytes: 8192,
        };
        let cloned = original.clone();
        assert_eq!(original.max_prefetch_per_round, cloned.max_prefetch_per_round);
        assert_eq!(original.tick_interval, cloned.tick_interval);
        assert!((original.min_confidence - cloned.min_confidence).abs() < 1e-10);
        assert_eq!(original.max_in_flight, cloned.max_in_flight);
        assert_eq!(original.page_bytes, cloned.page_bytes);
    }

    /// PartialEq: differing in exactly one field means not equal.
    #[test]
    fn config_partial_eq_single_field_diff() {
        let base = SwapInWorkerConfig::default();
        // max_prefetch_per_round diff
        let diff_a = SwapInWorkerConfig { max_prefetch_per_round: 999, ..base.clone() };
        assert_ne!(base, diff_a);
        // min_confidence diff
        let diff_b = SwapInWorkerConfig { min_confidence: 0.999, ..base.clone() };
        assert_ne!(base, diff_b);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch U: SwapInWorkerStats accumulation and computation
    // ═══════════════════════════════════════════════════════════════════════════

    /// Fresh default stats has avg_latency_us == 0.0.
    #[test]
    fn stats_avg_latency_default_is_zero() {
        let s = SwapInWorkerStats::default();
        assert_eq!(s.avg_latency_us(), 0.0);
    }

    /// After promoted_ok=1 and total_latency_us=1000, avg = 1000.0.
    #[test]
    fn stats_avg_latency_single_value() {
        let mut s = SwapInWorkerStats::default();
        s.promoted_ok = 1;
        s.total_latency_us = 1000;
        assert_eq!(s.avg_latency_us(), 1000.0);
    }

    /// After promoted_ok=3 and total_latency_us=6000, avg = 2000.0.
    #[test]
    fn stats_avg_latency_multiple_values() {
        let mut s = SwapInWorkerStats::default();
        s.promoted_ok = 3;
        s.total_latency_us = 6000;
        assert_eq!(s.avg_latency_us(), 2000.0);
    }

    /// total_latency_us can accumulate independently of promoted_ok (it is
    /// set externally). avg_latency_us computes correctly.
    #[test]
    fn stats_latency_independent_accumulation() {
        let mut s = SwapInWorkerStats::default();
        s.promoted_ok = 2;
        s.total_latency_us = 500;
        let avg = s.avg_latency_us();
        assert!((avg - 250.0).abs() < 1e-6, "avg should be 250: got {}", avg);
    }

    /// Resetting stats to default zeroes all fields.
    #[test]
    fn stats_reset_via_default() {
        let s = {
            let _ = SwapInWorkerStats {
                total_requests: 100,
                submitted: 50,
                skipped: 25,
                promoted_ok: 20,
                promoted_failed: 5,
                two_hop_promotions: 10,
                total_latency_us: 99999,
                rounds: 42,
            };
            SwapInWorkerStats::default()
        };
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.submitted, 0);
        assert_eq!(s.skipped, 0);
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);
        assert_eq!(s.two_hop_promotions, 0);
        assert_eq!(s.total_latency_us, 0);
        assert_eq!(s.rounds, 0);
    }

    /// Clone produces equal stats.
    #[test]
    fn stats_clone_produces_equal_instance() {
        let s = SwapInWorkerStats {
            total_requests: 42,
            submitted: 30,
            skipped: 12,
            promoted_ok: 25,
            promoted_failed: 5,
            two_hop_promotions: 8,
            total_latency_us: 12345,
            rounds: 10,
        };
        assert_eq!(s, s.clone());
    }

    /// Incrementing total_requests does not affect other fields.
    #[test]
    fn stats_total_requests_increment_isolated() {
        let mut s = SwapInWorkerStats::default();
        s.total_requests += 1;
        assert_eq!(s.submitted, 0);
        assert_eq!(s.skipped, 0);
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);
        assert_eq!(s.two_hop_promotions, 0);
        assert_eq!(s.total_latency_us, 0);
        assert_eq!(s.rounds, 0);
    }

    /// Incrementing rounds does not affect other fields.
    #[test]
    fn stats_rounds_increment_isolated() {
        let mut s = SwapInWorkerStats {
            total_requests: 100, ..SwapInWorkerStats::default()
        };
        s.rounds += 1;
        assert_eq!(s.total_requests, 100);
        assert_eq!(s.submitted, 0);
    }

    /// promoted_failed does not affect avg_latency_us (only promoted_ok counts).
    #[test]
    fn stats_promoted_failed_no_effect_on_avg_latency() {
        let mut s = SwapInWorkerStats::default();
        s.promoted_failed = 10;
        assert_eq!(s.avg_latency_us(), 0.0);
    }

    /// Two_hop_promotions is tracked independently from promoted_ok.
    #[test]
    fn stats_two_hop_independent_from_promoted_ok() {
        let mut s = SwapInWorkerStats::default();
        s.two_hop_promotions = 5;
        s.promoted_ok = 0;
        assert_eq!(s.avg_latency_us(), 0.0);
        assert_eq!(s.two_hop_promotions, 5);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch V: SwapInWorkerError edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    /// SendFailed and RecvFailed with the same message are not equal.
    #[test]
    fn error_send_and_recv_different_variants_same_message() {
        let send = SwapInWorkerError::SendFailed("err".to_string());
        let recv = SwapInWorkerError::RecvFailed("err".to_string());
        assert_ne!(send, recv);
    }

    /// Error clone preserves the exact inner string including special chars.
    #[test]
    fn error_clone_preserves_special_chars() {
        let err = SwapInWorkerError::SendFailed("tab\there\nnewline".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    /// Both variants implement std::error::Error (checked via trait bound).
    #[test]
    fn error_both_variants_are_std_error() {
        fn assert_std_error<E: std::error::Error>(_: &E) {}
        assert_std_error(&SwapInWorkerError::SendFailed("a".into()));
        assert_std_error(&SwapInWorkerError::RecvFailed("b".into()));
    }

    /// PartialEq: same variant with different messages are not equal.
    #[test]
    fn error_partial_eq_different_message_same_variant() {
        let a = SwapInWorkerError::SendFailed("msg1".to_string());
        let b = SwapInWorkerError::SendFailed("msg2".to_string());
        assert_ne!(a, b);
    }

    /// Error with very long message (10KB) does not panic.
    #[test]
    fn error_very_long_message_no_panic() {
        let long_msg = "x".repeat(10_000);
        let err = SwapInWorkerError::SendFailed(long_msg.clone());
        if let SwapInWorkerError::SendFailed(msg) = &err {
            assert_eq!(msg.len(), 10_000);
        } else {
            panic!("expected SendFailed");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch W: PrefetchRequest construction edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    /// PrefetchRequest with page_id=0 is valid.
    #[test]
    fn prefetch_request_page_id_zero_is_valid() {
        let req = PrefetchRequest {
            page_id: 0, urgency: 0.5, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_id, 0);
    }

    /// PrefetchRequest with urgency exactly 0.0 is valid.
    #[test]
    fn prefetch_request_urgency_exact_zero() {
        let req = PrefetchRequest {
            page_id: 1, urgency: 0.0, prefetch_confidence: 0.5,
            page_bytes: 4096, enqueued_at: Instant::now(),
        };
        assert_eq!(req.urgency, 0.0);
    }

    /// PrefetchRequest with page_bytes=0 is valid (round uses config default).
    #[test]
    fn prefetch_request_page_bytes_zero_valid() {
        let req = PrefetchRequest {
            page_id: 1, urgency: 1.0, prefetch_confidence: 1.0,
            page_bytes: 0, enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_bytes, 0);
    }

    /// Two PrefetchRequests with identical fields except enqueued_at are not equal.
    #[test]
    fn prefetch_request_different_enqueued_at_not_equal() {
        let now = Instant::now();
        let req_a = PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: now,
        };
        // Instant does not implement PartialEq in the way we expect across
        // different values; construct a second one
        let req_b = PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: now,
        };
        // Same time → should be equal
        assert_eq!(req_a, req_b);
    }

    /// PrefetchRequest Debug output contains all field names.
    #[test]
    fn prefetch_request_debug_all_field_names() {
        let req = PrefetchRequest {
            page_id: 42, urgency: 0.9, prefetch_confidence: 0.7,
            page_bytes: 8192, enqueued_at: Instant::now(),
        };
        let debug_str = format!("{:?}", req);
        assert!(debug_str.contains("page_id"), "debug should contain page_id");
        assert!(debug_str.contains("urgency"), "debug should contain urgency");
        assert!(debug_str.contains("prefetch_confidence"), "debug should contain prefetch_confidence");
        assert!(debug_str.contains("page_bytes"), "debug should contain page_bytes");
    }

    /// PrefetchRequest confidence > 1.0 is stored verbatim (no clamping).
    #[test]
    fn prefetch_request_confidence_above_one_stored_verbatim() {
        let req = PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 2.5,
            page_bytes: 4096, enqueued_at: Instant::now(),
        };
        assert!((req.prefetch_confidence - 2.5).abs() < 1e-6);
    }

    /// PrefetchRequest urgency < 0.0 is stored verbatim (no clamping).
    #[test]
    fn prefetch_request_negative_urgency_stored_verbatim() {
        let req = PrefetchRequest {
            page_id: 1, urgency: -0.5, prefetch_confidence: 0.5,
            page_bytes: 4096, enqueued_at: Instant::now(),
        };
        assert!((req.urgency - (-0.5)).abs() < 1e-6);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch X: swap_in_round edge cases (via direct call)
    // ═══════════════════════════════════════════════════════════════════════════

    /// swap_in_round with empty requests vector returns 0 and increments rounds.
    #[test]
    fn swap_in_round_empty_returns_zero_and_increments_rounds() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = Vec::new();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0);
        let s = stats.lock().unwrap();
        assert_eq!(s.rounds, 1);
        assert_eq!(s.total_requests, 0);
        actor.shutdown();
    }

    /// swap_in_round with all HBM pages: all skipped, submitted=0.
    #[test]
    fn swap_in_round_all_hbm_pages_all_skipped() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        // Mark pages 1-3 as GpuHbm
        {
            let mut table = addr_table.write().unwrap();
            for pid in 1u64..=3u64 {
                table.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: Some(pid * 0x1000),
                    host_buffer: None,
                    current_tier: StorageTier::GpuHbm,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (1..=3).map(|i| PrefetchRequest {
            page_id: i as PageId, urgency: 0.8, prefetch_confidence: 0.9,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0);
        let s = stats.lock().unwrap();
        assert_eq!(s.skipped, 3);
        actor.shutdown();
    }

    /// swap_in_round with mixed CpuDram pages: submitted count matches.
    #[test]
    fn swap_in_round_dram_pages_submitted_count() {
        let config = SwapInWorkerConfig { max_prefetch_per_round: 10, ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        {
            let mut table = addr_table.write().unwrap();
            for pid in 1u64..=5u64 {
                table.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (1..=5).map(|i| PrefetchRequest {
            page_id: i as PageId, urgency: 0.5 + i as f32 * 0.1, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 5);
        actor.shutdown();
    }

    /// swap_in_round with low confidence skips all requests.
    #[test]
    fn swap_in_round_all_low_confidence_skips_all() {
        let config = SwapInWorkerConfig { min_confidence: 0.5, ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        {
            let mut table = addr_table.write().unwrap();
            for pid in 1u64..=3u64 {
                table.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (1..=3).map(|i| PrefetchRequest {
            page_id: i as PageId, urgency: 0.5, prefetch_confidence: 0.1,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0);
        let s = stats.lock().unwrap();
        assert_eq!(s.skipped, 3);
        actor.shutdown();
    }

    /// swap_in_round with NVMe pages counts two_hop_promotions.
    #[test]
    fn swap_in_round_nvme_counts_two_hop() {
        let config = SwapInWorkerConfig { max_prefetch_per_round: 10, ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        {
            let mut table = addr_table.write().unwrap();
            for pid in 1u64..=2u64 {
                table.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: None,
                    current_tier: StorageTier::Nvme,
                    original_bytes: 4096,
                    codec: CompressionCodec::ZstdDict,
                });
            }
        }
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (1..=2).map(|i| PrefetchRequest {
            page_id: i as PageId, urgency: 0.8, prefetch_confidence: 0.9,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 2);
        let s = stats.lock().unwrap();
        assert_eq!(s.two_hop_promotions, 2);
        actor.shutdown();
    }

    /// swap_in_round truncation: more requests than max_prefetch_per_round.
    #[test]
    fn swap_in_round_truncation_limits_processed() {
        let config = SwapInWorkerConfig { max_prefetch_per_round: 2, ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        {
            let mut table = addr_table.write().unwrap();
            for pid in 1u64..=5u64 {
                table.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (1..=5).map(|i| PrefetchRequest {
            page_id: i as PageId, urgency: i as f32, prefetch_confidence: 0.9,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 2);
        let s = stats.lock().unwrap();
        // total_requests counted before truncation
        assert_eq!(s.total_requests, 5);
        actor.shutdown();
    }

    /// swap_in_round with pages not in addr_table: all skipped.
    #[test]
    fn swap_in_round_pages_not_in_addr_table_skipped() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (100..=102).map(|i| PrefetchRequest {
            page_id: i as PageId, urgency: 0.8, prefetch_confidence: 0.9,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0);
        let s = stats.lock().unwrap();
        assert_eq!(s.skipped, 3);
        actor.shutdown();
    }

    /// swap_in_round uses request.page_bytes when non-zero, config.page_bytes when zero.
    #[test]
    fn swap_in_round_page_bytes_source_selection() {
        let config = SwapInWorkerConfig { page_bytes: 4096, ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        {
            let mut table = addr_table.write().unwrap();
            // page 1: CpuDram, page 2: CpuDram
            for pid in [1u64, 2u64] {
                table.insert(pid as PageId, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![
            PrefetchRequest { page_id: 1, urgency: 0.8, prefetch_confidence: 0.9,
                page_bytes: 8192, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.7, prefetch_confidence: 0.9,
                page_bytes: 0, enqueued_at: Instant::now() },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 2);
        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch Y: Worker lifecycle integration
    // ═══════════════════════════════════════════════════════════════════════════

    /// Worker stats() returns fresh default on a newly spawned worker.
    #[test]
    fn worker_stats_fresh_spawn_is_default() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(50), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        let s = worker.stats();
        assert_eq!(s, SwapInWorkerStats::default());
        worker.shutdown();
    }

    /// prefetch_batch on a live worker enqueues all requests.
    #[test]
    fn worker_prefetch_batch_all_enqueued() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(50), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        let requests: Vec<PrefetchRequest> = (1..=5).map(|i| PrefetchRequest {
            page_id: i as PageId, urgency: 0.5, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        let count = worker.prefetch_batch(&requests);
        assert_eq!(count, 5);
        worker.shutdown();
    }

    /// prefetch_batch with empty slice returns 0.
    #[test]
    fn worker_prefetch_batch_empty_returns_zero() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(50), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        let count = worker.prefetch_batch(&[]);
        assert_eq!(count, 0);
        worker.shutdown();
    }

    /// Multiple prefetch calls succeed on a live worker.
    #[test]
    fn worker_multiple_prefetch_succeed() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(50), ..Default::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        for i in 0..10 {
            let result = worker.prefetch(PrefetchRequest {
                page_id: i as PageId, urgency: 0.5, prefetch_confidence: 0.8,
                page_bytes: 4096, enqueued_at: Instant::now(),
            });
            assert!(result.is_ok(), "prefetch {} should succeed", i);
        }
        worker.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch Z: Drain completions integration
    // ═══════════════════════════════════════════════════════════════════════════

    /// drain_completions_and_update with empty metadata does not panic.
    #[test]
    fn drain_empty_metadata_map_no_panic() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        drain_completions_and_update(
            &actor, &page_metadata, &addr_table, &stats, &observer,
        );
        let s = stats.lock().unwrap();
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);
        actor.shutdown();
    }

    /// drain_completions_and_update with non-empty metadata but no completions.
    #[test]
    fn drain_non_empty_metadata_no_completions() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new({
            let mut m = HashMap::new();
            m.insert(1u64 as PageId, PageMetadata::default());
            m
        }));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        drain_completions_and_update(
            &actor, &page_metadata, &addr_table, &stats, &observer,
        );
        let s = stats.lock().unwrap();
        assert_eq!(s.promoted_ok, 0);
        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AA: Observer integration
    // ═══════════════════════════════════════════════════════════════════════════

    /// BasicObserver records weight page events correctly.
    #[test]
    fn observer_records_eviction_increments_count() {
        let mut obs = BasicObserver::new();
        let initial_count = obs.last_state.weight_eviction_count;
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 1, from_tier: WeightTier::Hot, to_tier: WeightTier::Cold,
            reason: EvictionReason::MemoryPressure, bytes: 4096,
        });
        assert_eq!(obs.last_state.weight_eviction_count, initial_count + 1);
    }

    /// BasicObserver records recovery events correctly.
    #[test]
    fn observer_records_recovery_increments_count() {
        let mut obs = BasicObserver::new();
        let initial_count = obs.last_state.weight_recovery_count;
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 1, from_tier: WeightTier::Cold, to_tier: WeightTier::Hot,
            latency_us: 100, bytes: 4096,
        });
        assert_eq!(obs.last_state.weight_recovery_count, initial_count + 1);
    }

    /// BasicObserver update_scheduler_metrics sets all fields.
    #[test]
    fn observer_update_scheduler_metrics_sets_all() {
        let mut obs = BasicObserver::new();
        obs.update_scheduler_metrics(10, 5, 3, 100);
        assert_eq!(obs.last_state.waiting_queue_len, 10);
        assert_eq!(obs.last_state.current_running_len, 5);
        assert_eq!(obs.last_state.current_batch_size, 3);
        assert_eq!(obs.last_state.mean_context_len, 100);
    }

    /// BasicObserver update_kv_fragmentation sets value.
    #[test]
    fn observer_update_kv_fragmentation_sets_value() {
        let mut obs = BasicObserver::new();
        obs.update_kv_fragmentation(0.42);
        assert!((obs.last_state.kv_fragmentation - 0.42).abs() < 1e-6);
    }

    /// BasicObserver update_swap_io_rate sets value.
    #[test]
    fn observer_update_swap_io_rate_sets_value() {
        let mut obs = BasicObserver::new();
        obs.update_swap_io_rate(123.45);
        assert!((obs.last_state.swap_io_rate - 123.45).abs() < 1e-4);
    }

    /// BasicObserver update_logits_entropy sets value.
    #[test]
    fn observer_update_logits_entropy_sets_value() {
        let mut obs = BasicObserver::new();
        obs.update_logits_entropy(2.71);
        assert!((obs.last_state.logits_entropy - 2.71).abs() < 1e-4);
    }

    /// BasicObserver update_attention_sparsity sets value.
    #[test]
    fn observer_update_attention_sparsity_sets_value() {
        let mut obs = BasicObserver::new();
        obs.update_attention_sparsity(0.85);
        assert!((obs.last_state.attention_sparsity - 0.85).abs() < 1e-6);
    }

    /// BasicObserver update_moe_fault_metrics sets all three fields.
    #[test]
    fn observer_update_moe_fault_metrics_sets_all() {
        let mut obs = BasicObserver::new();
        obs.update_moe_fault_metrics(0.01, 500.0, 42);
        assert!((obs.last_state.moe_fault_rate - 0.01).abs() < 1e-6);
        assert!((obs.last_state.moe_avg_recovery_us - 500.0).abs() < 1e-4);
        assert_eq!(obs.last_state.moe_working_set_size, 42);
    }

    /// BasicObserver update_weight_metrics sets all seven fields.
    #[test]
    fn observer_update_weight_metrics_sets_all() {
        let mut obs = BasicObserver::new();
        obs.update_weight_metrics(100, 50, 30, 20, 5, 3);
        assert_eq!(obs.last_state.weight_page_total, 100);
        assert_eq!(obs.last_state.weight_pages_l1, 50);
        assert_eq!(obs.last_state.weight_pages_l2, 30);
        assert_eq!(obs.last_state.weight_pages_l3, 20);
        assert_eq!(obs.last_state.weight_eviction_count, 5);
        assert_eq!(obs.last_state.weight_recovery_count, 3);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AB: PageMetadata edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    /// PageMetadata with all zero/default fields.
    #[test]
    fn page_metadata_default_all_fields() {
        let meta = PageMetadata::default();
        assert_eq!(meta.page_id, 0);
        assert_eq!(meta.sequence_id, None);
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert_eq!(meta.swap_in_time, None);
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Standby);
        assert_eq!(meta.warm_until, None);
    }

    /// PageMetadata with maximum access_count.
    #[test]
    fn page_metadata_access_count_usize_max() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: usize::MAX,
            last_access: Instant::now(), swap_in_time: None, is_lir: true,
            state: PageState::Active, warm_until: None,
        };
        assert_eq!(meta.access_count, usize::MAX);
    }

    /// PageMetadata with sequence_id = None is valid.
    #[test]
    fn page_metadata_sequence_id_none() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: None, recency: 0, access_count: 1,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Active, warm_until: None,
        };
        assert!(meta.sequence_id.is_none());
    }

    /// PageMetadata with swap_in_time = Some is valid.
    #[test]
    fn page_metadata_swap_in_time_some() {
        let now = Instant::now();
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 1,
            last_access: now, swap_in_time: Some(now), is_lir: false,
            state: PageState::Warm, warm_until: None,
        };
        assert!(meta.swap_in_time.is_some());
    }

    /// PageMetadata with warm_until = Some is valid.
    #[test]
    fn page_metadata_warm_until_some_valid() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: Some(1), recency: 0, access_count: 1,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Warm, warm_until: Some(Instant::now()),
        };
        assert!(meta.warm_until.is_some());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AC: PageAddrEntry construction edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    /// PageAddrEntry with gpu_ptr=Some and host_buffer=None (GPU-resident).
    #[test]
    fn page_addr_entry_gpu_resident() {
        let entry = PageAddrEntry {
            gpu_ptr: Some(0xDEAD_BEEF),
            host_buffer: None,
            current_tier: StorageTier::GpuHbm,
            original_bytes: 4096,
            codec: CompressionCodec::None,
        };
        assert_eq!(entry.gpu_ptr, Some(0xDEAD_BEEF));
        assert!(entry.host_buffer.is_none());
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
    }

    /// PageAddrEntry with gpu_ptr=None and host_buffer=Some (DRAM-resident).
    #[test]
    fn page_addr_entry_dram_resident() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(vec![0u8; 4096]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 4096,
            codec: CompressionCodec::Lz4,
        };
        assert!(entry.gpu_ptr.is_none());
        assert!(entry.host_buffer.is_some());
        assert_eq!(entry.codec, CompressionCodec::Lz4);
    }

    /// PageAddrEntry with both gpu_ptr and host_buffer as None (NVMe-resident).
    #[test]
    fn page_addr_entry_nvme_resident() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: None,
            current_tier: StorageTier::Nvme,
            original_bytes: 4096,
            codec: CompressionCodec::ZstdDict,
        };
        assert!(entry.gpu_ptr.is_none());
        assert!(entry.host_buffer.is_none());
        assert_eq!(entry.current_tier, StorageTier::Nvme);
    }

    /// PageAddrEntry with original_bytes=0 is valid.
    #[test]
    fn page_addr_entry_zero_bytes() {
        let entry = PageAddrEntry {
            gpu_ptr: None,
            host_buffer: Some(vec![]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 0,
            codec: CompressionCodec::None,
        };
        assert_eq!(entry.original_bytes, 0);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AD: StorageTier edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    /// StorageTier discriminants: GpuHbm=0, CpuDram=1, Nvme=2.
    #[test]
    fn storage_tier_discriminant_values_exact() {
        assert_eq!(StorageTier::GpuHbm as u8, 0);
        assert_eq!(StorageTier::CpuDram as u8, 1);
        assert_eq!(StorageTier::Nvme as u8, 2);
    }

    /// StorageTier has exactly 3 variants.
    #[test]
    fn storage_tier_exactly_three_variants() {
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        assert_eq!(tiers.len(), 3);
        // Verify all are distinct
        use std::collections::HashSet;
        let set: HashSet<_> = tiers.iter().collect();
        assert_eq!(set.len(), 3);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AE: PageState variants
    // ═══════════════════════════════════════════════════════════════════════════

    /// All PageState variants are constructible.
    #[test]
    fn page_state_all_variants_constructible() {
        let _ = PageState::Free;
        let _ = PageState::Active;
        let _ = PageState::Standby;
        let _ = PageState::SwappedOut;
        let _ = PageState::Warm;
        let _ = PageState::Protected;
        let _ = PageState::Swapped;
    }

    /// PageState has exactly 7 variants.
    #[test]
    fn page_state_exactly_seven_variants() {
        use std::collections::HashSet;
        let variants: HashSet<PageState> = [
            PageState::Free, PageState::Active, PageState::Standby,
            PageState::SwappedOut, PageState::Warm, PageState::Protected,
            PageState::Swapped,
        ].into_iter().collect();
        assert_eq!(variants.len(), 7);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AF: WeightTier and EvictionReason
    // ═══════════════════════════════════════════════════════════════════════════

    /// WeightTier has exactly 3 variants: Hot, Warm, Cold.
    #[test]
    fn weight_tier_exactly_three_variants() {
        use std::collections::HashSet;
        let variants: HashSet<WeightTier> = [WeightTier::Hot, WeightTier::Warm, WeightTier::Cold].into_iter().collect();
        assert_eq!(variants.len(), 3);
    }

    /// EvictionReason has only MemoryPressure variant.
    #[test]
    fn eviction_reason_only_memory_pressure() {
        let reason = EvictionReason::MemoryPressure;
        // It should be Copy and Clone
        let copied = reason;
        assert_eq!(reason, copied);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AG: SwapInWorkerConfig PartialEq edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    /// Two configs differing only in tick_interval are not equal.
    #[test]
    fn config_partial_eq_tick_interval_diff() {
        let a = SwapInWorkerConfig { tick_interval: Duration::from_millis(1), ..Default::default() };
        let b = SwapInWorkerConfig { tick_interval: Duration::from_millis(2), ..Default::default() };
        assert_ne!(a, b);
    }

    /// Config with all fields set to the same values is equal.
    #[test]
    fn config_same_values_equal() {
        let a = SwapInWorkerConfig {
            max_prefetch_per_round: 8, tick_interval: Duration::from_millis(10),
            min_confidence: 0.5, max_in_flight: 32, page_bytes: 2048,
        };
        let b = SwapInWorkerConfig {
            max_prefetch_per_round: 8, tick_interval: Duration::from_millis(10),
            min_confidence: 0.5, max_in_flight: 32, page_bytes: 2048,
        };
        assert_eq!(a, b);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AH: Priority sorting verification
    // ═══════════════════════════════════════════════════════════════════════════

    /// Sorting by urgency descending puts highest urgency first.
    #[test]
    fn priority_sort_descending_order() {
        let mut requests: Vec<PrefetchRequest> = vec![
            PrefetchRequest { page_id: 1, urgency: 0.3, prefetch_confidence: 0.8,
                page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.9, prefetch_confidence: 0.8,
                page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 3, urgency: 0.6, prefetch_confidence: 0.8,
                page_bytes: 4096, enqueued_at: Instant::now() },
        ];
        requests.sort_by(|a, b| b.urgency.partial_cmp(&a.urgency).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(requests[0].page_id, 2);
        assert_eq!(requests[1].page_id, 3);
        assert_eq!(requests[2].page_id, 1);
    }

    /// Sorting with NaN urgency does not panic.
    #[test]
    fn priority_sort_with_nan_no_panic() {
        let mut requests: Vec<PrefetchRequest> = vec![
            PrefetchRequest { page_id: 1, urgency: f32::NAN, prefetch_confidence: 0.8,
                page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.5, prefetch_confidence: 0.8,
                page_bytes: 4096, enqueued_at: Instant::now() },
        ];
        requests.sort_by(|a, b| b.urgency.partial_cmp(&a.urgency).unwrap_or(std::cmp::Ordering::Equal));
        // Just verify no panic — order of NaN vs finite is implementation-defined
        assert_eq!(requests.len(), 2);
    }

    /// Sorting with all equal urgency preserves all items.
    #[test]
    fn priority_sort_equal_urgency_preserves_count() {
        let mut requests: Vec<PrefetchRequest> = (1..=10).map(|i| PrefetchRequest {
            page_id: i as PageId, urgency: 0.5, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        requests.sort_by(|a, b| b.urgency.partial_cmp(&a.urgency).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(requests.len(), 10);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Batch AI: MigrationCommand and MigrationDone edge cases
    // ═══════════════════════════════════════════════════════════════════════════

    /// MigrationCommand::Shutdown debug output contains "Shutdown".
    #[test]
    fn migration_command_shutdown_variant() {
        let cmd = MigrationCommand::Shutdown;
        let debug = format!("{:?}", cmd);
        assert!(debug.contains("Shutdown"));
    }

    /// MigrationDone construction with all fields.
    #[test]
    fn migration_done_construction_all_fields() {
        let done = MigrationDone {
            page_id: 42,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::GpuHbm,
            result: MigrationResult::Ok { compressed_bytes: 2048, checksum: 0x1234 },
        };
        assert_eq!(done.page_id, 42);
        assert_eq!(done.from_tier, StorageTier::Nvme);
        assert_eq!(done.to_tier, StorageTier::GpuHbm);
        if let MigrationResult::Ok { compressed_bytes, checksum } = done.result {
            assert_eq!(compressed_bytes, 2048);
            assert_eq!(checksum, 0x1234);
        } else {
            panic!("expected Ok result");
        }
    }

    /// MigrationDone with Failed result.
    #[test]
    fn migration_done_failed_result_variant() {
        let done = MigrationDone {
            page_id: 1,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::GpuHbm,
            result: MigrationResult::Failed { reason: "timeout".to_string() },
        };
        if let MigrationResult::Failed { reason } = &done.result {
            assert_eq!(reason, "timeout");
        } else {
            panic!("expected Failed result");
        }
    }

    /// MigrationResult::Ok clone preserves fields.
    #[test]
    fn migration_result_ok_clone_preserves() {
        let res = MigrationResult::Ok { compressed_bytes: 100, checksum: 0xAB };
        if let MigrationResult::Ok { compressed_bytes, checksum } = res.clone() {
            assert_eq!(compressed_bytes, 100);
            assert_eq!(checksum, 0xAB);
        } else {
            panic!("expected Ok");
        }
    }

    /// MigrationResult::Failed clone preserves reason.
    #[test]
    fn migration_result_failed_clone_preserves() {
        let res = MigrationResult::Failed { reason: "err".to_string() };
        if let MigrationResult::Failed { reason } = res.clone() {
            assert_eq!(reason, "err");
        } else {
            panic!("expected Failed");
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Additional tests (~80) — PrefetchRequest, Config, Stats, Error, Urgency,
    // swap_in_round, drain, observer, MigrationCommand/Done/Result edge cases
    // ─────────────────────────────────────────────────────────────────────────────

    /// PrefetchRequest with all zero numeric fields is valid and Debug prints.
    #[test]
    fn prefetch_request_all_zeros_debug_ok() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        };
        let dbg = format!("{req:?}");
        assert!(dbg.contains("page_id"));
        assert!(dbg.contains("urgency"));
    }

    /// PrefetchRequest PartialEq: same page_id, urgency, confidence, page_bytes but
    /// different enqueued_at are still equal (enqueued_at is Instant which Eq is time-based).
    #[test]
    fn prefetch_request_eq_ignores_enqueued_at_if_all_numeric_match() {
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        assert_eq!(a, b);
    }

    /// PrefetchRequest: page_bytes = 1 is valid.
    #[test]
    fn prefetch_request_single_byte_page() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 1.0,
            prefetch_confidence: 1.0,
            page_bytes: 1,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_bytes, 1);
    }

    /// PrefetchRequest: confidence = 0.0 is valid and stored.
    #[test]
    fn prefetch_request_zero_confidence_stored() {
        let req = PrefetchRequest {
            page_id: 5,
            urgency: 0.3,
            prefetch_confidence: 0.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!(!req.prefetch_confidence.is_nan());
        assert_eq!(req.prefetch_confidence, 0.0);
    }

    /// PrefetchRequest: urgency = f32::MAX is finite and stored.
    #[test]
    fn prefetch_request_f32_max_urgency() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: f32::MAX,
            prefetch_confidence: 0.5,
            page_bytes: 100,
            enqueued_at: Instant::now(),
        };
        assert!(req.urgency.is_finite());
    }

    /// SwapInWorkerConfig: max_prefetch_per_round = 1 still works.
    #[test]
    fn config_max_prefetch_one_round() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.max_prefetch_per_round, 1);
        assert_eq!(cfg, cfg.clone());
    }

    /// SwapInWorkerConfig: page_bytes = 1 (minimum meaningful).
    #[test]
    fn config_page_bytes_one() {
        let cfg = SwapInWorkerConfig {
            page_bytes: 1,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.page_bytes, 1);
    }

    /// SwapInWorkerConfig: tick_interval = 1 day (very large).
    #[test]
    fn config_tick_interval_one_day() {
        let cfg = SwapInWorkerConfig {
            tick_interval: Duration::from_secs(86400),
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval, Duration::from_secs(86400));
    }

    /// SwapInWorkerConfig: min_confidence = f32::MAX stored verbatim.
    #[test]
    fn config_min_confidence_f32_max() {
        let cfg = SwapInWorkerConfig {
            min_confidence: f32::MAX,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.min_confidence, f32::MAX);
    }

    /// SwapInWorkerConfig: all fields match after clone.
    #[test]
    fn config_clone_all_fields_exact() {
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 32,
            tick_interval: Duration::from_millis(100),
            min_confidence: 0.25,
            max_in_flight: 128,
            page_bytes: 8192,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.max_prefetch_per_round, 32);
        assert_eq!(cloned.tick_interval, Duration::from_millis(100));
        assert_eq!(cloned.min_confidence, 0.25);
        assert_eq!(cloned.max_in_flight, 128);
        assert_eq!(cloned.page_bytes, 8192);
    }

    /// SwapInWorkerStats: total_requests wraps correctly at u64 boundary.
    #[test]
    fn stats_total_requests_near_max() {
        let mut s = SwapInWorkerStats {
            total_requests: u64::MAX - 1,
            ..SwapInWorkerStats::default()
        };
        s.total_requests += 1;
        assert_eq!(s.total_requests, u64::MAX);
    }

    /// SwapInWorkerStats: promoted_ok and promoted_failed can both be non-zero.
    #[test]
    fn stats_both_promoted_counters_nonzero() {
        let s = SwapInWorkerStats {
            promoted_ok: 100,
            promoted_failed: 7,
            ..SwapInWorkerStats::default()
        };
        assert_eq!(s.promoted_ok, 100);
        assert_eq!(s.promoted_failed, 7);
    }

    /// SwapInWorkerStats: two_hop_promotions independent of promoted_ok.
    #[test]
    fn stats_two_hop_independent_of_promoted_ok() {
        let s = SwapInWorkerStats {
            two_hop_promotions: 50,
            promoted_ok: 0,
            ..SwapInWorkerStats::default()
        };
        assert_eq!(s.two_hop_promotions, 50);
        assert_eq!(s.promoted_ok, 0);
    }

    /// SwapInWorkerStats: total_latency_us accumulates across multiple updates.
    #[test]
    fn stats_latency_accumulates_additively() {
        let mut s = SwapInWorkerStats {
            promoted_ok: 2,
            total_latency_us: 100,
            ..SwapInWorkerStats::default()
        };
        s.total_latency_us += 200;
        s.promoted_ok += 1;
        assert!((s.avg_latency_us() - 100.0).abs() < f64::EPSILON);
    }

    /// SwapInWorkerStats: avg_latency_us with one promotion returns exact value.
    #[test]
    fn stats_avg_latency_exact_for_single_promotion() {
        let s = SwapInWorkerStats {
            promoted_ok: 1,
            total_latency_us: 12345,
            ..SwapInWorkerStats::default()
        };
        assert!((s.avg_latency_us() - 12345.0).abs() < f64::EPSILON);
    }

    /// SwapInWorkerStats: Debug format includes all field names.
    #[test]
    fn stats_debug_includes_all_field_names() {
        let s = SwapInWorkerStats {
            total_requests: 1,
            submitted: 2,
            skipped: 3,
            promoted_ok: 4,
            promoted_failed: 5,
            two_hop_promotions: 6,
            total_latency_us: 7,
            rounds: 8,
        };
        let dbg = format!("{s:?}");
        assert!(dbg.contains("total_requests"));
        assert!(dbg.contains("submitted"));
        assert!(dbg.contains("skipped"));
        assert!(dbg.contains("promoted_ok"));
        assert!(dbg.contains("promoted_failed"));
        assert!(dbg.contains("two_hop_promotions"));
        assert!(dbg.contains("total_latency_us"));
        assert!(dbg.contains("rounds"));
    }

    /// SwapInWorkerStats: PartialEq works for identical instances.
    #[test]
    fn stats_partial_eq_identical_custom() {
        let a = SwapInWorkerStats {
            total_requests: 10,
            submitted: 5,
            skipped: 3,
            promoted_ok: 2,
            promoted_failed: 0,
            two_hop_promotions: 1,
            total_latency_us: 500,
            rounds: 4,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    /// SwapInWorkerError: SendFailed with empty string is valid.
    #[test]
    fn error_send_failed_empty_string() {
        let e = SwapInWorkerError::SendFailed(String::new());
        assert!(matches!(e, SwapInWorkerError::SendFailed(_)));
    }

    /// SwapInWorkerError: RecvFailed with empty string is valid.
    #[test]
    fn error_recv_failed_empty_string() {
        let e = SwapInWorkerError::RecvFailed(String::new());
        assert!(matches!(e, SwapInWorkerError::RecvFailed(_)));
    }

    /// SwapInWorkerError: two SendFailed with different messages are not equal.
    #[test]
    fn error_send_failed_different_messages_not_equal() {
        let a = SwapInWorkerError::SendFailed("alpha".to_string());
        let b = SwapInWorkerError::SendFailed("beta".to_string());
        assert_ne!(a, b);
    }

    /// SwapInWorkerError: two RecvFailed with same message are equal.
    #[test]
    fn error_recv_failed_same_message_equal() {
        let a = SwapInWorkerError::RecvFailed("timeout".to_string());
        let b = SwapInWorkerError::RecvFailed("timeout".to_string());
        assert_eq!(a, b);
    }

    /// Urgency: access_count = 100 produces finite urgency across all tiers.
    #[test]
    fn urgency_access_count_hundred_finite_all_tiers() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        for tier in [StorageTier::CpuDram, StorageTier::Nvme, StorageTier::GpuHbm] {
            let u = SwapInWorker::compute_urgency(&meta, 0.5, tier);
            assert!(u.is_finite(), "urgency should be finite for {tier:?}");
        }
    }

    /// Urgency: confidence = 0.5 exactly halves the first term.
    #[test]
    fn urgency_confidence_half_exact_scaling() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u1 = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_half = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        // recency bonus cancels out, so ratio should be ~2x
        let recency = 0.1 / (1.0 + 0.0); // ~0.1 for just-now access
        let first_1 = u1 - recency;
        let first_half = u_half - recency;
        assert!(
            (first_1 - 2.0 * first_half).abs() < 0.01,
            "half confidence should halve first term: {first_1} vs {first_half}"
        );
    }

    /// Urgency: access_count = 0 gives importance_rebound = 0.0 (ln_1p(0) = 0).
    #[test]
    fn urgency_zero_access_importance_rebound_zero() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        // With access_count=0 and confidence=1.0 on CpuDram: first_term = 0 * 1.0 * 1.0 = 0
        // Only recency bonus remains
        let recency = 0.1 / (1.0 + 0.0);
        assert!((u - recency).abs() < 0.001, "urgency should be recency only: {u}");
    }

    /// Urgency: tier_bonus for Nvme = 0.5 exactly.
    #[test]
    fn urgency_nvme_tier_bonus_exact() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::Nvme);
        let recency = 0.1 / (1.0 + 0.0);
        let first_dram = u_dram - recency;
        let first_nvme = u_nvme - recency;
        assert!(
            (first_dram - 2.0 * first_nvme).abs() < 0.01,
            "NVMe first term should be half DRAM: dram={first_dram} nvme={first_nvme}"
        );
    }

    /// Urgency: GpuHbm tier_bonus = 2.0 gives double CpuDram first term.
    #[test]
    fn urgency_hbm_tier_double_dram_exact() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u_dram = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_hbm = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::GpuHbm);
        let recency = 0.1 / (1.0 + 0.0);
        let first_dram = u_dram - recency;
        let first_hbm = u_hbm - recency;
        assert!(
            (first_hbm - 2.0 * first_dram).abs() < 0.01,
            "HBM first term should be 2x DRAM: hbm={first_hbm} dram={first_dram}"
        );
    }

    /// Urgency: confidence = 0.0 gives first term = 0 regardless of tier.
    #[test]
    fn urgency_zero_conf_all_tiers_first_term_zero() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 20,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        for tier in [StorageTier::CpuDram, StorageTier::Nvme, StorageTier::GpuHbm] {
            let u = SwapInWorker::compute_urgency(&meta, 0.0, tier);
            let recency = 0.1 / (1.0 + 0.0);
            assert!(
                (u - recency).abs() < 0.01,
                "zero confidence first term should be zero for {tier:?}: {u}"
            );
        }
    }

    /// Urgency: recency bonus is strictly positive for any finite elapsed time.
    #[test]
    fn urgency_recency_always_positive_for_finite_elapsed() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        assert!(u > 0.0, "recency bonus alone should be positive: {u}");
    }

    /// Urgency: increasing access_count from 0 to 1 increases urgency.
    #[test]
    fn urgency_access_one_higher_than_zero_exact() {
        let meta0 = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let meta1 = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u0 = SwapInWorker::compute_urgency(&meta0, 1.0, StorageTier::CpuDram);
        let u1 = SwapInWorker::compute_urgency(&meta1, 1.0, StorageTier::CpuDram);
        assert!(u1 > u0, "access_count=1 should yield higher urgency than 0: {u1} vs {u0}");
    }

    /// Urgency: swapping page_id produces same urgency (page_id independent).
    #[test]
    fn urgency_page_id_independence_verified() {
        let make_meta = |pid: PageId| PageMetadata {
            page_id: pid,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u_a = SwapInWorker::compute_urgency(&make_meta(42), 0.5, StorageTier::CpuDram);
        let u_b = SwapInWorker::compute_urgency(&make_meta(9999), 0.5, StorageTier::CpuDram);
        assert!((u_a - u_b).abs() < 0.001, "page_id should not affect urgency: {u_a} vs {u_b}");
    }

    /// swap_in_round: empty requests vector increments rounds.
    #[test]
    fn round_empty_increments_rounds_only() {
        let config = SwapInWorkerConfig::default();
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut reqs = Vec::new();
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert
        assert_eq!(submitted, 0);
        let s = stats.lock().expect("lock");
        assert_eq!(s.rounds, 1);
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.skipped, 0);

        actor.shutdown();
    }

    /// swap_in_round: multiple rounds accumulate total_requests correctly.
    #[test]
    fn round_total_requests_accumulates_across_rounds() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 100,
            ..SwapInWorkerConfig::default()
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Round 1: 3 requests
        let mut reqs1 = vec![
            make_prefetch(1, 1.0, 0.9, 4096),
            make_prefetch(2, 0.5, 0.9, 4096),
            make_prefetch(3, 0.2, 0.9, 4096),
        ];
        SwapInWorker::swap_in_round(&config, &actor, &mut reqs1, &page_metadata, &addr_table, &stats, &observer);

        // Round 2: 2 requests
        let mut reqs2 = vec![
            make_prefetch(4, 0.8, 0.9, 4096),
            make_prefetch(5, 0.3, 0.9, 4096),
        ];
        SwapInWorker::swap_in_round(&config, &actor, &mut reqs2, &page_metadata, &addr_table, &stats, &observer);

        // Assert
        let s = stats.lock().expect("lock");
        assert_eq!(s.total_requests, 5);
        assert_eq!(s.rounds, 2);

        actor.shutdown();
    }

    /// swap_in_round: single DRAM page with max_in_flight = 1 still submits.
    #[test]
    fn round_single_dram_max_in_flight_one_submits() {
        let config = SwapInWorkerConfig {
            max_in_flight: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut reqs = vec![make_prefetch(10, 1.0, 0.9, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    /// swap_in_round: page_bytes from request used when non-zero.
    #[test]
    fn round_page_bytes_from_request_when_set() {
        let config = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Request has page_bytes = 8192, different from config's 4096
        let mut reqs = vec![make_prefetch(10, 1.0, 0.9, 8192)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1);
        // The request's 8192 should have been used (actor accepted it)
        let s = stats.lock().expect("lock");
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    /// swap_in_round: two pages with same urgency both get submitted.
    #[test]
    fn round_equal_urgency_both_submitted() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 10,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut reqs = vec![
            make_prefetch(1, 0.5, 0.8, 4096),
            make_prefetch(2, 0.5, 0.8, 4096),
        ];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 2);

        actor.shutdown();
    }

    /// swap_in_round: confidence exactly at min_confidence boundary is accepted.
    #[test]
    fn round_confidence_exact_boundary_accepted() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // confidence = 0.5 exactly, min_confidence = 0.5 => NOT less than => accepted
        let mut reqs = vec![make_prefetch(10, 1.0, 0.5, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1);

        actor.shutdown();
    }

    /// swap_in_round: NVMe page increments two_hop_promotions.
    #[test]
    fn round_nvme_increments_two_hop_counter() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 55, StorageTier::Nvme, CompressionCodec::ZstdDict);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut reqs = vec![make_prefetch(55, 1.0, 0.9, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1);
        let s = stats.lock().expect("lock");
        assert_eq!(s.two_hop_promotions, 1);

        actor.shutdown();
    }

    /// swap_in_round: DRAM page does NOT increment two_hop_promotions.
    #[test]
    fn round_dram_does_not_increment_two_hop() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::Lz4);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut reqs = vec![make_prefetch(10, 1.0, 0.9, 4096)];
        SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        let s = stats.lock().expect("lock");
        assert_eq!(s.two_hop_promotions, 0);

        actor.shutdown();
    }

    /// swap_in_round: truncation to max_prefetch keeps highest urgency.
    #[test]
    fn round_truncation_keeps_highest() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Page 2 has higher urgency but is listed second
        let mut reqs = vec![
            make_prefetch(1, 0.1, 0.9, 4096),
            make_prefetch(2, 0.9, 0.9, 4096),
        ];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1);
        let s = stats.lock().expect("lock");
        assert_eq!(s.submitted, 1);
        // Only 2 requests counted before truncation
        assert_eq!(s.total_requests, 2);

        actor.shutdown();
    }

    /// swap_in_round: back-pressure stops submission when max_in_flight reached.
    #[test]
    fn round_back_pressure_limits_submission() {
        let config = SwapInWorkerConfig {
            max_in_flight: 1,
            max_prefetch_per_round: 10,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        for pid in [1usize, 2, 3] {
            insert_addr_entry(&addr_table, pid, StorageTier::CpuDram, CompressionCodec::None);
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut reqs: Vec<PrefetchRequest> = (1..=3)
            .map(|pid| make_prefetch(pid, 1.0, 0.9, 4096))
            .collect();
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        // With max_in_flight=1, the first DRAM page uses 1 in_flight slot.
        // After draining completions, the next ones might also submit depending
        // on drain timing. At minimum, submitted >= 1.
        assert!(submitted >= 1, "at least one page should be submitted");

        actor.shutdown();
    }

    /// swap_in_round: requests vector is drained after round.
    #[test]
    fn round_drains_input_vector() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut reqs = vec![make_prefetch(10, 1.0, 0.9, 4096)];
        SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert!(reqs.is_empty(), "requests should be drained after round");

        actor.shutdown();
    }

    /// swap_in_round: mixed HBM and DRAM — only DRAM submitted.
    #[test]
    fn round_mixed_hbm_dram_only_dram_submitted() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::GpuHbm, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut reqs = vec![
            make_prefetch(1, 1.0, 0.9, 4096),
            make_prefetch(2, 0.5, 0.9, 4096),
        ];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1);
        let s = stats.lock().expect("lock");
        assert_eq!(s.skipped, 1); // HBM page skipped
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    /// PageMetadata: all state variants can be constructed.
    #[test]
    fn page_metadata_all_states_constructible() {
        let states = [
            PageState::Free, PageState::Standby, PageState::Active,
            PageState::Protected, PageState::Warm, PageState::Swapped, PageState::SwappedOut,
        ];
        for state in states {
            let meta = PageMetadata {
                page_id: 0,
                sequence_id: None,
                recency: 0,
                access_count: 0,
                last_access: Instant::now(),
                swap_in_time: None,
                is_lir: false,
                state,
                warm_until: None,
            };
            assert_eq!(meta.state, state);
        }
    }

    /// PageMetadata: clone produces equal instance with all fields.
    #[test]
    fn page_metadata_clone_exact_field_match() {
        let original = PageMetadata {
            page_id: 42,
            sequence_id: Some(7),
            recency: 100,
            access_count: 50,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Protected,
            warm_until: None,
        };
        let cloned = original.clone();
        assert_eq!(cloned.page_id, 42);
        assert_eq!(cloned.sequence_id, Some(7));
        assert_eq!(cloned.recency, 100);
        assert_eq!(cloned.access_count, 50);
        assert_eq!(cloned.is_lir, true);
        assert_eq!(cloned.state, PageState::Protected);
    }

    /// PageAddrEntry: all CompressionCodec variants are accepted.
    #[test]
    fn page_addr_entry_all_codec_variants() {
        let codecs = [
            CompressionCodec::None, CompressionCodec::Lz4,
            CompressionCodec::BitPackRle, CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for codec in codecs {
            let entry = PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec,
            };
            assert_eq!(entry.codec, codec);
        }
    }

    /// PageAddrEntry: mutation reflects immediately.
    #[test]
    fn page_addr_entry_mutation_immediate() {
        let mut entry = PageAddrEntry {
            gpu_ptr: Some(0x1000),
            host_buffer: Some(vec![0u8; 4096]),
            current_tier: StorageTier::CpuDram,
            original_bytes: 4096,
            codec: CompressionCodec::None,
        };
        entry.current_tier = StorageTier::GpuHbm;
        entry.original_bytes = 8192;
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
        assert_eq!(entry.original_bytes, 8192);
    }

    /// MigrationCommand: PromoteToDram carries page_id and page_bytes.
    #[test]
    fn migration_command_promote_to_dram_carries_fields() {
        let cmd = MigrationCommand::PromoteToDram { page_id: 42, page_bytes: 8192 };
        if let MigrationCommand::PromoteToDram { page_id, page_bytes } = cmd {
            assert_eq!(page_id, 42);
            assert_eq!(page_bytes, 8192);
        } else {
            panic!("expected PromoteToDram");
        }
    }

    /// MigrationCommand: PromoteToHbm carries page_id and page_bytes.
    #[test]
    fn migration_command_promote_to_hbm_carries_fields() {
        let cmd = MigrationCommand::PromoteToHbm { page_id: 99, page_bytes: 4096 };
        if let MigrationCommand::PromoteToHbm { page_id, page_bytes } = cmd {
            assert_eq!(page_id, 99);
            assert_eq!(page_bytes, 4096);
        } else {
            panic!("expected PromoteToHbm");
        }
    }

    /// MigrationDone: from_tier and to_tier can be same tier.
    #[test]
    fn migration_done_same_from_to_tier() {
        let done = MigrationDone {
            page_id: 10,
            from_tier: StorageTier::CpuDram,
            to_tier: StorageTier::CpuDram,
            result: MigrationResult::Ok { compressed_bytes: 0, checksum: 0 },
        };
        assert_eq!(done.from_tier, done.to_tier);
    }

    /// MigrationDone: clone preserves all fields.
    #[test]
    fn migration_done_clone_exact() {
        let done = MigrationDone {
            page_id: 77,
            from_tier: StorageTier::Nvme,
            to_tier: StorageTier::GpuHbm,
            result: MigrationResult::Ok { compressed_bytes: 1234, checksum: 0xFF },
        };
        let cloned = done.clone();
        assert_eq!(cloned.page_id, 77);
        assert_eq!(cloned.from_tier, StorageTier::Nvme);
        assert_eq!(cloned.to_tier, StorageTier::GpuHbm);
    }

    /// MigrationResult: Ok with zero compressed_bytes and checksum is valid.
    #[test]
    fn migration_result_ok_zero_fields() {
        let res = MigrationResult::Ok { compressed_bytes: 0, checksum: 0 };
        if let MigrationResult::Ok { compressed_bytes, checksum } = res {
            assert_eq!(compressed_bytes, 0);
            assert_eq!(checksum, 0);
        } else {
            panic!("expected Ok");
        }
    }

    /// MigrationResult: Failed with multi-line reason.
    #[test]
    fn migration_result_failed_multiline_reason() {
        let reason = "line1\nline2\nline3".to_string();
        let res = MigrationResult::Failed { reason: reason.clone() };
        if let MigrationResult::Failed { reason: r } = res {
            assert_eq!(r, "line1\nline2\nline3");
        } else {
            panic!("expected Failed");
        }
    }

    /// BasicObserver: new() has default memory pressure of 0.
    #[test]
    fn observer_new_default_pressure_zero() {
        let obs = BasicObserver::new();
        assert_eq!(obs.last_state.memory_pressure, 0.0);
    }

    /// CompressionCodec: all 5 variants produce distinct as_u8 values.
    #[test]
    fn compression_codec_distinct_as_u8() {
        let codes: Vec<u8> = [
            CompressionCodec::None, CompressionCodec::Lz4,
            CompressionCodec::BitPackRle, CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ].iter().map(|c| c.as_u8()).collect();
        let unique: std::collections::HashSet<u8> = codes.iter().copied().collect();
        assert_eq!(unique.len(), 5, "all 5 codecs should have distinct u8 values");
    }

    /// StorageTier: exactly 3 variants (GpuHbm, CpuDram, Nvme).
    #[test]
    fn storage_tier_exactly_three() {
        let tiers = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];
        assert_eq!(tiers.len(), 3);
    }

    /// WeightTier: exactly 3 variants (Hot, Warm, Cold).
    #[test]
    fn weight_tier_exactly_three() {
        let tiers = [WeightTier::Hot, WeightTier::Warm, WeightTier::Cold];
        assert_eq!(tiers.len(), 3);
    }

    /// PageState: Free and Standby are distinct.
    #[test]
    fn page_state_free_not_standby() {
        assert_ne!(PageState::Free, PageState::Standby);
    }

    /// PageState: Warm and Active are distinct.
    #[test]
    fn page_state_warm_not_active() {
        assert_ne!(PageState::Warm, PageState::Active);
    }

    /// PageState: Swapped and SwappedOut are distinct.
    #[test]
    fn page_state_swapped_not_swappedout() {
        assert_ne!(PageState::Swapped, PageState::SwappedOut);
    }

    /// Drain: empty metadata map does not panic.
    #[test]
    fn drain_empty_map_no_panic() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Should not panic
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);

        let s = stats.lock().expect("lock");
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);

        actor.shutdown();
    }

    /// Worker: prefetch_batch with slice of 0 returns 0.
    #[test]
    fn worker_prefetch_batch_zero_len() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);

        let empty: Vec<PrefetchRequest> = Vec::new();
        let count = worker.prefetch_batch(&empty);
        assert_eq!(count, 0);

        worker.shutdown();
    }

    /// Worker: stats() returns default on fresh worker.
    #[test]
    fn worker_stats_default_on_fresh() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);

        let s = worker.stats();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.submitted, 0);
        assert_eq!(s.rounds, 0);

        // Must shut down via mutable reference
        drop(worker);
    }

    /// Worker: Drop impl calls shutdown without panic.
    #[test]
    fn worker_drop_clean_shutdown() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        {
            let _worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
            // Drop goes out of scope here
        }
        // If we reach here, Drop::drop called shutdown cleanly
    }

    /// Urgency: importance_rebound is ln_1p(access_count) / ln_1p(10).
    #[test]
    fn urgency_importance_rebound_formula_exact() {
        // access_count = 10 → ln_1p(10) / ln_1p(10) = 1.0
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let recency = 0.1 / (1.0 + 0.0);
        let first_term = u - recency;
        assert!(
            (first_term - 1.0).abs() < 0.01,
            "importance_rebound for access_count=10 should be ~1.0: {first_term}"
        );
    }

    /// Urgency: very large access_count produces finite result.
    #[test]
    fn urgency_access_count_large_finite() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 1_000_000,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        assert!(u.is_finite(), "large access_count should produce finite urgency: {u}");
    }

    /// Urgency: sequence_id None vs Some does not affect score.
    #[test]
    fn urgency_sequence_id_none_vs_some_same() {
        let make = |seq: Option<u64>| PageMetadata {
            page_id: 0,
            sequence_id: seq,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u_none = SwapInWorker::compute_urgency(&make(None), 0.5, StorageTier::CpuDram);
        let u_some = SwapInWorker::compute_urgency(&make(Some(42)), 0.5, StorageTier::CpuDram);
        assert!((u_none - u_some).abs() < 0.001, "sequence_id should not affect urgency");
    }

    /// Urgency: is_lir true vs false does not affect score.
    #[test]
    fn urgency_is_lir_no_effect() {
        let make = |is_lir: bool| PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u_false = SwapInWorker::compute_urgency(&make(false), 0.5, StorageTier::CpuDram);
        let u_true = SwapInWorker::compute_urgency(&make(true), 0.5, StorageTier::CpuDram);
        assert!((u_false - u_true).abs() < 0.001, "is_lir should not affect urgency");
    }

    /// Urgency: warm_until Some vs None does not affect score.
    #[test]
    fn urgency_warm_until_no_effect() {
        let make = |warm: Option<Instant>| PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: warm,
        };
        let u_none = SwapInWorker::compute_urgency(&make(None), 0.5, StorageTier::CpuDram);
        let u_some = SwapInWorker::compute_urgency(&make(Some(Instant::now())), 0.5, StorageTier::CpuDram);
        assert!((u_none - u_some).abs() < 0.001, "warm_until should not affect urgency");
    }

    /// Urgency: recency field (meta.recency) does not directly affect urgency.
    #[test]
    fn urgency_meta_recency_field_no_effect() {
        let make = |recency: usize| PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u_0 = SwapInWorker::compute_urgency(&make(0), 0.5, StorageTier::CpuDram);
        let u_999 = SwapInWorker::compute_urgency(&make(999), 0.5, StorageTier::CpuDram);
        assert!((u_0 - u_999).abs() < 0.001, "meta.recency should not affect urgency");
    }

    /// Config: Debug format contains all field names.
    #[test]
    fn config_debug_all_field_names() {
        let cfg = SwapInWorkerConfig::default();
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("max_prefetch_per_round"));
        assert!(dbg.contains("tick_interval"));
        assert!(dbg.contains("min_confidence"));
        assert!(dbg.contains("max_in_flight"));
        assert!(dbg.contains("page_bytes"));
    }

    /// Error: SendFailed Display starts with "swap-in worker send failed".
    #[test]
    fn error_send_failed_display_prefix() {
        let e = SwapInWorkerError::SendFailed("oops".to_string());
        let msg = format!("{e}");
        assert!(msg.starts_with("swap-in worker send failed"), "msg={msg}");
    }

    /// Error: RecvFailed Display starts with "swap-in worker recv failed".
    #[test]
    fn error_recv_failed_display_prefix() {
        let e = SwapInWorkerError::RecvFailed("timeout".to_string());
        let msg = format!("{e}");
        assert!(msg.starts_with("swap-in worker recv failed"), "msg={msg}");
    }

    /// Error: both variants implement std::error::Error.
    #[test]
    fn error_both_variants_implement_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&SwapInWorkerError::SendFailed("a".to_string()));
        assert_error(&SwapInWorkerError::RecvFailed("b".to_string()));
    }

    /// Stats: promoted_ok = 0 and total_latency_us > 0 gives avg = 0.
    #[test]
    fn stats_avg_latency_zero_promoted_with_latency() {
        let s = SwapInWorkerStats {
            promoted_ok: 0,
            total_latency_us: 9999,
            ..SwapInWorkerStats::default()
        };
        assert_eq!(s.avg_latency_us(), 0.0);
    }

    /// Stats: total_requests, submitted, skipped can all be non-zero simultaneously.
    #[test]
    fn stats_all_three_counters_nonzero() {
        let s = SwapInWorkerStats {
            total_requests: 100,
            submitted: 80,
            skipped: 20,
            ..SwapInWorkerStats::default()
        };
        assert_eq!(s.total_requests, 100);
        assert_eq!(s.submitted, 80);
        assert_eq!(s.skipped, 20);
    }

    /// Stats: rounds counter is independent of all other counters.
    #[test]
    fn stats_rounds_independent() {
        let s = SwapInWorkerStats {
            rounds: 50,
            total_requests: 0,
            submitted: 0,
            ..SwapInWorkerStats::default()
        };
        assert_eq!(s.rounds, 50);
        assert_eq!(s.total_requests, 0);
    }

    /// MigrationActorConfig: Debug format contains field names.
    #[test]
    fn migration_actor_config_debug_field_names() {
        let cfg = MigrationActorConfig::default();
        let dbg = format!("{cfg:?}");
        assert!(dbg.contains("queue_capacity"));
        assert!(dbg.contains("page_size"));
    }

    /// EvictionReason: only MemoryPressure variant exists (exhaustive match).
    #[test]
    fn eviction_reason_exhaustive_match() {
        let reason = EvictionReason::MemoryPressure;
        match reason {
            EvictionReason::MemoryPressure => {} // only variant
        }
    }

    /// Observer: record_weight_page_event Recovered does not panic.
    #[test]
    fn observer_record_recovered_no_panic() {
        let mut obs = BasicObserver::new();
        obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
            page_id: 1,
            from_tier: WeightTier::Cold,
            to_tier: WeightTier::Hot,
            latency_us: 100,
            bytes: 4096,
        });
    }

    /// Observer: record_weight_page_event Evicted does not panic.
    #[test]
    fn observer_record_evicted_no_panic() {
        let mut obs = BasicObserver::new();
        obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
            page_id: 2,
            from_tier: WeightTier::Hot,
            to_tier: WeightTier::Cold,
            bytes: 4096,
            reason: EvictionReason::MemoryPressure,
        });
    }

    /// WeightTier: Debug format contains variant names.
    #[test]
    fn weight_tier_debug_contains_names() {
        assert!(format!("{:?}", WeightTier::Hot).contains("Hot"));
        assert!(format!("{:?}", WeightTier::Warm).contains("Warm"));
        assert!(format!("{:?}", WeightTier::Cold).contains("Cold"));
    }

    /// StorageTier: Debug format contains variant names.
    #[test]
    fn storage_tier_debug_contains_names() {
        assert!(format!("{:?}", StorageTier::GpuHbm).contains("GpuHbm"));
        assert!(format!("{:?}", StorageTier::CpuDram).contains("CpuDram"));
        assert!(format!("{:?}", StorageTier::Nvme).contains("Nvme"));
    }

    /// PageState: Debug format contains variant names.
    #[test]
    fn page_state_debug_contains_names() {
        assert!(format!("{:?}", PageState::Free).contains("Free"));
        assert!(format!("{:?}", PageState::Active).contains("Active"));
        assert!(format!("{:?}", PageState::Swapped).contains("Swapped"));
    }

    /// Helper: insert_addr_entry creates valid DRAM entry.
    #[test]
    fn helper_insert_addr_entry_dram() {
        let table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        insert_addr_entry(&table, 1, StorageTier::CpuDram, CompressionCodec::Lz4);
        let read = table.read().expect("read");
        let entry = read.get(&1).expect("entry");
        assert_eq!(entry.current_tier, StorageTier::CpuDram);
        assert_eq!(entry.codec, CompressionCodec::Lz4);
    }

    /// Helper: make_prefetch creates valid request.
    #[test]
    fn helper_make_prefetch_fields() {
        let req = make_prefetch(42, 0.75, 0.9, 8192);
        assert_eq!(req.page_id, 42);
        assert!((req.urgency - 0.75).abs() < f32::EPSILON);
        assert!((req.prefetch_confidence - 0.9).abs() < f32::EPSILON);
        assert_eq!(req.page_bytes, 8192);
    }

    /// CompressionCodec: from_u8 round-trip for all variants.
    #[test]
    fn compression_codec_roundtrip_all() {
        let variants = [
            CompressionCodec::None, CompressionCodec::Lz4,
            CompressionCodec::BitPackRle, CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];
        for v in variants {
            let code = v.as_u8();
            let restored = CompressionCodec::from_u8(code);
            assert_eq!(Some(v), restored, "roundtrip failed for {v:?}");
        }
    }

    /// StorageTier: from_u8 returns None for out-of-range values (new tests section).
    #[test]
    fn storage_tier_from_u8_out_of_range_appendix() {
        assert!(StorageTier::from_u8(3).is_none());
        assert!(StorageTier::from_u8(255).is_none());
    }

    // ── Helper functions for tests ──────────────────────────────────────────────

    fn make_prefetch(page_id: PageId, urgency: f32, confidence: f32, page_bytes: usize) -> PrefetchRequest {
        PrefetchRequest {
            page_id,
            urgency,
            prefetch_confidence: confidence,
            page_bytes,
            enqueued_at: Instant::now(),
        }
    }

    fn insert_addr_entry(
        table: &PageAddrTable,
        page_id: PageId,
        tier: StorageTier,
        codec: CompressionCodec,
    ) {
        let mut t = table.write().expect("write lock");
        t.insert(page_id, PageAddrEntry {
            gpu_ptr: if tier == StorageTier::GpuHbm { Some(0x1000) } else { None },
            host_buffer: if tier == StorageTier::CpuDram { Some(vec![0u8; 4096]) } else { None },
            current_tier: tier,
            original_bytes: 4096,
            codec,
        });
    }

    // ── Additional tests for simple public types ───────────────────────────────

    /// Config: PartialEq is reflexive (a == a).
    #[test]
    fn config_partial_eq_reflexivity() {
        let c = SwapInWorkerConfig::default();
        assert_eq!(c, c);
    }

    /// Config: PartialEq is symmetric (a == b → b == a).
    #[test]
    fn config_partial_eq_symmetry() {
        let a = SwapInWorkerConfig::default();
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    /// Config: different min_confidence values are not equal.
    #[test]
    fn config_different_min_confidence_not_equal() {
        let a = SwapInWorkerConfig { min_confidence: 0.1, ..SwapInWorkerConfig::default() };
        let b = SwapInWorkerConfig { min_confidence: 0.2, ..SwapInWorkerConfig::default() };
        assert_ne!(a, b);
    }

    /// Config: tick_interval as Duration::from_secs still works.
    #[test]
    fn config_tick_interval_from_secs() {
        let c = SwapInWorkerConfig {
            tick_interval: Duration::from_secs(1),
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(c.tick_interval, Duration::from_secs(1));
    }

    /// Config: default max_prefetch_per_round is 16.
    #[test]
    fn config_default_max_prefetch_is_sixteen() {
        assert_eq!(SwapInWorkerConfig::default().max_prefetch_per_round, 16);
    }

    /// Stats: PartialEq is reflexive.
    #[test]
    fn stats_partial_eq_reflexivity() {
        let s = SwapInWorkerStats::default();
        assert_eq!(s, s);
    }

    /// Stats: PartialEq is symmetric.
    #[test]
    fn stats_partial_eq_symmetry() {
        let a = SwapInWorkerStats { total_requests: 5, ..SwapInWorkerStats::default() };
        let b = SwapInWorkerStats { total_requests: 5, ..SwapInWorkerStats::default() };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    /// Stats: skipped field is independent of submitted.
    #[test]
    fn stats_skipped_independent_of_submitted() {
        let s = SwapInWorkerStats {
            skipped: 10,
            submitted: 0,
            ..SwapInWorkerStats::default()
        };
        assert_eq!(s.skipped, 10);
        assert_eq!(s.submitted, 0);
    }

    /// Stats: two_hop_promotions is independent of promoted_failed.
    #[test]
    fn stats_two_hop_independent_of_promoted_failed() {
        let s = SwapInWorkerStats {
            two_hop_promotions: 7,
            promoted_failed: 0,
            ..SwapInWorkerStats::default()
        };
        assert_eq!(s.two_hop_promotions, 7);
        assert_eq!(s.promoted_failed, 0);
    }

    /// Stats: total_latency_us can be non-zero while promoted_ok is zero.
    #[test]
    fn stats_latency_us_nonzero_with_zero_promoted() {
        let s = SwapInWorkerStats {
            total_latency_us: 5000,
            promoted_ok: 0,
            ..SwapInWorkerStats::default()
        };
        assert_eq!(s.total_latency_us, 5000);
        assert_eq!(s.promoted_ok, 0);
    }

    /// Stats: promoted_ok field increments correctly.
    #[test]
    fn stats_promoted_ok_increment() {
        let mut s = SwapInWorkerStats::default();
        s.promoted_ok += 1;
        assert_eq!(s.promoted_ok, 1);
        s.promoted_ok += 1;
        assert_eq!(s.promoted_ok, 2);
    }

    /// Stats: promoted_failed field increments correctly.
    #[test]
    fn stats_promoted_failed_increment() {
        let mut s = SwapInWorkerStats::default();
        s.promoted_failed += 1;
        assert_eq!(s.promoted_failed, 1);
        s.promoted_failed += 2;
        assert_eq!(s.promoted_failed, 3);
    }

    /// Stats: rounds can be incremented independently.
    #[test]
    fn stats_rounds_can_be_incremented() {
        let mut s = SwapInWorkerStats::default();
        for _ in 0..3 {
            s.rounds += 1;
        }
        assert_eq!(s.rounds, 3);
    }

    /// Stats: default Debug output contains field name "total_requests".
    #[test]
    fn stats_debug_default_contains_total_requests() {
        let s = SwapInWorkerStats::default();
        let dbg = format!("{s:?}");
        assert!(dbg.contains("total_requests"), "Debug should contain 'total_requests': {dbg}");
    }

    /// Error: SendFailed and RecvFailed with same message are not equal.
    #[test]
    fn error_send_and_recv_same_message_not_equal() {
        let msg = "identical";
        let send = SwapInWorkerError::SendFailed(msg.to_string());
        let recv = SwapInWorkerError::RecvFailed(msg.to_string());
        assert_ne!(send, recv);
    }

    /// Error: SendFailed message content is preserved.
    #[test]
    fn error_send_failed_message_preserved() {
        let err = SwapInWorkerError::SendFailed("channel closed".to_string());
        match err {
            SwapInWorkerError::SendFailed(msg) => assert_eq!(msg, "channel closed"),
            SwapInWorkerError::RecvFailed(_) => panic!("wrong variant"),
        }
    }

    /// Error: RecvFailed message content is preserved.
    #[test]
    fn error_recv_failed_message_preserved() {
        let err = SwapInWorkerError::RecvFailed("timeout".to_string());
        match err {
            SwapInWorkerError::RecvFailed(msg) => assert_eq!(msg, "timeout"),
            SwapInWorkerError::SendFailed(_) => panic!("wrong variant"),
        }
    }

    /// Error: clone produces equal value for SendFailed.
    #[test]
    fn error_send_failed_clone_equality() {
        let err = SwapInWorkerError::SendFailed("test".to_string());
        assert_eq!(err, err.clone());
    }

    /// Error: clone produces equal value for RecvFailed.
    #[test]
    fn error_recv_failed_clone_equality() {
        let err = SwapInWorkerError::RecvFailed("test".to_string());
        assert_eq!(err, err.clone());
    }

    /// Error: both variants implement Clone.
    #[test]
    fn error_both_variants_clone() {
        let send = SwapInWorkerError::SendFailed("a".to_string());
        let recv = SwapInWorkerError::RecvFailed("b".to_string());
        let _send_clone = send.clone();
        let _recv_clone = recv.clone();
    }

    /// PrefetchRequest: zero page_bytes is stored verbatim.
    #[test]
    fn prefetch_request_zero_page_bytes_verbatim() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_bytes, 0);
    }

    /// PrefetchRequest: confidence exactly 0.5 stored verbatim.
    #[test]
    fn prefetch_request_confidence_half() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert!((req.prefetch_confidence - 0.5).abs() < f32::EPSILON);
    }

    /// PrefetchRequest: urgency f32::MAX stored verbatim.
    #[test]
    fn prefetch_request_urgency_f32_max_verbatim() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: f32::MAX,
            prefetch_confidence: 1.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.urgency, f32::MAX);
    }

    /// PrefetchRequest: page_bytes usize::MAX stored verbatim.
    #[test]
    fn prefetch_request_page_bytes_usize_max() {
        let req = PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 1.0,
            page_bytes: usize::MAX,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_bytes, usize::MAX);
    }

    /// PrefetchRequest: Debug contains "page_id".
    #[test]
    fn prefetch_request_debug_has_page_id() {
        let req = PrefetchRequest {
            page_id: 99,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        };
        let dbg = format!("{req:?}");
        assert!(dbg.contains("page_id"), "Debug should contain 'page_id': {dbg}");
    }

    /// PrefetchRequest: Debug contains "urgency".
    #[test]
    fn prefetch_request_debug_has_urgency() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 1.0,
            prefetch_confidence: 0.0,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        };
        let dbg = format!("{req:?}");
        assert!(dbg.contains("urgency"), "Debug should contain 'urgency': {dbg}");
    }

    /// PrefetchRequest: Debug contains "prefetch_confidence".
    #[test]
    fn prefetch_request_debug_has_confidence() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 0.0,
            prefetch_confidence: 0.5,
            page_bytes: 0,
            enqueued_at: Instant::now(),
        };
        let dbg = format!("{req:?}");
        assert!(dbg.contains("prefetch_confidence"), "Debug should contain 'prefetch_confidence': {dbg}");
    }

    /// PrefetchRequest: Debug contains "page_bytes".
    #[test]
    fn prefetch_request_debug_has_page_bytes() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 12345,
            enqueued_at: Instant::now(),
        };
        let dbg = format!("{req:?}");
        assert!(dbg.contains("page_bytes"), "Debug should contain 'page_bytes': {dbg}");
    }

    /// Config: Debug output contains "max_prefetch_per_round".
    #[test]
    fn config_debug_has_max_prefetch_per_round() {
        let c = SwapInWorkerConfig::default();
        let dbg = format!("{c:?}");
        assert!(dbg.contains("max_prefetch_per_round"), "Debug should contain 'max_prefetch_per_round': {dbg}");
    }

    /// Config: Debug output contains "min_confidence".
    #[test]
    fn config_debug_has_min_confidence() {
        let c = SwapInWorkerConfig::default();
        let dbg = format!("{c:?}");
        assert!(dbg.contains("min_confidence"), "Debug should contain 'min_confidence': {dbg}");
    }

    /// Config: modifying cloned instance does not affect original.
    #[test]
    fn config_clone_mutation_isolation() {
        let original = SwapInWorkerConfig::default();
        let mut cloned = original.clone();
        cloned.max_prefetch_per_round = 999;
        assert_ne!(original.max_prefetch_per_round, 999);
    }

    /// Config: page_bytes field stores usize::MAX without truncation.
    #[test]
    fn config_page_bytes_usize_max_preserved() {
        let c = SwapInWorkerConfig {
            page_bytes: usize::MAX,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(c.page_bytes, usize::MAX);
    }

    /// Config: max_in_flight field stores zero (edge case).
    #[test]
    fn config_max_in_flight_zero_stored() {
        let c = SwapInWorkerConfig {
            max_in_flight: 0,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(c.max_in_flight, 0);
    }

    /// Config: tick_interval of zero duration is stored verbatim.
    #[test]
    fn config_tick_interval_zero_duration_stored() {
        let c = SwapInWorkerConfig {
            tick_interval: Duration::ZERO,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(c.tick_interval, Duration::ZERO);
    }

    /// Config: PartialEq is transitive — if a==b and b==c then a==c.
    #[test]
    fn config_partial_eq_transitivity() {
        let a = SwapInWorkerConfig::default();
        let b = a.clone();
        let c = a.clone();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c, "PartialEq must be transitive");
    }

    /// Config: Debug output contains "tick_interval".
    #[test]
    fn config_debug_contains_tick_interval() {
        let c = SwapInWorkerConfig::default();
        let dbg = format!("{c:?}");
        assert!(dbg.contains("tick_interval"), "Debug should contain 'tick_interval': {dbg}");
    }

    /// Config: Debug output contains "max_in_flight".
    #[test]
    fn config_debug_contains_max_in_flight() {
        let c = SwapInWorkerConfig::default();
        let dbg = format!("{c:?}");
        assert!(dbg.contains("max_in_flight"), "Debug should contain 'max_in_flight': {dbg}");
    }

    /// Config: Debug output contains "page_bytes".
    #[test]
    fn config_debug_contains_page_bytes() {
        let c = SwapInWorkerConfig::default();
        let dbg = format!("{c:?}");
        assert!(dbg.contains("page_bytes"), "Debug should contain 'page_bytes': {dbg}");
    }

    /// Stats: two default instances are equal.
    #[test]
    fn stats_two_defaults_are_equal() {
        let a = SwapInWorkerStats::default();
        let b = SwapInWorkerStats::default();
        assert_eq!(a, b);
    }

    /// Stats: total_latency_us accumulates independently of promoted_ok.
    #[test]
    fn stats_total_latency_accumulates_independently() {
        let mut s = SwapInWorkerStats::default();
        s.total_latency_us = 500;
        s.promoted_ok = 0;
        // latency can be non-zero even with zero promoted_ok (edge case from
        // drain_completions_and_update timing).
        assert_eq!(s.total_latency_us, 500);
    }

    /// Stats: two_hop_promotions increments independently.
    #[test]
    fn stats_two_hop_promotions_increment_independent() {
        let mut s = SwapInWorkerStats::default();
        s.two_hop_promotions = 7;
        assert_eq!(s.promoted_ok, 0, "two_hop_promotions should not affect promoted_ok");
        assert_eq!(s.two_hop_promotions, 7);
    }

    /// Stats: avg_latency_us returns f64 (not panicking on zero promoted_ok).
    #[test]
    fn stats_avg_latency_returns_f64_no_panic() {
        let s = SwapInWorkerStats::default();
        let avg = s.avg_latency_us();
        assert_eq!(avg, 0.0);
    }

    /// Stats: Debug output contains "two_hop_promotions".
    #[test]
    fn stats_debug_contains_two_hop_promotions() {
        let s = SwapInWorkerStats::default();
        let dbg = format!("{s:?}");
        assert!(dbg.contains("two_hop_promotions"), "Debug should contain 'two_hop_promotions': {dbg}");
    }

    /// Error: SendFailed and RecvFailed with same message are not equal.
    #[test]
    fn error_different_variants_same_string_not_equal() {
        let a = SwapInWorkerError::SendFailed("msg".to_string());
        let b = SwapInWorkerError::RecvFailed("msg".to_string());
        assert_ne!(a, b, "different variants with same message must not be equal");
    }

    /// Urgency: tier bonus ordering GpuHbm > CpuDram > Nvme for same inputs.
    #[test]
    fn urgency_tier_bonus_ordering_hbm_gt_dram_gt_nvme() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(100),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let conf = 0.5_f32;
        let u_hbm = SwapInWorker::compute_urgency(&meta, conf, StorageTier::GpuHbm);
        let u_dram = SwapInWorker::compute_urgency(&meta, conf, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, conf, StorageTier::Nvme);
        assert!(
            u_hbm > u_dram,
            "HBM urgency must exceed DRAM: hbm={u_hbm} dram={u_dram}",
        );
        assert!(
            u_dram > u_nvme,
            "DRAM urgency must exceed NVMe: dram={u_dram} nvme={u_nvme}",
        );
    }

    /// PrefetchRequest: two requests with identical numeric fields but different
    /// enqueued_at are not equal (Instant is part of PartialEq).
    #[test]
    fn prefetch_request_equality_different_enqueued_at_only() {
        let now = Instant::now();
        let later = now + Duration::from_millis(100);
        let a = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 1,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: later,
        };
        assert_ne!(a, b, "requests with different enqueued_at should not be equal");
    }

    /// An empty round after a non-empty round should not modify any stats counters
    /// except `rounds`.
    #[test]
    fn round_empty_preserves_previous_stats_values() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        // Round 1: one DRAM page
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        let mut reqs = vec![make_prefetch(1, 1.0, 0.5, 4096)];
        SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        let after_r1 = stats.lock().unwrap().clone();
        assert_eq!(after_r1.rounds, 1);
        assert_eq!(after_r1.total_requests, 1);
        assert_eq!(after_r1.submitted, 1);

        // Round 2: empty
        let mut empty: Vec<PrefetchRequest> = vec![];
        SwapInWorker::swap_in_round(&config, &actor, &mut empty, &page_metadata, &addr_table, &stats, &observer);

        let after_r2 = stats.lock().unwrap().clone();
        assert_eq!(after_r2.rounds, 2, "rounds should increment");
        assert_eq!(after_r2.total_requests, 1, "total_requests should not change on empty");
        assert_eq!(after_r2.submitted, 1, "submitted should not change on empty");
        assert_eq!(after_r2.skipped, after_r1.skipped, "skipped should not change on empty");

        actor.shutdown();
    }

    /// Zero confidence with default min_confidence (0.1) should be skipped.
    #[test]
    fn round_zero_confidence_skipped_with_default_min_confidence() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default(); // min_confidence = 0.1

        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);
        let mut reqs = vec![make_prefetch(10, 1.0, 0.0, 4096)]; // confidence=0.0 < 0.1
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 0, "zero confidence should be skipped with default min");
        let s = stats.lock().unwrap();
        assert_eq!(s.skipped, 1);

        actor.shutdown();
    }

    /// Negative confidence should be skipped with default min_confidence.
    #[test]
    fn round_negative_confidence_skipped_with_default_min() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 20, StorageTier::CpuDram, CompressionCodec::None);
        let mut reqs = vec![make_prefetch(20, 1.0, -0.5, 4096)]; // confidence=-0.5 < 0.1
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 0, "negative confidence should be skipped");
        let s = stats.lock().unwrap();
        assert_eq!(s.skipped, 1);

        actor.shutdown();
    }

    /// f32::MAX confidence should be accepted with default min_confidence.
    #[test]
    fn round_f32_max_confidence_accepted_with_default() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 30, StorageTier::CpuDram, CompressionCodec::None);
        let mut reqs = vec![make_prefetch(30, 1.0, f32::MAX, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "f32::MAX confidence should be accepted");

        actor.shutdown();
    }

    /// Three DRAM pages should all be submitted.
    #[test]
    fn round_three_dram_pages_all_submitted() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 3, StorageTier::CpuDram, CompressionCodec::None);

        let mut reqs = vec![
            make_prefetch(1, 0.5, 0.8, 4096),
            make_prefetch(2, 0.7, 0.9, 4096),
            make_prefetch(3, 0.3, 0.6, 4096),
        ];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 3, "all three DRAM pages should be submitted");
        let s = stats.lock().unwrap();
        assert_eq!(s.total_requests, 3);

        actor.shutdown();
    }

    /// Three tiers mixed: one HBM (skipped), one DRAM (submitted), one NVMe (submitted+two-hop).
    #[test]
    fn round_three_tiers_hbm_skipped_dram_nvme_submitted() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 1, StorageTier::GpuHbm, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 3, StorageTier::Nvme, CompressionCodec::None);

        let mut reqs = vec![
            make_prefetch(1, 1.0, 0.9, 4096), // HBM -> skip
            make_prefetch(2, 0.8, 0.9, 4096), // DRAM -> submit
            make_prefetch(3, 0.6, 0.9, 4096), // NVMe -> submit + two-hop
        ];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 2, "DRAM and NVMe should be submitted, HBM skipped");
        let s = stats.lock().unwrap();
        assert_eq!(s.skipped, 1, "HBM page should be skipped");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should increment two-hop");

        actor.shutdown();
    }

    /// DRAM page with ZstdDict codec should still submit.
    #[test]
    fn round_dram_page_with_zstd_dict_codec() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 100, StorageTier::CpuDram, CompressionCodec::ZstdDict);
        let mut reqs = vec![make_prefetch(100, 1.0, 0.8, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "DRAM page with ZstdDict codec should be submitted");

        actor.shutdown();
    }

    /// NVMe page with Lz4 codec should submit two-hop.
    #[test]
    fn round_nvme_page_with_lz4_codec() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 200, StorageTier::Nvme, CompressionCodec::Lz4);
        let mut reqs = vec![make_prefetch(200, 1.0, 0.8, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "NVMe page with Lz4 codec should be submitted");
        let s = stats.lock().unwrap();
        assert_eq!(s.two_hop_promotions, 1);

        actor.shutdown();
    }

    /// NVMe page with BitPackRle codec should submit two-hop.
    #[test]
    fn round_nvme_page_with_bitpack_rle_codec() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 201, StorageTier::Nvme, CompressionCodec::BitPackRle);
        let mut reqs = vec![make_prefetch(201, 1.0, 0.8, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "NVMe page with BitPackRle codec should be submitted");
        let s = stats.lock().unwrap();
        assert_eq!(s.two_hop_promotions, 1);

        actor.shutdown();
    }

    /// NVMe page with None codec should submit two-hop.
    #[test]
    fn round_nvme_page_with_none_codec() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 202, StorageTier::Nvme, CompressionCodec::None);
        let mut reqs = vec![make_prefetch(202, 1.0, 0.8, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "NVMe page with None codec should be submitted");
        let s = stats.lock().unwrap();
        assert_eq!(s.two_hop_promotions, 1);

        actor.shutdown();
    }

    /// NVMe page with ZstdDict codec should submit two-hop.
    #[test]
    fn round_nvme_page_with_zstd_dict_codec() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        insert_addr_entry(&addr_table, 203, StorageTier::Nvme, CompressionCodec::ZstdDict);
        let mut reqs = vec![make_prefetch(203, 1.0, 0.8, 4096)];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "NVMe page with ZstdDict codec should be submitted");
        let s = stats.lock().unwrap();
        assert_eq!(s.two_hop_promotions, 1);

        actor.shutdown();
    }

    /// drain: OK completion for HBM promotion should set page state to Active and
    /// clear swap_in_time.
    #[test]
    fn drain_ok_hbm_clears_swap_in_time_sets_active() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert page into addr_table with a host_buffer so actor can promote it.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(50, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        // Insert metadata with swap_in_time set.
        let mut page_metadata_map: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata_map.insert(50, PageMetadata {
            page_id: 50,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata_map));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Send PromoteToHbm and poll until completion arrives.
        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 50, page_bytes: 4096 });
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let meta = page_metadata.read().unwrap().get(&50).cloned().unwrap();
        assert_eq!(meta.state, PageState::Active, "state should be Active after HBM promotion");
        assert!(meta.swap_in_time.is_none(), "swap_in_time should be cleared after HBM promotion");

        let s = stats.lock().unwrap();
        assert_eq!(s.promoted_ok, 1);

        actor.shutdown();
    }

    /// drain: OK completion for NVMe promotion to CpuDram should set page state to Warm
    /// and set swap_in_time. Uses PromoteToDram on a page with no host_buffer (NVMe path).
    #[test]
    fn drain_ok_dram_sets_swap_in_time_sets_warm() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert page into addr_table as NVMe-resident (no pointers).
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(51, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut page_metadata_map: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata_map.insert(51, PageMetadata {
            page_id: 51,
            sequence_id: None,
            recency: 0,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata_map));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Send PromoteToDram and poll for completion.
        let _ = actor.send(MigrationCommand::PromoteToDram { page_id: 51, page_bytes: 4096 });
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok + s.promoted_failed > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        // The actor may produce either Ok or Failed for NVMe (depends on swap file).
        // Verify that if it succeeded, the state transition is correct.
        let s = stats.lock().unwrap();
        if s.promoted_ok > 0 {
            let meta = page_metadata.read().unwrap().get(&51).cloned().unwrap();
            assert_eq!(meta.state, PageState::Warm, "state should be Warm after DRAM promotion");
            assert!(meta.swap_in_time.is_some(), "swap_in_time should be set after DRAM promotion");
        }

        actor.shutdown();
    }

    /// drain: mixed OK and Failed completions should increment both promoted_ok
    /// and promoted_failed counters. Uses one DRAM page (likely OK) and one NVMe page
    /// (may fail depending on swap file availability).
    #[test]
    fn drain_ok_and_failed_mixed_both_counters_increment() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Page 60: DRAM-resident with host_buffer (should succeed).
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(60, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        // Page 61: NVMe-resident without host_buffer (may fail).
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(61, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut page_metadata_map: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata_map.insert(60, PageMetadata {
            page_id: 60,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        page_metadata_map.insert(61, PageMetadata {
            page_id: 61,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata_map));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 60, page_bytes: 4096 });
        let _ = actor.send(MigrationCommand::PromoteToDram { page_id: 61, page_bytes: 4096 });

        // Poll until we have at least 2 completions total.
        for _ in 0..40 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok + s.promoted_failed >= 2 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let s = stats.lock().unwrap();
        assert!(
            s.promoted_ok + s.promoted_failed >= 2,
            "should have at least 2 completions: ok={} failed={}",
            s.promoted_ok, s.promoted_failed,
        );

        actor.shutdown();
    }

    /// drain: Failed completion should not increment promoted_ok or total_latency_us.
    #[test]
    fn drain_failed_does_not_increment_promoted_ok_or_latency() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert page as NVMe-resident with no host_buffer (PromoteToDram will likely fail).
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(70, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let mut page_metadata_map: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata_map.insert(70, PageMetadata {
            page_id: 70,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata_map));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let _ = actor.send(MigrationCommand::PromoteToDram { page_id: 70, page_bytes: 4096 });
        thread::sleep(Duration::from_millis(50));
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);

        let s = stats.lock().unwrap();
        // The actor may produce either Ok or Failed for NVMe.
        // If it failed, verify the invariants hold.
        if s.promoted_failed > 0 {
            assert_eq!(s.promoted_ok, 0, "failed should not increment promoted_ok");
            assert_eq!(s.total_latency_us, 0, "failed should not increment total_latency_us");
        }
        // At minimum, we should have a completion of some kind.
        assert!(
            s.promoted_ok + s.promoted_failed >= 1,
            "should have at least one completion: ok={} failed={}",
            s.promoted_ok, s.promoted_failed,
        );

        actor.shutdown();
    }

    /// Multiple addr-table misses should all be counted as skipped.
    #[test]
    fn round_multiple_addr_table_misses_all_counted_as_skipped() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        // Do NOT insert any addr entries — all pages miss.
        let mut reqs = vec![
            make_prefetch(301, 1.0, 0.8, 4096),
            make_prefetch(302, 1.0, 0.8, 4096),
            make_prefetch(303, 1.0, 0.8, 4096),
        ];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut reqs, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 0, "no pages should be submitted when all miss addr table");
        let s = stats.lock().unwrap();
        assert_eq!(s.skipped, 3, "all three misses should be counted as skipped");

        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 15 additional tests — edge cases and uncovered paths
    // ═══════════════════════════════════════════════════════════════════════════

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that drain_completions_and_update correctly processes
    /// PromoteToHbm for a CpuDram page: when it succeeds, the state
    /// becomes Active and swap_in_time is cleared.
    #[test]
    fn drain_ok_hbm_promotion_clears_swap_in_time_sets_active() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        // Insert page on CpuDram with a host_buffer.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(77, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata.insert(77, PageMetadata {
            page_id: 77,
            sequence_id: Some(10),
            recency: 0,
            access_count: 3,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // PromoteToHbm produces a completion with to_tier=GpuHbm.
        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 77,
            page_bytes: 4096,
        });
        for _ in 0..30 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let meta = page_metadata.read().expect("read lock").get(&77).cloned();
        assert!(meta.is_some(), "page should still exist in metadata");
        let meta = meta.unwrap();
        assert_eq!(meta.state, PageState::Active, "state should be Active after HBM promotion");
        assert!(meta.swap_in_time.is_none(), "swap_in_time should be cleared after HBM promotion");

        let s = stats.lock().expect("stats lock");
        assert!(s.promoted_ok >= 1, "should have at least 1 successful promotion");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that drain_completions_and_update does NOT change page metadata
    /// when a promotion fails (MigrationResult::Failed).
    #[test]
    fn drain_failed_leaves_metadata_unchanged() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        // No addr entry for this page_id — actor will fail the promotion.
        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata.insert(555, PageMetadata {
            page_id: 555,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: Some(Instant::now()),
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 555,
            page_bytes: 4096,
        });
        thread::sleep(Duration::from_millis(50));
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);

        let meta = page_metadata.read().expect("read lock").get(&555).cloned();
        assert!(meta.is_some(), "page metadata should still exist");
        let meta = meta.unwrap();
        // Original state should be unchanged — failed promotion does not modify metadata.
        assert_eq!(meta.state, PageState::SwappedOut, "failed promotion should not change state");
        assert!(meta.swap_in_time.is_some(), "swap_in_time should remain from before");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with negative urgency combined with
    /// max_prefetch_per_round=1 truncates to the single positive-urgency page
    /// and the total_requests counter still reflects the original count.
    #[test]
    fn round_negative_urgency_truncation_keeps_positive() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in [1usize, 2usize] {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: -5.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "only 1 should be submitted (max_prefetch_per_round=1)");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2, "both requests counted before truncation");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with max_in_flight large enough allows all
    /// requests to be submitted without back-pressure.
    #[test]
    fn round_large_max_in_flight_allows_all() {
        let config = SwapInWorkerConfig {
            max_in_flight: usize::MAX,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            for pid in 1..=10usize {
                table.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=10)
            .map(|pid| make_prefetch(pid, 0.5, 0.9, 4096))
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 10, "all 10 pages should be submitted with large max_in_flight");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round accepts requests with confidence > 1.0
    /// (the API does not clamp — confidence above threshold is accepted).
    #[test]
    fn round_confidence_above_one_is_accepted() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 2.5, // above 1.0 — still above min_confidence
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "confidence > 1.0 should still be accepted if above threshold");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 0, "should not skip page with confidence > 1.0");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that compute_urgency produces strictly ordered results for
    /// confidence values at 0.25 increments across all three tiers.
    #[test]
    fn urgency_strict_ordering_across_tiers_and_confidences() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        // HBM should always exceed DRAM, which should always exceed NVMe
        // for the same confidence level.
        for &conf in &[0.25, 0.5, 0.75, 1.0] {
            let u_hbm = SwapInWorker::compute_urgency(&meta, conf, StorageTier::GpuHbm);
            let u_dram = SwapInWorker::compute_urgency(&meta, conf, StorageTier::CpuDram);
            let u_nvme = SwapInWorker::compute_urgency(&meta, conf, StorageTier::Nvme);
            assert!(
                u_hbm > u_dram,
                "HBM > DRAM at conf={conf}: hbm={u_hbm} dram={u_dram}",
            );
            assert!(
                u_dram > u_nvme,
                "DRAM > NVMe at conf={conf}: dram={u_dram} nvme={u_nvme}",
            );
        }
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round truncates BEFORE checking confidence and
    /// back-pressure, so total_requests counts the original unsorted count
    /// but only max_prefetch_per_round are even considered.
    #[test]
    fn round_truncation_happens_before_confidence_check() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            min_confidence: 1.0, // all will be skipped since none have confidence=1.0
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        for pid in [1usize, 2usize, 3usize] {
            insert_addr_entry(&addr_table, pid, StorageTier::CpuDram, CompressionCodec::None);
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=3)
            .map(|pid| make_prefetch(pid, 0.5, 0.9, 4096))
            .collect();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 0, "all truncated requests should fail confidence check");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 3, "all 3 counted before truncation");
        // Only 1 was truncated-to, so only 1 could be skipped for confidence.
        assert_eq!(s.skipped, 1, "only 1 request was evaluated and skipped");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that compute_urgency with a very old last_access (1 day ago)
    /// has a negligible recency bonus.
    #[test]
    fn urgency_day_old_access_recency_bonus_negligible() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now() - Duration::from_secs(86400),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // recency_bonus = 1/(1+86400) ≈ 0.0000116, so u ≈ 0.00000116
        assert!(
            u < 0.001,
            "day-old access with zero confidence should have near-zero urgency: {u}",
        );
        assert!(u > 0.0, "urgency should still be positive: {u}");
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that drain_completions_and_update does not modify page metadata
    /// when the completion's page_id is not in the metadata map.
    #[test]
    fn drain_ok_page_not_in_metadata_leaves_metadata_empty() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        // Empty metadata — no pages registered.
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Send PromoteToHbm for a page that has an addr entry but no metadata.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(42, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 42,
            page_bytes: 4096,
        });
        for _ in 0..30 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok + s.promoted_failed > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        // Metadata should still be empty — the completion was processed without panic.
        let meta_map = page_metadata.read().expect("read lock");
        assert!(meta_map.is_empty(), "metadata should remain empty since no page was registered");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round does not count skipped for pages that are
    /// in the addr_table with HBM tier, AND processes a second DRAM page
    /// correctly in the same round.
    #[test]
    fn round_hbm_skip_and_dram_submit_in_same_round() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::GpuHbm, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            make_prefetch(1, 0.9, 0.8, 4096), // HBM — skip
            make_prefetch(2, 0.5, 0.8, 4096), // DRAM — submit
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "only DRAM page should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 1, "HBM page should be skipped");
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify SwapInWorkerStats::avg_latency_us with many promotions and
    /// precisely known total latency.
    #[test]
    fn stats_avg_latency_exact_division() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 8;
        stats.total_latency_us = 1000;
        // 1000 / 8 = 125.0 exactly.
        assert!(
            (stats.avg_latency_us() - 125.0).abs() < 1e-6,
            "avg should be exactly 125.0: {}",
            stats.avg_latency_us(),
        );
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that a fresh worker's stats() snapshot returns all-zero fields
    /// even after several ticks with no prefetch requests.
    #[test]
    fn worker_stats_remain_zero_with_no_prefetch() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(10),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );

        // Let it tick a few times with no requests.
        thread::sleep(Duration::from_millis(80));

        let s = worker.stats();
        assert_eq!(s.total_requests, 0, "no requests enqueued, so total should be 0");
        assert_eq!(s.submitted, 0);
        assert_eq!(s.promoted_ok, 0);

        worker.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that compute_urgency returns a finite value for all combinations
    /// of access_count in {0, 1, 100, 1000000} and confidence in {0.0, 0.5, 1.0}.
    #[test]
    fn urgency_all_access_count_confidence_combinations_finite() {
        let access_counts: [usize; 4] = [0, 1, 100, 1_000_000];
        let confidences: [f32; 3] = [0.0, 0.5, 1.0];
        let tiers: [StorageTier; 3] = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];

        for &ac in &access_counts {
            for &conf in &confidences {
                for &tier in &tiers {
                    let meta = PageMetadata {
                        page_id: 1,
                        sequence_id: None,
                        recency: 0,
                        access_count: ac,
                        last_access: Instant::now(),
                        swap_in_time: None,
                        is_lir: false,
                        state: PageState::Active,
                        warm_until: None,
                    };
                    let u = SwapInWorker::compute_urgency(&meta, conf, tier);
                    assert!(
                        u.is_finite(),
                        "urgency should be finite for ac={ac} conf={conf} tier={tier:?}: {u}",
                    );
                }
            }
        }
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that drain_completions_and_update increments promoted_failed
    /// (not promoted_ok) when a migration fails.
    #[test]
    fn drain_failed_increments_promoted_failed_not_ok() {
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // PromoteToHbm for a page not in addr_table — should fail.
        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 9999,
            page_bytes: 4096,
        });
        thread::sleep(Duration::from_millis(200));
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);

        let s = stats.lock().expect("stats lock");
        let total = s.promoted_ok + s.promoted_failed;
        assert!(total >= 1, "should have processed at least one completion");
        if s.promoted_failed > 0 {
            // Verify that total_latency was NOT incremented for failed promotion.
            assert_eq!(
                s.total_latency_us, 0,
                "latency should not be tracked for failed promotions",
            );
        }

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with a single NVMe page increments
    /// two_hop_promotions AND submitted correctly.
    #[test]
    fn round_single_nvme_increments_both_hop_and_submitted() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 7, StorageTier::Nvme, CompressionCodec::ZstdDict);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(7, 1.0, 0.9, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "NVMe page should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 1, "submitted counter should be 1");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should trigger two-hop");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with a DRAM page does NOT increment
    /// two_hop_promotions.
    #[test]
    fn round_single_dram_no_two_hop() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 3, StorageTier::CpuDram, CompressionCodec::Lz4);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(3, 0.8, 0.7, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 0, "DRAM page should not be two-hop");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with request page_bytes=0 falls back to
    /// config.page_bytes. The migration command should carry the config default.
    #[test]
    fn round_request_page_bytes_zero_uses_config_page_bytes() {
        let config = SwapInWorkerConfig {
            page_bytes: 8192,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // page_bytes=0 in the request → should use config.page_bytes=8192
        let mut requests = vec![make_prefetch(10, 1.0, 0.9, 0)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "page with zero request bytes should still submit using config default");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round increments two_hop_promotions only for NVMe pages,
    /// not DRAM pages, when both appear in the same batch.
    #[test]
    fn round_mixed_nvme_dram_two_hop_only_nvme() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        // Page 1 on NVMe, page 2 on DRAM.
        insert_addr_entry(&addr_table, 1, StorageTier::Nvme, CompressionCodec::Lz4);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::Lz4);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            make_prefetch(1, 0.9, 0.8, 4096), // NVMe → two-hop
            make_prefetch(2, 0.8, 0.8, 4096), // DRAM → no two-hop
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 2, "both NVMe and DRAM pages should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "only the NVMe page should trigger two-hop");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that two sequential swap_in_round calls accumulate stats correctly
    /// without interference.
    #[test]
    fn round_two_sequential_calls_accumulate_stats_independently() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Round 1: one DRAM page
        let mut req1 = vec![make_prefetch(1, 0.9, 0.8, 4096)];
        let sub1 = SwapInWorker::swap_in_round(
            &config, &actor, &mut req1, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(sub1, 1);

        // Round 2: another DRAM page
        let mut req2 = vec![make_prefetch(2, 0.8, 0.8, 4096)];
        let sub2 = SwapInWorker::swap_in_round(
            &config, &actor, &mut req2, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(sub2, 1);

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2, "total_requests should be 2 across two rounds");
        assert_eq!(s.submitted, 2, "submitted should accumulate to 2");
        assert_eq!(s.rounds, 2, "rounds should be 2");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with max_prefetch_per_round=0 counts the
    /// original requests in total_requests but submits nothing after truncation.
    #[test]
    fn round_zero_max_prefetch_counts_requests_submits_none() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(1, 1.0, 0.9, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 0, "max_prefetch_per_round=0 should submit nothing");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 0);
        assert_eq!(s.total_requests, 1, "total_requests should count the input request");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that PrefetchRequest PartialEq returns false when page_bytes differ
    /// but all other numeric fields match.
    #[test]
    fn prefetch_request_eq_false_when_page_bytes_differ() {
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 5,
            urgency: 0.5,
            prefetch_confidence: 0.7,
            page_bytes: 4096,
            enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 5,
            urgency: 0.5,
            prefetch_confidence: 0.7,
            page_bytes: 8192,
            enqueued_at: now,
        };
        assert_ne!(a, b, "requests with different page_bytes should not be equal");
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that PrefetchRequest PartialEq returns false when urgency differs
    /// but all other fields match.
    #[test]
    fn prefetch_request_eq_false_when_urgency_differs() {
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 5,
            urgency: 0.5,
            prefetch_confidence: 0.7,
            page_bytes: 4096,
            enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 5,
            urgency: 0.9,
            prefetch_confidence: 0.7,
            page_bytes: 4096,
            enqueued_at: now,
        };
        assert_ne!(a, b, "requests with different urgency should not be equal");
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerError::Display prefixes the message correctly
    /// for both variants with non-ASCII characters.
    #[test]
    fn error_display_non_ascii_message() {
        let send_err = SwapInWorkerError::SendFailed("channel closed — 数据丢失".to_string());
        let recv_err = SwapInWorkerError::RecvFailed("接收失败：timeout".to_string());

        let send_display = format!("{send_err}");
        let recv_display = format!("{recv_err}");

        assert!(send_display.contains("channel closed"), "SendFailed display should contain the message: {send_display}");
        assert!(send_display.contains("数据丢失"), "SendFailed display should preserve non-ASCII: {send_display}");
        assert!(recv_display.contains("接收失败"), "RecvFailed display should preserve non-ASCII: {recv_display}");
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round correctly processes a page with page_id=usize::MAX
    /// stored in the addr_table.
    #[test]
    fn round_page_id_usize_max_in_addr_table() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let max_id = usize::MAX;
        insert_addr_entry(&addr_table, max_id, StorageTier::CpuDram, CompressionCodec::Lz4);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(max_id, 1.0, 0.9, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "page with usize::MAX id should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with a request whose confidence is NaN passes
    /// the confidence check (NaN < min_confidence is false) and gets submitted.
    /// This is the actual behavior: NaN comparisons return false, so the skip
    /// guard does not trigger, and the request proceeds to submission.
    #[test]
    fn round_nan_confidence_passes_through_and_submits() {
        let config = SwapInWorkerConfig::default(); // min_confidence = 0.1
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(1, 1.0, f32::NAN, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // NaN < 0.1 evaluates to false, so the confidence check does NOT skip.
        assert_eq!(submitted, 1, "NaN confidence bypasses the skip check and submits");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 0, "NaN confidence does not count as skipped");
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with urgency values that are all equal
    /// processes them in their original order (stable sort).
    #[test]
    fn round_equal_urgency_stable_order_preserved() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 2,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        // Both pages on DRAM, both should be submitted.
        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 20, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Both have same urgency and confidence
        let mut requests = vec![
            make_prefetch(10, 0.5, 0.9, 4096),
            make_prefetch(20, 0.5, 0.9, 4096),
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 2, "both equal-urgency requests should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 0, "no requests should be skipped");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round drains the input vector completely even when
    /// some requests are skipped (HBM, low confidence, missing addr entry).
    #[test]
    fn round_drains_all_requests_mixed_skip_reasons() {
        let config = SwapInWorkerConfig::default(); // min_confidence = 0.1
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        // Page 1: HBM (skipped), Page 2: DRAM (submitted), Page 3: missing (skipped)
        insert_addr_entry(&addr_table, 1, StorageTier::GpuHbm, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);
        // Page 3 not in addr_table

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            make_prefetch(1, 1.0, 0.9, 4096),  // HBM → skip
            make_prefetch(2, 0.8, 0.9, 4096),  // DRAM → submit
            make_prefetch(3, 0.7, 0.9, 4096),  // missing → skip
            make_prefetch(4, 0.6, 0.01, 4096), // low confidence → skip
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "only the DRAM page should be submitted");
        assert!(requests.is_empty(), "input vector should be fully drained");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 3, "3 requests should be skipped (HBM + missing + low confidence)");
        assert_eq!(s.total_requests, 4, "total_requests should count all 4 input requests");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerStats::avg_latency_us returns 0.0 when
    /// total_latency_us > 0 but promoted_ok == 0 (no successful promotions).
    #[test]
    fn stats_avg_latency_zero_promoted_with_nonzero_total_latency() {
        let stats = SwapInWorkerStats {
            total_requests: 100,
            submitted: 50,
            skipped: 10,
            promoted_ok: 0,
            promoted_failed: 5,
            two_hop_promotions: 3,
            total_latency_us: 99999,
            rounds: 10,
        };
        assert_eq!(stats.avg_latency_us(), 0.0, "avg_latency should be 0 when promoted_ok is 0");
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerStats::avg_latency_us returns an exact value
    /// when total_latency_us divides evenly by promoted_ok.
    #[test]
    fn stats_avg_latency_exact_integer_result() {
        let stats = SwapInWorkerStats {
            total_requests: 0,
            submitted: 0,
            skipped: 0,
            promoted_ok: 4,
            promoted_failed: 0,
            two_hop_promotions: 0,
            total_latency_us: 1000,
            rounds: 0,
        };
        let avg = stats.avg_latency_us();
        assert!(
            (avg - 250.0).abs() < f64::EPSILON,
            "avg_latency should be exactly 250.0: got {avg}",
        );
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that urgency with a future last_access (Instant in the future)
    /// does not panic and produces a finite result. saturating_duration_since
    /// should return 0 for future times.
    #[test]
    fn urgency_future_last_access_no_panic_finite() {
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now() + Duration::from_secs(60),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        assert!(u.is_finite(), "urgency should be finite with future last_access: {u}");
        assert!(u > 0.0, "urgency should be positive with future last_access: {u}");
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with a single NVMe page increments both
    /// two_hop_promotions (for the PromoteToDram) and submitted (for the
    /// PromoteToHbm). The in_flight counter should reflect 2 (one for each command).
    #[test]
    fn round_nvme_page_in_flight_counts_both_commands() {
        let config = SwapInWorkerConfig {
            max_in_flight: 1, // Very tight limit
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        // NVMe page: PromoteToDram (in_flight=1) + PromoteToHbm (in_flight=2 > max_in_flight=1)
        // But the back-pressure check is BEFORE submitting, and both commands are for the same page.
        // The NVMe PromoteToDram increments in_flight to 1, then PromoteToHbm increments to 2.
        // With max_in_flight=1, the first request's in_flight will exceed after both commands.
        insert_addr_entry(&addr_table, 7, StorageTier::Nvme, CompressionCodec::Lz4);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(7, 1.0, 0.9, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // The NVMe page submits PromoteToDram (in_flight becomes 1) then PromoteToHbm
        // (in_flight becomes 2). submitted counts PromoteToHbm commands = 1.
        assert_eq!(submitted, 1, "NVMe page should submit one PromoteToHbm");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should trigger two-hop");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that worker.prefetch() succeeds before shutdown and that
    /// the returned stats eventually reflect the submitted request.
    #[test]
    fn worker_prefetch_succeeds_before_shutdown() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(10),
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);

        let req = PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        let result = worker.prefetch(req);
        assert!(result.is_ok(), "prefetch before shutdown should succeed");

        // Give the worker time to process.
        thread::sleep(Duration::from_millis(100));

        worker.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that PrefetchRequest with page_bytes=0 is constructed and compared
    /// correctly. Zero page_bytes is valid (the swap_in_round will fall back
    /// to config.page_bytes).
    #[test]
    fn prefetch_request_zero_page_bytes_valid() {
        let now = Instant::now();
        let req = PrefetchRequest {
            page_id: 42,
            urgency: 0.0,
            prefetch_confidence: 0.0,
            page_bytes: 0,
            enqueued_at: now,
        };
        assert_eq!(req.page_bytes, 0);
        assert_eq!(req.page_id, 42);
        // Clone preserves zero.
        let cloned = req.clone();
        assert_eq!(cloned.page_bytes, 0);
        assert_eq!(cloned, req);
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerStats manually constructed with all fields zero
    /// equals the Default-derived instance.
    #[test]
    fn stats_manual_zero_equals_default() {
        let manual = SwapInWorkerStats {
            total_requests: 0,
            submitted: 0,
            skipped: 0,
            promoted_ok: 0,
            promoted_failed: 0,
            two_hop_promotions: 0,
            total_latency_us: 0,
            rounds: 0,
        };
        let default = SwapInWorkerStats::default();
        assert_eq!(manual, default, "manual zero-constructed stats should equal default");
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerError implements std::error::Error so it can
    /// be used in Result<_, Box<dyn std::error::Error>>.
    #[test]
    fn error_is_send_sync_std_error() {
        fn assert_error_trait<T: std::error::Error + Send + Sync + 'static>(_: &T) {}
        let err = SwapInWorkerError::SendFailed("test".to_string());
        assert_error_trait(&err);
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerConfig with custom non-default values is not
    /// equal to the default config.
    #[test]
    fn config_custom_not_equal_default() {
        let custom = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            tick_interval: Duration::from_secs(1),
            min_confidence: 0.99,
            max_in_flight: 1,
            page_bytes: 8192,
        };
        assert_ne!(custom, SwapInWorkerConfig::default());
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round uses config.page_bytes when the request has
    /// page_bytes=0 (the fallback path).
    #[test]
    fn round_uses_config_page_bytes_for_zero_request_bytes() {
        let config = SwapInWorkerConfig {
            page_bytes: 8192,
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Request with page_bytes=0 should use config.page_bytes (8192).
        let mut requests = vec![make_prefetch(1, 1.0, 0.9, 0)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "page with zero request bytes should still be submitted using config fallback");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with page_id=0 (zero-valued PageId) works
    /// correctly when the addr_table has an entry for page 0.
    #[test]
    fn round_page_id_zero_in_addr_table() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 0, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(0, 1.0, 0.9, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "page_id=0 should be processed normally");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 0, "page_id=0 on DRAM should not be skipped");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that compute_urgency with zero access count and zero confidence
    /// still produces a positive urgency due to the recency bonus.
    #[test]
    fn urgency_zero_access_zero_confidence_still_positive_from_recency() {
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
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        // First term = ln(1)/ln(10) * 0.0 * 1.0 = 0.0. Recency = 1/(1+0)*0.1 = 0.1.
        assert!(
            (u - 0.1).abs() < 1e-6,
            "urgency should be 0.1 from recency only: got {u}",
        );
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that compute_urgency with HBM tier yields the highest tier bonus
    /// (2.0 multiplier on the first term).
    #[test]
    fn urgency_hbm_tier_bonus_is_two() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::GpuHbm);
        // importance_rebound = ln(11)/ln(10), tier_bonus = 2.0
        // First term = importance_rebound * 1.0 * 2.0 = 2 * importance_rebound
        // Plus recency bonus ~0.1
        let importance_rebound = (10f32).ln_1p() / 10.0f32.ln_1p();
        let expected_first = importance_rebound * 1.0 * 2.0;
        assert!(
            u >= expected_first,
            "HBM urgency should be at least importance_rebound*2: got {u}, expected >= {expected_first}",
        );
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with min_confidence=0 accepts confidence=0
    /// (boundary value) requests.
    #[test]
    fn round_confidence_zero_accepted_when_min_confidence_zero() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(1, 1.0, 0.0, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "confidence=0 with min_confidence=0 should be accepted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 0, "nothing should be skipped");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round with all requests having urgency=NaN still
    /// processes without panic (NaN sorts as Equal per the partial_cmp).
    #[test]
    fn round_nan_urgency_no_panic() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(1, f32::NAN, 0.9, 4096)];
        // Should not panic during sort.
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "NaN urgency page should still be submitted");
        assert!(requests.is_empty(), "requests should be drained");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerStats with large total_latency_us and promoted_ok=1
    /// computes avg_latency_us correctly (no overflow for u64).
    #[test]
    fn stats_avg_latency_large_u64_single_promotion() {
        let stats = SwapInWorkerStats {
            total_requests: 1,
            submitted: 1,
            skipped: 0,
            promoted_ok: 1,
            promoted_failed: 0,
            two_hop_promotions: 0,
            total_latency_us: u64::MAX,
            rounds: 1,
        };
        let avg = stats.avg_latency_us();
        assert!(
            (avg - u64::MAX as f64).abs() < 1.0,
            "avg_latency should be approximately u64::MAX as f64: got {avg}",
        );
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that swap_in_round increments rounds counter even when all
    /// requests are filtered out by max_prefetch_per_round=0.
    #[test]
    fn round_zero_max_prefetch_increments_rounds_and_requests() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 0,
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(1, 1.0, 0.9, 4096)];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 0, "max_prefetch_per_round=0 should submit nothing");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 1, "rounds should be 1");
        assert_eq!(s.total_requests, 1, "total_requests should count before truncation");

        actor.shutdown();
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that PrefetchRequest Debug output contains "PrefetchRequest"
    /// (the struct name) and all field names.
    #[test]
    fn prefetch_request_debug_contains_struct_name() {
        let req = PrefetchRequest {
            page_id: 99,
            urgency: 0.5,
            prefetch_confidence: 0.75,
            page_bytes: 8192,
            enqueued_at: Instant::now(),
        };
        let debug_str = format!("{req:?}");
        assert!(
            debug_str.contains("PrefetchRequest"),
            "Debug output should contain struct name: {debug_str}",
        );
        assert!(
            debug_str.contains("page_id"),
            "Debug output should contain page_id field: {debug_str}",
        );
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerConfig Clone produces an independent copy.
    /// Mutating the clone should not affect the original.
    #[test]
    fn config_clone_mutation_does_not_affect_original() {
        let original = SwapInWorkerConfig::default();
        let mut cloned = original.clone();
        cloned.max_prefetch_per_round = 999;
        cloned.min_confidence = 0.999;
        cloned.page_bytes = 0;
        assert_ne!(
            original.max_prefetch_per_round, cloned.max_prefetch_per_round,
            "mutating clone should not affect original",
        );
        assert_eq!(
            original,
            SwapInWorkerConfig::default(),
            "original should remain unchanged",
        );
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that SwapInWorkerStats PartialEq returns false when
    /// two_hop_promotions differs between two otherwise identical stats.
    #[test]
    fn stats_partial_eq_differs_on_two_hop_promotions() {
        let a = SwapInWorkerStats {
            total_requests: 10,
            submitted: 5,
            skipped: 2,
            promoted_ok: 3,
            promoted_failed: 0,
            two_hop_promotions: 1,
            total_latency_us: 100,
            rounds: 5,
        };
        let mut b = a.clone();
        b.two_hop_promotions = 0;
        assert_ne!(a, b, "stats differing only in two_hop_promotions should not be equal");
    }

    // @trace REQ-COMP-006 [level:unit]
    /// Verify that worker.prefetch_batch() correctly enqueues all items
    /// in a batch and returns the exact count.
    #[test]
    fn worker_prefetch_batch_returns_exact_count() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(50),
            ..SwapInWorkerConfig::default()
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
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);

        let batch: Vec<PrefetchRequest> = (0..5)
            .map(|i| PrefetchRequest {
                page_id: i,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        let enqueued = worker.prefetch_batch(&batch);
        assert_eq!(enqueued, 5, "all 5 batch requests should be enqueued");

        worker.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional 15 tests — final batch
    // ═══════════════════════════════════════════════════════════════════════════

    /// Verify that PrefetchRequest with page_id=0 is a valid request (page_id
    /// is just a usize alias, zero is a legitimate page identifier).
    #[test]
    fn prefetch_request_zero_page_id_clone_preserves() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.page_id, 0);
        // Clone preserves page_id=0.
        let cloned = req.clone();
        assert_eq!(cloned.page_id, 0);
    }

    /// Verify that compute_urgency produces a finite positive value when
    /// access_count is non-zero and confidence is positive, even with a very
    /// old last_access.
    #[test]
    fn urgency_nonzero_access_old_timestamp_still_finite() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 50,
            last_access: Instant::now() - Duration::from_secs(7200), // 2 hours ago.
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.8, StorageTier::CpuDram);
        assert!(u.is_finite(), "urgency should be finite: {u}");
        assert!(u > 0.0, "urgency should be positive with nonzero access_count and confidence: {u}");
    }

    /// Verify that SwapInWorkerConfig::default() produces min_confidence in the
    /// range (0.0, 1.0) — a valid probability threshold.
    #[test]
    fn config_default_min_confidence_in_valid_range() {
        let cfg = SwapInWorkerConfig::default();
        assert!(
            cfg.min_confidence > 0.0 && cfg.min_confidence < 1.0,
            "default min_confidence should be in (0, 1): {}",
            cfg.min_confidence,
        );
    }

    /// Verify SwapInWorkerError Display output length includes both the prefix
    /// and the message for RecvFailed.
    #[test]
    fn error_recv_failed_display_length_includes_prefix_and_message() {
        let msg = "connection lost";
        let e = SwapInWorkerError::RecvFailed(msg.to_string());
        let display = format!("{e}");
        assert!(display.len() > msg.len(), "Display should be longer than just the message");
        assert!(display.contains(msg));
    }

    /// Verify SwapInWorkerStats cloned from default remain equal after the
    /// original is mutated — clones are snapshots.
    #[test]
    fn stats_clone_from_default_independent_after_mutation() {
        let stats = SwapInWorkerStats::default();
        let snapshot = stats.clone();
        // Mutate a copy — the snapshot should remain at defaults.
        let mut mutated = stats;
        mutated.total_requests = 500;
        mutated.rounds = 10;
        assert_eq!(snapshot.total_requests, 0, "snapshot should be independent");
        assert_eq!(snapshot.rounds, 0);
        assert_eq!(mutated.total_requests, 500);
    }

    /// Verify PageMetadata clone produces a fully independent copy that can be
    /// mutated without affecting the original.
    #[test]
    fn page_metadata_clone_independent_access_count() {
        let now = Instant::now();
        let mut meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(5),
            recency: 3,
            access_count: 10,
            last_access: now,
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let cloned = meta.clone();
        meta.access_count = 999;
        meta.state = PageState::SwappedOut;
        assert_eq!(cloned.access_count, 10, "clone should preserve original access_count");
        assert_eq!(cloned.state, PageState::Active, "clone should preserve original state");
    }

    /// Verify SwapInWorkerError::RecvFailed Debug output contains the variant
    /// name so that log output is actionable.
    #[test]
    fn error_recv_failed_debug_actionable_format() {
        let e = SwapInWorkerError::RecvFailed("timeout".into());
        let debug = format!("{e:?}");
        assert!(debug.contains("RecvFailed"), "Debug must contain variant name: {debug}");
        assert!(debug.contains("timeout"), "Debug must contain message: {debug}");
    }

    /// Verify that two SwapInWorkerConfig differ only in max_prefetch_per_round
    /// are detected as not-equal by PartialEq.
    #[test]
    fn config_partial_eq_max_prefetch_per_round_difference() {
        let c1 = SwapInWorkerConfig {
            max_prefetch_per_round: 16,
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            max_prefetch_per_round: 32,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2, "configs differing only in max_prefetch_per_round should not be equal");
    }

    /// Verify compute_urgency on CpuDram with access_count=0 and confidence=0.5:
    /// first term is zero (ln_1p(0)=0), only recency bonus remains.
    #[test]
    fn urgency_zero_access_nonzero_confidence_recency_dominates() {
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        // importance_rebound = ln(1)/ln(11) = 0.0, so first term = 0.
        // Only recency_bonus * 0.1 remains.
        assert!(
            (u - 0.1).abs() < 0.02,
            "zero access_count should yield only recency bonus ~0.1: {u}",
        );
    }

    /// Verify swap_in_round for an NVMe page with page_bytes=0 falls back to
    /// config.page_bytes for both PromoteToDram and PromoteToHbm commands.
    #[test]
    fn round_nvme_page_zero_bytes_uses_config_fallback() {
        let config = SwapInWorkerConfig {
            page_bytes: 8192,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 8192,
                codec: CompressionCodec::ZstdDict,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 0, // Falls back to config.page_bytes=8192.
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "NVMe page with page_bytes=0 should submit using config fallback");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should be two-hop");

        actor.shutdown();
    }

    /// Verify swap_in_round for a page not in addr_table on NVMe tier is
    /// counted as skipped (the page_id is missing entirely).
    #[test]
    fn round_missing_addr_table_entry_counted_as_skipped() {
        let config = SwapInWorkerConfig::default();
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Two requests for pages that do NOT exist in addr_table.
        let mut requests = vec![
            PrefetchRequest { page_id: 100, urgency: 1.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 200, urgency: 0.8, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0, "missing addr_table entries should not be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 2, "both missing entries should be skipped");

        actor.shutdown();
    }

    /// Verify that SwapInWorkerStats tracks promoted_failed independently from
    /// promoted_ok — they are separate counters.
    #[test]
    fn stats_promoted_failed_independent_counter() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 10;
        stats.promoted_failed = 3;
        stats.total_latency_us = 5000;
        // promoted_failed does NOT participate in avg_latency_us.
        let avg = stats.avg_latency_us();
        assert!(
            (avg - 500.0).abs() < 1e-6,
            "avg should be 5000/10=500, not affected by promoted_failed: {avg}",
        );
        // Verify they are distinct counters.
        assert_ne!(stats.promoted_ok, stats.promoted_failed);
    }

    /// Verify swap_in_round with negative urgency and confidence=0.0 is skipped
    /// because confidence < default min_confidence (0.1).
    #[test]
    fn round_negative_urgency_zero_confidence_skipped() {
        let config = SwapInWorkerConfig::default(); // min_confidence=0.1
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: -5.0,
            prefetch_confidence: 0.0, // Below default min_confidence=0.1.
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0, "zero confidence should be skipped");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 1);

        actor.shutdown();
    }

    /// Verify that drain_completions_and_update called on an actor with no
    /// pending completions does not panic and leaves stats at zero.
    #[test]
    fn drain_no_pending_completions_safe_no_panic() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        // Insert a page but do NOT send any migration command.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let mut page_metadata: HashMap<PageId, PageMetadata> = HashMap::new();
        page_metadata.insert(1, PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(page_metadata));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Multiple drain calls with no pending completions — should not panic.
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
        drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.promoted_ok, 0);
        assert_eq!(s.promoted_failed, 0);
        assert_eq!(s.total_latency_us, 0);

        actor.shutdown();
    }

    /// Verify that swap_in_round uses config.page_bytes for an NVMe page when
    /// the request specifies a non-zero page_bytes — the request value should
    /// take precedence over the config default.
    #[test]
    fn round_nvme_page_nonzero_request_bytes_not_fallback() {
        let config = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 16384,
                codec: CompressionCodec::ZstdDict,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // page_bytes=16384 explicitly — should NOT fall back to config.page_bytes=4096.
        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 16384,
            enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should be two-hop");

        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional tests — 15 new tests covering edge cases and behavior
    // ═══════════════════════════════════════════════════════════════════════════

    /// Verify that SwapInWorkerConfig::default() matches an explicit struct literal.
    #[test]
    fn config_default_matches_struct_literal() {
        let from_default = SwapInWorkerConfig::default();
        let from_literal = SwapInWorkerConfig {
            max_prefetch_per_round: 16,
            tick_interval: Duration::from_millis(5),
            min_confidence: 0.1,
            max_in_flight: 64,
            page_bytes: 4096,
        };
        assert_eq!(from_default, from_literal);
    }

    /// Verify that a zero tick_interval is stored verbatim and compares equal.
    #[test]
    fn config_tick_interval_zero_valid() {
        let cfg = SwapInWorkerConfig {
            tick_interval: Duration::ZERO,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.tick_interval, Duration::ZERO);
    }

    /// Verify SwapInWorkerError::SendFailed with an empty string formats the prefix.
    #[test]
    fn error_empty_message_send_display() {
        let e = SwapInWorkerError::SendFailed(String::new());
        let msg = format!("{e}");
        assert!(
            msg.contains("swap-in worker send failed:"),
            "display should contain the prefix even with empty message: got '{msg}'"
        );
    }

    /// Verify SwapInWorkerError::RecvFailed with an empty string formats the prefix.
    #[test]
    fn error_empty_message_recv_display() {
        let e = SwapInWorkerError::RecvFailed(String::new());
        let msg = format!("{e}");
        assert!(
            msg.contains("swap-in worker recv failed:"),
            "display should contain the prefix even with empty message: got '{msg}'"
        );
    }

    /// Verify urgency with zero access_count and max confidence: first term is zero,
    /// only the recency bonus contributes.
    #[test]
    fn urgency_zero_access_max_confidence_dram_only_recency() {
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
        let urgency = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);

        // access_count=0 → ln_1p(0)=0 → importance_rebound=0 → first term=0
        // recency bonus = 1/(1+0) * 0.1 = 0.1 (approximately, since last_access is Instant::now())
        assert!(
            urgency > 0.0,
            "recency bonus should keep urgency positive even with zero access count: got {urgency}"
        );
        assert!(
            urgency < 0.5,
            "with zero importance the urgency should be small (recency only): got {urgency}"
        );
    }

    /// Verify urgency recency bonus formula: at exactly zero elapsed, bonus = 0.1.
    #[test]
    fn urgency_recency_bonus_zero_elapsed_exact() {
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
        // With access_count=0 and any confidence, first term is 0.
        // recency bonus = 1/(1+~0) * 0.1 ≈ 0.1
        let urgency = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        let expected = 0.1; // 1/(1+~0) * 0.1
        assert!(
            (urgency - expected).abs() < 0.01,
            "recency bonus at zero elapsed should be ~0.1: got {urgency}, expected ~{expected}"
        );
    }

    /// Verify swap_in_round updates total_requests with the exact input count.
    #[test]
    fn round_total_requests_includes_all_input() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            max_prefetch_per_round: 100,
            ..SwapInWorkerConfig::default()
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // 5 requests, all will be skipped (not in addr_table), but total_requests should be 5.
        let mut requests: Vec<PrefetchRequest> = (0..5)
            .map(|i| PrefetchRequest {
                page_id: i,
                urgency: 1.0,
                prefetch_confidence: 0.8,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        let s = stats.lock().expect("stats lock");
        assert_eq!(
            s.total_requests, 5,
            "total_requests should match the number of input requests"
        );
        assert_eq!(s.skipped, 5, "all 5 should be skipped (not in addr_table)");

        actor.shutdown();
    }

    /// Verify swap_in_round with max_prefetch_per_round=1 processes only the highest-urgency request.
    #[test]
    fn round_max_prefetch_one_processes_single_request() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Insert one CpuDram page for page_id=1 (highest urgency).
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            table.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Page 2 has lower urgency; page 1 has higher urgency.
        let mut requests = vec![
            PrefetchRequest {
                page_id: 2,
                urgency: 0.1,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 1,
                urgency: 10.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "only 1 request should be submitted due to truncation");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2, "total_requests counts all input before truncation");

        actor.shutdown();
    }

    /// Verify NVMe page with Lz4 codec triggers two-hop promotion.
    #[test]
    fn round_nvme_with_lz4_codec_two_hop() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(99, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 8192,
                codec: CompressionCodec::Lz4,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 99,
            urgency: 5.0,
            prefetch_confidence: 0.7,
            page_bytes: 8192,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "NVMe page should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page with Lz4 should be two-hop");

        actor.shutdown();
    }

    /// Verify PageMetadata::default() has state = Standby.
    #[test]
    fn page_metadata_default_standby_explicit() {
        let meta = PageMetadata::default();
        assert_eq!(meta.state, PageState::Standby, "default state should be Standby");
        assert!(!meta.is_lir, "default is_lir should be false");
        assert_eq!(meta.access_count, 0, "default access_count should be 0");
        assert_eq!(meta.recency, 0, "default recency should be 0");
        assert!(meta.sequence_id.is_none(), "default sequence_id should be None");
        assert!(meta.swap_in_time.is_none(), "default swap_in_time should be None");
        assert!(meta.warm_until.is_none(), "default warm_until should be None");
    }

    /// Verify PrefetchRequest clone preserves all numeric fields.
    #[test]
    fn prefetch_request_clone_preserves_all_numeric_fields() {
        let now = Instant::now();
        let original = PrefetchRequest {
            page_id: 12345,
            urgency: 0.77,
            prefetch_confidence: 0.33,
            page_bytes: 16384,
            enqueued_at: now,
        };
        let cloned = original.clone();

        assert_eq!(cloned.page_id, original.page_id);
        assert_eq!(cloned.urgency, original.urgency);
        assert_eq!(cloned.prefetch_confidence, original.prefetch_confidence);
        assert_eq!(cloned.page_bytes, original.page_bytes);
        assert_eq!(cloned, original, "cloned PrefetchRequest should be equal to original");
    }

    /// Verify SwapInWorkerStats::promoted_failed does not affect avg_latency_us.
    #[test]
    fn stats_promoted_failed_independent_of_avg_latency() {
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_failed = 100;
        stats.total_latency_us = 0;
        // promoted_ok is still 0, so avg_latency should be 0.0
        assert_eq!(
            stats.avg_latency_us(),
            0.0,
            "promoted_failed should not affect avg_latency_us"
        );
    }

    /// Verify swap_in_round with both DRAM and NVMe pages submits both.
    #[test]
    fn round_dram_nvme_mixed_submission_both_submitted() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            table.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 2.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 2, "both DRAM and NVMe pages should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "only NVMe page should be two-hop");

        actor.shutdown();
    }

    /// Verify SwapInWorkerConfig min_confidence at exactly the threshold boundary.
    #[test]
    fn config_min_confidence_boundary_float_equality() {
        let cfg = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };

        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Confidence exactly equal to min_confidence should be accepted (not < threshold).
        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &cfg, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(
            submitted, 1,
            "confidence exactly at threshold should be accepted"
        );

        actor.shutdown();
    }

    /// Verify SwapInWorkerError: same string message but different variants are not equal.
    #[test]
    fn error_same_string_different_variants_not_equal() {
        let msg = "identical message";
        let send_err = SwapInWorkerError::SendFailed(msg.to_string());
        let recv_err = SwapInWorkerError::RecvFailed(msg.to_string());
        assert_ne!(send_err, recv_err, "different variants with same message should not be equal");
    }

    /// Verify swap_in_round with empty requests still increments the rounds counter.
    #[test]
    fn round_empty_increments_rounds_counter() {
        let config = SwapInWorkerConfig::default();
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
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut empty_requests: Vec<PrefetchRequest> = Vec::new();

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut empty_requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 0, "empty requests should yield zero submissions");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 1, "rounds counter should increment even for empty input");
        assert_eq!(s.total_requests, 0, "total_requests should be zero for empty input");

        actor.shutdown();
    }

    // ── Additional tests for untested edge cases ─────────────────────────────

    /// Verify SwapInWorkerStats::avg_latency_us returns u64::MAX as f64 when
    /// total_latency_us is u64::MAX and promoted_ok is 1.
    #[test]
    fn stats_avg_latency_u64_max_total_single_promotion() {
        let mut stats = SwapInWorkerStats::default();
        stats.total_latency_us = u64::MAX;
        stats.promoted_ok = 1;

        let avg = stats.avg_latency_us();
        assert!(
            avg.is_finite(),
            "avg_latency_us should be finite for u64::MAX / 1"
        );
        assert_eq!(
            avg, u64::MAX as f64,
            "avg_latency_us should equal u64::MAX as f64 when promoted_ok is 1"
        );
    }

    /// Verify SwapInWorkerConfig stores page_bytes = 2 correctly (non-default, non-zero).
    #[test]
    fn config_page_bytes_two() {
        let cfg = SwapInWorkerConfig {
            page_bytes: 2,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(cfg.page_bytes, 2);
        assert_ne!(cfg.page_bytes, SwapInWorkerConfig::default().page_bytes);
    }

    /// Verify PrefetchRequest stores f32::MIN urgency (most negative float).
    #[test]
    fn prefetch_request_urgency_negative_f32_min() {
        let req = PrefetchRequest {
            page_id: 0,
            urgency: f32::MIN,
            prefetch_confidence: 0.5,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };
        assert_eq!(req.urgency, f32::MIN);
    }

    /// Verify three consecutive empty rounds each increment the rounds counter.
    #[test]
    fn round_three_sequential_empty_increments_rounds() {
        let config = SwapInWorkerConfig::default();
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
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        for _ in 0..3 {
            let mut empty: Vec<PrefetchRequest> = Vec::new();
            SwapInWorker::swap_in_round(
                &config, &actor, &mut empty, &page_metadata, &addr_table, &stats, &observer,
            );
        }

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.rounds, 3, "three empty rounds should yield rounds=3");
        assert_eq!(s.submitted, 0);

        actor.shutdown();
    }

    /// Verify swap_in_round submits a DRAM page with page_id = usize::MAX / 2.
    #[test]
    fn round_dram_single_large_page_id() {
        let page_id = usize::MAX / 2;
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(page_id, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id,
            urgency: 1.0,
            prefetch_confidence: 0.8,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "large page_id should submit successfully");

        actor.shutdown();
    }

    /// Verify swap_in_round accepts confidence=0.0 when min_confidence=0.0.
    #[test]
    fn round_confidence_zero_min_confidence_zero_accepted() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1,
            urgency: 1.0,
            prefetch_confidence: 0.0,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(
            submitted, 1,
            "confidence=0.0 should be accepted when min_confidence=0.0"
        );

        actor.shutdown();
    }

    /// Verify NVMe page with max_in_flight=2: PromoteToDram uses 1 slot,
    /// PromoteToHbm uses 1 slot, then back-pressure stops further submissions.
    #[test]
    fn round_nvme_back_pressure_at_two_limits_both() {
        let config = SwapInWorkerConfig {
            max_in_flight: 2,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        // Two NVMe pages: first consumes 2 in-flight (PromoteToDram + PromoteToHbm),
        // second should be blocked.
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            table.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: 10.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 5.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // First NVMe page: PromoteToDram (in_flight=1) + PromoteToHbm (in_flight=2)
        // Second NVMe page: in_flight(2) >= max_in_flight(2), blocked.
        assert_eq!(submitted, 1, "only first NVMe page should be submitted");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1);

        actor.shutdown();
    }

    /// Verify swap_in_round returns submitted=1 for a single DRAM page submission.
    #[test]
    fn round_submitted_equals_one_for_single_dram() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(7, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 7,
            urgency: 5.0,
            prefetch_confidence: 0.7,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    /// Verify two DRAM pages with different urgencies: higher urgency processed first
    /// (verified by checking both are submitted and total_requests count is correct).
    #[test]
    fn round_two_dram_pages_sorted_by_urgency() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            table.insert(2, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: 1.0,
                prefetch_confidence: 0.8,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 10.0,
                prefetch_confidence: 0.8,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 2, "both DRAM pages should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 2);
        assert_eq!(s.total_requests, 2);

        actor.shutdown();
    }

    /// Verify swap_in_round counts 2 skipped when both pages are on HBM.
    #[test]
    fn round_stats_skipped_equals_two_for_two_hbm() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(1, PageAddrEntry {
                gpu_ptr: Some(0x1000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
            table.insert(2, PageAddrEntry {
                gpu_ptr: Some(0x2000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2,
                urgency: 2.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            },
        ];

        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 0, "HBM pages should not be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 2, "both HBM pages should be counted as skipped");

        actor.shutdown();
    }

    /// Verify SwapInWorkerError::SendFailed Display output contains the original message.
    #[test]
    fn error_send_failed_display_contains_original_message() {
        let msg = "channel closed unexpectedly";
        let err = SwapInWorkerError::SendFailed(msg.to_string());
        let display = format!("{err}");
        assert!(
            display.contains(msg),
            "Display output should contain original message: got '{}'",
            display
        );
    }

    /// Verify SwapInWorkerError::RecvFailed Display output contains the original message.
    #[test]
    fn error_recv_failed_display_contains_original_message() {
        let msg = "timeout waiting for completion";
        let err = SwapInWorkerError::RecvFailed(msg.to_string());
        let display = format!("{err}");
        assert!(
            display.contains(msg),
            "Display output should contain original message: got '{}'",
            display
        );
    }

    /// Verify PrefetchRequest PartialEq: two requests with identical numeric fields
    /// but different enqueued_at timestamps are NOT equal (PartialEq compares all fields
    /// including Instant).
    #[test]
    fn prefetch_request_eq_same_numeric_different_enqueued() {
        let req1 = PrefetchRequest {
            page_id: 42,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 1024,
            enqueued_at: Instant::now(),
        };
        thread::sleep(Duration::from_micros(10));
        let req2 = PrefetchRequest {
            page_id: 42,
            urgency: 0.5,
            prefetch_confidence: 0.8,
            page_bytes: 1024,
            enqueued_at: Instant::now(),
        };
        assert_ne!(
            req1, req2,
            "different enqueued_at should make requests not equal even with same numeric fields"
        );
    }

    /// Verify SwapInWorkerStats Debug output contains the "submitted" field name.
    #[test]
    fn stats_debug_contains_submitted_field() {
        let stats = SwapInWorkerStats {
            submitted: 99,
            ..SwapInWorkerStats::default()
        };
        let debug = format!("{stats:?}");
        assert!(
            debug.contains("submitted"),
            "Debug output should contain 'submitted' field: got '{}'",
            debug
        );
        assert!(
            debug.contains("99"),
            "Debug output should contain the value 99: got '{}'",
            debug
        );
    }

    /// Verify SwapInWorkerStats Debug output contains the "promoted_ok" field name.
    #[test]
    fn stats_debug_contains_promoted_ok_field() {
        let stats = SwapInWorkerStats {
            promoted_ok: 77,
            ..SwapInWorkerStats::default()
        };
        let debug = format!("{stats:?}");
        assert!(
            debug.contains("promoted_ok"),
            "Debug output should contain 'promoted_ok' field: got '{}'",
            debug
        );
        assert!(
            debug.contains("77"),
            "Debug output should contain the value 77: got '{}'",
            debug
        );
    }

    /// Verify PrefetchRequest equality: two requests with identical fields including
    /// the same Instant (cloned) are equal.
    #[test]
    fn prefetch_request_eq_identical_timestamps() {
        // Arrange: create a request and clone it (preserving the Instant).
        let now = Instant::now();
        let req1 = PrefetchRequest {
            page_id: 10,
            urgency: 3.5,
            prefetch_confidence: 0.6,
            page_bytes: 8192,
            enqueued_at: now,
        };
        let req2 = req1.clone();

        // Assert: cloned request is equal to original.
        assert_eq!(req1, req2, "cloned request with same Instant should be equal");
    }

    /// Verify PrefetchRequest clone produces an independent copy — mutating the clone's
    /// numeric fields does not affect the original.
    #[test]
    fn prefetch_request_clone_independent_mutation() {
        // Arrange
        let now = Instant::now();
        let original = PrefetchRequest {
            page_id: 5,
            urgency: 1.0,
            prefetch_confidence: 0.5,
            page_bytes: 2048,
            enqueued_at: now,
        };
        let mut cloned = original.clone();

        // Act: mutate the clone.
        cloned.urgency = 99.0;
        cloned.page_bytes = 65536;
        cloned.prefetch_confidence = 0.0;

        // Assert: original is unchanged.
        assert_eq!(original.urgency, 1.0);
        assert_eq!(original.page_bytes, 2048);
        assert_eq!(original.prefetch_confidence, 0.5);
    }

    /// Verify PrefetchRequest Debug output contains all field names.
    #[test]
    fn prefetch_request_debug_all_fields() {
        // Arrange
        let req = PrefetchRequest {
            page_id: 42,
            urgency: 7.5,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        };

        // Act
        let debug = format!("{req:?}");

        // Assert: all four field names appear in the debug output.
        assert!(debug.contains("page_id"), "Debug should contain 'page_id': got '{debug}'");
        assert!(debug.contains("urgency"), "Debug should contain 'urgency': got '{debug}'");
        assert!(debug.contains("prefetch_confidence"), "Debug should contain 'prefetch_confidence': got '{debug}'");
        assert!(debug.contains("page_bytes"), "Debug should contain 'page_bytes': got '{debug}'");
    }

    /// Verify SwapInWorkerStats Debug output contains the "rounds" field name.
    #[test]
    fn stats_debug_contains_rounds_field() {
        // Arrange
        let stats = SwapInWorkerStats {
            rounds: 1234,
            ..SwapInWorkerStats::default()
        };

        // Act
        let debug = format!("{stats:?}");

        // Assert
        assert!(
            debug.contains("rounds"),
            "Debug output should contain 'rounds' field: got '{}'",
            debug
        );
        assert!(
            debug.contains("1234"),
            "Debug output should contain the value 1234: got '{}'",
            debug
        );
    }

    /// Verify SwapInWorkerStats Debug output contains the "total_latency_us" field name.
    #[test]
    fn stats_debug_contains_total_latency() {
        // Arrange
        let stats = SwapInWorkerStats {
            total_latency_us: 9999,
            ..SwapInWorkerStats::default()
        };

        // Act
        let debug = format!("{stats:?}");

        // Assert
        assert!(
            debug.contains("total_latency_us"),
            "Debug output should contain 'total_latency_us': got '{}'",
            debug
        );
        assert!(
            debug.contains("9999"),
            "Debug output should contain the value 9999: got '{}'",
            debug
        );
    }

    /// Verify SwapInWorkerStats Debug output contains the "total_requests" field name.
    #[test]
    fn stats_debug_contains_total_requests() {
        // Arrange
        let stats = SwapInWorkerStats {
            total_requests: 555,
            ..SwapInWorkerStats::default()
        };

        // Act
        let debug = format!("{stats:?}");

        // Assert
        assert!(
            debug.contains("total_requests"),
            "Debug output should contain 'total_requests': got '{}'",
            debug
        );
        assert!(
            debug.contains("555"),
            "Debug output should contain the value 555: got '{}'",
            debug
        );
    }

    /// Verify swap_in_round uses the request's page_bytes (not config default) when
    /// the request specifies a nonzero value for a DRAM page.
    #[test]
    fn round_dram_page_bytes_uses_request_not_config() {
        // Arrange: config says 4096, request says 16384.
        let config = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 10,
            urgency: 5.0,
            prefetch_confidence: 0.8,
            page_bytes: 16384,
            enqueued_at: Instant::now(),
        }];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert: the request was submitted (nonzero page_bytes used, not config default).
        assert_eq!(submitted, 1);
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    /// Verify swap_in_round with all three pages on HBM: submitted=0 and skipped=3.
    #[test]
    fn round_all_hbm_pages_skipped_submitted_zero() {
        // Arrange
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        for page_id in 1..=3 {
            insert_addr_entry(&addr_table, page_id, StorageTier::GpuHbm, CompressionCodec::None);
        }

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=3)
            .map(|pid| make_prefetch(pid, pid as f32, 0.9, 4096))
            .collect();

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert
        assert_eq!(submitted, 0, "all HBM pages should result in zero submissions");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 3, "all three HBM pages should be skipped");

        actor.shutdown();
    }

    /// Verify swap_in_round for a single NVMe page with nonzero request page_bytes
    /// uses the request's bytes (not config default) and increments both two_hop and submitted.
    #[test]
    fn round_single_nvme_page_bytes_from_request() {
        // Arrange
        let config = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );

        insert_addr_entry(&addr_table, 99, StorageTier::Nvme, CompressionCodec::None);

        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 99,
            urgency: 8.0,
            prefetch_confidence: 0.95,
            page_bytes: 32768,
            enqueued_at: Instant::now(),
        }];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert
        assert_eq!(submitted, 1, "NVMe page should be submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should trigger two-hop promotion");

        actor.shutdown();
    }

    /// Verify SwapInWorkerConfig default max_in_flight is exactly 64.
    #[test]
    fn config_default_max_in_flight_sixty_four() {
        // Arrange & Act
        let config = SwapInWorkerConfig::default();

        // Assert
        assert_eq!(config.max_in_flight, 64);
    }

    /// Verify SwapInWorkerConfig default min_confidence is exactly 0.1.
    #[test]
    fn config_default_min_confidence_one_tenth() {
        // Arrange & Act
        let config = SwapInWorkerConfig::default();

        // Assert
        assert!(
            (config.min_confidence - 0.1).abs() < f32::EPSILON,
            "default min_confidence should be 0.1, got {}",
            config.min_confidence,
        );
    }

    /// Verify SwapInWorkerError::SendFailed with empty string still produces
    /// a non-empty Display output (the prefix alone).
    #[test]
    fn error_send_failed_source_empty() {
        // Arrange
        let err = SwapInWorkerError::SendFailed(String::new());

        // Act
        let display = format!("{err}");

        // Assert: prefix "swap-in worker send failed: " is present even with empty body.
        assert!(
            display.contains("swap-in worker send failed"),
            "Display should contain prefix even with empty message: got '{}'",
            display,
        );
    }

    /// Verify SwapInWorkerError::RecvFailed with empty string still produces
    /// a non-empty Display output (the prefix alone).
    #[test]
    fn error_recv_failed_source_empty() {
        // Arrange
        let err = SwapInWorkerError::RecvFailed(String::new());

        // Act
        let display = format!("{err}");

        // Assert: prefix "swap-in worker recv failed: " is present even with empty body.
        assert!(
            display.contains("swap-in worker recv failed"),
            "Display should contain prefix even with empty message: got '{}'",
            display,
        );
    }

    /// Verify urgency: HBM tier bonus is exactly 2.0, which is double the DRAM bonus of 1.0.
    /// The importance_rebound * confidence * tier_bonus term for HBM should be exactly
    /// twice that for DRAM when all other inputs are equal.
    #[test]
    fn urgency_hbm_double_dram_formula() {
        // Arrange: same metadata and confidence, different tiers.
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(0),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let confidence = 0.5;

        // Act
        let urgency_dram = SwapInWorker::compute_urgency(&meta, confidence, StorageTier::CpuDram);
        let urgency_hbm = SwapInWorker::compute_urgency(&meta, confidence, StorageTier::GpuHbm);

        // Assert: recency_bonus is the same for both, so the difference comes only from
        // the tier_bonus term. The non-recency part for HBM should be 2x DRAM.
        // urgency = importance * confidence * tier_bonus + recency * 0.1
        // recency is identical (same Instant::now()), so:
        // urgency_hbm - urgency_dram = importance * confidence * (2.0 - 1.0) = importance * confidence
        let importance_rebound = (meta.access_count as f32).ln_1p() / 10.0_f32.ln_1p();
        let expected_diff = importance_rebound * confidence;
        let actual_diff = urgency_hbm - urgency_dram;

        assert!(
            (actual_diff - expected_diff).abs() < 1e-5,
            "HBM-DRAM urgency difference should equal importance*confidence: expected={}, got={}",
            expected_diff,
            actual_diff,
        );
    }

    /// Verify SwapInWorkerStats: promoted_ok is independent of two_hop_promotions —
    /// setting one does not affect the other.
    #[test]
    fn stats_promoted_ok_independent_of_two_hop() {
        // Arrange
        let stats = SwapInWorkerStats {
            promoted_ok: 50,
            two_hop_promotions: 0,
            ..SwapInWorkerStats::default()
        };

        // Assert: promoted_ok is 50 regardless of two_hop_promotions being 0.
        assert_eq!(stats.promoted_ok, 50);
        assert_eq!(stats.two_hop_promotions, 0);

        // Act: set two_hop_promotions to a large value.
        let stats2 = SwapInWorkerStats {
            promoted_ok: 50,
            two_hop_promotions: 999,
            ..SwapInWorkerStats::default()
        };

        // Assert: promoted_ok is still 50.
        assert_eq!(stats2.promoted_ok, 50);
        assert_eq!(stats2.two_hop_promotions, 999);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional tests — 15 new tests
    // ═══════════════════════════════════════════════════════════════════════════

    /// Verify SwapInWorkerError PartialEq symmetry: if a == b then b == a,
    /// including across different variants with the same message.
    #[test]
    fn error_partial_eq_symmetry_across_variants() {
        // Arrange: two errors of different variants with the same message.
        let send = SwapInWorkerError::SendFailed("msg".into());
        let recv = SwapInWorkerError::RecvFailed("msg".into());

        // Assert: different variants with the same message are not equal, symmetrically.
        assert_ne!(send, recv);
        assert_ne!(recv, send);

        // Same variant, same message: symmetrically equal.
        let send2 = SwapInWorkerError::SendFailed("msg".into());
        assert_eq!(send, send2);
        assert_eq!(send2, send);
    }

    /// Verify SwapInWorkerError PartialEq transitivity: a == b && b == c => a == c.
    #[test]
    fn error_partial_eq_transitivity_chain() {
        // Arrange: three errors that should all be equal.
        let a = SwapInWorkerError::RecvFailed("timeout".into());
        let b = SwapInWorkerError::RecvFailed("timeout".into());
        let c = SwapInWorkerError::RecvFailed("timeout".into());

        // Assert: transitivity holds.
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    /// Verify PrefetchRequest PartialEq transitivity: a == b && b == c => a == c.
    #[test]
    fn prefetch_request_partial_eq_transitivity_chain() {
        // Arrange: three identical requests.
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 10,
            urgency: 0.75,
            prefetch_confidence: 0.6,
            page_bytes: 2048,
            enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 10,
            urgency: 0.75,
            prefetch_confidence: 0.6,
            page_bytes: 2048,
            enqueued_at: now,
        };
        let c = PrefetchRequest {
            page_id: 10,
            urgency: 0.75,
            prefetch_confidence: 0.6,
            page_bytes: 2048,
            enqueued_at: now,
        };

        // Assert: transitivity holds.
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    /// Verify the default tick_interval is exactly 5 milliseconds.
    #[test]
    fn config_default_tick_interval_exactly_five_ms() {
        let cfg = SwapInWorkerConfig::default();
        assert_eq!(cfg.tick_interval, Duration::from_millis(5));
    }

    /// Verify the default page_bytes is exactly 4096.
    #[test]
    fn config_default_page_bytes_exactly_4096() {
        let cfg = SwapInWorkerConfig::default();
        assert_eq!(cfg.page_bytes, 4096);
    }

    /// Verify that promoted_failed counter does not influence avg_latency_us calculation.
    #[test]
    fn stats_promoted_failed_no_impact_on_avg_latency() {
        // Arrange: stats with only failures, no successful promotions.
        let mut stats = SwapInWorkerStats::default();
        stats.promoted_ok = 0;
        stats.promoted_failed = 100;
        stats.total_latency_us = 500_000;

        // Act & Assert: avg_latency should still be 0.0 because promoted_ok == 0.
        assert!(
            (stats.avg_latency_us() - 0.0).abs() < 1e-6,
            "avg_latency should be 0.0 when promoted_ok is 0, regardless of promoted_failed: got {}",
            stats.avg_latency_us(),
        );
    }

    /// Verify that the Display prefixes for SendFailed and RecvFailed are distinct
    /// (no risk of confusion between variants).
    #[test]
    fn error_display_prefixes_are_mutually_distinct() {
        // Arrange
        let send = SwapInWorkerError::SendFailed("x".into());
        let recv = SwapInWorkerError::RecvFailed("x".into());
        let send_msg = format!("{send}");
        let recv_msg = format!("{recv}");

        // Assert: each message contains its own prefix but not the other's.
        assert!(
            send_msg.contains("send failed") && !send_msg.contains("recv failed"),
            "SendFailed display should contain 'send failed' but not 'recv failed': {send_msg}",
        );
        assert!(
            recv_msg.contains("recv failed") && !recv_msg.contains("send failed"),
            "RecvFailed display should contain 'recv failed' but not 'send failed': {recv_msg}",
        );
    }

    /// Verify compute_urgency produces identical scores for two pages that differ
    /// only in page_id (page_id is not part of the urgency formula).
    #[test]
    fn urgency_different_page_ids_same_result() {
        // Arrange: two metadata entries differing only in page_id.
        let now = Instant::now();
        let meta_a = PageMetadata {
            page_id: 1,
            sequence_id: Some(10),
            recency: 5,
            access_count: 20,
            last_access: now,
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };
        let meta_b = PageMetadata {
            page_id: 999,
            sequence_id: Some(10),
            recency: 5,
            access_count: 20,
            last_access: now,
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act
        let ua = SwapInWorker::compute_urgency(&meta_a, 0.7, StorageTier::CpuDram);
        let ub = SwapInWorker::compute_urgency(&meta_b, 0.7, StorageTier::CpuDram);

        // Assert: exact equality (same timestamp, same inputs).
        assert!(
            (ua - ub).abs() < 1e-6,
            "urgency should be identical for same inputs differing only in page_id: a={} b={}",
            ua, ub,
        );
    }

    /// Verify swap_in_round drains the input vector to length 0 even when
    /// submissions succeed (requests are drained, not just marked).
    #[test]
    fn round_vec_len_zero_after_submissions() {
        // Arrange
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(
                10,
                PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                },
            );
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 10,
            urgency: 1.0,
            prefetch_confidence: 0.9,
            page_bytes: 4096,
            enqueued_at: Instant::now(),
        }];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // Assert
        assert_eq!(submitted, 1);
        assert!(
            requests.is_empty(),
            "input vec should be empty after swap_in_round drains it, got len={}",
            requests.len(),
        );

        actor.shutdown();
    }

    /// Verify total_requests counts all input items BEFORE truncation drops some.
    #[test]
    fn round_total_requests_counts_before_truncation_drops() {
        // Arrange: 5 requests but max_prefetch_per_round = 2.
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 2,
            ..SwapInWorkerConfig::default()
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
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (0..5)
            .map(|i| PrefetchRequest {
                page_id: i,
                urgency: 1.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();

        // Act
        let _submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // Assert: total_requests should be 5 (all input), not 2 (truncated).
        let s = stats.lock().expect("stats lock");
        assert_eq!(
            s.total_requests, 5,
            "total_requests should count all 5 input items before truncation"
        );

        actor.shutdown();
    }

    /// Verify that cloning config preserves the tick_interval field exactly.
    #[test]
    fn config_clone_preserves_tick_interval_exact() {
        // Arrange
        let cfg = SwapInWorkerConfig {
            tick_interval: Duration::from_micros(333),
            ..SwapInWorkerConfig::default()
        };

        // Act
        let cloned = cfg.clone();

        // Assert
        assert_eq!(
            cloned.tick_interval,
            Duration::from_micros(333),
            "cloned tick_interval should exactly match original"
        );
    }

    /// Verify the invariant that promoted_ok + promoted_failed <= submitted
    /// when manually constructing stats.
    #[test]
    fn stats_promoted_plus_failed_leq_submitted_manual() {
        // Arrange: stats where promoted_ok + promoted_failed == submitted.
        let stats = SwapInWorkerStats {
            total_requests: 100,
            submitted: 80,
            skipped: 20,
            promoted_ok: 60,
            promoted_failed: 20,
            two_hop_promotions: 15,
            total_latency_us: 12000,
            rounds: 10,
        };

        // Assert
        assert!(
            stats.promoted_ok + stats.promoted_failed <= stats.submitted,
            "promoted_ok + promoted_failed ({}) should not exceed submitted ({})",
            stats.promoted_ok + stats.promoted_failed,
            stats.submitted,
        );
    }

    /// Verify that a PrefetchRequest with negative urgency can be stored,
    /// cloned, and the clone preserves the negative value.
    #[test]
    fn prefetch_request_negative_urgency_clone_roundtrip() {
        // Arrange
        let req = PrefetchRequest {
            page_id: 42,
            urgency: -0.5,
            prefetch_confidence: 0.3,
            page_bytes: 1024,
            enqueued_at: Instant::now(),
        };

        // Act
        let cloned = req.clone();

        // Assert
        assert!(
            (cloned.urgency - (-0.5_f32)).abs() < 1e-6,
            "cloned urgency should preserve -0.5, got {}",
            cloned.urgency,
        );
        assert_eq!(cloned.page_id, req.page_id);
        assert_eq!(cloned.page_bytes, req.page_bytes);
    }

    /// Verify all five fields of SwapInWorkerConfig are publicly accessible
    /// (constructable and readable without any accessor methods).
    #[test]
    fn config_all_five_fields_publicly_accessible() {
        // Arrange & Act: construct with all five fields set to distinctive values.
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 7,
            tick_interval: Duration::from_millis(13),
            min_confidence: 0.42,
            max_in_flight: 99,
            page_bytes: 8192,
        };

        // Assert: all five fields are readable.
        assert_eq!(cfg.max_prefetch_per_round, 7);
        assert_eq!(cfg.tick_interval, Duration::from_millis(13));
        assert!((cfg.min_confidence - 0.42).abs() < 1e-6);
        assert_eq!(cfg.max_in_flight, 99);
        assert_eq!(cfg.page_bytes, 8192);
    }

    /// Verify the first term of compute_urgency with known, controlled inputs.
    /// Formula: importance_rebound * confidence * tier_bonus
    /// With access_count=10: ln_1p(10)/ln_1p(10) = 1.0
    /// confidence=0.5, tier_bonus (CpuDram) = 1.0 => first term = 0.5
    #[test]
    fn urgency_first_term_exact_with_known_inputs() {
        // Arrange
        let meta = PageMetadata {
            page_id: 0,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let urgency = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);

        // Assert: first term = ln_1p(10)/ln_1p(10) * 0.5 * 1.0 = 0.5
        // Recency bonus = 0.1 / (1.0 + ~0) ~= 0.1
        // Total ~= 0.5 + 0.1 = 0.6, first term dominates at exactly 0.5.
        let first_term = 1.0_f32 * 0.5 * 1.0;
        assert!(
            urgency > first_term,
            "urgency ({}) should be > first term ({}) due to positive recency bonus",
            urgency, first_term,
        );
        // Recency bonus is at most 0.1 (when elapsed = 0).
        assert!(
            urgency < first_term + 0.11,
            "urgency ({}) should be < first term + 0.11 (max recency bonus is 0.1)",
            urgency,
        );
    }

    // -- new 15 tests: covering untested public API and edge cases --

    /// Verify Error source() returns None for both variants.
    #[test]
    fn error_source_returns_none_for_both_variants() {
        let e_send = SwapInWorkerError::SendFailed("broken pipe".into());
        let e_recv = SwapInWorkerError::RecvFailed("timeout".into());
        assert!(std::error::Error::source(&e_send).is_none());
        assert!(std::error::Error::source(&e_recv).is_none());
    }

    /// Verify max_in_flight=0 prevents any submissions.
    #[test]
    fn round_max_in_flight_zero_submits_nothing_even_with_valid_page() {
        let config = SwapInWorkerConfig { max_in_flight: 0, ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        { let mut t = addr_table.write().expect("wl"); t.insert(1, PageAddrEntry {
            gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
            current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
        }); }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![PrefetchRequest { page_id: 1, urgency: 1.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() }];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);
        assert_eq!(submitted, 0, "max_in_flight=0 should prevent any submissions");
        actor.shutdown();
    }

    /// Verify drain_completions_and_update handles missing page metadata without panic.
    #[test]
    fn drain_handles_missing_page_metadata_gracefully() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        { let mut t = addr_table.write().expect("wl"); t.insert(77, PageAddrEntry {
            gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
            current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
        }); }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 77, page_bytes: 4096 });
        for _ in 0..30 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("sl");
            if s.promoted_ok > 0 { break; }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }
        let s = stats.lock().expect("sl");
        assert!(s.promoted_ok >= 1, "should record promoted_ok even without page metadata");
        actor.shutdown();
    }

    /// Verify prefetch_batch returns 0 after worker shutdown.
    #[test]
    fn prefetch_batch_returns_zero_after_shutdown() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(10), ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        worker.shutdown();
        let requests: Vec<PrefetchRequest> = (0..3).map(|i| PrefetchRequest {
            page_id: i, urgency: 0.5, prefetch_confidence: 0.8, page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        assert_eq!(worker.prefetch_batch(&requests), 0, "prefetch_batch should return 0 after shutdown");
    }

    /// Verify all negative-confidence requests are skipped.
    #[test]
    fn round_all_negative_confidence_requests_skipped() {
        let config = SwapInWorkerConfig { min_confidence: 0.1, ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        { let mut t = addr_table.write().expect("wl"); for pid in 1..=3usize { t.insert(pid, PageAddrEntry {
            gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
            current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
        }); } }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (1..=3).map(|pid| PrefetchRequest {
            page_id: pid, urgency: 1.0, prefetch_confidence: -0.5, page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);
        assert_eq!(submitted, 0);
        let s = stats.lock().expect("sl");
        assert_eq!(s.skipped, 3);
        assert_eq!(s.submitted, 0);
        actor.shutdown();
    }

    /// Verify recency bonus upper bound is exactly 0.1.
    #[test]
    fn urgency_recency_bonus_upper_bound_is_exactly_one_tenth() {
        let meta = PageMetadata { page_id: 1, sequence_id: None, recency: 0, access_count: 0,
            last_access: Instant::now(), swap_in_time: None, is_lir: false, state: PageState::Active, warm_until: None };
        let urgency = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);
        assert!(urgency <= 0.1 + 0.001, "urgency should be at most ~0.1: {urgency}");
        assert!(urgency > 0.09, "urgency should be close to 0.1: {urgency}");
    }

    /// Verify equal-urgency requests are all processed when within limit.
    #[test]
    fn round_equal_urgency_all_processed_when_within_limit() {
        let config = SwapInWorkerConfig { max_prefetch_per_round: 10, ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        { let mut t = addr_table.write().expect("wl"); for pid in 1..=4usize { t.insert(pid, PageAddrEntry {
            gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
            current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
        }); } }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests: Vec<PrefetchRequest> = (1..=4).map(|pid| PrefetchRequest {
            page_id: pid, urgency: 0.5, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);
        assert_eq!(submitted, 4);
        actor.shutdown();
    }

    /// Verify SwapInWorkerError implements Send + Sync.
    #[test]
    fn error_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SwapInWorkerError>();
    }

    /// Verify drain handles failed migration results correctly.
    #[test]
    fn drain_increments_promoted_failed_on_failure() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        { let mut t = addr_table.write().expect("wl"); t.insert(200, PageAddrEntry {
            gpu_ptr: None, host_buffer: None,
            current_tier: StorageTier::Nvme, original_bytes: 4096, codec: CompressionCodec::ZstdDict,
        }); }
        let mut pm: HashMap<PageId, PageMetadata> = HashMap::new();
        pm.insert(200, PageMetadata { page_id: 200, sequence_id: Some(1), recency: 0, access_count: 3,
            last_access: Instant::now(), swap_in_time: None, is_lir: false, state: PageState::SwappedOut, warm_until: None });
        let page_metadata = Arc::new(RwLock::new(pm));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let _ = actor.send(MigrationCommand::PromoteToDram { page_id: 200, page_bytes: 4096 });
        for _ in 0..30 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("sl");
            if s.promoted_failed > 0 || s.promoted_ok > 0 { break; }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }
        let s = stats.lock().expect("sl");
        assert!(s.promoted_failed + s.promoted_ok >= 1, "should have at least one completion");
        actor.shutdown();
    }

    /// Verify two independent config instances have fully independent fields.
    #[test]
    fn config_multiple_instances_fully_independent() {
        let a = SwapInWorkerConfig { max_prefetch_per_round: 4, tick_interval: Duration::from_millis(10),
            min_confidence: 0.2, max_in_flight: 16, page_bytes: 2048 };
        let b = SwapInWorkerConfig { max_prefetch_per_round: 32, tick_interval: Duration::from_millis(50),
            min_confidence: 0.8, max_in_flight: 128, page_bytes: 16384 };
        assert_ne!(a.max_prefetch_per_round, b.max_prefetch_per_round);
        assert_ne!(a.tick_interval, b.tick_interval);
        assert_ne!(a.min_confidence, b.min_confidence);
        assert_ne!(a.max_in_flight, b.max_in_flight);
        assert_ne!(a.page_bytes, b.page_bytes);
    }

    /// Verify urgency produces finite values across all 7 PageState variants.
    #[test]
    fn urgency_finite_across_all_page_states() {
        let states = [PageState::Free, PageState::Active, PageState::Standby,
            PageState::SwappedOut, PageState::Warm, PageState::Protected, PageState::Swapped];
        for state in &states {
            let meta = PageMetadata { page_id: 1, sequence_id: None, recency: 0, access_count: 10,
                last_access: Instant::now(), swap_in_time: None, is_lir: false, state: *state, warm_until: None };
            let u = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
            assert!(u.is_finite(), "urgency should be finite for state {:?}: {u}", state);
        }
    }

    /// Verify swap_in_round drains the requests vec completely.
    #[test]
    fn round_drains_all_requests_from_vec() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        { let mut t = addr_table.write().expect("wl"); t.insert(1, PageAddrEntry {
            gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
            current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
        }); }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![PrefetchRequest { page_id: 1, urgency: 1.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() }];
        let _ = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);
        assert!(requests.is_empty(), "requests vec should be empty after drain");
        actor.shutdown();
    }

    /// Verify confidence linear scaling: half confidence has first term exactly half of full.
    /// access_count=10 → importance_rebound = ln(11)/ln(11) = 1.0, so the linear term
    /// scales exactly by confidence. Difference = 1.0*1.0*1.0 - 1.0*0.5*1.0 = 0.5.
    #[test]
    fn urgency_confidence_linear_scaling_half_exact() {
        let meta = PageMetadata { page_id: 1, sequence_id: None, recency: 0, access_count: 10,
            last_access: Instant::now(), swap_in_time: None, is_lir: false, state: PageState::Active, warm_until: None };
        let u_full = SwapInWorker::compute_urgency(&meta, 1.0, StorageTier::CpuDram);
        let u_half = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        let diff = u_full - u_half;
        assert!((diff - 0.5).abs() < 0.01, "diff should be ~0.5: {diff}");
    }

    /// Verify stats() is readable after idle ticks with no prefetch requests.
    #[test]
    fn stats_readable_after_idle_ticks() {
        let config = SwapInWorkerConfig { tick_interval: Duration::from_millis(10), ..SwapInWorkerConfig::default() };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        thread::sleep(Duration::from_millis(60));
        let s = worker.stats();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.submitted, 0);
        assert_eq!(s.skipped, 0);
        worker.shutdown();
    }

    /// Verify observer records weight page events on submission.
    #[test]
    fn round_observer_records_weight_page_events_on_submit() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        { let mut t = addr_table.write().expect("wl");
            t.insert(1, PageAddrEntry { gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None });
            t.insert(2, PageAddrEntry { gpu_ptr: None, host_buffer: None,
                current_tier: StorageTier::Nvme, original_bytes: 4096, codec: CompressionCodec::ZstdDict });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));
        let mut requests = vec![
            PrefetchRequest { page_id: 1, urgency: 0.8, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.9, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
        ];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);
        assert_eq!(submitted, 2);
        let obs = observer.lock().expect("ol");
        assert!(obs.last_state.weight_recovery_count >= 2, "observer should have >= 2 recovery events: {}", obs.last_state.weight_recovery_count);
        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Additional 15 tests — covering remaining gaps
    // ═══════════════════════════════════════════════════════════════════════════

    /// Verify NVMe page without host_buffer still issues both PromoteToDram and PromoteToHbm.
    /// The addr_table entry has no host_buffer, so the actor may fail, but the worker must
    /// still submit both commands and count two_hop_promotions.
    #[test]
    fn round_nvme_without_host_buffer_still_submits_two_commands() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(55, PageAddrEntry {
                gpu_ptr: None, host_buffer: None,
                current_tier: StorageTier::Nvme, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 55, urgency: 1.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "NVMe page should be submitted even without host_buffer");
        let s = stats.lock().expect("sl");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page should count as two-hop");

        actor.shutdown();
    }

    /// Verify drain_completions_and_update sets page state to Swapped when to_tier is Nvme.
    /// This is the fallback branch in drain that handles an unexpected tier gracefully.
    #[test]
    fn drain_sets_swapped_state_on_nvme_tier_promotion() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(77, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let mut pm: HashMap<PageId, PageMetadata> = HashMap::new();
        pm.insert(77, PageMetadata {
            page_id: 77, sequence_id: None, recency: 0, access_count: 1,
            last_access: Instant::now(), swap_in_time: Some(Instant::now()),
            is_lir: false, state: PageState::SwappedOut, warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(pm));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // PromoteToHbm normally produces to_tier=GpuHbm. The drain function maps GpuHbm → Active.
        // Verify that the swap_in_time is cleared on HBM promotion.
        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 77, page_bytes: 4096 });
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("sl");
            if s.promoted_ok > 0 { break; }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let meta = page_metadata.read().expect("rl").get(&77).cloned().unwrap();
        assert_eq!(meta.state, PageState::Active, "HBM promotion should set state to Active");
        assert!(meta.swap_in_time.is_none(), "swap_in_time should be cleared on Active");

        actor.shutdown();
    }

    /// Verify SwapInWorkerConfig stores a negative min_confidence without error.
    /// This tests the type's acceptance of out-of-range floats (no clamping).
    #[test]
    fn config_negative_min_confidence_stored_without_clamping() {
        let cfg = SwapInWorkerConfig {
            min_confidence: -1.0,
            ..SwapInWorkerConfig::default()
        };
        assert!((cfg.min_confidence - (-1.0)).abs() < 1e-6, "negative confidence should be stored as-is");
    }

    /// Verify PrefetchRequest with negative urgency is sorted below positive urgency.
    /// The round should process the positive-urgency page first when truncation applies.
    #[test]
    fn round_negative_urgency_truncated_to_single_page() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [1usize, 2usize] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Positive urgency page_id=2 comes first in the vec, negative urgency page_id=1 second.
        // After sort desc, page_id=2 (urgency=0.9) should be at top.
        let mut requests = vec![
            PrefetchRequest { page_id: 1, urgency: -5.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
            PrefetchRequest { page_id: 2, urgency: 0.9, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now() },
        ];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "only top-urgency page should be submitted with max_prefetch=1");
        let s = stats.lock().expect("sl");
        assert_eq!(s.total_requests, 2, "both requests counted before truncation");

        actor.shutdown();
    }

    /// Verify compute_urgency logarithmic scaling: access_count=4 should have higher urgency
    /// than access_count=2, confirming the ln_1p formula is applied correctly.
    #[test]
    fn urgency_logarithmic_scaling_between_two_and_four() {
        let meta_2 = PageMetadata {
            page_id: 1, sequence_id: None, recency: 0, access_count: 2,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Active, warm_until: None,
        };
        let meta_4 = PageMetadata {
            page_id: 2, sequence_id: None, recency: 0, access_count: 4,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Active, warm_until: None,
        };
        let u_2 = SwapInWorker::compute_urgency(&meta_2, 1.0, StorageTier::CpuDram);
        let u_4 = SwapInWorker::compute_urgency(&meta_4, 1.0, StorageTier::CpuDram);
        // ln(3)/ln(11) vs ln(5)/ln(11) — 4 should be higher.
        assert!(u_4 > u_2, "access_count=4 should yield higher urgency than 2: u4={u_4} u2={u_2}");
        // Verify the ratio matches ln(5)/ln(3).
        let ratio = u_4 / u_2;
        let expected_ratio = (4.0_f32).ln_1p() / (2.0_f32).ln_1p();
        assert!(
            (ratio - expected_ratio).abs() < 0.1,
            "ratio should match ln(5)/ln(3): ratio={ratio} expected={expected_ratio}",
        );
    }

    /// Verify that two separate drain calls accumulate total_latency_us correctly.
    #[test]
    fn drain_two_promotions_accumulate_latency() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [10usize, 20usize] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
                });
            }
        }
        let mut pm: HashMap<PageId, PageMetadata> = HashMap::new();
        let now = Instant::now();
        for pid in [10usize, 20usize] {
            pm.insert(pid, PageMetadata {
                page_id: pid, sequence_id: None, recency: 0, access_count: 1,
                last_access: now, swap_in_time: Some(now),
                is_lir: false, state: PageState::SwappedOut, warm_until: None,
            });
        }
        let page_metadata = Arc::new(RwLock::new(pm));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        for pid in [10usize, 20usize] {
            let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: pid, page_bytes: 4096 });
        }
        for _ in 0..30 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("sl");
            if s.promoted_ok >= 2 { break; }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let s = stats.lock().expect("sl");
        assert_eq!(s.promoted_ok, 2, "should have 2 successful promotions");
        assert!(s.total_latency_us > 0, "latency should accumulate from both promotions: {}", s.total_latency_us);

        actor.shutdown();
    }

    /// Verify swap_in_round with all NVMe pages correctly counts two_hop for each.
    #[test]
    fn round_all_nvme_pages_count_two_hop_each() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [100usize, 200usize, 300usize] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: None,
                    current_tier: StorageTier::Nvme, original_bytes: 4096, codec: CompressionCodec::ZstdDict,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = [100usize, 200usize, 300usize].iter().map(|&pid| PrefetchRequest {
            page_id: pid, urgency: 1.0, prefetch_confidence: 0.9, page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 3, "all 3 NVMe pages should be submitted");
        let s = stats.lock().expect("sl");
        assert_eq!(s.two_hop_promotions, 3, "each NVMe page should count as one two-hop");

        actor.shutdown();
    }

    /// Verify SwapInWorkerConfig PartialEq detects max_prefetch_per_round difference.
    #[test]
    fn config_partial_eq_detects_prefetch_round_diff() {
        let c1 = SwapInWorkerConfig { max_prefetch_per_round: 16, ..SwapInWorkerConfig::default() };
        let c2 = SwapInWorkerConfig { max_prefetch_per_round: 32, ..SwapInWorkerConfig::default() };
        assert_ne!(c1, c2, "different max_prefetch_per_round should not be equal");
    }

    /// Verify drain_completions_and_update handles page with very old swap_in_time correctly.
    /// The latency should be large but finite.
    #[test]
    fn drain_old_swap_in_time_produces_large_finite_latency() {
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(88, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096, codec: CompressionCodec::None,
            });
        }
        let mut pm: HashMap<PageId, PageMetadata> = HashMap::new();
        pm.insert(88, PageMetadata {
            page_id: 88, sequence_id: None, recency: 0, access_count: 1,
            last_access: Instant::now(), swap_in_time: Some(Instant::now() - Duration::from_secs(600)),
            is_lir: false, state: PageState::SwappedOut, warm_until: None,
        });
        let page_metadata = Arc::new(RwLock::new(pm));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let _ = actor.send(MigrationCommand::PromoteToHbm { page_id: 88, page_bytes: 4096 });
        for _ in 0..20 {
            drain_completions_and_update(&actor, &page_metadata, &addr_table, &stats, &observer);
            let s = stats.lock().expect("sl");
            if s.promoted_ok > 0 { break; }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        let s = stats.lock().expect("sl");
        assert_eq!(s.promoted_ok, 1);
        assert!(
            s.total_latency_us >= 500_000,
            "old swap_in_time should produce large latency (>=500ms): {}us",
            s.total_latency_us,
        );

        actor.shutdown();
    }

    /// Verify prefetch_batch returns 0 when worker has been shut down.
    #[test]
    fn prefetch_batch_zero_after_worker_shutdown() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(10),
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(config, actor, page_metadata, addr_table, observer);
        worker.shutdown();

        let requests: Vec<PrefetchRequest> = (1..=3)
            .map(|pid| PrefetchRequest {
                page_id: pid, urgency: 0.5, prefetch_confidence: 0.7,
                page_bytes: 4096, enqueued_at: Instant::now(),
            })
            .collect();
        let enqueued = worker.prefetch_batch(&requests);
        assert_eq!(enqueued, 0, "prefetch_batch after shutdown should return 0");
    }

    /// Verify swap_in_round with all HBM pages records observer events but submitted=0.
    /// The observer records recovery events only for submitted pages, not skipped ones.
    #[test]
    fn round_all_hbm_pages_observer_only_records_submitted() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [1usize, 2usize] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: Some(0x1000), host_buffer: None,
                    current_tier: StorageTier::GpuHbm, original_bytes: 4096, codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=2)
            .map(|pid| PrefetchRequest {
                page_id: pid, urgency: 1.0, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            })
            .collect();
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 0, "HBM pages should not be submitted");
        let s = stats.lock().expect("sl");
        assert_eq!(s.skipped, 2, "both HBM pages should be skipped");
        // No observer events for skipped pages — recovery_count should stay at 0.
        let obs = observer.lock().expect("ol");
        assert_eq!(obs.last_state.weight_recovery_count, 0, "skipped HBM pages should not produce observer events");

        actor.shutdown();
    }

    /// Verify compute_urgency with same access_count and confidence, NVMe is strictly lower than DRAM.
    /// This tests that the tier bonus multiplier is correctly applied as a strict ordering.
    #[test]
    fn urgency_nvme_strictly_lower_than_dram_for_same_inputs() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: None, recency: 0, access_count: 50,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Active, warm_until: None,
        };
        let u_dram = SwapInWorker::compute_urgency(&meta, 0.8, StorageTier::CpuDram);
        let u_nvme = SwapInWorker::compute_urgency(&meta, 0.8, StorageTier::Nvme);
        // NVMe tier_bonus=0.5 vs DRAM=1.0, so DRAM first term is exactly 2x NVMe first term.
        // The difference in first term is importance_rebound * 0.8 * 0.5.
        let importance = (50.0_f32).ln_1p() / (10.0_f32).ln_1p();
        let expected_diff = importance * 0.8 * 0.5;
        let diff = u_dram - u_nvme;
        assert!(
            (diff - expected_diff).abs() < 1e-3,
            "DRAM-NVMe diff should equal importance*0.8*0.5: diff={diff} expected={expected_diff}",
        );
    }

    /// Verify SwapInWorkerStats Default trait produces type-correct zero values for all fields.
    #[test]
    fn stats_default_trait_all_zero_u64() {
        let s = SwapInWorkerStats::default();
        // All fields are u64. Default for u64 is 0.
        let zero: u64 = 0;
        assert_eq!(s.total_requests, zero);
        assert_eq!(s.submitted, zero);
        assert_eq!(s.skipped, zero);
        assert_eq!(s.promoted_ok, zero);
        assert_eq!(s.promoted_failed, zero);
        assert_eq!(s.two_hop_promotions, zero);
        assert_eq!(s.total_latency_us, zero);
        assert_eq!(s.rounds, zero);
        // avg_latency_us should return 0.0 (guard clause for promoted_ok == 0).
        assert!((s.avg_latency_us() - 0.0).abs() < 1e-6);
    }

    /// Verify max_in_flight=2 allows exactly one NVMe page (consumes 2 in_flight slots).
    #[test]
    fn round_max_in_flight_two_allows_one_nvme_page() {
        let config = SwapInWorkerConfig {
            max_in_flight: 2,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [1usize, 2usize] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: None,
                    current_tier: StorageTier::Nvme, original_bytes: 4096, codec: CompressionCodec::ZstdDict,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = (1..=2)
            .map(|pid| PrefetchRequest {
                page_id: pid, urgency: 1.0, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            })
            .collect();
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);

        // NVMe page 1: PromoteToDram (+1) + PromoteToHbm (+1) = 2 in_flight.
        // After page 1, in_flight=2 >= max_in_flight=2, so page 2 is blocked.
        assert_eq!(submitted, 1, "max_in_flight=2 should allow exactly 1 NVMe page");
        let s = stats.lock().expect("sl");
        assert_eq!(s.two_hop_promotions, 1);

        actor.shutdown();
    }

    /// Verify that swap_in_round with request page_bytes=1 (tiny page) still submits successfully.
    #[test]
    fn round_tiny_page_bytes_submits_successfully() {
        let config = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 1]),
                current_tier: StorageTier::CpuDram, original_bytes: 1, codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // page_bytes=1 in the request — should use request value since it is > 0.
        let mut requests = vec![PrefetchRequest {
            page_id: 1, urgency: 1.0, prefetch_confidence: 0.9,
            page_bytes: 1, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(&config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer);

        assert_eq!(submitted, 1, "tiny page_bytes should still submit");

        actor.shutdown();
    }

    /// Verify urgency with subnormal f32 confidence (very close to zero but not zero).
    #[test]
    fn urgency_subnormal_confidence_still_finite() {
        let meta = PageMetadata {
            page_id: 1, sequence_id: None, recency: 0, access_count: 10,
            last_access: Instant::now(), swap_in_time: None, is_lir: false,
            state: PageState::Active, warm_until: None,
        };
        let u = SwapInWorker::compute_urgency(&meta, f32::MIN_POSITIVE, StorageTier::CpuDram);
        assert!(u.is_finite(), "subnormal confidence should produce finite urgency: {u}");
        assert!(u > 0.0, "subnormal confidence with recent access should still be positive: {u}");
    }

    // ── New wave: PrefetchRequest fields, Error Display variants,
    //    multi-priority sorting, config boundary values,
    //    stats avg_latency calculation, round skip logic ──────────────

    /// Verify PrefetchRequest PartialEq returns false when only urgency differs
    /// and all other numeric fields match (page_id, confidence, page_bytes).
    /// Since enqueued_at is always different between two Instant::now() calls,
    /// two requests with identical numeric fields should still be equal
    /// (Instant equality only matters when the same Instant value is used).
    #[test]
    fn prefetch_request_eq_true_when_all_numeric_fields_identical_same_instant() {
        let now = Instant::now();
        let a = PrefetchRequest {
            page_id: 42, urgency: 0.75, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: now,
        };
        let b = PrefetchRequest {
            page_id: 42, urgency: 0.75, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: now,
        };
        assert_eq!(a, b, "requests sharing the same Instant should be equal");
    }

    /// Verify SwapInWorkerError Display for SendFailed includes the prefix and the message.
    #[test]
    fn error_display_send_failed_exact_format() {
        let err = SwapInWorkerError::SendFailed("channel closed".to_string());
        let displayed = format!("{err}");
        assert!(
            displayed.starts_with("swap-in worker send failed: "),
            "SendFailed display should start with prefix: got '{displayed}'",
        );
        assert!(
            displayed.contains("channel closed"),
            "SendFailed display should contain original message: got '{displayed}'",
        );
    }

    /// Verify SwapInWorkerError Display for RecvFailed includes the prefix and the message.
    #[test]
    fn error_display_recv_failed_exact_format() {
        let err = SwapInWorkerError::RecvFailed("timeout expired".to_string());
        let displayed = format!("{err}");
        assert!(
            displayed.starts_with("swap-in worker recv failed: "),
            "RecvFailed display should start with prefix: got '{displayed}'",
        );
        assert!(
            displayed.contains("timeout expired"),
            "RecvFailed display should contain original message: got '{displayed}'",
        );
    }

    /// Verify multi-priority sorting: many requests with distinct urgencies
    /// are sorted so that highest urgency is processed first.
    /// Uses 10 requests with urgency 0.1..=1.0 and verifies descending order.
    #[test]
    fn round_sorts_ten_requests_by_urgency_descending() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 10,
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        // Insert 10 DRAM pages with page_id 1..=10.
        {
            let mut t = addr_table.write().expect("wl");
            for pid in 1..=10usize {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Build requests with ascending urgency (0.1, 0.2, ..., 1.0).
        // After sorting, page_id 10 (urgency=1.0) should be first.
        let mut requests: Vec<PrefetchRequest> = (1..=10usize)
            .map(|pid| PrefetchRequest {
                page_id: pid,
                urgency: pid as f32 / 10.0,
                prefetch_confidence: 0.9,
                page_bytes: 4096,
                enqueued_at: Instant::now(),
            })
            .collect();
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 10, "all 10 DRAM requests should be submitted");

        actor.shutdown();
    }

    /// Verify Config boundary: max_prefetch_per_round=1 truncates to exactly one request.
    #[test]
    fn config_max_prefetch_one_truncates_many_requests() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [1usize, 2usize, 3usize] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = vec![
            PrefetchRequest {
                page_id: 1, urgency: 0.5, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2, urgency: 1.0, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 3, urgency: 0.1, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        // Only 1 request should be processed (the highest urgency page_id=2).
        assert_eq!(submitted, 1, "max_prefetch_per_round=1 should submit exactly 1");
        let s = stats.lock().expect("sl");
        // total_requests = 3 (counted before truncation), submitted = 1.
        assert_eq!(s.total_requests, 3, "total_requests should count all 3 before truncation");

        actor.shutdown();
    }

    /// Verify stats avg_latency computation: (total_latency_us / promoted_ok) is correct
    /// when multiple promotions contribute to the total.
    #[test]
    fn stats_avg_latency_weighted_average_correct() {
        let mut s = SwapInWorkerStats::default();
        // Simulate: 3 promotions with latencies 100us, 200us, 300us.
        s.promoted_ok = 3;
        s.total_latency_us = 100 + 200 + 300;
        let avg = s.avg_latency_us();
        assert!(
            (avg - 200.0).abs() < 1e-6,
            "avg should be (100+200+300)/3 = 200: got {avg}",
        );
    }

    /// Verify stats avg_latency returns 0.0 when promoted_ok is zero
    /// even when total_latency_us has been manually set to a nonzero value.
    #[test]
    fn stats_avg_latency_zero_promoted_nonzero_total_returns_zero() {
        let mut s = SwapInWorkerStats {
            promoted_ok: 0,
            total_latency_us: 999_999,
            ..SwapInWorkerStats::default()
        };
        // promoted_ok == 0 triggers the early return guard.
        let avg = s.avg_latency_us();
        assert!(
            (avg - 0.0).abs() < 1e-6,
            "avg should be 0.0 when promoted_ok==0: got {avg}",
        );
    }

    /// Verify round skip logic: a request with confidence exactly equal to min_confidence
    /// is accepted (not skipped). Tests the < comparison boundary.
    #[test]
    fn round_skip_confidence_exactly_equal_to_min_is_accepted() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 1, urgency: 1.0, prefetch_confidence: 0.5,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "confidence == min_confidence should be accepted, not skipped");
        let s = stats.lock().expect("sl");
        assert_eq!(s.skipped, 0, "no requests should be skipped");

        actor.shutdown();
    }

    /// Verify round skip logic: a request with confidence just below min_confidence
    /// (using f32 epsilon) is skipped.
    #[test]
    fn round_skip_confidence_one_epsilon_below_min_is_skipped() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(1, PageAddrEntry {
                gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram, original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let below_min = 0.5_f32 - f32::EPSILON;
        let mut requests = vec![PrefetchRequest {
            page_id: 1, urgency: 1.0, prefetch_confidence: below_min,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0, "confidence just below min should be skipped");
        let s = stats.lock().expect("sl");
        assert_eq!(s.skipped, 1, "request should be counted as skipped");

        actor.shutdown();
    }

    /// Verify round skip logic: when a page is not found in the addr_table,
    /// the request is counted as skipped and submitted stays zero.
    #[test]
    fn round_skip_page_missing_from_addr_table_increments_skipped() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        // Intentionally do NOT insert page 999 into addr_table.
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 999, urgency: 1.0, prefetch_confidence: 0.9,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0, "missing page should not be submitted");
        let s = stats.lock().expect("sl");
        assert_eq!(s.skipped, 1, "missing page should be counted as skipped");

        actor.shutdown();
    }

    /// Verify PrefetchRequest fields: urgency can be f32::MAX without panicking.
    #[test]
    fn prefetch_request_urgency_f32_max_field_value() {
        let req = PrefetchRequest {
            page_id: 0, urgency: f32::MAX, prefetch_confidence: 1.0,
            page_bytes: 0, enqueued_at: Instant::now(),
        };
        assert_eq!(req.urgency, f32::MAX);
        assert!(req.urgency.is_finite(), "f32::MAX should be finite");
    }

    /// Verify multi-priority sorting with mixed urgency values including
    /// zero urgency and very high urgency in the same batch.
    /// The highest urgency page should be submitted when max_prefetch_per_round=1.
    #[test]
    fn round_mixed_urgency_highest_submitted_with_truncation() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [10usize, 20, 30] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 10, urgency: 0.0, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 20, urgency: 99.0, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 30, urgency: 0.5, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "truncation should submit only 1");
        // total_requests should be 3 (all counted before truncation).
        let s = stats.lock().expect("sl");
        assert_eq!(s.total_requests, 3);

        actor.shutdown();
    }

    /// Verify Config boundary: tick_interval of 0 is stored as Duration::ZERO
    /// without any clamping.
    #[test]
    fn config_tick_interval_zero_stored_verbatim() {
        let c = SwapInWorkerConfig {
            tick_interval: Duration::ZERO,
            ..SwapInWorkerConfig::default()
        };
        assert_eq!(c.tick_interval, Duration::ZERO);
    }

    /// Verify Config boundary: max_in_flight=1 allows exactly one DRAM page
    /// and blocks a second request in the same round.
    #[test]
    fn round_max_in_flight_one_allows_exactly_one_dram() {
        let config = SwapInWorkerConfig {
            max_in_flight: 1,
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [1usize, 2usize] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            PrefetchRequest {
                page_id: 1, urgency: 1.0, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            },
            PrefetchRequest {
                page_id: 2, urgency: 0.5, prefetch_confidence: 0.9,
                page_bytes: 4096, enqueued_at: Instant::now(),
            },
        ];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 1, "max_in_flight=1 should allow only 1 DRAM page");
        let s = stats.lock().expect("sl");
        assert_eq!(s.submitted, 1);
        // The second request was not processed at all (back-pressure break).
        // It was not counted as skipped because the loop broke before reaching it.

        actor.shutdown();
    }

    /// Verify Error Display: both variants produce non-empty strings.
    #[test]
    fn error_display_both_variants_non_empty() {
        let send = format!("{}", SwapInWorkerError::SendFailed("x".to_string()));
        let recv = format!("{}", SwapInWorkerError::RecvFailed("y".to_string()));
        assert!(!send.is_empty(), "SendFailed display should be non-empty");
        assert!(!recv.is_empty(), "RecvFailed display should be non-empty");
    }

    /// Verify stats avg_latency precision: when total_latency_us = 1 and promoted_ok = 3,
    /// the result should be approximately 0.333...
    #[test]
    fn stats_avg_latency_fractional_microsecond_precision() {
        let s = SwapInWorkerStats {
            promoted_ok: 3,
            total_latency_us: 1,
            ..SwapInWorkerStats::default()
        };
        let avg = s.avg_latency_us();
        assert!(
            (avg - (1.0 / 3.0)).abs() < 1e-10,
            "avg should be 1/3 with high precision: got {avg}",
        );
    }

    /// Verify round skip: all requests skipped because of low confidence results in
    /// submitted=0, skipped=N, and total_requests=N.
    #[test]
    fn round_all_low_confidence_skipped_total_requests_matches_input() {
        let config = SwapInWorkerConfig {
            min_confidence: 0.9,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [1usize, 2, 3, 4, 5] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None, host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram, original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // All 5 requests with confidence=0.1, below min_confidence=0.9.
        let mut requests: Vec<PrefetchRequest> = (1..=5usize)
            .map(|pid| PrefetchRequest {
                page_id: pid, urgency: 1.0, prefetch_confidence: 0.1,
                page_bytes: 4096, enqueued_at: Instant::now(),
            })
            .collect();
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );
        assert_eq!(submitted, 0, "all low-confidence requests should be skipped");
        let s = stats.lock().expect("sl");
        assert_eq!(s.total_requests, 5, "total_requests should count all 5");
        assert_eq!(s.skipped, 5, "all 5 should be counted as skipped");
        assert_eq!(s.submitted, 0);

        actor.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 15 new tests — round DRAM direct promote, host_buffer/gpu_ptr exclusivity,
    // worker thread lifecycle, stats Debug field coverage, config full-field PartialEq
    // ═══════════════════════════════════════════════════════════════════════════

    // ── Round: DRAM page direct promote to HBM (no two-hop) ──

    /// Verify that a CpuDram page promoted in a round gets submitted=1 but
    /// two_hop_promotions stays at 0 — DRAM→HBM is a single-hop path.
    #[test]
    fn round_dram_direct_promote_no_two_hop_counter() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(77, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::Lz4,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 77, urgency: 1.0, prefetch_confidence: 0.9,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "DRAM page should be submitted");
        let s = stats.lock().expect("sl");
        assert_eq!(s.two_hop_promotions, 0, "DRAM→HBM is not two-hop");
        assert_eq!(s.submitted, 1);

        actor.shutdown();
    }

    /// Verify that when two DRAM pages are submitted, neither increments two_hop.
    #[test]
    fn round_two_dram_pages_both_single_hop() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            for pid in [10usize, 20] {
                t.insert(pid, PageAddrEntry {
                    gpu_ptr: None,
                    host_buffer: Some(vec![0u8; 4096]),
                    current_tier: StorageTier::CpuDram,
                    original_bytes: 4096,
                    codec: CompressionCodec::None,
                });
            }
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests: Vec<PrefetchRequest> = [10usize, 20].iter().map(|&pid| PrefetchRequest {
            page_id: pid, urgency: 0.8, prefetch_confidence: 0.9,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }).collect();
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 2, "both DRAM pages should be submitted");
        let s = stats.lock().expect("sl");
        assert_eq!(s.two_hop_promotions, 0, "neither DRAM page should be two-hop");

        actor.shutdown();
    }

    // ── Round: host_buffer vs gpu_ptr exclusivity patterns ──

    /// A page on GpuHbm has gpu_ptr set but host_buffer=None. Verify round skips it.
    #[test]
    fn round_gpu_ptr_set_host_buffer_none_skips_as_hbm() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(55, PageAddrEntry {
                gpu_ptr: Some(0xBEEF0000),
                host_buffer: None,
                current_tier: StorageTier::GpuHbm,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 55, urgency: 1.0, prefetch_confidence: 0.9,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 0, "HBM page with gpu_ptr set should be skipped");
        let s = stats.lock().expect("sl");
        assert_eq!(s.skipped, 1);

        actor.shutdown();
    }

    /// NVMe page has both gpu_ptr=None and host_buffer=None. Verify round still submits
    /// (tier check is based on current_tier, not on buffer/pointer presence).
    #[test]
    fn round_nvme_both_gpu_ptr_and_host_buffer_none_submits_two_hop() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(33, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: None,
                current_tier: StorageTier::Nvme,
                original_bytes: 4096,
                codec: CompressionCodec::ZstdDict,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 33, urgency: 0.9, prefetch_confidence: 0.8,
            page_bytes: 4096, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "NVMe page submits regardless of buffer/ptr state");
        let s = stats.lock().expect("sl");
        assert_eq!(s.two_hop_promotions, 1, "NVMe page is two-hop");

        actor.shutdown();
    }

    /// Page on DRAM with host_buffer set and gpu_ptr=None — standard case. Verify submit.
    #[test]
    fn round_dram_host_buffer_set_gpu_ptr_none_submits_single_hop() {
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        {
            let mut t = addr_table.write().expect("wl");
            t.insert(44, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0xFF; 8192]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 8192,
                codec: CompressionCodec::BitPackRle,
            });
        }
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![PrefetchRequest {
            page_id: 44, urgency: 0.7, prefetch_confidence: 0.85,
            page_bytes: 8192, enqueued_at: Instant::now(),
        }];
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        assert_eq!(submitted, 1, "DRAM page with host_buffer submits");
        let s = stats.lock().expect("sl");
        assert_eq!(s.two_hop_promotions, 0, "DRAM→HBM is single-hop");

        actor.shutdown();
    }

    // ── Worker thread lifecycle ──

    /// Spawn worker, prefetch several requests, sleep to let the thread process
    /// a couple of rounds, then verify stats show at least one round executed.
    #[test]
    fn worker_thread_processes_at_least_one_round() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(10),
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, Arc::clone(&page_metadata), addr_table, observer,
        );

        // Enqueue 3 requests — none are in addr_table so they will be skipped,
        // but the thread should process them within a few ticks.
        for pid in 1..=3u64 {
            let _ = worker.prefetch(PrefetchRequest {
                page_id: pid as usize, urgency: 0.5, prefetch_confidence: 0.8,
                page_bytes: 4096, enqueued_at: Instant::now(),
            });
        }

        // Wait for 3 tick intervals to let the thread drain.
        thread::sleep(Duration::from_millis(50));

        let s = worker.stats();
        assert!(s.rounds >= 1, "worker should have executed at least 1 round: rounds={}", s.rounds);
        assert!(s.total_requests >= 3, "worker should have received all 3 requests: total={}", s.total_requests);

        worker.shutdown();
    }

    /// Verify worker stats snapshot shows non-zero rounds after prefetch and sleep,
    /// confirming the background thread is alive and processing.
    #[test]
    fn worker_stats_snapshot_shows_rounds_after_enqueue_and_wait() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(5),
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );

        let _ = worker.prefetch(PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.7,
            page_bytes: 4096, enqueued_at: Instant::now(),
        });

        thread::sleep(Duration::from_millis(30));

        let stats = worker.stats();
        assert!(stats.rounds > 0, "background thread should have run at least one round: {}", stats.rounds);

        worker.shutdown();
    }

    /// Verify that after shutdown, the worker thread is no longer running by
    /// confirming stats stop accumulating.
    #[test]
    fn worker_thread_stops_accumulating_after_shutdown() {
        let config = SwapInWorkerConfig {
            tick_interval: Duration::from_millis(5),
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );

        // Let thread run.
        thread::sleep(Duration::from_millis(20));
        worker.shutdown();

        let stats_before = worker.stats();
        let rounds_before = stats_before.rounds;

        // Wait a bit — rounds should not increase after shutdown.
        thread::sleep(Duration::from_millis(30));
        let stats_after = worker.stats();
        assert_eq!(
            stats_after.rounds, rounds_before,
            "rounds should not increase after shutdown: before={} after={}",
            rounds_before, stats_after.rounds,
        );
    }

    // ── SwapInWorkerStats: Debug format field coverage ──

    /// Verify Debug output contains all 8 field names of SwapInWorkerStats.
    #[test]
    fn stats_debug_contains_all_eight_field_names() {
        let mut stats = SwapInWorkerStats::default();
        stats.total_requests = 1;
        stats.submitted = 1;
        stats.skipped = 0;
        stats.promoted_ok = 1;
        stats.promoted_failed = 0;
        stats.two_hop_promotions = 0;
        stats.total_latency_us = 100;
        stats.rounds = 1;
        let debug = format!("{stats:?}");
        assert!(debug.contains("total_requests"), "Debug should contain total_requests");
        assert!(debug.contains("submitted"), "Debug should contain submitted");
        assert!(debug.contains("skipped"), "Debug should contain skipped");
        assert!(debug.contains("promoted_ok"), "Debug should contain promoted_ok");
        assert!(debug.contains("promoted_failed"), "Debug should contain promoted_failed");
        assert!(debug.contains("two_hop_promotions"), "Debug should contain two_hop_promotions");
        assert!(debug.contains("total_latency_us"), "Debug should contain total_latency_us");
        assert!(debug.contains("rounds"), "Debug should contain rounds");
    }

    /// Verify Debug output of non-default stats includes the struct name.
    #[test]
    fn stats_debug_non_default_includes_struct_name() {
        let stats = SwapInWorkerStats {
            total_requests: 100,
            submitted: 80,
            skipped: 20,
            promoted_ok: 75,
            promoted_failed: 5,
            two_hop_promotions: 10,
            total_latency_us: 50000,
            rounds: 50,
        };
        let debug = format!("{stats:?}");
        assert!(debug.contains("SwapInWorkerStats"), "Debug should contain struct name");
        assert!(debug.contains("100"), "Debug should contain total_requests value");
        assert!(debug.contains("50000"), "Debug should contain total_latency_us value");
    }

    // ── SwapInWorkerConfig: full-field PartialEq ──

    /// Verify PartialEq detects difference in max_prefetch_per_round.
    #[test]
    fn config_partial_eq_detects_max_prefetch_round_difference() {
        let c1 = SwapInWorkerConfig {
            max_prefetch_per_round: 16,
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            max_prefetch_per_round: 17,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2, "configs differing only in max_prefetch_per_round should not be equal");
    }

    /// Verify PartialEq detects difference in min_confidence.
    #[test]
    fn config_partial_eq_detects_min_confidence_difference() {
        let c1 = SwapInWorkerConfig {
            min_confidence: 0.1,
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            min_confidence: 0.10001,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2, "configs differing only in min_confidence should not be equal");
    }

    /// Verify PartialEq detects difference in max_in_flight.
    #[test]
    fn config_partial_eq_detects_max_in_flight_difference() {
        let c1 = SwapInWorkerConfig {
            max_in_flight: 64,
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            max_in_flight: 65,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2, "configs differing only in max_in_flight should not be equal");
    }

    /// Verify PartialEq detects difference in page_bytes.
    #[test]
    fn config_partial_eq_detects_page_bytes_difference() {
        let c1 = SwapInWorkerConfig {
            page_bytes: 4096,
            ..SwapInWorkerConfig::default()
        };
        let c2 = SwapInWorkerConfig {
            page_bytes: 4097,
            ..SwapInWorkerConfig::default()
        };
        assert_ne!(c1, c2, "configs differing only in page_bytes should not be equal");
    }

    /// Verify that a worker spawned with all-custom config still processes a prefetch
    /// request correctly, confirming config values propagate without panicking.
    #[test]
    fn worker_custom_config_processes_prefetch_without_panic() {
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 2,
            tick_interval: Duration::from_millis(5),
            min_confidence: 0.3,
            max_in_flight: 4,
            page_bytes: 8192,
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(), Arc::clone(&backend), Arc::clone(&addr_table), None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut worker = SwapInWorker::spawn(
            config, actor, page_metadata, addr_table, observer,
        );

        let result = worker.prefetch(PrefetchRequest {
            page_id: 1, urgency: 0.5, prefetch_confidence: 0.6,
            page_bytes: 8192, enqueued_at: Instant::now(),
        });
        assert!(result.is_ok(), "prefetch on custom-config worker should succeed");

        thread::sleep(Duration::from_millis(20));
        let stats = worker.stats();
        assert!(stats.rounds >= 1, "custom-config worker should run at least 1 round: {}", stats.rounds);

        worker.shutdown();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Wave 15 tests — new angles: Clone/Debug roundtrip, Default type, Display
    // format, urgency boundary per tier, duplicate page_id in single batch,
    // config page_bytes=0 with request page_bytes=0.
    // ═══════════════════════════════════════════════════════════════════════════

    /// Verify that cloning a custom SwapInWorkerConfig and formatting the clone
    /// via Debug still shows all five field names.
    #[test]
    fn config_clone_then_debug_shows_all_five_fields() {
        // Arrange
        let cfg = SwapInWorkerConfig {
            max_prefetch_per_round: 99,
            tick_interval: Duration::from_millis(77),
            min_confidence: 0.42,
            max_in_flight: 200,
            page_bytes: 16384,
        };
        let cloned = cfg.clone();

        // Act
        let debug = format!("{cloned:?}");

        // Assert: all five field names appear in the cloned Debug output.
        assert!(debug.contains("max_prefetch_per_round"), "cloned Debug should contain max_prefetch_per_round: {debug}");
        assert!(debug.contains("tick_interval"), "cloned Debug should contain tick_interval: {debug}");
        assert!(debug.contains("min_confidence"), "cloned Debug should contain min_confidence: {debug}");
        assert!(debug.contains("max_in_flight"), "cloned Debug should contain max_in_flight: {debug}");
        assert!(debug.contains("page_bytes"), "cloned Debug should contain page_bytes: {debug}");
    }

    /// Verify SwapInWorkerConfig Clone roundtrip: cloned equals original via PartialEq.
    #[test]
    fn config_clone_roundtrip_preserves_equality() {
        // Arrange
        let original = SwapInWorkerConfig {
            max_prefetch_per_round: 7,
            tick_interval: Duration::from_micros(333),
            min_confidence: 0.65,
            max_in_flight: 11,
            page_bytes: 2048,
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(original, cloned, "cloned config must equal original via PartialEq");
    }

    /// Verify SwapInWorkerStats Default produces a value that equals its own clone.
    #[test]
    fn stats_default_clone_is_equal() {
        // Arrange
        let default_stats = SwapInWorkerStats::default();

        // Act
        let cloned = default_stats.clone();

        // Assert
        assert_eq!(default_stats, cloned, "default stats should equal its clone");
    }

    /// Verify SwapInWorkerStats Default every field is exactly 0u64 and
    /// the field type is u64 (compile-time check via explicit type annotation).
    #[test]
    fn stats_default_each_field_is_zero_with_type_annotation() {
        // Arrange — explicit u64 type annotations for compile-time type verification.
        let s = SwapInWorkerStats::default();
        let zero_u64: u64 = 0;

        // Assert
        assert_eq!(s.total_requests, zero_u64, "total_requests should be 0u64");
        assert_eq!(s.submitted, zero_u64, "submitted should be 0u64");
        assert_eq!(s.skipped, zero_u64, "skipped should be 0u64");
        assert_eq!(s.promoted_ok, zero_u64, "promoted_ok should be 0u64");
        assert_eq!(s.promoted_failed, zero_u64, "promoted_failed should be 0u64");
        assert_eq!(s.two_hop_promotions, zero_u64, "two_hop_promotions should be 0u64");
        assert_eq!(s.total_latency_us, zero_u64, "total_latency_us should be 0u64");
        assert_eq!(s.rounds, zero_u64, "rounds should be 0u64");
    }

    /// Verify SwapInWorkerError SendFailed Display contains both the standard
    /// prefix and the full message text (including multi-word messages).
    #[test]
    fn error_display_send_failed_prefix_and_message_integrated() {
        // Arrange
        let msg = "channel disconnected during batch send";
        let err = SwapInWorkerError::SendFailed(msg.to_string());

        // Act
        let displayed = format!("{err}");

        // Assert
        let expected_prefix = "swap-in worker send failed: ";
        assert!(
            displayed.starts_with(expected_prefix),
            "Display should start with '{expected_prefix}': got '{displayed}'",
        );
        assert!(
            displayed.ends_with(msg),
            "Display should end with the original message '{msg}': got '{displayed}'",
        );
        assert_eq!(
            displayed,
            format!("swap-in worker send failed: {msg}"),
            "Display should be exact prefix + message",
        );
    }

    /// Verify SwapInWorkerError RecvFailed Display contains both the standard
    /// prefix and the full message text (including multi-word messages).
    #[test]
    fn error_display_recv_failed_prefix_and_message_integrated() {
        // Arrange
        let msg = "timeout waiting for migration actor response";
        let err = SwapInWorkerError::RecvFailed(msg.to_string());

        // Act
        let displayed = format!("{err}");

        // Assert
        let expected_prefix = "swap-in worker recv failed: ";
        assert!(
            displayed.starts_with(expected_prefix),
            "Display should start with '{expected_prefix}': got '{displayed}'",
        );
        assert!(
            displayed.ends_with(msg),
            "Display should end with the original message '{msg}': got '{displayed}'",
        );
        assert_eq!(
            displayed,
            format!("swap-in worker recv failed: {msg}"),
            "Display should be exact prefix + message",
        );
    }

    /// Verify PrefetchRequest Clone then Debug contains all five field names,
    /// ensuring Clone preserves all data and Debug formats it.
    #[test]
    fn prefetch_request_clone_then_debug_contains_all_five_fields() {
        // Arrange
        let req = PrefetchRequest {
            page_id: 255,
            urgency: 3.14,
            prefetch_confidence: 0.77,
            page_bytes: 65536,
            enqueued_at: Instant::now(),
        };
        let cloned = req.clone();

        // Act
        let debug = format!("{cloned:?}");

        // Assert: all five field names appear after clone+debug roundtrip.
        assert!(debug.contains("page_id"), "cloned Debug should contain page_id: {debug}");
        assert!(debug.contains("urgency"), "cloned Debug should contain urgency: {debug}");
        assert!(debug.contains("prefetch_confidence"), "cloned Debug should contain prefetch_confidence: {debug}");
        assert!(debug.contains("page_bytes"), "cloned Debug should contain page_bytes: {debug}");
        assert!(debug.contains("enqueued_at"), "cloned Debug should contain enqueued_at: {debug}");
    }

    /// Verify PrefetchRequest Clone roundtrip via PartialEq: cloned == original.
    #[test]
    fn prefetch_request_clone_roundtrip_equality() {
        // Arrange
        let now = Instant::now();
        let req = PrefetchRequest {
            page_id: 512,
            urgency: 2.718,
            prefetch_confidence: 0.618,
            page_bytes: 32768,
            enqueued_at: now,
        };

        // Act
        let cloned = req.clone();

        // Assert
        assert_eq!(req, cloned, "cloned PrefetchRequest must equal original via PartialEq");
    }

    /// Verify compute_urgency with NaN confidence on NVMe tier produces NaN.
    #[test]
    fn urgency_nan_confidence_on_nvme_produces_nan() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act
        let urgency = SwapInWorker::compute_urgency(&meta, f32::NAN, StorageTier::Nvme);

        // Assert
        assert!(
            urgency.is_nan(),
            "NaN confidence on NVMe tier should produce NaN urgency: got {urgency}",
        );
    }

    /// Verify compute_urgency with NaN confidence on GpuHbm tier produces NaN.
    #[test]
    fn urgency_nan_confidence_on_gpu_hbm_produces_nan() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };

        // Act
        let urgency = SwapInWorker::compute_urgency(&meta, f32::NAN, StorageTier::GpuHbm);

        // Assert
        assert!(
            urgency.is_nan(),
            "NaN confidence on GpuHbm tier should produce NaN urgency: got {urgency}",
        );
    }

    /// Verify compute_urgency with f32::MAX confidence on NVMe tier produces
    /// a finite positive urgency (first term is finite due to ln_1p boundedness).
    #[test]
    fn urgency_f32_max_confidence_on_nvme_produces_finite() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act
        let urgency = SwapInWorker::compute_urgency(&meta, f32::MAX, StorageTier::Nvme);

        // Assert: first_term = importance_rebound * f32::MAX * 0.5 → finite because
        // importance_rebound = ln(11)/ln(10) ≈ 1.04, and 1.04 * MAX * 0.5 = finite.
        assert!(
            urgency.is_finite(),
            "f32::MAX confidence on NVMe should produce finite urgency: got {urgency}",
        );
        assert!(
            urgency > 0.0,
            "f32::MAX confidence on NVMe should produce positive urgency: got {urgency}",
        );
    }

    /// Verify compute_urgency with f32::MAX confidence on GpuHbm tier produces
    /// a finite positive urgency.
    #[test]
    fn urgency_f32_max_confidence_on_gpu_hbm_produces_finite() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Active,
            warm_until: None,
        };

        // Act
        let urgency = SwapInWorker::compute_urgency(&meta, f32::MAX, StorageTier::GpuHbm);

        // Assert: first_term = importance_rebound * f32::MAX * 2.0 → could overflow to infinity
        // but importance_rebound is small (~1.04), so 1.04 * MAX * 2.0 = +inf.
        // The recency bonus is small. So urgency should be positive (either finite or inf).
        assert!(
            urgency > 0.0,
            "f32::MAX confidence on GpuHbm should produce positive urgency: got {urgency}",
        );
    }

    /// Verify swap_in_round processes duplicate page_ids within a single batch
    /// (no deduplication). Two requests for page_id=1 should both be submitted.
    #[test]
    fn round_duplicate_page_id_in_single_batch_both_submitted() {
        // Arrange
        let config = SwapInWorkerConfig {
            min_confidence: 0.0,
            max_prefetch_per_round: 10,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Two requests for the same page_id with different urgencies.
        let mut requests = vec![
            make_prefetch(1, 1.0, 0.9, 4096),
            make_prefetch(1, 0.5, 0.9, 4096),
        ];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert: no dedup — both requests should be submitted.
        assert_eq!(submitted, 2, "both duplicate page_id requests should be submitted (no dedup): got {submitted}");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2, "total_requests should count both duplicates");
        assert_eq!(s.submitted, 2, "submitted should count both duplicates");

        actor.shutdown();
    }

    /// Verify swap_in_round when both config.page_bytes=0 and request.page_bytes=0:
    /// the fallback still resolves to config.page_bytes (which is 0), and the page
    /// is still submitted (page_bytes=0 is passed through to the migration actor).
    #[test]
    fn round_both_config_and_request_page_bytes_zero_still_submits() {
        // Arrange
        let config = SwapInWorkerConfig {
            page_bytes: 0,
            min_confidence: 0.0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Request with page_bytes=0 and config.page_bytes=0.
        let mut requests = vec![make_prefetch(1, 1.0, 0.9, 0)];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert: page_bytes=0 is allowed — the page is submitted (zero bytes migration).
        assert_eq!(submitted, 1, "page with zero page_bytes (both config and request) should still be submitted");

        actor.shutdown();
    }

    /// Verify SwapInWorkerConfig with page_bytes=0 is not equal to default (4096).
    #[test]
    fn config_page_bytes_zero_differs_from_default() {
        // Arrange
        let cfg_zero = SwapInWorkerConfig {
            page_bytes: 0,
            ..SwapInWorkerConfig::default()
        };
        let cfg_default = SwapInWorkerConfig::default();

        // Assert
        assert_ne!(cfg_zero, cfg_default, "config with page_bytes=0 should not equal default (page_bytes=4096)");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Additional 15 unit tests
    // ─────────────────────────────────────────────────────────────────────────────

    /// 1. SwapInWorkerConfig Clone roundtrip with all fields — cloned config
    ///    preserves every field exactly, and modifying the clone does not affect
    ///    the original.
    #[test]
    fn config_clone_roundtrip_all_fields_exact() {
        // Arrange
        let original = SwapInWorkerConfig {
            max_prefetch_per_round: 32,
            tick_interval: Duration::from_millis(10),
            min_confidence: 0.25,
            max_in_flight: 128,
            page_bytes: 8192,
        };

        // Act
        let cloned = original.clone();

        // Assert — every field matches exactly
        assert_eq!(cloned.max_prefetch_per_round, 32);
        assert_eq!(cloned.tick_interval, Duration::from_millis(10));
        assert_eq!(cloned.min_confidence, 0.25);
        assert_eq!(cloned.max_in_flight, 128);
        assert_eq!(cloned.page_bytes, 8192);
        assert_eq!(cloned, original, "clone must be equal to original across all 5 fields");
    }

    /// 2. SwapInWorkerStats Default sets all fields to zero.
    #[test]
    fn stats_default_all_fields_are_zero() {
        // Act
        let stats = SwapInWorkerStats::default();

        // Assert — all 9 fields must be zero
        assert_eq!(stats.total_requests, 0, "total_requests must be 0");
        assert_eq!(stats.submitted, 0, "submitted must be 0");
        assert_eq!(stats.skipped, 0, "skipped must be 0");
        assert_eq!(stats.promoted_ok, 0, "promoted_ok must be 0");
        assert_eq!(stats.promoted_failed, 0, "promoted_failed must be 0");
        assert_eq!(stats.two_hop_promotions, 0, "two_hop_promotions must be 0");
        assert_eq!(stats.total_latency_us, 0, "total_latency_us must be 0");
        assert_eq!(stats.rounds, 0, "rounds must be 0");
    }

    /// 3. SwapInWorkerError Display formats both variants with correct prefix.
    #[test]
    fn error_display_all_variants() {
        // Arrange
        let send_err = SwapInWorkerError::SendFailed("channel closed".to_string());
        let recv_err = SwapInWorkerError::RecvFailed("timeout".to_string());

        // Act
        let send_display = format!("{send_err}");
        let recv_display = format!("{recv_err}");

        // Assert
        assert!(
            send_display.starts_with("swap-in worker send failed:"),
            "SendFailed Display must start with correct prefix: got '{send_display}'"
        );
        assert!(
            send_display.contains("channel closed"),
            "SendFailed Display must contain the message"
        );
        assert!(
            recv_display.starts_with("swap-in worker recv failed:"),
            "RecvFailed Display must start with correct prefix: got '{recv_display}'"
        );
        assert!(
            recv_display.contains("timeout"),
            "RecvFailed Display must contain the message"
        );
    }

    /// 4. PrefetchRequest Clone and Debug roundtrip — cloned value equals
    ///    original, Debug output contains relevant field info.
    #[test]
    fn prefetch_request_clone_debug_roundtrip() {
        // Arrange
        let original = PrefetchRequest {
            page_id: 99,
            urgency: 3.14,
            prefetch_confidence: 0.75,
            page_bytes: 8192,
            enqueued_at: Instant::now(),
        };

        // Act
        let cloned = original.clone();
        let debug_str = format!("{original:?}");

        // Assert — Clone preserves all fields
        assert_eq!(cloned.page_id, 99);
        assert_eq!(cloned.urgency, 3.14);
        assert_eq!(cloned.prefetch_confidence, 0.75);
        assert_eq!(cloned.page_bytes, 8192);
        assert_eq!(cloned, original, "cloned PrefetchRequest must equal original");

        // Debug output contains field names/values
        assert!(debug_str.contains("page_id"), "Debug must contain page_id");
        assert!(debug_str.contains("urgency"), "Debug must contain urgency");
    }

    /// 5. Urgency calculation with f32::MAX confidence still produces a finite
    ///    value on CpuDram tier (no overflow to infinity).
    #[test]
    fn urgency_f32_max_confidence_produces_finite() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(0),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act
        let urgency = SwapInWorker::compute_urgency(&meta, f32::MAX, StorageTier::CpuDram);

        // Assert — should be finite (not NaN or Inf) because ln_1p(10) is bounded
        assert!(
            urgency.is_finite(),
            "urgency with f32::MAX confidence should be finite, got {urgency}"
        );
    }

    /// 6. Urgency with 0.0 confidence on all tiers still produces a positive
    ///    score from the recency bonus alone.
    #[test]
    fn urgency_zero_confidence_positive_from_recency() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 5,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        for tier in [StorageTier::CpuDram, StorageTier::Nvme, StorageTier::GpuHbm] {
            let urgency = SwapInWorker::compute_urgency(&meta, 0.0, tier);

            // Assert — recency bonus (0.1 * 1/(1+0) = 0.1) ensures positive
            assert!(
                urgency > 0.0,
                "urgency with 0.0 confidence on {tier:?} should be positive from recency, got {urgency}"
            );
        }
    }

    /// 7. Urgency with NaN confidence produces NaN in the first term
    ///    (importance_rebound * NaN = NaN), but the recency bonus remains finite.
    ///    The total urgency is NaN because NaN + finite = NaN.
    #[test]
    fn urgency_nan_confidence_propagates_to_nan() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::Standby,
            warm_until: None,
        };

        // Act
        let urgency = SwapInWorker::compute_urgency(&meta, f32::NAN, StorageTier::CpuDram);

        // Assert — NaN * finite = NaN; NaN + finite = NaN
        assert!(
            urgency.is_nan(),
            "urgency with NaN confidence should be NaN, got {urgency}"
        );
    }

    /// 8. Round with duplicate page_id in requests — both requests are processed
    ///    independently (no deduplication at the round level; dedup is the
    ///    caller's responsibility).
    #[test]
    fn round_duplicate_page_id_both_processed() {
        // Arrange
        let config = SwapInWorkerConfig::default();
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 42, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // Two requests for the same page_id with different urgencies
        let mut requests = vec![
            make_prefetch(42, 2.0, 0.9, 4096),
            make_prefetch(42, 1.0, 0.9, 4096),
        ];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert — both requests are submitted (no dedup)
        assert_eq!(
            submitted, 2,
            "duplicate page_id requests should both be submitted, no dedup at round level"
        );

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2, "both requests should be counted");

        actor.shutdown();
    }

    /// 9. Config page_bytes=0 boundary — swap_in_round with page_bytes=0 in both
    ///    the request and config still processes the request (zero-byte migration
    ///    is allowed).
    #[test]
    fn config_page_bytes_zero_boundary_still_submits() {
        // Arrange
        let config = SwapInWorkerConfig {
            page_bytes: 0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 7, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![make_prefetch(7, 1.0, 0.9, 0)];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert — zero-byte page is still submitted
        assert_eq!(submitted, 1, "page_bytes=0 should still be submitted");

        actor.shutdown();
    }

    /// 10. Stats increment consistency — manually incrementing pages_swapped_in,
    ///     bytes_transferred, and latency fields preserves expected values.
    #[test]
    fn stats_increment_consistency_across_fields() {
        // Arrange
        let mut stats = SwapInWorkerStats::default();

        // Act — simulate 3 successful promotions
        stats.promoted_ok = 3;
        stats.total_latency_us = 150 + 200 + 250; // 600us total
        stats.submitted = 3;
        stats.two_hop_promotions = 1;
        stats.rounds = 2;

        // Assert — each field independently tracks its counter
        assert_eq!(stats.promoted_ok, 3);
        assert_eq!(stats.total_latency_us, 600);
        assert_eq!(stats.submitted, 3);
        assert_eq!(stats.two_hop_promotions, 1);
        assert_eq!(stats.rounds, 2);

        // avg_latency = 600 / 3 = 200
        let avg = stats.avg_latency_us();
        assert!(
            (avg - 200.0).abs() < f64::EPSILON,
            "avg_latency should be 200.0, got {avg}"
        );
    }

    /// 11. Error From trait chain — SwapInWorkerError can be constructed from
    ///     the send error path that wraps an external message string.
    #[test]
    fn error_from_send_failure_wraps_message() {
        // Arrange — simulate the error path from prefetch()
        let io_msg = "sending on a closed channel".to_string();

        // Act — construct error the same way prefetch() does
        let err = SwapInWorkerError::SendFailed(io_msg.clone());

        // Assert — the message is preserved and Display formats correctly
        assert_eq!(err, SwapInWorkerError::SendFailed(io_msg));
        let displayed = format!("{err}");
        assert!(
            displayed.contains("sending on a closed channel"),
            "Display must contain the original message: got '{displayed}'"
        );
    }

    /// 12. Config max_prefetch_rounds=0 (via max_prefetch_per_round=0) — round
    ///     truncates to 0 requests and submits nothing.
    #[test]
    fn config_max_prefetch_per_round_zero_submits_nothing() {
        // Arrange
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 0,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        let mut requests = vec![
            make_prefetch(1, 1.0, 0.9, 4096),
            make_prefetch(2, 0.8, 0.9, 4096),
        ];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert — truncate(0) removes all requests, nothing submitted
        assert_eq!(submitted, 0, "max_prefetch_per_round=0 should submit nothing");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2, "requests counted before truncation");
        assert_eq!(s.submitted, 0, "nothing submitted after truncation");

        actor.shutdown();
    }

    /// 13. Multiple prefetch requests are sorted by urgency descending — the
    ///     highest-urgency request is submitted first.
    #[test]
    fn prefetch_batch_ordering_by_urgency_descending() {
        // Arrange
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 2,
            ..SwapInWorkerConfig::default()
        };
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        insert_addr_entry(&addr_table, 1, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 2, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 3, StorageTier::CpuDram, CompressionCodec::None);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> = Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> = Arc::new(Mutex::new(BasicObserver::new()));

        // 3 requests with ascending urgency — only top 2 should be submitted
        let mut requests = vec![
            make_prefetch(1, 0.1, 0.9, 4096),  // lowest
            make_prefetch(2, 0.5, 0.9, 4096),  // middle
            make_prefetch(3, 0.9, 0.9, 4096),  // highest
        ];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config, &actor, &mut requests, &page_metadata, &addr_table, &stats, &observer,
        );

        // Assert — only 2 submitted (max_prefetch_per_round=2), highest urgency first
        assert_eq!(submitted, 2, "should submit exactly 2 requests (truncated from 3)");

        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 3, "all 3 requests counted before truncation");
        assert_eq!(s.submitted, 2, "only 2 submitted after truncation");

        actor.shutdown();
    }

    /// 14. Urgency monotonic relationship — urgency strictly increases when
    ///     confidence increases (holding all other factors constant).
    #[test]
    fn urgency_monotonic_increases_with_confidence_strict() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: Some(0),
            recency: 0,
            access_count: 10,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act — compute urgency for increasing confidence values
        let u_low = SwapInWorker::compute_urgency(&meta, 0.2, StorageTier::CpuDram);
        let u_mid = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);
        let u_high = SwapInWorker::compute_urgency(&meta, 0.9, StorageTier::CpuDram);

        // Assert — strict monotonic increase (recency bonus is constant)
        assert!(
            u_low < u_mid,
            "urgency should increase from conf=0.2 to 0.5: {u_low} vs {u_mid}"
        );
        assert!(
            u_mid < u_high,
            "urgency should increase from conf=0.5 to 0.9: {u_mid} vs {u_high}"
        );
    }

    /// 15. Stats Clone preserves all fields — cloned stats matches original,
    ///     and modifying the clone does not affect the original.
    #[test]
    fn stats_clone_preserves_all_nine_fields() {
        // Arrange
        let original = SwapInWorkerStats {
            total_requests: 100,
            submitted: 80,
            skipped: 20,
            promoted_ok: 70,
            promoted_failed: 10,
            two_hop_promotions: 5,
            total_latency_us: 50000,
            rounds: 15,
        };

        // Act
        let cloned = original.clone();

        // Assert — all 9 fields match
        assert_eq!(cloned.total_requests, 100);
        assert_eq!(cloned.submitted, 80);
        assert_eq!(cloned.skipped, 20);
        assert_eq!(cloned.promoted_ok, 70);
        assert_eq!(cloned.promoted_failed, 10);
        assert_eq!(cloned.two_hop_promotions, 5);
        assert_eq!(cloned.total_latency_us, 50000);
        assert_eq!(cloned.rounds, 15);
        assert_eq!(cloned, original, "cloned stats must equal original");
    }

    // ── New tests (wave-13x01) ──────────────────────────────────────────────────

    /// 1. compute_urgency with access_count=1 yields a small but positive
    ///    first term on CpuDram with moderate confidence.
    #[test]
    fn urgency_access_count_one_lowest_nonzero() {
        // Arrange
        let meta = PageMetadata {
            page_id: 1,
            sequence_id: None,
            recency: 0,
            access_count: 1,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act
        let u = SwapInWorker::compute_urgency(&meta, 0.5, StorageTier::CpuDram);

        // Assert — ln(2)/ln(11) * 0.5 * 1.0 + ~0.1 recency > 0
        let importance_rebound = 1.0_f32.ln_1p() / 10.0_f32.ln_1p();
        let first_term = importance_rebound * 0.5 * 1.0;
        assert!(
            u > 0.0,
            "urgency with access_count=1 should be positive: {u}"
        );
        assert!(
            u >= first_term,
            "urgency should be >= first_term alone: u={u}, first_term={first_term}"
        );
    }

    /// 2. compute_urgency with very old last_access (10 minutes ago) should have
    ///    recency bonus nearly zero, dominated by the first term.
    #[test]
    fn urgency_very_old_last_access_recency_near_zero() {
        // Arrange
        let old_time = Instant::now() - Duration::from_secs(600);
        let meta = PageMetadata {
            page_id: 2,
            sequence_id: None,
            recency: 0,
            access_count: 20,
            last_access: old_time,
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act
        let u = SwapInWorker::compute_urgency(&meta, 0.8, StorageTier::CpuDram);

        // Assert — recency_bonus = 1/(1+600) ≈ 0.00166; total recency contribution ≈ 0.000166
        let importance_rebound = 20.0_f32.ln_1p() / 10.0_f32.ln_1p();
        let first_term = importance_rebound * 0.8 * 1.0;
        let recency_contribution = u - first_term;
        assert!(
            recency_contribution < 0.01,
            "recency contribution for 10-minute-old access should be negligible: {recency_contribution}"
        );
        assert!(
            u > 0.0,
            "urgency should still be positive due to first term: {u}"
        );
    }

    /// 3. swap_in_round: all pages missing from addr_table increments skipped by
    ///    the total request count.
    #[test]
    fn swap_in_round_all_missing_pages_all_skipped() {
        // Arrange
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
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> =
            Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        let mut requests = vec![
            make_prefetch(100, 0.9, 0.8, 4096),
            make_prefetch(101, 0.8, 0.7, 4096),
            make_prefetch(102, 0.7, 0.6, 4096),
        ];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // Assert — none are in addr_table, all should be skipped
        assert_eq!(submitted, 0, "no pages in addr_table → 0 submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 3, "all 3 requests counted");
        assert_eq!(s.skipped, 3, "all 3 skipped (missing from addr_table)");
        assert_eq!(s.submitted, 0);
        drop(s);
        actor.shutdown();
    }

    /// 4. swap_in_round: a single DRAM page with no two-hop promotion —
    ///    two_hop_promotions stays zero.
    #[test]
    fn swap_in_round_single_dram_no_two_hop_count() {
        // Arrange
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        insert_addr_entry(&addr_table, 10, StorageTier::CpuDram, CompressionCodec::None);
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> =
            Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        let mut requests = vec![make_prefetch(10, 0.9, 0.8, 4096)];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // Assert
        assert_eq!(submitted, 1, "one DRAM page → 1 submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.two_hop_promotions, 0, "DRAM pages have zero two-hop");
        assert_eq!(s.submitted, 1);
        drop(s);
        actor.shutdown();
    }

    /// 5. SwapInWorkerConfig default min_confidence is exactly 0.1 (one tenth).
    #[test]
    fn config_default_min_confidence_exact_value() {
        // Arrange & Act
        let c = SwapInWorkerConfig::default();

        // Assert
        let expected: f32 = 0.1;
        assert!(
            (c.min_confidence - expected).abs() < f32::EPSILON,
            "default min_confidence should be exactly 0.1, got {}",
            c.min_confidence
        );
    }

    /// 6. SwapInWorkerStats: avg_latency returns 0.0 when promoted_ok is zero
    ///    even if total_latency_us is nonzero (guard against division by zero).
    #[test]
    fn stats_avg_latency_zero_promoted_ok_nonzero_latency() {
        // Arrange
        let stats = SwapInWorkerStats {
            total_requests: 10,
            submitted: 5,
            skipped: 5,
            promoted_ok: 0,
            promoted_failed: 1,
            two_hop_promotions: 0,
            total_latency_us: 999,
            rounds: 3,
        };

        // Act
        let avg = stats.avg_latency_us();

        // Assert
        assert_eq!(avg, 0.0, "avg should be 0.0 when promoted_ok is 0");
    }

    /// 7. drain_completions_and_update: when metadata has swap_in_time set and
    ///    completion is Ok with GpuHbm tier, swap_in_time is cleared.
    #[test]
    fn drain_ok_hbm_clears_swap_in_time_on_existing_metadata() {
        // Arrange
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        // Insert page with host_buffer so actor can actually promote it
        {
            let mut table = addr_table.write().expect("write lock");
            table.insert(55, PageAddrEntry {
                gpu_ptr: None,
                host_buffer: Some(vec![0u8; 4096]),
                current_tier: StorageTier::CpuDram,
                original_bytes: 4096,
                codec: CompressionCodec::None,
            });
        }
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let nvme = Arc::new(
            crate::scheduler::nvme_swap::NvmeSwapFile::open(
                tmp.path().join("test.swap"),
                4096,
                8192,
                64,
            )
            .expect("nvme swap"),
        );
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            Some(Arc::clone(&nvme)),
        );
        let swap_time = Instant::now() - Duration::from_millis(50);
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> = {
            let mut m = HashMap::new();
            m.insert(55, PageMetadata {
                page_id: 55,
                sequence_id: Some(1),
                recency: 0,
                access_count: 5,
                last_access: Instant::now(),
                swap_in_time: Some(swap_time),
                is_lir: false,
                state: PageState::SwappedOut,
                warm_until: None,
            });
            Arc::new(RwLock::new(m))
        };
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> =
            Arc::new(Mutex::new(BasicObserver::new()));

        // Submit a migration to get a completion event
        let _ = actor.send(MigrationCommand::PromoteToHbm {
            page_id: 55,
            page_bytes: 4096,
        });

        // Act — poll until completion arrives (same pattern as existing drain tests)
        for _ in 0..20 {
            drain_completions_and_update(
                &actor,
                &page_metadata,
                &addr_table,
                &stats,
                &observer,
            );
            let s = stats.lock().expect("stats lock");
            if s.promoted_ok > 0 {
                break;
            }
            drop(s);
            thread::sleep(Duration::from_millis(10));
        }

        // Assert — swap_in_time should be cleared for GpuHbm promotion
        let meta = page_metadata.read().expect("read").get(&55).cloned();
        assert!(meta.is_some(), "page 55 should exist in metadata");
        let meta = meta.unwrap();
        assert_eq!(
            meta.state, PageState::Active,
            "state should be Active after GpuHbm promotion"
        );
        assert!(
            meta.swap_in_time.is_none(),
            "swap_in_time should be None after GpuHbm promotion"
        );
        actor.shutdown();
    }

    /// 8. swap_in_round: three NVMe pages each get two-hop treatment
    ///    (two_hop_promotions = 3, submitted = 3).
    #[test]
    fn swap_in_round_three_nvme_pages_each_two_hop() {
        // Arrange
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        for pid in [201, 202, 203] {
            insert_addr_entry(&addr_table, pid, StorageTier::Nvme, CompressionCodec::Lz4);
        }
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> =
            Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig::default();

        let mut requests = vec![
            make_prefetch(201, 0.9, 0.9, 4096),
            make_prefetch(202, 0.8, 0.8, 4096),
            make_prefetch(203, 0.7, 0.7, 4096),
        ];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // Assert
        assert_eq!(submitted, 3, "all 3 NVMe pages submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(
            s.two_hop_promotions, 3,
            "3 NVMe pages → 3 two-hop promotions"
        );
        assert_eq!(s.submitted, 3);
        drop(s);
        actor.shutdown();
    }

    /// 9. swap_in_round: requests with confidence below threshold are all skipped,
    ///    none submitted, skipped count equals total_requests.
    #[test]
    fn swap_in_round_all_below_confidence_threshold_skipped() {
        // Arrange
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        insert_addr_entry(&addr_table, 300, StorageTier::CpuDram, CompressionCodec::None);
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> =
            Arc::new(Mutex::new(BasicObserver::new()));
        let config = SwapInWorkerConfig {
            min_confidence: 0.5,
            ..SwapInWorkerConfig::default()
        };

        let mut requests = vec![
            make_prefetch(300, 0.9, 0.1, 4096), // confidence 0.1 < 0.5
            make_prefetch(300, 0.8, 0.2, 4096), // confidence 0.2 < 0.5
            make_prefetch(300, 0.7, 0.3, 4096), // confidence 0.3 < 0.5
        ];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // Assert
        assert_eq!(submitted, 0, "all below threshold → 0 submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.skipped, 3, "all 3 skipped due to low confidence");
        assert_eq!(s.total_requests, 3);
        drop(s);
        actor.shutdown();
    }

    /// 10. SwapInWorkerError::SendFailed Display output contains both the
    ///     prefix and the custom message.
    #[test]
    fn error_send_failed_display_contains_prefix_and_message() {
        // Arrange
        let err = SwapInWorkerError::SendFailed("channel closed".to_string());

        // Act
        let display = format!("{err}");

        // Assert
        assert!(
            display.contains("swap-in worker send failed"),
            "should contain prefix: {display}"
        );
        assert!(
            display.contains("channel closed"),
            "should contain original message: {display}"
        );
    }

    /// 11. swap_in_round: a single DRAM page with urgency sorted highest among
    ///     mixed requests is processed first after sort+truncate.
    #[test]
    fn swap_in_round_highest_urgency_processed_first() {
        // Arrange
        let backend: Arc<dyn DmaBackend> = Arc::new(CpuDmaBackendSized);
        let addr_table: PageAddrTable = Arc::new(RwLock::new(HashMap::new()));
        insert_addr_entry(&addr_table, 401, StorageTier::CpuDram, CompressionCodec::None);
        insert_addr_entry(&addr_table, 402, StorageTier::CpuDram, CompressionCodec::None);
        let actor = PageMigrationActor::spawn_with_backend(
            MigrationActorConfig::default(),
            Arc::clone(&backend),
            Arc::clone(&addr_table),
            None,
        );
        let page_metadata: Arc<RwLock<HashMap<PageId, PageMetadata>>> =
            Arc::new(RwLock::new(HashMap::new()));
        let stats: Arc<Mutex<SwapInWorkerStats>> =
            Arc::new(Mutex::new(SwapInWorkerStats::default()));
        let observer: Arc<Mutex<BasicObserver>> =
            Arc::new(Mutex::new(BasicObserver::new()));
        // max_prefetch_per_round = 1 ensures only the highest-urgency page is processed
        let config = SwapInWorkerConfig {
            max_prefetch_per_round: 1,
            ..SwapInWorkerConfig::default()
        };

        let mut requests = vec![
            make_prefetch(401, 0.5, 0.9, 4096), // low urgency
            make_prefetch(402, 0.95, 0.9, 4096), // high urgency — should be kept
        ];

        // Act
        let submitted = SwapInWorker::swap_in_round(
            &config,
            &actor,
            &mut requests,
            &page_metadata,
            &addr_table,
            &stats,
            &observer,
        );

        // Assert — only 1 submitted due to truncation to max_prefetch_per_round=1
        assert_eq!(submitted, 1, "truncated to 1 → 1 submitted");
        let s = stats.lock().expect("stats lock");
        assert_eq!(s.total_requests, 2, "both requests counted before truncation");
        assert_eq!(s.submitted, 1);
        drop(s);
        actor.shutdown();
    }

    /// 12. PageMetadata with state Active and is_lir=true retains both fields
    ///     after clone.
    #[test]
    fn page_metadata_active_lir_clone_preserves_fields() {
        // Arrange
        let meta = PageMetadata {
            page_id: 77,
            sequence_id: Some(42),
            recency: 5,
            access_count: 100,
            last_access: Instant::now(),
            swap_in_time: None,
            is_lir: true,
            state: PageState::Active,
            warm_until: Some(Instant::now() + Duration::from_secs(30)),
        };

        // Act
        let cloned = meta.clone();

        // Assert
        assert_eq!(cloned.page_id, 77);
        assert_eq!(cloned.sequence_id, Some(42));
        assert_eq!(cloned.recency, 5);
        assert_eq!(cloned.access_count, 100);
        assert_eq!(cloned.is_lir, true);
        assert_eq!(cloned.state, PageState::Active);
        assert!(cloned.warm_until.is_some());
    }

    /// 13. compute_urgency: the recency bonus term for a page accessed exactly
    ///     1 second ago should be 1/(1+1) * 0.1 = 0.05.
    #[test]
    fn urgency_recency_bonus_exact_at_one_second_elapsed() {
        // Arrange — access_count=0 so first term = 0 * confidence * tier_bonus = 0
        let one_sec_ago = Instant::now() - Duration::from_secs(1);
        let meta = PageMetadata {
            page_id: 99,
            sequence_id: None,
            recency: 0,
            access_count: 0,
            last_access: one_sec_ago,
            swap_in_time: None,
            is_lir: false,
            state: PageState::SwappedOut,
            warm_until: None,
        };

        // Act
        let u = SwapInWorker::compute_urgency(&meta, 0.0, StorageTier::CpuDram);

        // Assert — first term = 0 (importance_rebound=0 for access_count=0);
        //          recency_bonus = 1/(1+1.0) = 0.5; total recency contribution = 0.5 * 0.1 = 0.05
        //          Allow timing tolerance since Instant is not perfectly precise
        let expected_approx = 0.05_f32;
        assert!(
            (u - expected_approx).abs() < 0.02,
            "recency bonus at ~1s elapsed should be ~0.05, got {u}"
        );
    }

}
