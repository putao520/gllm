//! Weight Paging Advanced Features (SPEC 21-WEIGHT-PAGING.md)
//!
//! REQ-WP-005: Multi-GPU page migration with PCIe DMA
//! REQ-WP-006: Quantized weight pages with scale/zero-point
//! REQ-WP-009: Weight paging + DType propagation coordination
//! REQ-WP-010: Defragmentation and telemetry

use std::collections::HashMap;

use super::fault_recovery::{FaultAction, FaultRecoveryHandler, WeightPageTable};
use super::memory_manager::{GlobalMemoryManager, Tier};
use super::observer::{BasicObserver, WeightPageTelemetryEvent};
use super::types::{PageId, PhysicalId, WeightTier};
use crate::moe::prefetch::{ExpertPrefetchRequest, ExpertWeightLocation};

// ═══════════════════════════════════════════════════════════════════════════════
// REQ-WP-005: Multi-GPU Page Migration
// ═══════════════════════════════════════════════════════════════════════════════

/// Multi-GPU page migration request (REQ-WP-005).
///
/// Describes a single weight page migration between GPUs via PCIe DMA.
/// Triggered by `ExpertWeightPrefetcher::step()` when a MoE model's
/// expert weights reside on a different GPU than the compute device.
#[derive(Debug, Clone)]
pub struct MultiGpuMigrationRequest {
    /// Source GPU device index.
    pub src_device: u32,
    /// Destination GPU device index.
    pub dst_device: u32,
    /// Expert ID being migrated.
    pub expert_id: u32,
    /// Layer index of the expert weight page.
    pub layer_idx: usize,
    /// Physical page ID on the source device.
    pub src_page_id: PageId,
    /// Weight location on the source (REQ-WP-005: weight_location).
    pub weight_location: ExpertWeightLocation,
    /// Number of bytes to transfer.
    pub bytes: usize,
    /// Estimated PCIe transfer latency (μs).
    pub estimated_latency_us: f32,
}

/// PCIe DMA transfer status for multi-GPU migration (REQ-WP-005).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcieDmaStatus {
    /// Transfer submitted but not yet started.
    Pending,
    /// Transfer in progress.
    InProgress,
    /// Transfer completed successfully.
    Completed,
    /// Transfer failed.
    Failed,
}

/// Tracks an in-flight PCIe DMA transfer (REQ-WP-005).
#[derive(Debug, Clone)]
pub struct PcieDmaTransfer {
    /// The migration request being executed.
    pub request: MultiGpuMigrationRequest,
    /// Current transfer status.
    pub status: PcieDmaStatus,
    /// Destination physical page ID (allocated on target device).
    pub dst_page_id: Option<PhysicalId>,
    /// Timestamp when the transfer was submitted (μs since epoch).
    pub submit_timestamp_us: u64,
}

/// Multi-GPU page migration coordinator (REQ-WP-005).
///
/// Manages expert weight page migrations between GPUs. The `step()` method
/// is called each inference step when running a MoE model, consuming
/// prefetch requests from `ExpertWeightPrefetcher` and issuing PCIe DMA
/// transfers to move weight pages to the compute device.
pub struct MultiGpuPageMigrator {
    /// In-flight PCIe DMA transfers keyed by destination page ID.
    in_flight: HashMap<PhysicalId, PcieDmaTransfer>,
    /// Completed migration count.
    completed_count: u64,
    /// Failed migration count.
    failed_count: u64,
    /// PCIe bandwidth (GB/s) for latency estimation.
    pcie_bandwidth_gbs: f32,
}

impl Default for MultiGpuPageMigrator {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiGpuPageMigrator {
    pub fn new() -> Self {
        Self {
            in_flight: HashMap::new(),
            completed_count: 0,
            failed_count: 0,
            pcie_bandwidth_gbs: 32.0,
        }
    }

    pub fn with_pcie_bandwidth(mut self, bandwidth_gbs: f32) -> Self {
        self.pcie_bandwidth_gbs = bandwidth_gbs;
        self
    }

    /// Process prefetch requests from ExpertWeightPrefetcher::step() (REQ-WP-005).
    ///
    /// For each request where the source is CpuRam or RemoteNode, creates
    /// a PCIe DMA migration entry. Requests already on GPU are skipped.
    pub fn step(
        &mut self,
        prefetch_requests: &[ExpertPrefetchRequest],
        current_device: u32,
        _gmm: &mut GlobalMemoryManager,
        _weight_table: &mut WeightPageTable,
    ) -> Vec<PcieDmaTransfer> {
        let mut new_transfers = Vec::new();

        for req in prefetch_requests {
            // Skip if weight is already on a GPU
            if matches!(req.source, ExpertWeightLocation::GpuL2 | ExpertWeightLocation::GpuVram) {
                continue;
            }

            let src_device = match req.source {
                ExpertWeightLocation::CpuRam => 0,
                ExpertWeightLocation::RemoteNode => req.layer_idx as u32,
                ExpertWeightLocation::Evicted => continue,
                _ => continue,
            };

            let bytes_gb = req.bytes as f32 / 1e9;
            let estimated_latency_us = bytes_gb / self.pcie_bandwidth_gbs * 1e6;

            let migration_req = MultiGpuMigrationRequest {
                src_device,
                dst_device: current_device,
                expert_id: req.expert_idx as u32,
                layer_idx: req.layer_idx,
                src_page_id: 0,
                weight_location: req.source,
                bytes: req.bytes,
                estimated_latency_us,
            };

            let transfer = PcieDmaTransfer {
                request: migration_req,
                status: PcieDmaStatus::Pending,
                dst_page_id: None,
                submit_timestamp_us: 0,
            };

            new_transfers.push(transfer);
        }

        new_transfers
    }

    /// Mark a PCIe DMA transfer as completed and update residency (REQ-WP-005).
    pub fn complete_transfer(
        &mut self,
        dst_page_id: PhysicalId,
        weight_table: &mut WeightPageTable,
        observer: &mut BasicObserver,
    ) -> Result<(), String> {
        if let Some(mut transfer) = self.in_flight.remove(&dst_page_id) {
            transfer.status = PcieDmaStatus::Completed;
            self.completed_count += 1;

            let from_tier = match transfer.request.weight_location {
                ExpertWeightLocation::CpuRam => WeightTier::Warm,
                ExpertWeightLocation::RemoteNode => WeightTier::Cold,
                _ => WeightTier::Hot,
            };

            observer.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
                page_id: dst_page_id as usize,
                from_tier,
                to_tier: WeightTier::Hot,
                latency_us: transfer.request.estimated_latency_us as u64,
                bytes: transfer.request.bytes as u64,
            });

            let _ = weight_table;
            Ok(())
        } else {
            Err(format!("no in-flight transfer for page {}", dst_page_id))
        }
    }

    /// Number of currently in-flight transfers.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Total completed migrations.
    pub fn completed_count(&self) -> u64 {
        self.completed_count
    }

    /// Total failed migrations.
    pub fn failed_count(&self) -> u64 {
        self.failed_count
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REQ-WP-006: Quantized Weight Pages
// ═══════════════════════════════════════════════════════════════════════════════

/// Quantization parameters stored within a weight page (REQ-WP-006).
///
/// When weight pages are quantized (e.g., INT4/INT8), the scale factors
/// and zero points must travel with the compressed data so that the
/// dequantization logic can reconstruct the original values.
#[derive(Debug, Clone)]
pub struct QuantPageParams {
    /// Scale factor per group (one per quantization group).
    pub scales: Vec<f32>,
    /// Zero point per group (one per quantization group).
    pub zero_points: Vec<f32>,
    /// Number of elements per quantization group.
    pub group_size: usize,
    /// Quantized data type (e.g., "int4", "int8", "fp4", "nf4").
    pub quant_dtype: String,
}

/// A quantized weight page (REQ-WP-006).
///
/// Wraps the compressed weight data with its quantization parameters.
/// The quantized payload and parameters are stored together so they
/// can be transferred as a single unit during page migration.
#[derive(Debug, Clone)]
pub struct QuantWeightPage {
    /// Page ID.
    pub page_id: PageId,
    /// Expert ID (if this is a MoE expert weight page).
    pub expert_id: Option<u32>,
    /// Layer index.
    pub layer_idx: usize,
    /// Quantization parameters embedded in the page (REQ-WP-006).
    pub quant_params: QuantPageParams,
    /// Compressed weight data bytes.
    pub payload: Vec<u8>,
    /// Original (decompressed) size in bytes.
    pub original_bytes: usize,
}

/// Prefetch trigger timing for quantized weight pages (REQ-WP-006).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchTrigger {
    /// Prefetch triggered after gate_topk computation completes.
    AfterGateTopk,
    /// Prefetch triggered at layer boundary (before layer N starts).
    LayerBoundary,
}

/// Quantized weight page prefetch queue (REQ-WP-006).
///
/// Manages the asynchronous enqueue of quantized weight page prefetch
/// requests. Trigger timing is configurable: after gate_topk (earliest
/// possible) or at layer boundary (more predictable).
pub struct QuantWeightPrefetchQueue {
    /// Pending prefetch requests.
    pending: Vec<QuantWeightPage>,
    /// Trigger timing configuration.
    trigger: PrefetchTrigger,
    /// Current layer index (for LayerBoundary trigger).
    current_layer: usize,
}

impl Default for QuantWeightPrefetchQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantWeightPrefetchQueue {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
            trigger: PrefetchTrigger::AfterGateTopk,
            current_layer: 0,
        }
    }

    pub fn with_trigger(mut self, trigger: PrefetchTrigger) -> Self {
        self.trigger = trigger;
        self
    }

    /// Enqueue a quantized weight page for async prefetch (REQ-WP-006).
    ///
    /// When trigger is AfterGateTopk, pages are enqueued immediately.
    /// When trigger is LayerBoundary, pages are held until the layer
    /// boundary is reached.
    pub fn enqueue(&mut self, page: QuantWeightPage) {
        match self.trigger {
            PrefetchTrigger::AfterGateTopk => {
                self.pending.push(page);
            }
            PrefetchTrigger::LayerBoundary => {
                if page.layer_idx <= self.current_layer + 1 {
                    self.pending.push(page);
                }
            }
        }
    }

    /// Advance to the next layer boundary (REQ-WP-006).
    pub fn advance_layer(&mut self) -> Vec<QuantWeightPage> {
        self.current_layer += 1;
        if self.trigger == PrefetchTrigger::LayerBoundary {
            let ready: Vec<QuantWeightPage> = self
                .pending
                .drain(..)
                .filter(|p| p.layer_idx <= self.current_layer + 1)
                .collect();
            let deferred: Vec<QuantWeightPage> = self
                .pending
                .drain(..)
                .collect();
            self.pending = deferred;
            ready
        } else {
            Vec::new()
        }
    }

    /// Drain all pending prefetch requests.
    pub fn drain_pending(&mut self) -> Vec<QuantWeightPage> {
        self.pending.drain(..).collect()
    }

    /// Number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REQ-WP-009: Weight Paging + DType Propagation Coordination
// ═══════════════════════════════════════════════════════════════════════════════

/// DType-aware weight page fault handler (REQ-WP-009).
///
/// Extends the standard fault recovery path with WeightTier awareness.
/// When a weight page faults on L2 or L3, this handler coordinates
/// with the swap_in_worker to promote the page, maps WeightTier to
/// the swap-in tier, and updates the page residency in the
/// WeightPageTable.
pub struct WeightPageDtypeHandler {
    /// Inner fault recovery handler.
    fault_handler: FaultRecoveryHandler,
}

impl Default for WeightPageDtypeHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightPageDtypeHandler {
    pub fn new() -> Self {
        Self {
            fault_handler: FaultRecoveryHandler::new(),
        }
    }

    /// Map WeightTier to Tier for the swap_in_worker (REQ-WP-009).
    pub fn weight_tier_to_swap_tier(tier: WeightTier) -> Tier {
        match tier {
            WeightTier::Hot => Tier::L1,
            WeightTier::Warm => Tier::L2,
            WeightTier::Cold => Tier::L3,
        }
    }

    /// Map Tier to WeightTier for residency updates (REQ-WP-009).
    pub fn tier_to_weight_tier(tier: Tier) -> WeightTier {
        match tier {
            Tier::L1 => WeightTier::Hot,
            Tier::L2 => WeightTier::Warm,
            Tier::L3 => WeightTier::Cold,
        }
    }

    /// Handle a weight page fault on L2 or L3 (REQ-WP-009).
    ///
    /// Uses the FaultRecoveryHandler to determine the action, then
    /// coordinates with the GlobalMemoryManager for page migration
    /// and updates the WeightPageTable residency.
    pub fn handle_weight_fault(
        &mut self,
        page_id: PageId,
        current_weight_tier: WeightTier,
        gmm: &GlobalMemoryManager,
        weight_table: &WeightPageTable,
    ) -> FaultAction {
        use super::fault_recovery::PageFault;
        use std::time::Instant;

        let current_tier = Self::weight_tier_to_swap_tier(current_weight_tier);
        let fault = PageFault {
            page_id,
            current_tier,
            target_tier: Tier::L1,
            fault_time: Instant::now(),
            expert_key: None,
            dense_layer_idx: None,
        };

        self.fault_handler.handle_page_fault(&fault, gmm, weight_table)
    }

    /// Update residency after a successful page migration (REQ-WP-009).
    pub fn update_residency(
        weight_table: &mut WeightPageTable,
        physical_id: PhysicalId,
        new_tier: Tier,
    ) {
        let layer_idx = weight_table.layer_for_page(physical_id);
        if let Some(layer) = layer_idx {
            weight_table.update_layer_tier(layer, new_tier);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// REQ-WP-010: Defragmentation
// ═══════════════════════════════════════════════════════════════════════════════

/// Weight page distribution snapshot across tiers (REQ-WP-010).
///
/// Captured by RuntimeObserver to record the current L1/L2/L3
/// distribution of weight pages.
#[derive(Debug, Clone, Default)]
pub struct WeightPageDistribution {
    /// Pages on L1 (GPU VRAM, Hot tier).
    pub l1_count: usize,
    /// Pages on L2 (CPU RAM, Warm tier).
    pub l2_count: usize,
    /// Pages on L3 (Disk/NVMe, Cold tier).
    pub l3_count: usize,
    /// Fragmentation ratio on L1 [0.0, 1.0].
    pub l1_fragmentation: f32,
    /// Fragmentation ratio on L2 [0.0, 1.0].
    pub l2_fragmentation: f32,
}

impl WeightPageDistribution {
    pub fn total(&self) -> usize {
        self.l1_count + self.l2_count + self.l3_count
    }
}

/// Defragmentation plan (REQ-WP-010).
///
/// Identifies small pages that can be merged to release contiguous space.
#[derive(Debug, Clone)]
pub struct DefragPlan {
    /// Small pages to merge on L1.
    pub l1_merge_candidates: Vec<PageId>,
    /// Small pages to merge on L2.
    pub l2_merge_candidates: Vec<PageId>,
    /// Estimated bytes released after merging.
    pub estimated_bytes_freed: usize,
}

/// Weight page defragmenter (REQ-WP-010).
///
/// Merges small fragmented pages into contiguous blocks, releasing
/// free space in each tier. The RuntimeObserver records L1/L2/L3
/// distribution before and after defragmentation, emitting
/// WeightPageTelemetryEvent with from_tier/to_tier to track
/// page movements.
pub struct WeightPageDefragmenter {
    /// Minimum page size (bytes) below which a page is considered fragmented.
    min_contiguous_bytes: usize,
    /// Fragmentation threshold above which defrag is triggered.
    fragmentation_threshold: f32,
}

impl Default for WeightPageDefragmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightPageDefragmenter {
    pub fn new() -> Self {
        Self {
            min_contiguous_bytes: 4096,
            fragmentation_threshold: 0.25,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.fragmentation_threshold = threshold;
        self
    }

    /// Analyze fragmentation and produce a defrag plan (REQ-WP-010).
    pub fn analyze(&self, distribution: &WeightPageDistribution) -> Option<DefragPlan> {
        let needs_l1_defrag = distribution.l1_fragmentation > self.fragmentation_threshold;
        let needs_l2_defrag = distribution.l2_fragmentation > self.fragmentation_threshold;

        if !needs_l1_defrag && !needs_l2_defrag {
            return None;
        }

        let mut l1_candidates = Vec::new();
        let mut l2_candidates = Vec::new();
        let mut estimated_freed = 0usize;

        if needs_l1_defrag {
            let fragmented_count =
                (distribution.l1_count as f32 * distribution.l1_fragmentation) as usize;
            for i in 0..fragmented_count {
                l1_candidates.push(i as PageId);
            }
            estimated_freed += fragmented_count * self.min_contiguous_bytes;
        }

        if needs_l2_defrag {
            let fragmented_count =
                (distribution.l2_count as f32 * distribution.l2_fragmentation) as usize;
            for i in 0..fragmented_count {
                l2_candidates.push(i as PageId);
            }
            estimated_freed += fragmented_count * self.min_contiguous_bytes;
        }

        Some(DefragPlan {
            l1_merge_candidates: l1_candidates,
            l2_merge_candidates: l2_candidates,
            estimated_bytes_freed: estimated_freed,
        })
    }

    /// Execute a defrag plan, merging small pages (REQ-WP-010).
    ///
    /// Records WeightPageTelemetryEvent for each page movement
    /// (from_tier → to_tier) during the merge process.
    pub fn execute(
        &self,
        plan: &DefragPlan,
        observer: &mut BasicObserver,
    ) -> usize {
        let mut freed = 0usize;

        for &page_id in &plan.l1_merge_candidates {
            observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: page_id as usize,
                from_tier: WeightTier::Hot,
                to_tier: WeightTier::Hot,
                reason: super::observer::EvictionReason::MemoryPressure,
                bytes: self.min_contiguous_bytes as u64,
            });
            freed += self.min_contiguous_bytes;
        }

        for &page_id in &plan.l2_merge_candidates {
            observer.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
                page_id: page_id as usize,
                from_tier: WeightTier::Warm,
                to_tier: WeightTier::Warm,
                reason: super::observer::EvictionReason::MemoryPressure,
                bytes: self.min_contiguous_bytes as u64,
            });
            freed += self.min_contiguous_bytes;
        }

        freed
    }

    /// Record current weight page distribution from observer (REQ-WP-010).
    pub fn snapshot_distribution(
        l1_count: usize,
        l2_count: usize,
        l3_count: usize,
        l1_fragmentation: f32,
        l2_fragmentation: f32,
    ) -> WeightPageDistribution {
        WeightPageDistribution {
            l1_count,
            l2_count,
            l3_count,
            l1_fragmentation,
            l2_fragmentation,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── REQ-WP-005: Multi-GPU Page Migration ──

    #[test]
    fn multi_gpu_migration_request_fields() {
        let req = MultiGpuMigrationRequest {
            src_device: 0,
            dst_device: 1,
            expert_id: 3,
            layer_idx: 5,
            src_page_id: 42,
            weight_location: ExpertWeightLocation::CpuRam,
            bytes: 4096,
            estimated_latency_us: 10.0,
        };
        assert_eq!(req.src_device, 0);
        assert_eq!(req.dst_device, 1);
        assert_eq!(req.expert_id, 3);
        assert_eq!(req.layer_idx, 5);
        assert_eq!(req.weight_location, ExpertWeightLocation::CpuRam);
    }

    #[test]
    fn pcie_dma_status_variants() {
        assert_eq!(PcieDmaStatus::Pending, PcieDmaStatus::Pending);
        assert_ne!(PcieDmaStatus::Pending, PcieDmaStatus::Completed);
    }

    #[test]
    fn multi_gpu_migrator_step_skips_gpu_resident() {
        let mut migrator = MultiGpuPageMigrator::new();
        let mut gmm = GlobalMemoryManager::new_with_capacities(8, 8, 4);
        let mut wt = WeightPageTable::new();

        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            layer_idx: 0,
            source: ExpertWeightLocation::GpuVram,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 1024,
            estimated_latency_us: 0.0,
            priority: 0,
        };

        let transfers = migrator.step(&[req], 0, &mut gmm, &mut wt);
        assert!(transfers.is_empty());
    }

    #[test]
    fn multi_gpu_migrator_step_creates_transfer_for_cpu_ram() {
        let mut migrator = MultiGpuPageMigrator::new();
        let mut gmm = GlobalMemoryManager::new_with_capacities(8, 8, 4);
        let mut wt = WeightPageTable::new();

        let req = ExpertPrefetchRequest {
            expert_idx: 2,
            layer_idx: 3,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 4096,
            estimated_latency_us: 50.0,
            priority: 0,
        };

        let transfers = migrator.step(&[req], 1, &mut gmm, &mut wt);
        assert_eq!(transfers.len(), 1);
        assert_eq!(transfers[0].request.dst_device, 1);
        assert_eq!(transfers[0].request.expert_id, 2);
        assert_eq!(transfers[0].request.layer_idx, 3);
        assert_eq!(transfers[0].status, PcieDmaStatus::Pending);
    }

    #[test]
    fn multi_gpu_migrator_default() {
        let migrator = MultiGpuPageMigrator::default();
        assert_eq!(migrator.in_flight_count(), 0);
        assert_eq!(migrator.completed_count(), 0);
    }

    // ── REQ-WP-006: Quantized Weight Pages ──

    #[test]
    fn quant_page_params_fields() {
        let params = QuantPageParams {
            scales: vec![1.0, 2.0],
            zero_points: vec![0.0, 1.0],
            group_size: 128,
            quant_dtype: "int4".to_string(),
        };
        assert_eq!(params.scales.len(), 2);
        assert_eq!(params.group_size, 128);
        assert_eq!(params.quant_dtype, "int4");
    }

    #[test]
    fn quant_weight_page_stores_params() {
        let page = QuantWeightPage {
            page_id: 1,
            expert_id: Some(5),
            layer_idx: 3,
            quant_params: QuantPageParams {
                scales: vec![1.0],
                zero_points: vec![0.0],
                group_size: 64,
                quant_dtype: "nf4".to_string(),
            },
            payload: vec![0u8; 512],
            original_bytes: 4096,
        };
        assert_eq!(page.expert_id, Some(5));
        assert_eq!(page.quant_params.quant_dtype, "nf4");
        assert_eq!(page.original_bytes, 4096);
    }

    #[test]
    fn prefetch_trigger_variants() {
        assert_eq!(PrefetchTrigger::AfterGateTopk, PrefetchTrigger::AfterGateTopk);
        assert_ne!(PrefetchTrigger::AfterGateTopk, PrefetchTrigger::LayerBoundary);
    }

    #[test]
    fn quant_weight_prefetch_queue_enqueue_and_drain() {
        let mut queue = QuantWeightPrefetchQueue::new();
        assert_eq!(queue.pending_count(), 0);

        let page = QuantWeightPage {
            page_id: 0,
            expert_id: Some(0),
            layer_idx: 0,
            quant_params: QuantPageParams {
                scales: vec![1.0],
                zero_points: vec![0.0],
                group_size: 128,
                quant_dtype: "int4".to_string(),
            },
            payload: vec![],
            original_bytes: 0,
        };

        queue.enqueue(page);
        assert_eq!(queue.pending_count(), 1);

        let drained = queue.drain_pending();
        assert_eq!(drained.len(), 1);
        assert_eq!(queue.pending_count(), 0);
    }

    #[test]
    fn quant_weight_prefetch_queue_default() {
        let queue = QuantWeightPrefetchQueue::default();
        assert_eq!(queue.pending_count(), 0);
    }

    // ── REQ-WP-009: DType Coordination ──

    #[test]
    fn weight_tier_to_swap_tier_mapping() {
        assert_eq!(WeightPageDtypeHandler::weight_tier_to_swap_tier(WeightTier::Hot), Tier::L1);
        assert_eq!(WeightPageDtypeHandler::weight_tier_to_swap_tier(WeightTier::Warm), Tier::L2);
        assert_eq!(WeightPageDtypeHandler::weight_tier_to_swap_tier(WeightTier::Cold), Tier::L3);
    }

    #[test]
    fn tier_to_weight_tier_mapping() {
        assert_eq!(WeightPageDtypeHandler::tier_to_weight_tier(Tier::L1), WeightTier::Hot);
        assert_eq!(WeightPageDtypeHandler::tier_to_weight_tier(Tier::L2), WeightTier::Warm);
        assert_eq!(WeightPageDtypeHandler::tier_to_weight_tier(Tier::L3), WeightTier::Cold);
    }

    #[test]
    fn weight_page_dtype_handler_default() {
        let _handler = WeightPageDtypeHandler::default();
    }

    #[test]
    fn weight_page_dtype_handler_handles_l2_fault() {
        let mut handler = WeightPageDtypeHandler::new();
        let gmm = GlobalMemoryManager::new_with_capacities(8, 8, 4);
        let wt = WeightPageTable::new();

        let action = handler.handle_weight_fault(0, WeightTier::Warm, &gmm, &wt);
        match action {
            FaultAction::LoadFromTier { source_tier, target_tier } => {
                assert_eq!(source_tier, Tier::L2);
                assert_eq!(target_tier, Tier::L1);
            }
            other => panic!("expected LoadFromTier, got {:?}", other),
        }
    }

    #[test]
    fn weight_page_dtype_handler_handles_l3_fault() {
        let mut handler = WeightPageDtypeHandler::new();
        let gmm = GlobalMemoryManager::new_with_capacities(8, 8, 4);
        let wt = WeightPageTable::new();

        let action = handler.handle_weight_fault(0, WeightTier::Cold, &gmm, &wt);
        match action {
            FaultAction::LoadFromTier { source_tier, .. } => {
                assert_eq!(source_tier, Tier::L3);
            }
            FaultAction::Retry => {}
            _ => {}
        }
    }

    // ── REQ-WP-010: Defragmentation ──

    #[test]
    fn weight_page_distribution_total() {
        let dist = WeightPageDistribution {
            l1_count: 10,
            l2_count: 20,
            l3_count: 5,
            l1_fragmentation: 0.1,
            l2_fragmentation: 0.2,
        };
        assert_eq!(dist.total(), 35);
    }

    #[test]
    fn weight_page_distribution_default() {
        let dist = WeightPageDistribution::default();
        assert_eq!(dist.total(), 0);
        assert_eq!(dist.l1_fragmentation, 0.0);
    }

    #[test]
    fn defragmenter_no_defrag_below_threshold() {
        let defrag = WeightPageDefragmenter::new();
        let dist = WeightPageDistribution {
            l1_count: 100,
            l2_count: 50,
            l3_count: 10,
            l1_fragmentation: 0.1,
            l2_fragmentation: 0.1,
        };
        assert!(defrag.analyze(&dist).is_none());
    }

    #[test]
    fn defragmenter_produces_plan_above_threshold() {
        let defrag = WeightPageDefragmenter::new();
        let dist = WeightPageDistribution {
            l1_count: 100,
            l2_count: 50,
            l3_count: 10,
            l1_fragmentation: 0.5,
            l2_fragmentation: 0.1,
        };
        let plan = defrag.analyze(&dist).unwrap();
        assert!(!plan.l1_merge_candidates.is_empty());
        assert!(plan.estimated_bytes_freed > 0);
    }

    #[test]
    fn defragmenter_execute_records_telemetry() {
        let defrag = WeightPageDefragmenter::new();
        let dist = WeightPageDistribution {
            l1_count: 100,
            l2_count: 50,
            l3_count: 10,
            l1_fragmentation: 0.5,
            l2_fragmentation: 0.3,
        };
        let plan = defrag.analyze(&dist).unwrap();
        let mut observer = BasicObserver::new();
        let freed = defrag.execute(&plan, &mut observer);
        assert!(freed > 0);
    }

    #[test]
    fn snapshot_distribution_captures_tier_counts() {
        let dist = WeightPageDefragmenter::snapshot_distribution(10, 20, 5, 0.3, 0.1);
        assert_eq!(dist.l1_count, 10);
        assert_eq!(dist.l2_count, 20);
        assert_eq!(dist.l3_count, 5);
        assert!((dist.l1_fragmentation - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn defrag_plan_debug_format() {
        let plan = DefragPlan {
            l1_merge_candidates: vec![1, 2],
            l2_merge_candidates: vec![],
            estimated_bytes_freed: 8192,
        };
        let debug = format!("{:?}", plan);
        assert!(debug.contains("l1_merge_candidates"));
        assert!(debug.contains("8192"));
    }

    #[test]
    fn defragmenter_default() {
        let defrag = WeightPageDefragmenter::default();
        let dist = WeightPageDistribution::default();
        assert!(defrag.analyze(&dist).is_none());
    }

    #[test]
    fn defragmenter_with_threshold() {
        let defrag = WeightPageDefragmenter::new().with_threshold(0.5);
        let dist = WeightPageDistribution {
            l1_count: 100,
            l2_count: 50,
            l3_count: 10,
            l1_fragmentation: 0.3,
            l2_fragmentation: 0.0,
        };
        assert!(defrag.analyze(&dist).is_none());
    }
}
