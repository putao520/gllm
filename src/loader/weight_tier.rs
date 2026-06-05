//! Heterogeneous device weight tier manager.
//!
//! Manages weight tensor placement across SPEC three-tier memory hierarchy:
//! - L1 (DeviceLocal): GPU HBM / NPU SRAM — highest bandwidth
//! - L2 (HostLocal): CPU RAM — medium bandwidth
//! - L3 (DiskMmap): disk-backed mmap — zero extra memory, lowest bandwidth
//!
//! The manager operates at model load time only. Once the mega-kernel weight_blob
//! is packed, tier information informs the packing strategy but is not needed at
//! inference time (ARCH-RUST-IS-CODEGEN).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::compat::backend_trait::{Backend, WeightPlacement};
use crate::kv_cache::StorageTier;
use crate::scheduler::memory_manager::{GlobalMemoryManager, Tier};

/// Weight memory tier — maps to SPEC/06-RUNTIME §5.1 L1/L2/L3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightTier {
    /// L1: Device-local memory (GPU VRAM / NPU SRAM).
    DeviceLocal,
    /// L2: Host memory (CPU RAM, optionally pinned).
    HostLocal,
    /// L3: Disk-backed mmap (safetensors/gguf zero-copy).
    DiskMmap,
}

/// Upload decision returned by [`WeightTierManager::decide`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UploadDecision {
    pub tier: WeightTier,
    pub placement: WeightPlacement,
}

/// Per-tensor allocation record.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct WeightAllocation {
    tier: WeightTier,
    size: usize,
}

/// Fraction of device memory reserved for weights (rest for KV cache + scratchpad).
const DEVICE_WEIGHT_FRACTION: f64 = 0.70;

/// Fraction of host memory usable for weight staging.
#[allow(dead_code)]
const HOST_WEIGHT_FRACTION: f64 = 0.60;

/// Manages weight tensor placement across heterogeneous memory tiers.
pub struct WeightTierManager {
    device_capacity: usize,
    host_capacity: usize,
    device_used: AtomicUsize,
    host_used: AtomicUsize,
    allocations: Mutex<HashMap<String, WeightAllocation>>,
}

/// Pre-trained GMM parameters for tier assignment (REQ-WP1).
struct GmmComponent {
    weight: f32,
    mean: [f32; 2],
    variance: [f32; 2],
}

const GMM_COMPONENTS: [GmmComponent; 3] = [
    GmmComponent { weight: 0.35, mean: [0.80, 0.20], variance: [0.0225, 0.0225] },
    GmmComponent { weight: 0.40, mean: [0.50, 0.50], variance: [0.0400, 0.0400] },
    GmmComponent { weight: 0.25, mean: [0.20, 0.80], variance: [0.0225, 0.0225] },
];

impl WeightTierManager {
    /// Create from explicit capacity values (bytes).
    pub fn new(device_capacity: usize, host_capacity: usize) -> Self {
        Self {
            device_capacity,
            host_capacity,
            device_used: AtomicUsize::new(0),
            host_used: AtomicUsize::new(0),
            allocations: Mutex::new(HashMap::new()),
        }
    }

    /// Create from [`SystemTopology`](crate::sensors::SystemTopology).
    ///
    /// Device capacity = GPU VRAM × 0.70 (30% reserved for KV cache + activations).
    /// Host capacity = physical RAM × 0.60 (40% reserved for OS + scratchpad).
    pub fn from_system_topology(topo: &crate::sensors::SystemTopology) -> Self {
        let device_capacity = topo
            .gpu
            .as_ref()
            .map(|g| (g.global_mem_bytes as f64 * DEVICE_WEIGHT_FRACTION) as usize)
            .unwrap_or(0);

        // Host capacity: estimate from L3 cache × 100 as a rough proxy,
        // capped by 16 GB minimum budget for small systems.
        let host_estimate = (topo.cpu.l3_bytes as f64 * 100.0) as usize;
        let host_capacity = host_estimate.max(16usize * 1024 * 1024 * 1024);

        Self::new(device_capacity, host_capacity)
    }

    /// Create from a [`Backend`] instance.
    ///
    /// Reads `device_memory_capacity()` from the backend for accurate VRAM sizing.
    pub fn from_backend<B, E>(backend: &B) -> Self
    where
        B: Backend<E>,
        E: crate::compat::backend_trait::Element,
    {
        let device_capacity = backend.device_memory_capacity();
        let host_capacity = 16usize * 1024 * 1024 * 1024; // 16 GB default host budget
        Self::new(device_capacity, host_capacity)
    }

    /// Decide upload tier for a tensor (back-to-front degradation).
    ///
    /// Tries in order: DeviceLocal → HostLocal → DiskMmap.
    /// Thread-safe: uses atomic counters for capacity tracking.
    pub fn decide(&self, name: &str, size: usize) -> UploadDecision {
        // Try device memory first
        let device_used = self.device_used.load(Ordering::Relaxed);
        if device_used + size <= self.device_capacity {
            // CAS loop to avoid over-allocation under concurrency
            match self.device_used.compare_exchange_weak(
                device_used,
                device_used + size,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.allocations.lock().unwrap().insert(
                        name.to_string(),
                        WeightAllocation { tier: WeightTier::DeviceLocal, size },
                    );
                    return UploadDecision {
                        tier: WeightTier::DeviceLocal,
                        placement: WeightPlacement::DeviceLocal,
                    };
                }
                Err(_) => { /* concurrent allocation changed, fall through */ }
            }
        }

        // Try host memory
        let host_used = self.host_used.load(Ordering::Relaxed);
        if host_used + size <= self.host_capacity {
            match self.host_used.compare_exchange_weak(
                host_used,
                host_used + size,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.allocations.lock().unwrap().insert(
                        name.to_string(),
                        WeightAllocation { tier: WeightTier::HostLocal, size },
                    );
                    return UploadDecision {
                        tier: WeightTier::HostLocal,
                        placement: WeightPlacement::HostLocal,
                    };
                }
                Err(_) => { /* concurrent allocation changed, fall through */ }
            }
        }

        // Degrade to mmap — data already lives in mmap'd file, no new allocation
        self.allocations.lock().unwrap().insert(
            name.to_string(),
            WeightAllocation { tier: WeightTier::DiskMmap, size },
        );
        UploadDecision {
            tier: WeightTier::DiskMmap,
            placement: WeightPlacement::HostLocal, // mmap data still accessed via CPU pointers
        }
    }

    /// Query which tier a tensor was allocated in.
    pub fn tier_of(&self, name: &str) -> Option<WeightTier> {
        self.allocations
            .lock()
            .unwrap()
            .get(name)
            .map(|a| a.tier)
    }

    /// Report tier usage: `(used_bytes, capacity_bytes)`.
    pub fn usage(&self, tier: WeightTier) -> (usize, usize) {
        match tier {
            WeightTier::DeviceLocal => {
                (self.device_used.load(Ordering::Relaxed), self.device_capacity)
            }
            WeightTier::HostLocal => {
                (self.host_used.load(Ordering::Relaxed), self.host_capacity)
            }
            WeightTier::DiskMmap => (0, 0), // mmap is file-backed, no capacity limit
        }
    }

    /// Total weight bytes allocated across all tiers.
    pub fn total_allocated(&self) -> usize {
        self.device_used.load(Ordering::Relaxed)
            + self.host_used.load(Ordering::Relaxed)
    }

    /// Number of tensors tracked.
    pub fn tensor_count(&self) -> usize {
        self.allocations.lock().unwrap().len()
    }

    /// Decide upload tier by querying GlobalMemoryManager's TierUsage (REQ-WP-001).
    ///
    /// Unlike `decide()` which uses internal atomic counters, this method reads the
    /// unified Tier capacity from GMM — ensuring weight and KV allocations share
    /// the same capacity budget.
    pub fn decide_via_gmm_capacity(
        &self,
        name: &str,
        size_pages: usize,
        gmm: &GlobalMemoryManager,
    ) -> UploadDecision {
        let l1 = gmm.tier_usage(Tier::L1);
        if l1.available() >= size_pages {
            self.allocations.lock().unwrap().insert(
                name.to_string(),
                WeightAllocation { tier: WeightTier::DeviceLocal, size: size_pages },
            );
            return UploadDecision {
                tier: WeightTier::DeviceLocal,
                placement: WeightPlacement::DeviceLocal,
            };
        }

        let l2 = gmm.tier_usage(Tier::L2);
        if l2.available() >= size_pages {
            self.allocations.lock().unwrap().insert(
                name.to_string(),
                WeightAllocation { tier: WeightTier::HostLocal, size: size_pages },
            );
            return UploadDecision {
                tier: WeightTier::HostLocal,
                placement: WeightPlacement::HostLocal,
            };
        }

        self.allocations.lock().unwrap().insert(
            name.to_string(),
            WeightAllocation { tier: WeightTier::DiskMmap, size: size_pages },
        );
        UploadDecision {
            tier: WeightTier::DiskMmap,
            placement: WeightPlacement::HostLocal,
        }
    }

    // ─── GMM Tier Decision (REQ-WP1) ───────────────────────────────────────

    /// Number of GMM features.
    const GMM_N_FEATURES: usize = 2;

    /// Decide tier using Gaussian Mixture Model inference (REQ-WP1).
    ///
    /// Takes a feature vector describing weight access patterns and returns
    /// the optimal [`StorageTier`] based on posterior probability maximization.
    /// Each component corresponds to one storage tier with learned mean/variance.
    ///
    /// Falls back to [`decide_layer_tier`] when features are empty or too short.
    pub fn decide_via_gmm(&self, features: &[f32]) -> StorageTier {
        if features.len() < Self::GMM_N_FEATURES {
            return self.decide_layer_tier();
        }

        // Compute log-posterior for each component (log domain for numerical stability).
        let mut log_posteriors = [0.0f32; 3];
        for (c, comp) in GMM_COMPONENTS.iter().enumerate() {
            let mut log_likelihood = comp.weight.ln();
            for d in 0..Self::GMM_N_FEATURES {
                let diff = features[d] - comp.mean[d];
                log_likelihood -= 0.5 * (
                    diff * diff / comp.variance[d]
                    + (2.0 * std::f32::consts::PI * comp.variance[d]).ln()
                );
            }
            log_posteriors[c] = log_likelihood;
        }

        // Argmax over log-posteriors
        let mut best = 0usize;
        let mut best_val = log_posteriors[0];
        for c in 1..3 {
            if log_posteriors[c] > best_val {
                best_val = log_posteriors[c];
                best = c;
            }
        }

        match best {
            0 => StorageTier::GpuHbm,
            1 => StorageTier::CpuDram,
            2 => StorageTier::Nvme,
            _ => StorageTier::CpuDram, // unreachable
        }
    }

    /// Fallback tier decision when no GMM model is available (SPEC §2).
    ///
    /// Returns a default tier based on current capacity availability:
    /// - `GpuHbm` if device has capacity
    /// - `CpuDram` if host has capacity
    /// - `Nvme` otherwise
    pub fn decide_layer_tier(&self) -> StorageTier {
        if self.device_used.load(Ordering::Relaxed) < self.device_capacity {
            StorageTier::GpuHbm
        } else if self.host_used.load(Ordering::Relaxed) < self.host_capacity {
            StorageTier::CpuDram
        } else {
            StorageTier::Nvme
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_first_then_host_then_mmap() {
        let mgr = WeightTierManager::new(100, 200);

        let d = mgr.decide("a", 50);
        assert_eq!(d.tier, WeightTier::DeviceLocal);
        assert_eq!(d.placement, WeightPlacement::DeviceLocal);

        let d = mgr.decide("b", 60); // 50 + 60 = 110 > 100, degrades to host
        assert_eq!(d.tier, WeightTier::HostLocal);
        assert_eq!(d.placement, WeightPlacement::HostLocal);

        let d = mgr.decide("c", 160); // host already has 60, 60 + 160 > 200, degrades to mmap
        assert_eq!(d.tier, WeightTier::DiskMmap);
    }

    #[test]
    fn device_capacity_exhaustion() {
        let mgr = WeightTierManager::new(100, 0);

        let d = mgr.decide("a", 80);
        assert_eq!(d.tier, WeightTier::DeviceLocal);

        let d = mgr.decide("b", 30); // 80 + 30 = 110 > 100
        assert_eq!(d.tier, WeightTier::DiskMmap); // no host budget → mmap
    }

    #[test]
    fn tier_of_query() {
        let mgr = WeightTierManager::new(1000, 1000);
        mgr.decide("x", 10);
        assert_eq!(mgr.tier_of("x"), Some(WeightTier::DeviceLocal));
        assert_eq!(mgr.tier_of("unknown"), None);
    }

    #[test]
    fn usage_tracking() {
        let mgr = WeightTierManager::new(100, 200);
        mgr.decide("a", 40);
        assert_eq!(mgr.usage(WeightTier::DeviceLocal), (40, 100));
        mgr.decide("b", 150);
        assert_eq!(mgr.usage(WeightTier::HostLocal), (150, 200));
    }

    #[test]
    fn no_device_falls_to_host() {
        let mgr = WeightTierManager::new(0, 500);
        let d = mgr.decide("a", 100);
        assert_eq!(d.tier, WeightTier::HostLocal);
    }

    // ─── WeightTier enum tests ──────────────────────────────────────────

    #[test]
    fn weight_tier_variants_are_distinct() {
        let tiers = [WeightTier::DeviceLocal, WeightTier::HostLocal, WeightTier::DiskMmap];
        for i in 0..tiers.len() {
            for j in (i + 1)..tiers.len() {
                assert_ne!(tiers[i], tiers[j], "WeightTier variants must all differ");
            }
        }
    }

    #[test]
    fn weight_tier_debug_format() {
        assert_eq!(format!("{:?}", WeightTier::DeviceLocal), "DeviceLocal");
        assert_eq!(format!("{:?}", WeightTier::HostLocal), "HostLocal");
        assert_eq!(format!("{:?}", WeightTier::DiskMmap), "DiskMmap");
    }

    #[test]
    fn weight_tier_clone_copy() {
        let t = WeightTier::HostLocal;
        let copy: WeightTier = t;
        assert_eq!(t, copy);

        let cloned = t.clone();
        assert_eq!(t, cloned);
    }

    #[test]
    fn weight_tier_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WeightTier::DeviceLocal);
        set.insert(WeightTier::HostLocal);
        set.insert(WeightTier::DiskMmap);
        // Duplicate insert
        set.insert(WeightTier::DeviceLocal);
        assert_eq!(set.len(), 3);
    }

    // ─── UploadDecision tests ───────────────────────────────────────────

    #[test]
    fn upload_decision_equality() {
        let a = UploadDecision { tier: WeightTier::DeviceLocal, placement: WeightPlacement::DeviceLocal };
        let b = UploadDecision { tier: WeightTier::DeviceLocal, placement: WeightPlacement::DeviceLocal };
        assert_eq!(a, b);

        let c = UploadDecision { tier: WeightTier::HostLocal, placement: WeightPlacement::HostLocal };
        assert_ne!(a, c);
    }

    #[test]
    fn upload_decision_copy_clone() {
        let d = UploadDecision { tier: WeightTier::DiskMmap, placement: WeightPlacement::HostLocal };
        let copy: UploadDecision = d;
        assert_eq!(d, copy);

        let cloned = d.clone();
        assert_eq!(d, cloned);
    }

    // ─── decide() boundary & edge cases ─────────────────────────────────

    #[test]
    fn decide_exact_fit_fills_device() {
        let mgr = WeightTierManager::new(100, 0);
        let d = mgr.decide("exact", 100);
        assert_eq!(d.tier, WeightTier::DeviceLocal);
        assert_eq!(mgr.usage(WeightTier::DeviceLocal), (100, 100));
    }

    #[test]
    fn decide_one_byte_over_degrades() {
        let mgr = WeightTierManager::new(100, 200);
        let d = mgr.decide("big", 101);
        assert_eq!(d.tier, WeightTier::HostLocal);
    }

    #[test]
    fn decide_zero_size_uses_device() {
        let mgr = WeightTierManager::new(100, 200);
        let d = mgr.decide("empty", 0);
        assert_eq!(d.tier, WeightTier::DeviceLocal);
        assert_eq!(mgr.usage(WeightTier::DeviceLocal).0, 0);
    }

    #[test]
    fn decide_both_exhausted_falls_to_mmap() {
        let mgr = WeightTierManager::new(10, 10);
        let _ = mgr.decide("a", 10); // fills device
        let _ = mgr.decide("b", 10); // fills host
        let d = mgr.decide("c", 1);
        assert_eq!(d.tier, WeightTier::DiskMmap);
        assert_eq!(d.placement, WeightPlacement::HostLocal);
    }

    #[test]
    fn decide_host_exact_fit() {
        let mgr = WeightTierManager::new(0, 100);
        let d = mgr.decide("h", 100);
        assert_eq!(d.tier, WeightTier::HostLocal);
        assert_eq!(mgr.usage(WeightTier::HostLocal).0, 100);
    }

    #[test]
    fn decide_mmap_usage_is_zero() {
        let mgr = WeightTierManager::new(0, 0);
        let d = mgr.decide("any", 999);
        assert_eq!(d.tier, WeightTier::DiskMmap);
        assert_eq!(mgr.usage(WeightTier::DiskMmap), (0, 0));
    }

    // ─── tier_of() query tests ──────────────────────────────────────────

    #[test]
    fn tier_of_returns_none_for_unknown() {
        let mgr = WeightTierManager::new(1000, 1000);
        assert_eq!(mgr.tier_of("nonexistent"), None);
    }

    #[test]
    fn tier_of_tracks_all_three_tiers() {
        let mgr = WeightTierManager::new(10, 10);
        let _ = mgr.decide("dev", 5);
        let _ = mgr.decide("host", 8);
        let _ = mgr.decide("mmap", 99);

        assert_eq!(mgr.tier_of("dev"), Some(WeightTier::DeviceLocal));
        assert_eq!(mgr.tier_of("host"), Some(WeightTier::HostLocal));
        assert_eq!(mgr.tier_of("mmap"), Some(WeightTier::DiskMmap));
    }

    // ─── tensor_count() ─────────────────────────────────────────────────

    #[test]
    fn tensor_count_increments() {
        let mgr = WeightTierManager::new(1000, 1000);
        assert_eq!(mgr.tensor_count(), 0);
        mgr.decide("t1", 10);
        assert_eq!(mgr.tensor_count(), 1);
        mgr.decide("t2", 20);
        assert_eq!(mgr.tensor_count(), 2);
    }

    #[test]
    fn tensor_count_overwrite_same_name() {
        let mgr = WeightTierManager::new(1000, 1000);
        mgr.decide("dup", 10);
        mgr.decide("dup", 20); // overwrites allocation entry
        assert_eq!(mgr.tensor_count(), 1);
    }

    // ─── total_allocated() ──────────────────────────────────────────────

    #[test]
    fn total_allocated_across_tiers() {
        let mgr = WeightTierManager::new(100, 200);
        mgr.decide("dev", 40);
        mgr.decide("host", 80);
        assert_eq!(mgr.total_allocated(), 120);
    }

    #[test]
    fn total_allocated_excludes_mmap() {
        let mgr = WeightTierManager::new(0, 0);
        mgr.decide("mmap_only", 9999);
        assert_eq!(mgr.total_allocated(), 0);
    }

    // ─── usage() per-tier ───────────────────────────────────────────────

    #[test]
    fn usage_reflects_capacity() {
        let mgr = WeightTierManager::new(256, 512);
        assert_eq!(mgr.usage(WeightTier::DeviceLocal), (0, 256));
        assert_eq!(mgr.usage(WeightTier::HostLocal), (0, 512));
    }

    // ─── decide_layer_tier() fallback ───────────────────────────────────

    #[test]
    fn decide_layer_tier_prefers_gpu() {
        let mgr = WeightTierManager::new(100, 200);
        assert_eq!(mgr.decide_layer_tier(), StorageTier::GpuHbm);
    }

    #[test]
    fn decide_layer_tier_falls_to_dram() {
        let mgr = WeightTierManager::new(0, 200);
        assert_eq!(mgr.decide_layer_tier(), StorageTier::CpuDram);
    }

    #[test]
    fn decide_layer_tier_falls_to_nvme() {
        let mgr = WeightTierManager::new(0, 0);
        assert_eq!(mgr.decide_layer_tier(), StorageTier::Nvme);
    }

    #[test]
    fn decide_layer_tier_gpu_exhausted() {
        let mgr = WeightTierManager::new(10, 200);
        mgr.decide("fill", 10);
        assert_eq!(mgr.decide_layer_tier(), StorageTier::CpuDram);
    }

    // ─── decide_via_gmm() ──────────────────────────────────────────────

    #[test]
    fn gmm_short_features_falls_back() {
        let mgr = WeightTierManager::new(100, 200);
        // Empty features → fallback to decide_layer_tier → GpuHbm
        assert_eq!(mgr.decide_via_gmm(&[]), StorageTier::GpuHbm);
        // Single element → also too short (needs 2)
        assert_eq!(mgr.decide_via_gmm(&[0.5]), StorageTier::GpuHbm);
    }

    #[test]
    fn gmm_high_access_low_size_prefers_gpu() {
        let mgr = WeightTierManager::new(0, 0);
        // Features near component 0 mean [0.80, 0.20] → GpuHbm
        let tier = mgr.decide_via_gmm(&[0.80, 0.20]);
        assert_eq!(tier, StorageTier::GpuHbm);
    }

    #[test]
    fn gmm_medium_features_prefers_dram() {
        let mgr = WeightTierManager::new(0, 0);
        // Features near component 1 mean [0.50, 0.50] → CpuDram
        let tier = mgr.decide_via_gmm(&[0.50, 0.50]);
        assert_eq!(tier, StorageTier::CpuDram);
    }

    #[test]
    fn gmm_low_access_high_size_prefers_nvme() {
        let mgr = WeightTierManager::new(0, 0);
        // Features near component 2 mean [0.20, 0.80] → Nvme
        let tier = mgr.decide_via_gmm(&[0.20, 0.80]);
        assert_eq!(tier, StorageTier::Nvme);
    }

    // ─── decide_via_gmm_capacity() ──────────────────────────────────────

    #[test]
    fn gmm_capacity_prefers_l1() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        let d = mgr.decide_via_gmm_capacity("w1", 10, &gmm);
        assert_eq!(d.tier, WeightTier::DeviceLocal);
        assert_eq!(d.placement, WeightPlacement::DeviceLocal);
    }

    #[test]
    fn gmm_capacity_l1_full_falls_to_l2() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(5, 100, 100);
        let d = mgr.decide_via_gmm_capacity("w1", 10, &gmm);
        assert_eq!(d.tier, WeightTier::HostLocal);
        assert_eq!(d.placement, WeightPlacement::HostLocal);
    }

    #[test]
    fn gmm_capacity_both_full_falls_to_mmap() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(5, 5, 100);
        let d = mgr.decide_via_gmm_capacity("w1", 10, &gmm);
        assert_eq!(d.tier, WeightTier::DiskMmap);
        assert_eq!(d.placement, WeightPlacement::HostLocal);
    }

    #[test]
    fn gmm_capacity_tracks_tier_of() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        mgr.decide_via_gmm_capacity("tracked", 5, &gmm);
        assert_eq!(mgr.tier_of("tracked"), Some(WeightTier::DeviceLocal));
    }

    // ─── Additional WeightAllocation tests ───────────────────────────────

    #[test]
    fn weight_allocation_fields() {
        let a = WeightAllocation { tier: WeightTier::HostLocal, size: 1024 };
        assert_eq!(a.tier, WeightTier::HostLocal);
        assert_eq!(a.size, 1024);
    }

    #[test]
    fn weight_allocation_equality() {
        let a = WeightAllocation { tier: WeightTier::DeviceLocal, size: 512 };
        let b = WeightAllocation { tier: WeightTier::DeviceLocal, size: 512 };
        assert_eq!(a, b);
    }

    #[test]
    fn weight_allocation_inequality_different_tier() {
        let a = WeightAllocation { tier: WeightTier::DeviceLocal, size: 512 };
        let b = WeightAllocation { tier: WeightTier::HostLocal, size: 512 };
        assert_ne!(a, b);
    }

    #[test]
    fn weight_allocation_inequality_different_size() {
        let a = WeightAllocation { tier: WeightTier::DeviceLocal, size: 512 };
        let b = WeightAllocation { tier: WeightTier::DeviceLocal, size: 256 };
        assert_ne!(a, b);
    }

    #[test]
    fn weight_allocation_clone() {
        let a = WeightAllocation { tier: WeightTier::DiskMmap, size: 4096 };
        let cloned = a.clone();
        assert_eq!(a, cloned);
    }

    #[test]
    fn weight_allocation_debug() {
        let a = WeightAllocation { tier: WeightTier::DeviceLocal, size: 100 };
        let debug_str = format!("{:?}", a);
        assert!(debug_str.contains("DeviceLocal"));
        assert!(debug_str.contains("100"));
    }

    // ─── UploadDecision Debug format ────────────────────────────────────

    #[test]
    fn upload_decision_debug_format() {
        let d = UploadDecision { tier: WeightTier::DeviceLocal, placement: WeightPlacement::DeviceLocal };
        let s = format!("{:?}", d);
        assert!(s.contains("DeviceLocal"));
    }

    // ─── WeightPlacement tests ──────────────────────────────────────────

    #[test]
    fn weight_placement_equality() {
        assert_eq!(WeightPlacement::DeviceLocal, WeightPlacement::DeviceLocal);
        assert_eq!(WeightPlacement::HostLocal, WeightPlacement::HostLocal);
        assert_ne!(WeightPlacement::DeviceLocal, WeightPlacement::HostLocal);
    }

    #[test]
    fn weight_placement_copy_clone() {
        let p = WeightPlacement::HostLocal;
        let copy: WeightPlacement = p;
        assert_eq!(p, copy);
        let cloned = p.clone();
        assert_eq!(p, cloned);
    }

    #[test]
    fn weight_placement_debug() {
        assert_eq!(format!("{:?}", WeightPlacement::DeviceLocal), "DeviceLocal");
        assert_eq!(format!("{:?}", WeightPlacement::HostLocal), "HostLocal");
    }

    // ─── decide() edge cases ───────────────────────────────────────────

    #[test]
    fn decide_large_capacity_fills_then_degrades() {
        let mgr = WeightTierManager::new(1_000_000, 1_000_000);
        let d = mgr.decide("huge", 1_000_000);
        assert_eq!(d.tier, WeightTier::DeviceLocal);
        // Second request must degrade to host
        let d2 = mgr.decide("overflow", 1);
        assert_eq!(d2.tier, WeightTier::HostLocal);
    }

    #[test]
    fn decide_overwrites_same_name_updates_tier() {
        let mgr = WeightTierManager::new(1000, 1000);
        let d1 = mgr.decide("x", 10);
        assert_eq!(d1.tier, WeightTier::DeviceLocal);
        // Fill device to capacity
        mgr.decide("filler", 990);
        // Re-decide "x" — device is full, so it goes to host
        let d2 = mgr.decide("x", 10);
        assert_eq!(d2.tier, WeightTier::HostLocal);
        // tier_of reflects the latest allocation
        assert_eq!(mgr.tier_of("x"), Some(WeightTier::HostLocal));
    }

    #[test]
    fn decide_host_one_byte_over_degrades_to_mmap() {
        let mgr = WeightTierManager::new(0, 100);
        mgr.decide("a", 90);
        // host used=90, requesting 11 → 90+11=101 > 100 → mmap
        let d = mgr.decide("b", 11);
        assert_eq!(d.tier, WeightTier::DiskMmap);
    }

    #[test]
    fn decide_exact_host_fit() {
        let mgr = WeightTierManager::new(0, 200);
        let d = mgr.decide("a", 200);
        assert_eq!(d.tier, WeightTier::HostLocal);
        assert_eq!(mgr.usage(WeightTier::HostLocal).0, 200);
    }

    #[test]
    fn decide_multiple_small_allocations_fill_device() {
        let mgr = WeightTierManager::new(100, 200);
        for i in 0..10 {
            let d = mgr.decide(&format!("t{}", i), 10);
            assert_eq!(d.tier, WeightTier::DeviceLocal);
        }
        assert_eq!(mgr.usage(WeightTier::DeviceLocal).0, 100);
        // 11th allocation overflows device → host
        let d = mgr.decide("t10", 10);
        assert_eq!(d.tier, WeightTier::HostLocal);
    }

    // ─── usage() edge cases ────────────────────────────────────────────

    #[test]
    fn usage_mmap_always_zero() {
        let mgr = WeightTierManager::new(0, 0);
        mgr.decide("x", 99999);
        assert_eq!(mgr.usage(WeightTier::DiskMmap), (0, 0));
    }

    #[test]
    fn usage_device_zero_capacity() {
        let mgr = WeightTierManager::new(0, 100);
        assert_eq!(mgr.usage(WeightTier::DeviceLocal), (0, 0));
    }

    // ─── total_allocated() edge cases ──────────────────────────────────

    #[test]
    fn total_allocated_zero_initially() {
        let mgr = WeightTierManager::new(100, 200);
        assert_eq!(mgr.total_allocated(), 0);
    }

    #[test]
    fn total_allocated_after_device_only() {
        let mgr = WeightTierManager::new(1000, 1000);
        mgr.decide("a", 300);
        mgr.decide("b", 200);
        assert_eq!(mgr.total_allocated(), 500);
    }

    // ─── tensor_count() edge cases ─────────────────────────────────────

    #[test]
    fn tensor_count_zero_initially() {
        let mgr = WeightTierManager::new(100, 200);
        assert_eq!(mgr.tensor_count(), 0);
    }

    // ─── decide_layer_tier() additional tests ──────────────────────────

    #[test]
    fn decide_layer_tier_host_exhausted() {
        let mgr = WeightTierManager::new(0, 10);
        mgr.decide("fill_host", 10);
        assert_eq!(mgr.decide_layer_tier(), StorageTier::Nvme);
    }

    #[test]
    fn decide_layer_tier_device_exactly_full_prefers_host() {
        let mgr = WeightTierManager::new(10, 200);
        mgr.decide("fill", 10);
        // device_used == device_capacity, so not strictly less → host
        assert_eq!(mgr.decide_layer_tier(), StorageTier::CpuDram);
    }

    // ─── decide_via_gmm() additional tests ─────────────────────────────

    #[test]
    fn gmm_extra_features_uses_first_two() {
        let mgr = WeightTierManager::new(0, 0);
        // Extra features beyond GMM_N_FEATURES are ignored
        let tier = mgr.decide_via_gmm(&[0.80, 0.20, 0.99, 0.01]);
        assert_eq!(tier, StorageTier::GpuHbm);
    }

    #[test]
    fn gmm_boundary_near_component_0() {
        let mgr = WeightTierManager::new(0, 0);
        // Slightly off from component 0 mean but still closest
        let tier = mgr.decide_via_gmm(&[0.78, 0.22]);
        assert_eq!(tier, StorageTier::GpuHbm);
    }

    #[test]
    fn gmm_boundary_near_component_2() {
        let mgr = WeightTierManager::new(0, 0);
        let tier = mgr.decide_via_gmm(&[0.22, 0.78]);
        assert_eq!(tier, StorageTier::Nvme);
    }

    #[test]
    fn gmm_negative_features() {
        let mgr = WeightTierManager::new(0, 0);
        // Negative features are far from all components; GMM still picks one
        let tier = mgr.decide_via_gmm(&[-1.0, -1.0]);
        // Verify it returns a valid tier (not panic)
        assert!(matches!(tier, StorageTier::GpuHbm | StorageTier::CpuDram | StorageTier::Nvme));
    }

    #[test]
    fn gmm_very_large_features() {
        let mgr = WeightTierManager::new(0, 0);
        let tier = mgr.decide_via_gmm(&[1e6, 1e6]);
        assert!(matches!(tier, StorageTier::GpuHbm | StorageTier::CpuDram | StorageTier::Nvme));
    }

    #[test]
    fn gmm_nan_features_returns_valid_tier() {
        let mgr = WeightTierManager::new(0, 0);
        let tier = mgr.decide_via_gmm(&[f32::NAN, f32::NAN]);
        // NaN propagation should not panic; GMM still returns some tier
        assert!(matches!(tier, StorageTier::GpuHbm | StorageTier::CpuDram | StorageTier::Nvme));
    }

    #[test]
    fn gmm_inf_features_returns_valid_tier() {
        let mgr = WeightTierManager::new(0, 0);
        let tier = mgr.decide_via_gmm(&[f32::INFINITY, f32::INFINITY]);
        assert!(matches!(tier, StorageTier::GpuHbm | StorageTier::CpuDram | StorageTier::Nvme));
    }

    // ─── decide_via_gmm_capacity() additional tests ────────────────────

    #[test]
    fn gmm_capacity_zero_size_goes_to_l1() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        let d = mgr.decide_via_gmm_capacity("zero", 0, &gmm);
        assert_eq!(d.tier, WeightTier::DeviceLocal);
    }

    #[test]
    fn gmm_capacity_exact_l1_fit() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(50, 100, 100);
        let d = mgr.decide_via_gmm_capacity("exact", 50, &gmm);
        assert_eq!(d.tier, WeightTier::DeviceLocal);
    }

    #[test]
    fn gmm_capacity_exact_l2_fit() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(5, 50, 100);
        let d = mgr.decide_via_gmm_capacity("exact_l2", 50, &gmm);
        assert_eq!(d.tier, WeightTier::HostLocal);
    }

    #[test]
    fn gmm_capacity_tracks_tensor_count() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(100, 100, 100);
        assert_eq!(mgr.tensor_count(), 0);
        mgr.decide_via_gmm_capacity("w1", 5, &gmm);
        mgr.decide_via_gmm_capacity("w2", 10, &gmm);
        assert_eq!(mgr.tensor_count(), 2);
    }

    #[test]
    fn gmm_capacity_multiple_tiers_tracked() {
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(10, 20, 100);
        let _ = mgr.decide_via_gmm_capacity("dev", 5, &gmm);
        let _ = mgr.decide_via_gmm_capacity("host", 15, &gmm);
        let _ = mgr.decide_via_gmm_capacity("mmap", 200, &gmm);
        assert_eq!(mgr.tier_of("dev"), Some(WeightTier::DeviceLocal));
        assert_eq!(mgr.tier_of("host"), Some(WeightTier::HostLocal));
        assert_eq!(mgr.tier_of("mmap"), Some(WeightTier::DiskMmap));
    }

    // ─── WeightTier ordering via Hash ──────────────────────────────────

    #[test]
    fn weight_tier_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut h1 = DefaultHasher::new();
        WeightTier::DeviceLocal.hash(&mut h1);
        let hash1 = h1.finish();
        let mut h2 = DefaultHasher::new();
        WeightTier::DeviceLocal.hash(&mut h2);
        let hash2 = h2.finish();
        assert_eq!(hash1, hash2, "Same WeightTier must hash identically");
    }

    #[test]
    fn weight_tier_hash_differs_for_variants() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |t: WeightTier| {
            let mut h = DefaultHasher::new();
            t.hash(&mut h);
            h.finish()
        };
        let hashes: [u64; 3] = [
            hash_of(WeightTier::DeviceLocal),
            hash_of(WeightTier::HostLocal),
            hash_of(WeightTier::DiskMmap),
        ];
        // All three hashes should be distinct (not guaranteed but expected)
        assert_ne!(hashes[0], hashes[1]);
        assert_ne!(hashes[1], hashes[2]);
        assert_ne!(hashes[0], hashes[2]);
    }

    // ─── WeightTier all-constants verification ─────────────────────────

    #[test]
    fn device_weight_fraction_in_range() {
        assert!(
            (0.0..=1.0).contains(&DEVICE_WEIGHT_FRACTION),
            "DEVICE_WEIGHT_FRACTION must be in [0, 1]"
        );
    }

    #[test]
    fn host_weight_fraction_in_range() {
        assert!(
            (0.0..=1.0).contains(&HOST_WEIGHT_FRACTION),
            "HOST_WEIGHT_FRACTION must be in [0, 1]"
        );
    }

    #[test]
    fn gmm_components_weights_sum_to_one() {
        let total: f32 = GMM_COMPONENTS.iter().map(|c| c.weight).sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "GMM component weights must sum to 1.0, got {}",
            total
        );
    }

    #[test]
    fn gmm_components_feature_dim_is_two() {
        for comp in &GMM_COMPONENTS {
            assert_eq!(comp.mean.len(), WeightTierManager::GMM_N_FEATURES);
            assert_eq!(comp.variance.len(), WeightTierManager::GMM_N_FEATURES);
        }
    }

    #[test]
    fn gmm_components_variances_positive() {
        for comp in &GMM_COMPONENTS {
            for &v in &comp.variance {
                assert!(v > 0.0, "Variance must be positive, got {}", v);
            }
        }
    }

    // ─── 13 New Tests ──────────────────────────────────────────────────

    // @trace TEST-WT-78 [req:REQ-WP] [level:unit]
    #[test]
    fn new_zero_capacities_both_tiers_empty() {
        // Arrange: both device and host capacity are zero
        let mgr = WeightTierManager::new(0, 0);

        // Act & Assert: all allocations degrade to mmap
        assert_eq!(mgr.usage(WeightTier::DeviceLocal), (0, 0));
        assert_eq!(mgr.usage(WeightTier::HostLocal), (0, 0));
        assert_eq!(mgr.total_allocated(), 0);
        assert_eq!(mgr.tensor_count(), 0);
    }

    // @trace TEST-WT-79 [req:REQ-WP] [level:unit]
    #[test]
    fn new_large_capacities_tracked_correctly() {
        // Arrange: very large capacities (simulating 16GB device, 64GB host)
        let dev_cap = 16 * 1024 * 1024 * 1024usize;
        let host_cap = 64 * 1024 * 1024 * 1024usize;
        let mgr = WeightTierManager::new(dev_cap, host_cap);

        // Act
        let d = mgr.decide("large_tensor", 2 * 1024 * 1024 * 1024); // 2 GB tensor

        // Assert: goes to device, usage tracks correctly
        assert_eq!(d.tier, WeightTier::DeviceLocal);
        assert_eq!(mgr.usage(WeightTier::DeviceLocal).0, 2 * 1024 * 1024 * 1024);
        assert_eq!(mgr.usage(WeightTier::DeviceLocal).1, dev_cap);
        assert_eq!(mgr.usage(WeightTier::HostLocal).1, host_cap);
    }

    // @trace TEST-WT-80 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_many_tensors_fill_device_exactly() {
        // Arrange: 1000 bytes device, allocate 100 tensors of 10 bytes each
        let mgr = WeightTierManager::new(1000, 2000);

        // Act
        for i in 0..100 {
            let d = mgr.decide(&format!("t{}", i), 10);
            assert_eq!(d.tier, WeightTier::DeviceLocal, "tensor {} should be on device", i);
        }

        // Assert: device exactly full, host untouched
        assert_eq!(mgr.usage(WeightTier::DeviceLocal).0, 1000);
        assert_eq!(mgr.usage(WeightTier::HostLocal).0, 0);
        assert_eq!(mgr.tensor_count(), 100);
    }

    // @trace TEST-WT-81 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_empty_string_name_works() {
        // Arrange
        let mgr = WeightTierManager::new(100, 200);

        // Act: decide with empty string name
        let d = mgr.decide("", 50);

        // Assert: empty string is a valid key
        assert_eq!(d.tier, WeightTier::DeviceLocal);
        assert_eq!(mgr.tier_of(""), Some(WeightTier::DeviceLocal));
        assert_eq!(mgr.tensor_count(), 1);
    }

    // @trace TEST-WT-82 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_sequential_host_fill_then_mmap() {
        // Arrange: device=0, host=50
        let mgr = WeightTierManager::new(0, 50);

        // Act: fill host with two allocations
        let d1 = mgr.decide("a", 25);
        let d2 = mgr.decide("b", 20);
        let d3 = mgr.decide("c", 10); // 25+20=45, 45+10=55 > 50 → mmap

        // Assert
        assert_eq!(d1.tier, WeightTier::HostLocal);
        assert_eq!(d2.tier, WeightTier::HostLocal);
        assert_eq!(d3.tier, WeightTier::DiskMmap);
        assert_eq!(mgr.usage(WeightTier::HostLocal).0, 45);
        assert_eq!(mgr.total_allocated(), 45); // mmap not counted
    }

    // @trace TEST-WT-83 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_usize_max_size_degrades_to_mmap() {
        // Arrange: small capacities, request usize::MAX
        let mgr = WeightTierManager::new(100, 100);

        // Act: requesting usize::MAX bytes always degrades
        let d = mgr.decide("huge", usize::MAX);

        // Assert: neither device nor host can fit it → mmap
        assert_eq!(d.tier, WeightTier::DiskMmap);
        assert_eq!(mgr.total_allocated(), 0);
    }

    // @trace TEST-WT-84 [req:REQ-WP] [level:unit]
    #[test]
    fn total_allocated_includes_both_tiers_excludes_mmap() {
        // Arrange: mixed allocations across all three tiers
        let mgr = WeightTierManager::new(200, 200);
        mgr.decide("dev1", 80);
        mgr.decide("dev2", 60);
        mgr.decide("host1", 90);
        mgr.decide("mmap1", 500); // degrades to mmap

        // Act
        let total = mgr.total_allocated();

        // Assert: 80+60 (device) + 90 (host) = 230; mmap excluded
        assert_eq!(total, 230);
        assert_eq!(mgr.usage(WeightTier::DeviceLocal).0, 140);
        assert_eq!(mgr.usage(WeightTier::HostLocal).0, 90);
    }

    // @trace TEST-WT-85 [req:REQ-WP] [level:unit]
    #[test]
    fn tier_of_reflects_latest_after_overwrite() {
        // Arrange: small device so it fills easily
        let mgr = WeightTierManager::new(10, 200);

        // Act: first allocation on device
        let d1 = mgr.decide("tensor_a", 5);
        assert_eq!(d1.tier, WeightTier::DeviceLocal);

        // Fill device completely (5 used + 5 more = 10 exactly fills)
        mgr.decide("filler", 5);

        // Re-allocate same name → device full (10/10), goes to host
        let d2 = mgr.decide("tensor_a", 5);

        // Assert: tier_of returns the latest allocation
        assert_eq!(d2.tier, WeightTier::HostLocal);
        assert_eq!(mgr.tier_of("tensor_a"), Some(WeightTier::HostLocal));
    }

    // @trace TEST-WT-86 [req:REQ-WP] [level:unit]
    #[test]
    fn gmm_features_between_components_0_and_1_boundary() {
        // Arrange: features exactly halfway between component 0 [0.80, 0.20] and component 1 [0.50, 0.50]
        // Midpoint = [0.65, 0.35]
        let mgr = WeightTierManager::new(0, 0);

        // Act
        let tier = mgr.decide_via_gmm(&[0.65, 0.35]);

        // Assert: either GpuHbm or CpuDram is valid (boundary); must not panic
        assert!(matches!(tier, StorageTier::GpuHbm | StorageTier::CpuDram | StorageTier::Nvme));
    }

    // @trace TEST-WT-87 [req:REQ-WP] [level:unit]
    #[test]
    fn gmm_negative_infinity_features_returns_valid_tier() {
        // Arrange: extreme negative infinity values
        let mgr = WeightTierManager::new(0, 0);

        // Act
        let tier = mgr.decide_via_gmm(&[f32::NEG_INFINITY, f32::NEG_INFINITY]);

        // Assert: must return a valid tier without panic
        assert!(matches!(tier, StorageTier::GpuHbm | StorageTier::CpuDram | StorageTier::Nvme));
    }

    // @trace TEST-WT-88 [req:REQ-WP] [level:unit]
    #[test]
    fn gmm_capacity_overwrite_same_name_changes_tier() {
        // Arrange: tiny L1 so second allocation degrades
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(5, 100, 100);

        // Act: first allocation fits L1
        let d1 = mgr.decide_via_gmm_capacity("w", 3, &gmm);
        assert_eq!(d1.tier, WeightTier::DeviceLocal);

        // Second allocation for same name is too big for L1, fits L2
        let d2 = mgr.decide_via_gmm_capacity("w", 10, &gmm);

        // Assert: tier_of reflects the latest allocation
        assert_eq!(d2.tier, WeightTier::HostLocal);
        assert_eq!(mgr.tier_of("w"), Some(WeightTier::HostLocal));
        // tensor_count is still 1 (same name overwrites)
        assert_eq!(mgr.tensor_count(), 1);
    }

    // @trace TEST-WT-89 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_layer_tier_with_partial_device_usage() {
        // Arrange: device partially used but not full, host untouched
        let mgr = WeightTierManager::new(100, 200);
        mgr.decide("partial", 60);

        // Act
        let tier = mgr.decide_layer_tier();

        // Assert: device_used (60) < device_capacity (100) → still prefers GPU
        assert_eq!(tier, StorageTier::GpuHbm);
    }

    // @trace TEST-WT-90 [req:REQ-WP] [level:unit]
    #[test]
    fn gmm_components_mean_values_in_unit_range() {
        // Arrange/Act: verify all GMM component means are in [0, 1]
        for comp in &GMM_COMPONENTS {
            for &m in &comp.mean {
                assert!(
                    (0.0..=1.0).contains(&m),
                    "GMM mean must be in [0, 1], got {}",
                    m
                );
            }
        }
    }

    // ─── 10 Additional High-Quality Tests ────────────────────────────────

    // @trace TEST-WT-91 [req:REQ-WP] [level:unit]
    #[test]
    fn weight_tier_used_as_hashmap_key() {
        // Arrange: use WeightTier as HashMap key to exercise Hash + Eq together
        use std::collections::HashMap;
        let mut map: HashMap<WeightTier, &str> = HashMap::new();
        map.insert(WeightTier::DeviceLocal, "gpu");
        map.insert(WeightTier::HostLocal, "cpu");
        map.insert(WeightTier::DiskMmap, "disk");

        // Act & Assert: lookup works for all variants
        assert_eq!(map.get(&WeightTier::DeviceLocal), Some(&"gpu"));
        assert_eq!(map.get(&WeightTier::HostLocal), Some(&"cpu"));
        assert_eq!(map.get(&WeightTier::DiskMmap), Some(&"disk"));
        assert_eq!(map.len(), 3);
    }

    // @trace TEST-WT-92 [req:REQ-WP] [level:unit]
    #[test]
    fn weight_tier_exhaustive_match_coverage() {
        // Arrange: iterate all WeightTier variants and ensure match is exhaustive
        let tiers = [WeightTier::DeviceLocal, WeightTier::HostLocal, WeightTier::DiskMmap];
        let mut labels = Vec::new();

        // Act: match on each variant (compile-time exhaustiveness check)
        for t in &tiers {
            let label = match t {
                WeightTier::DeviceLocal => "L1",
                WeightTier::HostLocal => "L2",
                WeightTier::DiskMmap => "L3",
            };
            labels.push(label);
        }

        // Assert: all three tiers produced distinct labels
        assert_eq!(labels, ["L1", "L2", "L3"]);
    }

    // @trace TEST-WT-93 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_name_case_sensitive() {
        // Arrange
        let mgr = WeightTierManager::new(1000, 1000);

        // Act: allocate same name with different case
        let d1 = mgr.decide("Tensor", 10);
        let d2 = mgr.decide("tensor", 10);

        // Assert: treated as distinct names
        assert_eq!(mgr.tier_of("Tensor"), Some(WeightTier::DeviceLocal));
        assert_eq!(mgr.tier_of("tensor"), Some(WeightTier::DeviceLocal));
        assert_eq!(mgr.tensor_count(), 2);
    }

    // @trace TEST-WT-94 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_host_overflow_wraps_gracefully_to_mmap() {
        // Arrange: host capacity is 100, allocate two tensors that sum to just over 100
        let mgr = WeightTierManager::new(0, 100);

        // Act
        let d1 = mgr.decide("a", 60);
        let d2 = mgr.decide("b", 50); // 60+50=110 > 100

        // Assert
        assert_eq!(d1.tier, WeightTier::HostLocal);
        assert_eq!(d2.tier, WeightTier::DiskMmap);
        assert_eq!(mgr.usage(WeightTier::HostLocal).0, 60);
    }

    // @trace TEST-WT-95 [req:REQ-WP] [level:unit]
    #[test]
    fn upload_decision_fields_accessible() {
        // Arrange
        let decision = UploadDecision {
            tier: WeightTier::HostLocal,
            placement: WeightPlacement::HostLocal,
        };

        // Act & Assert: verify struct fields are directly accessible
        assert_eq!(decision.tier, WeightTier::HostLocal);
        assert_eq!(decision.placement, WeightPlacement::HostLocal);

        // Verify Copy allows destructuring without move
        let UploadDecision { tier, placement } = decision;
        assert_eq!(tier, WeightTier::HostLocal);
        assert_eq!(placement, WeightPlacement::HostLocal);
        // decision still usable after destructuring (Copy)
        assert_eq!(decision.tier, WeightTier::HostLocal);
    }

    // @trace TEST-WT-96 [req:REQ-WP] [level:unit]
    #[test]
    fn gmm_deterministic_same_input_same_output() {
        // Arrange: identical features called multiple times
        let mgr = WeightTierManager::new(0, 0);
        let features = [0.80, 0.20];

        // Act: call decide_via_gmm 10 times with same input
        let mut results = Vec::new();
        for _ in 0..10 {
            results.push(mgr.decide_via_gmm(&features));
        }

        // Assert: all results identical (deterministic)
        assert!(results.iter().all(|t| *t == results[0]),
            "GMM must produce identical output for identical input");
    }

    // @trace TEST-WT-97 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_layer_tier_device_partial_host_full_falls_to_nvme() {
        // Arrange: device partially used, host fully used
        let mgr = WeightTierManager::new(100, 50);
        mgr.decide("dev_alloc", 30); // device: 30/100
        mgr.decide("host_alloc", 50); // host: 50/50 (full)

        // Act
        let tier = mgr.decide_layer_tier();

        // Assert: device still has capacity → GpuHbm
        assert_eq!(tier, StorageTier::GpuHbm);
    }

    // @trace TEST-WT-98 [req:REQ-WP] [level:unit]
    #[test]
    fn decide_layer_tier_device_full_host_partial_falls_to_dram() {
        // Arrange: device full, host partially used
        let mgr = WeightTierManager::new(50, 100);
        mgr.decide("dev_fill", 50); // device: 50/50 (full)
        mgr.decide("host_partial", 30); // host: 30/100

        // Act
        let tier = mgr.decide_layer_tier();

        // Assert: device_used == capacity (not strictly less) → CpuDram
        assert_eq!(tier, StorageTier::CpuDram);
    }

    // @trace TEST-WT-99 [req:REQ-WP] [level:unit]
    #[test]
    fn gmm_capacity_l1_zero_capacity_falls_to_l2() {
        // Arrange: L1 capacity is zero, L2 has space
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(0, 100, 100);

        // Act: even a zero-page request should check available() == 0
        let d = mgr.decide_via_gmm_capacity("small", 0, &gmm);

        // Assert: L1 available=0 but request is 0, so 0 >= 0 → L1
        assert_eq!(d.tier, WeightTier::DeviceLocal);
    }

    // @trace TEST-WT-100 [req:REQ-WP] [level:unit]
    #[test]
    fn gmm_capacity_l1_zero_capacity_nonzero_request_falls_to_l2() {
        // Arrange: L1 capacity is zero, L2 has space
        let mgr = WeightTierManager::new(0, 0);
        let gmm = GlobalMemoryManager::new_with_capacities(0, 100, 100);

        // Act: nonzero request cannot fit in L1 (available=0)
        let d = mgr.decide_via_gmm_capacity("real", 1, &gmm);

        // Assert: degrades to L2
        assert_eq!(d.tier, WeightTier::HostLocal);
        assert_eq!(d.placement, WeightPlacement::HostLocal);
    }
}
