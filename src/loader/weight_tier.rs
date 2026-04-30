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
#[derive(Debug, Clone)]
struct WeightAllocation {
    tier: WeightTier,
    size: usize,
}

/// Fraction of device memory reserved for weights (rest for KV cache + scratchpad).
const DEVICE_WEIGHT_FRACTION: f64 = 0.70;

/// Fraction of host memory usable for weight staging.
const HOST_WEIGHT_FRACTION: f64 = 0.60;

/// Manages weight tensor placement across heterogeneous memory tiers.
pub struct WeightTierManager {
    device_capacity: usize,
    host_capacity: usize,
    device_used: AtomicUsize,
    host_used: AtomicUsize,
    allocations: Mutex<HashMap<String, WeightAllocation>>,
}

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
}
