//! Expert Fault Handler — Page Fault Interrupt Model (SPEC §15.4)
//!
//! ## Core Design
//! Analogous to OS page faults: when routing selects an evicted expert,
//! a fault fires, the affected request is suspended, weights are reloaded
//! asynchronously, JIT code is restored, and the request resumes from the
//! faulted MoE layer (not from scratch).
//!
//! ## Key Properties
//! - Zero overhead on hot path (single `is_evicted` branch)
//! - Thundering-herd prevention: multiple faults on the same expert
//!   trigger only one page-in operation; all waiters resume together
//! - Fault statistics for adaptive working set sizing

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::hot_patch::HotPatchManager;
use super::prefetch::ExpertWeightLocation;
use super::thermal::ExpertThermalManager;

/// A fault event triggered when routing selects an evicted expert.
#[derive(Debug, Clone)]
pub struct ExpertFault {
    /// Expert that was accessed while evicted.
    pub expert_idx: usize,
    /// Transformer layer where the fault occurred.
    pub layer_idx: usize,
    /// Request that triggered the fault.
    pub request_id: u64,
    /// Monotonic timestamp of the fault.
    pub fault_time: Instant,
}

/// Outcome of handling a single fault.
#[derive(Debug, Clone)]
pub enum FaultResolution {
    /// Expert was restored and request can resume.
    Resumed { latency: Duration },
    /// Expert could not be restored (e.g. memory pressure).
    Rejected { reason: String },
}

/// Per-expert fault statistics.
#[derive(Debug, Clone)]
pub struct FaultStats {
    /// Total faults observed across all experts.
    pub total_faults: u64,
    /// Average recovery latency in microseconds.
    pub avg_recovery_us: f64,
    /// Faults per decode step (rolling ratio).
    pub fault_rate: f64,
    /// Number of experts currently being restored.
    pub in_flight_restorations: usize,
    /// Number of requests currently suspended.
    pub suspended_request_count: usize,
}

/// A suspended request waiting for an expert to be restored.
#[derive(Debug, Clone)]
struct SuspendedRequest {
    request_id: u64,
    layer_idx: usize,
    suspend_time: Instant,
}

/// Tracks the restoration state of a single expert.
#[derive(Debug)]
struct RestorationEntry {
    /// Requests waiting for this expert to come back.
    waiters: Vec<SuspendedRequest>,
    /// When the restoration was initiated.
    initiated_at: Instant,
    /// Source location for weight reload.
    weight_source: ExpertWeightLocation,
}

/// Expert Fault Handler — orchestrates fault detection, request suspension,
/// weight reload, JIT code restoration, and batch resume.
pub struct ExpertFaultHandler {
    /// Active restorations keyed by (expert_idx, layer_idx).
    restorations: HashMap<(usize, usize), RestorationEntry>,
    /// Per-expert cumulative fault count.
    per_expert_faults: Vec<u64>,
    /// Per-expert cumulative recovery latency (microseconds).
    per_expert_recovery_us: Vec<f64>,
    /// Total faults observed.
    total_faults: u64,
    /// Total decode steps observed (for fault-rate calculation).
    total_steps: u64,
    /// Total recovery latency (microseconds).
    total_recovery_us: f64,
    /// Total completed recoveries.
    total_recoveries: u64,
    /// Memory pressure threshold above which page-in is rejected.
    memory_pressure_limit: f32,
    /// Number of experts in the model.
    num_experts: usize,
}

impl ExpertFaultHandler {
    /// Create a new fault handler for a model with `num_experts` experts.
    pub fn new(num_experts: usize) -> Self {
        Self {
            restorations: HashMap::new(),
            per_expert_faults: vec![0; num_experts],
            per_expert_recovery_us: vec![0.0; num_experts],
            total_faults: 0,
            total_steps: 0,
            total_recovery_us: 0.0,
            total_recoveries: 0,
            memory_pressure_limit: 0.95,
            num_experts,
        }
    }

    /// Set the memory pressure limit above which page-in is rejected.
    pub fn with_memory_pressure_limit(mut self, limit: f32) -> Self {
        self.memory_pressure_limit = limit.clamp(0.0, 1.0);
        self
    }

    /// Record that one decode step has passed (for fault-rate calculation).
    pub fn record_step(&mut self) {
        self.total_steps += 1;
    }

    /// Handle a fault: suspend the request and initiate restoration if needed.
    ///
    /// Returns `FaultResolution::Rejected` if memory pressure is too high.
    /// Otherwise the request is suspended and `FaultResolution::Resumed` will
    /// be returned later via `complete_restoration`.
    pub fn handle_fault(
        &mut self,
        fault: ExpertFault,
        memory_pressure: f32,
        weight_source: ExpertWeightLocation,
    ) -> FaultResolution {
        // Reject if memory pressure is too high to page-in.
        if memory_pressure > self.memory_pressure_limit {
            return FaultResolution::Rejected {
                reason: format!(
                    "memory pressure {:.2} exceeds limit {:.2}",
                    memory_pressure, self.memory_pressure_limit,
                ),
            };
        }

        // Record fault statistics.
        self.total_faults += 1;
        if fault.expert_idx < self.num_experts {
            self.per_expert_faults[fault.expert_idx] += 1;
        }

        let key = (fault.expert_idx, fault.layer_idx);
        let suspended = SuspendedRequest {
            request_id: fault.request_id,
            layer_idx: fault.layer_idx,
            suspend_time: fault.fault_time,
        };

        // Thundering-herd prevention: if a restoration is already in flight
        // for this (expert, layer), just append the waiter.
        let entry = self.restorations.entry(key).or_insert_with(|| {
            RestorationEntry {
                waiters: Vec::new(),
                initiated_at: fault.fault_time,
                weight_source,
            }
        });
        entry.waiters.push(suspended);

        // The actual weight reload + JIT restore is async; caller drives it
        // via `complete_restoration` once weights are available.
        FaultResolution::Resumed {
            latency: Duration::ZERO,
        }
    }

    /// Check whether a restoration is already in-flight for (expert, layer).
    pub fn is_restoration_pending(&self, expert_idx: usize, layer_idx: usize) -> bool {
        self.restorations.contains_key(&(expert_idx, layer_idx))
    }

    /// Complete an in-flight restoration: restore JIT code, reactivate the
    /// expert in the thermal manager, and return all resumed request IDs.
    ///
    /// Returns the list of (request_id, recovery_latency) for each waiter.
    pub fn complete_restoration(
        &mut self,
        expert_idx: usize,
        layer_idx: usize,
        thermal: &mut ExpertThermalManager,
        patch_manager: &mut HotPatchManager,
    ) -> Vec<(u64, Duration)> {
        let key = (expert_idx, layer_idx);
        let entry = match self.restorations.remove(&key) {
            Some(e) => e,
            None => return Vec::new(),
        };

        // Restore JIT code via HotPatchManager.
        patch_manager.rollback_patch(expert_idx, layer_idx);

        // Reactivate in thermal manager.
        thermal.reactivate_expert(expert_idx);

        let now = Instant::now();
        let mut resumed = Vec::with_capacity(entry.waiters.len());
        for waiter in &entry.waiters {
            let latency = now.duration_since(waiter.suspend_time);
            let latency_us = latency.as_micros() as f64;

            self.total_recovery_us += latency_us;
            self.total_recoveries += 1;
            if expert_idx < self.num_experts {
                self.per_expert_recovery_us[expert_idx] += latency_us;
            }

            resumed.push((waiter.request_id, latency));
        }

        resumed
    }

    /// Get aggregate fault statistics.
    pub fn stats(&self) -> FaultStats {
        let avg_recovery_us = if self.total_recoveries > 0 {
            self.total_recovery_us / self.total_recoveries as f64
        } else {
            0.0
        };
        let fault_rate = if self.total_steps > 0 {
            self.total_faults as f64 / self.total_steps as f64
        } else {
            0.0
        };
        let suspended_count: usize =
            self.restorations.values().map(|e| e.waiters.len()).sum();

        FaultStats {
            total_faults: self.total_faults,
            avg_recovery_us,
            fault_rate,
            in_flight_restorations: self.restorations.len(),
            suspended_request_count: suspended_count,
        }
    }

    /// Per-expert fault count.
    pub fn expert_fault_count(&self, expert_idx: usize) -> u64 {
        self.per_expert_faults
            .get(expert_idx)
            .copied()
            .unwrap_or(0)
    }

    /// Number of experts being actively restored.
    pub fn in_flight_count(&self) -> usize {
        self.restorations.len()
    }

    /// Total suspended requests across all experts.
    pub fn suspended_request_count(&self) -> usize {
        self.restorations.values().map(|e| e.waiters.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_fault_and_complete() {
        let mut handler = ExpertFaultHandler::new(4);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);
        let config = super::super::routing::ExpertRouteConfig::new(4, 2);
        let mut patch = HotPatchManager::new(config);

        // Evict expert 2
        for _ in 0..6 {
            thermal.step(&[10, 5, 0, 3]);
        }
        thermal.evict_expert(2);
        patch.apply_patch(&super::super::hot_patch::PatchInstruction {
            target: super::super::hot_patch::PatchTarget::ExpertCode {
                expert_idx: 2,
                layer_idx: 0,
            },
            operation: super::super::hot_patch::PatchOperation::DeoptJump,
            consensus_steps: 6,
            reason: "test".to_string(),
            priority: 0,
        });

        // Fault on expert 2
        let fault = ExpertFault {
            expert_idx: 2,
            layer_idx: 0,
            request_id: 42,
            fault_time: Instant::now(),
        };
        let res = handler.handle_fault(fault, 0.5, ExpertWeightLocation::CpuRam);
        assert!(matches!(res, FaultResolution::Resumed { .. }));
        assert!(handler.is_restoration_pending(2, 0));
        assert_eq!(handler.suspended_request_count(), 1);

        // Complete restoration
        let resumed = handler.complete_restoration(2, 0, &mut thermal, &mut patch);
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].0, 42);
        assert!(!thermal.state(2).unwrap().is_evicted);
        assert!(!patch.is_expert_patched(2, 0));
    }

    #[test]
    fn test_thundering_herd_prevention() {
        let mut handler = ExpertFaultHandler::new(4);
        let now = Instant::now();

        // Multiple requests fault on the same expert
        for req_id in 0..5 {
            let fault = ExpertFault {
                expert_idx: 1,
                layer_idx: 3,
                request_id: req_id,
                fault_time: now,
            };
            handler.handle_fault(fault, 0.3, ExpertWeightLocation::CpuRam);
        }

        // Only one restoration in flight
        assert_eq!(handler.in_flight_count(), 1);
        // But 5 suspended requests
        assert_eq!(handler.suspended_request_count(), 5);
    }

    #[test]
    fn test_reject_on_high_memory_pressure() {
        let mut handler = ExpertFaultHandler::new(4)
            .with_memory_pressure_limit(0.9);

        let fault = ExpertFault {
            expert_idx: 0,
            layer_idx: 0,
            request_id: 1,
            fault_time: Instant::now(),
        };
        let res = handler.handle_fault(fault, 0.95, ExpertWeightLocation::CpuRam);
        assert!(matches!(res, FaultResolution::Rejected { .. }));
        assert_eq!(handler.in_flight_count(), 0);
    }

    #[test]
    fn test_fault_stats() {
        let mut handler = ExpertFaultHandler::new(4);
        handler.record_step();
        handler.record_step();

        let fault = ExpertFault {
            expert_idx: 0,
            layer_idx: 0,
            request_id: 1,
            fault_time: Instant::now(),
        };
        handler.handle_fault(fault, 0.3, ExpertWeightLocation::CpuRam);

        let stats = handler.stats();
        assert_eq!(stats.total_faults, 1);
        assert!((stats.fault_rate - 0.5).abs() < 1e-9);
        assert_eq!(stats.in_flight_restorations, 1);
        assert_eq!(stats.suspended_request_count, 1);
    }

    #[test]
    fn test_per_expert_fault_count() {
        let mut handler = ExpertFaultHandler::new(4);

        for i in 0..3 {
            let fault = ExpertFault {
                expert_idx: 2,
                layer_idx: i,
                request_id: i as u64,
                fault_time: Instant::now(),
            };
            handler.handle_fault(fault, 0.3, ExpertWeightLocation::CpuRam);
        }

        assert_eq!(handler.expert_fault_count(2), 3);
        assert_eq!(handler.expert_fault_count(0), 0);
    }

    #[test]
    fn test_complete_nonexistent_restoration() {
        let mut handler = ExpertFaultHandler::new(4);
        let mut thermal = ExpertThermalManager::new(4);
        let config = super::super::routing::ExpertRouteConfig::new(4, 2);
        let mut patch = HotPatchManager::new(config);

        // Completing a restoration that doesn't exist returns empty
        let resumed = handler.complete_restoration(0, 0, &mut thermal, &mut patch);
        assert!(resumed.is_empty());
    }
}
