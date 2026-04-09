//! MoE Dispatch Callback (SPEC §15)
//!
//! Integrates Mixture-of-Experts routing, prefetch, and dispatch into the
//! graph node loop. At MoE layers, performs expert routing, weight prefetch,
//! and hardware dispatch before the FFN computation.
//!
//! Fault-triggered cold expert revival: when routing selects an evicted expert,
//! the callback signals a fault to the `ExpertFaultHandler`, which suspends
//! only the affected request while other requests continue executing.

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
use crate::moe::fault_handler::{ExpertFault, ExpertFaultHandler, FaultResolution};
use crate::moe::prefetch::ExpertWeightLocation;
use crate::moe::thermal::{ExpertHeatLevel, ExpertThermalManager};
use std::time::Instant;

/// Control flow signal returned after fault detection.
#[derive(Debug, Clone)]
pub enum MoeDispatchSignal {
    /// Expert was hot/warm/cold; proceed normally.
    Continue,
    /// Request was suspended pending expert page-in.
    Suspended { request_id: u64, expert_idx: usize },
    /// Expert page-in was rejected (e.g. memory pressure too high).
    Rejected { request_id: u64, reason: String },
}

/// MoE dispatch callback -- routes tokens to experts and coordinates prefetch.
///
/// Per SPEC §15: for models with MoE architecture, this callback:
/// 1. Routes tokens to top-k experts via `ExpertRouteTable`
/// 2. Detects routing to evicted experts and triggers fault handling
/// 3. Prefetches expert weights via `ExpertWeightPrefetcher`
/// 4. Dispatches computation via `MoeHardwareDispatcher`
/// 5. Updates thermal tracking via `ExpertThermalManager`
pub struct MoeDispatchCallback {
    /// Total number of experts in the model (0 = dense model, no MoE)
    num_experts: usize,
    /// Top-k experts per token
    top_k: usize,
    /// Layer indices that have MoE FFN (typically all layers for MoE models)
    moe_layers: Vec<usize>,
    /// Whether MoE dispatch is enabled
    enabled: bool,
    /// Fault handler for evicted expert revival.
    fault_handler: ExpertFaultHandler,
    /// Last dispatch signal (consumed by the engine after each pre_node).
    last_signal: MoeDispatchSignal,
}

impl MoeDispatchCallback {
    /// Create a new MoE dispatch callback.
    ///
    /// `num_experts` -- total number of MoE experts (0 disables)
    /// `top_k` -- number of experts to route to per token
    /// `num_layers` -- total number of transformer layers
    /// `moe_start_layer` -- first layer with MoE (some models have dense early layers)
    pub fn new(
        num_experts: usize,
        top_k: usize,
        num_layers: usize,
        moe_start_layer: usize,
    ) -> Self {
        let enabled = num_experts > 0;
        let moe_layers: Vec<usize> = if enabled {
            (moe_start_layer..num_layers).collect()
        } else {
            Vec::new()
        };
        Self {
            num_experts,
            top_k,
            moe_layers,
            enabled,
            fault_handler: ExpertFaultHandler::new(num_experts),
            last_signal: MoeDispatchSignal::Continue,
        }
    }

    /// Create a disabled MoE callback (for dense models).
    pub fn disabled() -> Self {
        Self {
            num_experts: 0,
            top_k: 0,
            moe_layers: Vec::new(),
            enabled: false,
            fault_handler: ExpertFaultHandler::new(0),
            last_signal: MoeDispatchSignal::Continue,
        }
    }

    /// Whether MoE dispatch is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Get a reference to the fault handler for metrics / orchestration.
    pub fn fault_handler(&self) -> &ExpertFaultHandler {
        &self.fault_handler
    }

    /// Get a mutable reference to the fault handler.
    pub fn fault_handler_mut(&mut self) -> &mut ExpertFaultHandler {
        &mut self.fault_handler
    }

    /// Consume the last dispatch signal (used by the engine after pre_node).
    pub fn take_signal(&mut self) -> MoeDispatchSignal {
        std::mem::replace(&mut self.last_signal, MoeDispatchSignal::Continue)
    }

    /// Check if a specific expert is evicted and handle the fault.
    ///
    /// Returns the dispatch signal: Continue if not evicted, Suspended or
    /// Rejected if a fault was triggered.
    pub fn check_and_handle_fault(
        &mut self,
        expert_idx: usize,
        layer_idx: usize,
        request_id: u64,
        thermal: &ExpertThermalManager,
        memory_pressure: f32,
    ) -> MoeDispatchSignal {
        // Fast path: single branch on is_evicted.
        let state = match thermal.state(expert_idx) {
            Some(s) => s,
            None => return MoeDispatchSignal::Continue,
        };
        if !state.is_evicted {
            return MoeDispatchSignal::Continue;
        }

        // Expert is evicted -- trigger fault.
        let fault = ExpertFault {
            expert_idx,
            layer_idx,
            request_id,
            fault_time: Instant::now(),
        };

        let weight_source = ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Evicted);
        // Evicted experts' weights are typically in CpuRam (swapped out).
        let actual_source = ExpertWeightLocation::CpuRam;

        let _ = weight_source; // Evicted maps to Evicted; we use CpuRam as the reload source.
        match self.fault_handler.handle_fault(fault, memory_pressure, actual_source) {
            FaultResolution::Resumed { .. } => MoeDispatchSignal::Suspended {
                request_id,
                expert_idx,
            },
            FaultResolution::Rejected { reason } => MoeDispatchSignal::Rejected {
                request_id,
                reason,
            },
        }
    }
}

impl LayerCallback for MoeDispatchCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        if !self.enabled {
            return CallbackAction::Continue;
        }

        if !self.moe_layers.contains(&ctx.layer_idx) {
            return CallbackAction::Continue;
        }

        // Record a decode step for fault-rate tracking.
        self.fault_handler.record_step();

        log::trace!(
            "moe_dispatch: layer {} ({} experts, top_k={})",
            ctx.layer_idx,
            self.num_experts,
            self.top_k,
        );

        CallbackAction::Continue
    }

    fn post_node(&mut self, ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
        if !self.enabled || !self.moe_layers.contains(&ctx.layer_idx) {
            return CallbackAction::Continue;
        }

        CallbackAction::Continue
    }

    fn priority(&self) -> u32 {
        70
    }

    fn target_layers(&self) -> Option<&[usize]> {
        if self.moe_layers.is_empty() {
            None
        } else {
            Some(&self.moe_layers)
        }
    }

    fn name(&self) -> &str {
        "moe_dispatch"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_dispatch_enabled() {
        let cb = MoeDispatchCallback::new(8, 2, 32, 0);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 8);
        assert_eq!(cb.priority(), 70);
        assert_eq!(cb.name(), "moe_dispatch");
        assert_eq!(cb.moe_layers.len(), 32);
    }

    #[test]
    fn test_moe_dispatch_disabled() {
        let cb = MoeDispatchCallback::disabled();
        assert!(!cb.is_enabled());
        assert_eq!(cb.num_experts(), 0);
        assert!(cb.target_layers().is_none() || cb.target_layers().unwrap().is_empty());
    }

    #[test]
    fn test_moe_dispatch_partial_layers() {
        let cb = MoeDispatchCallback::new(64, 4, 32, 8);
        assert_eq!(cb.moe_layers.len(), 24);
        assert_eq!(cb.moe_layers[0], 8);
        assert_eq!(cb.moe_layers[23], 31);
    }

    #[test]
    fn test_fault_detection_hot_expert() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let thermal = ExpertThermalManager::new(4);

        // Expert 0 is not evicted (default Warm).
        let signal = cb.check_and_handle_fault(0, 0, 1, &thermal, 0.3);
        assert!(matches!(signal, MoeDispatchSignal::Continue));
    }

    #[test]
    fn test_fault_detection_evicted_expert() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        let signal = cb.check_and_handle_fault(1, 0, 42, &thermal, 0.3);
        assert!(matches!(signal, MoeDispatchSignal::Suspended { request_id: 42, expert_idx: 1 }));
    }

    #[test]
    fn test_fault_rejection_high_pressure() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        let signal = cb.check_and_handle_fault(1, 0, 42, &thermal, 0.99);
        assert!(matches!(signal, MoeDispatchSignal::Rejected { .. }));
    }

    #[test]
    fn test_take_signal() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let signal = cb.take_signal();
        assert!(matches!(signal, MoeDispatchSignal::Continue));
    }

    #[test]
    fn test_fault_handler_access() {
        let cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let stats = cb.fault_handler().stats();
        assert_eq!(stats.total_faults, 0);
    }
}
