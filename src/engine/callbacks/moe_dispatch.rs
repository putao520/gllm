//! MoE Dispatch Callback (SPEC §15)
//!
//! Integrates Mixture-of-Experts routing, prefetch, and dispatch into the
//! graph node loop. At MoE layers, performs expert routing, weight prefetch,
//! and hardware dispatch before the FFN computation.

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};

/// MoE dispatch callback — routes tokens to experts and coordinates prefetch.
///
/// Per SPEC §15: for models with MoE architecture, this callback:
/// 1. Routes tokens to top-k experts via `ExpertRouteTable`
/// 2. Prefetches expert weights via `ExpertWeightPrefetcher`
/// 3. Dispatches computation via `MoeHardwareDispatcher`
/// 4. Updates thermal tracking via `ExpertThermalTracker`
pub struct MoeDispatchCallback {
    /// Total number of experts in the model (0 = dense model, no MoE)
    num_experts: usize,
    /// Top-k experts per token
    top_k: usize,
    /// Layer indices that have MoE FFN (typically all layers for MoE models)
    moe_layers: Vec<usize>,
    /// Whether MoE dispatch is enabled
    enabled: bool,
}

impl MoeDispatchCallback {
    /// Create a new MoE dispatch callback.
    ///
    /// `num_experts` — total number of MoE experts (0 disables)
    /// `top_k` — number of experts to route to per token
    /// `num_layers` — total number of transformer layers
    /// `moe_start_layer` — first layer with MoE (some models have dense early layers)
    pub fn new(num_experts: usize, top_k: usize, num_layers: usize, moe_start_layer: usize) -> Self {
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
        }
    }

    /// Create a disabled MoE callback (for dense models).
    pub fn disabled() -> Self {
        Self {
            num_experts: 0,
            top_k: 0,
            moe_layers: Vec::new(),
            enabled: false,
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
}

impl LayerCallback for MoeDispatchCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        if !self.enabled {
            return CallbackAction::Continue;
        }

        if !self.moe_layers.contains(&ctx.layer_idx) {
            return CallbackAction::Continue;
        }

        // MoE routing, prefetch, and dispatch coordination.
        // In the current architecture, the actual routing is done by the JIT-compiled
        // MoERouting kernel. This callback provides a hook for:
        // 1. Pre-fetching expert weights to GPU (if using CPU-offloaded experts)
        // 2. Updating expert thermal tracking
        // 3. Load-balancing across GPUs
        //
        // For now, this is a no-op hook that will be extended when
        // heterogeneous MoE dispatch is implemented.

        log::trace!(
            "moe_dispatch: layer {} ({} experts, top_k={})",
            ctx.layer_idx, self.num_experts, self.top_k,
        );

        CallbackAction::Continue
    }

    fn post_node(&mut self, ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
        if !self.enabled || !self.moe_layers.contains(&ctx.layer_idx) {
            return CallbackAction::Continue;
        }

        // Post-node: update thermal tracking and load balance statistics.
        // Will be extended with actual thermal updates when heterogeneous
        // MoE execution is implemented.

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
        // Some models have dense first N layers, then MoE
        let cb = MoeDispatchCallback::new(64, 4, 32, 8);
        assert_eq!(cb.moe_layers.len(), 24); // layers 8..31
        assert_eq!(cb.moe_layers[0], 8);
        assert_eq!(cb.moe_layers[23], 31);
    }
}
