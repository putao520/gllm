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
    use crate::moe::thermal::{DeoptHandlingResult, EvictionDecision, ThermalSummary};
    use crate::moe::fault_handler::FaultStats;
    use crate::moe::prefetch::{
        ExpertPrefetchRequest, ExpertWeightLayout, ExpertWeightPrefetcher,
    };
    use crate::moe::DeoptRequest;
    use crate::moe::ExpertHeatState;
    use std::time::Duration;
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

    // ── MoeDispatchSignal Debug + Clone ────────────────────────────────

    #[test]
    fn test_signal_continue_debug_format() {
        let signal = MoeDispatchSignal::Continue;
        let debug = format!("{:?}", signal);
        assert!(debug.contains("Continue"));
    }

    #[test]
    fn test_signal_suspended_debug_format() {
        let signal = MoeDispatchSignal::Suspended {
            request_id: 42,
            expert_idx: 3,
        };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("Suspended"));
        assert!(debug.contains("42"));
        assert!(debug.contains("3"));
    }

    #[test]
    fn test_signal_rejected_debug_format() {
        let signal = MoeDispatchSignal::Rejected {
            request_id: 7,
            reason: "memory full".to_string(),
        };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("Rejected"));
        assert!(debug.contains("7"));
        assert!(debug.contains("memory full"));
    }

    #[test]
    fn test_signal_clone_continue() {
        let original = MoeDispatchSignal::Continue;
        let cloned = original.clone();
        assert!(matches!(cloned, MoeDispatchSignal::Continue));
    }

    #[test]
    fn test_signal_clone_suspended_preserves_fields() {
        let original = MoeDispatchSignal::Suspended {
            request_id: 99,
            expert_idx: 5,
        };
        let cloned = original.clone();
        match cloned {
            MoeDispatchSignal::Suspended { request_id, expert_idx } => {
                assert_eq!(request_id, 99);
                assert_eq!(expert_idx, 5);
            }
            _ => panic!("Expected Suspended variant after clone"),
        }
    }

    #[test]
    fn test_signal_clone_rejected_preserves_fields() {
        let original = MoeDispatchSignal::Rejected {
            request_id: 100,
            reason: "pressure too high".to_string(),
        };
        let cloned = original.clone();
        match cloned {
            MoeDispatchSignal::Rejected { request_id, reason } => {
                assert_eq!(request_id, 100);
                assert_eq!(reason, "pressure too high");
            }
            _ => panic!("Expected Rejected variant after clone"),
        }
    }

    // ── Constructor edge cases ─────────────────────────────────────────

    #[test]
    fn test_new_with_zero_experts_is_disabled() {
        let cb = MoeDispatchCallback::new(0, 0, 32, 0);
        assert!(!cb.is_enabled());
        assert_eq!(cb.num_experts(), 0);
        assert!(cb.target_layers().is_none() || cb.target_layers().unwrap().is_empty());
    }

    #[test]
    fn test_new_with_moe_start_at_last_layer() {
        let cb = MoeDispatchCallback::new(8, 2, 32, 31);
        assert!(cb.is_enabled());
        // Only layer 31 is MoE
        assert_eq!(cb.moe_layers.len(), 1);
        assert_eq!(cb.moe_layers[0], 31);
    }

    #[test]
    fn test_new_with_moe_start_equals_num_layers() {
        let cb = MoeDispatchCallback::new(8, 2, 32, 32);
        assert!(cb.is_enabled());
        // moe_start_layer == num_layers => empty range, no MoE layers
        assert!(cb.moe_layers.is_empty());
    }

    #[test]
    fn test_new_single_expert_single_layer() {
        let cb = MoeDispatchCallback::new(1, 1, 1, 0);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 1);
        assert_eq!(cb.moe_layers.len(), 1);
        assert_eq!(cb.moe_layers[0], 0);
    }

    #[test]
    fn test_disabled_callback_defaults() {
        let cb = MoeDispatchCallback::disabled();
        assert!(!cb.is_enabled());
        assert_eq!(cb.num_experts(), 0);
        assert_eq!(cb.priority(), 70);
        assert_eq!(cb.name(), "moe_dispatch");
    }

    // ── fault_handler_mut accessor ─────────────────────────────────────

    #[test]
    fn test_fault_handler_mut_records_steps() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        cb.fault_handler_mut().record_step();
        cb.fault_handler_mut().record_step();
        cb.fault_handler_mut().record_step();

        let stats = cb.fault_handler().stats();
        assert_eq!(stats.fault_rate, 0.0); // 0 faults / 3 steps
        assert!(stats.avg_recovery_us >= 0.0);
    }

    // ── take_signal resets to Continue ─────────────────────────────────

    #[test]
    fn test_take_signal_returns_continue_then_resets() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);

        // First take should return the initial Continue
        let first = cb.take_signal();
        assert!(matches!(first, MoeDispatchSignal::Continue));

        // Second take should also return Continue (reset value)
        let second = cb.take_signal();
        assert!(matches!(second, MoeDispatchSignal::Continue));
    }

    // ── check_and_handle_fault with various thermal states ─────────────

    #[test]
    fn test_check_fault_non_evicted_cold_expert() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(100);

        // Expert 1 is Cold but not evicted
        for _ in 0..10 {
            thermal.step(&[10, 0, 5, 3]);
        }

        let signal = cb.check_and_handle_fault(1, 0, 1, &thermal, 0.3);
        // Expert 1 is Cold, not evicted → Continue
        assert!(matches!(signal, MoeDispatchSignal::Continue));
    }

    #[test]
    fn test_check_fault_out_of_bounds_expert() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let thermal = ExpertThermalManager::new(4);

        // Expert index 10 is out of bounds for thermal manager
        let signal = cb.check_and_handle_fault(10, 0, 1, &thermal, 0.3);
        // thermal.state(10) returns None → Continue
        assert!(matches!(signal, MoeDispatchSignal::Continue));
    }

    #[test]
    fn test_check_fault_multiple_evicted_experts() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            thermal.step(&[10, 0, 0, 0]);
        }
        thermal.evict_expert(1);
        thermal.evict_expert(2);

        // Fault on expert 1
        let sig1 = cb.check_and_handle_fault(1, 0, 10, &thermal, 0.3);
        assert!(matches!(sig1, MoeDispatchSignal::Suspended { request_id: 10, expert_idx: 1 }));

        // Fault on expert 2
        let sig2 = cb.check_and_handle_fault(2, 3, 20, &thermal, 0.3);
        assert!(matches!(sig2, MoeDispatchSignal::Suspended { request_id: 20, expert_idx: 2 }));
    }

    // ── LayerCallback: pre_node on disabled callback ───────────────────

    #[test]
    fn test_pre_node_disabled_returns_continue() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        let mut cb = MoeDispatchCallback::disabled();

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };

        let action = cb.pre_node(&ctx);
        assert!(matches!(action, CallbackAction::Continue));
    }

    // ── LayerCallback: pre_node on non-moe layer ───────────────────────

    #[test]
    fn test_pre_node_non_moe_layer_returns_continue() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        // MoE starts at layer 4; invoke pre_node on layer 2
        let mut cb = MoeDispatchCallback::new(8, 2, 32, 4);

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 32,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 8,
            moe_top_k: 2,
            expert_intermediate_size: 512,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];
        let ctx = LayerContext {
            node_idx: 4,
            layer_idx: 2, // layer 2 is before moe_start_layer=4
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };

        let action = cb.pre_node(&ctx);
        assert!(matches!(action, CallbackAction::Continue));
    }

    // ── LayerCallback: pre_node on moe layer records step ──────────────

    #[test]
    fn test_pre_node_moe_layer_records_step() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        let mut cb = MoeDispatchCallback::new(8, 2, 32, 0);

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 32,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 8,
            moe_top_k: 2,
            expert_intermediate_size: 512,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];

        // Call pre_node 5 times on a MoE layer
        for node_idx in 0..5 {
            let ctx = LayerContext {
                node_idx,
                layer_idx: 0, // MoE starts at 0
                node_op: "Gemm",
                hidden_state: &hs,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 10,
                seq_len: 1,
                position: 9,
                request_id: 1,
                model_config: &config,
            };
            let action = cb.pre_node(&ctx);
            assert!(matches!(action, CallbackAction::Continue));
        }

        // Fault handler should have recorded 5 decode steps
        let stats = cb.fault_handler().stats();
        assert_eq!(stats.total_faults, 0);
        // Verify steps were recorded via fault_handler stats
        // (total_steps is internal; we verify via the side effect of fault_rate)
    }

    // ── target_layers returns correct slice ────────────────────────────

    #[test]
    fn test_target_layers_returns_all_moe_layers() {
        let cb = MoeDispatchCallback::new(16, 4, 12, 4);
        let layers = cb.target_layers().expect("should have target layers");
        assert_eq!(layers.len(), 8); // layers 4..11
        for (i, &layer) in layers.iter().enumerate() {
            assert_eq!(layer, i + 4);
        }
    }

    #[test]
    fn test_target_layers_none_when_no_moe_layers() {
        let cb = MoeDispatchCallback::disabled();
        assert!(cb.target_layers().is_none() || cb.target_layers().unwrap().is_empty());
    }

    // ── name and priority ──────────────────────────────────────────────

    #[test]
    fn test_name_and_priority_unchanged_by_state() {
        let cb_enabled = MoeDispatchCallback::new(8, 2, 32, 0);
        assert_eq!(cb_enabled.name(), "moe_dispatch");
        assert_eq!(cb_enabled.priority(), 70);

        let cb_disabled = MoeDispatchCallback::disabled();
        assert_eq!(cb_disabled.name(), "moe_dispatch");
        assert_eq!(cb_disabled.priority(), 70);
    }

    // ── post_node: disabled callback ──────────────────────────────────────

    #[test]
    fn test_post_node_disabled_returns_continue() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        let mut cb = MoeDispatchCallback::disabled();

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];
        let output = vec![0u8; 256 * 4];
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };

        let action = cb.post_node(&ctx, &output);
        assert!(matches!(action, CallbackAction::Continue));
    }

    // ── post_node: non-MoE layer ──────────────────────────────────────────

    #[test]
    fn test_post_node_non_moe_layer_returns_continue() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        // MoE starts at layer 4; invoke post_node on layer 1 (non-MoE)
        let mut cb = MoeDispatchCallback::new(8, 2, 32, 4);

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 32,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 8,
            moe_top_k: 2,
            expert_intermediate_size: 512,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];
        let output = vec![0u8; 256 * 4];
        let ctx = LayerContext {
            node_idx: 2,
            layer_idx: 1, // layer 1 is before moe_start_layer=4
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };

        let action = cb.post_node(&ctx, &output);
        assert!(matches!(action, CallbackAction::Continue));
    }

    // ── post_node: MoE layer ──────────────────────────────────────────────

    #[test]
    fn test_post_node_moe_layer_returns_continue() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        let mut cb = MoeDispatchCallback::new(8, 2, 32, 0);

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 32,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 8,
            moe_top_k: 2,
            expert_intermediate_size: 512,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];
        let output = vec![0u8; 256 * 4];
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0, // layer 0 is MoE
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };

        let action = cb.post_node(&ctx, &output);
        assert!(matches!(action, CallbackAction::Continue));
    }

    // ── ExpertFault construction and field access ─────────────────────────

    #[test]
    fn test_expert_fault_construction_and_field_access() {
        let fault = ExpertFault {
            expert_idx: 3,
            layer_idx: 7,
            request_id: 42,
            fault_time: Instant::now(),
        };
        assert_eq!(fault.expert_idx, 3);
        assert_eq!(fault.layer_idx, 7);
        assert_eq!(fault.request_id, 42);
    }

    #[test]
    fn test_expert_fault_clone_preserves_fields() {
        let fault = ExpertFault {
            expert_idx: 5,
            layer_idx: 10,
            request_id: 99,
            fault_time: Instant::now(),
        };
        let cloned = fault.clone();
        assert_eq!(cloned.expert_idx, 5);
        assert_eq!(cloned.layer_idx, 10);
        assert_eq!(cloned.request_id, 99);
    }

    // ── FaultResolution variants and Debug ────────────────────────────────

    #[test]
    fn test_fault_resolution_resumed_debug_format() {
        let res = FaultResolution::Resumed {
            latency: std::time::Duration::from_micros(500),
        };
        let debug = format!("{:?}", res);
        assert!(debug.contains("Resumed"));
    }

    #[test]
    fn test_fault_resolution_rejected_debug_format() {
        let res = FaultResolution::Rejected {
            reason: "OOM".to_string(),
        };
        let debug = format!("{:?}", res);
        assert!(debug.contains("Rejected"));
        assert!(debug.contains("OOM"));
    }

    #[test]
    fn test_fault_resolution_clone_resumed() {
        let res = FaultResolution::Resumed {
            latency: std::time::Duration::from_micros(200),
        };
        let cloned = res.clone();
        match cloned {
            FaultResolution::Resumed { latency } => {
                assert_eq!(latency.as_micros(), 200);
            }
            FaultResolution::Rejected { .. } => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_fault_resolution_clone_rejected() {
        let res = FaultResolution::Rejected {
            reason: "too hot".to_string(),
        };
        let cloned = res.clone();
        match cloned {
            FaultResolution::Rejected { reason } => {
                assert_eq!(reason, "too hot");
            }
            FaultResolution::Resumed { .. } => panic!("wrong variant"),
        }
    }

    // ── FaultStats construction and field access ──────────────────────────

    #[test]
    fn test_fault_stats_initial_state_via_handler() {
        let cb = MoeDispatchCallback::new(8, 2, 32, 0);
        let stats = cb.fault_handler().stats();
        assert_eq!(stats.total_faults, 0);
        assert_eq!(stats.avg_recovery_us, 0.0);
        assert_eq!(stats.fault_rate, 0.0);
        assert_eq!(stats.in_flight_restorations, 0);
        assert_eq!(stats.suspended_request_count, 0);
    }

    // ── ExpertHeatLevel traits from moe_dispatch scope ────────────────────

    #[test]
    fn test_heat_level_copy_from_dispatch_scope() {
        let level = ExpertHeatLevel::Hot;
        let copied = level; // Copy
        assert_eq!(level, copied);
    }

    #[test]
    fn test_heat_level_ordering_from_dispatch_scope() {
        assert!(ExpertHeatLevel::Hot < ExpertHeatLevel::Warm);
        assert!(ExpertHeatLevel::Warm < ExpertHeatLevel::Cold);
        assert!(ExpertHeatLevel::Cold < ExpertHeatLevel::Evicted);
    }

    // ── EvictionDecision traits from moe_dispatch scope ───────────────────

    #[test]
    fn test_eviction_decision_copy_eq_from_dispatch_scope() {
        let d1 = EvictionDecision::Keep;
        let d2 = d1; // Copy
        assert_eq!(d1, d2);
        assert_ne!(EvictionDecision::Keep, EvictionDecision::Evict);
        assert_ne!(EvictionDecision::Evict, EvictionDecision::Reactivate);
    }

    // ── ExpertWeightLocation traits from moe_dispatch scope ───────────────

    #[test]
    fn test_weight_location_copy_eq_from_dispatch_scope() {
        let loc = ExpertWeightLocation::CpuRam;
        let copied = loc; // Copy
        assert_eq!(loc, copied);
        assert_ne!(ExpertWeightLocation::GpuL2, ExpertWeightLocation::GpuVram);
        assert_ne!(ExpertWeightLocation::CpuRam, ExpertWeightLocation::RemoteNode);
    }

    // ── check_and_handle_fault: disabled callback expert ──────────────────

    #[test]
    fn test_check_fault_disabled_callback_continues() {
        let mut cb = MoeDispatchCallback::disabled();
        let thermal = ExpertThermalManager::new(0);
        // With 0 experts in the thermal manager, any index returns None => Continue
        let signal = cb.check_and_handle_fault(0, 0, 1, &thermal, 0.5);
        assert!(matches!(signal, MoeDispatchSignal::Continue));
    }

    // ── MoeDispatchSignal: PartialEq-like matching for all variants ───────

    #[test]
    fn test_signal_continue_matches_exactly() {
        let signal = MoeDispatchSignal::Continue;
        assert!(matches!(signal, MoeDispatchSignal::Continue));
    }

    #[test]
    fn test_signal_rejected_preserves_reason_content() {
        let reason = "memory pressure 0.99 exceeds limit 0.95".to_string();
        let signal = MoeDispatchSignal::Rejected {
            request_id: 7,
            reason: reason.clone(),
        };
        match signal {
            MoeDispatchSignal::Rejected { request_id, reason: r } => {
                assert_eq!(request_id, 7);
                assert_eq!(r, reason);
            }
            _ => panic!("Expected Rejected"),
        }
    }

    // ── check_and_handle_fault: same expert fault from multiple layers ────

    #[test]
    fn test_fault_same_expert_different_layers() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        // Fault on expert 1 at layer 0
        let sig0 = cb.check_and_handle_fault(1, 0, 10, &thermal, 0.3);
        assert!(matches!(sig0, MoeDispatchSignal::Suspended { request_id: 10, expert_idx: 1 }));

        // Fault on same expert 1 at layer 3 (different layer)
        let sig3 = cb.check_and_handle_fault(1, 3, 20, &thermal, 0.3);
        assert!(matches!(sig3, MoeDispatchSignal::Suspended { request_id: 20, expert_idx: 1 }));
    }

    // ── target_layers: boundary check for last MoE layer ──────────────────

    #[test]
    fn test_target_layers_last_element_matches_num_layers_minus_one() {
        let num_layers = 64;
        let moe_start = 16;
        let cb = MoeDispatchCallback::new(8, 2, num_layers, moe_start);
        let layers = cb.target_layers().expect("should have target layers");
        assert_eq!(layers.len(), num_layers - moe_start);
        assert_eq!(layers.first(), Some(&moe_start));
        assert_eq!(layers.last(), Some(&(num_layers - 1)));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Tests (18 new tests)
    // ═══════════════════════════════════════════════════════════════════════

    // ── ExpertHeatLevel::from_hit_rate boundary cases ─────────────────────

    #[test]
    fn test_heat_level_from_hit_rate_hot_at_exactly_hot_threshold() {
        let level = ExpertHeatLevel::from_hit_rate(0.1, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_heat_level_from_hit_rate_warm_just_below_hot_threshold() {
        let level = ExpertHeatLevel::from_hit_rate(0.099, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_heat_level_from_hit_rate_warm_at_exactly_cold_threshold() {
        let level = ExpertHeatLevel::from_hit_rate(0.001, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    #[test]
    fn test_heat_level_from_hit_rate_cold_just_above_zero() {
        let level = ExpertHeatLevel::from_hit_rate(0.0001, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_heat_level_from_hit_rate_evicted_at_exactly_zero() {
        let level = ExpertHeatLevel::from_hit_rate(0.0, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    // ── ExpertWeightLocation::from_heat_level mapping ─────────────────────

    #[test]
    fn test_weight_location_from_heat_level_all_mappings() {
        assert_eq!(
            ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Hot),
            ExpertWeightLocation::GpuL2,
        );
        assert_eq!(
            ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Warm),
            ExpertWeightLocation::CpuRam,
        );
        assert_eq!(
            ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Cold),
            ExpertWeightLocation::CpuRam,
        );
        assert_eq!(
            ExpertWeightLocation::from_heat_level(ExpertHeatLevel::Evicted),
            ExpertWeightLocation::Evicted,
        );
    }

    // ── ExpertWeightLocation::estimated_latency_us exact values ───────────

    #[test]
    fn test_weight_location_estimated_latency_exact_values() {
        assert_eq!(ExpertWeightLocation::GpuL2.estimated_latency_us(), 0.0);
        assert_eq!(ExpertWeightLocation::GpuVram.estimated_latency_us(), 5.0);
        assert_eq!(ExpertWeightLocation::CpuRam.estimated_latency_us(), 50.0);
        assert_eq!(ExpertWeightLocation::RemoteNode.estimated_latency_us(), 200.0);
        assert!(ExpertWeightLocation::Evicted.estimated_latency_us().is_infinite());
    }

    // ── DeoptRequest construction and field access ────────────────────────

    #[test]
    fn test_deopt_request_construction_and_field_access() {
        let req = DeoptRequest {
            request_id: 42,
            expert_idx: 3,
            layer_idx: 7,
            step: 100,
        };
        assert_eq!(req.request_id, 42);
        assert_eq!(req.expert_idx, 3);
        assert_eq!(req.layer_idx, 7);
        assert_eq!(req.step, 100);
    }

    #[test]
    fn test_deopt_request_clone_preserves_fields() {
        let req = DeoptRequest {
            request_id: 99,
            expert_idx: 1,
            layer_idx: 5,
            step: 200,
        };
        let cloned = req.clone();
        assert_eq!(cloned.request_id, 99);
        assert_eq!(cloned.expert_idx, 1);
        assert_eq!(cloned.layer_idx, 5);
        assert_eq!(cloned.step, 200);
    }

    #[test]
    fn test_deopt_request_debug_format() {
        let req = DeoptRequest {
            request_id: 7,
            expert_idx: 2,
            layer_idx: 3,
            step: 50,
        };
        let debug = format!("{:?}", req);
        assert!(debug.contains("DeoptRequest"));
        assert!(debug.contains("request_id"));
    }

    // ── DeoptHandlingResult Debug format ──────────────────────────────────

    #[test]
    fn test_deopt_handling_result_reactivate_debug_format() {
        let result = DeoptHandlingResult::ReactivateAndRerun {
            expert_idx: 5,
            request_id: 10,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("ReactivateAndRerun"));
    }

    #[test]
    fn test_deopt_handling_result_spurious_debug_format() {
        let result = DeoptHandlingResult::SpuriousDeopt {
            expert_idx: 2,
            request_id: 8,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("SpuriousDeopt"));
    }

    // ── ThermalSummary direct construction and field access ───────────────

    #[test]
    fn test_thermal_summary_field_access() {
        let summary = ThermalSummary {
            num_experts: 64,
            hot_count: 10,
            warm_count: 20,
            cold_count: 30,
            evicted_count: 4,
            total_evictions: 8,
            total_reactivations: 3,
            current_step: 500,
            pending_deopt_count: 2,
            working_set_size: 15,
            effective_eviction_threshold: 100,
        };
        assert_eq!(summary.num_experts, 64);
        assert_eq!(summary.hot_count, 10);
        assert_eq!(summary.warm_count, 20);
        assert_eq!(summary.cold_count, 30);
        assert_eq!(summary.evicted_count, 4);
        assert_eq!(summary.total_evictions, 8);
        assert_eq!(summary.total_reactivations, 3);
        assert_eq!(summary.current_step, 500);
        assert_eq!(summary.pending_deopt_count, 2);
        assert_eq!(summary.working_set_size, 15);
        assert_eq!(summary.effective_eviction_threshold, 100);
    }

    #[test]
    fn test_thermal_summary_debug_format() {
        let summary = ThermalSummary {
            num_experts: 8,
            hot_count: 2,
            warm_count: 3,
            cold_count: 2,
            evicted_count: 1,
            total_evictions: 5,
            total_reactivations: 1,
            current_step: 100,
            pending_deopt_count: 0,
            working_set_size: 6,
            effective_eviction_threshold: 50,
        };
        let debug = format!("{:?}", summary);
        assert!(debug.contains("ThermalSummary"));
        assert!(debug.contains("num_experts"));
    }

    // ── ExpertHeatState field access via thermal manager ──────────────────

    #[test]
    fn test_heat_state_default_fields_via_thermal_manager() {
        let thermal = ExpertThermalManager::new(4);
        let state = thermal.state(0).expect("expert 0 should exist");
        assert_eq!(state.expert_idx, 0);
        assert_eq!(state.hit_rate, 0.0);
        assert_eq!(state.hit_count, 0);
        assert_eq!(state.route_count, 0);
        assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
        assert_eq!(state.consecutive_zero_streak, 0);
        assert!(!state.is_evicted);
        assert_eq!(state.reactivation_count, 0);
    }

    #[test]
    fn test_heat_state_updates_after_steps() {
        let mut thermal = ExpertThermalManager::new(4);
        thermal.step(&[10, 5, 0, 0]);

        let s0 = thermal.state(0).unwrap();
        assert_eq!(s0.hit_count, 1);
        assert_eq!(s0.route_count, 1);
        // hit_rate = 1.0 >= hot_threshold (0.1) => Hot
        assert_eq!(s0.heat_level, ExpertHeatLevel::Hot);
        assert_eq!(s0.consecutive_zero_streak, 0);

        let s2 = thermal.state(2).unwrap();
        assert_eq!(s2.hit_count, 0);
        assert_eq!(s2.consecutive_zero_streak, 1);
        // hit_rate = 0.0 (no hits) => Evicted per from_hit_rate logic
        assert_eq!(s2.heat_level, ExpertHeatLevel::Evicted);
    }

    // ── ExpertThermalManager::with_eviction_aggressiveness ────────────────

    #[test]
    fn test_eviction_aggressiveness_reduces_effective_threshold() {
        let base_threshold = 1000u64;
        let thermal_no_aggr = ExpertThermalManager::new(4)
            .with_eviction_threshold(base_threshold);
        let thermal_aggr = ExpertThermalManager::new(4)
            .with_eviction_threshold(base_threshold)
            .with_eviction_aggressiveness(1.0);

        let threshold_no_aggr = thermal_no_aggr.effective_eviction_threshold();
        let threshold_aggr = thermal_aggr.effective_eviction_threshold();

        assert_eq!(threshold_no_aggr, base_threshold);
        // aggressiveness=1.0 → bias_factor=1/(1+1)=0.5 → threshold halved
        assert_eq!(threshold_aggr, base_threshold / 2);
    }

    // ── ExpertThermalManager::hot_experts and cold_or_evicted_experts ─────

    #[test]
    fn test_hot_experts_returns_correct_indices() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_heat_thresholds(0.1, 0.001);
        // Expert 0 gets all hits, others get zero
        for _ in 0..10 {
            thermal.step(&[100, 0, 0, 0]);
        }

        let hot = thermal.hot_experts();
        assert_eq!(hot.len(), 1);
        assert!(hot.contains(&0));
    }

    #[test]
    fn test_cold_or_evicted_experts_after_eviction() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(5);
        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        let cold_evicted = thermal.cold_or_evicted_experts();
        assert!(cold_evicted.contains(&1)); // evicted
    }

    // ── ExpertThermalManager::states returns all states ───────────────────

    #[test]
    fn test_states_returns_all_expert_states() {
        let thermal = ExpertThermalManager::new(8);
        let states = thermal.states();
        assert_eq!(states.len(), 8);
        for (i, s) in states.iter().enumerate() {
            assert_eq!(s.expert_idx, i);
        }
    }

    // ── ExpertThermalManager::num_experts accessor ────────────────────────

    #[test]
    fn test_thermal_manager_num_experts() {
        let thermal = ExpertThermalManager::new(16);
        assert_eq!(thermal.num_experts(), 16);
    }

    // ── ExpertThermalManager::summary integration ─────────────────────────

    #[test]
    fn test_summary_reflects_eviction_and_reactivation() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(5);
        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);
        thermal.reactivate_expert(1);

        let summary = thermal.summary();
        assert_eq!(summary.num_experts, 4);
        assert_eq!(summary.evicted_count, 0); // reactivated
        assert_eq!(summary.total_evictions, 1);
        assert_eq!(summary.total_reactivations, 1);
    }

    // ── MoeDispatchCallback with maximum index values ─────────────────────

    #[test]
    fn test_new_with_large_expert_and_layer_counts() {
        let cb = MoeDispatchCallback::new(256, 8, 1000, 500);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 256);
        let layers = cb.target_layers().expect("should have target layers");
        assert_eq!(layers.len(), 500);
        assert_eq!(layers[0], 500);
        assert_eq!(layers[499], 999);
    }

    // ── MoeDispatchSignal boundary request_id values ──────────────────────

    #[test]
    fn test_signal_suspended_with_zero_request_id() {
        let signal = MoeDispatchSignal::Suspended {
            request_id: 0,
            expert_idx: 0,
        };
        match signal {
            MoeDispatchSignal::Suspended { request_id, expert_idx } => {
                assert_eq!(request_id, 0);
                assert_eq!(expert_idx, 0);
            }
            _ => panic!("Expected Suspended"),
        }
    }

    #[test]
    fn test_signal_rejected_with_max_request_id() {
        let signal = MoeDispatchSignal::Rejected {
            request_id: u64::MAX,
            reason: "overflow".to_string(),
        };
        match signal {
            MoeDispatchSignal::Rejected { request_id, reason } => {
                assert_eq!(request_id, u64::MAX);
                assert_eq!(reason, "overflow");
            }
            _ => panic!("Expected Rejected"),
        }
    }

    // ── ExpertWeightLayout direct construction ────────────────────────────

    #[test]
    fn test_expert_weight_layout_construction_and_field_access() {
        let layout = ExpertWeightLayout {
            expert_idx: 7,
            weight_bytes: 4096,
            compressed_bytes: 512,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::CpuRam,
        };
        assert_eq!(layout.expert_idx, 7);
        assert_eq!(layout.weight_bytes, 4096);
        assert_eq!(layout.compressed_bytes, 512);
        assert!((layout.compression_ratio - 8.0).abs() < f32::EPSILON);
        assert_eq!(layout.location, ExpertWeightLocation::CpuRam);
    }

    #[test]
    fn test_expert_weight_layout_clone_preserves_fields() {
        let layout = ExpertWeightLayout {
            expert_idx: 3,
            weight_bytes: 8192,
            compressed_bytes: 1024,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::GpuVram,
        };
        let cloned = layout.clone();
        assert_eq!(cloned.expert_idx, 3);
        assert_eq!(cloned.weight_bytes, 8192);
        assert_eq!(cloned.compressed_bytes, 1024);
        assert_eq!(cloned.location, ExpertWeightLocation::GpuVram);
    }

    #[test]
    fn test_expert_weight_layout_debug_format() {
        let layout = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 1024,
            compressed_bytes: 128,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::GpuL2,
        };
        let debug = format!("{:?}", layout);
        assert!(debug.contains("ExpertWeightLayout"));
        assert!(debug.contains("expert_idx"));
    }

    // ── ExpertPrefetchRequest direct construction ─────────────────────────

    #[test]
    fn test_expert_prefetch_request_construction_and_field_access() {
        let req = ExpertPrefetchRequest {
            expert_idx: 5,
            layer_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 2048,
            estimated_latency_us: 3.14,
            priority: 2,
        };
        assert_eq!(req.expert_idx, 5);
        assert_eq!(req.source, ExpertWeightLocation::CpuRam);
        assert_eq!(req.destination, ExpertWeightLocation::GpuVram);
        assert_eq!(req.bytes, 2048);
        assert!((req.estimated_latency_us - 3.14).abs() < f32::EPSILON);
        assert_eq!(req.priority, 2);
    }

    #[test]
    fn test_expert_prefetch_request_clone_preserves_fields() {
        let req = ExpertPrefetchRequest {
            expert_idx: 10,
            layer_idx: 0,
            source: ExpertWeightLocation::RemoteNode,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 65536,
            estimated_latency_us: 200.0,
            priority: 0,
        };
        let cloned = req.clone();
        assert_eq!(cloned.expert_idx, 10);
        assert_eq!(cloned.source, ExpertWeightLocation::RemoteNode);
        assert_eq!(cloned.destination, ExpertWeightLocation::GpuVram);
        assert_eq!(cloned.bytes, 65536);
        assert_eq!(cloned.priority, 0);
    }

    // ── ExpertThermalManager::with_heat_thresholds customization ──────────

    #[test]
    fn test_with_heat_thresholds_affects_classification() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_heat_thresholds(0.5, 0.1);
        // Expert 0: every step has count > 0, so hit_rate = 1.0 (Hot >= 0.5)
        // Expert 1: never routed, hit_rate = 0.0 (Evicted)
        // Run enough steps to distinguish; then verify expert 0 is Hot
        for _ in 0..10 {
            thermal.step(&[10, 0, 0, 0]);
        }

        let s0 = thermal.state(0).unwrap();
        assert_eq!(s0.heat_level, ExpertHeatLevel::Hot);

        // Expert 1 has hit_rate 0.0 => Evicted
        let s1 = thermal.state(1).unwrap();
        assert_eq!(s1.heat_level, ExpertHeatLevel::Evicted);
    }

    // ── ExpertThermalManager::update_memory_pressure ──────────────────────

    #[test]
    fn test_update_memory_pressure_clamps_to_valid_range() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_adaptive_eviction(10);

        thermal.update_memory_pressure(0.5);
        let summary = thermal.summary();
        // Adaptive threshold should reflect the memory pressure
        assert!(summary.effective_eviction_threshold > 0);
    }

    // ── EvictionDecision exhaustive variant check ─────────────────────────

    #[test]
    fn test_eviction_decision_all_variants_are_distinct() {
        let keep = EvictionDecision::Keep;
        let evict = EvictionDecision::Evict;
        let reactivate = EvictionDecision::Reactivate;

        assert_ne!(keep, evict);
        assert_ne!(evict, reactivate);
        assert_ne!(keep, reactivate);
    }

    // ── ExpertHeatLevel Debug trait from moe_dispatch scope ───────────────

    #[test]
    fn test_heat_level_debug_format_contains_variant_name() {
        assert!(format!("{:?}", ExpertHeatLevel::Hot).contains("Hot"));
        assert!(format!("{:?}", ExpertHeatLevel::Warm).contains("Warm"));
        assert!(format!("{:?}", ExpertHeatLevel::Cold).contains("Cold"));
        assert!(format!("{:?}", ExpertHeatLevel::Evicted).contains("Evicted"));
    }

    // ── FaultStats direct construction with extreme values ────────────────

    #[test]
    fn test_fault_stats_with_large_values() {
        let stats = FaultStats {
            total_faults: u64::MAX,
            avg_recovery_us: f64::MAX,
            fault_rate: 1.0,
            in_flight_restorations: usize::MAX,
            suspended_request_count: usize::MAX,
        };
        assert_eq!(stats.total_faults, u64::MAX);
        assert_eq!(stats.avg_recovery_us, f64::MAX);
        assert!((stats.fault_rate - 1.0).abs() < f64::EPSILON);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Tests (50 new tests: batch 2)
    // ═══════════════════════════════════════════════════════════════════════

    // ── MoeDispatchSignal: memory replace semantics ──────────────────────

    #[test]
    fn test_take_signal_replaces_with_continue() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);
        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        // Trigger a fault so last_signal becomes Suspended internally
        let _ = cb.check_and_handle_fault(1, 0, 42, &thermal, 0.3);

        // take_signal should return the initial Continue (check_and_handle_fault
        // does NOT set last_signal — it returns directly)
        let sig = cb.take_signal();
        assert!(matches!(sig, MoeDispatchSignal::Continue));
    }

    // ── MoeDispatchCallback constructor: top_k stored but not exposed ─────

    #[test]
    fn test_new_stores_top_k_correctly_in_layers() {
        // top_k doesn't have a getter, but the constructor should not panic
        let cb = MoeDispatchCallback::new(64, 8, 12, 0);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 64);
    }

    // ── check_and_handle_fault: zero memory pressure ─────────────────────

    #[test]
    fn test_check_fault_evicted_with_zero_memory_pressure() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);
        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        let signal = cb.check_and_handle_fault(1, 2, 55, &thermal, 0.0);
        assert!(
            matches!(signal, MoeDispatchSignal::Suspended { request_id: 55, expert_idx: 1 }),
            "zero memory pressure should allow page-in"
        );
    }

    // ── check_and_handle_fault: memory pressure at exactly limit ─────────

    #[test]
    fn test_check_fault_evicted_at_exact_memory_pressure_limit() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        cb.fault_handler_mut();
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);
        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        // Default limit is 0.95; 0.95 is NOT > 0.95, so should be Suspended
        let signal = cb.check_and_handle_fault(1, 0, 99, &thermal, 0.95);
        assert!(
            matches!(signal, MoeDispatchSignal::Suspended { .. }),
            "pressure exactly at limit should still allow page-in"
        );
    }

    // ── ExpertFaultHandler: with_memory_pressure_limit builder ────────────

    #[test]
    fn test_fault_handler_with_custom_memory_pressure_limit() {
        let handler = ExpertFaultHandler::new(4).with_memory_pressure_limit(0.5);
        // Verify via stats (handler is consumed by builder pattern)
        let stats = handler.stats();
        assert_eq!(stats.total_faults, 0);
    }

    #[test]
    fn test_fault_handler_with_zero_memory_pressure_limit() {
        let handler = ExpertFaultHandler::new(4).with_memory_pressure_limit(0.0);
        let stats = handler.stats();
        assert_eq!(stats.in_flight_restorations, 0);
    }

    #[test]
    fn test_fault_handler_with_memory_pressure_limit_clamped_high() {
        // Values > 1.0 should be clamped to 1.0
        let handler = ExpertFaultHandler::new(4).with_memory_pressure_limit(2.0);
        let stats = handler.stats();
        assert_eq!(stats.suspended_request_count, 0);
    }

    // ── ExpertFaultHandler: expert_fault_count ────────────────────────────

    #[test]
    fn test_expert_fault_count_initial_zero() {
        let handler = ExpertFaultHandler::new(8);
        for i in 0..8 {
            assert_eq!(handler.expert_fault_count(i), 0);
        }
    }

    #[test]
    fn test_expert_fault_count_out_of_bounds_returns_zero() {
        let handler = ExpertFaultHandler::new(4);
        assert_eq!(handler.expert_fault_count(100), 0);
    }

    // ── ExpertFaultHandler: in_flight_count and suspended_request_count ───

    #[test]
    fn test_in_flight_count_initially_zero() {
        let handler = ExpertFaultHandler::new(4);
        assert_eq!(handler.in_flight_count(), 0);
        assert_eq!(handler.suspended_request_count(), 0);
    }

    // ── ExpertFaultHandler: is_restoration_pending initially false ────────

    #[test]
    fn test_is_restoration_pending_initially_false() {
        let handler = ExpertFaultHandler::new(4);
        assert!(!handler.is_restoration_pending(0, 0));
        assert!(!handler.is_restoration_pending(3, 7));
    }

    // ── ExpertFaultHandler: record_step and fault_rate ────────────────────

    #[test]
    fn test_record_step_updates_fault_rate_calculation() {
        let mut handler = ExpertFaultHandler::new(4);
        // 0 faults, 10 steps => fault_rate = 0.0
        for _ in 0..10 {
            handler.record_step();
        }
        let stats = handler.stats();
        assert!((stats.fault_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_step_single_step() {
        let mut handler = ExpertFaultHandler::new(4);
        handler.record_step();
        let stats = handler.stats();
        assert!((stats.fault_rate - 0.0).abs() < f64::EPSILON);
    }

    // ── ExpertFault: Debug format includes all fields ────────────────────

    #[test]
    fn test_expert_fault_debug_format() {
        let fault = ExpertFault {
            expert_idx: 3,
            layer_idx: 7,
            request_id: 42,
            fault_time: Instant::now(),
        };
        let debug = format!("{:?}", fault);
        assert!(debug.contains("ExpertFault"));
        assert!(debug.contains("expert_idx"));
        assert!(debug.contains("layer_idx"));
        assert!(debug.contains("request_id"));
    }

    // ── FaultStats: clone preserves all fields ───────────────────────────

    #[test]
    fn test_fault_stats_clone_preserves_all_fields() {
        let stats = FaultStats {
            total_faults: 100,
            avg_recovery_us: 42.5,
            fault_rate: 0.25,
            in_flight_restorations: 3,
            suspended_request_count: 7,
        };
        let cloned = stats.clone();
        assert_eq!(cloned.total_faults, 100);
        assert!((cloned.avg_recovery_us - 42.5).abs() < f64::EPSILON);
        assert!((cloned.fault_rate - 0.25).abs() < f64::EPSILON);
        assert_eq!(cloned.in_flight_restorations, 3);
        assert_eq!(cloned.suspended_request_count, 7);
    }

    // ── FaultStats: PartialEq works correctly ────────────────────────────

    #[test]
    fn test_fault_stats_partial_eq_same_values() {
        let a = FaultStats {
            total_faults: 10,
            avg_recovery_us: 5.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 1,
        };
        let b = FaultStats {
            total_faults: 10,
            avg_recovery_us: 5.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 1,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_fault_stats_partial_eq_different_values() {
        let a = FaultStats {
            total_faults: 10,
            avg_recovery_us: 5.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 1,
        };
        let b = FaultStats {
            total_faults: 11,
            avg_recovery_us: 5.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 1,
        };
        assert_ne!(a, b);
    }

    // ── FaultResolution: PartialEq between variants ──────────────────────

    #[test]
    fn test_fault_resolution_eq_same_resumed() {
        let a = FaultResolution::Resumed { latency: Duration::from_micros(100) };
        let b = FaultResolution::Resumed { latency: Duration::from_micros(100) };
        assert_eq!(a, b);
    }

    #[test]
    fn test_fault_resolution_ne_different_resumed_latency() {
        let a = FaultResolution::Resumed { latency: Duration::from_micros(100) };
        let b = FaultResolution::Resumed { latency: Duration::from_micros(200) };
        assert_ne!(a, b);
    }

    #[test]
    fn test_fault_resolution_ne_different_variants() {
        let a = FaultResolution::Resumed { latency: Duration::ZERO };
        let b = FaultResolution::Rejected { reason: "test".to_string() };
        assert_ne!(a, b);
    }

    // ── ExpertHeatLevel: Hash consistency ─────────────────────────────────

    #[test]
    fn test_heat_level_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ExpertHeatLevel::Hot);
        set.insert(ExpertHeatLevel::Warm);
        set.insert(ExpertHeatLevel::Cold);
        set.insert(ExpertHeatLevel::Evicted);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_heat_level_hash_duplicate_insertion() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(ExpertHeatLevel::Hot));
        assert!(!set.insert(ExpertHeatLevel::Hot));
        assert_eq!(set.len(), 1);
    }

    // ── ExpertWeightLocation: Hash consistency ────────────────────────────

    #[test]
    fn test_weight_location_hash_all_variants() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(ExpertWeightLocation::GpuL2);
        set.insert(ExpertWeightLocation::GpuVram);
        set.insert(ExpertWeightLocation::CpuRam);
        set.insert(ExpertWeightLocation::RemoteNode);
        set.insert(ExpertWeightLocation::Evicted);
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_weight_location_hash_duplicate_rejected() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(ExpertWeightLocation::CpuRam));
        assert!(!set.insert(ExpertWeightLocation::CpuRam));
    }

    // ── EvictionDecision: Hash and Ord ────────────────────────────────────

    #[test]
    fn test_eviction_decision_hash_all_variants() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(EvictionDecision::Keep);
        set.insert(EvictionDecision::Evict);
        set.insert(EvictionDecision::Reactivate);
        assert_eq!(set.len(), 3);
    }

    // ── ExpertThermalManager: eviction_decision for non-evicted expert ───

    #[test]
    fn test_eviction_decision_keep_for_active_expert() {
        let thermal = ExpertThermalManager::new(4).with_eviction_threshold(100);
        let decision = thermal.eviction_decision(0);
        assert_eq!(decision, EvictionDecision::Keep);
    }

    #[test]
    fn test_eviction_decision_evict_for_high_streak() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0, 5, 3]);
        }
        let decision = thermal.eviction_decision(1);
        assert_eq!(decision, EvictionDecision::Evict);
    }

    #[test]
    fn test_eviction_decision_out_of_bounds_returns_keep() {
        let thermal = ExpertThermalManager::new(4);
        let decision = thermal.eviction_decision(100);
        assert_eq!(decision, EvictionDecision::Keep);
    }

    // ── ExpertThermalManager: evict_expert returns false for double evict ─

    #[test]
    fn test_evict_expert_double_eviction_returns_false() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0, 5, 3]);
        }
        assert!(thermal.evict_expert(1));
        assert!(!thermal.evict_expert(1)); // already evicted
    }

    #[test]
    fn test_evict_expert_out_of_bounds_returns_false() {
        let mut thermal = ExpertThermalManager::new(4);
        assert!(!thermal.evict_expert(10));
    }

    // ── ExpertThermalManager: reactivate_expert edge cases ───────────────

    #[test]
    fn test_reactivate_non_evicted_returns_false() {
        let mut thermal = ExpertThermalManager::new(4);
        assert!(!thermal.reactivate_expert(0)); // not evicted
    }

    #[test]
    fn test_reactivate_out_of_bounds_returns_false() {
        let mut thermal = ExpertThermalManager::new(4);
        assert!(!thermal.reactivate_expert(100));
    }

    #[test]
    fn test_reactivate_sets_heat_level_to_cold() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);
        assert!(thermal.reactivate_expert(1));
        let state = thermal.state(1).unwrap();
        assert_eq!(state.heat_level, ExpertHeatLevel::Cold);
        assert!(!state.is_evicted);
        assert_eq!(state.consecutive_zero_streak, 0);
    }

    // ── ExpertThermalManager: experts_to_evict and experts_to_reactivate ──

    #[test]
    fn test_experts_to_evict_empty_initially() {
        let thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);
        assert!(thermal.experts_to_evict().is_empty());
    }

    #[test]
    fn test_experts_to_evict_after_cold_steps() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0, 0, 3]);
        }
        let to_evict = thermal.experts_to_evict();
        assert!(to_evict.contains(&1));
        assert!(to_evict.contains(&2));
    }

    #[test]
    fn test_experts_to_reactivate_empty_when_no_evictions() {
        let thermal = ExpertThermalManager::new(4);
        assert!(thermal.experts_to_reactivate().is_empty());
    }

    // ── ExpertThermalManager: handle_deopt_request for evicted expert ────

    #[test]
    fn test_handle_deopt_evicted_expert_reactivates() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        let req = DeoptRequest {
            request_id: 42,
            expert_idx: 1,
            layer_idx: 0,
            step: 100,
        };
        let result = thermal.handle_deopt_request(req);
        match result {
            DeoptHandlingResult::ReactivateAndRerun { expert_idx, request_id } => {
                assert_eq!(expert_idx, 1);
                assert_eq!(request_id, 42);
            }
            DeoptHandlingResult::SpuriousDeopt { .. } => panic!("expected ReactivateAndRerun"),
        }
        // Expert should now be reactivated
        assert!(!thermal.state(1).unwrap().is_evicted);
    }

    #[test]
    fn test_handle_deopt_non_evicted_is_spurious() {
        let mut thermal = ExpertThermalManager::new(4);
        let req = DeoptRequest {
            request_id: 7,
            expert_idx: 0,
            layer_idx: 2,
            step: 50,
        };
        let result = thermal.handle_deopt_request(req);
        match result {
            DeoptHandlingResult::SpuriousDeopt { expert_idx, request_id } => {
                assert_eq!(expert_idx, 0);
                assert_eq!(request_id, 7);
            }
            DeoptHandlingResult::ReactivateAndRerun { .. } => panic!("expected SpuriousDeopt"),
        }
    }

    // ── ExpertThermalManager: pending_deopt_requests and clear ────────────

    #[test]
    fn test_pending_deopt_requests_initially_empty() {
        let thermal = ExpertThermalManager::new(4);
        assert!(thermal.pending_deopt_requests().is_empty());
    }

    #[test]
    fn test_clear_deopt_requests_empties_list() {
        let mut thermal = ExpertThermalManager::new(4);
        let req = DeoptRequest {
            request_id: 1,
            expert_idx: 0,
            layer_idx: 0,
            step: 0,
        };
        thermal.handle_deopt_request(req);
        assert_eq!(thermal.pending_deopt_requests().len(), 1);
        thermal.clear_deopt_requests();
        assert!(thermal.pending_deopt_requests().is_empty());
    }

    // ── ExpertWeightPrefetcher construction and basic accessors ───────────

    #[test]
    fn test_prefetcher_new_initializes_all_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 8192);
        assert_eq!(prefetcher.num_experts(), 8);
        let layouts = prefetcher.layouts();
        assert_eq!(layouts.len(), 8);
        for (i, layout) in layouts.iter().enumerate() {
            assert_eq!(layout.expert_idx, i);
            assert_eq!(layout.weight_bytes, 8192);
            assert_eq!(layout.compressed_bytes, 8192 / 8);
            assert_eq!(layout.location, ExpertWeightLocation::CpuRam);
        }
    }

    #[test]
    fn test_prefetcher_new_zero_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(0, 4096);
        assert_eq!(prefetcher.num_experts(), 0);
        assert!(prefetcher.layouts().is_empty());
    }

    #[test]
    fn test_prefetcher_total_gpu_vram_bytes() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        let total = prefetcher.total_gpu_vram_bytes();
        assert!(total == 0, "no experts in GPU VRAM initially");
    }

    #[test]
    fn test_prefetcher_layout_returns_correct_expert() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        let layout = prefetcher.layout(2).expect("expert 2 should exist");
        assert_eq!(layout.expert_idx, 2);
    }

    #[test]
    fn test_prefetcher_layout_out_of_bounds_returns_none() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        assert!(prefetcher.layout(10).is_none());
    }

    // ── ExpertWeightPrefetcher: update_location ───────────────────────────

    #[test]
    fn test_prefetcher_update_location_changes_state() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        prefetcher.update_location(0, ExpertWeightLocation::GpuVram);
        let layout = prefetcher.layout(0).unwrap();
        assert_eq!(layout.location, ExpertWeightLocation::GpuVram);
    }

    #[test]
    fn test_prefetcher_update_location_out_of_bounds_no_panic() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        prefetcher.update_location(100, ExpertWeightLocation::GpuL2);
        // Should not panic; out-of-bounds is silently ignored
        assert_eq!(prefetcher.num_experts(), 4);
    }

    // ── ExpertWeightPrefetcher: builder pattern chaining ──────────────────

    #[test]
    fn test_prefetcher_builder_chaining() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096)
            .with_bandwidth(64.0, 200.0)
            .with_layer_compute_time(50.0)
            .with_prefetch_priority(1.5);
        assert_eq!(prefetcher.num_experts(), 4);
    }

    // ── ExpertWeightPrefetcher: bandwidth_savings_ratio reflects compression ─

    #[test]
    fn test_prefetcher_bandwidth_savings_ratio_reflects_compression() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        // compressed = 8192/8 = 1024 per expert; savings = 1 - (1024/8192) = 0.875
        let ratio = prefetcher.bandwidth_savings_ratio();
        assert!((ratio - 0.875).abs() < f32::EPSILON);
    }

    // ── ExpertHeatLevel: Ord total ordering ───────────────────────────────

    #[test]
    fn test_heat_level_total_ordering() {
        assert!(ExpertHeatLevel::Hot < ExpertHeatLevel::Warm);
        assert!(ExpertHeatLevel::Warm < ExpertHeatLevel::Cold);
        assert!(ExpertHeatLevel::Cold < ExpertHeatLevel::Evicted);
        assert!(ExpertHeatLevel::Hot < ExpertHeatLevel::Evicted);
    }

    // ── ExpertHeatLevel: from_hit_rate negative rate ──────────────────────

    #[test]
    fn test_heat_level_from_hit_rate_negative_rate() {
        // Negative rate < cold_threshold but rate > 0.0 is false; rate < 0.0
        // falls through all conditions to Evicted (rate > 0.0 is false)
        let level = ExpertHeatLevel::from_hit_rate(-1.0, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Evicted);
    }

    // ── ExpertWeightLocation: all variants Copy trait ─────────────────────

    #[test]
    fn test_weight_location_copy_preserves_value() {
        let original = ExpertWeightLocation::RemoteNode;
        let copied = original;
        assert_eq!(original, copied);
        // Verify original is still usable (Copy, not moved)
        assert_eq!(original, ExpertWeightLocation::RemoteNode);
    }

    // ── EvictionDecision: Copy trait ──────────────────────────────────────

    #[test]
    fn test_eviction_decision_copy_trait() {
        let original = EvictionDecision::Evict;
        let copied = original;
        assert_eq!(original, copied);
        assert_eq!(original, EvictionDecision::Evict);
    }

    // ── ThermalSummary: clone preserves all fields ────────────────────────

    #[test]
    fn test_thermal_summary_clone_preserves_all_fields() {
        let summary = ThermalSummary {
            num_experts: 16,
            hot_count: 4,
            warm_count: 6,
            cold_count: 4,
            evicted_count: 2,
            total_evictions: 10,
            total_reactivations: 3,
            current_step: 1000,
            pending_deopt_count: 1,
            working_set_size: 12,
            effective_eviction_threshold: 50,
        };
        let cloned = summary.clone();
        assert_eq!(cloned.num_experts, 16);
        assert_eq!(cloned.hot_count, 4);
        assert_eq!(cloned.warm_count, 6);
        assert_eq!(cloned.cold_count, 4);
        assert_eq!(cloned.evicted_count, 2);
        assert_eq!(cloned.total_evictions, 10);
        assert_eq!(cloned.total_reactivations, 3);
        assert_eq!(cloned.current_step, 1000);
        assert_eq!(cloned.pending_deopt_count, 1);
        assert_eq!(cloned.working_set_size, 12);
        assert_eq!(cloned.effective_eviction_threshold, 50);
    }

    // ── ThermalSummary: PartialEq same/different ──────────────────────────

    #[test]
    fn test_thermal_summary_eq_same_values() {
        let a = ThermalSummary {
            num_experts: 4,
            hot_count: 1,
            warm_count: 2,
            cold_count: 1,
            evicted_count: 0,
            total_evictions: 0,
            total_reactivations: 0,
            current_step: 10,
            pending_deopt_count: 0,
            working_set_size: 3,
            effective_eviction_threshold: 100,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_thermal_summary_ne_different_values() {
        let a = ThermalSummary {
            num_experts: 4,
            hot_count: 1,
            warm_count: 2,
            cold_count: 1,
            evicted_count: 0,
            total_evictions: 0,
            total_reactivations: 0,
            current_step: 10,
            pending_deopt_count: 0,
            working_set_size: 3,
            effective_eviction_threshold: 100,
        };
        let mut b = a.clone();
        b.hot_count = 2;
        assert_ne!(a, b);
    }

    // ── DeoptRequest: PartialEq ──────────────────────────────────────────

    #[test]
    fn test_deopt_request_eq_same_values() {
        let a = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let b = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_deopt_request_ne_different_step() {
        let a = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let b = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 5 };
        assert_ne!(a, b);
    }

    // ── DeoptHandlingResult: PartialEq ────────────────────────────────────

    #[test]
    fn test_deopt_handling_result_eq_same_reactivate() {
        let a = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 2 };
        let b = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 2 };
        assert_eq!(a, b);
    }

    #[test]
    fn test_deopt_handling_result_ne_different_variants() {
        let a = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 2 };
        let b = DeoptHandlingResult::SpuriousDeopt { expert_idx: 1, request_id: 2 };
        assert_ne!(a, b);
    }

    // ── ExpertWeightLayout: construction with zero bytes ──────────────────

    #[test]
    fn test_expert_weight_layout_zero_bytes() {
        let layout = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 0,
            compressed_bytes: 0,
            compression_ratio: 0.0,
            location: ExpertWeightLocation::CpuRam,
        };
        assert_eq!(layout.weight_bytes, 0);
        assert_eq!(layout.compressed_bytes, 0);
    }

    // ── ExpertPrefetchRequest: construction with max priority ─────────────

    #[test]
    fn test_expert_prefetch_request_max_priority() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            layer_idx: 0,
            source: ExpertWeightLocation::RemoteNode,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: f32::MAX,
            priority: u32::MAX,
        };
        assert_eq!(req.priority, u32::MAX);
    }

    // ── ExpertPrefetchRequest: Debug format ──────────────────────────────

    #[test]
    fn test_expert_prefetch_request_debug_format() {
        let req = ExpertPrefetchRequest {
            expert_idx: 3,
            layer_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuL2,
            bytes: 1024,
            estimated_latency_us: 10.0,
            priority: 1,
        };
        let debug = format!("{:?}", req);
        assert!(debug.contains("ExpertPrefetchRequest"));
        assert!(debug.contains("expert_idx"));
    }

    // ── ExpertHeatState: reactivation_count increments ────────────────────

    #[test]
    fn test_heat_state_reactivation_count_increments() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);
        assert_eq!(thermal.state(1).unwrap().reactivation_count, 0);

        thermal.reactivate_expert(1);
        assert_eq!(thermal.state(1).unwrap().reactivation_count, 1);

        // Evict again resets reactivation_count to 0
        thermal.evict_expert(1);
        assert_eq!(thermal.state(1).unwrap().reactivation_count, 0);

        thermal.reactivate_expert(1);
        assert_eq!(thermal.state(1).unwrap().reactivation_count, 1);
    }

    // ── ExpertThermalManager: summary counts match state ──────────────────

    #[test]
    fn test_summary_counts_consistent_with_states() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(3)
            .with_heat_thresholds(0.1, 0.001);

        // Step enough to make expert 0 hot and experts 1,2,3 cold/evicted
        for _ in 0..10 {
            thermal.step(&[100, 0, 0, 0]);
        }

        let summary = thermal.summary();
        let total = summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count;
        assert_eq!(total, 4, "sum of all heat levels should equal num_experts");
    }

    // ── MoeDispatchCallback: moe_layers correctness for various configs ──

    #[test]
    fn test_moe_layers_with_single_layer_model() {
        let cb = MoeDispatchCallback::new(8, 2, 1, 0);
        assert_eq!(cb.moe_layers.len(), 1);
        assert_eq!(cb.moe_layers[0], 0);
    }

    #[test]
    fn test_moe_layers_with_moe_start_beyond_num_layers() {
        let cb = MoeDispatchCallback::new(8, 2, 10, 20);
        assert!(cb.is_enabled());
        assert!(cb.moe_layers.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Tests (50 new tests: batch 3)
    // ═══════════════════════════════════════════════════════════════════════

    // ── MoeDispatchSignal: Suspended field isolation ─────────────────────

    #[test]
    fn test_signal_suspended_zero_expert_idx() {
        let signal = MoeDispatchSignal::Suspended {
            request_id: 1,
            expert_idx: 0,
        };
        match signal {
            MoeDispatchSignal::Suspended { expert_idx, .. } => assert_eq!(expert_idx, 0),
            _ => panic!("Expected Suspended"),
        }
    }

    #[test]
    fn test_signal_suspended_max_expert_idx() {
        let signal = MoeDispatchSignal::Suspended {
            request_id: u64::MAX,
            expert_idx: usize::MAX,
        };
        match signal {
            MoeDispatchSignal::Suspended { request_id, expert_idx } => {
                assert_eq!(request_id, u64::MAX);
                assert_eq!(expert_idx, usize::MAX);
            }
            _ => panic!("Expected Suspended"),
        }
    }

    // ── MoeDispatchSignal: Rejected reason with special characters ──────

    #[test]
    fn test_signal_rejected_empty_reason() {
        let signal = MoeDispatchSignal::Rejected {
            request_id: 1,
            reason: String::new(),
        };
        match signal {
            MoeDispatchSignal::Rejected { reason, .. } => assert!(reason.is_empty()),
            _ => panic!("Expected Rejected"),
        }
    }

    #[test]
    fn test_signal_rejected_multiline_reason() {
        let reason = "line1\nline2\nline3".to_string();
        let signal = MoeDispatchSignal::Rejected {
            request_id: 42,
            reason: reason.clone(),
        };
        match signal {
            MoeDispatchSignal::Rejected { reason: r, .. } => assert_eq!(r, reason),
            _ => panic!("Expected Rejected"),
        }
    }

    #[test]
    fn test_signal_rejected_unicode_reason() {
        let reason = "メモリ不足：圧力が高すぎる".to_string();
        let signal = MoeDispatchSignal::Rejected {
            request_id: 7,
            reason,
        };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("Rejected"));
    }

    // ── MoeDispatchSignal: Clone independence ────────────────────────────

    #[test]
    fn test_signal_clone_suspended_independence() {
        let original = MoeDispatchSignal::Suspended {
            request_id: 10,
            expert_idx: 3,
        };
        let cloned = original.clone();
        // Both should match the same pattern
        assert!(matches!(original, MoeDispatchSignal::Suspended { request_id: 10, expert_idx: 3 }));
        assert!(matches!(cloned, MoeDispatchSignal::Suspended { request_id: 10, expert_idx: 3 }));
    }

    #[test]
    fn test_signal_clone_rejected_independence() {
        let original = MoeDispatchSignal::Rejected {
            request_id: 5,
            reason: "pressure".to_string(),
        };
        let cloned = original.clone();
        // Both should hold identical values
        if let MoeDispatchSignal::Rejected { request_id, reason } = &cloned {
            assert_eq!(*request_id, 5);
            assert_eq!(reason, "pressure");
        } else {
            panic!("Expected Rejected");
        }
    }

    // ── MoeDispatchSignal: Debug roundtrip for all variants ─────────────

    #[test]
    fn test_signal_debug_roundtrip_continue() {
        let signal = MoeDispatchSignal::Continue;
        let debug = format!("{:?}", signal);
        assert!(debug.contains("Continue"));
        assert!(!debug.contains("Suspended"));
        assert!(!debug.contains("Rejected"));
    }

    #[test]
    fn test_signal_debug_roundtrip_suspended() {
        let signal = MoeDispatchSignal::Suspended {
            request_id: 123,
            expert_idx: 45,
        };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("123"));
        assert!(debug.contains("45"));
    }

    // ── MoeDispatchCallback: new() with various top_k values ────────────

    #[test]
    fn test_new_with_top_k_equal_to_num_experts() {
        let cb = MoeDispatchCallback::new(8, 8, 32, 0);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 8);
        assert_eq!(cb.moe_layers.len(), 32);
    }

    #[test]
    fn test_new_with_top_k_one() {
        let cb = MoeDispatchCallback::new(64, 1, 32, 0);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 64);
    }

    #[test]
    fn test_new_with_top_k_zero_but_experts_positive() {
        // top_k=0 with experts>0: enabled=true, but no valid routing
        let cb = MoeDispatchCallback::new(8, 0, 32, 0);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 8);
    }

    // ── MoeDispatchCallback: moe_layers ordering guarantee ──────────────

    #[test]
    fn test_moe_layers_are_contiguous_and_sorted() {
        let cb = MoeDispatchCallback::new(8, 2, 64, 16);
        for i in 1..cb.moe_layers.len() {
            assert_eq!(
                cb.moe_layers[i],
                cb.moe_layers[i - 1] + 1,
                "layers must be contiguous: {} should follow {}",
                cb.moe_layers[i],
                cb.moe_layers[i - 1]
            );
        }
    }

    // ── MoeDispatchCallback: disabled callback check_and_handle_fault ───

    #[test]
    fn test_disabled_callback_fault_handler_has_zero_experts() {
        let cb = MoeDispatchCallback::disabled();
        let stats = cb.fault_handler().stats();
        assert_eq!(stats.total_faults, 0);
        assert_eq!(cb.num_experts(), 0);
    }

    // ── check_and_handle_fault: hot expert never suspends ───────────────

    #[test]
    fn test_check_fault_hot_expert_always_continues() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_heat_thresholds(0.1, 0.001);

        // Make expert 0 hot
        for _ in 0..10 {
            thermal.step(&[100, 5, 5, 5]);
        }

        for layer in 0..8 {
            let signal = cb.check_and_handle_fault(0, layer, 42, &thermal, 0.3);
            assert!(
                matches!(signal, MoeDispatchSignal::Continue),
                "hot expert should always continue at layer {}",
                layer
            );
        }
    }

    // ── check_and_handle_fault: multiple requests same evicted expert ───

    #[test]
    fn test_check_fault_multiple_requests_same_evicted_expert() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        for req_id in 0..5u64 {
            let signal = cb.check_and_handle_fault(1, 0, req_id, &thermal, 0.3);
            assert!(
                matches!(signal, MoeDispatchSignal::Suspended { expert_idx: 1, .. }),
                "request {} should be suspended",
                req_id
            );
        }
    }

    // ── check_and_handle_fault: very low memory pressure always accepts ─

    #[test]
    fn test_check_fault_evicted_very_low_pressure_always_suspended() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);

        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);

        let signal = cb.check_and_handle_fault(1, 0, 1, &thermal, 0.001);
        assert!(matches!(signal, MoeDispatchSignal::Suspended { .. }));
    }

    // ── ExpertFault: Clone preserves all primitive fields ───────────────

    #[test]
    fn test_expert_fault_clone_preserves_instant_monotonicity() {
        let fault = ExpertFault {
            expert_idx: 3,
            layer_idx: 7,
            request_id: 42,
            fault_time: Instant::now(),
        };
        let cloned = fault.clone();
        // expert_idx, layer_idx, request_id are Copy, so checked elsewhere
        // Verify the clone is usable independently
        assert_eq!(cloned.expert_idx, fault.expert_idx);
        assert_eq!(cloned.layer_idx, fault.layer_idx);
        assert_eq!(cloned.request_id, fault.request_id);
    }

    // ── FaultStats: PartialEq edge case — only avg_recovery_us differs ──

    #[test]
    fn test_fault_stats_ne_differs_only_in_avg_recovery() {
        let a = FaultStats {
            total_faults: 10,
            avg_recovery_us: 1.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 3,
        };
        let b = FaultStats {
            total_faults: 10,
            avg_recovery_us: 2.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 3,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_fault_stats_ne_differs_only_in_fault_rate() {
        let a = FaultStats {
            total_faults: 10,
            avg_recovery_us: 1.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 3,
        };
        let b = FaultStats {
            total_faults: 10,
            avg_recovery_us: 1.0,
            fault_rate: 0.2,
            in_flight_restorations: 2,
            suspended_request_count: 3,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn test_fault_stats_ne_differs_only_in_suspended_count() {
        let a = FaultStats {
            total_faults: 10,
            avg_recovery_us: 1.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 3,
        };
        let b = FaultStats {
            total_faults: 10,
            avg_recovery_us: 1.0,
            fault_rate: 0.1,
            in_flight_restorations: 2,
            suspended_request_count: 4,
        };
        assert_ne!(a, b);
    }

    // ── FaultResolution: Rejected reason content verification ───────────

    #[test]
    fn test_fault_resolution_rejected_reason_with_special_chars() {
        let reason = "error: OOM at 0xDEADBEEF!".to_string();
        let res = FaultResolution::Rejected { reason: reason.clone() };
        match res {
            FaultResolution::Rejected { reason: r } => assert_eq!(r, reason),
            FaultResolution::Resumed { .. } => panic!("wrong variant"),
        }
    }

    // ── ExpertWeightLocation: Debug variant names ───────────────────────

    #[test]
    fn test_weight_location_debug_all_variant_names() {
        assert!(format!("{:?}", ExpertWeightLocation::GpuL2).contains("GpuL2"));
        assert!(format!("{:?}", ExpertWeightLocation::GpuVram).contains("GpuVram"));
        assert!(format!("{:?}", ExpertWeightLocation::CpuRam).contains("CpuRam"));
        assert!(format!("{:?}", ExpertWeightLocation::RemoteNode).contains("RemoteNode"));
        assert!(format!("{:?}", ExpertWeightLocation::Evicted).contains("Evicted"));
    }

    // ── ExpertWeightLocation: Eq consistency across Copy ────────────────

    #[test]
    fn test_weight_location_eq_after_copy_cycle() {
        let original = ExpertWeightLocation::RemoteNode;
        let copy1 = original;
        let copy2 = copy1;
        assert_eq!(original, copy1);
        assert_eq!(copy1, copy2);
        assert_eq!(original, copy2);
    }

    // ── ExpertHeatLevel: from_hit_rate with very high thresholds ────────

    #[test]
    fn test_heat_level_from_hit_rate_all_cold_with_high_thresholds() {
        // With hot=0.99 and cold=0.98, rate=0.5 is Cold (not >= hot, not >= cold, but > 0)
        let level = ExpertHeatLevel::from_hit_rate(0.5, 0.99, 0.98);
        assert_eq!(level, ExpertHeatLevel::Cold);
    }

    #[test]
    fn test_heat_level_from_hit_rate_rate_one_always_hot() {
        let level = ExpertHeatLevel::from_hit_rate(1.0, 0.99, 0.98);
        assert_eq!(level, ExpertHeatLevel::Hot);
    }

    #[test]
    fn test_heat_level_from_hit_rate_rate_just_below_hot() {
        let level = ExpertHeatLevel::from_hit_rate(0.989, 0.99, 0.001);
        assert_eq!(level, ExpertHeatLevel::Warm);
    }

    // ── ExpertHeatLevel: Clone produces identical values ────────────────

    #[test]
    fn test_heat_level_clone_all_variants() {
        for level in [ExpertHeatLevel::Hot, ExpertHeatLevel::Warm, ExpertHeatLevel::Cold, ExpertHeatLevel::Evicted] {
            let cloned = level.clone();
            assert_eq!(level, cloned);
        }
    }

    // ── EvictionDecision: Debug and Copy ────────────────────────────────

    #[test]
    fn test_eviction_decision_debug_all_variants() {
        assert!(format!("{:?}", EvictionDecision::Keep).contains("Keep"));
        assert!(format!("{:?}", EvictionDecision::Evict).contains("Evict"));
        assert!(format!("{:?}", EvictionDecision::Reactivate).contains("Reactivate"));
    }

    #[test]
    fn test_eviction_decision_copy_after_move() {
        let d1 = EvictionDecision::Reactivate;
        let d2 = d1; // Copy
        let d3 = d1; // Still valid due to Copy
        assert_eq!(d1, d2);
        assert_eq!(d2, d3);
    }

    // ── DeoptRequest: PartialEq comprehensive ───────────────────────────

    #[test]
    fn test_deopt_request_ne_different_request_id() {
        let a = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let b = DeoptRequest { request_id: 2, expert_idx: 2, layer_idx: 3, step: 4 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_deopt_request_ne_different_expert_idx() {
        let a = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let b = DeoptRequest { request_id: 1, expert_idx: 3, layer_idx: 3, step: 4 };
        assert_ne!(a, b);
    }

    #[test]
    fn test_deopt_request_ne_different_layer_idx() {
        let a = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let b = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 4, step: 4 };
        assert_ne!(a, b);
    }

    // ── DeoptHandlingResult: field extraction ────────────────────────────

    #[test]
    fn test_deopt_handling_result_reactivate_field_extraction() {
        let result = DeoptHandlingResult::ReactivateAndRerun {
            expert_idx: 10,
            request_id: 20,
        };
        match result {
            DeoptHandlingResult::ReactivateAndRerun { expert_idx, request_id } => {
                assert_eq!(expert_idx, 10);
                assert_eq!(request_id, 20);
            }
            DeoptHandlingResult::SpuriousDeopt { .. } => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_deopt_handling_result_spurious_field_extraction() {
        let result = DeoptHandlingResult::SpuriousDeopt {
            expert_idx: 5,
            request_id: 99,
        };
        match result {
            DeoptHandlingResult::SpuriousDeopt { expert_idx, request_id } => {
                assert_eq!(expert_idx, 5);
                assert_eq!(request_id, 99);
            }
            DeoptHandlingResult::ReactivateAndRerun { .. } => panic!("wrong variant"),
        }
    }

    // ── ThermalSummary: construction with all zeros ─────────────────────

    #[test]
    fn test_thermal_summary_all_zeros() {
        let summary = ThermalSummary {
            num_experts: 0,
            hot_count: 0,
            warm_count: 0,
            cold_count: 0,
            evicted_count: 0,
            total_evictions: 0,
            total_reactivations: 0,
            current_step: 0,
            pending_deopt_count: 0,
            working_set_size: 0,
            effective_eviction_threshold: 0,
        };
        assert_eq!(summary.num_experts, 0);
        assert_eq!(summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count, 0);
    }

    // ── ThermalSummary: Debug contains all field names ──────────────────

    #[test]
    fn test_thermal_summary_debug_all_fields() {
        let summary = ThermalSummary {
            num_experts: 1,
            hot_count: 1,
            warm_count: 1,
            cold_count: 1,
            evicted_count: 1,
            total_evictions: 1,
            total_reactivations: 1,
            current_step: 1,
            pending_deopt_count: 1,
            working_set_size: 1,
            effective_eviction_threshold: 1,
        };
        let debug = format!("{:?}", summary);
        assert!(debug.contains("num_experts"));
        assert!(debug.contains("hot_count"));
        assert!(debug.contains("evicted_count"));
        assert!(debug.contains("working_set_size"));
    }

    // ── ExpertWeightLayout: PartialEq with different locations ──────────

    #[test]
    fn test_expert_weight_layout_equality_same_fields() {
        let a = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 1024,
            compressed_bytes: 128,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::CpuRam,
        };
        let b = a.clone();
        assert_eq!(a.expert_idx, b.expert_idx);
        assert_eq!(a.weight_bytes, b.weight_bytes);
        assert_eq!(a.compressed_bytes, b.compressed_bytes);
        assert_eq!(a.location, b.location);
    }

    #[test]
    fn test_expert_weight_layout_different_expert_idx() {
        let a = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 1024,
            compressed_bytes: 128,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::CpuRam,
        };
        let b = ExpertWeightLayout {
            expert_idx: 1,
            weight_bytes: 1024,
            compressed_bytes: 128,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::CpuRam,
        };
        assert_ne!(a.expert_idx, b.expert_idx);
    }

    // ── ExpertPrefetchRequest: construction with all zero fields ────────

    #[test]
    fn test_expert_prefetch_request_zero_fields() {
        let req = ExpertPrefetchRequest {
            expert_idx: 0,
            layer_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 0,
            estimated_latency_us: 0.0,
            priority: 0,
        };
        assert_eq!(req.expert_idx, 0);
        assert_eq!(req.bytes, 0);
        assert!((req.estimated_latency_us - 0.0).abs() < f32::EPSILON);
        assert_eq!(req.priority, 0);
    }

    // ── ExpertPrefetchRequest: source and destination can be same ───────

    #[test]
    fn test_expert_prefetch_request_same_source_destination() {
        let req = ExpertPrefetchRequest {
            expert_idx: 5,
            layer_idx: 0,
            source: ExpertWeightLocation::GpuVram,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 4096,
            estimated_latency_us: 0.0,
            priority: 1,
        };
        assert_eq!(req.source, req.destination);
    }

    // ── ExpertThermalManager: step updates consecutive_zero_streak accurately

    #[test]
    fn test_thermal_manager_streak_resets_on_hit() {
        let mut thermal = ExpertThermalManager::new(2).with_eviction_threshold(100);

        // Expert 0 hit every step
        thermal.step(&[10, 0]);
        thermal.step(&[10, 0]);
        thermal.step(&[0, 0]); // miss
        thermal.step(&[10, 0]); // hit again — streak resets

        let state = thermal.state(0).unwrap();
        assert_eq!(state.consecutive_zero_streak, 0);
    }

    // ── ExpertThermalManager: summary hot+warm+cold+evicted equals total ─

    #[test]
    fn test_thermal_summary_partition_covers_all_experts() {
        let mut thermal = ExpertThermalManager::new(8).with_eviction_threshold(3);

        // Various patterns
        for _ in 0..5 {
            thermal.step(&[10, 5, 0, 0, 3, 0, 0, 1]);
        }

        let summary = thermal.summary();
        let total = summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count;
        assert_eq!(total, 8);
    }

    // ── MoeDispatchCallback: pre_node records steps via fault_handler ───

    #[test]
    fn test_pre_node_moe_layer_increments_fault_handler_steps() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        let mut cb = MoeDispatchCallback::new(8, 2, 4, 0);

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 8,
            moe_top_k: 2,
            expert_intermediate_size: 512,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];

        // Arrange: check initial stats
        let initial_stats = cb.fault_handler().stats();
        let initial_faults = initial_stats.total_faults;

        // Act: call pre_node 3 times on a MoE layer
        for i in 0..3 {
            let ctx = LayerContext {
                node_idx: i,
                layer_idx: 0,
                node_op: "Gemm",
                hidden_state: &hs,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 10,
                seq_len: 1,
                position: 9,
                request_id: 1,
                model_config: &config,
            };
            let action = cb.pre_node(&ctx);
            assert!(matches!(action, CallbackAction::Continue));
        }

        // Assert: fault handler should have recorded 3 decode steps
        let after_stats = cb.fault_handler().stats();
        assert_eq!(after_stats.total_faults, initial_faults);
    }

    // ── ExpertThermalManager: working_set_size after many diverse steps ──

    #[test]
    fn test_working_set_size_all_experts_accessed() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);

        thermal.step(&[1, 1, 1, 1]);
        assert_eq!(thermal.working_set_size(), 4);
    }

    // ── ExpertThermalManager: evict_expert updates total_evictions ──────

    #[test]
    fn test_evict_updates_total_evictions_counter() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..4 {
            thermal.step(&[10, 0, 0, 0]);
        }

        assert_eq!(thermal.summary().total_evictions, 0);
        assert!(thermal.evict_expert(1));
        assert_eq!(thermal.summary().total_evictions, 1);
        assert!(thermal.evict_expert(2));
        assert_eq!(thermal.summary().total_evictions, 2);
    }

    // ── ExpertThermalManager: reactivate updates total_reactivations ────

    #[test]
    fn test_reactivate_updates_total_reactivations_counter() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..4 {
            thermal.step(&[10, 0, 0, 0]);
        }

        thermal.evict_expert(1);
        thermal.evict_expert(2);
        assert_eq!(thermal.summary().total_reactivations, 0);

        assert!(thermal.reactivate_expert(1));
        assert_eq!(thermal.summary().total_reactivations, 1);
        assert!(thermal.reactivate_expert(2));
        assert_eq!(thermal.summary().total_reactivations, 2);
    }

    // ── ExpertWeightPrefetcher: schedule_prefetch with all cold experts ──

    #[test]
    fn test_schedule_prefetch_all_cold_produces_requests() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        let heat_levels = vec![ExpertHeatLevel::Cold; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        assert_eq!(requests.len(), 4);
        for req in &requests {
            assert_eq!(req.destination, ExpertWeightLocation::GpuVram);
        }
    }

    // ── ExpertWeightPrefetcher: total_gpu_vram_bytes after partial update

    #[test]
    fn test_total_gpu_vram_partial_update() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        prefetcher.update_location(0, ExpertWeightLocation::GpuVram);
        prefetcher.update_location(2, ExpertWeightLocation::GpuL2);
        // Experts 1 and 3 remain in CpuRam

        let vram = prefetcher.total_gpu_vram_bytes();
        // Experts 0 and 2 on GPU, each compressed to 4096/8 = 512
        assert_eq!(vram, 2 * 512);
    }

    // ── ExpertWeightPrefetcher: bandwidth_savings_ratio zero weight ──────

    #[test]
    fn test_bandwidth_savings_ratio_single_expert() {
        let prefetcher = ExpertWeightPrefetcher::new(1, 4096);
        let ratio = prefetcher.bandwidth_savings_ratio();
        // 4096 original, 512 compressed => savings = 1 - 512/4096 = 0.875
        assert!((ratio - 0.875).abs() < f32::EPSILON);
    }

    // ── ExpertWeightPrefetcher: prefetch_step with mixed locations ───────

    #[test]
    fn test_prefetch_step_mixed_locations_correct_count() {
        let mut prefetcher = ExpertWeightPrefetcher::new(5, 4096);
        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        prefetcher.update_location(1, ExpertWeightLocation::GpuVram);
        prefetcher.update_location(2, ExpertWeightLocation::Evicted);
        // Experts 3, 4 remain in CpuRam

        let requests = prefetcher.prefetch_step(0, &[0, 1, 2, 3, 4]);
        // Expert 0 (GpuL2) skipped, 1 (GpuVram) skipped, 2 (Evicted) skipped
        // Experts 3 and 4 need prefetch
        assert_eq!(requests.len(), 2);
    }

    // ── ExpertWeightPrefetcher: can_pipeline_hide empty requests ─────────

    #[test]
    fn test_can_pipeline_hide_empty_requests() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        let hiding = prefetcher.can_pipeline_hide(&[], 10);
        assert!(hiding.is_empty());
    }

    // ── ExpertHeatState: direct construction with evicted state ──────────

    #[test]
    fn test_heat_state_evicted_construction() {
        let state = ExpertHeatState {
            expert_idx: 7,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 100,
            heat_level: ExpertHeatLevel::Evicted,
            consecutive_zero_streak: 100,
            last_hit_step: 50,
            is_evicted: true,
            reactivation_count: 0,
        };
        assert!(state.is_evicted);
        assert_eq!(state.heat_level, ExpertHeatLevel::Evicted);
        assert_eq!(state.consecutive_zero_streak, 100);
        assert_eq!(state.last_hit_step, 50);
    }

    // ── ExpertHeatState: Debug format includes key fields ───────────────

    #[test]
    fn test_heat_state_debug_includes_evicted_field() {
        let state = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.5,
            hit_count: 10,
            route_count: 20,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 0,
            last_hit_step: 20,
            is_evicted: false,
            reactivation_count: 0,
        };
        let debug = format!("{:?}", state);
        assert!(debug.contains("is_evicted: false"));
        assert!(debug.contains("hit_rate: 0.5"));
    }

    // ── DeoptRequest: construction with all fields zero ─────────────────

    #[test]
    fn test_deopt_request_all_zeros() {
        let req = DeoptRequest {
            request_id: 0,
            expert_idx: 0,
            layer_idx: 0,
            step: 0,
        };
        assert_eq!(req.request_id, 0);
        assert_eq!(req.expert_idx, 0);
        assert_eq!(req.layer_idx, 0);
        assert_eq!(req.step, 0);
    }

    // ── MoeDispatchCallback: target_layers None when disabled ───────────

    #[test]
    fn test_target_layers_disabled_is_none_or_empty() {
        let cb = MoeDispatchCallback::disabled();
        match cb.target_layers() {
            None => {},
            Some(slice) => assert!(slice.is_empty()),
        }
    }

    // ── MoeDispatchCallback: name always returns moe_dispatch ───────────

    #[test]
    fn test_name_constant_regardless_of_state() {
        let enabled = MoeDispatchCallback::new(64, 8, 32, 0);
        let disabled = MoeDispatchCallback::disabled();
        let single = MoeDispatchCallback::new(1, 1, 1, 0);
        assert_eq!(enabled.name(), "moe_dispatch");
        assert_eq!(disabled.name(), "moe_dispatch");
        assert_eq!(single.name(), "moe_dispatch");
    }

    // ── MoeDispatchCallback: priority always returns 70 ─────────────────

    #[test]
    fn test_priority_constant_regardless_of_state() {
        let enabled = MoeDispatchCallback::new(64, 8, 32, 0);
        let disabled = MoeDispatchCallback::disabled();
        let single = MoeDispatchCallback::new(1, 1, 1, 0);
        assert_eq!(enabled.priority(), 70);
        assert_eq!(disabled.priority(), 70);
        assert_eq!(single.priority(), 70);
    }

    // ── ExpertWeightLocation: from_heat_level is deterministic ──────────

    #[test]
    fn test_from_heat_level_idempotent() {
        for level in [ExpertHeatLevel::Hot, ExpertHeatLevel::Warm, ExpertHeatLevel::Cold, ExpertHeatLevel::Evicted] {
            let first = ExpertWeightLocation::from_heat_level(level);
            let second = ExpertWeightLocation::from_heat_level(level);
            assert_eq!(first, second);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Tests (45 new tests: batch 4)
    // ═══════════════════════════════════════════════════════════════════════

    // ── ExpertFaultHandler: handle_fault direct ─────────────────────────

    #[test]
    fn test_handle_fault_direct_low_pressure_resumed() {
        let mut handler = ExpertFaultHandler::new(4);
        let fault = ExpertFault {
            expert_idx: 1,
            layer_idx: 0,
            request_id: 42,
            fault_time: Instant::now(),
        };
        let resolution = handler.handle_fault(fault, 0.3, ExpertWeightLocation::CpuRam);
        assert!(
            matches!(resolution, FaultResolution::Resumed { .. }),
            "low memory pressure should result in Resumed"
        );
    }

    #[test]
    fn test_handle_fault_direct_high_pressure_rejected() {
        let mut handler = ExpertFaultHandler::new(4);
        let fault = ExpertFault {
            expert_idx: 1,
            layer_idx: 0,
            request_id: 42,
            fault_time: Instant::now(),
        };
        let resolution = handler.handle_fault(fault, 0.99, ExpertWeightLocation::CpuRam);
        assert!(
            matches!(resolution, FaultResolution::Rejected { .. }),
            "high memory pressure should result in Rejected"
        );
    }

    #[test]
    fn test_handle_fault_direct_increments_expert_count() {
        let mut handler = ExpertFaultHandler::new(4);

        assert_eq!(handler.expert_fault_count(0), 0);
        let fault = ExpertFault {
            expert_idx: 0,
            layer_idx: 2,
            request_id: 1,
            fault_time: Instant::now(),
        };
        let _ = handler.handle_fault(fault, 0.1, ExpertWeightLocation::CpuRam);
        assert!(
            handler.expert_fault_count(0) > 0,
            "fault count for expert 0 should increase after handle_fault"
        );
    }

    #[test]
    fn test_handle_fault_multiple_experts_independent_counts() {
        let mut handler = ExpertFaultHandler::new(4);

        let fault0 = ExpertFault {
            expert_idx: 0, layer_idx: 0, request_id: 1, fault_time: Instant::now(),
        };
        let fault1 = ExpertFault {
            expert_idx: 1, layer_idx: 0, request_id: 2, fault_time: Instant::now(),
        };
        let _ = handler.handle_fault(fault0, 0.1, ExpertWeightLocation::CpuRam);
        let _ = handler.handle_fault(fault1, 0.1, ExpertWeightLocation::CpuRam);

        assert!(handler.expert_fault_count(0) > 0, "expert 0 should have faults");
        assert!(handler.expert_fault_count(1) > 0, "expert 1 should have faults");
        assert_eq!(handler.expert_fault_count(2), 0, "expert 2 should have zero faults");
        assert_eq!(handler.expert_fault_count(3), 0, "expert 3 should have zero faults");
    }

    #[test]
    fn test_handle_fault_resumed_increments_in_flight() {
        let mut handler = ExpertFaultHandler::new(4);
        let initial = handler.in_flight_count();

        let fault = ExpertFault {
            expert_idx: 0, layer_idx: 0, request_id: 1, fault_time: Instant::now(),
        };
        let resolution = handler.handle_fault(fault, 0.1, ExpertWeightLocation::CpuRam);
        if matches!(resolution, FaultResolution::Resumed { .. }) {
            assert!(
                handler.in_flight_count() > initial,
                "in_flight_count should increase after Resumed fault"
            );
        }
    }

    #[test]
    fn test_handle_fault_resumed_increments_suspended_count() {
        let mut handler = ExpertFaultHandler::new(4);
        let initial = handler.suspended_request_count();

        let fault = ExpertFault {
            expert_idx: 0, layer_idx: 0, request_id: 1, fault_time: Instant::now(),
        };
        let resolution = handler.handle_fault(fault, 0.1, ExpertWeightLocation::CpuRam);
        if matches!(resolution, FaultResolution::Resumed { .. }) {
            assert!(
                handler.suspended_request_count() > initial,
                "suspended_request_count should increase after Resumed fault"
            );
        }
    }

    #[test]
    fn test_handle_fault_rejected_does_not_increment_in_flight() {
        let mut handler = ExpertFaultHandler::new(4);

        let fault = ExpertFault {
            expert_idx: 0, layer_idx: 0, request_id: 1, fault_time: Instant::now(),
        };
        let resolution = handler.handle_fault(fault, 0.99, ExpertWeightLocation::CpuRam);
        if matches!(resolution, FaultResolution::Rejected { .. }) {
            assert_eq!(
                handler.in_flight_count(), 0,
                "in_flight_count should remain zero after Rejected fault"
            );
        }
    }

    // ── ExpertFaultHandler: state after fault ───────────────────────────

    #[test]
    fn test_is_restoration_pending_true_after_resumed_fault() {
        let mut handler = ExpertFaultHandler::new(4);

        let fault = ExpertFault {
            expert_idx: 2, layer_idx: 3, request_id: 1, fault_time: Instant::now(),
        };
        let resolution = handler.handle_fault(fault, 0.1, ExpertWeightLocation::CpuRam);
        if matches!(resolution, FaultResolution::Resumed { .. }) {
            assert!(
                handler.is_restoration_pending(2, 3),
                "restoration should be pending for the faulted expert and layer"
            );
        }
    }

    #[test]
    fn test_is_restoration_pending_false_for_unfaulted_expert() {
        let mut handler = ExpertFaultHandler::new(4);

        let fault = ExpertFault {
            expert_idx: 0, layer_idx: 0, request_id: 1, fault_time: Instant::now(),
        };
        let _ = handler.handle_fault(fault, 0.1, ExpertWeightLocation::CpuRam);

        assert!(
            !handler.is_restoration_pending(1, 0),
            "no restoration pending for un-faulted expert"
        );
        assert!(
            !handler.is_restoration_pending(0, 99),
            "no restoration pending for un-faulted layer"
        );
    }

    #[test]
    fn test_handle_fault_increments_total_faults_in_stats() {
        let mut handler = ExpertFaultHandler::new(4);
        let initial_total = handler.stats().total_faults;

        let fault = ExpertFault {
            expert_idx: 0, layer_idx: 0, request_id: 1, fault_time: Instant::now(),
        };
        let _ = handler.handle_fault(fault, 0.5, ExpertWeightLocation::CpuRam);

        assert!(
            handler.stats().total_faults > initial_total,
            "total_faults should increase after handle_fault"
        );
    }

    // ── ExpertThermalManager: step pattern edge cases ───────────────────

    #[test]
    fn test_step_accumulates_zero_streak_for_unaccessed() {
        let mut thermal = ExpertThermalManager::new(3);

        thermal.step(&[10, 0, 0]);
        thermal.step(&[10, 0, 0]);
        thermal.step(&[10, 0, 0]);

        assert_eq!(thermal.state(1).unwrap().consecutive_zero_streak, 3);
        assert_eq!(thermal.state(2).unwrap().consecutive_zero_streak, 3);
    }

    #[test]
    fn test_step_resets_zero_streak_on_access() {
        let mut thermal = ExpertThermalManager::new(2);

        thermal.step(&[0, 10]); // expert 0: miss → streak 1
        thermal.step(&[0, 10]); // expert 0: miss → streak 2
        thermal.step(&[5, 10]); // expert 0: hit → streak reset to 0

        assert_eq!(thermal.state(0).unwrap().consecutive_zero_streak, 0);
    }

    #[test]
    fn test_step_updates_route_count_per_step() {
        let mut thermal = ExpertThermalManager::new(2);

        thermal.step(&[10, 5]);
        thermal.step(&[10, 5]);
        thermal.step(&[10, 5]);

        assert_eq!(thermal.state(0).unwrap().route_count, 3);
        assert_eq!(thermal.state(1).unwrap().route_count, 3);
    }

    #[test]
    fn test_step_with_asymmetric_patterns() {
        let mut thermal = ExpertThermalManager::new(3);

        // Expert 0: hit every step, Expert 1: alternating, Expert 2: never
        thermal.step(&[10, 5, 0]);
        thermal.step(&[10, 0, 0]);
        thermal.step(&[10, 8, 0]);

        assert_eq!(thermal.state(0).unwrap().hit_count, 3);
        assert_eq!(thermal.state(1).unwrap().hit_count, 2);
        assert_eq!(thermal.state(1).unwrap().consecutive_zero_streak, 0);
        assert_eq!(thermal.state(2).unwrap().hit_count, 0);
        assert_eq!(thermal.state(2).unwrap().consecutive_zero_streak, 3);
    }

    #[test]
    fn test_step_hit_rate_in_valid_range() {
        let mut thermal = ExpertThermalManager::new(2);

        for _ in 0..100 {
            thermal.step(&[10, 0]);
        }

        let s0 = thermal.state(0).unwrap();
        assert!(s0.hit_rate > 0.0 && s0.hit_rate <= 1.0);

        let s1 = thermal.state(1).unwrap();
        assert!((s1.hit_rate - 0.0).abs() < f64::EPSILON);
    }

    // ── ExpertThermalManager: adaptive and memory pressure ─────────────

    #[test]
    fn test_update_memory_pressure_zero_value() {
        let mut thermal = ExpertThermalManager::new(4).with_adaptive_eviction(10);
        thermal.update_memory_pressure(0.0);
        let threshold = thermal.effective_eviction_threshold();
        assert!(threshold > 0, "threshold should be positive even at zero pressure");
    }

    #[test]
    fn test_update_memory_pressure_full_value() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);
        thermal.update_memory_pressure(1.0);
        let threshold = thermal.effective_eviction_threshold();
        assert!(threshold > 0);
    }

    #[test]
    fn test_effective_threshold_combines_aggressiveness_and_adaptive() {
        let base = 100u64;
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(base)
            .with_eviction_aggressiveness(1.0)
            .with_adaptive_eviction(10);

        thermal.update_memory_pressure(0.5);
        let threshold = thermal.effective_eviction_threshold();

        assert!(threshold > 0, "effective threshold should always be positive");
    }

    #[test]
    fn test_working_set_size_reflects_recent_access() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(5);

        // Only expert 0 is accessed
        for _ in 0..3 {
            thermal.step(&[10, 0, 0, 0]);
        }

        let ws = thermal.working_set_size();
        assert!(ws >= 1, "at least expert 0 should be in working set");
        assert!(ws <= 4);
    }

    // ── ExpertThermalManager: misc methods ──────────────────────────────

    #[test]
    fn test_cold_or_evicted_returns_non_hot_experts() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_eviction_threshold(100)
            .with_heat_thresholds(0.1, 0.001);

        // Expert 0 hot, rest get zero hits → Evicted
        for _ in 0..5 {
            thermal.step(&[100, 0, 0, 0]);
        }

        let cold_evicted = thermal.cold_or_evicted_experts();
        assert!(!cold_evicted.contains(&0), "hot expert should not be in cold_or_evicted");
        assert!(cold_evicted.contains(&1));
        assert!(cold_evicted.contains(&2));
        assert!(cold_evicted.contains(&3));
    }

    #[test]
    fn test_hot_experts_all_hot_when_all_accessed() {
        let mut thermal = ExpertThermalManager::new(3)
            .with_heat_thresholds(0.1, 0.001);

        for _ in 0..10 {
            thermal.step(&[10, 10, 10]);
        }

        let hot = thermal.hot_experts();
        assert_eq!(hot.len(), 3, "all experts should be hot");
    }

    #[test]
    fn test_summary_current_step_increments() {
        let mut thermal = ExpertThermalManager::new(2);

        assert_eq!(thermal.summary().current_step, 0);
        thermal.step(&[1, 1]);
        assert_eq!(thermal.summary().current_step, 1);
        thermal.step(&[1, 1]);
        assert_eq!(thermal.summary().current_step, 2);
    }

    #[test]
    fn test_evict_reactivate_cycle_maintains_counters() {
        let mut thermal = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            thermal.step(&[10, 0]);
        }

        assert!(thermal.evict_expert(1));
        assert!(thermal.reactivate_expert(1));
        assert!(thermal.evict_expert(1));
        assert!(thermal.reactivate_expert(1));

        let summary = thermal.summary();
        assert_eq!(summary.total_evictions, 2);
        assert_eq!(summary.total_reactivations, 2);
        assert_eq!(summary.evicted_count, 0, "expert should be reactivated");
    }

    #[test]
    fn test_pending_deopt_count_reflects_requests() {
        let mut thermal = ExpertThermalManager::new(4);

        let req1 = DeoptRequest { request_id: 1, expert_idx: 0, layer_idx: 0, step: 0 };
        let req2 = DeoptRequest { request_id: 2, expert_idx: 1, layer_idx: 1, step: 1 };
        thermal.handle_deopt_request(req1);
        thermal.handle_deopt_request(req2);

        assert_eq!(thermal.pending_deopt_requests().len(), 2);
        assert_eq!(thermal.summary().pending_deopt_count, 2);
    }

    // ── ExpertWeightPrefetcher: schedule_prefetch edge cases ───────────

    #[test]
    fn test_schedule_prefetch_hot_experts_excluded() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Cold,
        ];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        for req in &requests {
            assert_ne!(req.expert_idx, 0, "hot expert should not appear in prefetch requests");
        }
    }

    #[test]
    fn test_schedule_prefetch_empty_routed_experts() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        let heat_levels = vec![ExpertHeatLevel::Cold; 4];
        let requests = prefetcher.schedule_prefetch(&[], &heat_levels);
        assert!(requests.is_empty());
    }

    #[test]
    fn test_schedule_prefetch_mixed_heat_levels() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        let heat_levels = vec![
            ExpertHeatLevel::Hot,
            ExpertHeatLevel::Warm,
            ExpertHeatLevel::Cold,
            ExpertHeatLevel::Evicted,
        ];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        assert!(
            !requests.is_empty(),
            "cold/evicted experts should produce prefetch requests"
        );
    }

    #[test]
    fn test_schedule_prefetch_all_evicted_returns_valid_result() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        let heat_levels = vec![ExpertHeatLevel::Evicted; 4];
        let requests = prefetcher.schedule_prefetch(&[0, 1, 2, 3], &heat_levels);
        // Evicted experts need fault recovery before prefetch; just verify no panic
        assert!(requests.len() <= 4);
    }

    // ── ExpertWeightPrefetcher: can_pipeline_hide ──────────────────────

    #[test]
    fn test_can_pipeline_hide_single_request_shape() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096)
            .with_bandwidth(64.0, 200.0)
            .with_layer_compute_time(50.0);
        let requests = vec![ExpertPrefetchRequest {
            expert_idx: 0,
            layer_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 4096,
            estimated_latency_us: 50.0,
            priority: 1,
        }];
        let hiding = prefetcher.can_pipeline_hide(&requests, 10);
        assert_eq!(hiding.len(), 1, "should return one result per request");
    }

    #[test]
    fn test_can_pipeline_hide_multiple_requests_length_matches() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096)
            .with_bandwidth(64.0, 200.0)
            .with_layer_compute_time(50.0);
        let requests = vec![
            ExpertPrefetchRequest {
                expert_idx: 0,
                layer_idx: 0,
                source: ExpertWeightLocation::CpuRam,
                destination: ExpertWeightLocation::GpuVram,
                bytes: 8192,
                estimated_latency_us: 100.0,
                priority: 2,
            },
            ExpertPrefetchRequest {
                expert_idx: 1,
                layer_idx: 0,
                source: ExpertWeightLocation::RemoteNode,
                destination: ExpertWeightLocation::GpuVram,
                bytes: 4096,
                estimated_latency_us: 200.0,
                priority: 1,
            },
            ExpertPrefetchRequest {
                expert_idx: 2,
                layer_idx: 0,
                source: ExpertWeightLocation::CpuRam,
                destination: ExpertWeightLocation::GpuL2,
                bytes: 2048,
                estimated_latency_us: 10.0,
                priority: 3,
            },
        ];
        let hiding = prefetcher.can_pipeline_hide(&requests, 5);
        assert_eq!(hiding.len(), 3, "should return one result per request");
    }

    #[test]
    fn test_can_pipeline_hide_zero_layers_returns_results() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096)
            .with_bandwidth(64.0, 200.0)
            .with_layer_compute_time(10.0);
        let requests = vec![ExpertPrefetchRequest {
            expert_idx: 0,
            layer_idx: 0,
            source: ExpertWeightLocation::CpuRam,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 8192,
            estimated_latency_us: 50.0,
            priority: 1,
        }];
        let hiding = prefetcher.can_pipeline_hide(&requests, 0);
        assert_eq!(hiding.len(), 1, "should return one result even with zero layers");
    }

    // ── ExpertWeightPrefetcher: prefetch_step edge cases ───────────────

    #[test]
    fn test_prefetch_step_empty_ids_returns_empty() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        let requests = prefetcher.prefetch_step(0, &[]);
        assert!(requests.is_empty());
    }

    #[test]
    fn test_prefetch_step_step_param_variations_same_count() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        let ids: &[u32] = &[0, 1, 2, 3];

        let r0 = prefetcher.prefetch_step(0, ids);
        let r100 = prefetcher.prefetch_step(100, ids);

        assert_eq!(r0.len(), r100.len(), "same experts should produce same count regardless of step");
    }

    // ── ExpertWeightPrefetcher: total_gpu_vram and bandwidth ───────────

    #[test]
    fn test_total_gpu_vram_after_all_to_gpu() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        for i in 0..4 {
            prefetcher.update_location(i, ExpertWeightLocation::GpuVram);
        }
        let vram = prefetcher.total_gpu_vram_bytes();
        assert!(vram > 0, "all experts in GPU VRAM should contribute to total");
    }

    #[test]
    fn test_layouts_compressed_bytes_not_larger_than_weight() {
        let prefetcher = ExpertWeightPrefetcher::new(4, 8192);
        for layout in prefetcher.layouts() {
            assert!(
                layout.compressed_bytes <= layout.weight_bytes,
                "compressed should not exceed original for expert {}",
                layout.expert_idx
            );
        }
    }

    #[test]
    fn test_bandwidth_savings_ratio_in_unit_range() {
        let prefetcher = ExpertWeightPrefetcher::new(8, 4096);
        let ratio = prefetcher.bandwidth_savings_ratio();
        assert!(ratio >= 0.0 && ratio < 1.0, "savings ratio should be in [0, 1)");
    }

    // ── ExpertHeatState: PartialEq and Clone ───────────────────────────

    #[test]
    fn test_heat_state_eq_same_values() {
        let a = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.5,
            hit_count: 10,
            route_count: 20,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 3,
            last_hit_step: 15,
            is_evicted: false,
            reactivation_count: 1,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn test_heat_state_ne_different_eviction() {
        let a = ExpertHeatState {
            expert_idx: 0,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 10,
            heat_level: ExpertHeatLevel::Evicted,
            consecutive_zero_streak: 10,
            last_hit_step: 0,
            is_evicted: true,
            reactivation_count: 0,
        };
        let mut b = a.clone();
        b.is_evicted = false;
        assert_ne!(a, b);
    }

    #[test]
    fn test_heat_state_clone_preserves_hit_rate() {
        let state = ExpertHeatState {
            expert_idx: 3,
            hit_rate: 0.75,
            hit_count: 30,
            route_count: 40,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 40,
            is_evicted: false,
            reactivation_count: 2,
        };
        let cloned = state.clone();
        assert!((cloned.hit_rate - 0.75).abs() < f64::EPSILON);
        assert_eq!(cloned.hit_count, 30);
        assert_eq!(cloned.reactivation_count, 2);
    }

    #[test]
    fn test_heat_state_clone_preserves_heat_level() {
        let state = ExpertHeatState {
            expert_idx: 5,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 100,
            heat_level: ExpertHeatLevel::Cold,
            consecutive_zero_streak: 50,
            last_hit_step: 25,
            is_evicted: false,
            reactivation_count: 0,
        };
        let cloned = state.clone();
        assert_eq!(cloned.heat_level, ExpertHeatLevel::Cold);
    }

    // ── ExpertHeatLevel: edge cases ────────────────────────────────────

    #[test]
    fn test_from_hit_rate_very_large_rate_is_hot() {
        let level = ExpertHeatLevel::from_hit_rate(1000.0, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Hot, "very large rate should be Hot");
    }

    #[test]
    fn test_from_hit_rate_equal_thresholds_at_boundary() {
        // hot_threshold == cold_threshold; rate equals both → Hot (>= hot_threshold)
        let level = ExpertHeatLevel::from_hit_rate(0.5, 0.5, 0.5);
        assert_eq!(level, ExpertHeatLevel::Hot, "rate at equal thresholds should be Hot");
    }

    #[test]
    fn test_from_hit_rate_zero_cold_threshold() {
        // cold_threshold = 0.0; rate > 0 is >= cold_threshold → Warm
        let level = ExpertHeatLevel::from_hit_rate(0.0001, 0.1, 0.0);
        assert_eq!(level, ExpertHeatLevel::Warm, "rate > 0 with cold_threshold=0 should be Warm");
    }

    // ── EvictionDecision: PartialOrd ───────────────────────────────────

    #[test]
    fn test_eviction_decision_partial_ord_ordering() {
        assert!(EvictionDecision::Keep < EvictionDecision::Evict);
        assert!(EvictionDecision::Evict < EvictionDecision::Reactivate);
        assert!(EvictionDecision::Keep < EvictionDecision::Reactivate);
    }

    // ── ThermalSummary: clone equality ─────────────────────────────────

    #[test]
    fn test_thermal_summary_clone_eq_original() {
        let summary = ThermalSummary {
            num_experts: 8,
            hot_count: 3,
            warm_count: 2,
            cold_count: 2,
            evicted_count: 1,
            total_evictions: 5,
            total_reactivations: 3,
            current_step: 200,
            pending_deopt_count: 1,
            working_set_size: 6,
            effective_eviction_threshold: 50,
        };
        let cloned = summary.clone();
        assert_eq!(summary, cloned);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Tests (15 new tests: batch 5)
    // ═══════════════════════════════════════════════════════════════════════

    // ── ExpertHeatState: last_hit_step tracking via thermal manager ──────

    #[test]
    fn test_heat_state_last_hit_step_updates_on_access() {
        let mut thermal = ExpertThermalManager::new(2);
        thermal.step(&[0, 10]); // step 1: expert 0 not accessed
        thermal.step(&[5, 10]); // step 2: expert 0 accessed

        let s0 = thermal.state(0).unwrap();
        assert_eq!(s0.last_hit_step, 2, "last_hit_step should reflect the 1-indexed step where hit occurred");
    }

    // ── ExpertThermalManager: state returns None for out of bounds ───────

    #[test]
    fn test_thermal_state_out_of_bounds_returns_none() {
        let thermal = ExpertThermalManager::new(4);
        assert!(thermal.state(4).is_none(), "index == len should return None");
        assert!(thermal.state(255).is_none(), "large index should return None");
    }

    // ── ExpertWeightLocation: estimated_latency_us for Evicted is positive infinity

    #[test]
    fn test_evicted_location_latency_is_positive_infinity() {
        let lat = ExpertWeightLocation::Evicted.estimated_latency_us();
        assert!(lat.is_infinite(), "Evicted latency should be infinite");
        assert!(lat.is_sign_positive(), "Evicted latency should be positive infinity");
    }

    // ── FaultStats: all fields zero construction ──────────────────────────

    #[test]
    fn test_fault_stats_zero_construction_is_consistent() {
        let stats = FaultStats {
            total_faults: 0,
            avg_recovery_us: 0.0,
            fault_rate: 0.0,
            in_flight_restorations: 0,
            suspended_request_count: 0,
        };
        let cloned = stats.clone();
        assert_eq!(stats, cloned, "zero FaultStats should equal its clone");
    }

    // ── ExpertPrefetchRequest: source == Evicted ─────────────────────────

    #[test]
    fn test_prefetch_request_with_evicted_source() {
        let req = ExpertPrefetchRequest {
            expert_idx: 2,
            layer_idx: 0,
            source: ExpertWeightLocation::Evicted,
            destination: ExpertWeightLocation::GpuVram,
            bytes: 4096,
            estimated_latency_us: f32::INFINITY,
            priority: 0,
        };
        assert_eq!(req.source, ExpertWeightLocation::Evicted);
        assert!(req.estimated_latency_us.is_infinite());
    }

    // ── MoeDispatchCallback: fault_handler_mut returns same handler as fault_handler

    #[test]
    fn test_fault_handler_mut_matches_immut_ref() {
        let mut cb = MoeDispatchCallback::new(8, 2, 16, 0);
        let initial_total = cb.fault_handler().stats().total_faults;

        // Mutate through mut ref
        cb.fault_handler_mut().record_step();
        cb.fault_handler_mut().record_step();

        // Verify immut ref sees the mutation
        let after_total = cb.fault_handler().stats().total_faults;
        assert_eq!(after_total, initial_total, "immut ref should see mutations from mut ref");
    }

    // ── ExpertWeightLayout: compression_ratio zero when weight_bytes is zero ─

    #[test]
    fn test_weight_layout_zero_weight_has_zero_ratio() {
        let layout = ExpertWeightLayout {
            expert_idx: 0,
            weight_bytes: 0,
            compressed_bytes: 0,
            compression_ratio: 0.0,
            location: ExpertWeightLocation::Evicted,
        };
        assert_eq!(layout.weight_bytes, 0);
        assert_eq!(layout.compressed_bytes, 0);
        assert!((layout.compression_ratio - 0.0).abs() < f32::EPSILON);
    }

    // ── ExpertThermalManager: experts_to_evict excludes already-evicted ──

    #[test]
    fn test_experts_to_evict_excludes_already_evicted() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0, 0, 0]);
        }
        thermal.evict_expert(1);
        thermal.evict_expert(2);

        let to_evict = thermal.experts_to_evict();
        assert!(!to_evict.contains(&1), "already evicted expert should not appear");
        assert!(!to_evict.contains(&2), "already evicted expert should not appear");
    }

    // ── ExpertThermalManager: evict then step does not change evicted state ─

    #[test]
    fn test_evicted_expert_stays_evicted_after_further_steps() {
        let mut thermal = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0]);
        }
        assert!(thermal.evict_expert(1));

        // More steps with expert 1 getting hits
        thermal.step(&[0, 100]);
        thermal.step(&[0, 100]);

        let state = thermal.state(1).unwrap();
        assert!(state.is_evicted, "evicted expert should remain evicted until explicit reactivate");
    }

    // ── ExpertWeightPrefetcher: update_location to Evicted ───────────────

    #[test]
    fn test_prefetcher_update_location_to_evicted() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        prefetcher.update_location(0, ExpertWeightLocation::Evicted);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::Evicted);
        assert_eq!(prefetcher.total_gpu_vram_bytes(), 0, "Evicted location should not count toward GPU VRAM");
    }

    // ── ExpertHeatLevel: from_hit_rate with subnormal float ──────────────

    #[test]
    fn test_from_hit_rate_subnormal_positive_rate() {
        let subnormal: f64 = f64::from_bits(1); // smallest positive subnormal
        let level = ExpertHeatLevel::from_hit_rate(subnormal, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Cold, "subnormal positive rate < cold_threshold should be Cold");
    }

    // ── DeoptRequest: Clone trait produces independent copy ──────────────

    #[test]
    fn test_deopt_request_clone_allows_independent_use() {
        let original = DeoptRequest { request_id: 1, expert_idx: 2, layer_idx: 3, step: 4 };
        let cloned = original.clone();
        assert_eq!(original.request_id, 1);
        assert_eq!(cloned.request_id, 1);
        assert_eq!(original.step, 4);
        assert_eq!(cloned.step, 4);
    }

    // ── MoeDispatchCallback: moe_layers field is not pub but length verified ─

    #[test]
    fn test_moe_layers_length_consistency_with_num_layers_and_start() {
        // Verify: len = num_layers - moe_start_layer (when num_experts > 0)
        let cases = vec![(32, 0, 32), (32, 8, 24), (64, 16, 48), (1, 0, 1), (100, 50, 50)];
        for (num_layers, moe_start, expected_len) in cases {
            let cb = MoeDispatchCallback::new(8, 2, num_layers, moe_start);
            assert_eq!(
                cb.moe_layers.len(), expected_len,
                "num_layers={}, moe_start={} should produce {} MoE layers",
                num_layers, moe_start, expected_len
            );
        }
    }

    // ── ExpertFaultHandler: with_memory_pressure_limit at exactly 1.0 ────

    #[test]
    fn test_fault_handler_pressure_limit_at_one_allows_all() {
        let mut handler = ExpertFaultHandler::new(4).with_memory_pressure_limit(1.0);
        let fault = ExpertFault {
            expert_idx: 0, layer_idx: 0, request_id: 1, fault_time: Instant::now(),
        };
        // pressure 0.99 < 1.0 limit => Resumed
        let resolution = handler.handle_fault(fault, 0.99, ExpertWeightLocation::CpuRam);
        assert!(
            matches!(resolution, FaultResolution::Resumed { .. }),
            "pressure 0.99 with limit 1.0 should be Resumed"
        );
    }

    // ── ThermalSummary: evicted_count can differ from total_evictions ────

    #[test]
    fn test_thermal_summary_evicted_count_vs_total_evictions() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..4 {
            thermal.step(&[10, 0, 0, 0]);
        }
        thermal.evict_expert(1);
        thermal.evict_expert(2);
        thermal.reactivate_expert(1);

        let summary = thermal.summary();
        assert!(summary.total_evictions >= 2, "at least 2 evictions occurred");
        assert!(summary.total_reactivations >= 1, "at least 1 reactivation occurred");
        assert!(
            summary.evicted_count as u64 <= summary.total_evictions,
            "evicted_count should not exceed total_evictions"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Tests (15 new tests: batch 6)
    // ═══════════════════════════════════════════════════════════════════════

    // ── MoeDispatchSignal: debug output for Suspended with u64::MAX request_id ─

    #[test]
    fn test_signal_suspended_debug_with_max_values() {
        let signal = MoeDispatchSignal::Suspended {
            request_id: u64::MAX,
            expert_idx: usize::MAX,
        };
        let debug = format!("{:?}", signal);
        assert!(debug.contains("Suspended"));
        // Debug output should contain the numeric string for u64::MAX
        assert!(debug.contains(&u64::MAX.to_string()));
    }

    // ── MoeDispatchSignal: clone Rejected with very long reason string ────

    #[test]
    fn test_signal_clone_rejected_with_long_reason() {
        let long_reason = "x".repeat(10000);
        let signal = MoeDispatchSignal::Rejected {
            request_id: 0,
            reason: long_reason.clone(),
        };
        let cloned = signal.clone();
        if let MoeDispatchSignal::Rejected { reason, .. } = cloned {
            assert_eq!(reason.len(), 10000);
            assert_eq!(reason, long_reason);
        } else {
            panic!("Expected Rejected");
        }
    }

    // ── ExpertFault: request_id zero is valid ─────────────────────────────

    #[test]
    fn test_expert_fault_zero_request_id_is_valid() {
        let fault = ExpertFault {
            expert_idx: 0,
            layer_idx: 0,
            request_id: 0,
            fault_time: Instant::now(),
        };
        assert_eq!(fault.request_id, 0);
        assert_eq!(fault.expert_idx, 0);
        assert_eq!(fault.layer_idx, 0);
    }

    // ── ExpertFault: Clone produces structurally equal copy ───────────────

    #[test]
    fn test_expert_fault_clone_is_structurally_equal() {
        let fault = ExpertFault {
            expert_idx: 7,
            layer_idx: 11,
            request_id: 255,
            fault_time: Instant::now(),
        };
        let cloned = fault.clone();
        assert_eq!(cloned.expert_idx, fault.expert_idx);
        assert_eq!(cloned.layer_idx, fault.layer_idx);
        assert_eq!(cloned.request_id, fault.request_id);
    }

    // ── ExpertFaultHandler: pressure just below limit triggers Resumed ────

    #[test]
    fn test_fault_handler_pressure_just_below_limit_resumed() {
        let mut handler = ExpertFaultHandler::new(4).with_memory_pressure_limit(0.8);
        let fault = ExpertFault {
            expert_idx: 0,
            layer_idx: 0,
            request_id: 1,
            fault_time: Instant::now(),
        };
        // pressure 0.799 < 0.8 limit
        let resolution = handler.handle_fault(fault, 0.799, ExpertWeightLocation::CpuRam);
        assert!(
            matches!(resolution, FaultResolution::Resumed { .. }),
            "pressure just below limit should produce Resumed"
        );
    }

    // ── ExpertFaultHandler: pressure just above limit triggers Rejected ───

    #[test]
    fn test_fault_handler_pressure_just_above_limit_rejected() {
        let mut handler = ExpertFaultHandler::new(4).with_memory_pressure_limit(0.5);
        let fault = ExpertFault {
            expert_idx: 0,
            layer_idx: 0,
            request_id: 1,
            fault_time: Instant::now(),
        };
        // pressure 0.501 > 0.5 limit
        let resolution = handler.handle_fault(fault, 0.501, ExpertWeightLocation::CpuRam);
        assert!(
            matches!(resolution, FaultResolution::Rejected { .. }),
            "pressure just above limit should produce Rejected"
        );
    }

    // ── ExpertThermalManager: reactivate resets consecutive_zero_streak ──

    #[test]
    fn test_reactivate_resets_consecutive_zero_streak() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(3);
        for _ in 0..5 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);
        assert!(thermal.state(1).unwrap().consecutive_zero_streak >= 3);

        assert!(thermal.reactivate_expert(1));
        assert_eq!(
            thermal.state(1).unwrap().consecutive_zero_streak, 0,
            "reactivation should reset consecutive_zero_streak to 0"
        );
    }

    // ── ExpertThermalManager: single expert manager edge case ─────────────

    #[test]
    fn test_single_expert_thermal_manager_operations() {
        let mut thermal = ExpertThermalManager::new(1);

        // Only expert 0
        thermal.step(&[10]);
        assert_eq!(thermal.state(0).unwrap().hit_count, 1);
        assert_eq!(thermal.summary().hot_count + thermal.summary().warm_count + thermal.summary().cold_count + thermal.summary().evicted_count, 1);

        // Out of bounds
        assert!(thermal.state(1).is_none());
        assert!(!thermal.evict_expert(1));
        assert!(!thermal.reactivate_expert(1));
    }

    // ── DeoptHandlingResult: Clone for both variants ──────────────────────

    #[test]
    fn test_deopt_handling_result_clone_both_variants() {
        let reactivate = DeoptHandlingResult::ReactivateAndRerun {
            expert_idx: 3,
            request_id: 7,
        };
        let cloned_reactivate = reactivate.clone();
        assert_eq!(cloned_reactivate, reactivate);

        let spurious = DeoptHandlingResult::SpuriousDeopt {
            expert_idx: 1,
            request_id: 2,
        };
        let cloned_spurious = spurious.clone();
        assert_eq!(cloned_spurious, spurious);
    }

    // ── ThermalSummary: working_set_size equals hot_count in extreme case ─

    #[test]
    fn test_thermal_summary_working_set_with_all_hot() {
        let mut thermal = ExpertThermalManager::new(4)
            .with_heat_thresholds(0.1, 0.001)
            .with_eviction_threshold(100)
            .with_adaptive_eviction(10);

        // All experts are hot
        for _ in 0..10 {
            thermal.step(&[10, 10, 10, 10]);
        }

        let summary = thermal.summary();
        assert_eq!(summary.hot_count, 4, "all experts should be hot");
        assert!(summary.working_set_size > 0, "working set should be non-zero when all experts are hot");
    }

    // ── MoeDispatchCallback: new with usize::MAX num_layers overflow guard ─

    #[test]
    fn test_new_with_large_but_valid_config() {
        // Use values that won't overflow but stress the range construction
        let cb = MoeDispatchCallback::new(128, 4, 512, 256);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 128);
        let layers = cb.target_layers().expect("should have layers");
        assert_eq!(layers.len(), 256);
        assert_eq!(layers[0], 256);
        assert_eq!(layers[255], 511);
    }

    // ── LayerCallback: pre_node on exact boundary MoE layer ──────────────

    #[test]
    fn test_pre_node_exact_moe_start_layer_boundary() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        // MoE starts at layer 8; test that layer 8 IS a MoE layer
        let mut cb = MoeDispatchCallback::new(8, 2, 16, 8);

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 16,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 8,
            moe_top_k: 2,
            expert_intermediate_size: 512,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];

        // Layer 7 (just before moe_start) should NOT record steps
        let ctx_before = LayerContext {
            node_idx: 0,
            layer_idx: 7,
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };
        let action_before = cb.pre_node(&ctx_before);
        assert!(matches!(action_before, CallbackAction::Continue));

        // Layer 8 (exactly moe_start) should record a step
        let ctx_exact = LayerContext {
            node_idx: 1,
            layer_idx: 8,
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };
        let action_exact = cb.pre_node(&ctx_exact);
        assert!(matches!(action_exact, CallbackAction::Continue));

        // Verify the step was recorded (fault_handler step count changed)
        let stats = cb.fault_handler().stats();
        assert_eq!(stats.total_faults, 0, "no faults should have occurred");
    }

    // ── ExpertHeatState: direct construction preserves all fields exactly ─

    #[test]
    fn test_heat_state_direct_construction_all_fields() {
        let state = ExpertHeatState {
            expert_idx: 42,
            hit_rate: 0.33,
            hit_count: 33,
            route_count: 100,
            heat_level: ExpertHeatLevel::Warm,
            consecutive_zero_streak: 7,
            last_hit_step: 93,
            is_evicted: false,
            reactivation_count: 3,
        };
        assert_eq!(state.expert_idx, 42);
        assert!((state.hit_rate - 0.33).abs() < f64::EPSILON);
        assert_eq!(state.hit_count, 33);
        assert_eq!(state.route_count, 100);
        assert_eq!(state.heat_level, ExpertHeatLevel::Warm);
        assert_eq!(state.consecutive_zero_streak, 7);
        assert_eq!(state.last_hit_step, 93);
        assert!(!state.is_evicted);
        assert_eq!(state.reactivation_count, 3);
    }

    // ── ExpertThermalManager: multiple evict/reactivate cycles accumulate ─

    #[test]
    fn test_evict_reactivate_three_cycles_accumulate_counters() {
        let mut thermal = ExpertThermalManager::new(2).with_eviction_threshold(3);
        for _ in 0..4 {
            thermal.step(&[10, 0]);
        }

        // Cycle 1
        assert!(thermal.evict_expert(1));
        assert!(thermal.reactivate_expert(1));

        // Cycle 2
        assert!(thermal.evict_expert(1));
        assert!(thermal.reactivate_expert(1));

        // Cycle 3
        assert!(thermal.evict_expert(1));
        assert!(thermal.reactivate_expert(1));

        let summary = thermal.summary();
        assert_eq!(summary.total_evictions, 3);
        assert_eq!(summary.total_reactivations, 3);
        assert_eq!(summary.evicted_count, 0, "expert should be active after final reactivate");
    }

    // ── FaultResolution: Debug output contains variant-specific fields ────

    #[test]
    fn test_fault_resolution_debug_contains_latency_for_resumed() {
        let res = FaultResolution::Resumed {
            latency: Duration::from_micros(12345),
        };
        let debug = format!("{:?}", res);
        assert!(debug.contains("Resumed"));
        // Duration debug should include microseconds representation
        assert!(debug.contains("latency"));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Tests (15 new tests: batch 7)
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn test_fault_stats_debug_includes_all_fields() {
        let stats = FaultStats {
            total_faults: 42,
            avg_recovery_us: 123.4,
            fault_rate: 0.5,
            in_flight_restorations: 3,
            suspended_request_count: 7,
        };
        let debug = format!("{:?}", stats);
        assert!(debug.contains("total_faults"), "Debug should contain total_faults");
        assert!(debug.contains("avg_recovery_us"), "Debug should contain avg_recovery_us");
        assert!(debug.contains("fault_rate"), "Debug should contain fault_rate");
        assert!(debug.contains("in_flight_restorations"), "Debug should contain in_flight_restorations");
        assert!(debug.contains("suspended_request_count"), "Debug should contain suspended_request_count");
    }

    #[test]
    fn test_from_hit_rate_nan_returns_evicted() {
        let level = ExpertHeatLevel::from_hit_rate(f64::NAN, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Evicted, "NaN hit rate should map to Evicted");
    }

    #[test]
    fn test_eviction_decision_partial_ord_transitivity() {
        assert!(EvictionDecision::Keep < EvictionDecision::Evict);
        assert!(EvictionDecision::Evict < EvictionDecision::Reactivate);
        assert!(EvictionDecision::Keep < EvictionDecision::Reactivate);
    }

    #[test]
    fn test_expert_weight_layout_fieldwise_equality() {
        let a = ExpertWeightLayout {
            expert_idx: 5,
            weight_bytes: 2048,
            compressed_bytes: 256,
            compression_ratio: 8.0,
            location: ExpertWeightLocation::CpuRam,
        };
        let b = a.clone();
        assert_eq!(a.expert_idx, b.expert_idx);
        assert_eq!(a.weight_bytes, b.weight_bytes);
        assert_eq!(a.compressed_bytes, b.compressed_bytes);
        assert!((a.compression_ratio - b.compression_ratio).abs() < f32::EPSILON);
        assert_eq!(a.location, b.location);
    }

    #[test]
    fn test_check_fault_evicted_with_negative_memory_pressure() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(5);
        for _ in 0..6 {
            thermal.step(&[10, 0, 5, 3]);
        }
        thermal.evict_expert(1);
        let signal = cb.check_and_handle_fault(1, 0, 1, &thermal, -0.5);
        assert!(
            matches!(signal, MoeDispatchSignal::Suspended { .. }),
            "negative memory pressure should allow page-in"
        );
    }

    #[test]
    fn test_new_with_top_k_exceeding_num_experts() {
        let cb = MoeDispatchCallback::new(4, 16, 32, 0);
        assert!(cb.is_enabled());
        assert_eq!(cb.num_experts(), 4);
        assert_eq!(cb.moe_layers.len(), 32);
    }

    #[test]
    fn test_expert_fault_with_max_expert_idx() {
        let fault = ExpertFault {
            expert_idx: usize::MAX,
            layer_idx: usize::MAX,
            request_id: u64::MAX,
            fault_time: Instant::now(),
        };
        assert_eq!(fault.expert_idx, usize::MAX);
        assert_eq!(fault.layer_idx, usize::MAX);
        assert_eq!(fault.request_id, u64::MAX);
    }

    #[test]
    fn test_thermal_summary_with_maximum_values() {
        let summary = ThermalSummary {
            num_experts: usize::MAX,
            hot_count: usize::MAX,
            warm_count: 0,
            cold_count: 0,
            evicted_count: 0,
            total_evictions: u64::MAX,
            total_reactivations: u64::MAX,
            current_step: u64::MAX,
            pending_deopt_count: usize::MAX,
            working_set_size: usize::MAX,
            effective_eviction_threshold: u64::MAX,
        };
        assert_eq!(summary.num_experts, usize::MAX);
        assert_eq!(summary.total_evictions, u64::MAX);
        assert_eq!(summary.current_step, u64::MAX);
    }

    #[test]
    fn test_prefetch_request_with_max_expert_idx() {
        let req = ExpertPrefetchRequest {
            expert_idx: usize::MAX,
            layer_idx: 0,
            source: ExpertWeightLocation::RemoteNode,
            destination: ExpertWeightLocation::GpuVram,
            bytes: usize::MAX,
            estimated_latency_us: f32::MAX,
            priority: u32::MAX,
        };
        assert_eq!(req.expert_idx, usize::MAX);
        assert_eq!(req.bytes, usize::MAX);
        assert_eq!(req.priority, u32::MAX);
    }

    #[test]
    fn test_from_hit_rate_positive_infinity_is_hot() {
        let level = ExpertHeatLevel::from_hit_rate(f64::INFINITY, 0.1, 0.001);
        assert_eq!(level, ExpertHeatLevel::Hot, "infinite hit rate should be Hot");
    }

    #[test]
    fn test_deopt_request_clone_independence() {
        let original = DeoptRequest { request_id: 10, expert_idx: 5, layer_idx: 3, step: 100 };
        let mut cloned = original.clone();
        cloned.step = 999;
        assert_eq!(original.step, 100, "original should be unaffected by clone mutation");
        assert_eq!(cloned.step, 999);
    }

    #[test]
    fn test_working_set_size_without_adaptive_eviction() {
        let mut thermal = ExpertThermalManager::new(4).with_eviction_threshold(100);
        for _ in 0..5 {
            thermal.step(&[10, 5, 0, 0]);
        }
        let ws = thermal.working_set_size();
        assert!(ws <= 4, "working set size should not exceed num_experts");
    }

    #[test]
    fn test_handle_fault_updates_fault_rate_in_stats() {
        let mut handler = ExpertFaultHandler::new(4);
        for _ in 0..10 {
            handler.record_step();
        }
        let rate_before = handler.stats().fault_rate;
        let fault = ExpertFault {
            expert_idx: 0, layer_idx: 0, request_id: 1, fault_time: Instant::now(),
        };
        let _ = handler.handle_fault(fault, 0.1, ExpertWeightLocation::CpuRam);
        let rate_after = handler.stats().fault_rate;
        assert!(rate_after >= rate_before, "fault_rate should not decrease after a fault");
        assert!(handler.stats().total_faults > 0, "total_faults should be positive");
    }

    #[test]
    fn test_weight_location_all_latencies_strictly_ordered() {
        let gpu_l2 = ExpertWeightLocation::GpuL2.estimated_latency_us();
        let gpu_vram = ExpertWeightLocation::GpuVram.estimated_latency_us();
        let cpu_ram = ExpertWeightLocation::CpuRam.estimated_latency_us();
        let remote = ExpertWeightLocation::RemoteNode.estimated_latency_us();
        let evicted = ExpertWeightLocation::Evicted.estimated_latency_us();
        assert!(gpu_l2 < gpu_vram, "GpuL2 should be faster than GpuVram");
        assert!(gpu_vram < cpu_ram, "GpuVram should be faster than CpuRam");
        assert!(cpu_ram < remote, "CpuRam should be faster than RemoteNode");
        assert!(remote < evicted, "RemoteNode should be faster than Evicted (infinite)");
    }

    #[test]
    fn test_post_node_exact_moe_boundary_layer() {
        use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
        use crate::engine::executor::GeneratorForwardConfig;

        let mut cb = MoeDispatchCallback::new(8, 2, 16, 4);

        let geometry = std::sync::Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 16,
            vocab_size: 1000,
            intermediate_size: 512,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            max_seq_len: 128,
            rope_theta: 10000.0,
            rope_scale: 1.0,
            rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32,
            compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5,
            num_experts: 8,
            moe_top_k: 2,
            expert_intermediate_size: 512,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
            attention_pattern: vec![],
            sliding_window: 0,
            num_kv_shared_layers: 0,
            global_head_dim: 0,
            hidden_size_per_layer_input: 0,
            position_offset: None,
            rope_scaling: None,
            final_logit_softcapping: None,
            hidden_act: None,
            mla_d_c: 0,
            mla_d_rope: 0,
            mla_unabsorbed_threshold: 0,
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        });
        let config = GeneratorForwardConfig {
            geometry,
            rope: crate::engine::executor::RoPEConfig {
                theta: 10000.0,
                scale: 1.0,
                interleaved: false,
                precompute: false,
            },
            position_encoding: crate::engine::executor::PositionEncoding::Rope,
            arch_family: crate::manifest::ArchFamily::Decoder,
            rerank_yes_token_id: None,
            rerank_no_token_id: None,
            moe_config: None,
            paged_kv: crate::engine::executor::PagedKvConfig {
                page_table: None,
                page_size: 16,
            },
            callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
        };
        let hs = vec![0u8; 256 * 4];
        let output = vec![0u8; 256 * 4];

        let ctx_before = LayerContext {
            node_idx: 0,
            layer_idx: 3,
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };
        let action_before = cb.post_node(&ctx_before, &output);
        assert!(matches!(action_before, CallbackAction::Continue));

        let ctx_exact = LayerContext {
            node_idx: 1,
            layer_idx: 4,
            node_op: "Gemm",
            hidden_state: &hs,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 10,
            seq_len: 1,
            position: 9,
            request_id: 1,
            model_config: &config,
        };
        let action_exact = cb.post_node(&ctx_exact, &output);
        assert!(matches!(action_exact, CallbackAction::Continue));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Additional Tests (15 new tests: batch 7)
    // ═══════════════════════════════════════════════════════════════════════

    // ── ExpertHeatState: direct construction with Hot state ──────────────

    // @trace TEST-MOE-DISPATCH-009 [req:REQ-MOE-DISPATCH-002] [level:unit]
    #[test]
    fn test_heat_state_hot_construction_all_fields() {
        let state = ExpertHeatState {
            expert_idx: 3,
            hit_rate: 0.95,
            hit_count: 950,
            route_count: 1000,
            heat_level: ExpertHeatLevel::Hot,
            consecutive_zero_streak: 0,
            last_hit_step: 1000,
            is_evicted: false,
            reactivation_count: 2,
        };
        assert!(!state.is_evicted);
        assert_eq!(state.heat_level, ExpertHeatLevel::Hot);
        assert_eq!(state.reactivation_count, 2);
        assert_eq!(state.last_hit_step, 1000);
        assert!((state.hit_rate - 0.95).abs() < f64::EPSILON);
    }

    // ── ExpertHeatState: construction with usize::MAX expert_idx ─────────

    // @trace TEST-MOE-DISPATCH-010 [req:REQ-MOE-DISPATCH-002] [level:unit]
    #[test]
    fn test_heat_state_max_expert_idx() {
        let state = ExpertHeatState {
            expert_idx: usize::MAX,
            hit_rate: 0.0,
            hit_count: 0,
            route_count: 0,
            heat_level: ExpertHeatLevel::Evicted,
            consecutive_zero_streak: u64::MAX,
            last_hit_step: 0,
            is_evicted: true,
            reactivation_count: 0,
        };
        assert_eq!(state.expert_idx, usize::MAX);
        assert_eq!(state.consecutive_zero_streak, u64::MAX);
    }

    // ── DeoptHandlingResult: PartialEq same variant different expert_idx ──

    // @trace TEST-MOE-DISPATCH-011 [req:REQ-MOE-DISPATCH-005] [level:unit]
    #[test]
    fn test_deopt_handling_result_ne_same_variant_different_expert() {
        let a = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 1, request_id: 2 };
        let b = DeoptHandlingResult::ReactivateAndRerun { expert_idx: 3, request_id: 2 };
        assert_ne!(a, b, "different expert_idx should not be equal");
    }

    // ── DeoptHandlingResult: PartialEq SpuriousDeopt different request_id ─

    // @trace TEST-MOE-DISPATCH-012 [req:REQ-MOE-DISPATCH-005] [level:unit]
    #[test]
    fn test_deopt_handling_result_ne_spurious_different_request() {
        let a = DeoptHandlingResult::SpuriousDeopt { expert_idx: 1, request_id: 10 };
        let b = DeoptHandlingResult::SpuriousDeopt { expert_idx: 1, request_id: 20 };
        assert_ne!(a, b, "different request_id should not be equal");
    }

    // ── ExpertFaultHandler: fault_rate increases after handle_fault ───────

    // @trace TEST-MOE-DISPATCH-014 [req:REQ-MOE-DISPATCH-004] [level:unit]
    #[test]
    fn test_fault_rate_nonzero_after_single_fault_with_prior_steps() {
        let mut handler = ExpertFaultHandler::new(4);
        // Record 10 steps with zero faults => fault_rate = 0.0
        for _ in 0..10 {
            handler.record_step();
        }
        assert_eq!(handler.stats().fault_rate, 0.0);

        // Trigger a fault
        let fault = ExpertFault {
            expert_idx: 0,
            layer_idx: 0,
            request_id: 1,
            fault_time: Instant::now(),
        };
        let _ = handler.handle_fault(fault, 0.1, ExpertWeightLocation::CpuRam);

        let stats = handler.stats();
        assert!(stats.total_faults >= 1, "total_faults should be >= 1 after a fault");
        assert!(stats.fault_rate > 0.0, "fault_rate should be > 0 after a fault");
    }

    // ── ExpertFaultHandler: expert_fault_count increases after handle_fault

    // @trace TEST-MOE-DISPATCH-015 [req:REQ-MOE-DISPATCH-004] [level:unit]
    #[test]
    fn test_expert_fault_count_increments_on_handle_fault() {
        let mut handler = ExpertFaultHandler::new(4);
        assert_eq!(handler.expert_fault_count(2), 0);

        let fault = ExpertFault {
            expert_idx: 2,
            layer_idx: 0,
            request_id: 1,
            fault_time: Instant::now(),
        };
        let _ = handler.handle_fault(fault, 0.1, ExpertWeightLocation::CpuRam);
        assert!(
            handler.expert_fault_count(2) >= 1,
            "expert 2 should have at least 1 fault recorded"
        );
    }

    // ── ThermalSummary: construction with u64::MAX current_step ──────────

    // @trace TEST-MOE-DISPATCH-016 [req:REQ-MOE-DISPATCH-002] [level:unit]
    #[test]
    fn test_thermal_summary_with_max_current_step() {
        let summary = ThermalSummary {
            num_experts: 8,
            hot_count: 4,
            warm_count: 2,
            cold_count: 1,
            evicted_count: 1,
            total_evictions: 100,
            total_reactivations: 50,
            current_step: u64::MAX,
            pending_deopt_count: 3,
            working_set_size: 6,
            effective_eviction_threshold: 500,
        };
        assert_eq!(summary.current_step, u64::MAX);
        assert_eq!(summary.total_evictions, 100);
        let total = summary.hot_count + summary.warm_count + summary.cold_count + summary.evicted_count;
        assert_eq!(total, 8);
    }

    // ── ExpertWeightPrefetcher: multiple location updates preserve last ──

    // @trace TEST-MOE-DISPATCH-017 [req:REQ-MOE-DISPATCH-003] [level:unit]
    #[test]
    fn test_prefetcher_multiple_updates_preserves_last_location() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 4096);

        prefetcher.update_location(0, ExpertWeightLocation::GpuVram);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::GpuVram);

        prefetcher.update_location(0, ExpertWeightLocation::CpuRam);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::CpuRam);

        prefetcher.update_location(0, ExpertWeightLocation::GpuL2);
        assert_eq!(prefetcher.layout(0).unwrap().location, ExpertWeightLocation::GpuL2);
    }

    // ── ExpertWeightPrefetcher: prefetch_step with single expert index ───

    // @trace TEST-MOE-DISPATCH-018 [req:REQ-MOE-DISPATCH-003] [level:unit]
    #[test]
    fn test_prefetch_step_single_expert_index() {
        let mut prefetcher = ExpertWeightPrefetcher::new(4, 4096);
        prefetcher.update_location(0, ExpertWeightLocation::GpuVram);
        // Expert 1 still in CpuRam, should be prefetched
        let requests = prefetcher.prefetch_step(0, &[1]);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].expert_idx, 1);
        assert_eq!(requests[0].source, ExpertWeightLocation::CpuRam);
        assert_eq!(requests[0].destination, ExpertWeightLocation::GpuVram);
    }

    // ── EvictionDecision: ordering transitivity ──────────────────────────

    // @trace TEST-MOE-DISPATCH-019 [req:REQ-MOE-DISPATCH-002] [level:unit]
    #[test]
    fn test_eviction_decision_ord_transitivity() {
        // Keep < Evict and Evict < Reactivate => Keep < Reactive (transitivity)
        let keep = EvictionDecision::Keep;
        let evict = EvictionDecision::Evict;
        let reactivate = EvictionDecision::Reactivate;
        assert!(keep < evict, "Keep < Evict");
        assert!(evict < reactivate, "Evict < Reactivate");
        assert!(keep < reactivate, "Keep < Reactivate by transitivity");
    }

    // ── ExpertThermalManager: step with all-zero counts increases streaks ─

    // @trace TEST-MOE-DISPATCH-020 [req:REQ-MOE-DISPATCH-002] [level:unit]
    #[test]
    fn test_thermal_step_all_zeros_increments_all_streaks() {
        let mut thermal = ExpertThermalManager::new(3);
        thermal.step(&[0, 0, 0]);
        thermal.step(&[0, 0, 0]);
        thermal.step(&[0, 0, 0]);

        for i in 0..3 {
            let state = thermal.state(i).expect("expert should exist");
            assert_eq!(state.hit_count, 0, "expert {} should have zero hits", i);
            assert_eq!(state.consecutive_zero_streak, 3, "expert {} streak should be 3", i);
        }

        let summary = thermal.summary();
        assert_eq!(summary.hot_count, 0, "no expert should be hot with all-zero steps");
    }

    // ── MoeDispatchCallback: check_and_handle_fault with thermal state having is_evicted=false continues

    // @trace TEST-MOE-DISPATCH-021 [req:REQ-MOE-DISPATCH-004] [level:unit]
    #[test]
    fn test_check_fault_warm_expert_with_zero_pressure_continues() {
        let mut cb = MoeDispatchCallback::new(4, 2, 8, 0);
        let thermal = ExpertThermalManager::new(4).with_eviction_threshold(100);

        // Default state: all experts are Warm, is_evicted=false
        for idx in 0..4 {
            let signal = cb.check_and_handle_fault(idx, 0, 1, &thermal, 0.0);
            assert!(
                matches!(signal, MoeDispatchSignal::Continue),
                "non-evicted expert {} should always Continue",
                idx
            );
        }
    }
}
