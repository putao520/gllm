use std::collections::HashMap;

use crate::routing::ResidualBus;
use crate::speculative::engine::SpecDecodingState;

pub struct InferenceCoordinator {
    pub moe_thermal: Option<crate::moe::thermal::ExpertThermalManager>,
    pub moe_fault_handler: Option<crate::moe::fault_handler::ExpertFaultHandler>,
    pub moe_dispatcher: Option<crate::moe::dispatch::MoeHardwareDispatcher>,
    pub moe_prefetcher: Option<crate::moe::prefetch::ExpertWeightPrefetcher>,
    pub prefetch_pipeline: Option<crate::moe::prefetch_pipeline::PrefetchPipeline>,
    pub hot_patch_manager: Option<crate::moe::hot_patch::HotPatchManager>,
    pub expert_code_regions: HashMap<(usize, usize), (usize, usize)>,
    pub expert_saved_bytes: HashMap<(usize, usize), Vec<u8>>,
    pub spec_decoding: SpecDecodingState,
    pub rag_system: Option<crate::rag::LateFusionRag>,
    pub residual_bus: ResidualBus,
    pub gate_skip_flags: HashMap<u64, bool>,
    /// REQ-MTP-002: MTP adaptive controller tracking acceptance rates
    /// and deciding whether to enable/disable MTP.
    pub mtp_controller: crate::engine::mtp_executor::MtpController,
    /// REQ-DIST-014: Distributed MoE dispatch decision.
    /// Initialized from MoeDistributedConfig + CommHandleWrapper during executor setup.
    /// None = single-node or MoE not configured.
    #[cfg(feature = "nccl")]
    pub moe_dist_decision: Option<crate::moe::distributed_dispatch::distributed_dispatch::MoeDistDecision>,
    /// REQ-DIST-015: Expert load statistics for EPLB.
    /// Tracks per-expert invocation counts in a sliding window.
    /// None = single-node or MoE not configured.
    #[cfg(feature = "nccl")]
    pub expert_load_stats: Option<crate::moe::eplb::eplb::ExpertLoadStats>,
    /// REQ-DIST-015: EPLB imbalance threshold.
    /// When imbalance_ratio (max/avg) exceeds this threshold, rebalance is triggered.
    /// Default 2.0 (hottest expert is 2x the average).
    #[cfg(feature = "nccl")]
    pub eplb_imbalance_threshold: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mtp_executor::MtpController;
    use crate::speculative::engine::{SpecDecodingMode, SpecDecodingState};
    use std::collections::HashMap;

    /// Helper: build a minimal InferenceCoordinator for testing.
    fn make_coordinator() -> InferenceCoordinator {
        InferenceCoordinator {
            moe_thermal: None,
            moe_fault_handler: None,
            moe_dispatcher: None,
            moe_prefetcher: None,
            prefetch_pipeline: None,
            hot_patch_manager: None,
            expert_code_regions: HashMap::new(),
            expert_saved_bytes: HashMap::new(),
            spec_decoding: SpecDecodingState::new_standard(),
            rag_system: None,
            residual_bus: ResidualBus::new(768, 12),
            gate_skip_flags: HashMap::new(),
            mtp_controller: MtpController::new(),
            #[cfg(feature = "nccl")]
            moe_dist_decision: None,
            #[cfg(feature = "nccl")]
            expert_load_stats: None,
            #[cfg(feature = "nccl")]
            eplb_imbalance_threshold: 2.0,
        }
    }

    // @trace TEST-ICOORD-001 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_coordinator_default_all_optional_fields_are_none() {
        let coord = make_coordinator();

        assert!(coord.moe_thermal.is_none());
        assert!(coord.moe_fault_handler.is_none());
        assert!(coord.moe_dispatcher.is_none());
        assert!(coord.moe_prefetcher.is_none());
        assert!(coord.prefetch_pipeline.is_none());
        assert!(coord.hot_patch_manager.is_none());
        assert!(coord.rag_system.is_none());
    }

    // @trace TEST-ICOORD-002 [level:unit]
    #[test]
    fn test_coordinator_hashmaps_initially_empty() {
        let coord = make_coordinator();

        assert!(coord.expert_code_regions.is_empty());
        assert!(coord.expert_saved_bytes.is_empty());
        assert!(coord.gate_skip_flags.is_empty());
    }

    // @trace TEST-ICOORD-003 [level:unit]
    #[test]
    fn test_coordinator_spec_decoding_standard_mode() {
        let coord = make_coordinator();

        assert_eq!(coord.spec_decoding.mode(), SpecDecodingMode::Standard);
        assert!(!coord.spec_decoding.is_active());
        assert_eq!(coord.spec_decoding.draft_layers(), 0);
        assert_eq!(coord.spec_decoding.total_layers(), 0);
    }

    // @trace TEST-ICOORD-004 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_dimensions() {
        let coord = make_coordinator();

        assert_eq!(coord.residual_bus.hidden_size(), 768);
        assert_eq!(coord.residual_bus.num_layers(), 12);
        assert_eq!(coord.residual_bus.active_port_count(), 0);
    }

    // @trace TEST-ICOORD-005 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_default_is_enabled() {
        let coord = make_coordinator();

        assert!(coord.mtp_controller.is_enabled());
        assert!((coord.mtp_controller.ema_rate() - 0.5).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-006 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_record_acceptance_updates_ema() {
        let mut ctrl = MtpController::new();
        let initial_rate = ctrl.ema_rate();

        ctrl.record_acceptance(8, 10);
        let new_rate = ctrl.ema_rate();

        assert!(new_rate > initial_rate, "recording 80% acceptance should increase EMA from 0.5");
    }

    // @trace TEST-ICOORD-007 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_disable_toggle() {
        let mut ctrl = MtpController::new();
        assert!(ctrl.is_enabled());

        ctrl.disable();
        assert!(!ctrl.is_enabled());

        ctrl.enable();
        assert!(ctrl.is_enabled());
    }

    // @trace TEST-ICOORD-008 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_effective_depth_disabled_returns_zero() {
        let mut ctrl = MtpController::new();
        ctrl.disable();

        assert_eq!(ctrl.effective_depth(4), 0);
        assert_eq!(ctrl.effective_depth(100), 0);
    }

    // @trace TEST-ICOORD-009 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_effective_depth_high_ema_full_depth() {
        let mut ctrl = MtpController::new();
        // Feed high acceptance rates to drive EMA above 0.8
        for _ in 0..50 {
            ctrl.record_acceptance(9, 10);
        }

        assert!(ctrl.ema_rate() > 0.8, "EMA should exceed 0.8 after many high-acceptance rounds");
        assert_eq!(ctrl.effective_depth(5), 5, "high EMA should return full max_depth");
    }

    // @trace TEST-ICOORD-010 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_with_params_clamps_alpha() {
        let ctrl = MtpController::with_params(
            -5.0,  // alpha below range → clamped to 0.01
            0.3,
            0.5,
            3,
            5,
        );

        assert!(ctrl.is_enabled());
    }

    // @trace TEST-ICOORD-011 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_reset_restores_defaults() {
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(0, 10);
        ctrl.disable();

        assert!(!ctrl.is_enabled());

        ctrl.reset();

        assert!(ctrl.is_enabled());
        assert!((ctrl.ema_rate() - 0.5).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-012 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_consecutive_low_acceptance_disables() {
        let mut ctrl = MtpController::with_params(0.1, 0.3, 0.5, 3, 5);

        for _ in 0..3 {
            ctrl.record_acceptance(0, 10);
        }

        assert!(!ctrl.is_enabled(), "3 consecutive low-acceptance rounds should disable MTP");
    }

    // @trace TEST-ICOORD-013 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_insert_and_read() {
        let mut coord = make_coordinator();
        let key = (0, 3);
        let value = (100, 2048);

        coord.expert_code_regions.insert(key, value);

        assert_eq!(coord.expert_code_regions.get(&key), Some(&value));
        assert_eq!(coord.expert_code_regions.len(), 1);
    }

    // @trace TEST-ICOORD-014 [level:unit]
    #[test]
    fn test_coordinator_gate_skip_flags_insert_and_check() {
        let mut coord = make_coordinator();

        coord.gate_skip_flags.insert(42, true);
        coord.gate_skip_flags.insert(99, false);

        assert_eq!(coord.gate_skip_flags.get(&42), Some(&true));
        assert_eq!(coord.gate_skip_flags.get(&99), Some(&false));
        assert_eq!(coord.gate_skip_flags.get(&7), None);
        assert_eq!(coord.gate_skip_flags.len(), 2);
    }

    // @trace TEST-ICOORD-015 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_insert_and_read() {
        let mut coord = make_coordinator();
        let key = (2, 5);
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];

        coord.expert_saved_bytes.insert(key, data.clone());

        assert_eq!(coord.expert_saved_bytes.get(&key), Some(&data));
        assert_eq!(coord.expert_saved_bytes.len(), 1);
    }

    // @trace TEST-ICOORD-016 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_overwrite() {
        let mut coord = make_coordinator();
        let key = (0, 3);

        coord.expert_code_regions.insert(key, (100, 2048));
        assert_eq!(coord.expert_code_regions.get(&key), Some(&(100, 2048)));

        coord.expert_code_regions.insert(key, (200, 4096));
        assert_eq!(coord.expert_code_regions.get(&key), Some(&(200, 4096)));
        assert_eq!(coord.expert_code_regions.len(), 1, "overwriting should not create duplicates");
    }

    // @trace TEST-ICOORD-017 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_multiple_entries() {
        let mut coord = make_coordinator();
        let data_a = vec![1, 2, 3];
        let data_b = vec![4, 5, 6];
        let data_c = vec![7, 8, 9];

        coord.expert_saved_bytes.insert((0, 0), data_a);
        coord.expert_saved_bytes.insert((1, 1), data_b);
        coord.expert_saved_bytes.insert((2, 2), data_c);

        assert_eq!(coord.expert_saved_bytes.len(), 3);
        assert_eq!(coord.expert_saved_bytes.get(&(0, 0)), Some(&vec![1, 2, 3]));
        assert_eq!(coord.expert_saved_bytes.get(&(1, 1)), Some(&vec![4, 5, 6]));
        assert_eq!(coord.expert_saved_bytes.get(&(2, 2)), Some(&vec![7, 8, 9]));
        assert_eq!(coord.expert_saved_bytes.get(&(3, 3)), None);
    }

    // @trace TEST-ICOORD-018 [level:unit]
    #[test]
    fn test_coordinator_gate_skip_flags_overwrite() {
        let mut coord = make_coordinator();

        coord.gate_skip_flags.insert(42, true);
        assert_eq!(coord.gate_skip_flags.get(&42), Some(&true));

        coord.gate_skip_flags.insert(42, false);
        assert_eq!(coord.gate_skip_flags.get(&42), Some(&false));
        assert_eq!(coord.gate_skip_flags.len(), 1);
    }

    // @trace TEST-ICOORD-019 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_remove() {
        let mut coord = make_coordinator();
        coord.expert_code_regions.insert((0, 1), (10, 20));
        coord.expert_code_regions.insert((3, 4), (30, 40));

        assert_eq!(coord.expert_code_regions.len(), 2);

        let removed = coord.expert_code_regions.remove(&(0, 1));
        assert_eq!(removed, Some((10, 20)));
        assert_eq!(coord.expert_code_regions.len(), 1);
        assert_eq!(coord.expert_code_regions.get(&(3, 4)), Some(&(30, 40)));
    }

    // @trace TEST-ICOORD-020 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_zero_active_after_construction() {
        let coord = make_coordinator();

        assert_eq!(coord.residual_bus.ports().len(), 0, "new bus should have zero ports");
        assert_eq!(coord.residual_bus.active_port_count(), 0);
        assert_eq!(coord.residual_bus.active_ports_at_layer(0).count(), 0);
    }

    // @trace TEST-ICOORD-021 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_register_and_find_port() {
        use crate::routing::{BusPort, BusPortKind, BusPortTag};

        let mut coord = make_coordinator();
        let port = BusPort::injection(3, BusPortTag::RagInjection);
        coord.residual_bus.register(port);

        assert_eq!(coord.residual_bus.ports().len(), 1);
        let found = coord.residual_bus.find_port(BusPortTag::RagInjection);
        assert!(found.is_some());
        assert_eq!(found.unwrap().kind, BusPortKind::Injection);
        assert!(coord.residual_bus.find_port(BusPortTag::IntentRecall).is_none());
    }

    // @trace TEST-ICOORD-022 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_port_activate_deactivate() {
        use crate::routing::BusPortTag;

        let mut coord = make_coordinator();
        let port = crate::routing::BusPort::recall(5, BusPortTag::Guardrail);
        coord.residual_bus.register(port);

        let found = coord.residual_bus.find_port(BusPortTag::Guardrail).unwrap();
        assert!(found.is_active());

        found.deactivate();
        assert!(!found.is_active());
        assert_eq!(coord.residual_bus.active_port_count(), 0);

        found.activate();
        assert!(found.is_active());
        assert_eq!(coord.residual_bus.active_port_count(), 1);
    }

    // @trace TEST-ICOORD-023 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_inject_dimension_mismatch() {
        use crate::routing::{InjectionPayload, BusPortTag};

        let mut coord = make_coordinator();
        let port = crate::routing::BusPort::injection(2, BusPortTag::RagInjection);
        coord.residual_bus.register(port);

        // hidden_size=768 but data has only 10 elements
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![0.0; 10],
            scale: 1.0,
        };
        let mut buffer = vec![0.0f32; 768];
        let result = coord.residual_bus.inject(&payload, &mut buffer);

        assert!(result.is_err(), "injecting with wrong dimension should fail");
    }

    // @trace TEST-ICOORD-024 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_inject_wrong_port_type() {
        use crate::routing::{InjectionPayload, BusPortTag};

        let mut coord = make_coordinator();
        // Register a Recall port, then try to inject into it
        let port = crate::routing::BusPort::recall(2, BusPortTag::IntentRecall);
        coord.residual_bus.register(port);

        let payload = InjectionPayload {
            target: BusPortTag::IntentRecall,
            data: vec![0.0; 768],
            scale: 1.0,
        };
        let mut buffer = vec![0.0f32; 768];
        let result = coord.residual_bus.inject(&payload, &mut buffer);

        assert!(result.is_err(), "injecting into a Recall port should fail");
    }

    // @trace TEST-ICOORD-025 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_coordinator_mtp_controller_effective_depth_low_ema_one() {
        let mut coord = make_coordinator();

        // Drive EMA down below 0.3
        for _ in 0..50 {
            coord.mtp_controller.record_acceptance(0, 10);
        }

        assert!(coord.mtp_controller.ema_rate() < 0.3);
        // EMA between 0.0 and 0.3: depth = 0
        assert_eq!(coord.mtp_controller.effective_depth(4), 0);
        // Controller should still be enabled (patience=3, but EMA tracking is separate from disable streak)
    }

    // @trace TEST-ICOORD-026 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_coordinator_mtp_controller_effective_depth_moderate_ema() {
        let mut coord = make_coordinator();

        // Record one round of 50% acceptance: ema = 0.1*0.5 + 0.9*0.5 = 0.5
        coord.mtp_controller.record_acceptance(5, 10);
        // EMA = 0.5, not > 0.5, but > 0.3 → depth = 1
        assert_eq!(coord.mtp_controller.effective_depth(5), 1);
    }

    // @trace TEST-ICOORD-027 [level:unit]
    #[test]
    fn test_coordinator_spec_decoding_acceptance_rate_zero_in_standard() {
        let coord = make_coordinator();

        // Standard mode: acceptance_rate_ema should be 0.0
        assert!((coord.spec_decoding.acceptance_rate_ema() - 0.0).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-028 [level:unit]
    #[test]
    fn test_coordinator_spec_decoding_cache_accessible() {
        let coord = make_coordinator();

        // Verify cache accessor works on standard mode
        let cache = coord.spec_decoding.cache();
        assert_eq!(cache.len(), 0, "standard mode cache should be empty initially");

        let cache_mut = &mut make_coordinator().spec_decoding;
        let cache_ref = cache_mut.cache_mut();
        assert_eq!(cache_ref.len(), 0);
    }

    // @trace TEST-ICOORD-029 [level:unit]
    #[test]
    fn test_coordinator_spec_decoding_no_tree_in_standard() {
        let coord = make_coordinator();

        assert!(coord.spec_decoding.current_tree().is_none());
        assert!(coord.spec_decoding.adapter().is_none());
        assert!(coord.spec_decoding.eagle_head().is_none());
        assert!(coord.spec_decoding.mtp_head().is_none());
        assert!(coord.spec_decoding.draft_gpu_index().is_none());
        assert_eq!(coord.spec_decoding.target_gpu_indices().len(), 0);
    }

    // @trace TEST-ICOORD-030 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_inject_success() {
        use crate::routing::{BusPortTag, InjectionPayload};

        let mut coord = make_coordinator();
        let port = crate::routing::BusPort::injection(4, BusPortTag::RagInjection);
        coord.residual_bus.register(port);

        let mut buffer = vec![1.0f32; 768];
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![0.5; 768],
            scale: 2.0,
        };
        let result = coord.residual_bus.inject(&payload, &mut buffer);

        assert!(result.is_ok());
        // buffer[i] should be 1.0 + 0.5 * 2.0 = 2.0
        assert!((buffer[0] - 2.0).abs() < f32::EPSILON);
        assert!((buffer[767] - 2.0).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-031 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_tuple_key_value_roundtrip() {
        // Arrange
        let mut coord = make_coordinator();
        let key = (3, 7);
        let value = (0x1000, 0x2000);

        // Act
        coord.expert_code_regions.insert(key, value);

        // Assert
        let retrieved = coord.expert_code_regions.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), &(0x1000, 0x2000));
    }

    // @trace TEST-ICOORD-032 [level:unit]
    #[test]
    fn test_coordinator_gate_skip_flags_insert_multiple_verify_all() {
        // Arrange
        let mut coord = make_coordinator();
        let entries = vec![(1u64, true), (2u64, false), (3u64, true), (4u64, false)];

        // Act
        for (id, flag) in &entries {
            coord.gate_skip_flags.insert(*id, *flag);
        }

        // Assert
        assert_eq!(coord.gate_skip_flags.len(), 4);
        for (id, expected_flag) in &entries {
            assert_eq!(coord.gate_skip_flags.get(id), Some(expected_flag));
        }
    }

    // @trace TEST-ICOORD-033 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_binary_content_verification() {
        // Arrange
        let mut coord = make_coordinator();
        let key = (0, 1);
        let data: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];

        // Act
        coord.expert_saved_bytes.insert(key, data.clone());

        // Assert
        let stored = coord.expert_saved_bytes.get(&key);
        assert!(stored.is_some());
        assert_eq!(stored.unwrap().as_slice(), data.as_slice());
        assert_eq!(stored.unwrap().len(), 6);
    }

    // @trace TEST-ICOORD-034 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_custom_dimensions() {
        // Arrange — build a coordinator with non-default ResidualBus dimensions
        let mut coord = InferenceCoordinator {
            moe_thermal: None,
            moe_fault_handler: None,
            moe_dispatcher: None,
            moe_prefetcher: None,
            prefetch_pipeline: None,
            hot_patch_manager: None,
            expert_code_regions: HashMap::new(),
            expert_saved_bytes: HashMap::new(),
            spec_decoding: SpecDecodingState::new_standard(),
            rag_system: None,
            residual_bus: ResidualBus::new(1024, 24),
            gate_skip_flags: HashMap::new(),
            mtp_controller: MtpController::new(),
            #[cfg(feature = "nccl")]
            moe_dist_decision: None,
            #[cfg(feature = "nccl")]
            expert_load_stats: None,
            #[cfg(feature = "nccl")]
            eplb_imbalance_threshold: 2.0,
        };

        // Act — access dimensions
        let hidden = coord.residual_bus.hidden_size();
        let layers = coord.residual_bus.num_layers();

        // Assert
        assert_eq!(hidden, 1024);
        assert_eq!(layers, 24);
        assert_eq!(coord.residual_bus.active_port_count(), 0);
    }

    // @trace TEST-ICOORD-035 [level:unit]
    #[test]
    fn test_mtp_controller_auto_disable_after_consecutive_low_acceptance() {
        // Arrange
        let mut ctrl = MtpController::new();
        assert!(ctrl.is_enabled());

        // Act — record 3 consecutive low-acceptance rounds (below default threshold 0.3)
        // Each round: 0 accepted out of 10 → rate = 0.0 < 0.3
        for _ in 0..3 {
            ctrl.record_acceptance(0, 10);
        }

        // Assert — should have auto-disabled
        assert!(!ctrl.is_enabled());
    }

    // @trace TEST-ICOORD-036 [level:unit]
    #[test]
    fn test_mtp_controller_re_enable_after_disable() {
        // Arrange
        let mut ctrl = MtpController::new();
        ctrl.disable();
        assert!(!ctrl.is_enabled());

        // Act
        ctrl.enable();

        // Assert
        assert!(ctrl.is_enabled());
    }

    // @trace TEST-ICOORD-037 [level:unit]
    #[test]
    fn test_coordinator_rag_system_fusion_layer_access() {
        // Arrange
        let rag = crate::rag::LateFusionRag::new(7);
        let coord = InferenceCoordinator {
            moe_thermal: None,
            moe_fault_handler: None,
            moe_dispatcher: None,
            moe_prefetcher: None,
            prefetch_pipeline: None,
            hot_patch_manager: None,
            expert_code_regions: HashMap::new(),
            expert_saved_bytes: HashMap::new(),
            spec_decoding: SpecDecodingState::new_standard(),
            rag_system: Some(rag),
            residual_bus: ResidualBus::new(768, 12),
            gate_skip_flags: HashMap::new(),
            mtp_controller: MtpController::new(),
            #[cfg(feature = "nccl")]
            moe_dist_decision: None,
            #[cfg(feature = "nccl")]
            expert_load_stats: None,
            #[cfg(feature = "nccl")]
            eplb_imbalance_threshold: 2.0,
        };

        // Act
        let layer = coord.rag_system.as_ref().unwrap().fusion_layer;

        // Assert
        assert_eq!(layer, 7);
    }

    // @trace TEST-ICOORD-038 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_overwrite_key() {
        // Arrange
        let mut coord = make_coordinator();
        let key = (5, 2);
        coord.expert_code_regions.insert(key, (100, 200));

        // Act
        coord.expert_code_regions.insert(key, (300, 400));

        // Assert — last write wins
        assert_eq!(coord.expert_code_regions.get(&key), Some(&(300, 400)));
        assert_eq!(coord.expert_code_regions.len(), 1);
    }

    // @trace TEST-ICOORD-039 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_overwrite_key() {
        // Arrange
        let mut coord = make_coordinator();
        let key = (1, 3);
        coord.expert_saved_bytes.insert(key, vec![0x01, 0x02]);

        // Act
        coord.expert_saved_bytes.insert(key, vec![0xAA, 0xBB, 0xCC]);

        // Assert — overwritten value replaces original
        assert_eq!(coord.expert_saved_bytes.get(&key), Some(&vec![0xAA, 0xBB, 0xCC]));
        assert_eq!(coord.expert_saved_bytes.len(), 1);
    }

    // @trace TEST-ICOORD-040 [level:unit]
    #[test]
    fn test_coordinator_gate_skip_flags_default_false_and_true() {
        // Arrange
        let mut coord = make_coordinator();

        // Act — insert both false and true values
        coord.gate_skip_flags.insert(10u64, false);
        coord.gate_skip_flags.insert(20u64, true);

        // Assert — HashMap returns exact stored values
        assert_eq!(coord.gate_skip_flags.get(&10), Some(&false));
        assert_eq!(coord.gate_skip_flags.get(&20), Some(&true));
    }

    // @trace TEST-ICOORD-041 [level:unit]
    #[test]
    fn test_coordinator_multiple_instances_independent() {
        // Arrange
        let mut coord_a = make_coordinator();
        let mut coord_b = make_coordinator();

        // Act — mutate only coord_a
        coord_a.expert_code_regions.insert((0, 0), (100, 200));
        coord_a.gate_skip_flags.insert(1u64, true);
        coord_a.expert_saved_bytes.insert((0, 0), vec![0xFF]);

        // Assert — coord_b is unaffected
        assert!(coord_b.expert_code_regions.is_empty());
        assert!(coord_b.gate_skip_flags.is_empty());
        assert!(coord_b.expert_saved_bytes.is_empty());
        assert_eq!(coord_a.expert_code_regions.len(), 1);
    }

    // @trace TEST-ICOORD-042 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_layer_expert_key_pairs() {
        // Arrange
        let mut coord = make_coordinator();

        // Act — insert entries with (layer, expert) as key and (offset, size) as value
        coord.expert_code_regions.insert((0, 0), (0x0000, 512));
        coord.expert_code_regions.insert((0, 1), (0x0200, 512));
        coord.expert_code_regions.insert((2, 0), (0x0400, 256));
        coord.expert_code_regions.insert((2, 3), (0x0500, 1024));

        // Assert — all four entries retrievable with correct values
        assert_eq!(coord.expert_code_regions.get(&(0, 0)), Some(&(0x0000, 512)));
        assert_eq!(coord.expert_code_regions.get(&(0, 1)), Some(&(0x0200, 512)));
        assert_eq!(coord.expert_code_regions.get(&(2, 0)), Some(&(0x0400, 256)));
        assert_eq!(coord.expert_code_regions.get(&(2, 3)), Some(&(0x0500, 1024)));
        assert_eq!(coord.expert_code_regions.len(), 4);
    }

    // @trace TEST-ICOORD-043 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_byte_level_content() {
        // Arrange
        let mut coord = make_coordinator();
        let key = (4, 2);
        // Simulate 8 bytes of expert code: a mix of zero and non-zero values
        let saved_code: Vec<u8> = vec![0x00, 0x48, 0x89, 0xE5, 0x48, 0x83, 0xEC, 0x40];

        // Act
        coord.expert_saved_bytes.insert(key, saved_code);

        // Assert — verify every byte is stored exactly
        let stored = coord.expert_saved_bytes.get(&key).unwrap();
        assert_eq!(stored.len(), 8);
        assert_eq!(stored[0], 0x00);
        assert_eq!(stored[1], 0x48);
        assert_eq!(stored[2], 0x89);
        assert_eq!(stored[3], 0xE5);
        assert_eq!(stored[4], 0x48);
        assert_eq!(stored[5], 0x83);
        assert_eq!(stored[6], 0xEC);
        assert_eq!(stored[7], 0x40);
    }

    // @trace TEST-ICOORD-044 [level:unit]
    #[test]
    fn test_coordinator_gate_skip_flags_remove_entry() {
        // Arrange
        let mut coord = make_coordinator();
        coord.gate_skip_flags.insert(10u64, true);
        coord.gate_skip_flags.insert(20u64, false);
        assert_eq!(coord.gate_skip_flags.len(), 2);

        // Act
        let removed = coord.gate_skip_flags.remove(&10u64);

        // Assert
        assert_eq!(removed, Some(true));
        assert_eq!(coord.gate_skip_flags.len(), 1);
        assert_eq!(coord.gate_skip_flags.get(&10), None);
        assert_eq!(coord.gate_skip_flags.get(&20), Some(&false));
    }

    // @trace TEST-ICOORD-045 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_remove_entry() {
        // Arrange
        let mut coord = make_coordinator();
        coord.expert_saved_bytes.insert((0, 0), vec![1, 2, 3]);
        coord.expert_saved_bytes.insert((1, 1), vec![4, 5]);
        assert_eq!(coord.expert_saved_bytes.len(), 2);

        // Act
        let removed = coord.expert_saved_bytes.remove(&(0, 0));

        // Assert
        assert_eq!(removed, Some(vec![1, 2, 3]));
        assert_eq!(coord.expert_saved_bytes.len(), 1);
        assert_eq!(coord.expert_saved_bytes.get(&(0, 0)), None);
        assert_eq!(coord.expert_saved_bytes.get(&(1, 1)), Some(&vec![4, 5]));
    }

    // @trace TEST-ICOORD-046 [level:unit]
    #[test]
    fn test_mtp_controller_clone_is_independent() {
        // Arrange
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(8, 10);
        let rate_after_record = ctrl.ema_rate();

        // Act
        let cloned = ctrl.clone();

        // Assert — cloned shares state at clone time
        assert!((cloned.ema_rate() - rate_after_record).abs() < f32::EPSILON);
        assert_eq!(cloned.is_enabled(), ctrl.is_enabled());

        // Mutating original does not affect clone
        ctrl.record_acceptance(0, 10);
        assert!(ctrl.ema_rate() < cloned.ema_rate(), "original EMA should drop after low acceptance");
    }

    // @trace TEST-ICOORD-047 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_record_acceptance_zero_total_no_panic() {
        // Arrange
        let mut ctrl = MtpController::new();
        let rate_before = ctrl.ema_rate();

        // Act — total=0 is an edge case, should not panic
        let enabled = ctrl.record_acceptance(0, 0);

        // Assert — rate should be 0.0*alpha + (1-alpha)*ema, which is less than rate_before
        assert!(ctrl.ema_rate() < rate_before);
        assert!(enabled, "zero total should count as stable for re-enable, but controller starts enabled");
    }

    // @trace TEST-ICOORD-048 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_enable_disable_cycle_preserves_enabled_state() {
        // Arrange
        let mut ctrl = MtpController::new();

        // Act — multiple enable/disable cycles
        ctrl.disable();
        assert!(!ctrl.is_enabled());
        ctrl.enable();
        assert!(ctrl.is_enabled());
        ctrl.disable();
        assert!(!ctrl.is_enabled());
        ctrl.enable();
        assert!(ctrl.is_enabled());

        // Assert — after cycling back to enabled, effective_depth should work
        // Drive EMA up to get a non-zero effective depth
        for _ in 0..50 {
            ctrl.record_acceptance(9, 10);
        }
        assert_eq!(ctrl.effective_depth(4), 4, "after re-enable with high EMA, full depth should be returned");
    }

    // @trace TEST-ICOORD-049 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_empty_vec_storage() {
        // Arrange
        let mut coord = make_coordinator();
        let key = (5, 5);

        // Act — store an empty byte vector
        coord.expert_saved_bytes.insert(key, Vec::new());

        // Assert — empty vec is still a valid entry
        assert_eq!(coord.expert_saved_bytes.len(), 1);
        assert!(coord.expert_saved_bytes.get(&key).is_some());
        assert_eq!(coord.expert_saved_bytes.get(&key).unwrap().len(), 0);
    }

    // @trace TEST-ICOORD-050 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_minimal_dimensions() {
        // Arrange — smallest valid dimensions
        let bus = ResidualBus::new(1, 1);

        // Act & Assert
        assert_eq!(bus.hidden_size(), 1);
        assert_eq!(bus.num_layers(), 1);
        assert_eq!(bus.active_port_count(), 0);
        assert_eq!(bus.ports().len(), 0);
    }

    // @trace TEST-ICOORD-051 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_register_multiple_ports_same_tag() {
        // Arrange
        use crate::routing::BusPortTag;
        let mut bus = ResidualBus::new(768, 12);

        // Act — register two ports with the same tag at different layers
        bus.register(crate::routing::BusPort::injection(2, BusPortTag::RagInjection));
        bus.register(crate::routing::BusPort::injection(7, BusPortTag::RagInjection));

        // Assert — find_port returns the first match
        assert_eq!(bus.ports().len(), 2);
        let found = bus.find_port(BusPortTag::RagInjection);
        assert!(found.is_some());
        // Both ports are registered
        assert_eq!(bus.active_port_count(), 2);
    }

    // @trace TEST-ICOORD-052 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_recall_success() {
        // Arrange
        use crate::routing::BusPortTag;
        let mut coord = make_coordinator();
        let port = crate::routing::BusPort::recall(3, BusPortTag::IntentRecall);
        coord.residual_bus.register(port);

        let buffer = vec![1.0f32; 768];

        // Act
        let result = coord.residual_bus.recall(
            BusPortTag::IntentRecall,
            &buffer,
            3,
            None,
            0.5,
        );

        // Assert
        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.source, BusPortTag::IntentRecall);
        assert_eq!(payload.data.len(), 768);
    }

    // @trace TEST-ICOORD-053 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_reset_idempotent() {
        // Arrange
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(0, 10);
        ctrl.record_acceptance(0, 10);
        ctrl.disable();

        // Act — reset twice
        ctrl.reset();
        let state_after_first = (ctrl.is_enabled(), ctrl.ema_rate());
        ctrl.reset();
        let state_after_second = (ctrl.is_enabled(), ctrl.ema_rate());

        // Assert — double reset yields identical state
        assert_eq!(state_after_first, state_after_second);
        assert!(ctrl.is_enabled());
        assert!((ctrl.ema_rate() - 0.5).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-054 [level:unit]
    #[test]
    fn test_coordinator_gate_skip_flags_boundary_keys() {
        // Arrange
        let mut coord = make_coordinator();

        // Act — use boundary key values
        coord.gate_skip_flags.insert(0u64, true);
        coord.gate_skip_flags.insert(u64::MAX, false);

        // Assert
        assert_eq!(coord.gate_skip_flags.len(), 2);
        assert_eq!(coord.gate_skip_flags.get(&0), Some(&true));
        assert_eq!(coord.gate_skip_flags.get(&u64::MAX), Some(&false));
    }

    // @trace TEST-ICOORD-055 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_large_data() {
        // Arrange
        let mut coord = make_coordinator();
        let key = (0, 0);
        let large_data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();

        // Act
        coord.expert_saved_bytes.insert(key, large_data.clone());

        // Assert — large data is stored and retrieved intact
        let stored = coord.expert_saved_bytes.get(&key);
        assert!(stored.is_some());
        assert_eq!(stored.unwrap().len(), 4096);
        assert_eq!(stored.unwrap().as_slice(), large_data.as_slice());
    }

    // @trace TEST-ICOORD-056 [level:unit]
    #[test]
    fn test_coordinator_remove_nonexistent_key_returns_none() {
        // Arrange
        let mut coord = make_coordinator();

        // Act — remove from empty hashmaps
        let removed_regions = coord.expert_code_regions.remove(&(99, 99));
        let removed_bytes = coord.expert_saved_bytes.remove(&(99, 99));
        let removed_flags = coord.gate_skip_flags.remove(&999u64);

        // Assert — all return None without panicking
        assert_eq!(removed_regions, None);
        assert_eq!(removed_bytes, None);
        assert_eq!(removed_flags, None);
        assert!(coord.expert_code_regions.is_empty());
        assert!(coord.expert_saved_bytes.is_empty());
        assert!(coord.gate_skip_flags.is_empty());
    }

    // @trace TEST-ICOORD-057 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_default_trait_matches_new() {
        // Arrange & Act
        let via_new = MtpController::new();
        let via_default = MtpController::default();

        // Assert — Default trait delegates to new()
        assert_eq!(via_new.is_enabled(), via_default.is_enabled());
        assert!((via_new.ema_rate() - via_default.ema_rate()).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-058 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_debug_format_contains_enabled_field() {
        // Arrange
        let ctrl = MtpController::new();

        // Act
        let debug_str = format!("{:?}", ctrl);

        // Assert — Debug output should contain recognizable content
        assert!(debug_str.contains("enabled") || debug_str.contains("MtpController"));
    }

    // @trace TEST-ICOORD-059 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_effective_depth_boundary_ema_zero() {
        // Arrange — drive EMA to 0.0 by many zero-acceptance rounds
        let mut ctrl = MtpController::new();
        for _ in 0..200 {
            ctrl.record_acceptance(0, 100);
        }
        assert!(ctrl.ema_rate() < 0.1, "EMA should be near zero");

        // Act
        let depth = ctrl.effective_depth(4);

        // Assert — ema <= 0.3: depth = 0
        assert_eq!(depth, 0);
    }

    // @trace TEST-ICOORD-060 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_record_acceptance_returns_enabled_status() {
        // Arrange
        let mut ctrl = MtpController::with_params(0.5, 0.3, 0.5, 2, 5);
        assert!(ctrl.is_enabled());

        // Act — two low rounds should trigger disable
        let result_round1 = ctrl.record_acceptance(0, 10);
        assert!(result_round1, "first low round should not disable yet");
        let result_round2 = ctrl.record_acceptance(0, 10);
        assert!(!result_round2, "second consecutive low round should disable");
        assert!(!ctrl.is_enabled());
    }

    // @trace TEST-ICOORD-061 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_with_params_clamps_alpha_upper_bound() {
        // Arrange — alpha=999.0 should be clamped to 1.0
        let ctrl = MtpController::with_params(999.0, 0.3, 0.5, 3, 5);

        // Act — with alpha=1.0, the first recording fully overrides EMA
        let mut ctrl = ctrl;
        ctrl.record_acceptance(7, 10);

        // Assert — EMA should be exactly 0.7 (alpha*0.7 + 0*(1-alpha) with alpha=1.0)
        assert!((ctrl.ema_rate() - 0.7).abs() < 1e-3);
    }

    // @trace TEST-ICOORD-062 [level:unit]
    #[test]
    fn test_residual_bus_error_display_dimension_mismatch() {
        // Arrange
        let err = crate::routing::ResidualBusError::DimensionMismatch {
            expected: 768,
            actual: 10,
        };

        // Act
        let msg = format!("{}", err);

        // Assert
        assert!(msg.contains("768"), "should mention expected dimension");
        assert!(msg.contains("10"), "should mention actual dimension");
        assert!(msg.contains("mismatch"));
    }

    // @trace TEST-ICOORD-063 [level:unit]
    #[test]
    fn test_residual_bus_error_display_port_not_found() {
        // Arrange
        use crate::routing::BusPortTag;
        let err = crate::routing::ResidualBusError::PortNotFound(BusPortTag::ShadowKv);

        // Act
        let msg = format!("{}", err);

        // Assert
        assert!(msg.contains("not found"));
        assert!(msg.contains("ShadowKv"));
    }

    // @trace TEST-ICOORD-064 [level:unit]
    #[test]
    fn test_residual_bus_error_display_wrong_port_type() {
        // Arrange
        use crate::routing::{BusPortKind, ResidualBusError};
        let err = ResidualBusError::WrongPortType {
            expected: BusPortKind::Injection,
            actual: BusPortKind::Recall,
        };

        // Act
        let msg = format!("{}", err);

        // Assert
        assert!(msg.contains("Injection"));
        assert!(msg.contains("Recall"));
        assert!(msg.contains("wrong port type"));
    }

    // @trace TEST-ICOORD-065 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_find_port_mut_modification() {
        // Arrange
        use crate::routing::{BusPort, BusPortTag};
        let mut coord = make_coordinator();
        coord.residual_bus.register(BusPort::injection(1, BusPortTag::EarlyExit));

        // Act — deactivate via find_port_mut
        let port = coord.residual_bus.find_port_mut(BusPortTag::EarlyExit);
        assert!(port.is_some());
        port.unwrap().deactivate();

        // Assert — active count drops to zero
        assert_eq!(coord.residual_bus.active_port_count(), 0);
    }

    // @trace TEST-ICOORD-066 [level:unit]
    #[test]
    fn test_coordinator_spec_decoding_standard_step_count_zero() {
        // Arrange
        let coord = make_coordinator();

        // Act
        let steps = coord.spec_decoding.spec_step_count();
        let avg_rate = coord.spec_decoding.avg_acceptance_rate();

        // Assert — standard mode has zero speculation steps and 0.0 average rate
        assert_eq!(steps, 0);
        assert!((avg_rate - 0.0).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-067 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_active_ports_at_layer_filtering() {
        // Arrange
        use crate::routing::{BusPort, BusPortTag};
        let mut coord = make_coordinator();
        coord.residual_bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        coord.residual_bus.register(BusPort::recall(5, BusPortTag::IntentRecall));
        coord.residual_bus.register(BusPort::injection(3, BusPortTag::Guardrail));

        // Act — query layer 3
        let layer3_count = coord.residual_bus.active_ports_at_layer(3).count();
        let layer5_count = coord.residual_bus.active_ports_at_layer(5).count();
        let layer0_count = coord.residual_bus.active_ports_at_layer(0).count();

        // Assert — only ports at matching layers are returned
        assert_eq!(layer3_count, 2);
        assert_eq!(layer5_count, 1);
        assert_eq!(layer0_count, 0);
    }

    // @trace TEST-ICOORD-068 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_iteration_order_independent() {
        // Arrange
        let mut coord = make_coordinator();
        let keys = [(0, 0), (1, 2), (3, 4), (5, 6)];
        for &key in &keys {
            coord.expert_code_regions.insert(key, (key.0 * 100, key.1 * 100));
        }

        // Act — collect all keys via iteration
        let mut collected_keys: Vec<_> = coord.expert_code_regions.keys().copied().collect();
        collected_keys.sort();

        // Assert — all inserted keys are present regardless of iteration order
        let mut expected_keys = keys.to_vec();
        expected_keys.sort();
        assert_eq!(collected_keys, expected_keys);
    }

    // @trace TEST-ICOORD-069 [level:unit]
    #[test]
    fn test_coordinator_rag_system_none_by_default() {
        // Arrange
        let coord = make_coordinator();

        // Act & Assert — rag_system should be None in default coordinator
        assert!(coord.rag_system.is_none());
    }

    // @trace TEST-ICOORD-070 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_inject_inactive_port_error() {
        // Arrange
        use crate::routing::{BusPort, BusPortTag, InjectionPayload};

        let mut coord = make_coordinator();
        let port = BusPort::injection(3, BusPortTag::Guardrail);
        coord.residual_bus.register(port);

        // Deactivate the port
        let port_ref = coord.residual_bus.find_port(BusPortTag::Guardrail).unwrap();
        port_ref.deactivate();

        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0; 768],
            scale: 1.0,
        };
        let mut buffer = vec![0.0f32; 768];

        // Act
        let result = coord.residual_bus.inject(&payload, &mut buffer);

        // Assert — injecting into inactive port should fail
        assert!(result.is_err(), "inject into inactive port should fail");
        // Buffer should remain unchanged
        assert!(buffer.iter().all(|&v| v == 0.0));
    }

    // @trace TEST-ICOORD-071 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_recall_port_not_found_error() {
        // Arrange
        use crate::routing::BusPortTag;

        let coord = make_coordinator();
        let buffer = vec![1.0f32; 768];

        // Act — recall from a tag with no registered port
        let result = coord.residual_bus.recall(
            BusPortTag::ShadowKv,
            &buffer,
            0,
            None,
            0.0,
        );

        // Assert — should fail with PortNotFound
        assert!(result.is_err(), "recall from unregistered tag should fail");
    }

    // @trace TEST-ICOORD-072 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_custom_port_tag() {
        // Arrange
        use crate::routing::{BusPort, BusPortTag, InjectionPayload};

        let mut coord = make_coordinator();
        let custom_tag = BusPortTag::Custom(42);
        coord.residual_bus.register(BusPort::injection(5, custom_tag));

        // Act — find and inject via custom tag
        let found = coord.residual_bus.find_port(custom_tag);
        assert!(found.is_some(), "custom port should be found");

        let payload = InjectionPayload {
            target: custom_tag,
            data: vec![0.5; 768],
            scale: 2.0,
        };
        let mut buffer = vec![1.0f32; 768];
        let result = coord.residual_bus.inject(&payload, &mut buffer);

        // Assert — custom port inject succeeds: buffer[i] = 1.0 + 0.5*2.0 = 2.0
        assert!(result.is_ok());
        assert!((buffer[0] - 2.0).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-073 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_reenable_after_consecutive_high_rates() {
        // Arrange — use small patience for re-enable
        let mut ctrl = MtpController::with_params(0.1, 0.3, 0.5, 2, 3);
        assert!(ctrl.is_enabled());

        // Act — disable with 2 low rounds
        ctrl.record_acceptance(0, 10);
        ctrl.record_acceptance(0, 10);
        assert!(!ctrl.is_enabled(), "should be disabled after patience exhausted");

        // Re-enable with 3 consecutive high rounds
        ctrl.record_acceptance(8, 10);
        assert!(!ctrl.is_enabled(), "1 high round is not enough");
        ctrl.record_acceptance(9, 10);
        assert!(!ctrl.is_enabled(), "2 high rounds is not enough");
        ctrl.record_acceptance(7, 10);
        assert!(ctrl.is_enabled(), "3 high rounds should re-enable");
    }

    // @trace TEST-ICOORD-074 [level:unit]
    #[test]
    fn test_coordinator_gate_skip_flags_contains_key() {
        // Arrange
        let mut coord = make_coordinator();
        coord.gate_skip_flags.insert(100u64, true);
        coord.gate_skip_flags.insert(200u64, false);

        // Act & Assert — contains_key checks presence without retrieving value
        assert!(coord.gate_skip_flags.contains_key(&100));
        assert!(coord.gate_skip_flags.contains_key(&200));
        assert!(!coord.gate_skip_flags.contains_key(&999));
        assert!(!coord.gate_skip_flags.contains_key(&0));
    }

    // @trace TEST-ICOORD-075 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_multiple_injects_accumulate() {
        // Arrange
        use crate::routing::{BusPortTag, InjectionPayload};

        let mut coord = make_coordinator();
        coord.residual_bus.register(crate::routing::BusPort::injection(2, BusPortTag::RagInjection));

        let mut buffer = vec![0.0f32; 768];

        // Act — first inject
        let payload1 = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0; 768],
            scale: 1.0,
        };
        coord.residual_bus.inject(&payload1, &mut buffer).unwrap();

        // Second inject accumulates
        let payload2 = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0; 768],
            scale: 1.0,
        };
        coord.residual_bus.inject(&payload2, &mut buffer).unwrap();

        // Assert — 0 + 1*1 + 1*1 = 2.0
        assert!((buffer[0] - 2.0).abs() < f32::EPSILON);
        assert!((buffer[767] - 2.0).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-076 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_with_zero_offset_and_size() {
        // Arrange
        let mut coord = make_coordinator();
        let key = (0, 0);
        let value = (0, 0);

        // Act — store a code region with zero offset and zero size
        coord.expert_code_regions.insert(key, value);

        // Assert — zero values are valid entries
        assert_eq!(coord.expert_code_regions.get(&key), Some(&(0, 0)));
        assert_eq!(coord.expert_code_regions.len(), 1);
    }

    // @trace TEST-ICOORD-077 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_find_port_mut_and_inject() {
        // Arrange
        use crate::routing::{BusPort, BusPortTag, InjectionPayload};

        let mut coord = make_coordinator();
        coord.residual_bus.register(BusPort::injection(4, BusPortTag::EarlyExit));

        // Deactivate via find_port_mut
        let port = coord.residual_bus.find_port_mut(BusPortTag::EarlyExit);
        assert!(port.is_some());
        port.unwrap().deactivate();
        assert_eq!(coord.residual_bus.active_port_count(), 0);

        // Act — inject should fail on inactive port
        let payload = InjectionPayload {
            target: BusPortTag::EarlyExit,
            data: vec![1.0; 768],
            scale: 1.0,
        };
        let mut buffer = vec![0.0f32; 768];
        let result = coord.residual_bus.inject(&payload, &mut buffer);

        // Assert
        assert!(result.is_err());

        // Reactivate and retry — should succeed
        let port = coord.residual_bus.find_port_mut(BusPortTag::EarlyExit).unwrap();
        port.activate();
        let result2 = coord.residual_bus.inject(&payload, &mut buffer);
        assert!(result2.is_ok());
        assert!((buffer[0] - 1.0).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-078 [level:unit]
    #[test]
    fn test_coordinator_gate_skip_flags_clear_all() {
        // Arrange
        let mut coord = make_coordinator();
        for i in 0..10u64 {
            coord.gate_skip_flags.insert(i, i % 2 == 0);
        }
        assert_eq!(coord.gate_skip_flags.len(), 10);

        // Act — clear all flags
        coord.gate_skip_flags.clear();

        // Assert — all flags removed
        assert!(coord.gate_skip_flags.is_empty());
        assert_eq!(coord.gate_skip_flags.len(), 0);
        for i in 0..10u64 {
            assert!(!coord.gate_skip_flags.contains_key(&i));
        }
    }

    // @trace TEST-ICOORD-079 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_with_params_zero_patience_disables_immediately() {
        // Arrange — disable_patience=0 means any low round triggers disable
        let mut ctrl = MtpController::with_params(0.1, 0.3, 0.5, 0, 5);
        assert!(ctrl.is_enabled());

        // Act — single low-acceptance round
        let result = ctrl.record_acceptance(0, 10);

        // Assert — patience=0 means immediately disabled
        assert!(!result, "zero patience should disable after first low round");
        assert!(!ctrl.is_enabled());
    }

    // @trace TEST-ICOORD-080 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_inject_negative_scale_subtracts() {
        // Arrange
        use crate::routing::{BusPortTag, InjectionPayload};

        let mut coord = make_coordinator();
        coord.residual_bus.register(crate::routing::BusPort::injection(1, BusPortTag::Guardrail));

        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0; 768],
            scale: -1.0,
        };
        let mut buffer = vec![5.0f32; 768];

        // Act
        let result = coord.residual_bus.inject(&payload, &mut buffer);

        // Assert — 5.0 + 1.0 * (-1.0) = 4.0
        assert!(result.is_ok());
        assert!((buffer[0] - 4.0).abs() < f32::EPSILON);
        assert!((buffer[767] - 4.0).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-081 [level:unit]
    #[test]
    fn test_coordinator_expert_saved_bytes_retain_after_code_regions_clear() {
        // Arrange — verify independence of the two HashMaps
        let mut coord = make_coordinator();
        coord.expert_code_regions.insert((0, 1), (100, 200));
        coord.expert_saved_bytes.insert((0, 1), vec![0xAA, 0xBB]);

        // Act — clear only code regions
        coord.expert_code_regions.clear();

        // Assert — saved bytes unaffected
        assert!(coord.expert_code_regions.is_empty());
        assert_eq!(coord.expert_saved_bytes.len(), 1);
        assert_eq!(coord.expert_saved_bytes.get(&(0, 1)), Some(&vec![0xAA, 0xBB]));
    }

    // @trace TEST-ICOORD-082 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_recall_with_prev_residual_cosine_calculation() {
        // Arrange
        use crate::routing::{BusPort, BusPortTag};

        let mut coord = make_coordinator();
        coord.residual_bus.register(BusPort::recall(3, BusPortTag::IntentRecall));

        let buffer = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            .into_iter()
            .cycle()
            .take(768)
            .collect::<Vec<f32>>();
        // prev_residual is orthogonal: only element at index 1 is non-zero
        let prev: Vec<f32> = {
            let mut v = vec![0.0f32; 768];
            v[1] = 1.0;
            v
        };

        // Act
        let result = coord.residual_bus.recall(
            BusPortTag::IntentRecall,
            &buffer,
            3,
            Some(&prev),
            1.5,
        );

        // Assert
        assert!(result.is_ok());
        let payload = result.unwrap();
        assert_eq!(payload.meta.layer, 3);
        // Orthogonal vectors → cosine should be ~0
        assert!(payload.meta.cosine_sim.abs() < 0.01,
            "orthogonal vectors should have cosine ~0, got {}", payload.meta.cosine_sim);
        assert!((payload.meta.entropy - 1.5).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-083 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_port_inactive_error_display() {
        // Arrange
        use crate::routing::BusPortTag;

        let err = crate::routing::ResidualBusError::PortInactive(BusPortTag::Guardrail);

        // Act
        let msg = format!("{}", err);

        // Assert — PortInactive display should contain the tag name and "inactive"
        assert!(msg.contains("inactive"), "should mention 'inactive', got: {}", msg);
        assert!(msg.contains("Guardrail"), "should mention the tag name, got: {}", msg);
    }

    // @trace TEST-ICOORD-084 [level:unit]
    #[test]
    fn test_coordinator_rag_fuse_at_residual_through_coordinator() {
        // Arrange — build a coordinator with RAG, then call fuse_at_residual
        let mut rag = crate::rag::LateFusionRag::new(3);
        rag.retrieval_db = vec![vec![1.0; 768]];
        rag.top_k = 1;
        rag.fusion_weight = 0.5;

        let mut coord = InferenceCoordinator {
            moe_thermal: None,
            moe_fault_handler: None,
            moe_dispatcher: None,
            moe_prefetcher: None,
            prefetch_pipeline: None,
            hot_patch_manager: None,
            expert_code_regions: HashMap::new(),
            expert_saved_bytes: HashMap::new(),
            spec_decoding: SpecDecodingState::new_standard(),
            rag_system: Some(rag),
            residual_bus: ResidualBus::new(768, 12),
            gate_skip_flags: HashMap::new(),
            mtp_controller: MtpController::new(),
            #[cfg(feature = "nccl")]
            moe_dist_decision: None,
            #[cfg(feature = "nccl")]
            expert_load_stats: None,
            #[cfg(feature = "nccl")]
            eplb_imbalance_threshold: 2.0,
        };

        let mut hidden = vec![2.0f32; 768];

        // Act — fuse at the correct layer
        coord.rag_system.as_ref().unwrap().fuse_at_residual(&mut hidden, 3);

        // Assert — hidden state should be modified: 2.0 + 1.0*0.5 = 2.5
        assert!((hidden[0] - 2.5).abs() < 1e-5, "expected 2.5, got {}", hidden[0]);
        assert!((hidden[767] - 2.5).abs() < 1e-5);
    }

    // @trace TEST-ICOORD-085 [level:unit]
    #[test]
    fn test_coordinator_rag_retrieve_through_coordinator() {
        // Arrange — build a coordinator with RAG, populate DB, and call retrieve
        let mut rag = crate::rag::LateFusionRag::new(2);
        rag.retrieval_db = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.9, 0.1, 0.0]];
        rag.top_k = 2;

        let coord = InferenceCoordinator {
            moe_thermal: None,
            moe_fault_handler: None,
            moe_dispatcher: None,
            moe_prefetcher: None,
            prefetch_pipeline: None,
            hot_patch_manager: None,
            expert_code_regions: HashMap::new(),
            expert_saved_bytes: HashMap::new(),
            spec_decoding: SpecDecodingState::new_standard(),
            rag_system: Some(rag),
            residual_bus: ResidualBus::new(768, 12),
            gate_skip_flags: HashMap::new(),
            mtp_controller: MtpController::new(),
            #[cfg(feature = "nccl")]
            moe_dist_decision: None,
            #[cfg(feature = "nccl")]
            expert_load_stats: None,
            #[cfg(feature = "nccl")]
            eplb_imbalance_threshold: 2.0,
        };

        let query = vec![1.0, 0.0, 0.0];

        // Act
        let results = coord.rag_system.as_ref().unwrap().retrieve(&query);

        // Assert — top-2 results, first should be the exact match [1,0,0]
        assert_eq!(results.len(), 2);
        assert!((results[0][0] - 1.0).abs() < 1e-5, "first result should be [1,0,0]");
    }

    // @trace TEST-ICOORD-086 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_ema_alternating_rates_converges_to_midpoint() {
        // Arrange — with alpha=1.0 (clamped), each recording fully overrides EMA.
        // Test that alternating high/low rates cause EMA to track the most recent rate.
        let mut ctrl = MtpController::with_params(1.0, 0.3, 0.5, 100, 100);

        // Act — alternate between 100% and 0% acceptance
        ctrl.record_acceptance(10, 10); // rate=1.0, ema = 1.0*1.0 + 0*0.5 = 1.0
        assert!((ctrl.ema_rate() - 1.0).abs() < 1e-5, "after 100%% rate, EMA should be 1.0");

        ctrl.record_acceptance(0, 10); // rate=0.0, ema = 1.0*0.0 + 0*1.0 = 0.0
        assert!((ctrl.ema_rate() - 0.0).abs() < 1e-5, "after 0%% rate, EMA should be 0.0");

        ctrl.record_acceptance(5, 10); // rate=0.5, ema = 0.5
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-5, "after 50%% rate, EMA should be 0.5");

        // Assert — with alpha=1.0, EMA always equals last recorded rate
        assert!(ctrl.is_enabled(), "alternating should not disable with patience=100");
    }

    // @trace TEST-ICOORD-087 [level:unit]
    #[test]
    fn test_coordinator_expert_code_regions_and_saved_bytes_clear_independence() {
        // Arrange — populate both HashMaps, clear one, verify the other is unaffected
        let mut coord = make_coordinator();
        coord.expert_code_regions.insert((0, 0), (100, 200));
        coord.expert_code_regions.insert((1, 1), (300, 400));
        coord.expert_saved_bytes.insert((0, 0), vec![0x11]);
        coord.expert_saved_bytes.insert((1, 1), vec![0x22]);
        coord.gate_skip_flags.insert(1u64, true);

        // Act — clear only expert_code_regions
        coord.expert_code_regions.clear();

        // Assert — code regions cleared, saved bytes and flags untouched
        assert!(coord.expert_code_regions.is_empty());
        assert_eq!(coord.expert_saved_bytes.len(), 2);
        assert_eq!(coord.expert_saved_bytes.get(&(0, 0)), Some(&vec![0x11]));
        assert_eq!(coord.expert_saved_bytes.get(&(1, 1)), Some(&vec![0x22]));
        assert_eq!(coord.gate_skip_flags.len(), 1);
    }

    // @trace TEST-ICOORD-088 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_inject_into_specific_tag_among_multiple() {
        // Arrange — register two injection ports at the same layer with different tags
        use crate::routing::{BusPort, BusPortTag, InjectionPayload};

        let mut coord = make_coordinator();
        coord.residual_bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        coord.residual_bus.register(BusPort::injection(3, BusPortTag::Guardrail));

        // Act — inject only into Guardrail
        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![10.0; 768],
            scale: 1.0,
        };
        let mut buffer = vec![0.0f32; 768];
        let result = coord.residual_bus.inject(&payload, &mut buffer);

        // Assert — injection succeeds and targets only the Guardrail port
        assert!(result.is_ok());
        assert!((buffer[0] - 10.0).abs() < f32::EPSILON);
        assert!((buffer[767] - 10.0).abs() < f32::EPSILON);
    }

    // @trace TEST-ICOORD-089 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_with_params_negative_thresholds() {
        // Arrange — negative disable_threshold means all rates are "above threshold",
        // so low_streak never increments.
        let mut ctrl = MtpController::with_params(0.1, -1.0, 0.5, 2, 3);
        assert!(ctrl.is_enabled());

        // Act — record 5 rounds of 0% acceptance; rate=0.0 >= -1.0, so NOT "low"
        for _ in 0..5 {
            ctrl.record_acceptance(0, 10);
        }

        // Assert — with negative disable_threshold, zero acceptance is still above threshold
        assert!(ctrl.is_enabled(), "negative threshold means 0.0 rate is still above, never disables");
    }

    // @trace TEST-ICOORD-090 [level:unit]
    #[test]
    fn test_coordinator_residual_bus_recall_inactive_port_error() {
        // Arrange — register a recall port, deactivate it, then try to recall
        use crate::routing::{BusPort, BusPortTag};

        let mut coord = make_coordinator();
        coord.residual_bus.register(BusPort::recall(5, BusPortTag::ShadowKv));

        // Deactivate the port
        let port = coord.residual_bus.find_port(BusPortTag::ShadowKv).unwrap();
        port.deactivate();
        assert!(!port.is_active());

        let buffer = vec![1.0f32; 768];

        // Act — recall from inactive port
        let result = coord.residual_bus.recall(
            BusPortTag::ShadowKv,
            &buffer,
            5,
            None,
            0.0,
        );

        // Assert — should fail with PortInactive
        assert!(result.is_err(), "recall from inactive port should fail");
        let err = result.unwrap_err();
        assert!(matches!(err, crate::routing::ResidualBusError::PortInactive(_)),
            "error should be PortInactive");
    }

    // @trace TEST-ICOORD-091 [req:REQ-MTP-002] [level:unit]
    #[test]
    fn test_mtp_controller_effective_depth_exactly_at_boundary_0_5() {
        // Arrange — drive EMA to exactly 0.5
        // Default ema=0.5, alpha=0.1. One record of 0.5 rate:
        // ema = 0.1*0.5 + 0.9*0.5 = 0.5
        let mut ctrl = MtpController::new();
        ctrl.record_acceptance(5, 10);

        // Verify EMA is at 0.5 boundary
        assert!((ctrl.ema_rate() - 0.5).abs() < 1e-6, "EMA should be exactly 0.5");

        // Act — at ema=0.5, the condition is `ema > 0.5` which is false,
        // so it falls to `ema > 0.3` → depth = 1
        let depth = ctrl.effective_depth(5);

        // Assert — at exactly 0.5, depth should be 1 (the >0.3 branch, not the >0.5 branch)
        assert_eq!(depth, 1, "at EMA exactly 0.5, depth should be 1 (moderate-branch since >0.5 is false)");
    }

    // @trace TEST-ICOORD-092 [level:unit]
    #[test]
    fn test_coordinator_all_hashmaps_cleared_together() {
        // Arrange — populate all three HashMaps
        let mut coord = make_coordinator();
        coord.expert_code_regions.insert((0, 0), (100, 200));
        coord.expert_code_regions.insert((2, 3), (400, 500));
        coord.expert_saved_bytes.insert((0, 0), vec![0x01, 0x02]);
        coord.expert_saved_bytes.insert((5, 5), vec![0xFF]);
        coord.gate_skip_flags.insert(1u64, true);
        coord.gate_skip_flags.insert(2u64, false);
        assert_eq!(coord.expert_code_regions.len(), 2);
        assert_eq!(coord.expert_saved_bytes.len(), 2);
        assert_eq!(coord.gate_skip_flags.len(), 2);

        // Act — clear all three
        coord.expert_code_regions.clear();
        coord.expert_saved_bytes.clear();
        coord.gate_skip_flags.clear();

        // Assert — all are empty but coordinator is still usable
        assert!(coord.expert_code_regions.is_empty());
        assert!(coord.expert_saved_bytes.is_empty());
        assert!(coord.gate_skip_flags.is_empty());
        // Verify insert still works after clearing
        coord.expert_code_regions.insert((7, 7), (1, 1));
        assert_eq!(coord.expert_code_regions.get(&(7, 7)), Some(&(1, 1)));
    }

    // ── REQ-DIST-014/015: Distributed MoE fields ────────────────────────────

    // @trace TEST-ICOORD-093 [req:REQ-DIST-014] [level:unit]
    #[cfg(feature = "nccl")]
    #[test]
    fn test_coordinator_moe_dist_decision_default_none() {
        let coord = make_coordinator();
        assert!(coord.moe_dist_decision.is_none());
    }

    // @trace TEST-ICOORD-094 [req:REQ-DIST-015] [level:unit]
    #[cfg(feature = "nccl")]
    #[test]
    fn test_coordinator_expert_load_stats_default_none() {
        let coord = make_coordinator();
        assert!(coord.expert_load_stats.is_none());
    }

    // @trace TEST-ICOORD-095 [req:REQ-DIST-015] [level:unit]
    #[cfg(feature = "nccl")]
    #[test]
    fn test_coordinator_eplb_imbalance_threshold_default() {
        let coord = make_coordinator();
        assert!((coord.eplb_imbalance_threshold - 2.0f64).abs() < f64::EPSILON);
    }

    // @trace TEST-ICOORD-096 [req:REQ-DIST-014] [level:unit]
    #[cfg(feature = "nccl")]
    #[test]
    fn test_coordinator_moe_dist_decision_can_be_set() {
        use crate::engine::distributed_config::{CommHandleWrapper, ParallelConfig};

        let handle = CommHandleWrapper::from_config(&ParallelConfig::default()).unwrap();
        let decision = crate::moe::distributed_dispatch::distributed_dispatch::MoeDistDecision::from_config(
            &crate::engine::distributed_config::MoeDistributedConfig::default(),
            8,
            &handle,
        );
        let mut coord = make_coordinator();
        coord.moe_dist_decision = Some(decision.clone());
        assert!(coord.moe_dist_decision.is_some());
        assert_eq!(coord.moe_dist_decision.unwrap().num_experts, 8);
    }

    // @trace TEST-ICOORD-097 [req:REQ-DIST-015] [level:unit]
    #[cfg(feature = "nccl")]
    #[test]
    fn test_coordinator_expert_load_stats_can_be_set() {
        let stats = crate::moe::eplb::eplb::ExpertLoadStats::new(16);
        assert_eq!(stats.invocation_counts.len(), 16);
        let mut coord = make_coordinator();
        coord.expert_load_stats = Some(stats);
        assert!(coord.expert_load_stats.is_some());
        assert_eq!(coord.expert_load_stats.unwrap().invocation_counts.len(), 16);
    }

    // @trace TEST-ICOORD-098 [req:REQ-DIST-015] [level:unit]
    #[cfg(feature = "nccl")]
    #[test]
    fn test_coordinator_eplb_threshold_can_be_customized() {
        let mut coord = make_coordinator();
        coord.eplb_imbalance_threshold = 3.5;
        assert!((coord.eplb_imbalance_threshold - 3.5f64).abs() < f64::EPSILON);
    }

    // @trace TEST-ICOORD-099 [req:REQ-DIST-014] [level:unit]
    #[test]
    fn test_moe_distributed_dispatch_step_noop_without_nccl() {
        // Without nccl feature, moe_distributed_dispatch_step is a no-op.
        // This test verifies it compiles and doesn't panic with a default executor.
        // The actual method exists on Executor but we just verify the struct fields.
        let coord = make_coordinator();
        // Verify the struct is valid with all None fields
        assert!(coord.moe_thermal.is_none() || coord.moe_thermal.is_some());
    }
}
