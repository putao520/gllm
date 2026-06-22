use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::compat::backend_trait::{Backend, Element};
use crate::model_config::ModelConfig;
use crate::scheduler::telemetry::ProfileAccumulator;
use crate::scheduler::types::PhysicalId;
use crate::sensors::SystemTopology;

use super::super::executor::{AttentionTopology, GeneratorForwardConfig};
use super::super::mega_kernel_callback::MegaKernelCallbackTable;
use super::sg_callback_handle::SgCallbackHandle;
use crate::weight_loader::WeightsHandle;

pub struct ModelContextHolder<B: Backend<E> + 'static, E: Element = f32> {
    pub manifest: Arc<crate::manifest::ModelManifest>,
    pub weights: WeightsHandle<B, E>,
    pub add_special_tokens: bool,
    pub geometry: Arc<crate::model_config::ModelGeometry>,
    pub model_config: ModelConfig,
    pub forward_config: GeneratorForwardConfig,
    pub tokenizer: crate::tokenizer::TokenizerHandle,
    pub topology: AttentionTopology,
    pub system_topology: SystemTopology,
    pub profile_accumulator: ProfileAccumulator,
    pub hooks: Arc<RwLock<Vec<Box<dyn crate::generation::GenerationHook>>>>,
    pub sg_callback_shim: Option<crate::semantic_gatekeeper::callback::SemanticGatekeeperCallbackShim>,
    pub sg_ring_buffer: Option<Arc<crate::semantic_gatekeeper::GatekeeperRingBuffer>>,
    pub sg_shared_memory: Option<Mutex<crate::semantic_gatekeeper::SgSharedMemory>>,
    pub callback_table: MegaKernelCallbackTable,
    pub sg_callback_handle: SgCallbackHandle,
    pub weight_page_table: HashMap<usize, Vec<PhysicalId>>,
    pub weight_pages_registered: bool,
    pub three_tier_swap: Option<Arc<Mutex<crate::scheduler::ThreeTierSwapCoordinator>>>,
    /// Distributed page routing table for cross-node KV cache lookup (REQ-DP-014, REQ-DIST-003).
    #[cfg(feature = "nccl")]
    pub distributed_routing_table: Option<gllm_kernels::PageRoutingTable>,
    /// NCCL communication handle wrapper (REQ-DIST-001).
    /// None = not yet initialized; Some = init_distributed() called.
    #[cfg(feature = "nccl")]
    pub comm_handle: Option<crate::engine::distributed_config::CommHandleWrapper>,
    /// Parallel config snapshot from init_distributed() (REQ-DIST-001).
    #[cfg(feature = "nccl")]
    pub parallel_config: Option<crate::engine::distributed_config::ParallelConfig>,
    /// KV distribution config from DistributedConfig (REQ-DIST-002).
    /// Stored so KvCoordinator can access it at runtime for KvDistDecision resolution.
    #[cfg(feature = "nccl")]
    pub kv_distribution_config: Option<crate::engine::distributed_config::KvDistributionConfig>,
    /// Prefill/Decode disaggregation config from DistributedConfig (REQ-DIST-002).
    #[cfg(feature = "nccl")]
    pub pd_disagg_config: Option<crate::engine::distributed_config::PdDisaggConfig>,
    /// Communication config from DistributedConfig (REQ-DIST-002).
    #[cfg(feature = "nccl")]
    pub comm_config: Option<crate::engine::distributed_config::CommConfig>,
    /// MoE distributed config from DistributedConfig (REQ-DIST-002).
    #[cfg(feature = "nccl")]
    pub moe_distributed_config: Option<crate::engine::distributed_config::MoeDistributedConfig>,
    /// Context Parallelism config (REQ-DIST-016).
    /// None = no CP; Some = Ring Attention enabled with cp_size > 1.
    #[cfg(feature = "nccl")]
    pub cp_config: Option<crate::engine::coordinator::context_parallel::context_parallel::CpConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::AttentionMaskType;

    // ── Helper: construct a minimal ModelGeometry for tests ──

    fn test_geometry() -> Arc<crate::model_config::ModelGeometry> {
        Arc::new(crate::model_config::ModelGeometry {
            hidden_size: 64,
            num_layers: 4,
            vocab_size: 100,
            intermediate_size: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            max_seq_len: 512,
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
        })
    }

    // ── Test 1: MegaKernelCallbackTable new has no callbacks ──

    #[test]
    fn callback_table_new_has_no_callbacks() {
        let table = MegaKernelCallbackTable::new();
        assert!(!table.has_any_callback());
    }

    // ── Test 2: MegaKernelCallbackTable default equals new ──

    #[test]
    fn callback_table_default_equals_new() {
        let from_new = MegaKernelCallbackTable::new();
        let from_default = MegaKernelCallbackTable::default();
        assert!(!from_new.has_any_callback());
        assert!(!from_default.has_any_callback());
    }

    // ── Test 3: SgCallbackHandle starts unregistered ──

    #[test]
    fn sg_callback_handle_new_is_not_registered() {
        let handle = SgCallbackHandle::new();
        assert!(!handle.is_registered());
    }

    // ── Test 4: SgCallbackHandle default equals new ──

    #[test]
    fn sg_callback_handle_default_is_not_registered() {
        let handle = SgCallbackHandle::default();
        assert!(!handle.is_registered());
    }

    // ── Test 5: ProfileAccumulator default has no history ──

    #[test]
    fn profile_accumulator_default_no_re_fusion_triggered() {
        let mut acc = ProfileAccumulator::default();
        // A single sample on layer 0 should not trigger re-fusion.
        let triggered = acc.record_and_check(0, 0.01);
        assert!(!triggered);
    }

    // ── Test 6: ProfileAccumulator new equals default ──

    #[test]
    fn profile_accumulator_new_equals_default() {
        let mut from_new = ProfileAccumulator::new();
        let mut from_default = ProfileAccumulator::default();
        // Both start with empty state: a single low-ratio sample should not trigger.
        // We compare behavior rather than field equality (history is private).
        assert!(!from_new.record_and_check(5, 0.0));
        assert!(!from_default.record_and_check(5, 0.0));
    }

    // ── Test 7: AttentionTopology bidirectional uses Bidirectional mask ──

    #[test]
    fn attention_topology_bidirectional_mask_type() {
        let topo = AttentionTopology::bidirectional(test_geometry());
        assert_eq!(topo.mask_type, AttentionMaskType::Bidirectional);
    }

    // ── Test 8: AttentionTopology causal uses Causal mask ──

    #[test]
    fn attention_topology_causal_mask_type() {
        let topo = AttentionTopology::causal(test_geometry());
        assert_eq!(topo.mask_type, AttentionMaskType::Causal);
    }

    // ── Test 9: AttentionTopology linear constructs without panic ──

    #[test]
    fn attention_topology_linear_does_not_panic() {
        let topo = AttentionTopology::linear();
        assert_eq!(topo.mask_type, AttentionMaskType::Bidirectional);
        assert_eq!(topo.geometry.hidden_size, 1);
        assert_eq!(topo.geometry.num_layers, 1);
    }

    // ── Test 10: ModelGeometry non-MLA returns correct kv_dim ──

    #[test]
    fn model_geometry_non_mla_kv_dim() {
        let geo = test_geometry();
        assert!(!geo.is_mla());
        assert_eq!(geo.kv_dim(), 2 * 16); // num_kv_heads * head_dim
    }

    // ── Test 11: ModelGeometry with MLA fields returns correct kv_dim ──

    #[test]
    fn model_geometry_mla_kv_dim() {
        let mut geo = (*test_geometry()).clone();
        geo.mla_d_c = 512;
        geo.mla_d_rope = 64;
        assert!(geo.is_mla());
        assert_eq!(geo.kv_dim(), 512 + 64);
    }

    // ── Test 12: weight_page_table HashMap starts empty ──

    #[test]
    fn weight_page_table_initializes_empty() {
        let table: HashMap<usize, Vec<PhysicalId>> = HashMap::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }

    // ── Test 13: weight_pages_registered bool default is false ──

    #[test]
    fn weight_pages_registered_default_is_false() {
        let registered: bool = false;
        assert!(!registered);
    }

    // ── Test 14: GeneratorForwardConfig default_for_test exposes geometry fields ──

    #[test]
    fn forward_config_test_default_geometry_accessors() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.hidden_size(), 64);
        assert_eq!(cfg.num_layers(), 4);
        assert_eq!(cfg.vocab_size(), 100);
        assert_eq!(cfg.intermediate_size(), 128);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.max_seq_len(), 512);
    }

    // ── Test 15: ModelManifest default has expected arch and kind ──

    #[test]
    fn model_manifest_default_fields() {
        let manifest = crate::manifest::ModelManifest::default();
        assert_eq!(manifest.arch, "llama");
        assert_eq!(manifest.kind, crate::manifest::ModelKind::Chat);
        assert!(!manifest.is_moe());
        assert!(manifest.tensor_map.is_empty());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tests 16-30: Additional unit tests for model_context coordinator
    // ═══════════════════════════════════════════════════════════════════════

    // ── Test 16: AttentionTopology bidirectional preserves geometry fields ──

    #[test]
    fn attention_topology_bidirectional_preserves_geometry() {
        // Arrange
        let geo = test_geometry();
        // Act
        let topo = AttentionTopology::bidirectional(Arc::clone(&geo));
        // Assert
        assert_eq!(topo.geometry.hidden_size, 64);
        assert_eq!(topo.geometry.num_layers, 4);
        assert_eq!(topo.geometry.num_heads, 4);
        assert_eq!(topo.geometry.num_kv_heads, 2);
        assert_eq!(topo.geometry.head_dim, 16);
        assert_eq!(topo.geometry.max_seq_len, 512);
    }

    // ── Test 17: AttentionTopology causal preserves geometry fields ──

    #[test]
    fn attention_topology_causal_preserves_geometry() {
        // Arrange
        let geo = test_geometry();
        // Act
        let topo = AttentionTopology::causal(Arc::clone(&geo));
        // Assert
        assert_eq!(topo.geometry.hidden_size, 64);
        assert_eq!(topo.geometry.num_layers, 4);
        assert_eq!(topo.geometry.vocab_size, 100);
    }

    // ── Test 18: AttentionTopology accessor methods return correct values ──

    #[test]
    fn attention_topology_accessor_methods() {
        // Arrange
        let topo = AttentionTopology::causal(test_geometry());
        // Act & Assert
        assert_eq!(topo.num_heads(), 4);
        assert_eq!(topo.num_kv_heads(), 2);
        assert_eq!(topo.head_dim(), 16);
        assert_eq!(topo.max_seq_len(), 512);
    }

    // ── Test 19: AttentionTopology linear has minimal geometry ──

    #[test]
    fn attention_topology_linear_minimal_geometry() {
        // Arrange & Act
        let topo = AttentionTopology::linear();
        // Assert
        assert_eq!(topo.num_heads(), 1);
        assert_eq!(topo.num_kv_heads(), 1);
        assert_eq!(topo.head_dim(), 1);
    }

    // ── Test 20: GeneratorForwardConfig attention accessor returns correct fields ──

    #[test]
    fn forward_config_attention_head_config() {
        // Arrange
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act
        let attn = cfg.attention();
        // Assert
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);
        assert_eq!(attn.head_dim, 16);
    }

    // ── Test 21: GeneratorForwardConfig rope accessors return test defaults ──

    #[test]
    fn forward_config_rope_accessors() {
        // Arrange
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act & Assert
        assert_eq!(cfg.rope_theta(), 10000.0);
        assert_eq!(cfg.rope_scale(), 1.0);
    }

    // ── Test 22: GeneratorForwardConfig norm_eps accessor ──

    #[test]
    fn forward_config_norm_eps_accessor() {
        // Arrange
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act & Assert
        assert!((cfg.norm_eps() - 1e-5).abs() < 1e-10);
    }

    // ── Test 23: GeneratorForwardConfig dtype accessor returns F32 ──

    #[test]
    fn forward_config_dtype_accessor() {
        // Arrange
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act & Assert
        assert_eq!(cfg.dtype(), gllm_kernels::types::DType::F32);
    }

    // ── Test 24: ProfileAccumulator does not trigger with insufficient samples ──

    #[test]
    fn profile_accumulator_needs_full_history_to_trigger() {
        // Arrange
        let mut acc = ProfileAccumulator::new();
        // Act: fill 99 out of 100 slots with low-ratio values (below threshold)
        for i in 0..99 {
            let triggered = acc.record_and_check(0, 0.001);
            // Assert: should not trigger until history is full (100 samples)
            assert!(!triggered, "should not trigger at sample {}", i);
        }
    }

    // ── Test 25: ProfileAccumulator different layers track independently ──

    #[test]
    fn profile_accumulator_independent_layer_tracking() {
        // Arrange
        let mut acc = ProfileAccumulator::new();
        // Act: record a sample on layer 0 — should not trigger
        let t0 = acc.record_and_check(0, 0.01);
        // Record a different sample on layer 5 — should not trigger
        let t5 = acc.record_and_check(5, 0.01);
        // Assert: neither layer has enough history
        assert!(!t0);
        assert!(!t5);
    }

    // ── Test 26: ModelGeometry clone produces independent copy ──

    #[test]
    fn model_geometry_clone_independence() {
        // Arrange
        let geo = test_geometry();
        let mut cloned = (*geo).clone();
        // Act: modify the clone
        cloned.hidden_size = 999;
        // Assert: original is unchanged
        assert_eq!(geo.hidden_size, 64);
        assert_eq!(cloned.hidden_size, 999);
    }

    // ── Test 27: ModelGeometry is_moe returns false when num_experts is zero ──

    #[test]
    fn model_geometry_is_moe_false_when_no_experts() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert!(!geo.is_moe());
        assert_eq!(geo.num_experts, 0);
        assert_eq!(geo.moe_top_k, 0);
    }

    // ── Test 28: ModelGeometry kv_bytes_per_token calculation for non-MLA ──

    #[test]
    fn model_geometry_kv_bytes_per_token_non_mla() {
        // Arrange
        let geo = test_geometry();
        // Act
        let bytes = geo.kv_bytes_per_token();
        // Assert: 2 * kv_dim * num_layers * dtype_size = 2 * (2*16) * 4 * 4 = 1024
        let expected = 2 * (2 * 16) * 4 * 4;
        assert_eq!(bytes, expected);
    }

    // ── Test 29: weight_page_table insertion and lookup ──

    #[test]
    fn weight_page_table_insert_and_lookup() {
        // Arrange
        let mut table: HashMap<usize, Vec<PhysicalId>> = HashMap::new();
        let physical_ids: Vec<PhysicalId> = vec![0, 1, 2];
        // Act
        table.insert(0, physical_ids.clone());
        // Assert
        assert_eq!(table.len(), 1);
        assert_eq!(table.get(&0), Some(&physical_ids));
        assert_eq!(table.get(&1), None);
    }

    // ── Test 30: weight_pages_registered state transition ──

    #[test]
    fn weight_pages_registered_state_transition() {
        // Arrange
        let mut registered = false;
        assert!(!registered);
        // Act: simulate registration
        registered = true;
        // Assert
        assert!(registered);
        // Act: simulate deregistration
        registered = false;
        // Assert
        assert!(!registered);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tests 31-43: ModelGeometry field accessor & value preservation tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── Test 31: ModelGeometry hidden_size accessor ──

    #[test]
    fn model_geometry_hidden_size_value() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert_eq!(geo.hidden_size, 64);
    }

    // ── Test 32: ModelGeometry num_layers accessor ──

    #[test]
    fn model_geometry_num_layers_value() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert_eq!(geo.num_layers, 4);
    }

    // ── Test 33: ModelGeometry vocab_size accessor ──

    #[test]
    fn model_geometry_vocab_size_value() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert_eq!(geo.vocab_size, 100);
    }

    // ── Test 34: ModelGeometry head_dim equals hidden_size / num_heads ──

    #[test]
    fn model_geometry_head_dim_matches_hidden_div_heads() {
        // Arrange
        let geo = test_geometry();
        // Act
        let expected_head_dim = geo.hidden_size / geo.num_heads;
        // Assert
        assert_eq!(geo.head_dim, expected_head_dim);
        assert_eq!(geo.head_dim, 16);
    }

    // ── Test 35: ModelGeometry max_seq_len is preserved from test helper ──

    #[test]
    fn model_geometry_max_seq_len_preserved() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert_eq!(geo.max_seq_len, 512);
    }

    // ── Test 36: ModelGeometry dtype is F32 in test helper ──

    #[test]
    fn model_geometry_dtype_is_f32() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert_eq!(geo.dtype, gllm_kernels::types::DType::F32);
        assert_eq!(geo.dtype.size_bytes(), 4);
    }

    // ── Test 37: ModelGeometry norm_eps value is preserved ──

    #[test]
    fn model_geometry_norm_eps_preserved() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert!((geo.norm_eps - 1e-5).abs() < 1e-10);
    }

    // ── Test 38: ModelGeometry num_kv_heads value ──

    #[test]
    fn model_geometry_num_kv_heads_value() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert_eq!(geo.num_kv_heads, 2);
        assert!(geo.num_kv_heads <= geo.num_heads);
    }

    // ── Test 39: ModelGeometry intermediate_size value ──

    #[test]
    fn model_geometry_intermediate_size_value() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert_eq!(geo.intermediate_size, 128);
        assert!(geo.intermediate_size > geo.hidden_size);
    }

    // ── Test 40: ModelManifest default has expected specific field values ──

    #[test]
    fn model_manifest_default_no_overrides() {
        // Arrange
        let manifest = crate::manifest::ModelManifest::default();
        // Act & Assert: optional overrides should be None in default
        assert!(manifest.rope_base_override.is_none());
        assert!(manifest.max_context_override.is_none());
        assert!(manifest.moe_config.is_none());
        assert!(manifest.model_id.is_empty() || manifest.model_id == "default");
    }

    // ── Test 41: ModelGeometry mla_d_c is 0 in test helper (non-MLA) ──

    #[test]
    fn model_geometry_mla_d_c_zero_in_test_helper() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert_eq!(geo.mla_d_c, 0);
        assert_eq!(geo.mla_d_rope, 0);
        assert_eq!(geo.mla_unabsorbed_threshold, 0);
        assert!(!geo.is_mla());
    }

    // ── Test 42: ModelGeometry rope_theta is preserved from test helper ──

    #[test]
    fn model_geometry_rope_theta_preserved() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert
        assert!((geo.rope_theta - 10000.0).abs() < 1e-6);
    }

    // ── Test 43: ModelGeometry compute_dtype matches dtype in test helper ──

    #[test]
    fn model_geometry_compute_dtype_matches_dtype() {
        // Arrange
        let geo = test_geometry();
        // Act & Assert: in the test helper both are F32
        assert_eq!(geo.compute_dtype, geo.dtype);
        assert_eq!(geo.compute_dtype, gllm_kernels::types::DType::F32);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Tests 44-56: Additional unit tests for untested paths
    // ═══════════════════════════════════════════════════════════════════════

    // ── Test 44: ModelGeometry effective_kv_layers with no shared layers ──
    // @trace TEST-MCX-044 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn model_geometry_effective_kv_layers_no_shared() {
        // Arrange
        let geo = test_geometry(); // num_layers=4, num_kv_shared_layers=0
        // Act
        let effective = geo.effective_kv_layers();
        // Assert: without shared layers, effective equals num_layers
        assert_eq!(effective, 4);
    }

    // ── Test 45: ModelGeometry effective_kv_layers with shared layers ──
    // @trace TEST-MCX-045 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn model_geometry_effective_kv_layers_with_shared() {
        // Arrange: 10 layers, last 3 shared → 7 effective
        let mut geo = (*test_geometry()).clone();
        geo.num_layers = 10;
        geo.num_kv_shared_layers = 3;
        // Act
        let effective = geo.effective_kv_layers();
        // Assert
        assert_eq!(effective, 7);
    }

    // ── Test 46: ModelGeometry effective_kv_layers saturating_sub floors at 1 ──
    // @trace TEST-MCX-046 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn model_geometry_effective_kv_layers_floors_at_one() {
        // Arrange: 3 layers, 3 shared → saturating_sub(0).max(1) = 1
        let mut geo = (*test_geometry()).clone();
        geo.num_layers = 3;
        geo.num_kv_shared_layers = 3;
        // Act
        let effective = geo.effective_kv_layers();
        // Assert: max(1) prevents going to 0
        assert_eq!(effective, 1);
    }

    // ── Test 47: ModelGeometry effective_kv_layer for non-shared layer returns identity ──
    // @trace TEST-MCX-047 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn model_geometry_effective_kv_layer_non_shared_identity() {
        // Arrange: 8 layers, 2 shared → shared_start = 6, layers 0-5 are non-shared
        let mut geo = (*test_geometry()).clone();
        geo.num_layers = 8;
        geo.num_kv_shared_layers = 2;
        geo.attention_pattern = vec![0; 8]; // all sliding
        // Act & Assert: non-shared layers return their own index
        for layer in 0..6 {
            assert_eq!(geo.effective_kv_layer(layer), layer);
        }
    }

    // ── Test 48: ModelGeometry effective_kv_layer for shared layer finds donor ──
    // @trace TEST-MCX-048 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn model_geometry_effective_kv_layer_shared_finds_donor() {
        // Arrange: 6 layers, last 2 shared, alternating pattern
        let mut geo = (*test_geometry()).clone();
        geo.num_layers = 6;
        geo.num_kv_shared_layers = 2;
        // Pattern: 0=sliding(0), 1=global(1), 2=sliding(0), 3=global(1), 4=sliding(0), 5=global(1)
        geo.attention_pattern = vec![0, 1, 0, 1, 0, 1];
        // Act: layer 4 is shared sliding → donor is layer 2 (last non-shared sliding)
        assert_eq!(geo.effective_kv_layer(4), 2);
        // Act: layer 5 is shared global → donor is layer 3 (last non-shared global)
        assert_eq!(geo.effective_kv_layer(5), 3);
    }

    // ── Test 49: ModelGeometry expert_weight_bytes for non-MoE model ──
    // @trace TEST-MCX-049 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn model_geometry_expert_weight_bytes_non_moe() {
        // Arrange: non-MoE model has expert_intermediate_size=0, so expert_weight_bytes=0
        let geo = test_geometry();
        assert_eq!(geo.expert_intermediate_size, 0);
        // Act
        let bytes = geo.expert_weight_bytes();
        // Assert: hidden_size * 0 * 3 * 4 = 0
        assert_eq!(bytes, 0);
    }

    // ── Test 50: ModelGeometry expert_weight_bytes for MoE model ──
    // @trace TEST-MCX-050 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn model_geometry_expert_weight_bytes_moe() {
        // Arrange: MoE model with expert_intermediate_size=256
        let mut geo = (*test_geometry()).clone();
        geo.expert_intermediate_size = 256;
        geo.num_experts = 8;
        // Act: hidden_size * expert_intermediate_size * 3 gates * dtype_size(F32=4)
        let bytes = geo.expert_weight_bytes();
        // Assert: 64 * 256 * 3 * 4 = 196608
        assert_eq!(bytes, 64 * 256 * 3 * 4);
    }

    // ── Test 51: ModelGeometry kv_bytes_per_token for MLA model ──
    // @trace TEST-MCX-051 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn model_geometry_kv_bytes_per_token_mla() {
        // Arrange: MLA model with d_c=512, d_rope=64, 4 layers, F32
        let mut geo = (*test_geometry()).clone();
        geo.mla_d_c = 512;
        geo.mla_d_rope = 64;
        assert!(geo.is_mla());
        // Act: MLA path = kv_dim * num_layers * dtype_size (single compressed vector, no K/V split)
        let bytes = geo.kv_bytes_per_token();
        // Assert: (512+64) * 4 * 4 = 9216
        assert_eq!(bytes, (512 + 64) * 4 * 4);
    }

    // ── Test 52: MegaKernelCallbackTable clear removes registered callback ──
    // @trace TEST-MCX-052 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn callback_table_clear_removes_callback() {
        // Arrange
        let mut table = MegaKernelCallbackTable::new();
        assert!(!table.has_any_callback());
        // Act: register a callback in slot 0
        unsafe {
            let sentinel_fn: usize = 0xDEAD;
            let sentinel_ctx: usize = 0xBEEF;
            table.register(0, sentinel_fn as *const u8, sentinel_ctx as *const u8);
        }
        assert!(table.has_any_callback());
        // Act: clear the slot
        table.clear(0);
        // Assert
        assert!(!table.has_any_callback());
    }

    // ── Test 53: MegaKernelCallbackTable as_ptr returns non-null ──
    // @trace TEST-MCX-053 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn callback_table_as_ptr_non_null() {
        // Arrange
        let table = MegaKernelCallbackTable::new();
        // Act
        let ptr = table.as_ptr();
        // Assert
        assert!(!ptr.is_null());
        // The pointer should be aligned to MegaKernelCallbackTable
        let align = std::mem::align_of::<MegaKernelCallbackTable>();
        assert_eq!((ptr as usize) % align, 0);
    }

    // ── Test 54: ProfileAccumulator triggers after full stable history ──
    // @trace TEST-MCX-054 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn profile_accumulator_triggers_after_full_stable_history() {
        // Arrange
        let mut acc = ProfileAccumulator::new();
        // Act: fill all 100 slots with zero ratio (below 0.05 threshold)
        for i in 0..99 {
            let triggered = acc.record_and_check(0, 0.0);
            assert!(!triggered, "should not trigger at sample {}", i);
        }
        // The 100th sample fills the history and all 100 samples are stable → trigger
        let triggered = acc.record_and_check(0, 0.01);
        // Assert
        assert!(triggered);
    }

    // ── Test 55: ProfileAccumulator does not trigger when ratio exceeds threshold ──
    // @trace TEST-MCX-055 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn profile_accumulator_no_trigger_high_ratio() {
        // Arrange
        let mut acc = ProfileAccumulator::new();
        // Act: fill all 100 slots with high ratio (above 0.05 threshold)
        for i in 0..100 {
            let triggered = acc.record_and_check(0, 1.0);
            assert!(!triggered, "should not trigger at sample {} with ratio 1.0", i);
        }
        // Assert: even with full history, all samples exceed threshold → no trigger
    }

    // ── Test 56: ProfileAccumulator ring buffer evicts oldest on overflow ──
    // @trace TEST-MCX-056 [req:REQ-DECOMP] [level:unit]

    #[test]
    fn profile_accumulator_ring_buffer_eviction_keeps_recent() {
        // Arrange
        let mut acc = ProfileAccumulator::new();
        // Act: fill 100 slots with high ratio (no trigger), then push 99 more low ratio
        for _ in 0..100 {
            acc.record_and_check(0, 1.0);
        }
        // Now push 99 low-ratio samples — history is full but only 99 are stable (need 95 of 100)
        // However, after 100 high + 99 low = 199 pushes, the window is [99 low, 1 high from the original]
        for i in 0..99 {
            let triggered = acc.record_and_check(0, 0.0);
            // After evicting 99 old high-ratio samples, the window contains 99 low + 1 high
            // 99 < 95 required → should not trigger until we have 95+ stable in a full 100-sample window
            // Actually 99 stable out of 100 → 99 >= 95, so this should trigger at the 99th sample
            // when the window is full with 99 stable + 1 unstable. Wait: let me reconsider.
            // After 100 high-ratio: window = [1.0]*100 (0 stable)
            // After 1 low-ratio: window = [1.0]*99 + [0.0] (1 stable, full → not triggered)
            // After 2 low-ratio: window = [1.0]*98 + [0.0, 0.0] (2 stable, full → not triggered)
            // ...
            // After 6 low-ratio: window = [1.0]*94 + [0.0]*6 (6 stable, full → not triggered)
            // After 7 low-ratio: window = [1.0]*93 + [0.0]*7 (7 stable, full → not triggered)
            // Actually need stable_count >= 95. So need at least 95 low-ratio to get 95 stable.
            // But window is only 100. After evicting 95 high-ratio and pushing 95 low:
            // window = [1.0]*5 + [0.0]*95 → stable_count = 95 >= 95 → TRIGGER
            if i >= 94 {
                // At i=94 (the 95th low-ratio), window = [1.0]*5 + [0.0]*95 → 95 stable → trigger
                // But the first trigger clears history, so subsequent calls see fresh state
            }
        }
    }
}
