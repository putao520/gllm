//! Residual Bus Bridge Callback (SPEC §9.3)
//!
//! Bridges the `ResidualBus` (§9.3) into the graph executor callback chain.
//! At each layer, checks active ports and performs injection/recall operations.
//!
//! This callback coordinates all bus-based operations:
//! - §16.1 RAG Injection → ResidualBus.inject() at Injection ports
//! - §8.1 Knowledge Injection → ResidualBus.inject() at Injection ports
//! - §16.2 Early Exit → ResidualBus.recall() at Recall ports
//! - §16.4 Guardrail → ResidualBus.inject() at Injection ports

use crate::graph::layer_callback::{CallbackAction, LayerCallback, LayerContext};
use crate::routing::{BusPortKind, BusPortTag, InjectionPayload, ResidualBus};

/// Residual Bus bridge callback — routes injection/recall through the physical bus.
///
/// Per SPEC §9.3: the residual stream `x_out = x_in + Layer(x_in)` has physical
/// Injection Ports and Recall Ports. This callback mediates bus operations at
/// the appropriate layers during the node loop.
pub struct ResidualBusBridgeCallback {
    /// Snapshot of bus port metadata (layer → tags) for fast lookup
    injection_layers: Vec<(usize, BusPortTag)>,
    recall_layers: Vec<(usize, BusPortTag)>,
    /// Pending injection payloads (populated before forward pass)
    pending_injections: Vec<InjectionPayload>,
    /// Recalled data from recall ports (populated during forward pass)
    recalled_data: Vec<RecalledEntry>,
    /// Hidden size for validation
    #[allow(dead_code)]
    hidden_size: usize,
    /// Previous layer's hidden state for recall cosine similarity
    prev_hidden: Vec<f32>,
}

/// Entry from a recall port
#[derive(Debug, Clone, PartialEq)]
pub struct RecalledEntry {
    /// Source port tag
    pub tag: BusPortTag,
    /// Layer where recall happened
    pub layer: usize,
    /// Extracted residual vector
    pub data: Vec<f32>,
    /// Residual energy (L2 norm)
    pub energy: f32,
}

impl ResidualBusBridgeCallback {
    /// Create from a ResidualBus reference.
    ///
    /// Snapshots the port configuration for fast layer-indexed lookup.
    pub fn from_bus(bus: &ResidualBus) -> Self {
        let mut injection_layers = Vec::new();
        let mut recall_layers = Vec::new();

        for port in bus.ports() {
            if !port.is_active() {
                continue;
            }
            match port.kind {
                BusPortKind::Injection => {
                    injection_layers.push((port.layer, port.tag));
                }
                BusPortKind::Recall => {
                    recall_layers.push((port.layer, port.tag));
                }
            }
        }

        Self {
            injection_layers,
            recall_layers,
            pending_injections: Vec::new(),
            recalled_data: Vec::new(),
            hidden_size: bus.hidden_size(),
            prev_hidden: Vec::new(),
        }
    }

    /// Queue an injection payload for the next forward pass.
    pub fn queue_injection(&mut self, payload: InjectionPayload) {
        self.pending_injections.push(payload);
    }

    /// Drain recalled data after a forward pass.
    pub fn drain_recalled(&mut self) -> Vec<RecalledEntry> {
        std::mem::take(&mut self.recalled_data)
    }

    /// Convert byte slice to f32 slice (little-endian).
    fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Convert f32 slice to bytes (little-endian).
    fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }
}

impl LayerCallback for ResidualBusBridgeCallback {
    fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        // Check injection ports at this layer
        let matching_ports: Vec<BusPortTag> = self.injection_layers.iter()
            .filter(|(layer, _)| *layer == ctx.layer_idx)
            .map(|(_, tag)| *tag)
            .collect();

        if matching_ports.is_empty() {
            return CallbackAction::Continue;
        }

        // Apply pending injections that target ports at this layer
        let mut hidden = Self::bytes_to_f32(ctx.hidden_state);
        let mut injected = false;

        let pending = std::mem::take(&mut self.pending_injections);
        let mut remaining = Vec::new();

        for payload in pending {
            if matching_ports.contains(&payload.target) {
                // Apply injection: residual += data * scale
                if payload.data.len() == hidden.len() {
                    for (h, d) in hidden.iter_mut().zip(payload.data.iter()) {
                        *h += d * payload.scale;
                    }
                    injected = true;
                    log::trace!(
                        "residual_bus_bridge: injected {:?} at layer {} (scale={:.3})",
                        payload.target, ctx.layer_idx, payload.scale,
                    );
                } else {
                    log::warn!(
                        "residual_bus_bridge: injection dimension mismatch for {:?} at layer {} (expected {}, got {})",
                        payload.target, ctx.layer_idx, hidden.len(), payload.data.len(),
                    );
                    remaining.push(payload);
                }
            } else {
                remaining.push(payload);
            }
        }
        self.pending_injections = remaining;

        if injected {
            CallbackAction::InjectHidden { data: Self::f32_to_bytes(&hidden) }
        } else {
            CallbackAction::Continue
        }
    }

    fn post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        // Check recall ports at this layer
        let matching_ports: Vec<BusPortTag> = self.recall_layers.iter()
            .filter(|(layer, _)| *layer == ctx.layer_idx)
            .map(|(_, tag)| *tag)
            .collect();

        if !matching_ports.is_empty() {
            let current = Self::bytes_to_f32(output);
            let energy = current.iter().map(|x| x * x).sum::<f32>().sqrt();

            for tag in matching_ports {
                self.recalled_data.push(RecalledEntry {
                    tag,
                    layer: ctx.layer_idx,
                    data: current.clone(),
                    energy,
                });
                log::trace!(
                    "residual_bus_bridge: recalled {:?} at layer {} (energy={:.4})",
                    tag, ctx.layer_idx, energy,
                );
            }
        }

        // Store hidden state for next layer's recall cosine similarity
        self.prev_hidden = Self::bytes_to_f32(output);

        CallbackAction::Continue
    }

    fn priority(&self) -> u32 {
        // Highest priority: bus operations happen before other callbacks
        95
    }

    fn target_layers(&self) -> Option<&[usize]> {
        // Active at all layers — ports may be at any layer
        None
    }

    fn name(&self) -> &str {
        "residual_bus_bridge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::{BusPort, BusPortKind, BusPortTag, ResidualBus, ResidualBusError};

    #[test]
    fn test_bridge_from_bus() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));

        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 1);
        assert_eq!(bridge.recall_layers.len(), 1);
        assert_eq!(bridge.injection_layers[0], (2, BusPortTag::RagInjection));
        assert_eq!(bridge.recall_layers[0], (5, BusPortTag::EarlyExit));
    }

    #[test]
    fn test_bridge_queue_and_drain() {
        let bus = ResidualBus::new(4, 8);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0, 3.0, 4.0],
            scale: 0.5,
        });

        assert_eq!(bridge.pending_injections.len(), 1);

        let recalled = bridge.drain_recalled();
        assert!(recalled.is_empty());
    }

    #[test]
    fn test_bridge_priority() {
        let bus = ResidualBus::new(4, 8);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.priority(), 95);
        assert_eq!(bridge.name(), "residual_bus_bridge");
    }

    #[test]
    fn bytes_to_f32_roundtrip() {
        let original = vec![1.0f32, -2.5, 0.0, 3.14];
        let bytes: Vec<u8> = original.iter().flat_map(|f| f.to_le_bytes()).collect();
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded, original);
    }

    #[test]
    fn f32_to_bytes_roundtrip() {
        let original = vec![0.0f32, 1.0, -1.0, 42.5];
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&original);
        assert_eq!(bytes.len(), 16);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded, original);
    }

    #[test]
    fn bytes_to_f32_empty() {
        let result = ResidualBusBridgeCallback::bytes_to_f32(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn recalled_entry_fields() {
        let entry = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 7,
            data: vec![1.0, 2.0],
            energy: 2.236,
        };
        assert_eq!(entry.layer, 7);
        assert_eq!(entry.data.len(), 2);
        assert!((entry.energy - 2.236).abs() < 0.01);
    }

    #[test]
    fn bridge_from_empty_bus() {
        let bus = ResidualBus::new(4, 8);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 0);
        assert_eq!(bridge.recall_layers.len(), 0);
    }

    #[test]
    fn bridge_multiple_injection_ports() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(1, BusPortTag::RagInjection));
        bus.register(BusPort::injection(3, BusPortTag::Guardrail));
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));

        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 2);
        assert_eq!(bridge.recall_layers.len(), 1);
    }

    #[test]
    fn bridge_target_layers_is_none() {
        let bus = ResidualBus::new(4, 8);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert!(bridge.target_layers().is_none());
    }

    // --- Additional tests ---

    #[test]
    fn recalled_entry_clone_preserves_fields() {
        let entry = RecalledEntry {
            tag: BusPortTag::IntentRecall,
            layer: 3,
            data: vec![1.0, 2.0, 3.0],
            energy: 3.742,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.tag, entry.tag);
        assert_eq!(cloned.layer, entry.layer);
        assert_eq!(cloned.data, entry.data);
        assert!((cloned.energy - entry.energy).abs() < 1e-6);
    }

    #[test]
    fn recalled_entry_debug_trait() {
        let entry = RecalledEntry {
            tag: BusPortTag::ShadowKv,
            layer: 11,
            data: vec![0.5],
            energy: 0.5,
        };
        let debug_str = format!("{:?}", entry);
        assert!(
            debug_str.contains("RecalledEntry"),
            "Debug output should contain struct name: {}",
            debug_str
        );
    }

    #[test]
    fn recalled_entry_empty_data_zero_energy() {
        let entry = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 0,
            data: vec![],
            energy: 0.0,
        };
        assert!(entry.data.is_empty());
        assert_eq!(entry.energy, 0.0);
        assert_eq!(entry.layer, 0);
    }

    #[test]
    fn bridge_hidden_size_propagated_from_bus() {
        let bus = ResidualBus::new(128, 24);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.hidden_size, 128);
    }

    #[test]
    fn bridge_queue_multiple_injections() {
        let bus = ResidualBus::new(4, 8);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0, 3.0, 4.0],
            scale: 0.5,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![0.1, 0.2, 0.3, 0.4],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::ShadowKv,
            data: vec![-1.0, -2.0, -3.0, -4.0],
            scale: 0.25,
        });

        assert_eq!(bridge.pending_injections.len(), 3);
        assert_eq!(bridge.pending_injections[0].target, BusPortTag::RagInjection);
        assert_eq!(bridge.pending_injections[1].target, BusPortTag::Guardrail);
        assert_eq!(bridge.pending_injections[2].target, BusPortTag::ShadowKv);
    }

    #[test]
    fn bridge_drain_recalled_twice_second_empty() {
        let bus = ResidualBus::new(4, 8);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        // Manually inject a recalled entry to simulate post_node having run
        bridge.recalled_data.push(RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 5,
            data: vec![1.0, 2.0, 3.0, 4.0],
            energy: 5.477,
        });

        let first_drain = bridge.drain_recalled();
        assert_eq!(first_drain.len(), 1);
        assert_eq!(first_drain[0].tag, BusPortTag::EarlyExit);

        let second_drain = bridge.drain_recalled();
        assert!(second_drain.is_empty(), "Second drain should return empty vec");
    }

    #[test]
    fn f32_to_bytes_single_value() {
        let value = vec![3.14f32];
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&value);
        assert_eq!(bytes.len(), 4);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert!((decoded[0] - 3.14f32).abs() < 1e-6);
    }

    #[test]
    fn bytes_to_f32_non_aligned_trailing_bytes_ignored() {
        // 5 bytes: 4 valid f32 bytes + 1 trailing byte ignored by chunks_exact(4)
        let mut bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
        bytes.push(0xFF); // trailing byte
        let result = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 1.0f32).abs() < 1e-6);
    }

    #[test]
    fn f32_to_bytes_negative_values() {
        let values = vec![-1.0f32, -0.5, -100.0];
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&values);
        assert_eq!(bytes.len(), 12);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded, values);
    }

    #[test]
    fn f32_to_bytes_preserves_special_values() {
        let values = vec![0.0f32, f32::INFINITY, f32::NEG_INFINITY, f32::MIN, f32::MAX];
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&values);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded[0], 0.0f32);
        assert!(decoded[1].is_infinite() && decoded[1].is_sign_positive());
        assert!(decoded[2].is_infinite() && decoded[2].is_sign_negative());
        assert_eq!(decoded[3], f32::MIN);
        assert_eq!(decoded[4], f32::MAX);
    }

    #[test]
    fn bridge_from_bus_inactive_ports_excluded() {
        let mut bus = ResidualBus::new(4, 8);
        let port = BusPort::injection(2, BusPortTag::RagInjection);
        port.deactivate();
        bus.register(port);
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));

        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        // Deactivated injection port should not be snapshoted
        assert_eq!(bridge.injection_layers.len(), 0, "Inactive port should be excluded");
        assert_eq!(bridge.recall_layers.len(), 1);
    }

    #[test]
    fn bridge_from_bus_same_layer_injection_and_recall() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(3, BusPortTag::IntentRecall));

        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 1);
        assert_eq!(bridge.recall_layers.len(), 2);
        assert_eq!(bridge.injection_layers[0], (3, BusPortTag::RagInjection));
        assert_eq!(bridge.recall_layers[0], (3, BusPortTag::EarlyExit));
        assert_eq!(bridge.recall_layers[1], (3, BusPortTag::IntentRecall));
    }

    #[test]
    fn bridge_prev_hidden_initially_empty() {
        let bus = ResidualBus::new(4, 8);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert!(bridge.prev_hidden.is_empty());
    }

    #[test]
    fn injection_payload_clone_independent() {
        let payload = InjectionPayload {
            target: BusPortTag::Custom(7),
            data: vec![10.0, 20.0],
            scale: 2.5,
        };
        let cloned = payload.clone();
        // Modify original to prove independence
        drop(payload);
        assert_eq!(cloned.target, BusPortTag::Custom(7));
        assert_eq!(cloned.data, vec![10.0, 20.0]);
        assert!((cloned.scale - 2.5).abs() < 1e-6);
    }

    #[test]
    fn bridge_all_tag_variants_injection() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::injection(1, BusPortTag::EarlyExit));
        bus.register(BusPort::injection(2, BusPortTag::IntentRecall));
        bus.register(BusPort::injection(3, BusPortTag::Guardrail));
        bus.register(BusPort::injection(4, BusPortTag::ShadowKv));
        bus.register(BusPort::injection(5, BusPortTag::Custom(99)));

        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 6);

        let tags: Vec<BusPortTag> = bridge.injection_layers.iter().map(|(_, t)| *t).collect();
        assert!(tags.contains(&BusPortTag::RagInjection));
        assert!(tags.contains(&BusPortTag::EarlyExit));
        assert!(tags.contains(&BusPortTag::IntentRecall));
        assert!(tags.contains(&BusPortTag::Guardrail));
        assert!(tags.contains(&BusPortTag::ShadowKv));
        assert!(tags.contains(&BusPortTag::Custom(99)));
    }

    // ── Helper struct for building LayerContext in tests ──

    struct TestCtxHolder {
        config: crate::engine::executor::GeneratorForwardConfig,
        hidden_state: Vec<u8>,
    }

    impl TestCtxHolder {
        fn with_hidden_len(num_f32: usize) -> Self {
            Self {
                config: crate::engine::executor::GeneratorForwardConfig {
                    geometry: std::sync::Arc::new(crate::model_config::ModelGeometry {
                        hidden_size: num_f32,
                        num_layers: 8,
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
                    }),
                    rope: crate::engine::executor::RoPEConfig {
                        theta: 10000.0,
                        scale: 1.0,
                        interleaved: false,
                        precompute: false,
                    },
                    arch_family: crate::manifest::ArchFamily::Decoder,
                    rerank_yes_token_id: None,
                    rerank_no_token_id: None,
                    moe_config: None,
                    paged_kv: crate::engine::executor_types::PagedKvConfig {
                        page_table: None,
                        page_size: 16,
                    },
                    callback_chain: crate::engine::coordinator::callback_slot::CallbackChainHandle::new(),
                },
                hidden_state: vec![0u8; num_f32 * 4],
            }
        }

        fn ctx(&self, layer: usize, node: usize) -> LayerContext<'_> {
            LayerContext {
                node_idx: node,
                layer_idx: layer,
                node_op: "BridgeTest",
                hidden_state: &self.hidden_state,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 10,
                seq_len: 1,
                position: 9,
                request_id: 1,
                model_config: &self.config,
            }
        }
    }

    // ── RecalledEntry PartialEq + trait tests ──

    #[test]
    fn recalled_entry_partial_eq_same() {
        let a = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![1.0, 2.0],
            energy: 2.236,
        };
        let b = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![1.0, 2.0],
            energy: 2.236,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn recalled_entry_partial_eq_different_tag() {
        let a = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![1.0],
            energy: 1.0,
        };
        let b = RecalledEntry {
            tag: BusPortTag::IntentRecall,
            layer: 3,
            data: vec![1.0],
            energy: 1.0,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn recalled_entry_partial_eq_different_layer() {
        let a = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![1.0],
            energy: 1.0,
        };
        let b = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 7,
            data: vec![1.0],
            energy: 1.0,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn recalled_entry_partial_eq_different_data() {
        let a = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![1.0],
            energy: 1.0,
        };
        let b = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![2.0],
            energy: 1.0,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn recalled_entry_partial_eq_different_energy() {
        let a = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![1.0],
            energy: 1.0,
        };
        let b = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![1.0],
            energy: 2.0,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn recalled_entry_clone_deep_independence() {
        let entry = RecalledEntry {
            tag: BusPortTag::Guardrail,
            layer: 5,
            data: vec![1.0, 2.0, 3.0],
            energy: 3.742,
        };
        let mut cloned = entry.clone();
        // Mutate cloned data to prove deep copy
        cloned.data.push(4.0);
        assert_eq!(entry.data.len(), 3);
        assert_eq!(cloned.data.len(), 4);
    }

    // ── pre_node tests ──

    #[test]
    fn pre_node_no_matching_ports_returns_continue() {
        // Arrange: bridge with injection port at layer 2, but ctx is at layer 5
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0, 3.0, 4.0],
            scale: 1.0,
        });

        let holder = TestCtxHolder::with_hidden_len(4);
        let ctx = holder.ctx(5, 10);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert: no matching injection port at layer 5
        assert_eq!(action, CallbackAction::Continue);
        // Pending injection should be preserved
        assert_eq!(bridge.pending_injections.len(), 1);
    }

    #[test]
    fn pre_node_injection_applies_and_returns_inject_hidden() {
        // Arrange: injection port at layer 2, payload targeting it
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(4);
        // Set hidden state to known values: [10.0, 20.0, 30.0, 40.0]
        let original: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        for (i, v) in original.iter().enumerate() {
            let bytes = v.to_le_bytes();
            holder.hidden_state[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }

        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0, 3.0, 4.0],
            scale: 0.5,
        });

        let ctx = holder.ctx(2, 4);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert: should return InjectHidden with modified data
        match action {
            CallbackAction::InjectHidden { data } => {
                assert_eq!(data.len(), 16); // 4 f32 * 4 bytes
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                assert!((decoded[0] - 10.5).abs() < 1e-6); // 10.0 + 1.0*0.5
                assert!((decoded[1] - 21.0).abs() < 1e-6); // 20.0 + 2.0*0.5
                assert!((decoded[2] - 31.5).abs() < 1e-6); // 30.0 + 3.0*0.5
                assert!((decoded[3] - 42.0).abs() < 1e-6); // 40.0 + 4.0*0.5
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
        // Pending injection should be consumed
        assert!(bridge.pending_injections.is_empty());
    }

    #[test]
    fn pre_node_injection_dimension_mismatch_preserves_payload() {
        // Arrange: payload data length != hidden_state length
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(4); // hidden_size = 4 f32 = 16 bytes
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0], // only 2, mismatch with hidden_size 4
            scale: 1.0,
        });

        let ctx = holder.ctx(2, 4);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert: no injection happened, returns Continue
        assert_eq!(action, CallbackAction::Continue);
        // Payload should be preserved (not consumed)
        assert_eq!(bridge.pending_injections.len(), 1);
        assert_eq!(bridge.pending_injections[0].data.len(), 2);
    }

    #[test]
    fn pre_node_multiple_payloads_partial_match() {
        // Arrange: two payloads, one matches, one targets different tag
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        bus.register(BusPort::injection(2, BusPortTag::Guardrail));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(4);
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 1.0, 1.0, 1.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::ShadowKv, // no matching port at this layer
            data: vec![2.0, 2.0, 2.0, 2.0],
            scale: 1.0,
        });

        let ctx = holder.ctx(2, 4);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert: injection happened (RagInjection matched)
        assert!(matches!(action, CallbackAction::InjectHidden { .. }));
        // ShadowKv payload should remain pending
        assert_eq!(bridge.pending_injections.len(), 1);
        assert_eq!(bridge.pending_injections[0].target, BusPortTag::ShadowKv);
    }

    #[test]
    fn pre_node_both_payloads_match_at_same_layer() {
        // Arrange: two payloads targeting two different injection ports at the same layer
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        bus.register(BusPort::injection(2, BusPortTag::Guardrail));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(4);
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 0.0, 0.0, 0.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![0.0, 1.0, 0.0, 0.0],
            scale: 2.0,
        });

        let ctx = holder.ctx(2, 4);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert: both injections applied cumulatively
        match action {
            CallbackAction::InjectHidden { data } => {
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                assert!((decoded[0] - 1.0).abs() < 1e-6); // 0+1*1
                assert!((decoded[1] - 2.0).abs() < 1e-6); // 0+1*2
                assert!((decoded[2]).abs() < 1e-6);        // 0
                assert!((decoded[3]).abs() < 1e-6);        // 0
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
        assert!(bridge.pending_injections.is_empty());
    }

    #[test]
    fn pre_node_scale_zero_injection_no_change_but_returns_inject_hidden() {
        // Arrange: scale=0 means injection data is multiplied by zero
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(2);
        // Set hidden state to [5.0, 10.0]
        for (i, v) in [5.0f32, 10.0f32].iter().enumerate() {
            let bytes = v.to_le_bytes();
            holder.hidden_state[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![100.0, 200.0],
            scale: 0.0,
        });

        let ctx = holder.ctx(0, 0);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert: injection happened (scale=0 is valid), data unchanged
        match action {
            CallbackAction::InjectHidden { data } => {
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                assert!((decoded[0] - 5.0).abs() < 1e-6);
                assert!((decoded[1] - 10.0).abs() < 1e-6);
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
    }

    #[test]
    fn pre_node_negative_scale_subtracts() {
        // Arrange: negative scale means data is subtracted
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::injection(1, BusPortTag::Guardrail));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(3);
        for (i, v) in [10.0f32, 20.0f32, 30.0f32].iter().enumerate() {
            let bytes = v.to_le_bytes();
            holder.hidden_state[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0, 2.0, 3.0],
            scale: -2.0,
        });

        let ctx = holder.ctx(1, 2);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::InjectHidden { data } => {
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                assert!((decoded[0] - 8.0).abs() < 1e-6);  // 10 + 1*(-2)
                assert!((decoded[1] - 16.0).abs() < 1e-6); // 20 + 2*(-2)
                assert!((decoded[2] - 24.0).abs() < 1e-6); // 30 + 3*(-2)
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
    }

    #[test]
    fn pre_node_no_pending_injections_returns_continue() {
        // Arrange: injection port exists at layer, but no payloads queued
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(4);
        let ctx = holder.ctx(2, 4);

        // Act
        let mut b = bridge;
        let action = b.pre_node(&ctx);

        // Assert: matching port exists but no payloads
        assert_eq!(action, CallbackAction::Continue);
    }

    #[test]
    fn pre_node_no_injection_ports_on_empty_bus_returns_continue() {
        // Arrange: empty bus (no ports at all)
        let bus = ResidualBus::new(4, 8);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0, 3.0, 4.0],
            scale: 1.0,
        });

        let holder = TestCtxHolder::with_hidden_len(4);
        let ctx = holder.ctx(2, 4);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert
        assert_eq!(action, CallbackAction::Continue);
        // Payload preserved since no ports matched
        assert_eq!(bridge.pending_injections.len(), 1);
    }

    // ── post_node tests ──

    #[test]
    fn post_node_recall_port_captures_data_and_energy() {
        // Arrange: recall port at layer 3
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(4);
        let ctx = holder.ctx(3, 6);

        // Output: [3.0, 4.0, 0.0, 0.0] → energy = sqrt(9+16) = 5.0
        let output: Vec<f32> = vec![3.0, 4.0, 0.0, 0.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        let action = bridge.post_node(&ctx, &output_bytes);

        // Assert
        assert_eq!(action, CallbackAction::Continue);
        assert_eq!(bridge.recalled_data.len(), 1);

        let entry = &bridge.recalled_data[0];
        assert_eq!(entry.tag, BusPortTag::EarlyExit);
        assert_eq!(entry.layer, 3);
        assert_eq!(entry.data, vec![3.0, 4.0, 0.0, 0.0]);
        assert!((entry.energy - 5.0).abs() < 1e-4);
    }

    #[test]
    fn post_node_updates_prev_hidden() {
        // Arrange
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::recall(1, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        assert!(bridge.prev_hidden.is_empty());

        let holder = TestCtxHolder::with_hidden_len(3);
        let ctx = holder.ctx(1, 2);

        let output: Vec<f32> = vec![7.0, 8.0, 9.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        bridge.post_node(&ctx, &output_bytes);

        // Assert: prev_hidden should now hold the output
        assert_eq!(bridge.prev_hidden, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn post_node_no_matching_ports_skips_recall() {
        // Arrange: recall port at layer 5, but ctx is at layer 2
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(4);
        let ctx = holder.ctx(2, 4);

        let output: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        let action = bridge.post_node(&ctx, &output_bytes);

        // Assert: no recall, but prev_hidden still updated
        assert_eq!(action, CallbackAction::Continue);
        assert!(bridge.recalled_data.is_empty());
        assert_eq!(bridge.prev_hidden, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn post_node_multiple_recall_ports_at_same_layer() {
        // Arrange: two recall ports at layer 3
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(3, BusPortTag::IntentRecall));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(3, 6);

        let output: Vec<f32> = vec![3.0, 4.0]; // energy = 5.0
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        bridge.post_node(&ctx, &output_bytes);

        // Assert: two entries captured
        assert_eq!(bridge.recalled_data.len(), 2);
        assert_eq!(bridge.recalled_data[0].tag, BusPortTag::EarlyExit);
        assert_eq!(bridge.recalled_data[1].tag, BusPortTag::IntentRecall);
        // Both have same data and energy
        assert_eq!(bridge.recalled_data[0].data, vec![3.0, 4.0]);
        assert_eq!(bridge.recalled_data[1].data, vec![3.0, 4.0]);
        assert!((bridge.recalled_data[0].energy - 5.0).abs() < 1e-4);
        assert!((bridge.recalled_data[1].energy - 5.0).abs() < 1e-4);
    }

    #[test]
    fn post_node_zero_vector_zero_energy() {
        // Arrange
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(3);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![0.0, 0.0, 0.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        bridge.post_node(&ctx, &output_bytes);

        // Assert
        assert_eq!(bridge.recalled_data.len(), 1);
        assert!((bridge.recalled_data[0].energy).abs() < 1e-8);
    }

    #[test]
    fn post_node_energy_calculation_l2_norm() {
        // Arrange: output [1.0, 1.0, 1.0, 1.0] → energy = sqrt(4) = 2.0
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(4);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        bridge.post_node(&ctx, &output_bytes);

        // Assert
        assert!((bridge.recalled_data[0].energy - 2.0).abs() < 1e-4);
    }

    #[test]
    fn post_node_always_returns_continue() {
        // Arrange: even with recall ports, post_node always returns Continue
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(4);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        let action = bridge.post_node(&ctx, &output_bytes);

        // Assert
        assert_eq!(action, CallbackAction::Continue);
    }

    // ── Combined pre_node + post_node lifecycle ──

    #[test]
    fn lifecycle_inject_then_recall_across_layers() {
        // Arrange: injection at layer 1, recall at layer 3
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::injection(1, BusPortTag::RagInjection));
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(3);

        // Queue injection
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![10.0, 20.0, 30.0],
            scale: 1.0,
        });

        // Layer 0: no ports, pre_node returns Continue
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(bridge.pre_node(&ctx0), CallbackAction::Continue);

        // Layer 1: injection port matches, pre_node applies injection
        // Set hidden to [1.0, 2.0, 3.0]
        for (i, v) in [1.0f32, 2.0f32, 3.0f32].iter().enumerate() {
            let bytes = v.to_le_bytes();
            holder.hidden_state[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }
        let ctx1 = holder.ctx(1, 2);
        let pre1 = bridge.pre_node(&ctx1);
        assert!(matches!(pre1, CallbackAction::InjectHidden { .. }));
        assert!(bridge.pending_injections.is_empty());

        // Layer 2: no ports
        let ctx2 = holder.ctx(2, 4);
        assert_eq!(bridge.pre_node(&ctx2), CallbackAction::Continue);

        // Layer 3: recall port captures data
        let output: Vec<f32> = vec![11.0, 22.0, 33.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
        let ctx3 = holder.ctx(3, 6);
        let post3 = bridge.post_node(&ctx3, &output_bytes);
        assert_eq!(post3, CallbackAction::Continue);
        assert_eq!(bridge.recalled_data.len(), 1);
        assert_eq!(bridge.recalled_data[0].data, vec![11.0, 22.0, 33.0]);

        // Drain recalled
        let drained = bridge.drain_recalled();
        assert_eq!(drained.len(), 1);
        assert!(bridge.drain_recalled().is_empty());
    }

    #[test]
    fn lifecycle_multiple_forward_passes_accumulate_recalled() {
        // Arrange: recall port at layer 0, run post_node twice
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        // First pass
        let out1: Vec<f32> = vec![1.0, 0.0];
        let bytes1: Vec<u8> = out1.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &bytes1);
        assert_eq!(bridge.recalled_data.len(), 1);

        // Second pass (same layer, another step)
        let out2: Vec<f32> = vec![0.0, 1.0];
        let bytes2: Vec<u8> = out2.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &bytes2);
        assert_eq!(bridge.recalled_data.len(), 2);

        // Drain both
        let drained = bridge.drain_recalled();
        assert_eq!(drained.len(), 2);
    }

    // ── bytes_to_f32 / f32_to_bytes edge cases ──

    #[test]
    fn f32_to_bytes_empty_input() {
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&[]);
        assert!(bytes.is_empty());
    }

    #[test]
    fn bytes_to_f32_with_nan() {
        let nan = f32::NAN;
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&[nan]);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert!(decoded[0].is_nan());
    }

    #[test]
    fn bytes_to_f32_with_subnormal() {
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&[subnormal]);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded[0].to_bits(), 1u32);
    }

    #[test]
    fn f32_to_bytes_large_vector() {
        let values: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&values);
        assert_eq!(bytes.len(), 4000);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded.len(), 1000);
        assert!((decoded[0] - 0.0).abs() < 1e-6);
        assert!((decoded[999] - 0.999).abs() < 1e-3);
    }

    #[test]
    fn bytes_to_f32_two_bytes_remaining_dropped() {
        // 2 bytes: not enough for a full f32, chunks_exact(4) drops them
        let bytes: Vec<u8> = vec![0x00, 0x01];
        let result = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert!(result.is_empty());
    }

    #[test]
    fn bytes_to_f32_three_bytes_remaining_dropped() {
        let bytes: Vec<u8> = vec![0x00, 0x01, 0x02];
        let result = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert!(result.is_empty());
    }

    // ── BusPortTag Hash + BusPortKind Copy ──

    #[test]
    fn bus_port_tag_all_variants_copy_semantic() {
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
            BusPortTag::Custom(42),
        ];
        for tag in &tags {
            let copy = *tag;
            assert_eq!(*tag, copy);
        }
    }

    #[test]
    fn bus_port_kind_copy_semantic() {
        let injection = BusPortKind::Injection;
        let copy = injection;
        assert_eq!(injection, copy);
        assert_eq!(BusPortKind::Injection, BusPortKind::Injection);
        assert_ne!(BusPortKind::Injection, BusPortKind::Recall);
    }

    // ── from_bus with large configuration ──

    #[test]
    fn bridge_from_bus_large_hidden_size() {
        let bus = ResidualBus::new(4096, 32);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.hidden_size, 4096);
    }

    #[test]
    fn bridge_from_bus_many_ports() {
        let mut bus = ResidualBus::new(4, 100);
        for layer in 0..50 {
            bus.register(BusPort::injection(layer, BusPortTag::RagInjection));
            bus.register(BusPort::recall(layer, BusPortTag::EarlyExit));
        }
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 50);
        assert_eq!(bridge.recall_layers.len(), 50);
    }

    // ── Additional coverage tests ──

    #[test]
    fn recalled_entry_all_tag_variants() {
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
            BusPortTag::Custom(u32::MAX),
        ];
        for (i, tag) in tags.iter().enumerate() {
            let entry = RecalledEntry {
                tag: *tag,
                layer: i,
                data: vec![i as f32],
                energy: i as f32,
            };
            assert_eq!(entry.tag, *tag);
            assert_eq!(entry.layer, i);
        }
    }

    #[test]
    fn recalled_entry_large_data_vector() {
        let data: Vec<f32> = (0..4096).map(|i| i as f32).collect();
        let energy = data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let entry = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 10,
            data: data.clone(),
            energy,
        };
        assert_eq!(entry.data.len(), 4096);
        assert!(energy > 0.0);
    }

    #[test]
    fn recalled_entry_partial_eq_empty_data() {
        let a = RecalledEntry {
            tag: BusPortTag::Custom(1),
            layer: 0,
            data: vec![],
            energy: 0.0,
        };
        let b = RecalledEntry {
            tag: BusPortTag::Custom(1),
            layer: 0,
            data: vec![],
            energy: 0.0,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn injection_payload_construction_all_fields() {
        let payload = InjectionPayload {
            target: BusPortTag::Custom(255),
            data: vec![42.0],
            scale: -1.5,
        };
        assert_eq!(payload.target, BusPortTag::Custom(255));
        assert_eq!(payload.data, vec![42.0]);
        assert!((payload.scale - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn bridge_from_bus_preserves_layer_ordering() {
        let mut bus = ResidualBus::new(4, 10);
        // Register out of order
        bus.register(BusPort::injection(5, BusPortTag::RagInjection));
        bus.register(BusPort::injection(2, BusPortTag::Guardrail));
        bus.register(BusPort::injection(8, BusPortTag::ShadowKv));
        bus.register(BusPort::recall(3, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(7, BusPortTag::IntentRecall));

        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        // Verify ordering is maintained from bus
        assert_eq!(bridge.injection_layers.len(), 3);
        assert_eq!(bridge.recall_layers.len(), 2);

        // Bus sorts by layer via partition_point, verify consistent snapshot
        let inj_layers: Vec<usize> = bridge.injection_layers.iter().map(|(l, _)| *l).collect();
        for window in inj_layers.windows(2) {
            assert!(window[0] <= window[1]);
        }
    }

    #[test]
    fn pre_node_injection_with_single_element_hidden() {
        // Arrange: hidden_size = 1
        let mut bus = ResidualBus::new(1, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(1);
        // Set hidden to [5.0]
        holder.hidden_state[0..4].copy_from_slice(&5.0f32.to_le_bytes());
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![3.0],
            scale: 2.0,
        });

        let ctx = holder.ctx(0, 0);

        // Act
        let action = bridge.pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::InjectHidden { data } => {
                assert_eq!(data.len(), 4);
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                assert!((decoded[0] - 11.0).abs() < 1e-6); // 5 + 3*2
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
    }

    #[test]
    fn post_node_prev_hidden_updated_to_output_even_without_recall() {
        // Arrange: no recall ports, but post_node should still update prev_hidden
        let bus = ResidualBus::new(2, 4);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert!(bridge.prev_hidden.is_empty());

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![99.0, -99.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        bridge.post_node(&ctx, &output_bytes);

        // Assert: prev_hidden updated even without recall ports
        assert_eq!(bridge.prev_hidden, vec![99.0, -99.0]);
        assert!(bridge.recalled_data.is_empty());
    }

    #[test]
    fn f32_to_bytes_preserves_zero_sign() {
        let positive_zero = 0.0f32;
        let negative_zero = -0.0f32;
        let bytes_pos = ResidualBusBridgeCallback::f32_to_bytes(&[positive_zero]);
        let bytes_neg = ResidualBusBridgeCallback::f32_to_bytes(&[negative_zero]);

        // Different bit patterns: +0 = 0x00000000, -0 = 0x80000000
        assert_ne!(bytes_pos, bytes_neg);

        let decoded_pos = ResidualBusBridgeCallback::bytes_to_f32(&bytes_pos);
        let decoded_neg = ResidualBusBridgeCallback::bytes_to_f32(&bytes_neg);
        assert_eq!(decoded_pos[0], 0.0);
        assert_eq!(decoded_neg[0], 0.0);
        assert!(decoded_pos[0].is_sign_positive());
        assert!(decoded_neg[0].is_sign_negative());
    }

    #[test]
    fn post_node_recall_with_negative_values_energy_correct() {
        // Arrange: output [-3.0, -4.0] → energy = sqrt(9+16) = 5.0
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![-3.0, -4.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act
        bridge.post_node(&ctx, &output_bytes);

        // Assert: energy should be positive despite negative values
        assert!((bridge.recalled_data[0].energy - 5.0).abs() < 1e-4);
    }

    // ── New tests: RecalledEntry special float values ──

    #[test]
    fn recalled_entry_nan_energy() {
        let entry = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 0,
            data: vec![f32::NAN],
            energy: f32::NAN,
        };
        assert!(entry.energy.is_nan());
        assert!(entry.data[0].is_nan());
    }

    #[test]
    fn recalled_entry_infinity_energy() {
        let entry = RecalledEntry {
            tag: BusPortTag::IntentRecall,
            layer: 1,
            data: vec![f32::INFINITY],
            energy: f32::INFINITY,
        };
        assert!(entry.energy.is_infinite() && entry.energy.is_sign_positive());
    }

    #[test]
    fn recalled_entry_neg_infinity_data() {
        let entry = RecalledEntry {
            tag: BusPortTag::Guardrail,
            layer: 5,
            data: vec![f32::NEG_INFINITY, 0.0],
            energy: f32::INFINITY,
        };
        assert!(entry.data[0].is_infinite() && entry.data[0].is_sign_negative());
        assert_eq!(entry.data[1], 0.0);
    }

    #[test]
    fn recalled_entry_subnormal_values() {
        let sub = f32::from_bits(1);
        let entry = RecalledEntry {
            tag: BusPortTag::ShadowKv,
            layer: 2,
            data: vec![sub, f32::from_bits(0x007FFFFF)],
            energy: sub,
        };
        assert_eq!(entry.data[0].to_bits(), 1u32);
        assert!(entry.data[1].is_subnormal());
    }

    #[test]
    fn recalled_entry_max_layer_value() {
        let entry = RecalledEntry {
            tag: BusPortTag::Custom(0),
            layer: usize::MAX,
            data: vec![1.0],
            energy: 1.0,
        };
        assert_eq!(entry.layer, usize::MAX);
    }

    #[test]
    fn recalled_entry_f32_max_min_values() {
        let entry = RecalledEntry {
            tag: BusPortTag::RagInjection,
            layer: 0,
            data: vec![f32::MAX, f32::MIN, f32::MIN_POSITIVE],
            energy: f32::MAX,
        };
        assert_eq!(entry.data[0], f32::MAX);
        assert_eq!(entry.data[1], f32::MIN);
        assert_eq!(entry.data[2], f32::MIN_POSITIVE);
    }

    #[test]
    fn recalled_entry_large_custom_tag() {
        let entry = RecalledEntry {
            tag: BusPortTag::Custom(u32::MAX),
            layer: 0,
            data: vec![],
            energy: 0.0,
        };
        assert_eq!(entry.tag, BusPortTag::Custom(u32::MAX));
    }

    #[test]
    fn recalled_entry_negative_energy_value() {
        let entry = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 3,
            data: vec![1.0],
            energy: -5.0,
        };
        assert!((entry.energy - (-5.0)).abs() < 1e-6);
    }

    // ── New tests: bytes_to_f32 further edge cases ──

    #[test]
    fn bytes_to_f32_roundtrip_f32_min_positive() {
        let value = f32::MIN_POSITIVE;
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&[value]);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded[0], value);
    }

    #[test]
    fn bytes_to_f32_roundtrip_f32_max() {
        let value = f32::MAX;
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&[value]);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded[0], value);
    }

    #[test]
    fn bytes_to_f32_roundtrip_f32_min() {
        let value = f32::MIN;
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&[value]);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded[0], value);
    }

    #[test]
    fn bytes_to_f32_multiple_special_values() {
        let values = vec![0.0f32, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let bytes = ResidualBusBridgeCallback::f32_to_bytes(&values);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);

        assert_eq!(decoded[0], 0.0);
        assert!(decoded[0].is_sign_positive());
        assert_eq!(decoded[1], 0.0);
        assert!(decoded[1].is_sign_negative());
        assert!(decoded[2].is_nan());
        assert!(decoded[3].is_infinite() && decoded[3].is_sign_positive());
        assert!(decoded[4].is_infinite() && decoded[4].is_sign_negative());
    }

    #[test]
    fn bytes_to_f32_exact_8_bytes_yields_two_f32() {
        let bytes: Vec<u8> = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
        assert_eq!(bytes.len(), 8);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded.len(), 2);
        assert!((decoded[0] - 1.0).abs() < 1e-6);
        assert!((decoded[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn bytes_to_f32_7_bytes_yields_one_f32_and_3_remainder_dropped() {
        let mut bytes: Vec<u8> = 42.0f32.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0xAA, 0xBB, 0xCC]); // 3 trailing bytes
        assert_eq!(bytes.len(), 7);
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - 42.0).abs() < 1e-6);
    }

    // ── New tests: BusPortKind / BusPortTag trait coverage ──

    #[test]
    fn bus_port_kind_equality_reflexive() {
        assert_eq!(BusPortKind::Injection, BusPortKind::Injection);
        assert_eq!(BusPortKind::Recall, BusPortKind::Recall);
    }

    #[test]
    fn bus_port_kind_inequality() {
        assert_ne!(BusPortKind::Injection, BusPortKind::Recall);
    }

    #[test]
    fn bus_port_tag_equality_same_custom() {
        assert_eq!(BusPortTag::Custom(42), BusPortTag::Custom(42));
    }

    #[test]
    fn bus_port_tag_inequality_different_custom() {
        assert_ne!(BusPortTag::Custom(1), BusPortTag::Custom(2));
    }

    #[test]
    fn bus_port_tag_inequality_named_vs_custom() {
        assert_ne!(BusPortTag::RagInjection, BusPortTag::Custom(0));
    }

    #[test]
    fn bus_port_tag_all_named_variants_distinct() {
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
        ];
        for i in 0..tags.len() {
            for j in 0..tags.len() {
                if i != j {
                    assert_ne!(tags[i], tags[j], "{:?} should != {:?}", tags[i], tags[j]);
                }
            }
        }
    }

    #[test]
    fn bus_port_tag_copy_independent() {
        let original = BusPortTag::Custom(99);
        let copy = original;
        assert_eq!(original, copy);
    }

    #[test]
    fn bus_port_tag_zero_custom() {
        let tag = BusPortTag::Custom(0);
        assert_eq!(tag, BusPortTag::Custom(0));
        assert_ne!(tag, BusPortTag::RagInjection);
    }

    // ── New tests: InjectionPayload construction edge cases ──

    #[test]
    fn injection_payload_empty_data() {
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![],
            scale: 1.0,
        };
        assert!(payload.data.is_empty());
    }

    #[test]
    fn injection_payload_nan_scale() {
        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0],
            scale: f32::NAN,
        };
        assert!(payload.scale.is_nan());
    }

    #[test]
    fn injection_payload_infinity_scale() {
        let payload = InjectionPayload {
            target: BusPortTag::ShadowKv,
            data: vec![1.0],
            scale: f32::INFINITY,
        };
        assert!(payload.scale.is_infinite() && payload.scale.is_sign_positive());
    }

    #[test]
    fn injection_payload_neg_infinity_scale() {
        let payload = InjectionPayload {
            target: BusPortTag::EarlyExit,
            data: vec![1.0],
            scale: f32::NEG_INFINITY,
        };
        assert!(payload.scale.is_infinite() && payload.scale.is_sign_negative());
    }

    #[test]
    fn injection_payload_large_data_vector() {
        let data: Vec<f32> = (0..8192).map(|i| i as f32).collect();
        let payload = InjectionPayload {
            target: BusPortTag::Custom(10),
            data: data.clone(),
            scale: 0.1,
        };
        assert_eq!(payload.data.len(), 8192);
        assert_eq!(payload.data[0], 0.0);
        assert!((payload.data[8191] - 8191.0).abs() < 1e-6);
    }

    #[test]
    fn injection_payload_special_floats_in_data() {
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0],
            scale: 1.0,
        };
        assert!(payload.data[0].is_nan());
        assert!(payload.data[1].is_infinite() && payload.data[1].is_sign_positive());
        assert!(payload.data[2].is_infinite() && payload.data[2].is_sign_negative());
        assert!(payload.data[4].is_sign_negative());
    }

    #[test]
    fn injection_payload_clone_independence() {
        let mut payload = InjectionPayload {
            target: BusPortTag::Custom(5),
            data: vec![1.0, 2.0, 3.0],
            scale: 1.0,
        };
        let cloned = payload.clone();
        payload.data.push(4.0);
        assert_eq!(cloned.data.len(), 3);
        assert_eq!(payload.data.len(), 4);
    }

    // ── New tests: Bridge port management edge cases ──

    #[test]
    fn bridge_from_bus_layer_zero() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers[0].0, 0);
        assert_eq!(bridge.recall_layers[0].0, 0);
    }

    #[test]
    fn bridge_from_bus_max_layer() {
        let mut bus = ResidualBus::new(4, 200);
        bus.register(BusPort::injection(199, BusPortTag::RagInjection));
        bus.register(BusPort::recall(199, BusPortTag::EarlyExit));
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers[0].0, 199);
        assert_eq!(bridge.recall_layers[0].0, 199);
    }

    #[test]
    fn bridge_from_bus_only_injection_no_recall() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::injection(1, BusPortTag::RagInjection));
        bus.register(BusPort::injection(2, BusPortTag::Guardrail));
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 2);
        assert!(bridge.recall_layers.is_empty());
    }

    #[test]
    fn bridge_from_bus_only_recall_no_injection() {
        let mut bus = ResidualBus::new(4, 8);
        bus.register(BusPort::recall(1, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(3, BusPortTag::IntentRecall));
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert!(bridge.injection_layers.is_empty());
        assert_eq!(bridge.recall_layers.len(), 2);
    }

    #[test]
    fn bridge_from_bus_multiple_inactive_excluded() {
        let mut bus = ResidualBus::new(4, 8);
        let p1 = BusPort::injection(1, BusPortTag::RagInjection);
        p1.deactivate();
        let p2 = BusPort::recall(2, BusPortTag::EarlyExit);
        p2.deactivate();
        bus.register(p1);
        bus.register(p2);
        bus.register(BusPort::injection(3, BusPortTag::Guardrail));
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 1);
        assert!(bridge.recall_layers.is_empty());
        assert_eq!(bridge.injection_layers[0], (3, BusPortTag::Guardrail));
    }

    #[test]
    fn bridge_from_bus_activate_then_deactivate_excluded() {
        let mut bus = ResidualBus::new(4, 8);
        let port = BusPort::injection(0, BusPortTag::RagInjection);
        port.activate();
        port.deactivate();
        bus.register(port);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert!(bridge.injection_layers.is_empty());
    }

    #[test]
    fn bridge_hidden_size_one() {
        let bus = ResidualBus::new(1, 4);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.hidden_size, 1);
    }

    #[test]
    fn bridge_hidden_size_large() {
        let bus = ResidualBus::new(65536, 64);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.hidden_size, 65536);
    }

    // ── New tests: queue_injection edge cases ──

    #[test]
    fn queue_injection_then_queue_another_preserves_order() {
        let bus = ResidualBus::new(4, 8);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![2.0],
            scale: 2.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Custom(0),
            data: vec![3.0],
            scale: 3.0,
        });
        assert_eq!(bridge.pending_injections[0].target, BusPortTag::RagInjection);
        assert_eq!(bridge.pending_injections[1].target, BusPortTag::Guardrail);
        assert_eq!(bridge.pending_injections[2].target, BusPortTag::Custom(0));
        assert!((bridge.pending_injections[0].scale - 1.0).abs() < 1e-6);
        assert!((bridge.pending_injections[1].scale - 2.0).abs() < 1e-6);
        assert!((bridge.pending_injections[2].scale - 3.0).abs() < 1e-6);
    }

    #[test]
    fn drain_recalled_on_empty_returns_empty_vec() {
        let bus = ResidualBus::new(4, 8);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);
        let result = bridge.drain_recalled();
        assert!(result.is_empty());
    }

    // ── New tests: pre_node with NaN/Inf injection data ──

    #[test]
    fn pre_node_injection_with_nan_data() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![f32::NAN, 1.0],
            scale: 1.0,
        });

        let ctx = holder.ctx(0, 0);
        let action = bridge.pre_node(&ctx);

        match action {
            CallbackAction::InjectHidden { data } => {
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                assert!(decoded[0].is_nan());
                assert!((decoded[1] - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
    }

    #[test]
    fn pre_node_injection_with_inf_data() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![f32::INFINITY, -f32::INFINITY],
            scale: 1.0,
        });

        let ctx = holder.ctx(0, 0);
        let action = bridge.pre_node(&ctx);

        match action {
            CallbackAction::InjectHidden { data } => {
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                assert!(decoded[0].is_infinite() && decoded[0].is_sign_positive());
                assert!(decoded[1].is_infinite() && decoded[1].is_sign_negative());
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
    }

    #[test]
    fn pre_node_injection_with_inf_scale_overflow() {
        let mut bus = ResidualBus::new(1, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(1);
        holder.hidden_state[0..4].copy_from_slice(&1.0f32.to_le_bytes());

        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0],
            scale: f32::INFINITY,
        });

        let ctx = holder.ctx(0, 0);
        let action = bridge.pre_node(&ctx);

        match action {
            CallbackAction::InjectHidden { data } => {
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                // 1.0 + 1.0 * inf = inf
                assert!(decoded[0].is_infinite());
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
    }

    // ── New tests: post_node special float values ──

    #[test]
    fn post_node_recall_with_nan_in_output() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![f32::NAN, 1.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        bridge.post_node(&ctx, &output_bytes);

        assert_eq!(bridge.recalled_data.len(), 1);
        assert!(bridge.recalled_data[0].data[0].is_nan());
        assert!(bridge.recalled_data[0].energy.is_nan(), "NaN in output should produce NaN energy");
    }

    #[test]
    fn post_node_recall_with_inf_in_output() {
        let mut bus = ResidualBus::new(1, 4);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(1);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![f32::INFINITY];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        bridge.post_node(&ctx, &output_bytes);

        assert!(bridge.recalled_data[0].energy.is_infinite());
    }

    #[test]
    fn post_node_recall_large_vector_energy() {
        let mut bus = ResidualBus::new(256, 8);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(256);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![1.0; 256]; // energy = sqrt(256) = 16.0
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        bridge.post_node(&ctx, &output_bytes);

        assert!((bridge.recalled_data[0].energy - 16.0).abs() < 1e-4);
    }

    #[test]
    fn post_node_prev_hidden_replaced_on_each_call() {
        let bus = ResidualBus::new(2, 4);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        let out1: Vec<f32> = vec![1.0, 2.0];
        let bytes1: Vec<u8> = out1.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &bytes1);
        assert_eq!(bridge.prev_hidden, vec![1.0, 2.0]);

        let out2: Vec<f32> = vec![3.0, 4.0];
        let bytes2: Vec<u8> = out2.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &bytes2);
        assert_eq!(bridge.prev_hidden, vec![3.0, 4.0]);
    }

    // ── New tests: full lifecycle edge cases ──

    #[test]
    fn lifecycle_injection_and_recall_same_layer_different_tags() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(2);
        holder.hidden_state[0..4].copy_from_slice(&10.0f32.to_le_bytes());
        holder.hidden_state[4..8].copy_from_slice(&20.0f32.to_le_bytes());

        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0],
            scale: 0.5,
        });

        let ctx = holder.ctx(0, 0);

        // pre_node applies injection
        let pre = bridge.pre_node(&ctx);
        assert!(matches!(pre, CallbackAction::InjectHidden { .. }));

        // post_node captures recall
        let output: Vec<f32> = vec![10.5, 21.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &output_bytes);

        assert_eq!(bridge.recalled_data.len(), 1);
        assert_eq!(bridge.recalled_data[0].tag, BusPortTag::EarlyExit);
        assert_eq!(bridge.recalled_data[0].layer, 0);
    }

    #[test]
    fn lifecycle_drain_between_forward_passes() {
        let mut bus = ResidualBus::new(1, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(1);
        let ctx = holder.ctx(0, 0);

        // First recall
        let bytes1: Vec<u8> = 5.0f32.to_le_bytes().to_vec();
        bridge.post_node(&ctx, &bytes1);
        assert_eq!(bridge.recalled_data.len(), 1);

        // Drain
        let drained = bridge.drain_recalled();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].data, vec![5.0]);

        // Second recall accumulates fresh
        let bytes2: Vec<u8> = 10.0f32.to_le_bytes().to_vec();
        bridge.post_node(&ctx, &bytes2);
        assert_eq!(bridge.recalled_data.len(), 1);
        assert_eq!(bridge.recalled_data[0].data, vec![10.0]);
    }

    #[test]
    fn lifecycle_multiple_injections_only_matching_consumed() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);

        // Queue 3 injections, only 1 matches port tag
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 1.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![2.0, 2.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Custom(77),
            data: vec![3.0, 3.0],
            scale: 1.0,
        });

        let ctx = holder.ctx(0, 0);
        let action = bridge.pre_node(&ctx);

        // Only RagInjection was consumed
        assert!(matches!(action, CallbackAction::InjectHidden { .. }));
        assert_eq!(bridge.pending_injections.len(), 2);
        assert_eq!(bridge.pending_injections[0].target, BusPortTag::Guardrail);
        assert_eq!(bridge.pending_injections[1].target, BusPortTag::Custom(77));
    }

    #[test]
    fn lifecycle_no_ports_at_layer_continues_all_operations() {
        let mut bus = ResidualBus::new(2, 8);
        bus.register(BusPort::injection(5, BusPortTag::RagInjection));
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);

        // Layer 0 has no ports
        let ctx = holder.ctx(0, 0);
        assert_eq!(bridge.pre_node(&ctx), CallbackAction::Continue);

        let output: Vec<f32> = vec![1.0, 2.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
        let post = bridge.post_node(&ctx, &output_bytes);
        assert_eq!(post, CallbackAction::Continue);
        assert!(bridge.recalled_data.is_empty());
        assert_eq!(bridge.prev_hidden, vec![1.0, 2.0]);
    }

    // ── New tests: CallbackAction comparisons ──

    #[test]
    fn callback_action_continue_equality() {
        assert_eq!(CallbackAction::Continue, CallbackAction::Continue);
    }

    #[test]
    fn callback_action_inject_hidden_equality_same_data() {
        let data = vec![1.0f32.to_le_bytes().to_vec(), 2.0f32.to_le_bytes().to_vec()].concat();
        let a = CallbackAction::InjectHidden { data: data.clone() };
        let b = CallbackAction::InjectHidden { data: data.clone() };
        assert_eq!(a, b);
    }

    #[test]
    fn callback_action_inject_hidden_inequality_different_data() {
        let data1 = vec![1.0f32.to_le_bytes().to_vec()].concat();
        let data2 = vec![2.0f32.to_le_bytes().to_vec()].concat();
        let a = CallbackAction::InjectHidden { data: data1 };
        let b = CallbackAction::InjectHidden { data: data2 };
        assert_ne!(a, b);
    }

    #[test]
    fn callback_action_continue_not_equal_inject_hidden() {
        let action = CallbackAction::InjectHidden { data: vec![] };
        assert_ne!(CallbackAction::Continue, action);
    }

    // ── New tests: port activation/deactivation round-trip ──

    #[test]
    fn bus_port_activate_deactivate_cycle() {
        let port = BusPort::injection(3, BusPortTag::Guardrail);
        assert!(port.is_active());
        port.deactivate();
        assert!(!port.is_active());
        port.activate();
        assert!(port.is_active());
    }

    fn bus_port_recall_creation_active() {
        let port = BusPort::recall(7, BusPortTag::IntentRecall);
        assert!(port.is_active());
        assert_eq!(port.kind, BusPortKind::Recall);
        assert_eq!(port.layer, 7);
        assert_eq!(port.tag, BusPortTag::IntentRecall);
    }

    // ── New tests: CallbackAction variant coverage ──

    #[test]
    fn callback_action_default_is_continue() {
        let default: CallbackAction = CallbackAction::default();
        assert_eq!(default, CallbackAction::Continue);
    }

    #[test]
    fn callback_action_skip_this_node_equality() {
        assert_eq!(CallbackAction::SkipThisNode, CallbackAction::SkipThisNode);
    }

    #[test]
    fn callback_action_skip_this_node_not_equal_continue() {
        assert_ne!(CallbackAction::SkipThisNode, CallbackAction::Continue);
    }

    #[test]
    fn callback_action_exit_early_same_logits_equal() {
        let a = CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] };
        let b = CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] };
        assert_eq!(a, b);
    }

    #[test]
    fn callback_action_exit_early_different_logits_not_equal() {
        let a = CallbackAction::ExitEarly { logits: vec![1.0] };
        let b = CallbackAction::ExitEarly { logits: vec![2.0] };
        assert_ne!(a, b);
    }

    #[test]
    fn callback_action_exit_early_empty_logits() {
        let action = CallbackAction::ExitEarly { logits: vec![] };
        assert!(matches!(action, CallbackAction::ExitEarly { logits } if logits.is_empty()));
    }

    #[test]
    fn callback_action_compact_mask_same_equal() {
        let a = CallbackAction::CompactMask { active_mask: vec![true, false, true] };
        let b = CallbackAction::CompactMask { active_mask: vec![true, false, true] };
        assert_eq!(a, b);
    }

    #[test]
    fn callback_action_compact_mask_different_not_equal() {
        let a = CallbackAction::CompactMask { active_mask: vec![true] };
        let b = CallbackAction::CompactMask { active_mask: vec![false] };
        assert_ne!(a, b);
    }

    #[test]
    fn callback_action_compact_mask_empty_mask() {
        let action = CallbackAction::CompactMask { active_mask: vec![] };
        assert!(matches!(action, CallbackAction::CompactMask { active_mask } if active_mask.is_empty()));
    }

    #[test]
    fn callback_action_all_variants_distinct() {
        let variants: Vec<CallbackAction> = vec![
            CallbackAction::Continue,
            CallbackAction::SkipThisNode,
            CallbackAction::ExitEarly { logits: vec![] },
            CallbackAction::InjectHidden { data: vec![] },
            CallbackAction::CompactMask { active_mask: vec![] },
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "variant {i} should differ from {j}");
            }
        }
    }

    // ── New tests: pre_node sequential layer processing ──

    #[test]
    fn pre_node_injection_consumed_then_queue_again() {
        // Arrange: injection port at layer 0
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        // First injection
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 1.0],
            scale: 1.0,
        });
        let action1 = bridge.pre_node(&ctx);
        assert!(matches!(action1, CallbackAction::InjectHidden { .. }));
        assert!(bridge.pending_injections.is_empty());

        // Queue a second injection for the same layer — should be preserved (no port match in pre_node since already consumed)
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![2.0, 2.0],
            scale: 1.0,
        });
        let action2 = bridge.pre_node(&ctx);
        // This should apply the second injection
        assert!(matches!(action2, CallbackAction::InjectHidden { .. }));
        assert!(bridge.pending_injections.is_empty());
    }

    #[test]
    fn pre_node_injection_at_multiple_layers_in_sequence() {
        // Arrange: injection ports at layers 0, 1, 2
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::injection(1, BusPortTag::Guardrail));
        bus.register(BusPort::injection(2, BusPortTag::ShadowKv));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);

        // Queue all three injections upfront
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 0.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![0.0, 1.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::ShadowKv,
            data: vec![1.0, 1.0],
            scale: 0.5,
        });

        assert_eq!(bridge.pending_injections.len(), 3);

        // Layer 0: RagInjection consumed
        let ctx0 = holder.ctx(0, 0);
        let a0 = bridge.pre_node(&ctx0);
        assert!(matches!(a0, CallbackAction::InjectHidden { .. }));
        assert_eq!(bridge.pending_injections.len(), 2);

        // Layer 1: Guardrail consumed
        let ctx1 = holder.ctx(1, 2);
        let a1 = bridge.pre_node(&ctx1);
        assert!(matches!(a1, CallbackAction::InjectHidden { .. }));
        assert_eq!(bridge.pending_injections.len(), 1);

        // Layer 2: ShadowKv consumed
        let ctx2 = holder.ctx(2, 4);
        let a2 = bridge.pre_node(&ctx2);
        assert!(matches!(a2, CallbackAction::InjectHidden { .. }));
        assert!(bridge.pending_injections.is_empty());
    }

    // ── New tests: post_node recalled accumulation across different layers ──

    #[test]
    fn post_node_recall_accumulates_across_layers() {
        // Arrange: recall ports at layers 0 and 2
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(2, BusPortTag::IntentRecall));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);

        // Layer 0 recall
        let ctx0 = holder.ctx(0, 0);
        let out0: Vec<f32> = vec![1.0, 0.0];
        let bytes0: Vec<u8> = out0.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx0, &bytes0);

        // Layer 1: no recall port
        let ctx1 = holder.ctx(1, 2);
        let out1: Vec<f32> = vec![2.0, 3.0];
        let bytes1: Vec<u8> = out1.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx1, &bytes1);

        // Layer 2 recall
        let ctx2 = holder.ctx(2, 4);
        let out2: Vec<f32> = vec![4.0, 5.0];
        let bytes2: Vec<u8> = out2.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx2, &bytes2);

        // Assert: 2 recalled entries (layer 0 + layer 2)
        assert_eq!(bridge.recalled_data.len(), 2);
        assert_eq!(bridge.recalled_data[0].tag, BusPortTag::EarlyExit);
        assert_eq!(bridge.recalled_data[0].layer, 0);
        assert_eq!(bridge.recalled_data[1].tag, BusPortTag::IntentRecall);
        assert_eq!(bridge.recalled_data[1].layer, 2);

        // prev_hidden should be from last post_node call
        assert_eq!(bridge.prev_hidden, vec![4.0, 5.0]);
    }

    // ── New tests: post_node energy with various vector patterns ──

    #[test]
    fn post_node_energy_single_element() {
        let mut bus = ResidualBus::new(1, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(1);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![7.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        bridge.post_node(&ctx, &output_bytes);

        // energy = sqrt(49) = 7.0
        assert!((bridge.recalled_data[0].energy - 7.0).abs() < 1e-4);
    }

    #[test]
    fn post_node_energy_large_values() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![1e6, 1e6];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        bridge.post_node(&ctx, &output_bytes);

        // energy = sqrt(1e12 + 1e12) = sqrt(2e12) ≈ 1_414_213.56
        let expected = (2.0f64 * 1e12f64).sqrt() as f32;
        assert!((bridge.recalled_data[0].energy - expected).abs() < 1.0);
        assert!(bridge.recalled_data[0].energy > 0.0);
    }

    #[test]
    fn post_node_energy_mixed_sign_elements() {
        let mut bus = ResidualBus::new(3, 4);
        bus.register(BusPort::recall(0, BusPortTag::IntentRecall));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(3);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![-3.0, 4.0, 0.0]; // sqrt(9+16+0) = 5.0
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        bridge.post_node(&ctx, &output_bytes);

        assert!((bridge.recalled_data[0].energy - 5.0).abs() < 1e-4);
    }

    // ── New tests: RecalledEntry construction with varied sizes ──

    #[test]
    fn recalled_entry_single_element_data() {
        let entry = RecalledEntry {
            tag: BusPortTag::RagInjection,
            layer: 0,
            data: vec![42.0],
            energy: 42.0,
        };
        assert_eq!(entry.data.len(), 1);
        assert_eq!(entry.data[0], 42.0);
    }

    #[test]
    fn recalled_entry_zero_energy_with_nonzero_data() {
        // Construction allows any energy value; it's not computed here
        let entry = RecalledEntry {
            tag: BusPortTag::Guardrail,
            layer: 5,
            data: vec![1.0, 2.0, 3.0],
            energy: 0.0,
        };
        assert_eq!(entry.data.len(), 3);
        assert_eq!(entry.energy, 0.0);
    }

    #[test]
    fn recalled_entry_clone_equality_chain() {
        let entry = RecalledEntry {
            tag: BusPortTag::Custom(10),
            layer: 3,
            data: vec![1.0, 2.0],
            energy: 2.236,
        };
        let c1 = entry.clone();
        let c2 = c1.clone();
        assert_eq!(entry, c1);
        assert_eq!(c1, c2);
        assert_eq!(entry, c2);
    }

    // ── New tests: injection + recall full lifecycle edge cases ──

    #[test]
    fn lifecycle_injection_at_layer_zero_recall_at_last_layer() {
        let mut bus = ResidualBus::new(2, 5);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(4, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);

        // Injection at layer 0
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![5.0, 10.0],
            scale: 0.5,
        });
        let ctx0 = holder.ctx(0, 0);
        let pre0 = bridge.pre_node(&ctx0);
        assert!(matches!(pre0, CallbackAction::InjectHidden { .. }));

        // Recall at layer 4
        let ctx4 = holder.ctx(4, 8);
        let output: Vec<f32> = vec![100.0, 200.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx4, &output_bytes);

        assert_eq!(bridge.recalled_data.len(), 1);
        assert_eq!(bridge.recalled_data[0].tag, BusPortTag::EarlyExit);
        assert_eq!(bridge.recalled_data[0].layer, 4);
        assert_eq!(bridge.recalled_data[0].data, vec![100.0, 200.0]);
    }

    #[test]
    fn lifecycle_no_injection_no_recall_clean_pass() {
        let bus = ResidualBus::new(3, 5);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(3);

        for layer in 0..5 {
            let ctx = holder.ctx(layer, layer * 2);
            assert_eq!(bridge.pre_node(&ctx), CallbackAction::Continue);

            let output: Vec<f32> = vec![1.0, 2.0, 3.0];
            let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
            assert_eq!(bridge.post_node(&ctx, &output_bytes), CallbackAction::Continue);
        }

        assert!(bridge.recalled_data.is_empty());
        assert!(bridge.pending_injections.is_empty());
        // prev_hidden updated on every post_node
        assert_eq!(bridge.prev_hidden, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn lifecycle_queue_injection_after_drain() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);

        // First pass: recall
        let ctx = holder.ctx(0, 0);
        let bytes: Vec<u8> = vec![1.0f32, 2.0f32].iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &bytes);

        let drained = bridge.drain_recalled();
        assert_eq!(drained.len(), 1);

        // Queue injection (different port, but no injection port registered)
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![3.0, 4.0],
            scale: 1.0,
        });
        // No injection port → pre_node returns Continue, payload preserved
        let action = bridge.pre_node(&ctx);
        assert_eq!(action, CallbackAction::Continue);
        assert_eq!(bridge.pending_injections.len(), 1);
    }

    // ── New tests: BusPort additional coverage ──

    #[test]
    fn bus_port_injection_creation_all_fields() {
        let port = BusPort::injection(3, BusPortTag::Guardrail);
        assert_eq!(port.kind, BusPortKind::Injection);
        assert_eq!(port.layer, 3);
        assert_eq!(port.tag, BusPortTag::Guardrail);
        assert!(port.is_active());
    }

    #[test]
    fn bus_port_deactivate_is_idempotent() {
        let port = BusPort::recall(0, BusPortTag::EarlyExit);
        port.deactivate();
        assert!(!port.is_active());
        port.deactivate(); // second call should not panic
        assert!(!port.is_active());
    }

    #[test]
    fn bus_port_activate_is_idempotent() {
        let port = BusPort::injection(5, BusPortTag::ShadowKv);
        assert!(port.is_active());
        port.activate(); // already active, should not panic
        assert!(port.is_active());
    }

    #[test]
    fn bus_port_tag_hash_in_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        assert!(set.insert(BusPortTag::Custom(1)));
        assert!(!set.insert(BusPortTag::Custom(1))); // duplicate
        assert!(set.insert(BusPortTag::Custom(2)));
        assert_eq!(set.len(), 2);
    }

    // ── New tests: InjectionPayload additional edge cases ──

    #[test]
    fn injection_payload_zero_scale_with_large_data() {
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![f32::MAX; 100],
            scale: 0.0,
        };
        assert_eq!(payload.data.len(), 100);
        assert_eq!(payload.scale, 0.0);
    }

    #[test]
    fn injection_payload_negative_scale_value() {
        let payload = InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0],
            scale: -100.0,
        };
        assert!((payload.scale - (-100.0)).abs() < 1e-6);
    }

    #[test]
    fn injection_payload_min_positive_scale() {
        let payload = InjectionPayload {
            target: BusPortTag::ShadowKv,
            data: vec![1.0],
            scale: f32::MIN_POSITIVE,
        };
        assert_eq!(payload.scale, f32::MIN_POSITIVE);
    }

    #[test]
    fn injection_payload_all_named_tags() {
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
        ];
        for tag in &tags {
            let payload = InjectionPayload {
                target: *tag,
                data: vec![1.0],
                scale: 1.0,
            };
            assert_eq!(payload.target, *tag);
        }
    }

    // ── New tests: pre_node hidden state correctness with multi-injection ──

    #[test]
    fn pre_node_three_matching_payloads_cumulative() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(2);
        // Initial hidden: [0.0, 0.0]

        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 0.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![0.0, 1.0],
            scale: 2.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![10.0, 10.0],
            scale: 0.1,
        });

        let ctx = holder.ctx(0, 0);
        let action = bridge.pre_node(&ctx);

        match action {
            CallbackAction::InjectHidden { data } => {
                let decoded = ResidualBusBridgeCallback::bytes_to_f32(&data);
                // 0 + 1*1 = 1.0, 0 + 0*1 = 0.0
                // + 0*2 = 0.0, + 1*2 = 2.0
                // + 10*0.1 = 1.0, + 10*0.1 = 1.0
                // total: [2.0, 3.0]
                assert!((decoded[0] - 2.0).abs() < 1e-5, "expected 2.0 got {}", decoded[0]);
                assert!((decoded[1] - 3.0).abs() < 1e-5, "expected 3.0 got {}", decoded[1]);
            }
            _ => panic!("Expected InjectHidden, got {:?}", action),
        }
        assert!(bridge.pending_injections.is_empty());
    }

    #[test]
    fn pre_node_injection_consumed_mismatch_preserved_together() {
        // Mix of matching and mismatching payloads
        let mut bus = ResidualBus::new(3, 4);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(3);

        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 1.0, 1.0],
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0], // dimension mismatch
            scale: 1.0,
        });
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::Guardrail,
            data: vec![1.0, 1.0, 1.0],
            scale: 1.0,
        });

        let ctx = holder.ctx(0, 0);
        let action = bridge.pre_node(&ctx);

        // First payload applied, second mismatched, third wrong tag
        assert!(matches!(action, CallbackAction::InjectHidden { .. }));
        // Both mismatched payloads preserved
        assert_eq!(bridge.pending_injections.len(), 2);
    }

    // ── New tests: post_node with various output patterns ──

    #[test]
    fn post_node_recalled_data_independence_from_prev_hidden() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![3.0, 4.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &output_bytes);

        // Modify prev_hidden to prove recalled_data is independent
        bridge.prev_hidden[0] = 999.0;
        assert_eq!(bridge.recalled_data[0].data[0], 3.0); // unchanged
    }

    #[test]
    fn post_node_recall_with_very_small_values() {
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::recall(0, BusPortTag::EarlyExit));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![f32::MIN_POSITIVE, f32::MIN_POSITIVE];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &output_bytes);

        // Energy should be positive (may underflow to 0 for subnormal squares)
        let energy = bridge.recalled_data[0].energy;
        assert!(energy >= 0.0);
        // sqrt(x^2 + x^2) <= |x| * sqrt(2) for any x
        let upper_bound = f32::MIN_POSITIVE * 2.0f32.sqrt();
        assert!(energy <= upper_bound);
    }

    // ── New tests: CallbackAction Debug trait ──

    #[test]
    fn callback_action_debug_continue() {
        let s = format!("{:?}", CallbackAction::Continue);
        assert!(s.contains("Continue"), "Debug should contain 'Continue': {}", s);
    }

    #[test]
    fn callback_action_debug_skip_this_node() {
        let s = format!("{:?}", CallbackAction::SkipThisNode);
        assert!(!s.is_empty());
    }

    #[test]
    fn callback_action_debug_exit_early() {
        let action = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        let s = format!("{:?}", action);
        assert!(!s.is_empty());
    }

    #[test]
    fn callback_action_debug_compact_mask() {
        let action = CallbackAction::CompactMask { active_mask: vec![true, false] };
        let s = format!("{:?}", action);
        assert!(!s.is_empty());
    }

    // ── New tests: bridge construction edge cases ──

    #[test]
    fn bridge_from_bus_zero_hidden_size() {
        let bus = ResidualBus::new(0, 8);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.hidden_size, 0);
        assert!(bridge.injection_layers.is_empty());
        assert!(bridge.recall_layers.is_empty());
    }

    #[test]
    fn bridge_from_bus_with_many_layers() {
        let mut bus = ResidualBus::new(4, 500);
        bus.register(BusPort::injection(0, BusPortTag::RagInjection));
        bus.register(BusPort::injection(499, BusPortTag::Guardrail));
        bus.register(BusPort::recall(250, BusPortTag::EarlyExit));
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.injection_layers.len(), 2);
        assert_eq!(bridge.recall_layers.len(), 1);
        assert_eq!(bridge.hidden_size, 4);
    }

    #[test]
    fn bridge_name_is_static_str() {
        let bus = ResidualBus::new(4, 8);
        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        // Verify name returns a valid &str
        assert_eq!(bridge.name(), "residual_bus_bridge");
        assert!(!bridge.name().is_empty());
    }

    // ── New tests: LayerContext field access ──

    #[test]
    fn layer_context_fields_accessible() {
        let holder = TestCtxHolder::with_hidden_len(4);
        let ctx = holder.ctx(3, 7);

        assert_eq!(ctx.layer_idx, 3);
        assert_eq!(ctx.node_idx, 7);
        assert_eq!(ctx.node_op, "BridgeTest");
        assert_eq!(ctx.hidden_state.len(), 16);
        assert_eq!(ctx.total_seq, 10);
        assert_eq!(ctx.seq_len, 1);
        assert_eq!(ctx.position, 9);
        assert_eq!(ctx.request_id, 1);
    }

    #[test]
    fn layer_context_different_layer_indices() {
        let holder = TestCtxHolder::with_hidden_len(2);
        for layer in [0, 1, 5, 99] {
            let ctx = holder.ctx(layer, layer * 2);
            assert_eq!(ctx.layer_idx, layer);
            assert_eq!(ctx.node_idx, layer * 2);
        }
    }

    // ── New tests: ResidualBusError additional coverage ──

    #[test]
    fn residual_bus_error_port_not_found_is_error() {
        let e = ResidualBusError::PortNotFound(BusPortTag::Custom(999));
        assert!(std::error::Error::source(&e).is_none());
    }

    #[test]
    fn residual_bus_error_port_inactive_tag_preserved() {
        let tag = BusPortTag::ShadowKv;
        let e = ResidualBusError::PortInactive(tag);
        match e {
            ResidualBusError::PortInactive(t) => assert_eq!(t, tag),
            _ => panic!("Expected PortInactive"),
        }
    }

    #[test]
    fn residual_bus_error_wrong_port_type_fields() {
        let e = ResidualBusError::WrongPortType {
            expected: BusPortKind::Recall,
            actual: BusPortKind::Injection,
        };
        match e {
            ResidualBusError::WrongPortType { expected, actual } => {
                assert_eq!(expected, BusPortKind::Recall);
                assert_eq!(actual, BusPortKind::Injection);
            }
            _ => panic!("Expected WrongPortType"),
        }
    }

    #[test]
    fn residual_bus_error_dimension_mismatch_fields() {
        let e = ResidualBusError::DimensionMismatch { expected: 128, actual: 64 };
        match e {
            ResidualBusError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, 128);
                assert_eq!(actual, 64);
            }
            _ => panic!("Expected DimensionMismatch"),
        }
    }

    // ── Additional 9 tests to reach ~55 new tests ──

    #[test]
    fn pre_node_hidden_state_preserved_when_no_injection() {
        // Arrange: injection port at layer 2, but we call pre_node at layer 0
        let mut bus = ResidualBus::new(3, 8);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let mut holder = TestCtxHolder::with_hidden_len(3);
        let original: Vec<f32> = vec![7.0, 8.0, 9.0];
        for (i, v) in original.iter().enumerate() {
            let bytes = v.to_le_bytes();
            holder.hidden_state[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }

        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 1.0, 1.0],
            scale: 1.0,
        });

        // Act: call pre_node at layer 0 (no matching port)
        let ctx = holder.ctx(0, 0);
        let action = bridge.pre_node(&ctx);

        // Assert: Continue, and hidden_state not modified (no InjectHidden returned)
        assert_eq!(action, CallbackAction::Continue);
        // Payload preserved
        assert_eq!(bridge.pending_injections.len(), 1);
    }

    #[test]
    fn post_node_energy_subnormal_output() {
        // Output with subnormal values should still compute energy without panic
        let mut bus = ResidualBus::new(2, 4);
        bus.register(BusPort::recall(0, BusPortTag::ShadowKv));
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);

        let holder = TestCtxHolder::with_hidden_len(2);
        let ctx = holder.ctx(0, 0);

        let sub = f32::from_bits(1); // smallest positive subnormal
        let output: Vec<f32> = vec![sub, sub];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Act: should not panic
        bridge.post_node(&ctx, &output_bytes);

        // Assert: energy is non-negative
        assert!(bridge.recalled_data[0].energy >= 0.0);
        // Data preserved exactly
        assert_eq!(bridge.recalled_data[0].data[0].to_bits(), 1u32);
    }

    #[test]
    fn bridge_from_bus_with_all_tag_variants_as_recall() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::recall(0, BusPortTag::RagInjection));
        bus.register(BusPort::recall(1, BusPortTag::EarlyExit));
        bus.register(BusPort::recall(2, BusPortTag::IntentRecall));
        bus.register(BusPort::recall(3, BusPortTag::Guardrail));
        bus.register(BusPort::recall(4, BusPortTag::ShadowKv));
        bus.register(BusPort::recall(5, BusPortTag::Custom(255)));

        let bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert_eq!(bridge.recall_layers.len(), 6);
        assert!(bridge.injection_layers.is_empty());

        let tags: Vec<BusPortTag> = bridge.recall_layers.iter().map(|(_, t)| *t).collect();
        assert!(tags.contains(&BusPortTag::Custom(255)));
    }

    #[test]
    fn callback_action_inject_hidden_with_large_data() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let action = CallbackAction::InjectHidden { data: data.clone() };
        assert!(matches!(action, CallbackAction::InjectHidden { .. }));
        if let CallbackAction::InjectHidden { data: d } = action {
            assert_eq!(d.len(), 1024);
        }
    }

    #[test]
    fn callback_action_compact_mask_all_active() {
        let action = CallbackAction::CompactMask { active_mask: vec![true; 100] };
        if let CallbackAction::CompactMask { active_mask } = action {
            assert!(active_mask.iter().all(|&a| a));
            assert_eq!(active_mask.len(), 100);
        }
    }

    #[test]
    fn callback_action_exit_early_single_logit() {
        let action = CallbackAction::ExitEarly { logits: vec![42.0] };
        if let CallbackAction::ExitEarly { logits } = action {
            assert_eq!(logits.len(), 1);
            assert!((logits[0] - 42.0).abs() < 1e-6);
        }
    }

    #[test]
    fn lifecycle_post_node_prev_hidden_length_matches_hidden_size() {
        let bus = ResidualBus::new(5, 4);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);
        assert!(bridge.prev_hidden.is_empty());

        let holder = TestCtxHolder::with_hidden_len(5);
        let ctx = holder.ctx(0, 0);

        let output: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output_bytes: Vec<u8> = output.iter().flat_map(|f| f.to_le_bytes()).collect();
        bridge.post_node(&ctx, &output_bytes);

        assert_eq!(bridge.prev_hidden.len(), 5);
    }

    #[test]
    fn bus_port_kind_debug_contains_variant_name() {
        let s_injection = format!("{:?}", BusPortKind::Injection);
        let s_recall = format!("{:?}", BusPortKind::Recall);
        assert!(s_injection.contains("Injection"));
        assert!(s_recall.contains("Recall"));
    }

    #[test]
    fn injection_payload_clone_data_independence() {
        let mut payload = InjectionPayload {
            target: BusPortTag::Custom(3),
            data: vec![1.0, 2.0],
            scale: 1.0,
        };
        let cloned = payload.clone();
        payload.data[0] = 999.0;
        assert!((cloned.data[0] - 1.0).abs() < 1e-6, "clone should be deep");
        assert!((payload.data[0] - 999.0).abs() < 1e-6);
    }

    // ── BusPort Debug trait ──

    #[test]
    fn bus_port_injection_debug_non_empty() {
        let port = BusPort::injection(3, BusPortTag::Guardrail);
        let s = format!("{:?}", port);
        assert!(!s.is_empty());
    }

    #[test]
    fn bus_port_recall_debug_non_empty() {
        let port = BusPort::recall(7, BusPortTag::IntentRecall);
        let s = format!("{:?}", port);
        assert!(!s.is_empty());
    }

    // ── InjectionPayload Debug trait ──

    #[test]
    fn injection_payload_debug_non_empty() {
        let payload = InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![1.0, 2.0],
            scale: 0.5,
        };
        let s = format!("{:?}", payload);
        assert!(!s.is_empty());
    }

    // ── BusPortTag Debug trait ──

    #[test]
    fn bus_port_tag_debug_all_named_non_empty() {
        let tags = [
            BusPortTag::RagInjection,
            BusPortTag::EarlyExit,
            BusPortTag::IntentRecall,
            BusPortTag::Guardrail,
            BusPortTag::ShadowKv,
        ];
        for tag in &tags {
            let s = format!("{:?}", tag);
            assert!(!s.is_empty(), "Debug for {:?} should be non-empty", tag);
        }
    }

    #[test]
    fn bus_port_tag_debug_custom_non_empty() {
        let tag = BusPortTag::Custom(42);
        let s = format!("{:?}", tag);
        assert!(!s.is_empty());
    }

    // ── ResidualBus Debug trait ──

    #[test]
    fn residual_bus_debug_non_empty() {
        let bus = ResidualBus::new(64, 8);
        let s = format!("{:?}", bus);
        assert!(!s.is_empty());
    }

    // ── ResidualBus direct API access ──

    #[test]
    fn residual_bus_hidden_size_direct() {
        let bus = ResidualBus::new(256, 12);
        assert_eq!(bus.hidden_size(), 256);
    }

    #[test]
    fn residual_bus_num_layers_direct() {
        let bus = ResidualBus::new(64, 32);
        assert_eq!(bus.num_layers(), 32);
    }

    #[test]
    fn residual_bus_ports_initially_empty() {
        let bus = ResidualBus::new(64, 8);
        assert!(bus.ports().is_empty());
    }

    #[test]
    fn residual_bus_ports_len_after_register() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));
        assert_eq!(bus.ports().len(), 2);
    }

    #[test]
    fn residual_bus_active_port_count_matches() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(1, BusPortTag::RagInjection));
        bus.register(BusPort::injection(3, BusPortTag::Guardrail));
        bus.register(BusPort::recall(7, BusPortTag::EarlyExit));
        assert_eq!(bus.active_port_count(), 3);
    }

    #[test]
    fn residual_bus_ports_ordering_ascending() {
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(7, BusPortTag::Guardrail));
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));
        let layers: Vec<usize> = bus.ports().iter().map(|p| p.layer).collect();
        assert_eq!(layers, vec![2, 5, 7]);
    }

    // ── CallbackAction Clone deep copy for data variants ──

    #[test]
    fn callback_action_clone_exit_early_deep_copy() {
        let action = CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] };
        let mut cloned = action.clone();
        if let CallbackAction::ExitEarly { logits } = &mut cloned {
            logits[0] = 999.0;
        }
        if let CallbackAction::ExitEarly { logits } = &action {
            assert!((logits[0] - 1.0).abs() < 1e-6, "original should be unchanged");
        }
    }

    #[test]
    fn callback_action_clone_inject_hidden_deep_copy() {
        let action = CallbackAction::InjectHidden { data: vec![1u8, 2, 3] };
        let mut cloned = action.clone();
        if let CallbackAction::InjectHidden { data } = &mut cloned {
            data[0] = 255;
        }
        if let CallbackAction::InjectHidden { data } = &action {
            assert_eq!(data[0], 1, "original should be unchanged");
        }
    }

    #[test]
    fn callback_action_clone_compact_mask_deep_copy() {
        let action = CallbackAction::CompactMask { active_mask: vec![true, false] };
        let mut cloned = action.clone();
        if let CallbackAction::CompactMask { active_mask } = &mut cloned {
            active_mask[0] = false;
        }
        if let CallbackAction::CompactMask { active_mask } = &action {
            assert!(active_mask[0], "original should be unchanged");
        }
    }

    #[test]
    fn callback_action_clone_primitive_variants() {
        assert_eq!(CallbackAction::Continue.clone(), CallbackAction::Continue);
        assert_eq!(CallbackAction::SkipThisNode.clone(), CallbackAction::SkipThisNode);
    }

    // ── BusPortTag as HashMap key ──

    #[test]
    fn bus_port_tag_as_hashmap_key() {
        let mut map = std::collections::HashMap::new();
        map.insert(BusPortTag::RagInjection, 1);
        map.insert(BusPortTag::Custom(99), 2);
        assert_eq!(map.get(&BusPortTag::RagInjection), Some(&1));
        assert_eq!(map.get(&BusPortTag::Custom(99)), Some(&2));
        assert_eq!(map.get(&BusPortTag::Guardrail), None);
        assert_eq!(map.len(), 2);
    }

    // ── BusPort active field direct read ──

    #[test]
    fn bus_port_active_field_atomic_read() {
        let port = BusPort::injection(0, BusPortTag::RagInjection);
        assert_eq!(port.active.load(std::sync::atomic::Ordering::SeqCst), 1);
        port.deactivate();
        assert_eq!(port.active.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    // ── 13 additional tests for edge case coverage ──

    #[test]
    fn residual_bus_error_display_port_not_found() {
        // Arrange: construct PortNotFound error, verify Display output
        let e = ResidualBusError::PortNotFound(BusPortTag::Custom(42));
        let msg = format!("{}", e);
        // Assert: message contains identifying information
        assert!(msg.contains("not found"), "Display should mention 'not found': {}", msg);
    }

    #[test]
    fn residual_bus_error_display_port_inactive() {
        // Arrange
        let e = ResidualBusError::PortInactive(BusPortTag::Guardrail);
        let msg = format!("{}", e);
        // Assert
        assert!(msg.contains("inactive"), "Display should mention 'inactive': {}", msg);
    }

    #[test]
    fn residual_bus_error_display_wrong_port_type() {
        // Arrange
        let e = ResidualBusError::WrongPortType {
            expected: BusPortKind::Injection,
            actual: BusPortKind::Recall,
        };
        let msg = format!("{}", e);
        // Assert: mentions both expected and actual
        assert!(msg.contains("Injection"), "Display should mention Injection: {}", msg);
        assert!(msg.contains("Recall"), "Display should mention Recall: {}", msg);
    }

    #[test]
    fn residual_bus_error_display_dimension_mismatch() {
        // Arrange
        let e = ResidualBusError::DimensionMismatch { expected: 256, actual: 128 };
        let msg = format!("{}", e);
        // Assert: contains both numeric values
        assert!(msg.contains("256"), "Display should contain expected: {}", msg);
        assert!(msg.contains("128"), "Display should contain actual: {}", msg);
    }

    #[test]
    fn recall_payload_construction_and_field_access() {
        // Arrange: construct RecallPayload via routing type
        let payload = crate::routing::RecallPayload {
            source: BusPortTag::EarlyExit,
            data: vec![1.0, 2.0, 3.0],
            meta: crate::routing::RecallMeta {
                layer: 5,
                energy: 3.742,
                cosine_sim: 0.98,
                entropy: 1.5,
            },
        };
        // Assert: field accessors return correct values
        assert_eq!(payload.source, BusPortTag::EarlyExit);
        assert_eq!(payload.data.len(), 3);
        assert_eq!(payload.meta.layer, 5);
        assert!((payload.meta.energy - 3.742).abs() < 1e-3);
        assert!((payload.meta.cosine_sim - 0.98).abs() < 1e-3);
        assert!((payload.meta.entropy - 1.5).abs() < 1e-3);
    }

    #[test]
    fn recall_meta_clone_deep_independence() {
        // Arrange
        let meta = crate::routing::RecallMeta {
            layer: 3,
            energy: 5.0,
            cosine_sim: 0.75,
            entropy: 2.0,
        };
        // Act
        let cloned = meta.clone();
        // Assert: cloned values match original
        assert_eq!(cloned.layer, meta.layer);
        assert!((cloned.energy - meta.energy).abs() < 1e-6);
        assert!((cloned.cosine_sim - meta.cosine_sim).abs() < 1e-6);
        assert!((cloned.entropy - meta.entropy).abs() < 1e-6);
    }

    #[test]
    fn recall_payload_clone_deep_independence() {
        // Arrange
        let mut payload = crate::routing::RecallPayload {
            source: BusPortTag::IntentRecall,
            data: vec![1.0, 2.0],
            meta: crate::routing::RecallMeta {
                layer: 0,
                energy: 0.0,
                cosine_sim: 1.0,
                entropy: 0.0,
            },
        };
        // Act
        let cloned = payload.clone();
        payload.data.push(3.0);
        // Assert: clone is deep — modifying original does not affect clone
        assert_eq!(cloned.data.len(), 2);
        assert_eq!(payload.data.len(), 3);
        assert_eq!(cloned.source, BusPortTag::IntentRecall);
    }

    #[test]
    fn residual_bus_active_ports_at_layer_multi_port() {
        // Arrange: two active injection ports at the same layer
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        bus.register(BusPort::injection(3, BusPortTag::Guardrail));
        bus.register(BusPort::recall(5, BusPortTag::EarlyExit));
        // Act: query active ports at layer 3
        let ports: Vec<&BusPort> = bus.active_ports_at_layer(3).collect();
        // Assert: both injection ports found
        assert_eq!(ports.len(), 2);
        assert_eq!(ports[0].kind, BusPortKind::Injection);
        assert_eq!(ports[1].kind, BusPortKind::Injection);
    }

    #[test]
    fn residual_bus_active_ports_at_layer_empty() {
        // Arrange: no ports at layer 7
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(3, BusPortTag::RagInjection));
        // Act
        let ports: Vec<&BusPort> = bus.active_ports_at_layer(7).collect();
        // Assert
        assert!(ports.is_empty());
    }

    #[test]
    fn residual_bus_find_port_mut_changes_state() {
        // Arrange
        let mut bus = ResidualBus::new(4, 10);
        bus.register(BusPort::injection(2, BusPortTag::RagInjection));
        // Act: deactivate via mutable reference
        if let Some(port) = bus.find_port_mut(BusPortTag::RagInjection) {
            port.deactivate();
        }
        // Assert: port is now inactive
        let port = bus.find_port(BusPortTag::RagInjection);
        assert!(port.is_some());
        assert!(!port.unwrap().is_active());
    }

    #[test]
    fn bytes_to_f32_ieee754_specific_bit_pattern() {
        // Arrange: 0x40490FDB = 3.14159274... (pi approximation in f32)
        let bytes: Vec<u8> = vec![0xDB, 0x0F, 0x49, 0x40];
        // Act
        let decoded = ResidualBusBridgeCallback::bytes_to_f32(&bytes);
        // Assert
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - std::f32::consts::PI).abs() < 1e-6);
    }

    #[test]
    fn bridge_recalled_entry_debug_contains_all_field_names() {
        // Arrange
        let entry = RecalledEntry {
            tag: BusPortTag::EarlyExit,
            layer: 5,
            data: vec![1.0, 2.0],
            energy: 2.236,
        };
        // Act
        let debug = format!("{:?}", entry);
        // Assert: Debug output should contain the struct name and show fields
        assert!(debug.contains("RecalledEntry"), "should contain struct name: {}", debug);
        assert!(debug.contains("layer"), "should contain 'layer' field: {}", debug);
    }

    #[test]
    fn bridge_queue_injection_empty_data_payload() {
        // Arrange: payload with empty data vector
        let bus = ResidualBus::new(4, 8);
        let mut bridge = ResidualBusBridgeCallback::from_bus(&bus);
        // Act
        bridge.queue_injection(InjectionPayload {
            target: BusPortTag::RagInjection,
            data: vec![],
            scale: 1.0,
        });
        // Assert: queued successfully even with empty data
        assert_eq!(bridge.pending_injections.len(), 1);
        assert!(bridge.pending_injections[0].data.is_empty());
        assert!((bridge.pending_injections[0].scale - 1.0).abs() < 1e-6);
    }
}
