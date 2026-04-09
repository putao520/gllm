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
    hidden_size: usize,
    /// Previous layer's hidden state for recall cosine similarity
    prev_hidden: Vec<f32>,
}

/// Entry from a recall port
#[derive(Debug, Clone)]
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
    use crate::routing::{BusPort, BusPortTag, ResidualBus};

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
}
