//! Per-Node Callback Implementations (SPEC §9-§18)
//!
//! Concrete LayerCallback implementations that bridge existing optimization
//! modules to the mega-kernel node loop via the callback system.
//!
//! ## Callback Priority Table (per SPEC/05-OPTIMIZATIONS.md §8)
//!
//! | Priority | Callback | Trigger | Action |
//! |----------|----------|---------|--------|
//! | 100 | Prefetch | pre_node | KV cache prefetch |
//! | 90 | Semantic Gatekeeper | pre_node | InjectHidden (residual add) |
//! | 80 | RAG Inject | pre_node | InjectHidden |
//! | 70 | MoE Dispatch | pre_node | Expert routing |
//! | 60 | Gate Skip | pre_node | CompactMask (§14.2) |
//! | 50 | Early Exit | post_node | ExitEarly |
//! | 20 | Residual Bypass | pre_node | SkipThisNode |

pub mod early_exit;
pub mod gate_skip;
pub mod guardrail_probe;
pub mod mid_layer_encode;
pub mod moe_dispatch;
pub mod rag_inject;
pub mod residual_bus_bridge;

pub use early_exit::EarlyExitCallback;
pub use gate_skip::GateSkipCallback;
pub use guardrail_probe::GuardrailProbeCallback;
pub use mid_layer_encode::MidLayerEncodeCallback;
pub use moe_dispatch::MoeDispatchCallback;
pub use rag_inject::RagInjectCallback;
pub use residual_bus_bridge::ResidualBusBridgeCallback;

use crate::graph::layer_callback::{CallbackChain, LayerCallback};

/// Build a callback chain from a list of boxed callbacks.
///
/// Convenience function that wraps `CallbackChain::new()`.
pub fn build_callback_chain(callbacks: Vec<Box<dyn LayerCallback + Send>>) -> CallbackChain {
    CallbackChain::new(callbacks)
}
