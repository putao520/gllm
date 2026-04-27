//! Per-Node Callback Infrastructure (SPEC §9-§18)
//!
//! Provides a zero-overhead callback mechanism for mid-layer optimization hooks.
//! The mega-kernel node loop calls `pre_node()` / `post_node()` between
//! each JIT-compiled kernel, enabling:
//!
//! - **Early Exit** (§16.2): Stop execution after confidence threshold
//! - **Gate Skip** (§13.1): Skip FFN layers when dead_density is high
//! - **RAG Injection** (§16.1): Inject retrieved context at fusion layer
//! - **Knowledge Injection** (§8.1): Inject knowledge vectors at target layer
//! - **Guardrail Probe** (§16.4): Safety classification at target layer
//! - **Intent Recall** (§16.3): Extract intent embedding at target layer
//! - **MoE Dispatch** (§15): Expert routing and prefetch at MoE layers
//! - **Residual Bypass** (§13.3): Skip residual connections when delta_rho is low
//!
//! ## Zero-Overhead Guarantee
//!
//! When no callbacks are registered, the execution path is identical to the
//! original `run_with_kv_cache()` — a single `if callbacks.is_empty()` guard.

use crate::engine::executor::GeneratorForwardConfig;

// ============================================================================
// Callback Action — 回调返回值
// ============================================================================

/// Action returned by a callback to control execution flow.
#[derive(Debug, Clone)]
pub enum CallbackAction {
    /// Continue normal execution (no intervention).
    Continue,
    /// Skip the current node entirely (used by Residual Bypass when delta_rho < threshold).
    ///
    /// Per SPEC §14.3: only used for per-request block-level residual bypass,
    /// NOT for gate/FFN skip (which must use CompactMask per §14.2).
    SkipThisNode,
    /// Exit execution early with the provided logits (used by Early Exit, Guardrail).
    ///
    /// The `logits` field contains pre-softmax scores for the last token position.
    /// When empty, the caller should project the current hidden_state through lm_head.
    ExitEarly {
        logits: Vec<f32>,
    },
    /// Replace the hidden state before this node executes (used by RAG, Knowledge injection).
    ///
    /// The `data` field contains the modified hidden state in the model's dtype.
    InjectHidden {
        data: Vec<u8>,
    },
    /// Register-level compaction: compact active neurons using hardware masks (SPEC §14.2).
    ///
    /// Dead neurons are compacted out via hardware predicated execution (AVX-512 vcompress,
    /// GPU prefix sum, SVE predicate). The FFN executes on the dense compacted tensor at
    /// full throughput, then results are scattered back to original positions.
    ///
    /// `active_mask` is a boolean mask per neuron: true = active, false = dead.
    /// The executor must apply RaggedCompaction (Compact→Execute→Scatter) instead of skipping.
    CompactMask {
        active_mask: Vec<bool>,
    },
}

impl Default for CallbackAction {
    fn default() -> Self {
        Self::Continue
    }
}

// ============================================================================
// Layer Context — 传递给回调的上下文
// ============================================================================

/// Context provided to each callback invocation.
///
/// Contains all information needed for a callback to make a decision:
/// current layer/node indices, hidden state, KV cache pointers, and model config.
pub struct LayerContext<'a> {
    /// Current graph node index (0-based).
    pub node_idx: usize,
    /// Estimated layer index derived from node_idx.
    /// For typical transformer graphs: attention_node → layer = node_idx / 2,
    /// ffn_node → same layer = node_idx / 2.
    pub layer_idx: usize,
    /// Reference to the node's operation name.
    pub node_op: &'a str,
    /// Current hidden state tensor (byte representation, model dtype).
    pub hidden_state: &'a [u8],
    /// KV cache K-half pointer (all layers, flat buffer).
    pub kv_cache_k: *mut f32,
    /// KV cache V-half pointer (all layers, flat buffer).
    pub kv_cache_v: *mut f32,
    /// Total sequence length (cached + new tokens).
    pub total_seq: usize,
    /// New sequence length (tokens being processed in this step).
    pub seq_len: usize,
    /// Starting position of new tokens in the sequence.
    pub position: usize,
    /// Request ID for this sequence.
    pub request_id: u64,
    /// Static model configuration.
    pub model_config: &'a GeneratorForwardConfig,
}

// SAFETY: LayerContext holds raw pointers to KV cache buffers that are
// only accessed during the synchronous execution of the node loop.
unsafe impl Send for LayerContext<'_> {}

// ============================================================================
// Layer Callback Trait
// ============================================================================

/// Trait for mid-layer optimization callbacks.
///
/// Implementations are called between graph node executions in
/// `run_with_kv_cache_with_callbacks()` on the mega-kernel path.
///
/// # Priority
///
/// Callbacks are sorted by priority (highest first). Typical order:
/// - 100: Prefetch
/// - 90: Knowledge Inject
/// - 80: RAG Inject
/// - 70: MoE Dispatch
/// - 60: Gate Skip
/// - 50: Early Exit
/// - 40: Guardrail Probe
/// - 30: Intent Recall
/// - 20: Residual Bypass
/// - 10: Telemetry
///
/// # Target Layers
///
/// If `target_layers()` returns `Some(layers)`, the callback is only invoked
/// for nodes in those layers. `None` means all layers.
pub trait LayerCallback {
    /// Called before a graph node executes.
    ///
    /// Can return `SkipThisNode` to skip the node, `InjectHidden` to modify
    /// the hidden state, or `Continue` to proceed normally.
    fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Called after a graph node executes.
    ///
    /// Receives the output tensor from the node. Can return `ExitEarly` to
    /// stop execution, or `Continue` to proceed.
    fn post_node(&mut self, _ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
        CallbackAction::Continue
    }

    /// Callback priority (higher = called first).
    fn priority(&self) -> u32 {
        0
    }

    /// Layers this callback targets. `None` = all layers.
    fn target_layers(&self) -> Option<&[usize]> {
        None
    }

    /// Human-readable name for logging.
    fn name(&self) -> &str {
        "unnamed"
    }
}

// ============================================================================
// Callback Chain — 多回调管理
// ============================================================================

/// Manages a sorted list of callbacks and dispatches them efficiently.
///
/// Callbacks are sorted by priority (highest first) at construction time.
/// For each node, only callbacks whose `target_layers` include the current
/// layer are invoked.
pub struct CallbackChain {
    callbacks: Vec<Box<dyn LayerCallback + Send>>,
    /// Cached sorted order (by priority, descending).
    sorted_indices: Vec<usize>,
}

impl CallbackChain {
    /// Create an empty chain (zero-overhead when no optimizations are enabled).
    pub fn empty() -> Self {
        Self {
            callbacks: Vec::new(),
            sorted_indices: Vec::new(),
        }
    }

    /// Create a chain from a list of callbacks, sorted by priority.
    pub fn new(callbacks: Vec<Box<dyn LayerCallback + Send>>) -> Self {
        let mut sorted_indices: Vec<usize> = (0..callbacks.len()).collect();
        // Sort by priority descending (highest priority first)
        sorted_indices.sort_by(|&a, &b| {
            callbacks[b].priority().cmp(&callbacks[a].priority())
        });
        Self {
            callbacks,
            sorted_indices,
        }
    }

    /// Returns true if no callbacks are registered.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }

    /// Returns the number of registered callbacks.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Dispatch `pre_node()` to all applicable callbacks for the given layer.
    ///
    /// Returns the first non-Continue action, or Continue if all callbacks agree.
    pub fn dispatch_pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
        let layer_idx = ctx.layer_idx;
        for &idx in &self.sorted_indices {
            let cb = &mut self.callbacks[idx];
            // Check target_layers filter
            if let Some(layers) = cb.target_layers() {
                if !layers.contains(&layer_idx) {
                    continue;
                }
            }
            let action = cb.pre_node(ctx);
            if !matches!(action, CallbackAction::Continue) {
                log::trace!(
                    "callback[{}] pre_node layer={} → {:?}",
                    cb.name(),
                    layer_idx,
                    action_variant_name(&action),
                );
                return action;
            }
        }
        CallbackAction::Continue
    }

    /// Dispatch `post_node()` to all applicable callbacks for the given layer.
    ///
    /// Returns the first non-Continue action, or Continue if all callbacks agree.
    pub fn dispatch_post_node(&mut self, ctx: &LayerContext, output: &[u8]) -> CallbackAction {
        let layer_idx = ctx.layer_idx;
        for &idx in &self.sorted_indices {
            let cb = &mut self.callbacks[idx];
            if let Some(layers) = cb.target_layers() {
                if !layers.contains(&layer_idx) {
                    continue;
                }
            }
            let action = cb.post_node(ctx, output);
            if !matches!(action, CallbackAction::Continue) {
                log::trace!(
                    "callback[{}] post_node layer={} → {:?}",
                    cb.name(),
                    layer_idx,
                    action_variant_name(&action),
                );
                return action;
            }
        }
        CallbackAction::Continue
    }
}

fn action_variant_name(action: &CallbackAction) -> &'static str {
    match action {
        CallbackAction::Continue => "Continue",
        CallbackAction::SkipThisNode => "SkipThisNode",
        CallbackAction::ExitEarly { .. } => "ExitEarly",
        CallbackAction::InjectHidden { .. } => "InjectHidden",
        CallbackAction::CompactMask { .. } => "CompactMask",
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test callback with configurable behavior ──

    struct TestCallback {
        name: &'static str,
        priority: u32,
        layers: Option<Vec<usize>>,
        pre_action: CallbackAction,
        post_action: CallbackAction,
    }

    impl TestCallback {
        fn new(name: &'static str, priority: u32) -> Self {
            Self {
                name,
                priority,
                layers: None,
                pre_action: CallbackAction::Continue,
                post_action: CallbackAction::Continue,
            }
        }

        fn with_layers(mut self, layers: Vec<usize>) -> Self {
            self.layers = Some(layers);
            self
        }

        fn with_pre_action(mut self, action: CallbackAction) -> Self {
            self.pre_action = action;
            self
        }

        fn with_post_action(mut self, action: CallbackAction) -> Self {
            self.post_action = action;
            self
        }
    }

    impl LayerCallback for TestCallback {
        fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
            self.pre_action.clone()
        }

        fn post_node(&mut self, _ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
            self.post_action.clone()
        }

        fn priority(&self) -> u32 {
            self.priority
        }

        fn target_layers(&self) -> Option<&[usize]> {
            self.layers.as_deref()
        }

        fn name(&self) -> &str {
            self.name
        }
    }

    // ── Holder struct to own data that LayerContext borrows ──

    struct CtxHolder {
        op: &'static str,
        config: GeneratorForwardConfig,
        hidden_state: Vec<u8>,
    }

    impl CtxHolder {
        fn new() -> Self {
            Self {
                op: "Test",
                config: GeneratorForwardConfig {
                    geometry: std::sync::Arc::new(crate::model_config::ModelGeometry {
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
                        norm_eps: 1e-5,
                        num_experts: 0,
                        moe_top_k: 0,
                        expert_intermediate_size: 0,
                        global_rope_theta: 0.0,
                        rope_partial_ratio: 1.0,
                        attention_pattern: vec![],
                        sliding_window: 0,
                        num_kv_shared_layers: 0,
                        global_head_dim: 0,
                        hidden_size_per_layer_input: 0,
                        position_offset: None,
                        rope_scaling: None,
                        final_logit_softcapping: None,
                        hidden_act: None,
                    }),
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
                    callback_chain_ptr: std::ptr::null_mut(),
                },
                hidden_state: vec![0u8; 256 * 4],
            }
        }

        fn ctx(&self, layer: usize, node: usize) -> LayerContext<'_> {
            LayerContext {
                node_idx: node,
                layer_idx: layer,
                node_op: &self.op,
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

    // ── Tests ──

    #[test]
    fn test_empty_chain() {
        let mut chain = CallbackChain::empty();
        assert!(chain.is_empty());
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(matches!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue));
        assert!(matches!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue));
    }

    #[test]
    fn test_priority_ordering() {
        let cb_low = TestCallback::new("low", 10);
        let cb_high = TestCallback::new("high", 100);
        let mut chain = CallbackChain::new(vec![Box::new(cb_low), Box::new(cb_high)]);
        assert_eq!(chain.len(), 2);
        // High priority should be first in sorted_indices
        assert_eq!(chain.sorted_indices[0], 1); // index 1 = "high"
        assert_eq!(chain.sorted_indices[1], 0); // index 0 = "low"
    }

    #[test]
    fn test_target_layers_filter() {
        let cb = TestCallback::new("targeted", 50)
            .with_layers(vec![2, 4]);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();

        // Layer 1 → callback not targeted → Continue
        let ctx1 = holder.ctx(1, 2);
        assert!(matches!(chain.dispatch_pre_node(&ctx1), CallbackAction::Continue));

        // Layer 2 → callback targeted → Continue (default pre_action)
        let ctx2 = holder.ctx(2, 4);
        assert!(matches!(chain.dispatch_pre_node(&ctx2), CallbackAction::Continue));
    }

    #[test]
    fn test_skip_action() {
        let cb = TestCallback::new("skipper", 60)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_exit_early_action() {
        let cb = TestCallback::new("exiter", 50)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_post_node(&ctx, &[]);
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![1.0, 2.0, 3.0]);
            }
            _ => panic!("Expected ExitEarly, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_inject_hidden_action() {
        let cb = TestCallback::new("injector", 80)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![42u8; 1024] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(3, 6);

        let action = chain.dispatch_pre_node(&ctx);
        match action {
            CallbackAction::InjectHidden { data } => {
                assert_eq!(data.len(), 1024);
                assert!(data.iter().all(|&b| b == 42));
            }
            _ => panic!("Expected InjectHidden"),
        }
    }

    #[test]
    fn test_multiple_callbacks_priority() {
        // Two callbacks: high priority returns Continue, low returns SkipThisNode
        let cb_high = TestCallback::new("high_continue", 100);
        let cb_low = TestCallback::new("low_skip", 10)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // High priority returns Continue first, then low returns SkipThisNode
        let action = chain.dispatch_pre_node(&ctx);
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_default_callback_action() {
        let action = CallbackAction::default();
        assert!(matches!(action, CallbackAction::Continue));
    }
}
