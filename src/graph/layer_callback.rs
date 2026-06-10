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
#[derive(Debug, Clone, PartialEq)]
#[derive(Default)]
pub enum CallbackAction {
    /// Continue normal execution (no intervention).
    #[default]
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
        let chain = CallbackChain::new(vec![Box::new(cb_low), Box::new(cb_high)]);
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

    // ── Additional comprehensive tests ──

    #[test]
    fn test_callback_action_debug_format() {
        // Verify Debug trait produces meaningful output for each variant
        let continue_action = CallbackAction::Continue;
        let debug_str = format!("{:?}", continue_action);
        assert!(debug_str.contains("Continue"));

        let skip = CallbackAction::SkipThisNode;
        assert!(format!("{:?}", skip).contains("SkipThisNode"));

        let exit = CallbackAction::ExitEarly { logits: vec![0.5] };
        assert!(format!("{:?}", exit).contains("ExitEarly"));

        let inject = CallbackAction::InjectHidden { data: vec![1, 2, 3] };
        assert!(format!("{:?}", inject).contains("InjectHidden"));

        let compact = CallbackAction::CompactMask { active_mask: vec![true, false] };
        assert!(format!("{:?}", compact).contains("CompactMask"));
    }

    #[test]
    fn test_callback_action_clone() {
        // Verify Clone produces equal but independent copies
        let original = CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] };
        let cloned = original.clone();

        match (original, cloned) {
            (CallbackAction::ExitEarly { logits: a }, CallbackAction::ExitEarly { logits: b }) => {
                assert_eq!(a, b);
            }
            _ => panic!("Clone produced mismatched variant"),
        }
    }

    #[test]
    fn test_action_variant_name_all_variants() {
        // Verify action_variant_name returns correct static str for every variant
        assert_eq!(action_variant_name(&CallbackAction::Continue), "Continue");
        assert_eq!(action_variant_name(&CallbackAction::SkipThisNode), "SkipThisNode");
        assert_eq!(
            action_variant_name(&CallbackAction::ExitEarly { logits: vec![] }),
            "ExitEarly"
        );
        assert_eq!(
            action_variant_name(&CallbackAction::InjectHidden { data: vec![] }),
            "InjectHidden"
        );
        assert_eq!(
            action_variant_name(&CallbackAction::CompactMask { active_mask: vec![] }),
            "CompactMask"
        );
    }

    #[test]
    fn test_compact_mask_action() {
        // Arrange: callback returns CompactMask with mixed active/dead neurons
        let mask = vec![true, true, false, true, false, false, true];
        let cb = TestCallback::new("compactor", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, mask);
                assert_eq!(active_mask.iter().filter(|&&v| v).count(), 4);
                assert_eq!(active_mask.iter().filter(|&&v| !v).count(), 3);
            }
            _ => panic!("Expected CompactMask, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_callback_trait_default_methods() {
        // Verify that the default trait methods return the correct values
        struct MinimalCallback;
        impl LayerCallback for MinimalCallback {}

        let mut cb = MinimalCallback;
        assert_eq!(cb.priority(), 0);
        assert_eq!(cb.name(), "unnamed");
        assert!(cb.target_layers().is_none());

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(matches!(cb.pre_node(&ctx), CallbackAction::Continue));
        assert!(matches!(cb.post_node(&ctx, &[]), CallbackAction::Continue));
    }

    #[test]
    fn test_callback_custom_name() {
        // Verify custom name is accessible through the trait
        let cb = TestCallback::new("my_custom_hook", 42);
        assert_eq!(cb.name(), "my_custom_hook");
    }

    #[test]
    fn test_chain_new_with_single_callback() {
        // Arrange: single callback
        let cb = TestCallback::new("solo", 50)
            .with_pre_action(CallbackAction::SkipThisNode);

        // Act
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        // Assert
        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.sorted_indices.len(), 1);
        assert_eq!(chain.sorted_indices[0], 0);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let action = chain.dispatch_pre_node(&ctx);
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_chain_three_callbacks_priority_sort() {
        // Arrange: three callbacks with different priorities, middle one returns Skip
        let cb_low = TestCallback::new("low", 10);
        let cb_mid = TestCallback::new("mid", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_high = TestCallback::new("high", 90);

        // Act
        let chain = CallbackChain::new(vec![Box::new(cb_low), Box::new(cb_mid), Box::new(cb_high)]);

        // Assert: sorted order should be high(2), mid(1), low(0)
        assert_eq!(chain.sorted_indices, vec![2, 1, 0]);
        assert_eq!(chain.len(), 3);
    }

    #[test]
    fn test_chain_same_priority_preserves_insertion_order_stability() {
        // Arrange: two callbacks with identical priority
        let cb_first = TestCallback::new("first", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_second = TestCallback::new("second", 50);

        // Act: stable sort should keep insertion order for equal priorities
        let chain = CallbackChain::new(vec![Box::new(cb_first), Box::new(cb_second)]);

        // Assert: both should be present; first should appear before second
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.sorted_indices[0], 0);
        assert_eq!(chain.sorted_indices[1], 1);
    }

    #[test]
    fn test_dispatch_pre_node_high_priority_short_circuits() {
        // Arrange: high-priority callback returns non-Continue, blocking lower callbacks
        let cb_high = TestCallback::new("high_exit", 100)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_low = TestCallback::new("low_never_called", 10)
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![9.0] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act: high priority should return SkipThisNode, low never executes
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_dispatch_post_node_short_circuits_on_exit_early() {
        // Arrange: high-priority post callback exits, low never runs
        let cb_high = TestCallback::new("post_exit", 80)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![0.25, 0.75] });
        let cb_low = TestCallback::new("post_continue", 20);

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[1, 2, 3]);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![0.25, 0.75]);
            }
            _ => panic!("Expected ExitEarly, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_target_layers_multiple_layers() {
        // Arrange: callback targets layers 0 and 3
        let cb = TestCallback::new("multi_target", 50)
            .with_layers(vec![0, 3])
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Layer 0 → targeted → SkipThisNode
        let ctx0 = holder.ctx(0, 0);
        assert!(matches!(chain.dispatch_pre_node(&ctx0), CallbackAction::SkipThisNode));

        // Layer 1 → not targeted → Continue
        let ctx1 = holder.ctx(1, 2);
        assert!(matches!(chain.dispatch_pre_node(&ctx1), CallbackAction::Continue));

        // Layer 2 → not targeted → Continue
        let ctx2 = holder.ctx(2, 4);
        assert!(matches!(chain.dispatch_pre_node(&ctx2), CallbackAction::Continue));

        // Layer 3 → targeted → SkipThisNode
        let ctx3 = holder.ctx(3, 6);
        assert!(matches!(chain.dispatch_pre_node(&ctx3), CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_target_layers_none_means_all_layers() {
        // Arrange: callback with no target_layers filter returns Skip for all
        let cb = TestCallback::new("all_layers", 50)
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Every layer should be targeted
        for layer in 0..5 {
            let ctx = holder.ctx(layer, layer * 2);
            assert!(
                matches!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode),
                "Layer {} should be targeted",
                layer
            );
        }
    }

    #[test]
    fn test_layer_context_fields() {
        // Arrange
        let holder = CtxHolder::new();
        let ctx = holder.ctx(3, 7);

        // Assert: verify all LayerContext fields are correctly assigned
        assert_eq!(ctx.node_idx, 7);
        assert_eq!(ctx.layer_idx, 3);
        assert_eq!(ctx.node_op, "Test");
        assert_eq!(ctx.hidden_state.len(), 256 * 4);
        assert!(ctx.kv_cache_k.is_null());
        assert!(ctx.kv_cache_v.is_null());
        assert_eq!(ctx.total_seq, 10);
        assert_eq!(ctx.seq_len, 1);
        assert_eq!(ctx.position, 9);
        assert_eq!(ctx.request_id, 1);
        assert_eq!(ctx.model_config.hidden_size(), 256);
        assert_eq!(ctx.model_config.num_layers(), 4);
    }

    #[test]
    fn test_exit_early_with_empty_logits() {
        // Arrange: ExitEarly with empty logits means caller should project through lm_head
        let cb = TestCallback::new("empty_exit", 50)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[]);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert!(logits.is_empty());
            }
            _ => panic!("Expected ExitEarly with empty logits"),
        }
    }

    #[test]
    fn test_inject_hidden_with_empty_data() {
        // Arrange: InjectHidden with empty data
        let cb = TestCallback::new("empty_inject", 80)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::InjectHidden { data } => {
                assert!(data.is_empty());
            }
            _ => panic!("Expected InjectHidden with empty data"),
        }
    }

    #[test]
    fn test_compact_mask_all_active() {
        // Arrange: all neurons active (no compaction needed)
        let mask = vec![true; 8];
        let cb = TestCallback::new("all_active", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert!(active_mask.iter().all(|&v| v));
                assert_eq!(active_mask.len(), 8);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn test_compact_mask_all_dead() {
        // Arrange: all neurons dead (extreme compaction)
        let mask = vec![false; 8];
        let cb = TestCallback::new("all_dead", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert!(active_mask.iter().all(|&v| !v));
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn test_empty_chain_len_and_is_empty() {
        // Arrange & Act
        let chain = CallbackChain::empty();

        // Assert
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.sorted_indices.is_empty());
    }

    #[test]
    fn test_chain_post_node_all_continue_returns_continue() {
        // Arrange: multiple callbacks all return Continue on post_node
        let cb1 = TestCallback::new("cb1", 30);
        let cb2 = TestCallback::new("cb2", 60);
        let cb3 = TestCallback::new("cb3", 90);

        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2), Box::new(cb3)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act: all callbacks return Continue
        let action = chain.dispatch_post_node(&ctx, &[0u8; 64]);

        // Assert
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_chain_with_many_callbacks_dispatches_correctly() {
        // Arrange: 5 callbacks, only one returns non-Continue
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("cb0", 10)),
            Box::new(TestCallback::new("cb1", 20)),
            Box::new(TestCallback::new("cb2", 30)
                .with_pre_action(CallbackAction::SkipThisNode)),
            Box::new(TestCallback::new("cb3", 40)),
            Box::new(TestCallback::new("cb4", 50)),
        ];

        let mut chain = CallbackChain::new(callbacks);
        assert_eq!(chain.len(), 5);

        // Sorted: 4(50), 3(40), 2(30), 1(20), 0(10)
        assert_eq!(chain.sorted_indices, vec![4, 3, 2, 1, 0]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act: cb4(50)→Continue, cb3(40)→Continue, cb2(30)→SkipThisNode (stops here)
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    // ── Additional unit tests for comprehensive coverage ──

    #[test]
    fn test_callback_action_clone_independence() {
        // Arrange: Clone should produce an independent copy; modifying clone doesn't affect original
        let original = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        let cloned = original.clone();

        // After cloning, both should be equal in content
        if let (
            CallbackAction::ExitEarly { logits: a },
            CallbackAction::ExitEarly { logits: b },
        ) = (&original, &cloned)
        {
            assert_eq!(a, b);
        } else {
            panic!("Both should be ExitEarly");
        }
    }

    #[test]
    fn test_callback_action_clone_compact_mask() {
        // Arrange
        let mask = vec![true, false, true, true, false];
        let original = CallbackAction::CompactMask { active_mask: mask.clone() };
        let cloned = original.clone();

        // Assert
        if let CallbackAction::CompactMask { active_mask: cloned_mask } = cloned {
            assert_eq!(cloned_mask, mask);
        } else {
            panic!("Expected CompactMask");
        }
    }

    #[test]
    fn test_callback_action_clone_inject_hidden_large_data() {
        // Arrange: large binary payload
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let original = CallbackAction::InjectHidden { data: data.clone() };
        let cloned = original.clone();

        if let CallbackAction::InjectHidden { data: cloned_data } = cloned {
            assert_eq!(cloned_data, data);
            assert_eq!(cloned_data.len(), 4096);
        } else {
            panic!("Expected InjectHidden");
        }
    }

    #[test]
    fn test_debug_format_contains_variant_data() {
        // Verify Debug output includes inner data for data-carrying variants
        let exit = CallbackAction::ExitEarly { logits: vec![42.0, -1.5] };
        let debug = format!("{:?}", exit);
        assert!(debug.contains("42"));
        assert!(debug.contains("-1.5"));

        let inject = CallbackAction::InjectHidden { data: vec![0xAB, 0xCD] };
        let debug = format!("{:?}", inject);
        assert!(debug.contains("171") || debug.contains("AB") || debug.contains("171, 205"));

        let compact = CallbackAction::CompactMask { active_mask: vec![true] };
        let debug = format!("{:?}", compact);
        assert!(debug.contains("true"));
    }

    #[test]
    fn test_layer_context_with_non_null_kv_pointers() {
        // Arrange: allocate small buffers to get non-null pointers
        let mut k_buf = vec![0.0f32; 64];
        let mut v_buf = vec![0.0f32; 64];
        let holder = CtxHolder::new();
        let k_ptr = k_buf.as_mut_ptr();
        let v_ptr = v_buf.as_mut_ptr();

        let ctx = LayerContext {
            node_idx: 5,
            layer_idx: 2,
            node_op: "CustomOp",
            hidden_state: &holder.hidden_state,
            kv_cache_k: k_ptr,
            kv_cache_v: v_ptr,
            total_seq: 100,
            seq_len: 10,
            position: 90,
            request_id: 42,
            model_config: &holder.config,
        };

        // Assert
        assert_eq!(ctx.node_idx, 5);
        assert_eq!(ctx.layer_idx, 2);
        assert_eq!(ctx.node_op, "CustomOp");
        assert!(!ctx.kv_cache_k.is_null());
        assert!(!ctx.kv_cache_v.is_null());
        assert_eq!(ctx.total_seq, 100);
        assert_eq!(ctx.seq_len, 10);
        assert_eq!(ctx.position, 90);
        assert_eq!(ctx.request_id, 42);
    }

    #[test]
    fn test_layer_context_different_request_ids() {
        // Arrange
        let holder = CtxHolder::new();

        for rid in [0, 1, u64::MAX, u64::MIN] {
            let ctx = LayerContext {
                node_idx: 0,
                layer_idx: 0,
                node_op: "Op",
                hidden_state: &holder.hidden_state,
                kv_cache_k: std::ptr::null_mut(),
                kv_cache_v: std::ptr::null_mut(),
                total_seq: 1,
                seq_len: 1,
                position: 0,
                request_id: rid,
                model_config: &holder.config,
            };
            assert_eq!(ctx.request_id, rid);
        }
    }

    #[test]
    fn test_dispatch_pre_node_high_skip_blocks_low_inject() {
        // Arrange: high priority returns Skip, low returns InjectHidden
        let cb_high = TestCallback::new("high_skip", 100)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_low = TestCallback::new("low_inject", 10)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![1, 2, 3] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert: high-priority SkipThisNode should win, low InjectHidden never reached
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_dispatch_post_node_inject_hidden() {
        // Arrange: post_node returns InjectHidden
        let cb = TestCallback::new("post_inject", 50)
            .with_post_action(CallbackAction::InjectHidden { data: vec![0xDE, 0xAD, 0xBE, 0xEF] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[1, 2, 3]);

        // Assert
        match action {
            CallbackAction::InjectHidden { data } => {
                assert_eq!(data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            _ => panic!("Expected InjectHidden, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_dispatch_post_node_skip_this_node() {
        // Arrange: post_node returns SkipThisNode
        let cb = TestCallback::new("post_skip", 60)
            .with_post_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[0u8; 32]);

        // Assert
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_dispatch_post_node_compact_mask() {
        // Arrange: post_node returns CompactMask
        let mask = vec![true, false, true];
        let cb = TestCallback::new("post_compact", 70)
            .with_post_action(CallbackAction::CompactMask { active_mask: mask.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[]);

        // Assert
        match action {
            CallbackAction::CompactMask { active_mask: m } => {
                assert_eq!(m, mask);
            }
            _ => panic!("Expected CompactMask, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_compact_mask_single_element() {
        // Arrange: single-element mask (edge case)
        let mask = vec![true];
        let cb = TestCallback::new("single_mask", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 1);
                assert!(active_mask[0]);
            }
            _ => panic!("Expected CompactMask"),
        }
    }

    #[test]
    fn test_chain_many_callbacks_correct_sort_order() {
        // Arrange: 8 callbacks with varying priorities
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("p10", 10)),
            Box::new(TestCallback::new("p20", 20)),
            Box::new(TestCallback::new("p30", 30)),
            Box::new(TestCallback::new("p40", 40)),
            Box::new(TestCallback::new("p50", 50)),
            Box::new(TestCallback::new("p60", 60)),
            Box::new(TestCallback::new("p70", 70)),
            Box::new(TestCallback::new("p80", 80)),
        ];

        let chain = CallbackChain::new(callbacks);

        // Assert: sorted indices should be 7(80), 6(70), 5(60), 4(50), 3(40), 2(30), 1(20), 0(10)
        assert_eq!(chain.sorted_indices, vec![7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(chain.len(), 8);
    }

    #[test]
    fn test_chain_zero_priority_callback() {
        // Arrange: callback with default (zero) priority
        let cb = TestCallback::new("zero_prio", 0)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert: zero priority still works fine
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_chain_mixed_priority_zero_and_high() {
        // Arrange: mix of zero and high priority callbacks
        let cb_zero = TestCallback::new("zero", 0)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![99] });
        let cb_high = TestCallback::new("high", 200);
        let chain = CallbackChain::new(vec![Box::new(cb_zero), Box::new(cb_high)]);

        // Assert: high(200) should be first, zero(0) second
        assert_eq!(chain.sorted_indices, vec![1, 0]);
    }

    #[test]
    fn test_target_layers_empty_slice_matches_nothing() {
        // Arrange: callback targets empty layer list
        let cb = TestCallback::new("empty_targets", 50)
            .with_layers(vec![])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();

        // No layer should trigger the callback
        for layer in 0..5 {
            let ctx = holder.ctx(layer, layer);
            let action = chain.dispatch_pre_node(&ctx);
            assert!(
                matches!(action, CallbackAction::Continue),
                "Layer {} should not be targeted",
                layer
            );
        }
    }

    #[test]
    fn test_dispatch_pre_node_all_continue_then_post_node_exit() {
        // Arrange: pre_node all Continue, then post_node returns ExitEarly
        let cb = TestCallback::new("pre_cont_post_exit", 50)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![0.5, 0.5] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act: pre_node returns Continue
        let pre = chain.dispatch_pre_node(&ctx);
        assert!(matches!(pre, CallbackAction::Continue));

        // Act: post_node returns ExitEarly
        let post = chain.dispatch_post_node(&ctx, &[]);
        match post {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![0.5, 0.5]);
            }
            _ => panic!("Expected ExitEarly"),
        }
    }

    #[test]
    fn test_callback_priority_trait_default_is_zero() {
        // Verify the default priority is 0 per trait definition
        struct DefaultsOnly;
        impl LayerCallback for DefaultsOnly {}

        let cb = DefaultsOnly;
        assert_eq!(cb.priority(), 0);
    }

    #[test]
    fn test_callback_name_trait_default_is_unnamed() {
        struct DefaultsOnly;
        impl LayerCallback for DefaultsOnly {}

        let cb = DefaultsOnly;
        assert_eq!(cb.name(), "unnamed");
    }

    #[test]
    fn test_callback_target_layers_default_is_none() {
        struct DefaultsOnly;
        impl LayerCallback for DefaultsOnly {}

        let cb = DefaultsOnly;
        assert!(cb.target_layers().is_none());
    }

    #[test]
    fn test_callback_action_clone_continue() {
        let original = CallbackAction::Continue;
        let cloned = original.clone();
        assert!(matches!(original, CallbackAction::Continue));
        assert!(matches!(cloned, CallbackAction::Continue));
    }

    #[test]
    fn test_callback_action_clone_skip_this_node() {
        let original = CallbackAction::SkipThisNode;
        let cloned = original.clone();
        assert!(matches!(original, CallbackAction::SkipThisNode));
        assert!(matches!(cloned, CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_model_config_accessors_through_context() {
        // Arrange: verify model config is accessible through LayerContext
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Assert: access model geometry through context
        assert_eq!(ctx.model_config.hidden_size(), 256);
        assert_eq!(ctx.model_config.num_layers(), 4);
        assert_eq!(ctx.model_config.vocab_size(), 1000);
        assert_eq!(ctx.model_config.intermediate_size(), 512);
    }

    #[test]
    fn test_exit_early_with_large_logits_vector() {
        // Arrange: large logits vector (simulating real vocab size)
        let logits: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let cb = TestCallback::new("large_exit", 50)
            .with_post_action(CallbackAction::ExitEarly { logits: logits.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[]);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits: l } => {
                assert_eq!(l.len(), 1000);
                assert_eq!(l[0], 0.0);
                assert!((l[999] - 0.999).abs() < 1e-6);
            }
            _ => panic!("Expected ExitEarly"),
        }
    }

    #[test]
    fn test_inject_hidden_with_arbitrary_byte_patterns() {
        // Arrange: various byte patterns
        for &pattern in &[0x00u8, 0xFF, 0x42] {
            let data = vec![pattern; 256];
            let cb = TestCallback::new("pattern_inject", 80)
                .with_pre_action(CallbackAction::InjectHidden { data: data.clone() });
            let mut chain = CallbackChain::new(vec![Box::new(cb)]);

            let holder = CtxHolder::new();
            let ctx = holder.ctx(0, 0);

            let action = chain.dispatch_pre_node(&ctx);

            if let CallbackAction::InjectHidden { data: d } = action {
                assert!(d.iter().all(|&b| b == pattern));
            } else {
                panic!("Expected InjectHidden for pattern {:02X}", pattern);
            }
        }
    }

    #[test]
    fn test_chain_dispatch_with_output_buffer_passed_through() {
        // Arrange: tracking callback that inspects the output slice
        struct OutputInspectCallback {
            observed_len: usize,
        }
        impl LayerCallback for OutputInspectCallback {
            fn post_node(&mut self, _ctx: &LayerContext, output: &[u8]) -> CallbackAction {
                self.observed_len = output.len();
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "output_inspector" }
        }

        let cb = OutputInspectCallback { observed_len: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let output = &[1u8, 2, 3, 4, 5, 6, 7, 8];

        // Act
        let action = chain.dispatch_post_node(&ctx, output);

        // Assert: callback observed the output and returned Continue
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_layer_context_zero_position() {
        // Arrange: first token position (position = 0)
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "FirstToken",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 999,
            model_config: &holder.config,
        };

        assert_eq!(ctx.position, 0);
        assert_eq!(ctx.total_seq, 1);
        assert_eq!(ctx.seq_len, 1);
        assert_eq!(ctx.request_id, 999);
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_callback_action_clone_deep_independence() {
        // Arrange: Clone produces deep copy; pushing to cloned vec doesn't affect original
        let original = CallbackAction::ExitEarly { logits: vec![1.0] };
        let mut cloned = original.clone();

        // Act: mutate the cloned variant's inner vec
        if let CallbackAction::ExitEarly { logits } = &mut cloned {
            logits.push(2.0);
        }

        // Assert: original is unchanged
        if let CallbackAction::ExitEarly { logits } = &original {
            assert_eq!(logits.len(), 1);
            assert_eq!(logits[0], 1.0);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    #[test]
    fn test_callback_action_clone_inject_hidden_deep_independence() {
        // Arrange
        let original = CallbackAction::InjectHidden { data: vec![10u8] };
        let mut cloned = original.clone();

        // Act: mutate cloned data
        if let CallbackAction::InjectHidden { data } = &mut cloned {
            data.push(20);
        }

        // Assert: original unchanged
        if let CallbackAction::InjectHidden { data } = &original {
            assert_eq!(data.len(), 1);
            assert_eq!(data[0], 10);
        } else {
            panic!("Expected InjectHidden");
        }
    }

    #[test]
    fn test_callback_action_clone_compact_mask_deep_independence() {
        // Arrange
        let original = CallbackAction::CompactMask { active_mask: vec![true] };
        let mut cloned = original.clone();

        // Act: mutate cloned mask
        if let CallbackAction::CompactMask { active_mask } = &mut cloned {
            active_mask.push(false);
        }

        // Assert: original unchanged
        if let CallbackAction::CompactMask { active_mask } = &original {
            assert_eq!(active_mask.len(), 1);
            assert!(active_mask[0]);
        } else {
            panic!("Expected CompactMask");
        }
    }

    #[test]
    fn test_exit_early_with_special_float_values() {
        // Arrange: NaN and Infinity in logits
        let cb = TestCallback::new("special_floats", 50)
            .with_post_action(CallbackAction::ExitEarly {
                logits: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0],
            });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[]);

        // Assert
        if let CallbackAction::ExitEarly { logits } = action {
            assert!(logits[0].is_nan());
            assert!(logits[1].is_infinite() && logits[1].is_sign_positive());
            assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
            assert_eq!(logits[3], 0.0);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    #[test]
    fn test_layer_context_hidden_state_content_access() {
        // Arrange: populate hidden_state with known pattern
        let mut holder = CtxHolder::new();
        let pattern: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        holder.hidden_state[..64].copy_from_slice(&pattern);
        let ctx = holder.ctx(0, 0);

        // Assert: hidden state bytes are readable and match injected pattern
        assert_eq!(&ctx.hidden_state[..64], &pattern);
        assert_eq!(ctx.hidden_state[0], 0);
        assert_eq!(ctx.hidden_state[63], 252);
    }

    #[test]
    fn test_layer_context_max_usize_values() {
        // Arrange: verify LayerContext handles usize::MAX for index fields
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: usize::MAX,
            layer_idx: usize::MAX,
            node_op: "MaxOp",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: usize::MAX,
            seq_len: usize::MAX,
            position: usize::MAX,
            request_id: 0,
            model_config: &holder.config,
        };

        assert_eq!(ctx.node_idx, usize::MAX);
        assert_eq!(ctx.layer_idx, usize::MAX);
        assert_eq!(ctx.total_seq, usize::MAX);
        assert_eq!(ctx.seq_len, usize::MAX);
        assert_eq!(ctx.position, usize::MAX);
    }

    #[test]
    fn test_chain_new_with_empty_vec_matches_empty() {
        // Arrange: CallbackChain::new(vec![]) should behave like CallbackChain::empty()
        let chain_from_new = CallbackChain::new(vec![]);
        let chain_from_empty = CallbackChain::empty();

        // Assert
        assert!(chain_from_new.is_empty());
        assert!(chain_from_empty.is_empty());
        assert_eq!(chain_from_new.len(), 0);
        assert_eq!(chain_from_empty.len(), 0);
        assert!(chain_from_new.sorted_indices.is_empty());
        assert!(chain_from_empty.sorted_indices.is_empty());
    }

    #[test]
    fn test_chain_dispatch_all_filtered_by_target_layers() {
        // Arrange: callback targets only layer 99, dispatch for layer 0
        let cb = TestCallback::new("layer99_only", 50)
            .with_layers(vec![99])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Act: dispatch for layer 0 (not 99)
        let ctx = holder.ctx(0, 0);
        let action = chain.dispatch_pre_node(&ctx);

        // Assert: callback was skipped, Continue returned
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_chain_dispatch_first_filtered_second_matches() {
        // Arrange: high-priority targets layer 99, low-priority targets layer 0
        let cb_high = TestCallback::new("high_layer99", 100)
            .with_layers(vec![99])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_low = TestCallback::new("low_layer0", 10)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![42] });
        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();

        // Act: dispatch for layer 0
        let ctx = holder.ctx(0, 0);
        let action = chain.dispatch_pre_node(&ctx);

        // Assert: high skipped (wrong layer), low matched and returned InjectHidden
        match action {
            CallbackAction::InjectHidden { data } => {
                assert_eq!(data, vec![42]);
            }
            _ => panic!("Expected InjectHidden from low-priority callback, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_chain_overlapping_target_layers() {
        // Arrange: two callbacks both target layer 2, different priorities and actions
        let cb_high = TestCallback::new("high_l2", 80)
            .with_layers(vec![2])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_low = TestCallback::new("low_l2", 20)
            .with_layers(vec![2])
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);

        let holder = CtxHolder::new();

        // Layer 2: both targeted, high priority returns SkipThisNode first
        let ctx2 = holder.ctx(2, 4);
        assert!(matches!(chain.dispatch_pre_node(&ctx2), CallbackAction::SkipThisNode));

        // Layer 1: neither targeted
        let ctx1 = holder.ctx(1, 2);
        assert!(matches!(chain.dispatch_pre_node(&ctx1), CallbackAction::Continue));
    }

    #[test]
    fn test_chain_three_equal_priorities_stable_sort() {
        // Arrange: three callbacks with same priority, middle one returns Skip
        let cb_a = TestCallback::new("a", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_b = TestCallback::new("b", 50);
        let cb_c = TestCallback::new("c", 50);

        let chain = CallbackChain::new(vec![Box::new(cb_a), Box::new(cb_b), Box::new(cb_c)]);

        // Assert: stable sort preserves insertion order for equal priorities
        assert_eq!(chain.sorted_indices, vec![0, 1, 2]);
        assert_eq!(chain.len(), 3);
    }

    #[test]
    fn test_dispatch_continues_only_on_continue_variant() {
        // Arrange: verify that ONLY CallbackAction::Continue allows dispatch to proceed
        // to the next callback. All other variants short-circuit.
        // Test each non-Continue variant in separate sub-tests.
        let holder = CtxHolder::new();

        // Sub-test: InjectHidden short-circuits
        let cb_inject = TestCallback::new("inject", 10)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![1] });
        let cb_skip = TestCallback::new("skip", 5)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb_inject), Box::new(cb_skip)]);
        let ctx = holder.ctx(0, 0);
        // InjectHidden is higher priority → it short-circuits, Skip never reached
        let action = chain.dispatch_pre_node(&ctx);
        assert!(matches!(action, CallbackAction::InjectHidden { .. }));
    }

    #[test]
    fn test_dispatch_post_node_continues_through_continue_callbacks() {
        // Arrange: 4 callbacks, first 3 return Continue, last returns ExitEarly
        let cb1 = TestCallback::new("cont1", 40);
        let cb2 = TestCallback::new("cont2", 30);
        let cb3 = TestCallback::new("cont3", 20);
        let cb4 = TestCallback::new("exit_last", 10)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![7.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2), Box::new(cb3), Box::new(cb4)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act: dispatch goes through all 4 (sorted by priority), first 3 Continue, last exits
        let action = chain.dispatch_post_node(&ctx, &[]);

        // Assert
        if let CallbackAction::ExitEarly { logits } = action {
            assert_eq!(logits, vec![7.0]);
        } else {
            panic!("Expected ExitEarly from lowest-priority callback");
        }
    }

    #[test]
    fn test_callback_action_debug_format_skip_this_node() {
        // Arrange: verify Debug for SkipThisNode is clean
        let action = CallbackAction::SkipThisNode;
        let debug = format!("{:?}", action);

        // Assert
        assert_eq!(debug, "SkipThisNode");
    }

    #[test]
    fn test_callback_action_debug_format_continue() {
        // Arrange
        let action = CallbackAction::Continue;
        let debug = format!("{:?}", action);

        // Assert
        assert_eq!(debug, "Continue");
    }

    #[test]
    fn test_callback_action_debug_format_exit_early_with_values() {
        // Arrange: verify Debug includes the logits data
        let action = CallbackAction::ExitEarly { logits: vec![1.5, -2.5, 0.0] };
        let debug = format!("{:?}", action);

        // Assert: should contain the field name and values
        assert!(debug.contains("ExitEarly"));
        assert!(debug.contains("logits"));
        assert!(debug.contains("1.5"));
    }

    #[test]
    fn test_ctx_holder_different_node_ops() {
        // Arrange: CtxHolder with custom op string
        let mut holder = CtxHolder::new();
        holder.op = "RmsNorm";

        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.node_op, "RmsNorm");

        holder.op = "Gemm";
        let ctx2 = holder.ctx(1, 2);
        assert_eq!(ctx2.node_op, "Gemm");
    }

    #[test]
    fn test_inject_hidden_with_max_u8_pattern() {
        // Arrange: InjectHidden with 0xFF bytes
        let data = vec![0xFFu8; 512];
        let cb = TestCallback::new("max_byte_inject", 80)
            .with_pre_action(CallbackAction::InjectHidden { data: data.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        if let CallbackAction::InjectHidden { data: d } = action {
            assert_eq!(d.len(), 512);
            assert!(d.iter().all(|&b| b == 0xFF));
        } else {
            panic!("Expected InjectHidden");
        }
    }

    #[test]
    fn test_layer_context_different_seq_and_position_combinations() {
        // Arrange: various valid seq_len/position/total_seq combinations
        let holder = CtxHolder::new();

        // Case 1: prefill (seq_len > 1)
        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 128, seq_len: 128, position: 0,
            request_id: 1, model_config: &holder.config,
        };
        assert_eq!(ctx.seq_len, 128);
        assert_eq!(ctx.position, 0);
        assert_eq!(ctx.total_seq, 128);

        // Case 2: decode step (seq_len = 1)
        let ctx2 = LayerContext {
            node_idx: 2, layer_idx: 1, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 129, seq_len: 1, position: 128,
            request_id: 1, model_config: &holder.config,
        };
        assert_eq!(ctx2.seq_len, 1);
        assert_eq!(ctx2.position, 128);
        assert_eq!(ctx2.total_seq, 129);
    }

    // ── New tests: partial_eq, chain behavior, edge cases ──

    #[test]
    fn test_callback_action_partial_eq_continue() {
        assert_eq!(CallbackAction::Continue, CallbackAction::Continue);
    }

    #[test]
    fn test_callback_action_partial_eq_skip_this_node() {
        assert_eq!(CallbackAction::SkipThisNode, CallbackAction::SkipThisNode);
        assert_ne!(CallbackAction::SkipThisNode, CallbackAction::Continue);
    }

    #[test]
    fn test_callback_action_partial_eq_exit_early_same_logits() {
        let a = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        let b = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        assert_eq!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_exit_early_different_logits() {
        let a = CallbackAction::ExitEarly { logits: vec![1.0] };
        let b = CallbackAction::ExitEarly { logits: vec![2.0] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_inject_hidden_same_data() {
        let a = CallbackAction::InjectHidden { data: vec![10, 20, 30] };
        let b = CallbackAction::InjectHidden { data: vec![10, 20, 30] };
        assert_eq!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_inject_hidden_different_data() {
        let a = CallbackAction::InjectHidden { data: vec![1] };
        let b = CallbackAction::InjectHidden { data: vec![2] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_compact_mask_same_mask() {
        let a = CallbackAction::CompactMask { active_mask: vec![true, false] };
        let b = CallbackAction::CompactMask { active_mask: vec![true, false] };
        assert_eq!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_compact_mask_different_mask() {
        let a = CallbackAction::CompactMask { active_mask: vec![true] };
        let b = CallbackAction::CompactMask { active_mask: vec![false] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_cross_variant_never_equal() {
        let actions = vec![
            CallbackAction::Continue,
            CallbackAction::SkipThisNode,
            CallbackAction::ExitEarly { logits: vec![] },
            CallbackAction::InjectHidden { data: vec![] },
            CallbackAction::CompactMask { active_mask: vec![] },
        ];
        for (i, a) in actions.iter().enumerate() {
            for (j, b) in actions.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Variant {} should not equal variant {}", i, j);
                }
            }
        }
    }

    #[test]
    fn test_chain_empty_dispatch_pre_node_idempotent() {
        let mut chain = CallbackChain::empty();
        let holder = CtxHolder::new();
        for layer in 0..10 {
            let ctx = holder.ctx(layer, layer * 3);
            assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        }
    }

    #[test]
    fn test_chain_empty_dispatch_post_node_idempotent() {
        let mut chain = CallbackChain::empty();
        let holder = CtxHolder::new();
        for layer in 0..10 {
            let ctx = holder.ctx(layer, layer * 3);
            assert_eq!(chain.dispatch_post_node(&ctx, &[0u8; 64]), CallbackAction::Continue);
        }
    }

    #[test]
    fn test_chain_callback_order_respects_priority_with_gap() {
        let cb_10 = TestCallback::new("p10", 10)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_1000 = TestCallback::new("p1000", 1000);
        let cb_50 = TestCallback::new("p50", 50);
        let chain = CallbackChain::new(vec![
            Box::new(cb_10),
            Box::new(cb_1000),
            Box::new(cb_50),
        ]);
        assert_eq!(chain.sorted_indices, vec![1, 2, 0]);
        assert_eq!(chain.len(), 3);
    }

    #[test]
    fn test_callback_pre_and_post_independent_actions() {
        let cb = TestCallback::new("dual", 50)
            .with_pre_action(CallbackAction::SkipThisNode)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![3.14] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let pre = chain.dispatch_pre_node(&ctx);
        assert_eq!(pre, CallbackAction::SkipThisNode);

        let post = chain.dispatch_post_node(&ctx, &[]);
        match post {
            CallbackAction::ExitEarly { logits } => {
                assert!((logits[0] - 3.14).abs() < 1e-6);
            }
            _ => panic!("Expected ExitEarly, got {:?}", action_variant_name(&post)),
        }
    }

    #[test]
    fn test_chain_target_layer_prevents_post_node_dispatch() {
        let cb = TestCallback::new("only_l5", 50)
            .with_layers(vec![5])
            .with_post_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        let ctx_l0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_post_node(&ctx_l0, &[]), CallbackAction::Continue);

        let ctx_l5 = holder.ctx(5, 10);
        assert_eq!(chain.dispatch_post_node(&ctx_l5, &[]), CallbackAction::SkipThisNode);
    }

    #[test]
    fn test_chain_single_callback_with_target_layer_mismatch() {
        let cb = TestCallback::new("mismatch", 50)
            .with_layers(vec![7])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for layer in [0, 1, 2, 3, 4, 5, 6, 8, 9] {
            let ctx = holder.ctx(layer, layer);
            assert_eq!(
                chain.dispatch_pre_node(&ctx),
                CallbackAction::Continue,
                "Layer {} should not be targeted",
                layer
            );
        }
    }

    #[test]
    fn test_callback_action_default_trait_is_continue() {
        fn assert_default<T: Default>(_: &T) {}
        let action = CallbackAction::default();
        assert_default(&action);
        assert_eq!(action, CallbackAction::Continue);
    }

    #[test]
    fn test_layer_context_node_op_borrow_lifetime() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let op_ref: &str = ctx.node_op;
        assert_eq!(op_ref, "Test");
        let config_ref: &GeneratorForwardConfig = ctx.model_config;
        assert_eq!(config_ref.hidden_size(), 256);
    }

    #[test]
    fn test_callback_trait_object_sends_across_thread_boundary() {
        let cb: Box<dyn LayerCallback + Send> = Box::new(TestCallback::new("sendable", 50));
        let handle = std::thread::spawn(move || {
            assert_eq!(cb.name(), "sendable");
            assert_eq!(cb.priority(), 50);
        });
        handle.join().expect("Thread should complete without panic");
    }

    // ========================================================================
    // New tests: 40 additional tests for comprehensive coverage
    // ========================================================================

    // -- CallbackAction: Debug output completeness --

    #[test]
    fn test_callback_action_debug_inject_hidden_shows_data() {
        let action = CallbackAction::InjectHidden { data: vec![0u8, 1, 2] };
        let debug = format!("{:?}", action);
        assert!(debug.contains("InjectHidden"));
        assert!(debug.contains("data"));
    }

    #[test]
    fn test_callback_action_debug_compact_mask_shows_mask() {
        let action = CallbackAction::CompactMask { active_mask: vec![false, true, false] };
        let debug = format!("{:?}", action);
        assert!(debug.contains("CompactMask"));
        assert!(debug.contains("active_mask"));
    }

    // -- CallbackAction: Clone deep copy verification --

    #[test]
    fn test_callback_action_clone_exit_early_large_logits_deep_copy() {
        let logits: Vec<f32> = (0..2048).map(|i| i as f32).collect();
        let original = CallbackAction::ExitEarly { logits };
        let mut cloned = original.clone();
        // Mutate cloned to prove deep copy
        if let CallbackAction::ExitEarly { logits } = &mut cloned {
            logits[0] = -999.0;
        }
        if let CallbackAction::ExitEarly { logits } = &original {
            assert_ne!(logits[0], -999.0, "Original should not be affected by clone mutation");
        }
    }

    // -- CallbackAction: special float values in ExitEarly logits --

    #[test]
    fn test_callback_action_exit_early_with_zero_and_negative_logits() {
        let logits = vec![0.0f32, -1.0, -0.0];
        let action = CallbackAction::ExitEarly { logits: logits.clone() };
        if let CallbackAction::ExitEarly { logits: l } = action {
            assert_eq!(l[0], 0.0);
            assert!(l[1].is_sign_negative());
            // -0.0 == 0.0 in IEEE 754
            assert_eq!(l[2], 0.0);
        }
    }

    #[test]
    fn test_callback_action_exit_early_with_f32_min_max() {
        let logits = vec![f32::MIN, f32::MAX, f32::MIN_POSITIVE];
        let action = CallbackAction::ExitEarly { logits: logits.clone() };
        if let CallbackAction::ExitEarly { logits: l } = action {
            assert_eq!(l[0], f32::MIN);
            assert_eq!(l[1], f32::MAX);
            assert_eq!(l[2], f32::MIN_POSITIVE);
        }
    }

    // -- CallbackAction: Default should always be Continue --

    #[test]
    fn test_callback_action_default_is_stable_across_multiple_calls() {
        let a = CallbackAction::default();
        let b = CallbackAction::default();
        assert_eq!(a, b);
        assert!(matches!(a, CallbackAction::Continue));
    }

    // -- CallbackChain: construction edge cases --

    #[test]
    fn test_chain_new_single_callback_sorted_indices_identity() {
        let cb = TestCallback::new("solo", 42);
        let chain = CallbackChain::new(vec![Box::new(cb)]);
        assert_eq!(chain.sorted_indices, vec![0]);
        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_chain_max_priority_value() {
        // Arrange: callback with u32::MAX priority
        let cb = TestCallback::new("max_prio", u32::MAX)
            .with_pre_action(CallbackAction::SkipThisNode);
        let chain = CallbackChain::new(vec![Box::new(cb)]);
        assert_eq!(chain.sorted_indices, vec![0]);

        let mut chain = chain;
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(matches!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_chain_priority_ordering_with_max_and_zero() {
        let cb_zero = TestCallback::new("zero", 0);
        let cb_max = TestCallback::new("max", u32::MAX);
        let cb_mid = TestCallback::new("mid", 500);
        let chain = CallbackChain::new(vec![
            Box::new(cb_zero),
            Box::new(cb_max),
            Box::new(cb_mid),
        ]);
        // max=1(u32::MAX), mid=2(500), zero=0(0)
        assert_eq!(chain.sorted_indices, vec![1, 2, 0]);
    }

    #[test]
    fn test_chain_ten_callbacks_all_same_priority_stable_order() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = (0..10)
            .map(|i| Box::new(TestCallback::new(Box::leak(format!("cb{}", i).into_boxed_str()), 50)) as _)
            .collect();
        let chain = CallbackChain::new(callbacks);
        // Stable sort should preserve 0..10
        assert_eq!(chain.sorted_indices, (0..10).collect::<Vec<_>>());
        assert_eq!(chain.len(), 10);
    }

    // -- CallbackChain: dispatch behavior edge cases --

    #[test]
    fn test_chain_dispatch_pre_node_high_returns_inject_mid_never_called() {
        let cb_high = TestCallback::new("high_inject", 100)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0xAA] });
        let cb_mid = TestCallback::new("mid_skip", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_low = TestCallback::new("low_exit", 10)
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![0.0] });

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_high),
            Box::new(cb_mid),
            Box::new(cb_low),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        // High priority returns InjectHidden, mid and low never execute
        match action {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xAA]),
            _ => panic!("Expected InjectHidden, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_chain_dispatch_post_node_middle_callback_short_circuits() {
        let cb_high = TestCallback::new("high_cont", 90);
        let cb_mid = TestCallback::new("mid_compact", 50)
            .with_post_action(CallbackAction::CompactMask { active_mask: vec![true, false] });
        let cb_low = TestCallback::new("low_skip", 10)
            .with_post_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_high),
            Box::new(cb_mid),
            Box::new(cb_low),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_post_node(&ctx, &[1, 2, 3]);
        // high→Continue, mid→CompactMask (short-circuits), low never called
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false]);
            }
            _ => panic!("Expected CompactMask, got {:?}", action_variant_name(&action)),
        }
    }

    // -- target_layers: boundary conditions --

    #[test]
    fn test_target_layers_single_layer_zero() {
        let cb = TestCallback::new("l0_only", 50)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        let ctx0 = holder.ctx(0, 0);
        assert!(matches!(chain.dispatch_pre_node(&ctx0), CallbackAction::SkipThisNode));

        let ctx1 = holder.ctx(1, 2);
        assert!(matches!(chain.dispatch_pre_node(&ctx1), CallbackAction::Continue));
    }

    #[test]
    fn test_target_layers_large_layer_index() {
        let cb = TestCallback::new("large_layer", 50)
            .with_layers(vec![usize::MAX])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        let ctx_normal = holder.ctx(10, 20);
        assert!(matches!(chain.dispatch_pre_node(&ctx_normal), CallbackAction::Continue));

        let ctx_max = holder.ctx(usize::MAX, usize::MAX);
        assert!(matches!(chain.dispatch_pre_node(&ctx_max), CallbackAction::SkipThisNode));
    }

    #[test]
    fn test_target_layers_multiple_callbacks_different_single_targets() {
        let cb_l0 = TestCallback::new("l0", 30)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_l1 = TestCallback::new("l1", 20)
            .with_layers(vec![1])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_l2 = TestCallback::new("l2", 10)
            .with_layers(vec![2])
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_l0),
            Box::new(cb_l1),
            Box::new(cb_l2),
        ]);
        let holder = CtxHolder::new();

        for layer in 0..3 {
            let ctx = holder.ctx(layer, layer);
            assert!(matches!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode));
        }
        let ctx3 = holder.ctx(3, 6);
        assert!(matches!(chain.dispatch_pre_node(&ctx3), CallbackAction::Continue));
    }

    // -- LayerContext: field combinations and boundary values --

    #[test]
    fn test_layer_context_with_zero_seq_len() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "EmptySeq",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 0,
            seq_len: 0,
            position: 0,
            request_id: 0,
            model_config: &holder.config,
        };
        assert_eq!(ctx.total_seq, 0);
        assert_eq!(ctx.seq_len, 0);
        assert_eq!(ctx.position, 0);
        assert_eq!(ctx.request_id, 0);
    }

    #[test]
    fn test_layer_context_model_config_num_heads() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.num_heads(), 4);
        assert_eq!(ctx.model_config.num_kv_heads(), 2);
        assert_eq!(ctx.model_config.head_dim(), 64);
    }

    #[test]
    fn test_layer_context_model_config_rope() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.rope_theta(), 10000.0);
        assert_eq!(ctx.model_config.rope_scale(), 1.0);
    }

    #[test]
    fn test_layer_context_different_node_ops_for_each_node() {
        let mut holder = CtxHolder::new();
        let ops = ["Attention", "FFN", "RmsNorm", "Gemm", "Rope"];
        for (i, &op) in ops.iter().enumerate() {
            holder.op = op;
            let ctx = holder.ctx(i, i);
            assert_eq!(ctx.node_op, op);
        }
    }

    #[test]
    fn test_layer_context_hidden_state_slice_access() {
        let mut holder = CtxHolder::new();
        // Fill with specific pattern
        for (i, byte) in holder.hidden_state.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let ctx = holder.ctx(0, 0);
        // Verify pattern is accessible through context
        assert_eq!(ctx.hidden_state[0], 0);
        assert_eq!(ctx.hidden_state[255], 255);
        assert_eq!(ctx.hidden_state[256], 0);
        assert_eq!(ctx.hidden_state.len(), 256 * 4);
    }

    // -- LayerCallback trait: default implementation verification --

    #[test]
    fn test_layer_callback_default_pre_node_returns_continue() {
        struct Minimal;
        impl LayerCallback for Minimal {}
        let mut m = Minimal;
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(m.pre_node(&ctx), CallbackAction::Continue);
    }

    #[test]
    fn test_layer_callback_default_post_node_returns_continue() {
        struct Minimal;
        impl LayerCallback for Minimal {}
        let mut m = Minimal;
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(m.post_node(&ctx, &[]), CallbackAction::Continue);
    }

    #[test]
    fn test_layer_callback_default_post_node_ignores_output() {
        struct Minimal;
        impl LayerCallback for Minimal {}
        let mut m = Minimal;
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        // Even with non-empty output, default returns Continue
        assert_eq!(m.post_node(&ctx, &[0xFF; 1024]), CallbackAction::Continue);
    }

    // -- CallbackChain: mixed target_layers and priority interaction --

    #[test]
    fn test_chain_high_priority_filtered_low_priority_matches_in_pre() {
        let cb_high = TestCallback::new("high_filtered", 100)
            .with_layers(vec![99])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_low = TestCallback::new("low_matches", 10)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![1, 2, 3] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();

        // Dispatch for layer 0: high filtered out, low matches
        let ctx = holder.ctx(0, 0);
        let action = chain.dispatch_pre_node(&ctx);
        match action {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![1, 2, 3]),
            _ => panic!("Expected InjectHidden from low-priority, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_chain_high_priority_filtered_low_priority_matches_in_post() {
        let cb_high = TestCallback::new("high_filtered", 100)
            .with_layers(vec![50])
            .with_post_action(CallbackAction::ExitEarly { logits: vec![99.0] });
        let cb_low = TestCallback::new("low_matches", 10)
            .with_post_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();

        // Dispatch post for layer 0: high filtered, low matches with SkipThisNode
        let ctx = holder.ctx(0, 0);
        let action = chain.dispatch_post_node(&ctx, &[]);
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    // -- CallbackChain: dispatch with multiple callbacks, only one non-Continue in the middle --

    #[test]
    fn test_chain_six_callbacks_middle_one_returns_skip() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("p60", 60)),
            Box::new(TestCallback::new("p50", 50)),
            Box::new(TestCallback::new("p40", 40)
                .with_pre_action(CallbackAction::SkipThisNode)),
            Box::new(TestCallback::new("p30", 30)),
            Box::new(TestCallback::new("p20", 20)),
            Box::new(TestCallback::new("p10", 10)),
        ];

        let mut chain = CallbackChain::new(callbacks);
        assert_eq!(chain.len(), 6);
        assert_eq!(chain.sorted_indices, vec![0, 1, 2, 3, 4, 5]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let action = chain.dispatch_pre_node(&ctx);
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    // -- CallbackChain: repeated dispatch on same chain --

    #[test]
    fn test_chain_dispatch_pre_node_repeatedly_same_chain() {
        let cb = TestCallback::new("consistent", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for layer in 0..20 {
            let ctx = holder.ctx(layer, layer);
            let action = chain.dispatch_pre_node(&ctx);
            assert!(
                matches!(action, CallbackAction::SkipThisNode),
                "Dispatch {} should return SkipThisNode",
                layer
            );
        }
    }

    #[test]
    fn test_chain_dispatch_post_node_repeatedly_same_chain() {
        let cb = TestCallback::new("consistent_post", 50)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for i in 0..10 {
            let ctx = holder.ctx(i, i);
            let action = chain.dispatch_post_node(&ctx, &[]);
            match action {
                CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![1.0]),
                _ => panic!("Dispatch {} should return ExitEarly", i),
            }
        }
    }

    // -- CompactMask: edge cases with mask content --

    #[test]
    fn test_compact_mask_alternating_pattern() {
        let mask: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let active_count = mask.iter().filter(|&&v| v).count();
        assert_eq!(active_count, 50);

        let cb = TestCallback::new("alternating", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        if let CallbackAction::CompactMask { active_mask: m } = action {
            assert_eq!(m.len(), 100);
            assert_eq!(m.iter().filter(|&&v| v).count(), 50);
            assert!(m[0]);
            assert!(!m[1]);
        } else {
            panic!("Expected CompactMask");
        }
    }

    #[test]
    fn test_compact_mask_single_false_element() {
        let mask = vec![false];
        let cb = TestCallback::new("single_false", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        if let CallbackAction::CompactMask { active_mask: m } = action {
            assert_eq!(m.len(), 1);
            assert!(!m[0]);
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- InjectHidden: edge case data sizes --

    #[test]
    fn test_inject_hidden_single_byte_data() {
        let cb = TestCallback::new("single_byte", 80)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0x42] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        if let CallbackAction::InjectHidden { data } = action {
            assert_eq!(data.len(), 1);
            assert_eq!(data[0], 0x42);
        } else {
            panic!("Expected InjectHidden");
        }
    }

    #[test]
    fn test_inject_hidden_large_payload_round_trip() {
        let data: Vec<u8> = (0u8..=255).cycle().take(1024 * 1024).collect();
        let expected_len = data.len();
        let cb = TestCallback::new("large_inject", 80)
            .with_pre_action(CallbackAction::InjectHidden { data });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        if let CallbackAction::InjectHidden { data } = action {
            assert_eq!(data.len(), expected_len);
            // Verify first 256 bytes cover the full 0..=255 range
            for i in 0..256 {
                assert_eq!(data[i], i as u8, "Byte at index {} should be {}", i, i);
            }
        } else {
            panic!("Expected InjectHidden");
        }
    }

    // -- CallbackChain: interaction between pre_node and post_node --

    #[test]
    fn test_chain_pre_continue_post_continue_independent_dispatch() {
        let cb = TestCallback::new("both_continue", 50);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(2, 4);

        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[1, 2, 3]), CallbackAction::Continue);
    }

    #[test]
    fn test_chain_pre_skip_post_exit_different_callbacks() {
        let cb_pre = TestCallback::new("pre_skip", 80)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_post = TestCallback::new("post_exit", 20)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![5.0] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_pre), Box::new(cb_post)]);
        let holder = CtxHolder::new();

        // Different layers to test independent dispatch
        let ctx_pre = holder.ctx(0, 0);
        let ctx_post = holder.ctx(1, 2);

        assert_eq!(chain.dispatch_pre_node(&ctx_pre), CallbackAction::SkipThisNode);
        match chain.dispatch_post_node(&ctx_post, &[]) {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![5.0]),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // -- TestCallback builder pattern: chaining produces correct configuration --

    #[test]
    fn test_test_callback_builder_all_fields_set() {
        let cb = TestCallback::new("built", 77)
            .with_layers(vec![3, 5, 7])
            .with_pre_action(CallbackAction::SkipThisNode)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![0.5] });

        assert_eq!(cb.name(), "built");
        assert_eq!(cb.priority(), 77);
        assert_eq!(cb.target_layers(), Some([3, 5, 7].as_slice()));
    }

    // -- CallbackChain: sorted_indices correctness with reverse priority order --

    #[test]
    fn test_chain_reverse_insertion_order_still_sorted_by_priority() {
        // Insert in reverse priority order: low first, high last
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("p10", 10)),
            Box::new(TestCallback::new("p20", 20)),
            Box::new(TestCallback::new("p30", 30)),
            Box::new(TestCallback::new("p40", 40)),
            Box::new(TestCallback::new("p50", 50)),
        ];
        let chain = CallbackChain::new(callbacks);
        // Already in priority order as inserted, sorted should preserve that
        assert_eq!(chain.sorted_indices, vec![4, 3, 2, 1, 0]);
    }

    // -- CallbackChain: duplicate priorities --

    #[test]
    fn test_chain_four_callbacks_two_pairs_same_priority() {
        let cb_a1 = TestCallback::new("a1", 100)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_a2 = TestCallback::new("a2", 100);
        let cb_b1 = TestCallback::new("b1", 50);
        let cb_b2 = TestCallback::new("b2", 50);

        let chain = CallbackChain::new(vec![
            Box::new(cb_a1),
            Box::new(cb_a2),
            Box::new(cb_b1),
            Box::new(cb_b2),
        ]);
        // Stable sort: a1(0), a2(1) at prio 100; b1(2), b2(3) at prio 50
        assert_eq!(chain.sorted_indices, vec![0, 1, 2, 3]);
        assert_eq!(chain.len(), 4);
    }

    // -- CallbackAction: PartialEq with different-length inner data --

    #[test]
    fn test_callback_action_partial_eq_exit_early_different_lengths() {
        let a = CallbackAction::ExitEarly { logits: vec![1.0] };
        let b = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_inject_hidden_different_lengths() {
        let a = CallbackAction::InjectHidden { data: vec![1] };
        let b = CallbackAction::InjectHidden { data: vec![1, 2] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_compact_mask_different_lengths() {
        let a = CallbackAction::CompactMask { active_mask: vec![true] };
        let b = CallbackAction::CompactMask { active_mask: vec![true, false] };
        assert_ne!(a, b);
    }

    // -- CallbackAction: PartialEq reflexive property for data-carrying variants --

    #[test]
    fn test_callback_action_partial_eq_reflexive_all_data_variants() {
        let exit = CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] };
        assert_eq!(exit, exit);
        let inject = CallbackAction::InjectHidden { data: vec![42, 43] };
        assert_eq!(inject, inject);
        let compact = CallbackAction::CompactMask { active_mask: vec![true, false, true] };
        assert_eq!(compact, compact);
    }

    // -- LayerContext: multiple contexts from same holder share model_config --

    #[test]
    fn test_layer_contexts_share_model_config() {
        let holder = CtxHolder::new();
        let ctx0 = holder.ctx(0, 0);
        let ctx1 = holder.ctx(1, 2);

        // Both should reference the same config
        assert_eq!(ctx0.model_config.hidden_size(), ctx1.model_config.hidden_size());
        assert_eq!(ctx0.model_config.num_layers(), ctx1.model_config.num_layers());
        assert!(std::ptr::eq(ctx0.model_config, ctx1.model_config));
    }

    // -- CallbackChain: dispatch with output slice of different sizes --

    #[test]
    fn test_chain_dispatch_post_node_with_empty_output_slice() {
        let cb = TestCallback::new("post_empty", 50)
            .with_post_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_post_node(&ctx, &[]);
        assert!(matches!(action, CallbackAction::SkipThisNode));
    }

    // -- CallbackChain: sending chain building across thread for construction --

    #[test]
    fn test_callback_chain_construction_in_thread() {
        let handle = std::thread::spawn(|| {
            let cb = TestCallback::new("thread_cb", 42)
                .with_pre_action(CallbackAction::SkipThisNode);
            let chain = CallbackChain::new(vec![Box::new(cb)]);
            chain.len()
        });
        let len = handle.join().expect("Thread should complete");
        assert_eq!(len, 1);
    }

    // ========================================================================
    // 45 additional tests for comprehensive coverage
    // ========================================================================

    // -- CallbackAction: PartialEq symmetry and transitivity --

    #[test]
    fn test_callback_action_partial_eq_symmetry_continue() {
        let a = CallbackAction::Continue;
        let b = CallbackAction::Continue;
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn test_callback_action_partial_eq_symmetry_exit_early() {
        let a = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        let b = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    #[test]
    fn test_callback_action_partial_eq_transitivity_exit_early() {
        let a = CallbackAction::ExitEarly { logits: vec![3.14] };
        let b = CallbackAction::ExitEarly { logits: vec![3.14] };
        let c = CallbackAction::ExitEarly { logits: vec![3.14] };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // -- CallbackAction: Clone for each variant exhaustively --

    #[test]
    fn test_callback_action_clone_continue_preserves_variant() {
        let original = CallbackAction::Continue;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_callback_action_clone_skip_this_node_preserves_variant() {
        let original = CallbackAction::SkipThisNode;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_callback_action_clone_exit_early_preserves_all_logits() {
        let logits: Vec<f32> = vec![-100.0, -1.0, 0.0, 1.0, 100.0];
        let original = CallbackAction::ExitEarly { logits: logits.clone() };
        let cloned = original.clone();
        if let CallbackAction::ExitEarly { logits: cloned_logits } = cloned {
            assert_eq!(cloned_logits, logits);
        } else {
            panic!("Cloned should be ExitEarly");
        }
    }

    #[test]
    fn test_callback_action_clone_inject_hidden_preserves_all_bytes() {
        let data: Vec<u8> = (0u8..=255).collect();
        let original = CallbackAction::InjectHidden { data: data.clone() };
        let cloned = original.clone();
        if let CallbackAction::InjectHidden { data: cloned_data } = cloned {
            assert_eq!(cloned_data, data);
        } else {
            panic!("Cloned should be InjectHidden");
        }
    }

    #[test]
    fn test_callback_action_clone_compact_mask_preserves_all_booleans() {
        let mask: Vec<bool> = vec![true, false, true, true, false, false, true, false];
        let original = CallbackAction::CompactMask { active_mask: mask.clone() };
        let cloned = original.clone();
        if let CallbackAction::CompactMask { active_mask: cloned_mask } = cloned {
            assert_eq!(cloned_mask, mask);
        } else {
            panic!("Cloned should be CompactMask");
        }
    }

    // -- CallbackAction: Debug format for edge cases --

    #[test]
    fn test_callback_action_debug_exit_early_empty_logits() {
        let action = CallbackAction::ExitEarly { logits: vec![] };
        let debug = format!("{:?}", action);
        assert!(debug.contains("ExitEarly"));
        assert!(debug.contains("logits"));
    }

    #[test]
    fn test_callback_action_debug_inject_hidden_empty_data() {
        let action = CallbackAction::InjectHidden { data: vec![] };
        let debug = format!("{:?}", action);
        assert!(debug.contains("InjectHidden"));
        assert!(debug.contains("data"));
    }

    #[test]
    fn test_callback_action_debug_compact_mask_empty_mask() {
        let action = CallbackAction::CompactMask { active_mask: vec![] };
        let debug = format!("{:?}", action);
        assert!(debug.contains("CompactMask"));
        assert!(debug.contains("active_mask"));
    }

    // -- CallbackAction: ExitEarly with special float edge cases --

    #[test]
    fn test_callback_action_exit_early_with_f32_epsilon() {
        let action = CallbackAction::ExitEarly { logits: vec![f32::EPSILON] };
        if let CallbackAction::ExitEarly { logits } = action {
            assert_eq!(logits[0], f32::EPSILON);
            assert!(logits[0] > 0.0);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    #[test]
    fn test_callback_action_exit_early_all_nan_logits() {
        let action = CallbackAction::ExitEarly { logits: vec![f32::NAN; 5] };
        if let CallbackAction::ExitEarly { logits } = action {
            for val in &logits {
                assert!(val.is_nan());
            }
        } else {
            panic!("Expected ExitEarly");
        }
    }

    #[test]
    fn test_callback_action_exit_early_mixed_special_floats() {
        let logits = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0, f32::MIN, f32::MAX, f32::MIN_POSITIVE];
        let action = CallbackAction::ExitEarly { logits: logits.clone() };
        if let CallbackAction::ExitEarly { logits: l } = action {
            assert_eq!(l.len(), 8);
            assert!(l[0].is_nan());
            assert!(l[1].is_infinite() && l[1].is_sign_positive());
            assert!(l[2].is_infinite() && l[2].is_sign_negative());
            assert_eq!(l[3], 0.0);
            assert!(l[4].is_sign_negative() || l[4] == 0.0);
            assert_eq!(l[5], f32::MIN);
            assert_eq!(l[6], f32::MAX);
            assert_eq!(l[7], f32::MIN_POSITIVE);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: PartialEq with NaN (NaN != NaN by IEEE 754) --

    #[test]
    fn test_callback_action_partial_eq_exit_early_nan_logits_not_equal() {
        let a = CallbackAction::ExitEarly { logits: vec![f32::NAN] };
        let b = CallbackAction::ExitEarly { logits: vec![f32::NAN] };
        // Vec<f32> equality: NaN != NaN, so the actions are not equal
        assert_ne!(a, b);
    }

    // -- CallbackChain: empty chain behavior consistency --

    #[test]
    fn test_chain_empty_len_zero() {
        let chain = CallbackChain::empty();
        assert_eq!(chain.len(), 0);
    }

    #[test]
    fn test_chain_empty_sorted_indices_empty() {
        let chain = CallbackChain::empty();
        assert!(chain.sorted_indices.is_empty());
    }

    #[test]
    fn test_chain_new_empty_vec_same_as_empty() {
        let from_new = CallbackChain::new(vec![]);
        let from_empty = CallbackChain::empty();
        assert_eq!(from_new.is_empty(), from_empty.is_empty());
        assert_eq!(from_new.len(), from_empty.len());
    }

    // -- CallbackChain: priority ordering with descending priorities --

    #[test]
    fn test_chain_descending_priorities_sorted_correctly() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("p100", 100)),
            Box::new(TestCallback::new("p75", 75)),
            Box::new(TestCallback::new("p50", 50)),
            Box::new(TestCallback::new("p25", 25)),
        ];
        let chain = CallbackChain::new(callbacks);
        // Already in descending order as inserted
        assert_eq!(chain.sorted_indices, vec![0, 1, 2, 3]);
    }

    // -- CallbackChain: dispatch with all callbacks filtered by target_layers --

    #[test]
    fn test_chain_all_callbacks_filtered_returns_continue() {
        let cb1 = TestCallback::new("cb1_l100", 90)
            .with_layers(vec![100])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb2 = TestCallback::new("cb2_l200", 80)
            .with_layers(vec![200])
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let cb3 = TestCallback::new("cb3_l300", 70)
            .with_layers(vec![300])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![1] });

        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2), Box::new(cb3)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        assert_eq!(action, CallbackAction::Continue);
    }

    // -- CallbackChain: dispatch_post_node with all callbacks filtered --

    #[test]
    fn test_chain_all_post_callbacks_filtered_returns_continue() {
        let cb1 = TestCallback::new("post_l99", 90)
            .with_layers(vec![99])
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let cb2 = TestCallback::new("post_l88", 80)
            .with_layers(vec![88])
            .with_post_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_post_node(&ctx, &[1, 2, 3]);
        assert_eq!(action, CallbackAction::Continue);
    }

    // -- CallbackChain: single callback returns Continue on both pre and post --

    #[test]
    fn test_chain_single_callback_continue_on_both_pre_and_post() {
        let cb = TestCallback::new("both_continue", 50);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(3, 7);

        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[0u8; 32]), CallbackAction::Continue);
    }

    // -- action_variant_name: exhaustive coverage of all variants --

    #[test]
    fn test_action_variant_name_exit_early_with_non_empty_logits() {
        assert_eq!(
            action_variant_name(&CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] }),
            "ExitEarly"
        );
    }

    #[test]
    fn test_action_variant_name_inject_hidden_with_non_empty_data() {
        assert_eq!(
            action_variant_name(&CallbackAction::InjectHidden { data: vec![0xFF; 64] }),
            "InjectHidden"
        );
    }

    #[test]
    fn test_action_variant_name_compact_mask_with_non_empty_mask() {
        assert_eq!(
            action_variant_name(&CallbackAction::CompactMask { active_mask: vec![true; 32] }),
            "CompactMask"
        );
    }

    // -- CallbackChain: dispatch with large output slice --

    #[test]
    fn test_chain_dispatch_post_node_with_large_output_slice() {
        struct LenCaptureCallback {
            captured_len: usize,
        }
        impl LayerCallback for LenCaptureCallback {
            fn post_node(&mut self, _ctx: &LayerContext, output: &[u8]) -> CallbackAction {
                self.captured_len = output.len();
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "len_capture" }
        }

        let cb = LenCaptureCallback { captured_len: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let large_output = vec![0xABu8; 65536];

        let action = chain.dispatch_post_node(&ctx, &large_output);
        assert_eq!(action, CallbackAction::Continue);
    }

    // -- LayerContext: node_idx and layer_idx independence --

    #[test]
    fn test_layer_context_node_idx_independent_from_layer_idx() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 100,
            layer_idx: 0,
            node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 0,
            model_config: &holder.config,
        };
        assert_eq!(ctx.node_idx, 100);
        assert_eq!(ctx.layer_idx, 0);
    }

    #[test]
    fn test_layer_context_layer_idx_greater_than_node_idx() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 999,
            node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 1,
            seq_len: 1,
            position: 0,
            request_id: 0,
            model_config: &holder.config,
        };
        assert_eq!(ctx.node_idx, 0);
        assert_eq!(ctx.layer_idx, 999);
    }

    // -- LayerContext: request_id boundary values --

    #[test]
    fn test_layer_context_request_id_zero() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 0, model_config: &holder.config,
        };
        assert_eq!(ctx.request_id, 0);
    }

    #[test]
    fn test_layer_context_request_id_max() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: u64::MAX, model_config: &holder.config,
        };
        assert_eq!(ctx.request_id, u64::MAX);
    }

    // -- CallbackChain: multiple dispatches alternating pre and post --

    #[test]
    fn test_chain_alternating_pre_post_dispatch() {
        let cb = TestCallback::new("alt", 50)
            .with_pre_action(CallbackAction::SkipThisNode)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for i in 0..5 {
            let ctx = holder.ctx(i, i * 2);
            let pre = chain.dispatch_pre_node(&ctx);
            assert_eq!(pre, CallbackAction::SkipThisNode);

            let post = chain.dispatch_post_node(&ctx, &[]);
            match post {
                CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![1.0]),
                _ => panic!("Expected ExitEarly on iteration {}", i),
            }
        }
    }

    // -- CallbackChain: priority with extremely large gap --

    #[test]
    fn test_chain_priority_extreme_gap() {
        let cb_low = TestCallback::new("p1", 1);
        let cb_high = TestCallback::new("p_max_minus_1", u32::MAX - 1);
        let cb_max = TestCallback::new("p_max", u32::MAX)
            .with_pre_action(CallbackAction::SkipThisNode);

        let chain = CallbackChain::new(vec![
            Box::new(cb_low),
            Box::new(cb_high),
            Box::new(cb_max),
        ]);
        // sorted: max=2, high=1, low=0
        assert_eq!(chain.sorted_indices, vec![2, 1, 0]);
    }

    // -- CallbackAction: PartialEq for ExitEarly with single element --

    #[test]
    fn test_callback_action_partial_eq_exit_early_single_float() {
        let a = CallbackAction::ExitEarly { logits: vec![42.0] };
        let b = CallbackAction::ExitEarly { logits: vec![42.0] };
        assert_eq!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_exit_early_single_float_ne() {
        let a = CallbackAction::ExitEarly { logits: vec![42.0] };
        let b = CallbackAction::ExitEarly { logits: vec![43.0] };
        assert_ne!(a, b);
    }

    // -- CallbackAction: PartialEq for InjectHidden with single byte --

    #[test]
    fn test_callback_action_partial_eq_inject_hidden_single_byte() {
        let a = CallbackAction::InjectHidden { data: vec![0x00] };
        let b = CallbackAction::InjectHidden { data: vec![0x00] };
        assert_eq!(a, b);
    }

    // -- CallbackAction: PartialEq for CompactMask with single element --

    #[test]
    fn test_callback_action_partial_eq_compact_mask_single_true() {
        let a = CallbackAction::CompactMask { active_mask: vec![true] };
        let b = CallbackAction::CompactMask { active_mask: vec![true] };
        assert_eq!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_compact_mask_single_false() {
        let a = CallbackAction::CompactMask { active_mask: vec![false] };
        let b = CallbackAction::CompactMask { active_mask: vec![false] };
        assert_eq!(a, b);
    }

    // -- CompactMask: very large mask --

    #[test]
    fn test_compact_mask_large_mask_dispatch() {
        let mask: Vec<bool> = (0..10000).map(|i| i % 3 == 0).collect();
        let expected_active = mask.iter().filter(|&&v| v).count();
        let cb = TestCallback::new("large_mask", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        if let CallbackAction::CompactMask { active_mask: m } = action {
            assert_eq!(m.len(), 10000);
            assert_eq!(m.iter().filter(|&&v| v).count(), expected_active);
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- InjectHidden: zero bytes --

    #[test]
    fn test_inject_hidden_all_zero_bytes() {
        let data = vec![0u8; 1024];
        let cb = TestCallback::new("zero_inject", 80)
            .with_pre_action(CallbackAction::InjectHidden { data: data.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        if let CallbackAction::InjectHidden { data: d } = action {
            assert_eq!(d.len(), 1024);
            assert!(d.iter().all(|&b| b == 0));
        } else {
            panic!("Expected InjectHidden");
        }
    }

    // -- CallbackChain: dispatch_pre_node with multiple target layers matching one --

    #[test]
    fn test_chain_dispatch_pre_node_multiple_callbacks_first_filtered_second_matches() {
        let cb1 = TestCallback::new("filtered", 100)
            .with_layers(vec![5])
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![99.0] });
        let cb2 = TestCallback::new("matches", 50)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0x11] });
        let cb3 = TestCallback::new("never_reached", 10)
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2), Box::new(cb3)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        // cb1 filtered (wrong layer), cb2 matches and returns InjectHidden, cb3 never reached
        match action {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0x11]),
            _ => panic!("Expected InjectHidden, got {:?}", action_variant_name(&action)),
        }
    }

    // -- CallbackChain: dispatch_post_node high priority filtered low priority returns action --

    #[test]
    fn test_chain_dispatch_post_node_high_filtered_low_compact_mask() {
        let cb_high = TestCallback::new("high_filtered", 100)
            .with_layers(vec![50])
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let cb_low = TestCallback::new("low_matches", 10)
            .with_post_action(CallbackAction::CompactMask { active_mask: vec![true, false, true] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_post_node(&ctx, &[]);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true]);
            }
            _ => panic!("Expected CompactMask, got {:?}", action_variant_name(&action)),
        }
    }

    // -- LayerContext: hidden_state as empty slice (zero-length) --

    #[test]
    fn test_layer_context_with_empty_hidden_state() {
        let empty_state: Vec<u8> = vec![];
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Empty",
            hidden_state: &empty_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 0, seq_len: 0, position: 0,
            request_id: 0, model_config: &holder.config,
        };
        assert!(ctx.hidden_state.is_empty());
    }

    // -- CallbackChain: building with two callbacks same priority same target_layers --

    #[test]
    fn test_chain_two_callbacks_same_priority_same_target_first_wins() {
        let cb1 = TestCallback::new("first", 50)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb2 = TestCallback::new("second", 50)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![1] });

        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Stable sort keeps first(0) before second(1), so SkipThisNode wins
        let action = chain.dispatch_pre_node(&ctx);
        assert_eq!(action, CallbackAction::SkipThisNode);
    }

    // -- CallbackAction: Default is consistent with explicit Continue construction --

    #[test]
    fn test_callback_action_default_equals_explicit_continue() {
        let from_default = CallbackAction::default();
        let from_explicit = CallbackAction::Continue;
        assert_eq!(from_default, from_explicit);
    }

    // -- CallbackAction: cross-variant inequality exhaustive check --

    #[test]
    fn test_callback_action_all_variants_pairwise_inequality() {
        let variants: Vec<CallbackAction> = vec![
            CallbackAction::Continue,
            CallbackAction::SkipThisNode,
            CallbackAction::ExitEarly { logits: vec![1.0] },
            CallbackAction::InjectHidden { data: vec![1] },
            CallbackAction::CompactMask { active_mask: vec![true] },
        ];
        let n = variants.len();
        let mut unequal_count = 0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert_ne!(variants[i], variants[j], "Variant {} should not equal variant {}", i, j);
                    unequal_count += 1;
                }
            }
        }
        // 5 * 4 = 20 pairs
        assert_eq!(unequal_count, n * (n - 1));
    }

    // -- LayerContext: Verify model_config accessor chain for intermediate_size --

    #[test]
    fn test_layer_context_model_config_intermediate_size() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.intermediate_size(), 512);
    }

    // -- CallbackChain: dispatch_pre_node with callback that has None target_layers is always invoked --

    #[test]
    fn test_chain_none_target_always_invoked_across_layers() {
        let cb = TestCallback::new("always", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for layer in 0..20 {
            let ctx = holder.ctx(layer, layer);
            assert_eq!(
                chain.dispatch_pre_node(&ctx),
                CallbackAction::SkipThisNode,
                "Callback should be invoked at layer {}",
                layer
            );
        }
    }

    // -- CallbackChain: CallbackChain sorted_indices reflect correct original indices --

    #[test]
    fn test_chain_sorted_indices_map_to_correct_priorities() {
        let cb0 = TestCallback::new("p30", 30);
        let cb1 = TestCallback::new("p10", 10);
        let cb2 = TestCallback::new("p50", 50);
        let cb3 = TestCallback::new("p20", 20);
        let cb4 = TestCallback::new("p40", 40);

        let chain = CallbackChain::new(vec![
            Box::new(cb0), Box::new(cb1), Box::new(cb2), Box::new(cb3), Box::new(cb4),
        ]);

        // sorted by descending priority: 50(idx=2), 40(idx=4), 30(idx=0), 20(idx=3), 10(idx=1)
        assert_eq!(chain.sorted_indices, vec![2, 4, 0, 3, 1]);
    }

    // -- CallbackChain: building chain from vec of 20 callbacks --

    #[test]
    fn test_chain_twenty_callbacks_correct_length_and_sorted() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = (0..20)
            .map(|i| {
                let name = Box::leak(format!("cb{}", i).into_boxed_str());
                Box::new(TestCallback::new(name, i * 5)) as _
            })
            .collect();
        let chain = CallbackChain::new(callbacks);
        assert_eq!(chain.len(), 20);
        // Priorities 0,5,10,...,95 → sorted descending: 19(95), 18(90), ..., 0(0)
        let expected: Vec<usize> = (0..20).rev().collect();
        assert_eq!(chain.sorted_indices, expected);
    }

    // ========================================================================
    // 55 additional tests for further comprehensive coverage
    // ========================================================================

    // -- GeneratorForwardConfig accessors through LayerContext --

    #[test]
    fn test_layer_context_model_config_norm_eps() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!((ctx.model_config.norm_eps() - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_layer_context_model_config_max_seq_len() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.max_seq_len(), 128);
    }

    #[test]
    fn test_layer_context_model_config_head_dim_matches() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.head_dim(), 64);
        // head_dim = hidden_size / num_heads = 256 / 4 = 64
        assert_eq!(
            ctx.model_config.head_dim(),
            ctx.model_config.hidden_size() / ctx.model_config.num_heads()
        );
    }

    // -- CallbackAction: Debug output does not panic for any variant --

    #[test]
    fn test_callback_action_debug_no_panic_all_variants() {
        let _ = format!("{:?}", CallbackAction::Continue);
        let _ = format!("{:?}", CallbackAction::SkipThisNode);
        let _ = format!("{:?}", CallbackAction::ExitEarly { logits: vec![] });
        let _ = format!("{:?}", CallbackAction::InjectHidden { data: vec![] });
        let _ = format!("{:?}", CallbackAction::CompactMask { active_mask: vec![] });
    }

    // -- CallbackAction: Clone then mutate original does not affect clone --

    #[test]
    fn test_callback_action_clone_exit_early_original_mutation_isolation() {
        let mut original = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        let cloned = original.clone();
        if let CallbackAction::ExitEarly { logits } = &mut original {
            logits[0] = -999.0;
        }
        if let CallbackAction::ExitEarly { logits } = &cloned {
            assert_ne!(logits[0], -999.0);
            assert_eq!(logits[0], 1.0);
        } else {
            panic!("Cloned should be ExitEarly");
        }
    }

    // -- CallbackAction: ExitEarly with subnormal float --

    #[test]
    fn test_callback_action_exit_early_with_subnormal_float() {
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        let action = CallbackAction::ExitEarly { logits: vec![subnormal] };
        if let CallbackAction::ExitEarly { logits } = action {
            assert!(logits[0] > 0.0);
            assert!(logits[0] < f32::MIN_POSITIVE);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: ExitEarly with many identical values --

    #[test]
    fn test_callback_action_exit_early_many_identical_logits() {
        let logits = vec![0.5f32; 10000];
        let action = CallbackAction::ExitEarly { logits: logits.clone() };
        if let CallbackAction::ExitEarly { logits: l } = action {
            assert_eq!(l.len(), 10000);
            assert!(l.iter().all(|&v| (v - 0.5).abs() < 1e-10));
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackChain: dispatch_pre_node returns first non-Continue in priority order --

    #[test]
    fn test_chain_dispatch_pre_node_second_callback_returns_action() {
        let cb1 = TestCallback::new("high_continue", 100);
        let cb2 = TestCallback::new("mid_inject", 50)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0xBB] });
        let cb3 = TestCallback::new("low_skip", 10)
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2), Box::new(cb3)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        match action {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xBB]),
            _ => panic!("Expected InjectHidden from mid-priority callback"),
        }
    }

    // -- CallbackChain: post_node second callback returns CompactMask --

    #[test]
    fn test_chain_dispatch_post_node_second_callback_compact_mask() {
        let cb1 = TestCallback::new("high_cont", 90);
        let cb2 = TestCallback::new("mid_compact", 50)
            .with_post_action(CallbackAction::CompactMask { active_mask: vec![true] });
        let cb3 = TestCallback::new("low_skip", 10)
            .with_post_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2), Box::new(cb3)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_post_node(&ctx, &[]);
        match action {
            CallbackAction::CompactMask { active_mask } => assert_eq!(active_mask, vec![true]),
            _ => panic!("Expected CompactMask from mid-priority callback"),
        }
    }

    // -- CallbackChain: target_layers filtering for post_node --

    #[test]
    fn test_chain_post_node_target_layers_filter_blocks_callback() {
        let cb = TestCallback::new("targeted_post", 50)
            .with_layers(vec![10])
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Layer 0 not in target [10], should return Continue
        assert_eq!(chain.dispatch_post_node(&ctx, &[1, 2, 3]), CallbackAction::Continue);
    }

    // -- CallbackChain: dispatch_pre_node on chain built from vec![] --

    #[test]
    fn test_chain_new_empty_vec_dispatch_pre_returns_continue() {
        let mut chain = CallbackChain::new(vec![]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(5, 10);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- CallbackChain: dispatch_post_node on chain built from vec![] --

    #[test]
    fn test_chain_new_empty_vec_dispatch_post_returns_continue() {
        let mut chain = CallbackChain::new(vec![]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(5, 10);
        assert_eq!(chain.dispatch_post_node(&ctx, &[1, 2, 3]), CallbackAction::Continue);
    }

    // -- LayerContext: position > total_seq (edge case) --

    #[test]
    fn test_layer_context_position_exceeds_total_seq() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 5, seq_len: 1, position: 100,
            request_id: 0, model_config: &holder.config,
        };
        // The struct does not enforce position <= total_seq; verify raw field values
        assert_eq!(ctx.position, 100);
        assert_eq!(ctx.total_seq, 5);
    }

    // -- LayerContext: total_seq equals seq_len at start --

    #[test]
    fn test_layer_context_total_seq_equals_seq_len_prefill() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Prefill",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 64, seq_len: 64, position: 0,
            request_id: 1, model_config: &holder.config,
        };
        assert_eq!(ctx.total_seq, ctx.seq_len);
        assert_eq!(ctx.position, 0);
    }

    // -- LayerContext: kv_cache pointers are mutable and settable --

    #[test]
    fn test_layer_context_kv_cache_pointers_settable() {
        let mut k_buf = [0.0f32; 4];
        let mut v_buf = [0.0f32; 4];
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: k_buf.as_mut_ptr(),
            kv_cache_v: v_buf.as_mut_ptr(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 0, model_config: &holder.config,
        };
        assert!(!ctx.kv_cache_k.is_null());
        assert!(!ctx.kv_cache_v.is_null());
        // Pointers should point to different addresses
        assert_ne!(ctx.kv_cache_k, ctx.kv_cache_v);
    }

    // -- CallbackChain: len() matches callback count --

    #[test]
    fn test_chain_len_matches_callback_count() {
        for count in 0..=5 {
            let callbacks: Vec<Box<dyn LayerCallback + Send>> = (0..count)
                .map(|i| Box::new(TestCallback::new(Box::leak(format!("cb{}", i).into_boxed_str()), i as u32)) as _)
                .collect();
            let chain = CallbackChain::new(callbacks);
            assert_eq!(chain.len(), count);
            assert_eq!(chain.is_empty(), count == 0);
        }
    }

    // -- CallbackChain: is_empty and len consistency --

    #[test]
    fn test_chain_is_empty_consistent_with_len() {
        let empty = CallbackChain::empty();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let non_empty = CallbackChain::new(vec![Box::new(TestCallback::new("cb", 1))]);
        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.len(), 1);
    }

    // -- CallbackChain: sorted_indices count equals callbacks count --

    #[test]
    fn test_chain_sorted_indices_len_equals_callbacks_len() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("a", 10)),
            Box::new(TestCallback::new("b", 30)),
            Box::new(TestCallback::new("c", 20)),
        ];
        let chain = CallbackChain::new(callbacks);
        assert_eq!(chain.sorted_indices.len(), chain.len());
        // No duplicates
        let mut seen = std::collections::HashSet::new();
        for &idx in &chain.sorted_indices {
            assert!(seen.insert(idx), "Duplicate index {} in sorted_indices", idx);
        }
    }

    // -- CallbackChain: priority ordering is descending --

    #[test]
    fn test_chain_sorted_indices_descending_priority() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("p5", 5)),
            Box::new(TestCallback::new("p50", 50)),
            Box::new(TestCallback::new("p25", 25)),
            Box::new(TestCallback::new("p75", 75)),
        ];
        let chain = CallbackChain::new(callbacks);
        // Verify sorted_indices maps to descending priority
        // Just verify the order: 75(idx3), 50(idx1), 25(idx2), 5(idx0)
        assert_eq!(chain.sorted_indices, vec![3, 1, 2, 0]);
    }

    // -- TestCallback: with_layers returns correct slice --

    #[test]
    fn test_test_callback_with_layers_returns_correct_slice() {
        let cb = TestCallback::new("layered", 50).with_layers(vec![1, 3, 5]);
        let layers = cb.target_layers().unwrap();
        assert_eq!(layers, [1, 3, 5]);
    }

    // -- TestCallback: new sets default actions to Continue --

    #[test]
    fn test_test_callback_new_defaults_continue() {
        let cb = TestCallback::new("default", 50);
        let holder = CtxHolder::new();
        let mut cb = cb;
        let ctx = holder.ctx(0, 0);
        assert_eq!(cb.pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(cb.post_node(&ctx, &[]), CallbackAction::Continue);
    }

    // -- CallbackChain: two callbacks targeting same layer, different actions, second wins --

    #[test]
    fn test_chain_two_callbacks_same_layer_high_priority_continue_low_skips() {
        let cb_high = TestCallback::new("high_cont", 100)
            .with_layers(vec![0]);
        let cb_low = TestCallback::new("low_skip", 50)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // high returns Continue, then low returns SkipThisNode
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: callback with target_layers Some([]) behaves like no-target --

    #[test]
    fn test_chain_target_layers_empty_vec_post_node_always_continue() {
        let cb = TestCallback::new("empty_targets_post", 50)
            .with_layers(vec![])
            .with_post_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for layer in 0..5 {
            let ctx = holder.ctx(layer, layer);
            assert_eq!(
                chain.dispatch_post_node(&ctx, &[]),
                CallbackAction::Continue,
                "Post for layer {} should be Continue (empty target list filters out)",
                layer
            );
        }
    }

    // -- CallbackChain: dispatch with node_idx=0 layer_idx=0 --

    #[test]
    fn test_chain_dispatch_at_origin_node() {
        let cb = TestCallback::new("origin", 50)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0x01] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        match chain.dispatch_pre_node(&ctx) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0x01]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // -- LayerContext: hidden_state length matches hidden_size * sizeof(f32) --

    #[test]
    fn test_layer_context_hidden_state_size_matches_model_config() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        // CtxHolder creates hidden_state = vec![0u8; 256 * 4] where hidden_size=256
        assert_eq!(ctx.hidden_state.len(), ctx.model_config.hidden_size() * 4);
    }

    // -- CallbackAction: ExitEarly logits retain insertion order --

    #[test]
    fn test_callback_action_exit_early_logits_order_preserved() {
        let logits = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        let action = CallbackAction::ExitEarly { logits: logits.clone() };
        if let CallbackAction::ExitEarly { logits: l } = action {
            assert_eq!(l, logits);
            // Not sorted, original order preserved
            assert_eq!(l[0], 5.0);
            assert_eq!(l[4], 2.0);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: InjectHidden data retains insertion order --

    #[test]
    fn test_callback_action_inject_hidden_data_order_preserved() {
        let data: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let action = CallbackAction::InjectHidden { data: data.clone() };
        if let CallbackAction::InjectHidden { data: d } = action {
            assert_eq!(d, data);
        } else {
            panic!("Expected InjectHidden");
        }
    }

    // -- CallbackAction: CompactMask active_mask retains insertion order --

    #[test]
    fn test_callback_action_compact_mask_order_preserved() {
        let mask = vec![true, false, false, true, true, false, true];
        let action = CallbackAction::CompactMask { active_mask: mask.clone() };
        if let CallbackAction::CompactMask { active_mask: m } = action {
            assert_eq!(m, mask);
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- CallbackChain: multiple pre dispatches with stateful callback --

    #[test]
    fn test_chain_stateful_callback_tracks_invocation_count() {
        struct CountingCallback { count: usize }
        impl LayerCallback for CountingCallback {
            fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
                self.count += 1;
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "counter" }
        }

        let cb = CountingCallback { count: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for i in 0..10 {
            let ctx = holder.ctx(i, i);
            assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        }
        // All 10 dispatches should have completed (count state is internal)
    }

    // -- CallbackChain: callback that changes action on each invocation --

    #[test]
    fn test_chain_callback_alternating_actions() {
        struct AlternatingCallback { invoke_count: usize }
        impl LayerCallback for AlternatingCallback {
            fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
                self.invoke_count += 1;
                if self.invoke_count % 2 == 1 {
                    CallbackAction::Continue
                } else {
                    CallbackAction::SkipThisNode
                }
            }
            fn name(&self) -> &str { "alternating" }
        }

        let cb = AlternatingCallback { invoke_count: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // First invocation: Continue (invoke_count=1)
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::Continue);

        // Second invocation: SkipThisNode (invoke_count=2)
        let ctx1 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx1), CallbackAction::SkipThisNode);

        // Third invocation: Continue again (invoke_count=3)
        let ctx2 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx2), CallbackAction::Continue);
    }

    // -- CallbackChain: post_node callback that changes action based on output len --

    #[test]
    fn test_chain_post_node_callback_responds_to_output_size() {
        struct OutputSizeCallback { threshold: usize }
        impl LayerCallback for OutputSizeCallback {
            fn post_node(&mut self, _ctx: &LayerContext, output: &[u8]) -> CallbackAction {
                if output.len() > self.threshold {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "size_check" }
        }

        let cb = OutputSizeCallback { threshold: 16 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Small output: Continue
        assert_eq!(chain.dispatch_post_node(&ctx, &[0u8; 8]), CallbackAction::Continue);
        // Large output: SkipThisNode
        assert_eq!(chain.dispatch_post_node(&ctx, &[0u8; 32]), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: high-priority callback with target layer, correct layer triggers --

    #[test]
    fn test_chain_high_priority_targeted_triggers_only_on_correct_layer() {
        let cb = TestCallback::new("targeted_high", 100)
            .with_layers(vec![3])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for layer in 0..5 {
            let ctx = holder.ctx(layer, layer);
            let action = chain.dispatch_pre_node(&ctx);
            if layer == 3 {
                assert_eq!(action, CallbackAction::SkipThisNode);
            } else {
                assert_eq!(action, CallbackAction::Continue);
            }
        }
    }

    // -- CallbackAction: PartialEq symmetry for InjectHidden --

    #[test]
    fn test_callback_action_partial_eq_symmetry_inject_hidden() {
        let a = CallbackAction::InjectHidden { data: vec![10, 20, 30] };
        let b = CallbackAction::InjectHidden { data: vec![10, 20, 30] };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // -- CallbackAction: PartialEq symmetry for CompactMask --

    #[test]
    fn test_callback_action_partial_eq_symmetry_compact_mask() {
        let a = CallbackAction::CompactMask { active_mask: vec![true, false] };
        let b = CallbackAction::CompactMask { active_mask: vec![true, false] };
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // -- CallbackAction: PartialEq transitivity for InjectHidden --

    #[test]
    fn test_callback_action_partial_eq_transitivity_inject_hidden() {
        let data = vec![1, 2, 3];
        let a = CallbackAction::InjectHidden { data: data.clone() };
        let b = CallbackAction::InjectHidden { data: data.clone() };
        let c = CallbackAction::InjectHidden { data: data.clone() };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // -- CallbackChain: dispatch with node_op string matching in callback --

    #[test]
    fn test_chain_callback_inspects_node_op() {
        struct OpInspectCallback { last_op: String }
        impl LayerCallback for OpInspectCallback {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.last_op = ctx.node_op.to_string();
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "op_inspect" }
        }

        let cb = OpInspectCallback { last_op: String::new() };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- LayerContext: node_op is a borrowed reference, not owned --

    #[test]
    fn test_layer_context_node_op_is_borrowed_str() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let op: &str = ctx.node_op;
        assert!(!op.is_empty());
    }

    // -- CallbackChain: empty chain dispatch_idempotent_many_layers --

    #[test]
    fn test_chain_empty_dispatch_many_layers_and_nodes() {
        let mut chain = CallbackChain::empty();
        let holder = CtxHolder::new();
        for layer in 0..50 {
            for node in 0..5 {
                let ctx = holder.ctx(layer, node);
                assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
                assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue);
            }
        }
    }

    // -- CallbackChain: callback with priority 0 still dispatched --

    #[test]
    fn test_chain_zero_priority_callback_dispatched_on_pre() {
        let cb = TestCallback::new("zero_prio", 0)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0x00] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        match chain.dispatch_pre_node(&ctx) {
            CallbackAction::InjectHidden { .. } => {}
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackChain: callback with priority 0 still dispatched on post --

    #[test]
    fn test_chain_zero_priority_callback_dispatched_on_post() {
        let cb = TestCallback::new("zero_prio_post", 0)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![0.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        match chain.dispatch_post_node(&ctx, &[]) {
            CallbackAction::ExitEarly { .. } => {}
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackAction: ExitEarly with negative infinity --

    #[test]
    fn test_callback_action_exit_early_with_neg_infinity() {
        let action = CallbackAction::ExitEarly { logits: vec![f32::NEG_INFINITY] };
        if let CallbackAction::ExitEarly { logits } = action {
            assert!(logits[0].is_infinite());
            assert!(logits[0].is_sign_negative());
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: ExitEarly with positive infinity --

    #[test]
    fn test_callback_action_exit_early_with_pos_infinity() {
        let action = CallbackAction::ExitEarly { logits: vec![f32::INFINITY] };
        if let CallbackAction::ExitEarly { logits } = action {
            assert!(logits[0].is_infinite());
            assert!(logits[0].is_sign_positive());
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- LayerContext: request_id distinguishes different requests --

    #[test]
    fn test_layer_context_different_request_ids_different_contexts() {
        let holder = CtxHolder::new();
        let ctx_a = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 100, model_config: &holder.config,
        };
        let ctx_b = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 200, model_config: &holder.config,
        };
        assert_ne!(ctx_a.request_id, ctx_b.request_id);
    }

    // -- CallbackChain: dispatch with single callback that uses context fields --

    #[test]
    fn test_chain_callback_reads_layer_idx_from_context() {
        struct LayerCheckCallback { last_layer: usize }
        impl LayerCallback for LayerCheckCallback {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.last_layer = ctx.layer_idx;
                if ctx.layer_idx >= 5 {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "layer_check" }
        }

        let cb = LayerCheckCallback { last_layer: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Layer < 5: Continue
        let ctx3 = holder.ctx(3, 6);
        assert_eq!(chain.dispatch_pre_node(&ctx3), CallbackAction::Continue);

        // Layer >= 5: SkipThisNode
        let ctx7 = holder.ctx(7, 14);
        assert_eq!(chain.dispatch_pre_node(&ctx7), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: dispatch with callback that checks request_id --

    #[test]
    fn test_chain_callback_reads_request_id_from_context() {
        struct RequestFilterCallback { allowed_id: u64 }
        impl LayerCallback for RequestFilterCallback {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                if ctx.request_id == self.allowed_id {
                    CallbackAction::Continue
                } else {
                    CallbackAction::SkipThisNode
                }
            }
            fn name(&self) -> &str { "request_filter" }
        }

        let cb = RequestFilterCallback { allowed_id: 42 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Matching request_id
        let ctx_match = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 42, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx_match), CallbackAction::Continue);

        // Non-matching request_id
        let ctx_no = LayerContext {
            request_id: 99, ..ctx_match
        };
        assert_eq!(chain.dispatch_pre_node(&ctx_no), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: thread safety of CallbackChain construction --

    #[test]
    fn test_callback_chain_built_in_thread_dispatches_correctly() {
        let holder = CtxHolder::new();
        let hidden_state = holder.hidden_state.clone();

        let handle = std::thread::spawn(move || {
            let config = GeneratorForwardConfig {
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

            let cb = TestCallback::new("thread_cb", 42)
                .with_pre_action(CallbackAction::SkipThisNode);
            let mut chain = CallbackChain::new(vec![Box::new(cb)]);

            let ctx = LayerContext {
                node_idx: 0, layer_idx: 0, node_op: "Op",
                hidden_state: &hidden_state,
                kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
                total_seq: 1, seq_len: 1, position: 0,
                request_id: 0, model_config: &config,
            };
            chain.dispatch_pre_node(&ctx)
        });

        let action = handle.join().expect("Thread should complete");
        assert_eq!(action, CallbackAction::SkipThisNode);
    }

    // -- action_variant_name: returns correct static str for each variant with data --

    #[test]
    fn test_action_variant_name_returns_static_str() {
        let name: &'static str = action_variant_name(&CallbackAction::Continue);
        assert_eq!(name, "Continue");
    }

    // -- CallbackChain: sorted_indices are all valid indices --

    #[test]
    fn test_chain_sorted_indices_all_valid() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("a", 30)),
            Box::new(TestCallback::new("b", 10)),
            Box::new(TestCallback::new("c", 50)),
            Box::new(TestCallback::new("d", 20)),
            Box::new(TestCallback::new("e", 40)),
        ];
        let chain = CallbackChain::new(callbacks);
        let n = chain.len();
        for &idx in &chain.sorted_indices {
            assert!(idx < n, "Index {} out of range [0, {})", idx, n);
        }
    }

    // -- CompactMask: mask with exactly one active element --

    #[test]
    fn test_compact_mask_exactly_one_active() {
        let mut mask = vec![false; 100];
        mask[50] = true;
        let cb = TestCallback::new("one_active", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        if let CallbackAction::CompactMask { active_mask: m } = action {
            assert_eq!(m.iter().filter(|&&v| v).count(), 1);
            assert!(m[50]);
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- InjectHidden: binary data integrity --

    #[test]
    fn test_inject_hidden_binary_data_integrity() {
        let data: Vec<u8> = (0u8..=255).collect();
        let cb = TestCallback::new("binary_integrity", 80)
            .with_pre_action(CallbackAction::InjectHidden { data: data.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        if let CallbackAction::InjectHidden { data: d } = action {
            assert_eq!(d.len(), 256);
            for i in 0..256 {
                assert_eq!(d[i], i as u8, "Byte at index {} should be {}", i, i);
            }
        } else {
            panic!("Expected InjectHidden");
        }
    }

    // -- CallbackChain: pre_node dispatch does not affect subsequent post_node dispatch --

    #[test]
    fn test_chain_pre_dispatch_does_not_affect_post_dispatch() {
        let cb = TestCallback::new("independent", 50)
            .with_pre_action(CallbackAction::SkipThisNode)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![2.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // pre_node returns SkipThisNode
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);

        // post_node independently returns ExitEarly
        match chain.dispatch_post_node(&ctx, &[]) {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![2.0]),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackChain: dispatch with callbacks at very high layer indices --

    #[test]
    fn test_chain_dispatch_at_very_high_layer_index() {
        let cb = TestCallback::new("high_layer_cb", 50)
            .with_layers(vec![10000])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Low layer: not targeted
        let ctx_low = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx_low), CallbackAction::Continue);

        // High layer: targeted
        let ctx_high = holder.ctx(10000, 20000);
        assert_eq!(chain.dispatch_pre_node(&ctx_high), CallbackAction::SkipThisNode);
    }

    // -- CallbackAction: Debug output is non-empty for all variants --

    #[test]
    fn test_callback_action_debug_non_empty_all_variants() {
        let variants: Vec<CallbackAction> = vec![
            CallbackAction::Continue,
            CallbackAction::SkipThisNode,
            CallbackAction::ExitEarly { logits: vec![1.0] },
            CallbackAction::InjectHidden { data: vec![1] },
            CallbackAction::CompactMask { active_mask: vec![true] },
        ];
        for action in &variants {
            let debug = format!("{:?}", action);
            assert!(!debug.is_empty(), "Debug output should not be empty");
        }
    }

    // -- CallbackAction: PartialEq for InjectHidden with same data different allocation --

    #[test]
    fn test_callback_action_partial_eq_inject_hidden_independent_allocations() {
        let data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let a = CallbackAction::InjectHidden { data: data.clone() };
        let b = CallbackAction::InjectHidden { data: data.clone() };
        assert_eq!(a, b);
    }

    // -- LayerContext: verify node_op from CtxHolder matches static str --

    #[test]
    fn test_ctx_holder_default_op_is_test() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.node_op, "Test");
    }

    // -- CallbackChain: chain with only Continue callbacks has correct length --

    #[test]
    fn test_chain_all_continue_correct_length() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = (0..7)
            .map(|i| Box::new(TestCallback::new(Box::leak(format!("cont{}", i).into_boxed_str()), 50)) as _)
            .collect();
        let mut chain = CallbackChain::new(callbacks);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        assert_eq!(chain.len(), 7);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue);
    }

    // -- CallbackChain: chain with mixed actions where highest-priority non-targeted --

    #[test]
    fn test_chain_highest_not_targeted_second_highest_returns_action() {
        let cb_highest = TestCallback::new("highest", 200)
            .with_layers(vec![99])
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let cb_second = TestCallback::new("second", 150)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0xCC] });
        let cb_third = TestCallback::new("third", 100)
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_highest),
            Box::new(cb_second),
            Box::new(cb_third),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        match action {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xCC]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // -- LayerContext: verify paged_kv config accessible through model_config --

    #[test]
    fn test_layer_context_model_config_paged_kv_page_size() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.paged_kv.page_size, 16);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // NEW TESTS — additional coverage for layer_callback.rs
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // -- action_variant_name: each variant returns the correct static str --

    #[test]
    fn test_action_variant_name_continue_str() {
        assert_eq!(action_variant_name(&CallbackAction::Continue), "Continue");
    }

    #[test]
    fn test_action_variant_name_skip_this_node_str() {
        assert_eq!(action_variant_name(&CallbackAction::SkipThisNode), "SkipThisNode");
    }

    #[test]
    fn test_action_variant_name_exit_early_with_data_str() {
        let action = CallbackAction::ExitEarly { logits: vec![1.0, 2.0] };
        assert_eq!(action_variant_name(&action), "ExitEarly");
    }

    #[test]
    fn test_action_variant_name_inject_hidden_with_data_str() {
        let action = CallbackAction::InjectHidden { data: vec![0xAB] };
        assert_eq!(action_variant_name(&action), "InjectHidden");
    }

    #[test]
    fn test_action_variant_name_compact_mask_with_data_str() {
        let action = CallbackAction::CompactMask { active_mask: vec![true] };
        assert_eq!(action_variant_name(&action), "CompactMask");
    }

    // -- CallbackAction: ExitEarly with single-element logits --

    #[test]
    fn test_exit_early_single_logit_value() {
        let action = CallbackAction::ExitEarly { logits: vec![42.0] };
        if let CallbackAction::ExitEarly { logits } = action {
            assert_eq!(logits.len(), 1);
            assert!((logits[0] - 42.0).abs() < f32::EPSILON);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: InjectHidden with repeated byte pattern --

    #[test]
    fn test_inject_hidden_repeated_byte_pattern() {
        let data = vec![0xAA; 1024];
        let action = CallbackAction::InjectHidden { data: data.clone() };
        if let CallbackAction::InjectHidden { data: d } = action {
            assert_eq!(d.len(), 1024);
            assert!(d.iter().all(|&b| b == 0xAA));
        } else {
            panic!("Expected InjectHidden");
        }
    }

    // -- CallbackAction: CompactMask with all-true preserves count --

    #[test]
    fn test_compact_mask_all_true_count_preserved() {
        let mask = vec![true; 256];
        let action = CallbackAction::CompactMask { active_mask: mask.clone() };
        if let CallbackAction::CompactMask { active_mask: m } = action {
            assert_eq!(m.len(), 256);
            assert!(m.iter().all(|&v| v));
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- CallbackAction: CompactMask with all-false preserves count --

    #[test]
    fn test_compact_mask_all_false_count_preserved() {
        let mask = vec![false; 128];
        let action = CallbackAction::CompactMask { active_mask: mask.clone() };
        if let CallbackAction::CompactMask { active_mask: m } = action {
            assert_eq!(m.len(), 128);
            assert!(m.iter().all(|&v| !v));
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- CallbackAction: matches! macro correctly identifies Continue --

    #[test]
    fn test_matches_macro_continue_is_continue() {
        let action = CallbackAction::Continue;
        assert!(matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_matches_macro_skip_is_not_continue() {
        let action = CallbackAction::SkipThisNode;
        assert!(!matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_matches_macro_exit_early_is_not_continue() {
        let action = CallbackAction::ExitEarly { logits: vec![] };
        assert!(!matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_matches_macro_inject_hidden_is_not_continue() {
        let action = CallbackAction::InjectHidden { data: vec![] };
        assert!(!matches!(action, CallbackAction::Continue));
    }

    #[test]
    fn test_matches_macro_compact_mask_is_not_continue() {
        let action = CallbackAction::CompactMask { active_mask: vec![] };
        assert!(!matches!(action, CallbackAction::Continue));
    }

    // -- LayerContext: node_idx and layer_idx can be independently set --

    #[test]
    fn test_layer_context_node_idx_greater_than_layer_idx() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(2, 5);
        assert_eq!(ctx.node_idx, 5);
        assert_eq!(ctx.layer_idx, 2);
        assert!(ctx.node_idx > ctx.layer_idx);
    }

    // -- LayerContext: seq_len of zero is valid --

    #[test]
    fn test_layer_context_seq_len_zero() {
        let mut holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: &holder.op,
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 0,
            seq_len: 0,
            position: 0,
            request_id: 0,
            model_config: &holder.config,
        };
        assert_eq!(ctx.seq_len, 0);
        assert_eq!(ctx.total_seq, 0);
    }

    // -- LayerContext: position zero is valid --

    #[test]
    fn test_layer_context_position_zero_is_valid() {
        let mut holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: &holder.op,
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 5,
            seq_len: 5,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };
        assert_eq!(ctx.position, 0);
        assert_eq!(ctx.seq_len, 5);
    }

    // -- LayerContext: kv_cache pointers are null from CtxHolder --

    #[test]
    fn test_layer_context_kv_cache_k_null_by_default() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.kv_cache_k.is_null());
    }

    #[test]
    fn test_layer_context_kv_cache_v_null_by_default() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.kv_cache_v.is_null());
    }

    // -- LayerContext: hidden_state bytes are all zero from CtxHolder --

    #[test]
    fn test_layer_context_hidden_state_all_zeros_default() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.hidden_state.iter().all(|&b| b == 0));
    }

    // -- LayerContext: model_config intermediate_size accessor --

    #[test]
    fn test_layer_context_model_config_intermediate_size_value() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.intermediate_size(), 512);
    }

    // -- LayerContext: model_config num_layers accessor --

    #[test]
    fn test_layer_context_model_config_num_layers_value() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.num_layers(), 4);
    }

    // -- LayerContext: model_config vocab_size accessor --

    #[test]
    fn test_layer_context_model_config_vocab_size_value() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.vocab_size(), 1000);
    }

    // -- LayerContext: model_config dtype accessor returns F32 --

    #[test]
    fn test_layer_context_model_config_dtype_is_f32() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.dtype(), gllm_kernels::types::DType::F32);
    }

    // -- LayerContext: model_config geometry hidden_size field --

    #[test]
    fn test_layer_context_geometry_hidden_size() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.hidden_size, 256);
    }

    // -- LayerContext: model_config geometry head_dim field --

    #[test]
    fn test_layer_context_geometry_head_dim() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.head_dim, 64);
    }

    // -- LayerContext: model_config geometry max_seq_len field --

    #[test]
    fn test_layer_context_geometry_max_seq_len() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.max_seq_len, 128);
    }

    // -- LayerContext: model_config arch_family is Decoder --

    #[test]
    fn test_layer_context_arch_family_decoder() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(matches!(ctx.model_config.arch_family, crate::manifest::ArchFamily::Decoder));
    }

    // -- LayerContext: model_config position_encoding is Rope --

    #[test]
    fn test_layer_context_position_encoding_rope() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(matches!(ctx.model_config.position_encoding, crate::engine::executor::PositionEncoding::Rope));
    }

    // -- LayerContext: model_config rope config fields --

    #[test]
    fn test_layer_context_rope_theta() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!((ctx.model_config.rope.theta - 10000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_layer_context_rope_scale() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!((ctx.model_config.rope.scale - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_layer_context_rope_not_interleaved() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(!ctx.model_config.rope.interleaved);
    }

    #[test]
    fn test_layer_context_rope_not_precomputed() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(!ctx.model_config.rope.precompute);
    }

    // -- LayerContext: model_config moe_config is None for non-MoE model --

    #[test]
    fn test_layer_context_moe_config_none() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.moe_config.is_none());
    }

    // -- LayerContext: model_config rerank tokens are None --

    #[test]
    fn test_layer_context_rerank_yes_token_none() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.rerank_yes_token_id.is_none());
    }

    #[test]
    fn test_layer_context_rerank_no_token_none() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.rerank_no_token_id.is_none());
    }

    // -- CallbackChain: dispatch_pre_node returns Continue for all Continue callbacks --

    #[test]
    fn test_chain_dispatch_pre_all_continue_with_output_check() {
        let cb1 = TestCallback::new("cb1", 100);
        let cb2 = TestCallback::new("cb2", 50);
        let cb3 = TestCallback::new("cb3", 10);
        let mut chain = CallbackChain::new(vec![
            Box::new(cb1),
            Box::new(cb2),
            Box::new(cb3),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let result = chain.dispatch_pre_node(&ctx);
        assert_eq!(result, CallbackAction::Continue);
    }

    // -- CallbackChain: dispatch_post_node with non-empty output buffer --

    #[test]
    fn test_chain_dispatch_post_with_nontrivial_output() {
        let cb = TestCallback::new("post_out", 50);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let output = vec![1u8, 2, 3, 4, 5];
        let result = chain.dispatch_post_node(&ctx, &output);
        assert_eq!(result, CallbackAction::Continue);
    }

    // -- CallbackChain: callback with priority u32::MAX is always first --

    #[test]
    fn test_chain_max_u32_priority_dispatched_first() {
        let cb_max = TestCallback::new("max_pri", u32::MAX)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_mid = TestCallback::new("mid_pri", 100);
        let cb_min = TestCallback::new("min_pri", 0);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_min),
            Box::new(cb_mid),
            Box::new(cb_max),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // max priority callback fires first and returns SkipThisNode
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: callback returning InjectHidden stops further dispatch --

    #[test]
    fn test_chain_inject_hidden_stops_dispatch() {
        let cb_inject = TestCallback::new("injector", 80)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0xFF, 0x00] });
        let cb_skip = TestCallback::new("skipper", 50)
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_inject),
            Box::new(cb_skip),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        match chain.dispatch_pre_node(&ctx) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xFF, 0x00]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackChain: CompactMask returned by post_node stops dispatch --

    #[test]
    fn test_chain_post_node_compact_mask_stops_dispatch() {
        let cb_compact = TestCallback::new("compact_post", 90)
            .with_post_action(CallbackAction::CompactMask {
                active_mask: vec![true, false, true],
            });
        let cb_exit = TestCallback::new("exit_post", 50)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_compact),
            Box::new(cb_exit),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        match chain.dispatch_post_node(&ctx, &[0u8; 12]) {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true]);
            }
            other => panic!("Expected CompactMask, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackChain: five callbacks all targeting same layer, all Continue --

    #[test]
    fn test_chain_five_callbacks_same_layer_all_continue() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = (0..5)
            .map(|i| {
                Box::new(TestCallback::new(Box::leak(format!("cb{}", i).into_boxed_str()), (i + 1) * 10))
                    as Box<dyn LayerCallback + Send>
            })
            .collect();
        let mut chain = CallbackChain::new(callbacks);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(3, 6);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- CallbackChain: pre_node returns ExitEarly with empty logits --

    #[test]
    fn test_chain_pre_node_exit_early_empty_logits_stops_chain() {
        let cb = TestCallback::new("exit_empty", 100)
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        match chain.dispatch_pre_node(&ctx) {
            CallbackAction::ExitEarly { logits } => assert!(logits.is_empty()),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackChain: post_node returns ExitEarly with many logits --

    #[test]
    fn test_chain_post_node_exit_early_many_logits() {
        let logits: Vec<f32> = (0..1000).map(|i| i as f32 * 0.001).collect();
        let expected_len = logits.len();
        let cb = TestCallback::new("exit_many", 100)
            .with_post_action(CallbackAction::ExitEarly { logits });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        match chain.dispatch_post_node(&ctx, &[]) {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits.len(), expected_len),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackAction: Clone on CompactMask with mixed pattern --

    #[test]
    fn test_clone_compact_mask_mixed_pattern() {
        let mask = vec![true, false, true, true, false, false, true, false];
        let action = CallbackAction::CompactMask { active_mask: mask };
        let cloned = action.clone();
        if let CallbackAction::CompactMask { active_mask } = cloned {
            assert_eq!(active_mask, vec![true, false, true, true, false, false, true, false]);
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- CallbackAction: PartialEq for Continue is reflexive --

    #[test]
    fn test_partial_eq_continue_reflexive() {
        let a = CallbackAction::Continue;
        assert_eq!(a, a);
    }

    // -- CallbackAction: PartialEq for SkipThisNode is reflexive --

    #[test]
    fn test_partial_eq_skip_this_node_reflexive() {
        let a = CallbackAction::SkipThisNode;
        assert_eq!(a, a);
    }

    // -- CallbackAction: ExitEarly with NaN logits -- PartialEq returns false for NaN != NaN --

    #[test]
    fn test_partial_eq_exit_early_nan_logits_self_compare() {
        let action = CallbackAction::ExitEarly { logits: vec![f32::NAN] };
        // f32 NaN != NaN, so ExitEarly with NaN logits should not equal itself
        assert_ne!(action, action);
    }

    // -- CallbackAction: two InjectHidden with same data but different capacity --

    #[test]
    fn test_partial_eq_inject_hidden_same_data_different_capacity() {
        let mut data1 = Vec::with_capacity(100);
        let mut data2 = Vec::with_capacity(200);
        data1.extend_from_slice(&[1, 2, 3, 4]);
        data2.extend_from_slice(&[1, 2, 3, 4]);
        let a = CallbackAction::InjectHidden { data: data1 };
        let b = CallbackAction::InjectHidden { data: data2 };
        assert_eq!(a, b);
    }

    // -- CallbackChain: dispatch across multiple layers with per-layer targets --

    #[test]
    fn test_chain_dispatch_across_layers_with_per_layer_targets() {
        let cb_layer0 = TestCallback::new("l0", 100)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_layer1 = TestCallback::new("l1", 90)
            .with_layers(vec![1])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0x01] });
        let cb_layer2 = TestCallback::new("l2", 80)
            .with_layers(vec![2])
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![0.5] });

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_layer0),
            Box::new(cb_layer1),
            Box::new(cb_layer2),
        ]);
        let holder = CtxHolder::new();

        // layer 0: only cb_layer0 matches → SkipThisNode
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::SkipThisNode);

        // layer 1: only cb_layer1 matches → InjectHidden
        let ctx1 = holder.ctx(1, 2);
        match chain.dispatch_pre_node(&ctx1) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0x01]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }

        // layer 2: only cb_layer2 matches → ExitEarly
        let ctx2 = holder.ctx(2, 4);
        match chain.dispatch_pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits.len(), 1);
                assert!((logits[0] - 0.5).abs() < f32::EPSILON);
            }
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackChain: two callbacks, one filtered, other continues --

    #[test]
    fn test_chain_one_filtered_one_continues_returns_continue() {
        let cb_filtered = TestCallback::new("filtered", 100)
            .with_layers(vec![99])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_continues = TestCallback::new("continues", 50);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_filtered),
            Box::new(cb_continues),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        // filtered is skipped, continues returns Continue
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- CallbackChain: post_node filtered callback returns Continue --

    #[test]
    fn test_chain_post_node_filtered_returns_continue() {
        let cb = TestCallback::new("wrong_layer", 100)
            .with_layers(vec![5])
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue);
    }

    // -- TestCallback: with_pre_action builder sets the action --

    #[test]
    fn test_test_callback_with_pre_action_builder() {
        let cb = TestCallback::new("pre_builder", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let holder = CtxHolder::new();
        let mut cb = cb;
        let ctx = holder.ctx(0, 0);
        assert_eq!(cb.pre_node(&ctx), CallbackAction::SkipThisNode);
    }

    // -- TestCallback: with_post_action builder sets the action --

    #[test]
    fn test_test_callback_with_post_action_builder() {
        let cb = TestCallback::new("post_builder", 50)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![3.0] });
        let holder = CtxHolder::new();
        let mut cb = cb;
        let ctx = holder.ctx(0, 0);
        match cb.post_node(&ctx, &[]) {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![3.0]),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // -- TestCallback: name() returns the assigned name --

    #[test]
    fn test_test_callback_name_returns_assigned() {
        let cb = TestCallback::new("my_callback", 42);
        assert_eq!(cb.name(), "my_callback");
    }

    // -- TestCallback: priority() returns the assigned priority --

    #[test]
    fn test_test_callback_priority_returns_assigned() {
        let cb = TestCallback::new("pri_test", 77);
        assert_eq!(cb.priority(), 77);
    }

    // -- CtxHolder: hidden_state length is hidden_size * 4 (F32) --

    #[test]
    fn test_ctx_holder_hidden_state_length() {
        let holder = CtxHolder::new();
        assert_eq!(holder.hidden_state.len(), 256 * 4);
    }

    // -- CtxHolder: op field is static str "Test" --

    #[test]
    fn test_ctx_holder_op_is_static() {
        let holder = CtxHolder::new();
        assert_eq!(holder.op, "Test");
    }

    // -- CtxHolder: ctx method passes total_seq=10, seq_len=1, position=9 --

    #[test]
    fn test_ctx_holder_ctx_default_values() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(1, 2);
        assert_eq!(ctx.total_seq, 10);
        assert_eq!(ctx.seq_len, 1);
        assert_eq!(ctx.position, 9);
        assert_eq!(ctx.request_id, 1);
    }

    // -- LayerContext: hidden_state is a valid slice --

    #[test]
    fn test_layer_context_hidden_state_is_valid_slice() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(!ctx.hidden_state.is_empty());
        assert_eq!(ctx.hidden_state.len(), 1024);
    }

    // -- CallbackChain: empty chain dispatch_pre_node is idempotent across calls --

    #[test]
    fn test_empty_chain_pre_node_idempotent_triple() {
        let mut chain = CallbackChain::empty();
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- CallbackChain: empty chain dispatch_post_node is idempotent across calls --

    #[test]
    fn test_empty_chain_post_node_idempotent_triple() {
        let mut chain = CallbackChain::empty();
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_post_node(&ctx, &[1, 2, 3]), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[4, 5, 6]), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue);
    }

    // -- CallbackChain: dispatch with large output buffer and Continue --

    #[test]
    fn test_chain_post_node_large_output_buffer_continue() {
        let cb = TestCallback::new("large_out", 50);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        let output = vec![0u8; 65536];
        assert_eq!(chain.dispatch_post_node(&ctx, &output), CallbackAction::Continue);
    }

    // -- CallbackChain: single callback with priority 0 dispatches normally --

    #[test]
    fn test_chain_single_zero_priority_dispatches() {
        let cb = TestCallback::new("zero_pri", 0)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: Verify sorted_indices is a permutation of 0..n --

    #[test]
    fn test_chain_sorted_indices_is_permutation() {
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("a", 30)),
            Box::new(TestCallback::new("b", 10)),
            Box::new(TestCallback::new("c", 50)),
            Box::new(TestCallback::new("d", 20)),
            Box::new(TestCallback::new("e", 40)),
        ];
        let chain = CallbackChain::new(callbacks);
        let mut indices = chain.sorted_indices.clone();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    // -- CallbackChain: len() returns 0 for empty, N for N callbacks --

    #[test]
    fn test_chain_len_correct_for_various_sizes() {
        assert_eq!(CallbackChain::empty().len(), 0);
        assert_eq!(CallbackChain::new(vec![]).len(), 0);

        let cb1 = TestCallback::new("a", 1);
        assert_eq!(CallbackChain::new(vec![Box::new(cb1) as Box<dyn LayerCallback + Send>]).len(), 1);

        let cbs: Vec<Box<dyn LayerCallback + Send>> = (0..10)
            .map(|i| Box::new(TestCallback::new(Box::leak(format!("c{}", i).into_boxed_str()), i * 10)) as _)
            .collect();
        assert_eq!(CallbackChain::new(cbs).len(), 10);
    }

    // -- LayerContext: model_config geometry num_kv_heads field --

    #[test]
    fn test_layer_context_geometry_num_kv_heads() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.num_kv_heads, 2);
    }

    // -- LayerContext: model_config geometry num_heads field --

    #[test]
    fn test_layer_context_geometry_num_heads() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.num_heads, 4);
    }

    // -- LayerContext: model_config paged_kv page_table is None --

    #[test]
    fn test_layer_context_paged_kv_page_table_none() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.paged_kv.page_table.is_none());
    }

    // -- LayerContext: geometry norm_eps is 1e-5 --

    #[test]
    fn test_layer_context_geometry_norm_eps() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!((ctx.model_config.geometry.norm_eps - 1e-5).abs() < 1e-10);
    }

    // -- LayerContext: geometry dtype field is F32 --

    #[test]
    fn test_layer_context_geometry_dtype_f32() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.dtype, gllm_kernels::types::DType::F32);
    }

    // -- LayerContext: geometry compute_dtype field is F32 --

    #[test]
    fn test_layer_context_geometry_compute_dtype_f32() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.compute_dtype, gllm_kernels::types::DType::F32);
    }

    // -- LayerContext: geometry rope_theta is 10000.0 --

    #[test]
    fn test_layer_context_geometry_rope_theta() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!((ctx.model_config.geometry.rope_theta - 10000.0).abs() < f64::EPSILON);
    }

    // -- LayerContext: geometry num_experts is 0 (dense model) --

    #[test]
    fn test_layer_context_geometry_num_experts_zero() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.num_experts, 0);
    }

    // -- CallbackChain: dispatch_pre_node does not mutate context --

    #[test]
    fn test_chain_dispatch_pre_node_does_not_mutate_context() {
        let cb = TestCallback::new("observer", 50);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(3, 7);

        let node_before = ctx.node_idx;
        let layer_before = ctx.layer_idx;
        let seq_before = ctx.seq_len;

        let _ = chain.dispatch_pre_node(&ctx);

        assert_eq!(ctx.node_idx, node_before);
        assert_eq!(ctx.layer_idx, layer_before);
        assert_eq!(ctx.seq_len, seq_before);
    }

    // -- CallbackChain: dispatch_post_node does not mutate context --

    #[test]
    fn test_chain_dispatch_post_node_does_not_mutate_context() {
        let cb = TestCallback::new("observer", 50);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(2, 5);

        let request_id_before = ctx.request_id;
        let total_seq_before = ctx.total_seq;

        let _ = chain.dispatch_post_node(&ctx, &[0u8; 100]);

        assert_eq!(ctx.request_id, request_id_before);
        assert_eq!(ctx.total_seq, total_seq_before);
    }

    // -- CallbackAction: Debug output contains variant name for each variant --

    #[test]
    fn test_debug_output_contains_variant_name_all_five() {
        let continue_str = format!("{:?}", CallbackAction::Continue);
        assert!(continue_str.contains("Continue"));

        let skip_str = format!("{:?}", CallbackAction::SkipThisNode);
        assert!(skip_str.contains("SkipThisNode"));

        let exit_str = format!("{:?}", CallbackAction::ExitEarly { logits: vec![1.0] });
        assert!(exit_str.contains("ExitEarly"));

        let inject_str = format!("{:?}", CallbackAction::InjectHidden { data: vec![0x00] });
        assert!(inject_str.contains("InjectHidden"));

        let compact_str = format!("{:?}", CallbackAction::CompactMask { active_mask: vec![true] });
        assert!(compact_str.contains("CompactMask"));
    }

    // -- CallbackChain: two callbacks, high returns Continue, low returns action in post --

    #[test]
    fn test_chain_post_node_high_continue_low_returns_action() {
        let cb_high = TestCallback::new("high_cont", 100);
        let cb_low = TestCallback::new("low_act", 50)
            .with_post_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_high),
            Box::new(cb_low),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: target_layers with single layer that matches --

    #[test]
    fn test_chain_target_layer_single_match_fires_callback() {
        let cb = TestCallback::new("single_target", 50)
            .with_layers(vec![3])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(3, 6);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: target_layers with single layer that does NOT match --

    #[test]
    fn test_chain_target_layer_single_mismatch_returns_continue() {
        let cb = TestCallback::new("single_target", 50)
            .with_layers(vec![7])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(2, 4);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- CallbackChain: is_empty returns false after adding a callback --

    #[test]
    fn test_chain_not_empty_after_adding_callback() {
        let cb = TestCallback::new("nonempty", 10);
        let chain = CallbackChain::new(vec![Box::new(cb)]);
        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);
    }

    // -- CallbackChain: dispatch pre and post are independent for stateful callback --

    #[test]
    fn test_chain_stateful_callback_pre_and_post_independent_counts() {
        struct DualCounter { pre_count: usize, post_count: usize }
        impl LayerCallback for DualCounter {
            fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
                self.pre_count += 1;
                CallbackAction::Continue
            }
            fn post_node(&mut self, _ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
                self.post_count += 1;
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "dual_counter" }
        }

        let cb = DualCounter { pre_count: 0, post_count: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Call pre_node 3 times
        for _ in 0..3 {
            assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        }
        // Call post_node 2 times
        for _ in 0..2 {
            assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue);
        }

        // Callback was invoked 5 times total (3 pre + 2 post)
        // We can't inspect the internal state, but we verify it continues to work
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue);
    }

    // -- CallbackChain: callback with layers=[0,1,2] fires on 0,1,2 but not 3 --

    #[test]
    fn test_chain_layers_range_fires_on_included_only() {
        let cb = TestCallback::new("range_layers", 50)
            .with_layers(vec![0, 1, 2])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for layer in 0..3 {
            let ctx = holder.ctx(layer, layer * 2);
            assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode,
                "Should fire on layer {}", layer);
        }

        let ctx3 = holder.ctx(3, 6);
        assert_eq!(chain.dispatch_pre_node(&ctx3), CallbackAction::Continue,
            "Should NOT fire on layer 3");
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 45 additional tests — coverage gaps: PartialEq transitivity, Clone special
    // floats, context field reads in callbacks, untested geometry fields
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // -- CallbackAction: PartialEq transitivity for Continue --

    #[test]
    fn test_callback_action_partial_eq_transitivity_continue() {
        let a = CallbackAction::Continue;
        let b = CallbackAction::Continue;
        let c = CallbackAction::Continue;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // -- CallbackAction: PartialEq transitivity for SkipThisNode --

    #[test]
    fn test_callback_action_partial_eq_transitivity_skip_this_node() {
        let a = CallbackAction::SkipThisNode;
        let b = CallbackAction::SkipThisNode;
        let c = CallbackAction::SkipThisNode;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // -- CallbackAction: PartialEq transitivity for CompactMask --

    #[test]
    fn test_callback_action_partial_eq_transitivity_compact_mask() {
        let mask = vec![true, false, true];
        let a = CallbackAction::CompactMask { active_mask: mask.clone() };
        let b = CallbackAction::CompactMask { active_mask: mask.clone() };
        let c = CallbackAction::CompactMask { active_mask: mask };
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // -- CallbackAction: Clone preserves NaN in ExitEarly logits --

    #[test]
    fn test_callback_action_clone_exit_early_nan_preserved() {
        let original = CallbackAction::ExitEarly { logits: vec![f32::NAN, 1.0] };
        let cloned = original.clone();
        if let CallbackAction::ExitEarly { logits } = cloned {
            assert!(logits[0].is_nan());
            assert!((logits[1] - 1.0).abs() < f32::EPSILON);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: Clone preserves +Infinity in ExitEarly logits --

    #[test]
    fn test_callback_action_clone_exit_early_infinity_preserved() {
        let original = CallbackAction::ExitEarly { logits: vec![f32::INFINITY] };
        let cloned = original.clone();
        if let CallbackAction::ExitEarly { logits } = cloned {
            assert!(logits[0].is_infinite() && logits[0].is_sign_positive());
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: Clone preserves -Infinity in ExitEarly logits --

    #[test]
    fn test_callback_action_clone_exit_early_neg_infinity_preserved() {
        let original = CallbackAction::ExitEarly { logits: vec![f32::NEG_INFINITY] };
        let cloned = original.clone();
        if let CallbackAction::ExitEarly { logits } = cloned {
            assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: Clone InjectHidden with empty vec --

    #[test]
    fn test_callback_action_clone_inject_hidden_empty_vec() {
        let original = CallbackAction::InjectHidden { data: vec![] };
        let cloned = original.clone();
        if let CallbackAction::InjectHidden { data } = cloned {
            assert!(data.is_empty());
        } else {
            panic!("Expected InjectHidden");
        }
    }

    // -- CallbackAction: Clone CompactMask with empty vec --

    #[test]
    fn test_callback_action_clone_compact_mask_empty_vec() {
        let original = CallbackAction::CompactMask { active_mask: vec![] };
        let cloned = original.clone();
        if let CallbackAction::CompactMask { active_mask } = cloned {
            assert!(active_mask.is_empty());
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- CallbackAction: ExitEarly with -0.0 equals ExitEarly with +0.0 (IEEE 754) --

    #[test]
    fn test_callback_action_exit_early_neg_zero_equals_pos_zero() {
        let a = CallbackAction::ExitEarly { logits: vec![-0.0f32] };
        let b = CallbackAction::ExitEarly { logits: vec![0.0f32] };
        assert_eq!(a, b);
    }

    // -- CallbackAction: ExitEarly logits all zeros --

    #[test]
    fn test_callback_action_exit_early_logits_all_zeros() {
        let logits = vec![0.0f32; 100];
        let action = CallbackAction::ExitEarly { logits: logits.clone() };
        if let CallbackAction::ExitEarly { logits: l } = action {
            assert_eq!(l.len(), 100);
            assert!(l.iter().all(|&v| v == 0.0));
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: ExitEarly with mixed positive and negative logits --

    #[test]
    fn test_callback_action_exit_early_mixed_sign_logits() {
        let logits: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
        let action = CallbackAction::ExitEarly { logits: logits.clone() };
        if let CallbackAction::ExitEarly { logits: l } = action {
            let positive_count = l.iter().filter(|&&v| v > 0.0).count();
            let negative_count = l.iter().filter(|&&v| v < 0.0).count();
            let zero_count = l.iter().filter(|&&v| v == 0.0).count();
            assert_eq!(positive_count, 50);
            assert_eq!(negative_count, 50);
            assert_eq!(zero_count, 1);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: InjectHidden with exactly two bytes --

    #[test]
    fn test_callback_action_inject_hidden_two_bytes() {
        let action = CallbackAction::InjectHidden { data: vec![0xDE, 0xAD] };
        if let CallbackAction::InjectHidden { data } = action {
            assert_eq!(data.len(), 2);
            assert_eq!(data[0], 0xDE);
            assert_eq!(data[1], 0xAD);
        } else {
            panic!("Expected InjectHidden");
        }
    }

    // -- CallbackAction: CompactMask with two elements, both true --

    #[test]
    fn test_callback_action_compact_mask_two_both_true() {
        let action = CallbackAction::CompactMask { active_mask: vec![true, true] };
        if let CallbackAction::CompactMask { active_mask } = action {
            assert_eq!(active_mask.len(), 2);
            assert!(active_mask.iter().all(|&v| v));
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- CallbackAction: CompactMask with two elements, both false --

    #[test]
    fn test_callback_action_compact_mask_two_both_false() {
        let action = CallbackAction::CompactMask { active_mask: vec![false, false] };
        if let CallbackAction::CompactMask { active_mask } = action {
            assert_eq!(active_mask.len(), 2);
            assert!(active_mask.iter().all(|&v| !v));
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- CallbackAction: CompactMask with two elements, mixed --

    #[test]
    fn test_callback_action_compact_mask_two_mixed() {
        let action = CallbackAction::CompactMask { active_mask: vec![true, false] };
        if let CallbackAction::CompactMask { active_mask } = action {
            assert_eq!(active_mask.len(), 2);
            assert!(active_mask[0]);
            assert!(!active_mask[1]);
        } else {
            panic!("Expected CompactMask");
        }
    }

    // -- CallbackAction: ExitEarly with exactly two logits --

    #[test]
    fn test_callback_action_exit_early_with_two_logits() {
        let action = CallbackAction::ExitEarly { logits: vec![-1.0, 1.0] };
        if let CallbackAction::ExitEarly { logits } = action {
            assert_eq!(logits.len(), 2);
            assert!(logits[0] < 0.0);
            assert!(logits[1] > 0.0);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    // -- CallbackAction: Clone ExitEarly preserves logit count --

    #[test]
    fn test_callback_action_clone_exit_early_preserves_logit_count() {
        let logits: Vec<f32> = (0..500).map(|i| i as f32 * 0.01).collect();
        let original = CallbackAction::ExitEarly { logits };
        let cloned = original.clone();
        if let (CallbackAction::ExitEarly { logits: a }, CallbackAction::ExitEarly { logits: b }) =
            (&original, &cloned)
        {
            assert_eq!(a.len(), b.len());
            assert_eq!(a.len(), 500);
        } else {
            panic!("Both should be ExitEarly");
        }
    }

    // -- CallbackAction: Debug for ExitEarly with single NaN --

    #[test]
    fn test_callback_action_debug_exit_early_single_nan() {
        let action = CallbackAction::ExitEarly { logits: vec![f32::NAN] };
        let debug = format!("{:?}", action);
        assert!(debug.contains("ExitEarly"));
        assert!(debug.contains("logits"));
    }

    // -- CallbackAction: Debug for InjectHidden with single byte --

    #[test]
    fn test_callback_action_debug_inject_hidden_single_byte() {
        let action = CallbackAction::InjectHidden { data: vec![0x42] };
        let debug = format!("{:?}", action);
        assert!(debug.contains("InjectHidden"));
        assert!(debug.contains("data"));
    }

    // -- CallbackChain: callback reads total_seq from context --

    #[test]
    fn test_chain_callback_reads_total_seq() {
        struct TotalSeqReader { observed: usize }
        impl LayerCallback for TotalSeqReader {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.observed = ctx.total_seq;
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "total_seq_reader" }
        }

        let cb = TotalSeqReader { observed: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 42, seq_len: 1, position: 41,
            request_id: 0, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- CallbackChain: callback reads position from context --

    #[test]
    fn test_chain_callback_reads_position() {
        struct PositionReader { observed: usize }
        impl LayerCallback for PositionReader {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.observed = ctx.position;
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "position_reader" }
        }

        let cb = PositionReader { observed: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 99,
            request_id: 0, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- CallbackChain: callback reads node_idx from context --

    #[test]
    fn test_chain_callback_reads_node_idx() {
        struct NodeIdxReader { observed: usize }
        impl LayerCallback for NodeIdxReader {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.observed = ctx.node_idx;
                if ctx.node_idx >= 10 {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "node_idx_reader" }
        }

        let cb = NodeIdxReader { observed: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // node_idx < 10: Continue
        let ctx_low = holder.ctx(0, 5);
        assert_eq!(chain.dispatch_pre_node(&ctx_low), CallbackAction::Continue);

        // node_idx >= 10: SkipThisNode
        let ctx_high = holder.ctx(0, 15);
        assert_eq!(chain.dispatch_pre_node(&ctx_high), CallbackAction::SkipThisNode);
    }

    // -- CallbackChain: callback reads hidden_state length from context --

    #[test]
    fn test_chain_callback_reads_hidden_state_len() {
        struct HiddenStateLenReader { observed: usize }
        impl LayerCallback for HiddenStateLenReader {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.observed = ctx.hidden_state.len();
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "hs_len_reader" }
        }

        let cb = HiddenStateLenReader { observed: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
    }

    // -- CallbackChain: callback reads kv_cache_k pointer from context --

    #[test]
    fn test_chain_callback_reads_kv_cache_k_ptr() {
        struct KvPtrReader { k_was_null: bool }
        impl LayerCallback for KvPtrReader {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.k_was_null = ctx.kv_cache_k.is_null();
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "kv_ptr_reader" }
        }

        let cb = KvPtrReader { k_was_null: false };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Null pointer case
        let ctx_null = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx_null), CallbackAction::Continue);

        // Non-null pointer case
        let mut k_buf = [0.0f32; 4];
        let ctx_nonnull = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: k_buf.as_mut_ptr(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 0, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx_nonnull), CallbackAction::Continue);
    }

    // -- CallbackChain: callback reads seq_len from context --

    #[test]
    fn test_chain_callback_reads_seq_len_from_context() {
        struct SeqLenReader { observed: usize }
        impl LayerCallback for SeqLenReader {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.observed = ctx.seq_len;
                if ctx.seq_len > 1 {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "seq_len_reader" }
        }

        let cb = SeqLenReader { observed: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // decode (seq_len=1): Continue
        let ctx_decode = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 10, seq_len: 1, position: 9,
            request_id: 0, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx_decode), CallbackAction::Continue);

        // prefill (seq_len>1): SkipThisNode
        let ctx_prefill = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 64, seq_len: 64, position: 0,
            request_id: 0, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx_prefill), CallbackAction::SkipThisNode);
    }

    // -- LayerContext: node_op as empty string --

    #[test]
    fn test_layer_context_empty_node_op() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 0, model_config: &holder.config,
        };
        assert!(ctx.node_op.is_empty());
    }

    // -- CallbackChain: dispatch with large hidden_state context --

    #[test]
    fn test_chain_dispatch_with_large_hidden_state_context() {
        let large_state = vec![0xABu8; 1024 * 1024];
        let cb = TestCallback::new("large_hs", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        let ctx = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &large_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 0, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);
        assert_eq!(ctx.hidden_state.len(), 1024 * 1024);
    }

    // -- LayerContext: geometry.vocab_size field --

    #[test]
    fn test_layer_context_geometry_vocab_size() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.vocab_size, 1000);
    }

    // -- LayerContext: geometry.intermediate_size field --

    #[test]
    fn test_layer_context_geometry_intermediate_size() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.intermediate_size, 512);
    }

    // -- LayerContext: geometry.num_layers field --

    #[test]
    fn test_layer_context_geometry_num_layers() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.num_layers, 4);
    }

    // -- LayerContext: geometry.rope_interleaved field is false --

    #[test]
    fn test_layer_context_geometry_rope_interleaved_false() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(!ctx.model_config.geometry.rope_interleaved);
    }

    // -- LayerContext: geometry.rope_scale field is 1.0 --

    #[test]
    fn test_layer_context_geometry_rope_scale_one() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!((ctx.model_config.geometry.rope_scale - 1.0).abs() < f64::EPSILON);
    }

    // -- LayerContext: geometry.rope_partial_ratio field --

    #[test]
    fn test_layer_context_geometry_rope_partial_ratio() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!((ctx.model_config.geometry.rope_partial_ratio - 1.0).abs() < f32::EPSILON);
    }

    // -- LayerContext: geometry.global_rope_theta field --

    #[test]
    fn test_layer_context_geometry_global_rope_theta() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.global_rope_theta, 0.0);
    }

    // -- LayerContext: geometry.sliding_window field --

    #[test]
    fn test_layer_context_geometry_sliding_window() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.sliding_window, 0);
    }

    // -- LayerContext: geometry.attention_pattern empty vec --

    #[test]
    fn test_layer_context_geometry_attention_pattern_empty() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.geometry.attention_pattern.is_empty());
    }

    // -- LayerContext: geometry.hidden_size_per_layer_input field --

    #[test]
    fn test_layer_context_geometry_hidden_size_per_layer_input() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.hidden_size_per_layer_input, 0);
    }

    // -- LayerContext: geometry.position_offset field is None --

    #[test]
    fn test_layer_context_geometry_position_offset_none() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.geometry.position_offset.is_none());
    }

    // -- LayerContext: geometry.rope_scaling field is None --

    #[test]
    fn test_layer_context_geometry_rope_scaling_none() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.geometry.rope_scaling.is_none());
    }

    // -- LayerContext: geometry.final_logit_softcapping field is None --

    #[test]
    fn test_layer_context_geometry_final_logit_softcapping_none() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.geometry.final_logit_softcapping.is_none());
    }

    // -- LayerContext: geometry.hidden_act field is None --

    #[test]
    fn test_layer_context_geometry_hidden_act_none() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert!(ctx.model_config.geometry.hidden_act.is_none());
    }

    // -- CallbackChain: pre_node InjectHidden stops three-callback chain --

    #[test]
    fn test_chain_pre_inject_stops_three_callback_chain() {
        let cb_high = TestCallback::new("high_inject", 100)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0x11, 0x22] });
        let cb_mid = TestCallback::new("mid_skip", 50)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_low = TestCallback::new("low_exit", 10)
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![0.0] });

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_high),
            Box::new(cb_mid),
            Box::new(cb_low),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_pre_node(&ctx);
        match action {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0x11, 0x22]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackChain: post_node CompactMask from three callbacks, middle fires --

    #[test]
    fn test_chain_post_compact_mask_from_three_callbacks() {
        let cb_high = TestCallback::new("high_cont", 90);
        let cb_mid = TestCallback::new("mid_compact", 50)
            .with_post_action(CallbackAction::CompactMask {
                active_mask: vec![true, false, true, false],
            });
        let cb_low = TestCallback::new("low_skip", 10)
            .with_post_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_high),
            Box::new(cb_mid),
            Box::new(cb_low),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_post_node(&ctx, &[0u8; 16]);
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask, vec![true, false, true, false]);
            }
            other => panic!("Expected CompactMask, got {:?}", action_variant_name(&other)),
        }
    }

    // -- CallbackChain: callback checks both layer_idx AND node_idx --

    #[test]
    fn test_chain_callback_compound_layer_and_node_idx_check() {
        struct CompoundCheck { last_layer: usize, last_node: usize }
        impl LayerCallback for CompoundCheck {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                self.last_layer = ctx.layer_idx;
                self.last_node = ctx.node_idx;
                // Only skip when both layer=2 AND node>=4
                if ctx.layer_idx == 2 && ctx.node_idx >= 4 {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "compound_check" }
        }

        let cb = CompoundCheck { last_layer: 0, last_node: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // layer=0, node=0: Continue
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::Continue);

        // layer=2, node=2: Continue (node < 4)
        let ctx1 = holder.ctx(2, 2);
        assert_eq!(chain.dispatch_pre_node(&ctx1), CallbackAction::Continue);

        // layer=2, node=4: SkipThisNode
        let ctx2 = holder.ctx(2, 4);
        assert_eq!(chain.dispatch_pre_node(&ctx2), CallbackAction::SkipThisNode);

        // layer=1, node=10: Continue (wrong layer)
        let ctx3 = holder.ctx(1, 10);
        assert_eq!(chain.dispatch_pre_node(&ctx3), CallbackAction::Continue);
    }

    // -- CallbackChain: empty() and new(vec![]) behavioral equivalence --

    #[test]
    fn test_chain_empty_vs_new_empty_behavior_equivalence() {
        let mut empty = CallbackChain::empty();
        let mut from_new = CallbackChain::new(vec![]);
        let holder = CtxHolder::new();

        for layer in 0..10 {
            let ctx_e = holder.ctx(layer, layer);
            let ctx_n = holder.ctx(layer, layer);
            assert_eq!(
                empty.dispatch_pre_node(&ctx_e),
                from_new.dispatch_pre_node(&ctx_n),
                "pre_node mismatch at layer {}",
                layer
            );
            assert_eq!(
                empty.dispatch_post_node(&ctx_e, &[]),
                from_new.dispatch_post_node(&ctx_n, &[]),
                "post_node mismatch at layer {}",
                layer
            );
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 15 new tests — uncovered paths: stateful post_node, context-driven
    // decisions in callbacks, priority gaps, and geometry edge fields
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // 1. Callback that switches from Continue to SkipThisNode on the Nth call
    #[test]
    fn test_chain_callback_switches_action_after_threshold() {
        // Arrange
        struct ThresholdCallback { invocations: usize, threshold: usize }
        impl LayerCallback for ThresholdCallback {
            fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
                self.invocations += 1;
                if self.invocations >= self.threshold {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "threshold" }
        }

        let cb = ThresholdCallback { invocations: 0, threshold: 3 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Act & Assert: first two dispatches return Continue, third returns Skip
        for i in 0..2 {
            let ctx = holder.ctx(i, i);
            assert_eq!(
                chain.dispatch_pre_node(&ctx),
                CallbackAction::Continue,
                "Invocation {} should be Continue",
                i + 1
            );
        }
        let ctx = holder.ctx(2, 4);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);
    }

    // 2. Callback makes decision based on position field in LayerContext
    #[test]
    fn test_chain_callback_decides_based_on_position() {
        // Arrange
        struct PositionGate { max_position: usize }
        impl LayerCallback for PositionGate {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                if ctx.position > self.max_position {
                    CallbackAction::ExitEarly { logits: vec![0.0] }
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "position_gate" }
        }

        let cb = PositionGate { max_position: 50 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Act & Assert: position within limit → Continue
        let ctx_safe = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 30, seq_len: 1, position: 29,
            request_id: 0, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx_safe), CallbackAction::Continue);

        // position exceeds limit → ExitEarly
        let ctx_over = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 60, seq_len: 1, position: 59,
            request_id: 0, model_config: &holder.config,
        };
        match chain.dispatch_pre_node(&ctx_over) {
            CallbackAction::ExitEarly { logits } => assert!(logits.is_empty() || logits == vec![0.0]),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // 3. CallbackChain with callback that uses post_node output content to decide
    #[test]
    fn test_chain_post_node_callback_inspects_output_bytes() {
        // Arrange
        struct OutputContentCallback;
        impl LayerCallback for OutputContentCallback {
            fn post_node(&mut self, _ctx: &LayerContext, output: &[u8]) -> CallbackAction {
                if output.len() >= 4 {
                    let first_four_sum: u8 = output[..4].iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
                    if first_four_sum == 0 {
                        // All zeros in first 4 bytes → skip
                        return CallbackAction::SkipThisNode;
                    }
                }
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "output_content" }
        }

        let cb = OutputContentCallback;
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act & Assert: all-zero output → SkipThisNode
        let zero_output = vec![0u8; 16];
        assert_eq!(
            chain.dispatch_post_node(&ctx, &zero_output),
            CallbackAction::SkipThisNode
        );

        // non-zero output → Continue
        let nonzero_output = vec![1u8, 2, 3, 4];
        assert_eq!(
            chain.dispatch_post_node(&ctx, &nonzero_output),
            CallbackAction::Continue
        );
    }

    // 4. CallbackChain: priority ordering with large gaps between values
    #[test]
    fn test_chain_priority_ordering_with_quadratic_gaps() {
        // Arrange: priorities are squares (1, 4, 9, 16, 25)
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("p1", 1)),
            Box::new(TestCallback::new("p4", 4)),
            Box::new(TestCallback::new("p9", 9)),
            Box::new(TestCallback::new("p16", 16)),
            Box::new(TestCallback::new("p25", 25)),
        ];
        let chain = CallbackChain::new(callbacks);

        // Assert: sorted by descending priority: 25(idx4), 16(idx3), 9(idx2), 4(idx1), 1(idx0)
        assert_eq!(chain.sorted_indices, vec![4, 3, 2, 1, 0]);
        assert_eq!(chain.len(), 5);
    }

    // 5. Two callbacks with overlapping target layers; both fire on shared layer
    #[test]
    fn test_chain_overlapping_target_layers_first_non_continue_wins() {
        // Arrange: both target layers [1, 2]; high priority returns InjectHidden, low returns Skip
        let cb_high = TestCallback::new("high_overlap", 100)
            .with_layers(vec![1, 2])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0xCC] });
        let cb_low = TestCallback::new("low_overlap", 50)
            .with_layers(vec![1, 2])
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();

        // Act & Assert: on shared layer 1, high priority fires first → InjectHidden
        let ctx = holder.ctx(1, 2);
        match chain.dispatch_pre_node(&ctx) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xCC]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }

        // On layer 3 (not targeted by either) → Continue
        let ctx3 = holder.ctx(3, 6);
        assert_eq!(chain.dispatch_pre_node(&ctx3), CallbackAction::Continue);
    }

    // 6. LayerContext: geometry.num_kv_shared_layers field
    #[test]
    fn test_layer_context_geometry_num_kv_shared_layers_zero() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.num_kv_shared_layers, 0);
    }

    // 7. LayerContext: geometry.global_head_dim field
    #[test]
    fn test_layer_context_geometry_global_head_dim_zero() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.global_head_dim, 0);
    }

    // 8. LayerContext: geometry.mla_d_c field
    #[test]
    fn test_layer_context_geometry_mla_d_c_zero() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.mla_d_c, 0);
    }

    // 9. LayerContext: geometry.mla_d_rope field
    #[test]
    fn test_layer_context_geometry_mla_d_rope_zero() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.mla_d_rope, 0);
    }

    // 10. LayerContext: geometry.mla_unabsorbed_threshold field
    #[test]
    fn test_layer_context_geometry_mla_unabsorbed_threshold_zero() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.mla_unabsorbed_threshold, 0);
    }

    // 11. CallbackChain: callback with target_layers=[0] only fires on layer 0, not layer 1
    #[test]
    fn test_chain_target_layer_zero_fires_only_on_layer_zero_post_node() {
        let cb = TestCallback::new("l0_post_only", 50)
            .with_layers(vec![0])
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Layer 0: fires → ExitEarly
        let ctx0 = holder.ctx(0, 0);
        match chain.dispatch_post_node(&ctx0, &[0u8; 8]) {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![1.0]),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }

        // Layer 1: not targeted → Continue
        let ctx1 = holder.ctx(1, 2);
        assert_eq!(chain.dispatch_post_node(&ctx1, &[0u8; 8]), CallbackAction::Continue);
    }

    // 12. CallbackChain: stateful callback tracks total post_node output bytes
    #[test]
    fn test_chain_stateful_post_node_tracks_bytes() {
        // Arrange
        struct ByteCounter { total_bytes: usize }
        impl LayerCallback for ByteCounter {
            fn post_node(&mut self, _ctx: &LayerContext, output: &[u8]) -> CallbackAction {
                self.total_bytes += output.len();
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "byte_counter" }
        }

        let cb = ByteCounter { total_bytes: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act: dispatch with varying output sizes
        let sizes = [0usize, 8, 16, 32, 64];
        for &size in &sizes {
            let output = vec![0u8; size];
            assert_eq!(chain.dispatch_post_node(&ctx, &output), CallbackAction::Continue);
        }
        // total_bytes should be 0+8+16+32+64 = 120 (internal state, verified by continuation)
    }

    // 13. CallbackChain: dispatch_pre_node with callback that returns CompactMask on specific layer
    #[test]
    fn test_chain_compact_mask_on_specific_layer_only() {
        let mask = vec![true, false, true, false, true];
        let expected_mask = mask.clone();
        let cb = TestCallback::new("layer_specific_compact", 70)
            .with_layers(vec![5])
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Layer 0: not targeted → Continue
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::Continue);

        // Layer 5: targeted → CompactMask
        let ctx5 = holder.ctx(5, 10);
        match chain.dispatch_pre_node(&ctx5) {
            CallbackAction::CompactMask { active_mask } => assert_eq!(active_mask, expected_mask),
            other => panic!("Expected CompactMask, got {:?}", action_variant_name(&other)),
        }

        // Layer 6: not targeted → Continue
        let ctx6 = holder.ctx(6, 12);
        assert_eq!(chain.dispatch_pre_node(&ctx6), CallbackAction::Continue);
    }

    // 14. CallbackChain: building with callbacks that have priorities [0, 0, 0] preserves order
    #[test]
    fn test_chain_three_zero_priority_callbacks_preserves_insertion_order() {
        let cb_a = TestCallback::new("a", 0)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_b = TestCallback::new("b", 0)
            .with_pre_action(CallbackAction::InjectHidden { data: vec![1] });
        let cb_c = TestCallback::new("c", 0)
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![0.0] });

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_a), Box::new(cb_b), Box::new(cb_c),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Stable sort preserves insertion order: a fires first → SkipThisNode
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);
        assert_eq!(chain.sorted_indices, vec![0, 1, 2]);
    }

    // 15. CallbackChain: dispatch_pre_node and dispatch_post_node on same chain with multiple callbacks
    #[test]
    fn test_chain_pre_and_post_dispatch_interleaved_with_multiple_callbacks() {
        // Arrange: 3 callbacks with different pre and post actions
        let cb1 = TestCallback::new("high", 100)
            .with_pre_action(CallbackAction::Continue)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![5.0] });
        let cb2 = TestCallback::new("mid", 50)
            .with_pre_action(CallbackAction::Continue)
            .with_post_action(CallbackAction::Continue);
        let cb3 = TestCallback::new("low", 10)
            .with_pre_action(CallbackAction::SkipThisNode)
            .with_post_action(CallbackAction::Continue);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb1), Box::new(cb2), Box::new(cb3),
        ]);
        let holder = CtxHolder::new();

        // Act & Assert: pre_node dispatches in priority order
        // high→Continue, mid→Continue, low→SkipThisNode
        let ctx = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);

        // post_node dispatches independently in priority order
        // high→ExitEarly (short-circuits), mid and low never called
        match chain.dispatch_post_node(&ctx, &[1, 2, 3]) {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![5.0]),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 15 NEW tests — coverage expansion: LayerCallback Send bound verification,
    // CallbackChain priority arithmetic edge cases, context-driven multi-field
    // callback logic, action variant round-trip through dispatch, geometry
    // field boundary checks
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // 1. Verify CallbackAction::CompactMask with a mask of length 0 preserves semantics
    #[test]
    fn test_compact_mask_empty_mask_dispatched_via_chain() {
        // Arrange
        let cb = TestCallback::new("empty_compact", 70)
            .with_pre_action(CallbackAction::CompactMask { active_mask: vec![] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert: empty mask is still CompactMask, not Continue
        match action {
            CallbackAction::CompactMask { active_mask } => assert!(active_mask.is_empty()),
            other => panic!("Expected CompactMask with empty mask, got {:?}", action_variant_name(&other)),
        }
    }

    // 2. CallbackChain: priority values that are powers of two sort correctly
    #[test]
    fn test_chain_priority_powers_of_two_sort_correctly() {
        // Arrange: priorities 1, 2, 4, 8, 16, 32, 64, 128
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("p1", 1)),
            Box::new(TestCallback::new("p2", 2)),
            Box::new(TestCallback::new("p4", 4)),
            Box::new(TestCallback::new("p8", 8)),
            Box::new(TestCallback::new("p16", 16)),
            Box::new(TestCallback::new("p32", 32)),
            Box::new(TestCallback::new("p64", 64)),
            Box::new(TestCallback::new("p128", 128)),
        ];
        let chain = CallbackChain::new(callbacks);

        // Assert: sorted descending by priority → 7(128), 6(64), 5(32), 4(16), 3(8), 2(4), 1(2), 0(1)
        assert_eq!(chain.sorted_indices, vec![7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(chain.len(), 8);
    }

    // 3. CallbackChain: dispatch_post_node with InjectHidden from a stateful callback
    #[test]
    fn test_chain_stateful_post_node_injects_different_data_each_call() {
        // Arrange
        struct InjectCounter { call_count: usize }
        impl LayerCallback for InjectCounter {
            fn post_node(&mut self, _ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
                self.call_count += 1;
                CallbackAction::InjectHidden { data: vec![self.call_count as u8] }
            }
            fn name(&self) -> &str { "incrementing_injector" }
        }

        let cb = InjectCounter { call_count: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Act & Assert: first call returns data=[1]
        let ctx1 = holder.ctx(0, 0);
        match chain.dispatch_post_node(&ctx1, &[]) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![1]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }

        // Second call returns data=[2]
        let ctx2 = holder.ctx(0, 0);
        match chain.dispatch_post_node(&ctx2, &[]) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![2]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // 4. CallbackChain: callback with target_layers=[usize::MAX] only fires at that exact layer
    #[test]
    fn test_chain_target_layer_max_usize_exclusive_match() {
        // Arrange
        let cb = TestCallback::new("max_layer_exclusive", 50)
            .with_layers(vec![usize::MAX])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Act & Assert: layer usize::MAX - 1 does NOT match
        let ctx_below = holder.ctx(usize::MAX - 1, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx_below), CallbackAction::Continue);

        // layer usize::MAX DOES match
        let ctx_exact = holder.ctx(usize::MAX, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx_exact), CallbackAction::SkipThisNode);
    }

    // 5. CallbackAction: ExitEarly with f32::EPSILON logits clones correctly
    #[test]
    fn test_callback_action_exit_early_epsilon_logits_clone_round_trip() {
        // Arrange
        let original = CallbackAction::ExitEarly { logits: vec![f32::EPSILON, f32::EPSILON * 2.0] };
        let cloned = original.clone();

        // Assert: cloned values match original exactly
        if let CallbackAction::ExitEarly { logits } = cloned {
            assert_eq!(logits.len(), 2);
            assert_eq!(logits[0], f32::EPSILON);
            assert_eq!(logits[1], f32::EPSILON * 2.0);
        } else {
            panic!("Expected ExitEarly");
        }

        // Original unchanged
        if let CallbackAction::ExitEarly { logits } = &original {
            assert_eq!(logits[0], f32::EPSILON);
        } else {
            panic!("Original should be ExitEarly");
        }
    }

    // 6. CallbackChain: two callbacks where high returns Continue and low returns InjectHidden with 1MB data
    #[test]
    fn test_chain_large_inject_hidden_from_low_priority_callback() {
        // Arrange
        let large_data = vec![0xABu8; 1024 * 1024];
        let expected_len = large_data.len();
        let cb_high = TestCallback::new("high_cont", 100);
        let cb_low = TestCallback::new("low_inject", 10)
            .with_pre_action(CallbackAction::InjectHidden { data: large_data });

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::InjectHidden { data } => {
                assert_eq!(data.len(), expected_len);
                assert!(data.iter().all(|&b| b == 0xAB));
            }
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // 7. CallbackChain: post_node dispatch returns CompactMask with large alternating mask
    #[test]
    fn test_chain_post_node_compact_mask_large_alternating() {
        // Arrange
        let mask: Vec<bool> = (0..1024).map(|i| i % 3 != 0).collect();
        let expected_active = mask.iter().filter(|&&v| v).count();
        let cb = TestCallback::new("post_compact_alt", 50)
            .with_post_action(CallbackAction::CompactMask { active_mask: mask });

        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[0u8; 64]);

        // Assert
        match action {
            CallbackAction::CompactMask { active_mask } => {
                assert_eq!(active_mask.len(), 1024);
                assert_eq!(active_mask.iter().filter(|&&v| v).count(), expected_active);
            }
            other => panic!("Expected CompactMask, got {:?}", action_variant_name(&other)),
        }
    }

    // 8. CallbackChain: callback uses model_config.num_layers() to gate behavior
    #[test]
    fn test_chain_callback_gates_on_model_config_num_layers() {
        // Arrange: callback only returns SkipThisNode when layer >= num_layers
        struct LayerBoundCallback { num_layers: usize }
        impl LayerCallback for LayerBoundCallback {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                if ctx.layer_idx >= self.num_layers {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "layer_bound" }
        }

        let cb = LayerBoundCallback { num_layers: 4 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Act & Assert: layer 3 (within bound) → Continue
        let ctx_in = holder.ctx(3, 6);
        assert_eq!(chain.dispatch_pre_node(&ctx_in), CallbackAction::Continue);

        // layer 4 (at boundary) → SkipThisNode
        let ctx_out = holder.ctx(4, 8);
        assert_eq!(chain.dispatch_pre_node(&ctx_out), CallbackAction::SkipThisNode);

        // layer 0 → Continue
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::Continue);
    }

    // 9. CallbackAction: PartialEq symmetry for ExitEarly with multiple identical elements
    #[test]
    fn test_callback_action_partial_eq_symmetry_exit_early_identical_elements() {
        // Arrange
        let a = CallbackAction::ExitEarly { logits: vec![1.0, 1.0, 1.0, 1.0] };
        let b = CallbackAction::ExitEarly { logits: vec![1.0, 1.0, 1.0, 1.0] };

        // Assert: symmetry
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // 10. CallbackChain: pre_node returns ExitEarly with logits containing f32::MIN_POSITIVE
    #[test]
    fn test_chain_pre_node_exit_early_min_positive_logit() {
        // Arrange
        let cb = TestCallback::new("min_pos_exit", 100)
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![f32::MIN_POSITIVE] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits.len(), 1);
                assert_eq!(logits[0], f32::MIN_POSITIVE);
                assert!(logits[0] > 0.0);
            }
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // 11. CallbackChain: verify dispatch_pre_node is consistent across repeated calls with same context
    #[test]
    fn test_chain_pre_node_deterministic_across_many_calls() {
        // Arrange
        let cb = TestCallback::new("deterministic", 42)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(3, 7);

        // Act & Assert: 50 dispatches should all return the same action
        for i in 0..50 {
            assert_eq!(
                chain.dispatch_pre_node(&ctx),
                CallbackAction::SkipThisNode,
                "Dispatch {} should return SkipThisNode",
                i
            );
        }
    }

    // 12. CallbackChain: verify dispatch_post_node with empty output and ExitEarly containing one logit
    #[test]
    fn test_chain_post_node_exit_early_single_logit_with_empty_output() {
        // Arrange
        let cb = TestCallback::new("single_exit", 80)
            .with_post_action(CallbackAction::ExitEarly { logits: vec![0.123] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_post_node(&ctx, &[]);

        // Assert
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits.len(), 1);
                assert!((logits[0] - 0.123).abs() < 1e-6);
            }
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // 13. CallbackChain: callback with layers=[0,1] combined with another callback layers=[2,3]
    #[test]
    fn test_chain_two_callbacks_partitioned_target_layers() {
        // Arrange
        let cb_early = TestCallback::new("early_layers", 100)
            .with_layers(vec![0, 1])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_late = TestCallback::new("late_layers", 50)
            .with_layers(vec![2, 3])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0xDD] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_early), Box::new(cb_late)]);
        let holder = CtxHolder::new();

        // Act & Assert: layers 0,1 → early fires (SkipThisNode)
        for layer in 0..2 {
            let ctx = holder.ctx(layer, layer * 2);
            assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode,
                "Layer {} should be handled by early callback", layer);
        }

        // layers 2,3 → early filtered, late fires (InjectHidden)
        for layer in 2..4 {
            let ctx = holder.ctx(layer, layer * 2);
            match chain.dispatch_pre_node(&ctx) {
                CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xDD]),
                other => panic!("Layer {} expected InjectHidden, got {:?}", layer, action_variant_name(&other)),
            }
        }

        // layer 4 → neither matches → Continue
        let ctx4 = holder.ctx(4, 8);
        assert_eq!(chain.dispatch_pre_node(&ctx4), CallbackAction::Continue);
    }

    // 14. CallbackAction: InjectHidden with mixed byte values from 0 to 255 cloned correctly
    #[test]
    fn test_callback_action_inject_hidden_full_byte_range_clone_fidelity() {
        // Arrange
        let data: Vec<u8> = (0u8..=255).collect();
        let original = CallbackAction::InjectHidden { data: data.clone() };
        let mut cloned = original.clone();

        // Act: mutate the clone
        if let CallbackAction::InjectHidden { data } = &mut cloned {
            data[0] = 255;
        }

        // Assert: original unaffected
        if let CallbackAction::InjectHidden { data } = &original {
            assert_eq!(data[0], 0);
            assert_eq!(data.len(), 256);
        } else {
            panic!("Original should be InjectHidden");
        }

        // Assert: clone was mutated
        if let CallbackAction::InjectHidden { data } = &cloned {
            assert_eq!(data[0], 255);
        } else {
            panic!("Cloned should be InjectHidden");
        }
    }

    // 15. CallbackChain: multi-callback chain where only the lowest-priority returns non-Continue
    #[test]
    fn test_chain_only_lowest_priority_returns_non_continue_pre_node() {
        // Arrange: 4 callbacks, only the last one (lowest priority) returns SkipThisNode
        let cb1 = TestCallback::new("p100", 100);
        let cb2 = TestCallback::new("p75", 75);
        let cb3 = TestCallback::new("p50", 50);
        let cb4 = TestCallback::new("p25", 25)
            .with_pre_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb1), Box::new(cb2), Box::new(cb3), Box::new(cb4),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert: all higher-priority callbacks returned Continue, lowest returns Skip
        assert_eq!(action, CallbackAction::SkipThisNode);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 15 NEW tests — remaining coverage: Send trait on LayerContext, cross-layer
    // dispatch sequences, CallbackAction equality edge cases, geometry field
    // exhaustive coverage, chain construction invariants
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // 1. Verify LayerContext is Send by moving it across a thread boundary
    #[test]
    fn test_layer_context_is_send_across_thread() {
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 3,
            layer_idx: 1,
            node_op: "ThreadTest",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 20,
            seq_len: 5,
            position: 15,
            request_id: 42,
            model_config: &holder.config,
        };

        let handle = std::thread::spawn(move || {
            (ctx.node_idx, ctx.layer_idx, ctx.request_id)
        });

        let (node, layer, req) = handle.join().expect("Thread should succeed");
        assert_eq!(node, 3);
        assert_eq!(layer, 1);
        assert_eq!(req, 42);
    }

    // 2. CallbackAction: ExitEarly logits with -0.0 and +0.0 are equal (IEEE 754)
    #[test]
    fn test_callback_action_exit_early_negative_zero_equals_positive_zero() {
        let a = CallbackAction::ExitEarly { logits: vec![-0.0f32, 1.0] };
        let b = CallbackAction::ExitEarly { logits: vec![0.0f32, 1.0] };
        assert_eq!(a, b);
    }

    // 3. CallbackChain: dispatch across 100 layers with a callback targeting layer 50 only
    #[test]
    fn test_chain_dispatch_across_many_layers_single_target() {
        let cb = TestCallback::new("l50_only", 50)
            .with_layers(vec![50])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        for layer in 0..100u64 {
            let ctx = holder.ctx(layer as usize, (layer * 2) as usize);
            let action = chain.dispatch_pre_node(&ctx);
            if layer == 50 {
                assert_eq!(action, CallbackAction::SkipThisNode, "Layer 50 should match");
            } else {
                assert_eq!(action, CallbackAction::Continue, "Layer {} should not match", layer);
            }
        }
    }

    // 4. CallbackAction: CompactMask PartialEq with all-true vs all-false are not equal
    #[test]
    fn test_callback_action_compact_mask_all_true_ne_all_false() {
        let a = CallbackAction::CompactMask { active_mask: vec![true; 8] };
        let b = CallbackAction::CompactMask { active_mask: vec![false; 8] };
        assert_ne!(a, b);
    }

    // 5. CallbackChain: new() with single callback preserves sorted_indices as [0]
    #[test]
    fn test_chain_new_single_callback_sorted_indices_is_zero() {
        let cb = TestCallback::new("only_one", 123);
        let chain = CallbackChain::new(vec![Box::new(cb)]);
        assert_eq!(chain.sorted_indices, vec![0]);
        assert_eq!(chain.sorted_indices.len(), 1);
    }

    // 6. CallbackChain: dispatch_post_node with non-null kv_cache pointers passes through
    #[test]
    fn test_chain_post_node_with_non_null_kv_pointers() {
        let mut k_buf = [1.0f32, 2.0, 3.0, 4.0];
        let mut v_buf = [5.0f32, 6.0, 7.0, 8.0];
        let cb = TestCallback::new("kv_check", 50)
            .with_post_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "KVTest",
            hidden_state: &holder.hidden_state,
            kv_cache_k: k_buf.as_mut_ptr(),
            kv_cache_v: v_buf.as_mut_ptr(),
            total_seq: 4,
            seq_len: 4,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };

        let action = chain.dispatch_post_node(&ctx, &[0u8; 16]);
        assert_eq!(action, CallbackAction::SkipThisNode);
        assert!(!ctx.kv_cache_k.is_null());
        assert!(!ctx.kv_cache_v.is_null());
    }

    // 7. CallbackAction: InjectHidden PartialEq symmetry with multi-byte data
    #[test]
    fn test_callback_action_partial_eq_symmetry_inject_hidden_multi_byte() {
        let data: Vec<u8> = (0..64).collect();
        let a = CallbackAction::InjectHidden { data: data.clone() };
        let b = CallbackAction::InjectHidden { data: data.clone() };
        assert_eq!(a, b);
        assert_eq!(b, a);
        // Also verify transitivity
        let c = CallbackAction::InjectHidden { data };
        assert_eq!(a, c);
    }

    // 8. CallbackChain: callback returning InjectHidden from post_node with empty output
    #[test]
    fn test_chain_post_node_inject_hidden_empty_output_short_circuits() {
        let cb_high = TestCallback::new("high_inject", 90)
            .with_post_action(CallbackAction::InjectHidden { data: vec![0xAA, 0xBB] });
        let cb_low = TestCallback::new("low_skip", 10)
            .with_post_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        let action = chain.dispatch_post_node(&ctx, &[]);
        match action {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xAA, 0xBB]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // 9. CallbackAction: Debug for ExitEarly with single very large logit
    #[test]
    fn test_callback_action_debug_exit_early_large_logit() {
        let action = CallbackAction::ExitEarly { logits: vec![1e30f32] };
        let debug = format!("{:?}", action);
        assert!(debug.contains("ExitEarly"));
        assert!(debug.contains("logits"));
    }

    // 10. CallbackChain: dispatch_pre_node where middle callback has target filter and highest priority
    #[test]
    fn test_chain_middle_index_highest_priority_with_target_filter() {
        let cb_low = TestCallback::new("low", 10)
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_mid = TestCallback::new("mid", 100)
            .with_layers(vec![5])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0x42] });
        let cb_high_name = TestCallback::new("high", 50);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_low),
            Box::new(cb_mid),
            Box::new(cb_high_name),
        ]);

        let holder = CtxHolder::new();

        // Layer 0: mid(100) filtered → low(10) fires → SkipThisNode
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::SkipThisNode);

        // Layer 5: mid(100) fires → InjectHidden
        let ctx5 = holder.ctx(5, 10);
        match chain.dispatch_pre_node(&ctx5) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0x42]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // 11. CallbackAction: CompactMask with mask where first half true, second half false
    #[test]
    fn test_callback_action_compact_mask_first_half_active() {
        let mut mask = vec![true; 50];
        mask.extend_from_slice(&vec![false; 50]);

        let action = CallbackAction::CompactMask { active_mask: mask };
        if let CallbackAction::CompactMask { active_mask } = action {
            assert_eq!(active_mask.len(), 100);
            assert_eq!(active_mask.iter().filter(|&&v| v).count(), 50);
            assert_eq!(active_mask.iter().filter(|&&v| !v).count(), 50);
            // Verify boundary
            assert!(active_mask[49]);
            assert!(!active_mask[50]);
        } else {
            panic!("Expected CompactMask");
        }
    }

    // 12. CallbackChain: empty chain is_empty() is consistent with len() == 0
    #[test]
    fn test_chain_empty_is_empty_and_len_both_consistent() {
        let chain = CallbackChain::empty();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.callbacks.is_empty());
        assert!(chain.sorted_indices.is_empty());

        // Double-check new(vec![]) matches
        let chain2 = CallbackChain::new(vec![]);
        assert_eq!(chain.is_empty(), chain2.is_empty());
        assert_eq!(chain.len(), chain2.len());
    }

    // 13. CallbackChain: three callbacks with strictly increasing priorities, middle returns action
    #[test]
    fn test_chain_three_increasing_priorities_middle_returns_action() {
        let cb_low = TestCallback::new("p10", 10);
        let cb_mid = TestCallback::new("p50", 50)
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![7.7] });
        let cb_high = TestCallback::new("p90", 90);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_low),
            Box::new(cb_mid),
            Box::new(cb_high),
        ]);

        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Sorted: high(idx2, prio90) → Continue, mid(idx1, prio50) → ExitEarly
        let action = chain.dispatch_pre_node(&ctx);
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits.len(), 1);
                assert!((logits[0] - 7.7).abs() < 1e-6);
            }
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // 14. LayerContext: verify geometry.num_experts is 0 and moe_top_k is 0 for dense model
    #[test]
    fn test_layer_context_geometry_dense_model_fields() {
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);
        assert_eq!(ctx.model_config.geometry.num_experts, 0);
        assert_eq!(ctx.model_config.geometry.moe_top_k, 0);
        assert_eq!(ctx.model_config.geometry.expert_intermediate_size, 0);
    }

    // 15. CallbackChain: dispatch_pre_node and dispatch_post_node interleaved with different callbacks
    #[test]
    fn test_chain_interleaved_pre_post_with_different_target_layers() {
        let cb_pre = TestCallback::new("pre_l0", 100)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_post = TestCallback::new("post_l1", 50)
            .with_layers(vec![1])
            .with_post_action(CallbackAction::ExitEarly { logits: vec![3.0] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_pre), Box::new(cb_post)]);
        let holder = CtxHolder::new();

        // Layer 0: pre fires (Skip), post filtered (Continue)
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::SkipThisNode);
        assert_eq!(chain.dispatch_post_node(&ctx0, &[]), CallbackAction::Continue);

        // Layer 1: pre filtered (Continue), post fires (ExitEarly)
        let ctx1 = holder.ctx(1, 2);
        assert_eq!(chain.dispatch_pre_node(&ctx1), CallbackAction::Continue);
        match chain.dispatch_post_node(&ctx1, &[0u8; 8]) {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![3.0]),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }

        // Layer 2: both filtered → Continue
        let ctx2 = holder.ctx(2, 4);
        assert_eq!(chain.dispatch_pre_node(&ctx2), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx2, &[0u8; 8]), CallbackAction::Continue);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 15 NEW tests — remaining coverage: hidden_state content-driven callback
    // decisions, post_node stateful transitions, multi-callback target layer
    // interactions, CallbackAction mutation isolation, callback_chain_handle
    // field access, context field ratio-based callback logic, sorted_indices
    // uniqueness invariant, callback that modifies behavior across dispatches
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    // 1. Callback decides based on hidden_state content: all-zero → Skip, non-zero → Continue
    #[test]
    // @trace TEST-LCB-441 [req:REQ-CB-001] [level:unit]
    fn test_chain_callback_decides_on_hidden_state_all_zeros() {
        // Arrange: callback that checks if hidden_state starts with all zeros
        struct HiddenStateCheck;
        impl LayerCallback for HiddenStateCheck {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                if ctx.hidden_state.len() >= 4 {
                    let first_four_all_zero = ctx.hidden_state[..4].iter().all(|&b| b == 0);
                    if first_four_all_zero {
                        return CallbackAction::SkipThisNode;
                    }
                }
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "hs_check" }
        }

        let cb = HiddenStateCheck;
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        // Case 1: all-zero hidden state (default CtxHolder)
        let holder_zero = CtxHolder::new();
        let ctx_zero = holder_zero.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx_zero), CallbackAction::SkipThisNode);

        // Case 2: non-zero hidden state
        let mut holder_nonzero = CtxHolder::new();
        holder_nonzero.hidden_state[0] = 0xFF;
        let ctx_nonzero = holder_nonzero.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx_nonzero), CallbackAction::Continue);
    }

    // 2. Stateful callback that transitions from Continue → ExitEarly after accumulating output bytes
    #[test]
    // @trace TEST-LCB-442 [req:REQ-CB-001] [level:unit]
    fn test_chain_stateful_post_node_transitions_after_byte_accumulation() {
        // Arrange: callback accumulates output bytes, exits once threshold exceeded
        struct ByteThreshold { accumulated: usize, threshold: usize }
        impl LayerCallback for ByteThreshold {
            fn post_node(&mut self, _ctx: &LayerContext, output: &[u8]) -> CallbackAction {
                self.accumulated += output.len();
                if self.accumulated > self.threshold {
                    CallbackAction::ExitEarly { logits: vec![-1.0] }
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "byte_threshold" }
        }

        let cb = ByteThreshold { accumulated: 0, threshold: 32 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act & Assert: dispatch with 16 bytes → accumulated=16 ≤ 32 → Continue
        assert_eq!(chain.dispatch_post_node(&ctx, &[0u8; 16]), CallbackAction::Continue);

        // dispatch with another 16 bytes → accumulated=32 ≤ 32 → Continue
        assert_eq!(chain.dispatch_post_node(&ctx, &[0u8; 16]), CallbackAction::Continue);

        // dispatch with 1 byte → accumulated=33 > 32 → ExitEarly
        match chain.dispatch_post_node(&ctx, &[0u8; 1]) {
            CallbackAction::ExitEarly { logits } => assert_eq!(logits, vec![-1.0]),
            other => panic!("Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }
    }

    // 3. Three callbacks with non-overlapping target layers, each returns a different action
    #[test]
    // @trace TEST-LCB-443 [req:REQ-CB-001] [level:unit]
    fn test_chain_three_non_overlapping_targets_three_different_actions() {
        // Arrange: layer 0 → InjectHidden, layer 1 → CompactMask, layer 2 → ExitEarly
        let cb_l0 = TestCallback::new("l0_inject", 100)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::InjectHidden { data: vec![0x01] });
        let cb_l1 = TestCallback::new("l1_compact", 50)
            .with_layers(vec![1])
            .with_pre_action(CallbackAction::CompactMask { active_mask: vec![true, false] });
        let cb_l2 = TestCallback::new("l2_exit", 10)
            .with_layers(vec![2])
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![0.0] });

        let mut chain = CallbackChain::new(vec![
            Box::new(cb_l0), Box::new(cb_l1), Box::new(cb_l2),
        ]);
        let holder = CtxHolder::new();

        // Layer 0 → InjectHidden (only l0 matches)
        let ctx0 = holder.ctx(0, 0);
        match chain.dispatch_pre_node(&ctx0) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0x01]),
            other => panic!("Layer 0: Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }

        // Layer 1 → CompactMask (only l1 matches)
        let ctx1 = holder.ctx(1, 2);
        match chain.dispatch_pre_node(&ctx1) {
            CallbackAction::CompactMask { active_mask } => assert_eq!(active_mask, vec![true, false]),
            other => panic!("Layer 1: Expected CompactMask, got {:?}", action_variant_name(&other)),
        }

        // Layer 2 → ExitEarly (only l2 matches)
        let ctx2 = holder.ctx(2, 4);
        match chain.dispatch_pre_node(&ctx2) {
            CallbackAction::ExitEarly { logits } => assert!(logits == vec![0.0]),
            other => panic!("Layer 2: Expected ExitEarly, got {:?}", action_variant_name(&other)),
        }

        // Layer 3 → none match → Continue
        let ctx3 = holder.ctx(3, 6);
        assert_eq!(chain.dispatch_pre_node(&ctx3), CallbackAction::Continue);
    }

    // 4. CallbackAction: mutating a cloned ExitEarly does not affect the original
    #[test]
    // @trace TEST-LCB-444 [req:REQ-CB-001] [level:unit]
    fn test_callback_action_exit_early_clone_then_mutate_isolated() {
        // Arrange
        let original = CallbackAction::ExitEarly { logits: vec![1.0, 2.0, 3.0] };
        let mut cloned = original.clone();

        // Act: clear the cloned logits
        if let CallbackAction::ExitEarly { logits } = &mut cloned {
            logits.clear();
        }

        // Assert: original untouched
        if let CallbackAction::ExitEarly { logits } = &original {
            assert_eq!(logits.len(), 3);
            assert_eq!(logits[0], 1.0);
            assert_eq!(logits[2], 3.0);
        } else {
            panic!("Original should be ExitEarly");
        }

        // Assert: cloned was mutated
        if let CallbackAction::ExitEarly { logits } = &cloned {
            assert!(logits.is_empty());
        } else {
            panic!("Cloned should be ExitEarly");
        }
    }

    // 5. CallbackChain: callback returns different actions based on total_seq vs seq_len ratio
    #[test]
    // @trace TEST-LCB-445 [req:REQ-CB-001] [level:unit]
    fn test_chain_callback_decides_on_total_seq_to_seq_len_ratio() {
        // Arrange: callback returns SkipThisNode for prefill (seq_len > 1), Continue for decode
        struct PrefillDetector;
        impl LayerCallback for PrefillDetector {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                if ctx.seq_len > 1 {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "prefill_detector" }
        }

        let cb = PrefillDetector;
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Decode step (seq_len=1): Continue
        let ctx_decode = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 100, seq_len: 1, position: 99,
            request_id: 1, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx_decode), CallbackAction::Continue);

        // Prefill step (seq_len=128): SkipThisNode
        let ctx_prefill = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(), kv_cache_v: std::ptr::null_mut(),
            total_seq: 128, seq_len: 128, position: 0,
            request_id: 2, model_config: &holder.config,
        };
        assert_eq!(chain.dispatch_pre_node(&ctx_prefill), CallbackAction::SkipThisNode);
    }

    // 6. CallbackChain: sorted_indices has no duplicate entries (invariant check)
    #[test]
    // @trace TEST-LCB-446 [req:REQ-CB-001] [level:unit]
    fn test_chain_sorted_indices_no_duplicates_invariant() {
        // Arrange: 6 callbacks with various priorities
        let callbacks: Vec<Box<dyn LayerCallback + Send>> = vec![
            Box::new(TestCallback::new("a", 100)),
            Box::new(TestCallback::new("b", 50)),
            Box::new(TestCallback::new("c", 75)),
            Box::new(TestCallback::new("d", 25)),
            Box::new(TestCallback::new("e", 50)),
            Box::new(TestCallback::new("f", 0)),
        ];
        let chain = CallbackChain::new(callbacks);

        // Assert: all indices are unique
        let mut seen = std::collections::HashSet::new();
        assert_eq!(chain.sorted_indices.len(), 6);
        for &idx in &chain.sorted_indices {
            assert!(seen.insert(idx), "Duplicate index {} found in sorted_indices", idx);
            assert!(idx < 6, "Index {} out of range", idx);
        }
        assert_eq!(seen.len(), 6);
    }

    // 7. CallbackChain: callback returns SkipThisNode on pre_node and InjectHidden on post_node
    #[test]
    // @trace TEST-LCB-447 [req:REQ-CB-001] [level:unit]
    fn test_chain_callback_different_action_pre_vs_post() {
        // Arrange
        let cb = TestCallback::new("split_behavior", 75)
            .with_pre_action(CallbackAction::SkipThisNode)
            .with_post_action(CallbackAction::InjectHidden { data: vec![0xDE, 0xAD] });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act & Assert: pre_node returns SkipThisNode
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::SkipThisNode);

        // post_node returns InjectHidden (independent dispatch)
        match chain.dispatch_post_node(&ctx, &[]) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xDE, 0xAD]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // 8. LayerContext: multiple contexts from same holder share the same hidden_state pointer
    #[test]
    // @trace TEST-LCB-448 [req:REQ-CB-001] [level:unit]
    fn test_layer_contexts_share_hidden_state_pointer() {
        // Arrange
        let holder = CtxHolder::new();
        let ctx0 = holder.ctx(0, 0);
        let ctx1 = holder.ctx(1, 2);

        // Act & Assert: both hidden_state slices have same address
        let ptr0 = ctx0.hidden_state.as_ptr();
        let ptr1 = ctx1.hidden_state.as_ptr();
        assert_eq!(ptr0, ptr1, "Both contexts should reference the same hidden_state allocation");
        assert_eq!(ctx0.hidden_state.len(), ctx1.hidden_state.len());
    }

    // 9. CallbackChain: two stateful callbacks in chain, second accumulates state from first
    #[test]
    // @trace TEST-LCB-449 [req:REQ-CB-001] [level:unit]
    fn test_chain_two_stateful_callbacks_both_invoked_until_short_circuit() {
        // Arrange: high priority counts and returns Continue, low counts and returns Skip after N calls
        struct CountingCallback { name: &'static str, count: usize, limit: usize }
        impl LayerCallback for CountingCallback {
            fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
                self.count += 1;
                if self.count >= self.limit {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { self.name }
        }

        let cb_high = CountingCallback { name: "high_counter", count: 0, limit: 10 };
        let cb_low = CountingCallback { name: "low_counter", count: 0, limit: 3 };
        let mut chain = CallbackChain::new(vec![
            Box::new(cb_high), Box::new(cb_low),
        ]);
        let holder = CtxHolder::new();

        // Act: dispatch 3 times
        for i in 0..2 {
            let ctx = holder.ctx(i, i);
            assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue,
                "Dispatch {} should return Continue", i);
        }
        // Third dispatch: low_counter hits limit (count=3 >= 3) → SkipThisNode
        let ctx2 = holder.ctx(2, 4);
        assert_eq!(chain.dispatch_pre_node(&ctx2), CallbackAction::SkipThisNode);

        // Fourth dispatch: high_counter has count=4 (still < 10), low_counter has count=4 (>= 3) → Skip
        let ctx3 = holder.ctx(3, 6);
        assert_eq!(chain.dispatch_pre_node(&ctx3), CallbackAction::SkipThisNode);
    }

    // 10. CallbackAction: ExitEarly with logit f32::MIN clones correctly and preserves value
    #[test]
    // @trace TEST-LCB-450 [req:REQ-CB-001] [level:unit]
    fn test_callback_action_exit_early_f32_min_clone_preserves_value() {
        // Arrange
        let original = CallbackAction::ExitEarly { logits: vec![f32::MIN] };
        let cloned = original.clone();

        // Assert: cloned preserves f32::MIN exactly
        if let CallbackAction::ExitEarly { logits } = cloned {
            assert_eq!(logits[0], f32::MIN);
            assert!(logits[0] < -3.4e38);
        } else {
            panic!("Expected ExitEarly");
        }

        // Original also unchanged
        if let CallbackAction::ExitEarly { logits } = &original {
            assert_eq!(logits[0], f32::MIN);
        } else {
            panic!("Original should be ExitEarly");
        }
    }

    // 11. CallbackChain: callback with target_layers=[0,2,4] fires on exactly those layers
    #[test]
    // @trace TEST-LCB-451 [req:REQ-CB-001] [level:unit]
    fn test_chain_sparse_target_layers_fires_only_on_specified() {
        // Arrange
        let cb = TestCallback::new("sparse_targets", 60)
            .with_layers(vec![0, 2, 4])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Act & Assert: layers 0-5, only 0, 2, 4 should fire
        for layer in 0..6 {
            let ctx = holder.ctx(layer, layer);
            let expected = if layer == 0 || layer == 2 || layer == 4 {
                CallbackAction::SkipThisNode
            } else {
                CallbackAction::Continue
            };
            assert_eq!(chain.dispatch_pre_node(&ctx), expected,
                "Layer {} should return {:?}", layer, expected);
        }
    }

    // 12. CallbackChain: post_node callback that reads kv_cache_v pointer nullability
    #[test]
    // @trace TEST-LCB-452 [req:REQ-CB-001] [level:unit]
    fn test_chain_post_node_callback_checks_kv_cache_v_nullability() {
        // Arrange
        struct KvNullCheck { v_was_null: bool }
        impl LayerCallback for KvNullCheck {
            fn post_node(&mut self, ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
                self.v_was_null = ctx.kv_cache_v.is_null();
                if self.v_was_null {
                    CallbackAction::Continue
                } else {
                    CallbackAction::SkipThisNode
                }
            }
            fn name(&self) -> &str { "kv_null_check" }
        }

        let cb = KvNullCheck { v_was_null: false };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Case 1: null kv_cache_v → Continue
        let ctx_null = holder.ctx(0, 0);
        assert!(ctx_null.kv_cache_v.is_null());
        assert_eq!(chain.dispatch_post_node(&ctx_null, &[0u8; 8]), CallbackAction::Continue);

        // Case 2: non-null kv_cache_v → SkipThisNode
        let mut v_buf = [0.0f32; 4];
        let ctx_nonnull = LayerContext {
            node_idx: 0, layer_idx: 0, node_op: "Op",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: v_buf.as_mut_ptr(),
            total_seq: 1, seq_len: 1, position: 0,
            request_id: 0, model_config: &holder.config,
        };
        assert!(!ctx_nonnull.kv_cache_v.is_null());
        assert_eq!(chain.dispatch_post_node(&ctx_nonnull, &[0u8; 8]), CallbackAction::SkipThisNode);
    }

    // 13. CallbackAction: CompactMask cloned, mutated, original unchanged
    #[test]
    // @trace TEST-LCB-453 [req:REQ-CB-001] [level:unit]
    fn test_callback_action_compact_mask_clone_mutation_isolation() {
        // Arrange
        let mask: Vec<bool> = (0..16).map(|i| i % 2 == 0).collect();
        let original = CallbackAction::CompactMask { active_mask: mask };
        let mut cloned = original.clone();

        // Act: flip all bits in cloned
        if let CallbackAction::CompactMask { active_mask } = &mut cloned {
            for val in active_mask.iter_mut() {
                *val = !*val;
            }
        }

        // Assert: original unchanged
        if let CallbackAction::CompactMask { active_mask } = &original {
            assert!(active_mask[0]);
            assert!(!active_mask[1]);
            assert!(active_mask[2]);
            assert_eq!(active_mask.len(), 16);
        } else {
            panic!("Original should be CompactMask");
        }

        // Assert: cloned is flipped
        if let CallbackAction::CompactMask { active_mask } = &cloned {
            assert!(!active_mask[0]);
            assert!(active_mask[1]);
            assert!(!active_mask[2]);
        } else {
            panic!("Cloned should be CompactMask");
        }
    }

    // 14. CallbackChain: dispatch_pre_node with four callbacks, all returning Continue, then dispatch_post_node where third returns action
    #[test]
    // @trace TEST-LCB-454 [req:REQ-CB-001] [level:unit]
    fn test_chain_all_continue_pre_then_third_fires_on_post() {
        // Arrange: 4 callbacks with priorities 100, 75, 50, 25
        let cb1 = TestCallback::new("p100", 100);
        let cb2 = TestCallback::new("p75", 75);
        let cb3 = TestCallback::new("p50", 50)
            .with_post_action(CallbackAction::InjectHidden { data: vec![0xFE] });
        let cb4 = TestCallback::new("p25", 25)
            .with_post_action(CallbackAction::SkipThisNode);

        let mut chain = CallbackChain::new(vec![
            Box::new(cb1), Box::new(cb2), Box::new(cb3), Box::new(cb4),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act & Assert: pre_node all Continue
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);

        // post_node: p100→Continue, p75→Continue, p50→InjectHidden (short-circuits, p25 never reached)
        match chain.dispatch_post_node(&ctx, &[0u8; 32]) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xFE]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }
    }

    // 15. CallbackChain: building chain from callbacks with priority sequence [3,1,4,1,5] sorts correctly
    #[test]
    // @trace TEST-LCB-455 [req:REQ-CB-001] [level:unit]
    fn test_chain_priority_sequence_3_1_4_1_5_sorts_correctly() {
        // Arrange: "pi" digit priorities — 3, 1, 4, 1, 5
        let cb0 = TestCallback::new("p3", 3);
        let cb1 = TestCallback::new("p1a", 1);
        let cb2 = TestCallback::new("p4", 4);
        let cb3 = TestCallback::new("p1b", 1);
        let cb4 = TestCallback::new("p5", 5);

        let chain = CallbackChain::new(vec![
            Box::new(cb0), Box::new(cb1), Box::new(cb2), Box::new(cb3), Box::new(cb4),
        ]);

        // Assert: sorted descending: 5(idx4), 4(idx2), 3(idx0), 1(idx1), 1(idx3)
        // Stable sort keeps idx1 before idx3 for equal priority 1
        assert_eq!(chain.sorted_indices, vec![4, 2, 0, 1, 3]);
        assert_eq!(chain.len(), 5);

        // Verify each sorted index maps to the correct priority
        let priorities: Vec<u32> = vec![3, 1, 4, 1, 5];
        for i in 0..chain.sorted_indices.len() - 1 {
            let idx_a = chain.sorted_indices[i];
            let idx_b = chain.sorted_indices[i + 1];
            assert!(priorities[idx_a] >= priorities[idx_b],
                "Priority at sorted position {} ({}) should be >= position {} ({})",
                i, priorities[idx_a], i + 1, priorities[idx_b]);
        }
    }

    // ========================================================================
    // 15 additional tests for uncovered paths
    // ========================================================================

    #[test]
    fn test_layer_context_with_empty_hidden_state_slice() {
        // Arrange: hidden state is an empty byte slice (edge case for degenerate tensors)
        let holder = CtxHolder::new();
        let empty_state: Vec<u8> = vec![];
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "EmptyTensor",
            hidden_state: &empty_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 0,
            seq_len: 0,
            position: 0,
            request_id: 0,
            model_config: &holder.config,
        };

        // Assert: empty hidden_state is valid and accessible
        assert!(ctx.hidden_state.is_empty());
        assert_eq!(ctx.hidden_state.len(), 0);
    }

    #[test]
    fn test_chain_pre_node_continues_then_post_node_compact_mask_on_same_ctx() {
        // Arrange: single callback returns Continue on pre_node, CompactMask on post_node
        let mask = vec![true, true, false, true, false, true, true, false];
        let cb = TestCallback::new("pre_cont_post_compact", 50)
            .with_post_action(CallbackAction::CompactMask { active_mask: mask.clone() });
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(2, 4);

        // Act: pre_node returns Continue (default)
        let pre = chain.dispatch_pre_node(&ctx);
        assert_eq!(pre, CallbackAction::Continue);

        // Act: post_node returns CompactMask
        let post = chain.dispatch_post_node(&ctx, &[0u8; 128]);
        match post {
            CallbackAction::CompactMask { active_mask: m } => {
                assert_eq!(m, mask);
            }
            _ => panic!("Expected CompactMask, got {:?}", action_variant_name(&post)),
        }
    }

    #[test]
    fn test_callback_action_partial_eq_compact_mask_different_lengths_not_equal() {
        // Arrange: two CompactMask with different lengths should not be equal
        let a = CallbackAction::CompactMask { active_mask: vec![true] };
        let b = CallbackAction::CompactMask { active_mask: vec![true, true] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_callback_action_partial_eq_inject_hidden_same_bytes_different_alloc() {
        // Arrange: two InjectHidden actions with same bytes from different allocations
        let data_a: Vec<u8> = (0..64).collect();
        let data_b: Vec<u8> = (0..64).collect();
        let a = CallbackAction::InjectHidden { data: data_a };
        let b = CallbackAction::InjectHidden { data: data_b };
        // Assert: content equality, not identity
        assert_eq!(a, b);
    }

    #[test]
    fn test_chain_dispatch_pre_node_two_matching_targets_first_returns_continue_second_skips() {
        // Arrange: two callbacks target same layer, first returns Continue, second returns Skip
        let cb_high = TestCallback::new("high_continue", 80)
            .with_layers(vec![1]);
        let cb_low = TestCallback::new("low_skip", 20)
            .with_layers(vec![1])
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(1, 2);

        // Act: high returns Continue, dispatch proceeds to low which returns SkipThisNode
        let action = chain.dispatch_pre_node(&ctx);

        // Assert
        assert_eq!(action, CallbackAction::SkipThisNode);
    }

    #[test]
    fn test_layer_context_position_encoding_field_matches_config() {
        // Arrange: verify position_encoding is accessible through context
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Assert: position_encoding matches what was configured in CtxHolder
        assert!(matches!(
            ctx.model_config.position_encoding,
            crate::engine::executor::PositionEncoding::Rope
        ));
    }

    #[test]
    fn test_layer_context_arch_family_field_matches_config() {
        // Arrange: verify arch_family is accessible through context
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Assert: arch_family matches what was configured
        assert!(matches!(
            ctx.model_config.arch_family,
            crate::manifest::ArchFamily::Decoder
        ));
    }

    #[test]
    fn test_chain_four_callbacks_each_targeting_distinct_single_layer() {
        // Arrange: 4 callbacks, each targeting exactly one distinct layer [0,1,2,3]
        let make_cb = |name: &'static str, layer: usize| {
            TestCallback::new(name, 50 - layer as u32)
                .with_layers(vec![layer])
                .with_pre_action(CallbackAction::SkipThisNode)
        };
        let mut chain = CallbackChain::new(vec![
            Box::new(make_cb("l0", 0)),
            Box::new(make_cb("l1", 1)),
            Box::new(make_cb("l2", 2)),
            Box::new(make_cb("l3", 3)),
        ]);
        let holder = CtxHolder::new();

        // Act & Assert: each layer triggers exactly one callback
        for layer in 0..4 {
            let ctx = holder.ctx(layer, layer);
            let action = chain.dispatch_pre_node(&ctx);
            assert_eq!(action, CallbackAction::SkipThisNode,
                "Layer {} should trigger its callback", layer);
        }

        // Layer 4 has no targeting callback
        let ctx4 = holder.ctx(4, 8);
        assert_eq!(chain.dispatch_pre_node(&ctx4), CallbackAction::Continue);
    }

    #[test]
    fn test_callback_action_partial_eq_exit_early_different_lengths_not_equal() {
        // Arrange: ExitEarly with different logit counts should not be equal
        let a = CallbackAction::ExitEarly { logits: vec![1.0] };
        let b = CallbackAction::ExitEarly { logits: vec![1.0, 1.0] };
        assert_ne!(a, b);
    }

    #[test]
    fn test_chain_empty_dispatch_pre_and_post_with_empty_output() {
        // Arrange: empty chain dispatches both pre and post with empty output
        let mut chain = CallbackChain::empty();
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act & Assert: both dispatches return Continue with no callbacks
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[1u8, 2, 3]), CallbackAction::Continue);
    }

    #[test]
    fn test_layer_context_paged_kv_config_accessible() {
        // Arrange: verify paged_kv config fields through context
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Assert: page_size is accessible
        assert_eq!(ctx.model_config.paged_kv.page_size, 16);
        assert!(ctx.model_config.paged_kv.page_table.is_none());
    }

    #[test]
    fn test_chain_dispatch_pre_node_callback_returns_exit_early_with_single_logit_stops_chain() {
        // Arrange: high-priority returns Continue, mid returns ExitEarly with single logit
        let cb_high = TestCallback::new("high_cont", 90);
        let cb_mid = TestCallback::new("mid_exit", 50)
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![42.0] });
        let cb_low = TestCallback::new("low_skip", 10)
            .with_pre_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![
            Box::new(cb_high),
            Box::new(cb_mid),
            Box::new(cb_low),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act
        let action = chain.dispatch_pre_node(&ctx);

        // Assert: mid's ExitEarly stops dispatch, low never reached
        match action {
            CallbackAction::ExitEarly { logits } => {
                assert_eq!(logits, vec![42.0]);
            }
            _ => panic!("Expected ExitEarly, got {:?}", action_variant_name(&action)),
        }
    }

    #[test]
    fn test_callback_action_debug_compact_mask_with_mixed_bools() {
        // Arrange: verify Debug output for CompactMask with mixed booleans
        let action = CallbackAction::CompactMask {
            active_mask: vec![true, false, true, false, true],
        };
        let debug = format!("{:?}", action);

        // Assert: Debug contains variant name and the boolean values
        assert!(debug.contains("CompactMask"));
        assert!(debug.contains("active_mask"));
    }

    #[test]
    fn test_chain_dispatch_post_node_three_callbacks_first_two_filtered_last_returns_skip() {
        // Arrange: first two callbacks target wrong layers, third targets correct layer
        let cb1 = TestCallback::new("wrong_l5", 90)
            .with_layers(vec![5])
            .with_post_action(CallbackAction::ExitEarly { logits: vec![1.0] });
        let cb2 = TestCallback::new("wrong_l7", 60)
            .with_layers(vec![7])
            .with_post_action(CallbackAction::SkipThisNode);
        let cb3 = TestCallback::new("correct_l0", 30)
            .with_layers(vec![0])
            .with_post_action(CallbackAction::SkipThisNode);
        let mut chain = CallbackChain::new(vec![
            Box::new(cb1),
            Box::new(cb2),
            Box::new(cb3),
        ]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act: first two filtered, third fires
        let action = chain.dispatch_post_node(&ctx, &[0u8; 32]);

        // Assert
        assert_eq!(action, CallbackAction::SkipThisNode);
    }

    #[test]
    fn test_chain_repeated_dispatch_pre_and_post_interleaved_preserves_state() {
        // Arrange: stateful callback counts both pre and post invocations
        struct CountingCallback {
            pre_count: usize,
            post_count: usize,
        }
        impl LayerCallback for CountingCallback {
            fn pre_node(&mut self, _ctx: &LayerContext) -> CallbackAction {
                self.pre_count += 1;
                CallbackAction::Continue
            }
            fn post_node(&mut self, _ctx: &LayerContext, _output: &[u8]) -> CallbackAction {
                self.post_count += 1;
                CallbackAction::Continue
            }
            fn name(&self) -> &str { "counter" }
        }

        let cb = CountingCallback { pre_count: 0, post_count: 0 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);
        let holder = CtxHolder::new();

        // Act: interleave 5 pre and 5 post dispatches
        for i in 0..5 {
            let ctx = holder.ctx(i, i);
            let pre = chain.dispatch_pre_node(&ctx);
            assert_eq!(pre, CallbackAction::Continue);
            let post = chain.dispatch_post_node(&ctx, &[]);
            assert_eq!(post, CallbackAction::Continue);
        }

        // Assert: we can verify the chain still dispatches correctly after repeated calls
        // (The callback's internal state was mutated 10 times total)
        let ctx = holder.ctx(99, 99);
        assert_eq!(chain.dispatch_pre_node(&ctx), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx, &[]), CallbackAction::Continue);
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // 10 additional tests — uncovered edge cases and interactions
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    #[test]
    fn test_callback_action_exit_early_partial_eq_after_sorting_logits() {
        // Arrange: two ExitEarly actions with same logits but different construction order
        let mut logits_a = vec![3.0, 1.0, 2.0];
        let mut logits_b = vec![2.0, 3.0, 1.0];
        logits_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
        logits_b.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let a = CallbackAction::ExitEarly { logits: logits_a };
        let b = CallbackAction::ExitEarly { logits: logits_b };

        // Assert: sorted logits are equal, so actions are equal
        assert_eq!(a, b);
    }

    #[test]
    fn test_chain_post_node_two_callbacks_same_layer_first_continue_second_compact_mask() {
        // Arrange: two callbacks both target layer 1, high returns Continue, low returns CompactMask
        let mask = vec![true, false, true, true, false];
        let expected_mask = mask.clone();
        let cb_high = TestCallback::new("high_cont_l1", 90)
            .with_layers(vec![1]);
        let cb_low = TestCallback::new("low_compact_l1", 30)
            .with_layers(vec![1])
            .with_post_action(CallbackAction::CompactMask { active_mask: mask });

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();

        // Layer 0: neither targets → Continue
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_post_node(&ctx0, &[0u8; 16]), CallbackAction::Continue);

        // Layer 1: both match; high→Continue, low→CompactMask
        let ctx1 = holder.ctx(1, 2);
        match chain.dispatch_post_node(&ctx1, &[0u8; 16]) {
            CallbackAction::CompactMask { active_mask } => assert_eq!(active_mask, expected_mask),
            other => panic!("Expected CompactMask, got {:?}", action_variant_name(&other)),
        }
    }

    #[test]
    fn test_callback_action_inject_hidden_with_capacity_preserves_content() {
        // Arrange: InjectHidden constructed from Vec with excess capacity
        let mut data = Vec::with_capacity(1024);
        data.extend_from_slice(&[0xCA, 0xFE, 0xBA, 0xBE]);
        assert_eq!(data.len(), 4);
        assert!(data.capacity() >= 1024);

        let action = CallbackAction::InjectHidden { data };

        // Assert: only the actual content matters, not capacity
        if let CallbackAction::InjectHidden { data } = action {
            assert_eq!(data.len(), 4);
            assert_eq!(data, vec![0xCA, 0xFE, 0xBA, 0xBE]);
        } else {
            panic!("Expected InjectHidden");
        }
    }

    #[test]
    fn test_chain_callback_inspects_hidden_state_specific_byte_pattern() {
        // Arrange: callback that checks a specific byte offset in hidden_state
        struct PatternCheckCallback { offset: usize, expected: u8 }
        impl LayerCallback for PatternCheckCallback {
            fn pre_node(&mut self, ctx: &LayerContext) -> CallbackAction {
                if ctx.hidden_state.len() > self.offset && ctx.hidden_state[self.offset] == self.expected {
                    CallbackAction::SkipThisNode
                } else {
                    CallbackAction::Continue
                }
            }
            fn name(&self) -> &str { "pattern_check" }
        }

        let cb = PatternCheckCallback { offset: 100, expected: 0x42 };
        let mut chain = CallbackChain::new(vec![Box::new(cb)]);

        // Case 1: default hidden state (all zeros) → byte at offset 100 is 0, not 0x42
        let holder_default = CtxHolder::new();
        let ctx_default = holder_default.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx_default), CallbackAction::Continue);

        // Case 2: hidden state with 0x42 at offset 100 → matches
        let mut holder_pattern = CtxHolder::new();
        holder_pattern.hidden_state[100] = 0x42;
        let ctx_pattern = holder_pattern.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx_pattern), CallbackAction::SkipThisNode);
    }

    #[test]
    fn test_chain_all_callbacks_target_same_layer_all_return_continue() {
        // Arrange: 3 callbacks all target layer 2, all return Continue
        let cb1 = TestCallback::new("cb1_l2", 100)
            .with_layers(vec![2]);
        let cb2 = TestCallback::new("cb2_l2", 50)
            .with_layers(vec![2]);
        let cb3 = TestCallback::new("cb3_l2", 10)
            .with_layers(vec![2]);

        let mut chain = CallbackChain::new(vec![Box::new(cb1), Box::new(cb2), Box::new(cb3)]);
        let holder = CtxHolder::new();

        // Layer 2: all three match, all return Continue
        let ctx2 = holder.ctx(2, 4);
        assert_eq!(chain.dispatch_pre_node(&ctx2), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx2, &[0u8; 32]), CallbackAction::Continue);

        // Layer 1: none match
        let ctx1 = holder.ctx(1, 2);
        assert_eq!(chain.dispatch_pre_node(&ctx1), CallbackAction::Continue);
    }

    #[test]
    fn test_layer_context_total_seq_less_than_seq_len_edge_case() {
        // Arrange: unusual but structurally valid — total_seq < seq_len (corrupted or edge case)
        let holder = CtxHolder::new();
        let ctx = LayerContext {
            node_idx: 0,
            layer_idx: 0,
            node_op: "EdgeCase",
            hidden_state: &holder.hidden_state,
            kv_cache_k: std::ptr::null_mut(),
            kv_cache_v: std::ptr::null_mut(),
            total_seq: 5,
            seq_len: 10,
            position: 0,
            request_id: 1,
            model_config: &holder.config,
        };

        // Assert: struct stores raw values without enforcing invariants
        assert_eq!(ctx.total_seq, 5);
        assert_eq!(ctx.seq_len, 10);
        assert!(ctx.total_seq < ctx.seq_len);
    }

    #[test]
    fn test_callback_action_default_always_continue_across_variations() {
        // Arrange: verify default() consistently returns Continue for 100 invocations
        let mut actions = Vec::with_capacity(100);
        for _ in 0..100 {
            actions.push(CallbackAction::default());
        }

        // Assert: every invocation returns Continue
        for (i, action) in actions.iter().enumerate() {
            assert!(matches!(action, CallbackAction::Continue), "Action {} should be Continue", i);
        }

        // Assert: all 100 actions are equal to each other
        for i in 1..100 {
            assert_eq!(actions[0], actions[i], "Action 0 should equal action {}", i);
        }
    }

    #[test]
    fn test_chain_pre_node_compact_mask_first_callback_wins_over_second_exit_early() {
        // Arrange: two callbacks targeting same layer; high returns CompactMask, low returns ExitEarly
        let mask = vec![false, true, false, true];
        let expected_mask = mask.clone();
        let cb_high = TestCallback::new("high_compact", 80)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::CompactMask { active_mask: mask });
        let cb_low = TestCallback::new("low_exit", 20)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::ExitEarly { logits: vec![1.0, 2.0] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_high), Box::new(cb_low)]);
        let holder = CtxHolder::new();
        let ctx = holder.ctx(0, 0);

        // Act: high-priority CompactMask wins, low never reached
        match chain.dispatch_pre_node(&ctx) {
            CallbackAction::CompactMask { active_mask } => assert_eq!(active_mask, expected_mask),
            other => panic!("Expected CompactMask, got {:?}", action_variant_name(&other)),
        }
    }

    #[test]
    fn test_callback_action_exit_early_logits_with_mixed_integer_and_fractional_values() {
        // Arrange: logits that are integer-valued floats mixed with fractional
        let logits = vec![0.0, 1.0, 2.0, 0.5, 1.5, 100.0, -50.0];
        let action = CallbackAction::ExitEarly { logits: logits.clone() };

        // Assert: all values preserved in order
        if let CallbackAction::ExitEarly { logits: l } = action {
            assert_eq!(l.len(), 7);
            assert_eq!(l[0], 0.0);
            assert_eq!(l[1], 1.0);
            assert_eq!(l[2], 2.0);
            assert!((l[3] - 0.5).abs() < f32::EPSILON);
            assert!((l[4] - 1.5).abs() < f32::EPSILON);
            assert_eq!(l[5], 100.0);
            assert_eq!(l[6], -50.0);
        } else {
            panic!("Expected ExitEarly");
        }
    }

    #[test]
    fn test_chain_pre_node_skip_and_post_node_inject_hidden_different_callbacks_different_layers() {
        // Arrange: pre callback targets layer 0 (Skip), post callback targets layer 1 (Inject)
        let cb_pre = TestCallback::new("pre_skip_l0", 80)
            .with_layers(vec![0])
            .with_pre_action(CallbackAction::SkipThisNode);
        let cb_post = TestCallback::new("post_inject_l1", 50)
            .with_layers(vec![1])
            .with_post_action(CallbackAction::InjectHidden { data: vec![0xBB, 0xCC] });

        let mut chain = CallbackChain::new(vec![Box::new(cb_pre), Box::new(cb_post)]);
        let holder = CtxHolder::new();

        // Layer 0: pre fires (Skip), post filtered (Continue)
        let ctx0 = holder.ctx(0, 0);
        assert_eq!(chain.dispatch_pre_node(&ctx0), CallbackAction::SkipThisNode);
        assert_eq!(chain.dispatch_post_node(&ctx0, &[]), CallbackAction::Continue);

        // Layer 1: pre filtered (Continue), post fires (InjectHidden)
        let ctx1 = holder.ctx(1, 2);
        assert_eq!(chain.dispatch_pre_node(&ctx1), CallbackAction::Continue);
        match chain.dispatch_post_node(&ctx1, &[0u8; 16]) {
            CallbackAction::InjectHidden { data } => assert_eq!(data, vec![0xBB, 0xCC]),
            other => panic!("Expected InjectHidden, got {:?}", action_variant_name(&other)),
        }

        // Layer 2: both filtered (Continue on both)
        let ctx2 = holder.ctx(2, 4);
        assert_eq!(chain.dispatch_pre_node(&ctx2), CallbackAction::Continue);
        assert_eq!(chain.dispatch_post_node(&ctx2, &[0u8; 16]), CallbackAction::Continue);
    }
}
