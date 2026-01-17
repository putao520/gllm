//! Inference hooks for model introspection.
//!
//! This module provides a callback mechanism to extract intermediate states
//! during model inference, particularly useful for:
//! - Extracting embeddings from the last hidden layer
//! - Monitoring layer outputs for debugging
//! - Implementing custom post-processing logic
//!
//! # Example
//!
//! ```ignore
//! use gllm::hooks::{InferenceHook, HookManager};
//! use std::sync::{Arc, Mutex};
//!
//! struct EmbeddingExtractor {
//!     embeddings: Mutex<Vec<Vec<f32>>>,
//! }
//!
//! impl InferenceHook for EmbeddingExtractor {
//!     fn on_last_hidden_state(&self, hidden_states: &[f32], shape: [usize; 3], _layer_idx: usize) {
//!         let [_batch, seq_len, hidden_size] = shape;
//!         // Extract the last token's hidden state
//!         let last_token_start = (seq_len - 1) * hidden_size;
//!         let embedding = hidden_states[last_token_start..last_token_start + hidden_size].to_vec();
//!         self.embeddings.lock().unwrap().push(embedding);
//!     }
//! }
//!
//! let extractor = Arc::new(EmbeddingExtractor {
//!     embeddings: Mutex::new(Vec::new()),
//! });
//! model.register_hook(extractor.clone());
//! ```

use std::sync::Arc;

/// Trait for inference hooks that receive intermediate model states.
///
/// Implement this trait to create custom hooks that can observe or extract
/// data during model inference.
pub trait InferenceHook: Send + Sync {
    /// Called after the final transformer layer, before the LM head.
    ///
    /// This is the primary hook point for extracting sentence/token embeddings.
    ///
    /// # Arguments
    /// * `hidden_states` - Flattened hidden states: [batch * seq_len * hidden_size]
    /// * `shape` - Shape as [batch_size, seq_len, hidden_size]
    /// * `layer_idx` - Index of the last transformer layer
    fn on_last_hidden_state(&self, hidden_states: &[f32], shape: [usize; 3], layer_idx: usize);

    /// Called after each transformer layer output (optional).
    ///
    /// Default implementation does nothing. Override to monitor layer-by-layer
    /// activations for debugging or analysis.
    ///
    /// # Arguments
    /// * `hidden_states` - Flattened hidden states: [batch * seq_len * hidden_size]
    /// * `shape` - Shape as [batch_size, seq_len, hidden_size]
    /// * `layer_idx` - Index of the current layer
    fn on_layer_output(&self, _hidden_states: &[f32], _shape: [usize; 3], _layer_idx: usize) {}

    /// Called after logits computation (optional).
    ///
    /// Default implementation does nothing. Override to observe raw logits
    /// before sampling.
    ///
    /// # Arguments
    /// * `logits` - Flattened logits: [batch * seq_len * vocab_size] or [batch * vocab_size]
    /// * `shape` - Shape as [batch_size (or batch*seq), vocab_size]
    fn on_logits(&self, _logits: &[f32], _shape: [usize; 2]) {}

    /// Hook priority for ordering multiple hooks (higher = earlier).
    ///
    /// Default is 0. Hooks with higher priority are called first.
    fn priority(&self) -> i32 {
        0
    }

    /// Optional name for debugging/logging purposes.
    fn name(&self) -> &str {
        "unnamed_hook"
    }
}

/// Manager for inference hooks.
///
/// Maintains a list of registered hooks and provides methods to fire events
/// to all hooks in priority order.
pub struct HookManager {
    hooks: Vec<Arc<dyn InferenceHook>>,
    enabled: bool,
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HookManager {
    /// Create a new empty hook manager.
    pub fn new() -> Self {
        Self {
            hooks: Vec::new(),
            enabled: true,
        }
    }

    /// Register a new hook.
    ///
    /// Hooks are maintained in priority order (highest first).
    pub fn register(&mut self, hook: Arc<dyn InferenceHook>) {
        self.hooks.push(hook);
        // Sort by priority (descending)
        self.hooks.sort_by(|a, b| b.priority().cmp(&a.priority()));
    }

    /// Unregister all hooks.
    pub fn unregister_all(&mut self) {
        self.hooks.clear();
    }

    /// Unregister a hook by name.
    ///
    /// Returns true if a hook was removed.
    pub fn unregister_by_name(&mut self, name: &str) -> bool {
        let initial_len = self.hooks.len();
        self.hooks.retain(|h| h.name() != name);
        self.hooks.len() < initial_len
    }

    /// Enable or disable all hooks.
    ///
    /// When disabled, fire_* methods do nothing.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Check if hooks are enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Check if no hooks are registered.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Fire the last hidden state event to all hooks.
    ///
    /// Called by the model after the final transformer layer.
    pub fn fire_last_hidden(&self, data: &[f32], shape: [usize; 3], layer_idx: usize) {
        if !self.enabled || self.hooks.is_empty() {
            return;
        }
        for hook in &self.hooks {
            hook.on_last_hidden_state(data, shape, layer_idx);
        }
    }

    /// Fire the layer output event to all hooks.
    ///
    /// Called by the model after each transformer layer.
    pub fn fire_layer_output(&self, data: &[f32], shape: [usize; 3], layer_idx: usize) {
        if !self.enabled || self.hooks.is_empty() {
            return;
        }
        for hook in &self.hooks {
            hook.on_layer_output(data, shape, layer_idx);
        }
    }

    /// Fire the logits event to all hooks.
    ///
    /// Called by the model after computing logits.
    pub fn fire_logits(&self, data: &[f32], shape: [usize; 2]) {
        if !self.enabled || self.hooks.is_empty() {
            return;
        }
        for hook in &self.hooks {
            hook.on_logits(data, shape);
        }
    }

    /// Get list of registered hook names.
    pub fn hook_names(&self) -> Vec<&str> {
        self.hooks.iter().map(|h| h.name()).collect()
    }
}

// ============================================================================
// Common hook implementations
// ============================================================================

/// A simple hook that collects last hidden states.
///
/// Useful for extracting embeddings without implementing a custom hook.
pub struct LastHiddenStateCollector {
    states: std::sync::Mutex<Vec<CollectedState>>,
    max_states: usize,
}

/// A collected hidden state with metadata.
#[derive(Clone, Debug)]
pub struct CollectedState {
    /// The hidden state data (flattened).
    pub data: Vec<f32>,
    /// Shape as [batch_size, seq_len, hidden_size].
    pub shape: [usize; 3],
    /// Layer index.
    pub layer_idx: usize,
}

impl LastHiddenStateCollector {
    /// Create a new collector with optional maximum capacity.
    ///
    /// If max_states is 0, unlimited states are collected.
    pub fn new(max_states: usize) -> Self {
        Self {
            states: std::sync::Mutex::new(Vec::new()),
            max_states,
        }
    }

    /// Get all collected states.
    pub fn get_states(&self) -> Vec<CollectedState> {
        self.states.lock().unwrap().clone()
    }

    /// Clear all collected states.
    pub fn clear(&self) {
        self.states.lock().unwrap().clear();
    }

    /// Get the number of collected states.
    pub fn len(&self) -> usize {
        self.states.lock().unwrap().len()
    }

    /// Check if no states have been collected.
    pub fn is_empty(&self) -> bool {
        self.states.lock().unwrap().is_empty()
    }

    /// Get the last collected state, if any.
    pub fn last(&self) -> Option<CollectedState> {
        self.states.lock().unwrap().last().cloned()
    }

    /// Extract embeddings from collected states.
    ///
    /// For each state, extracts the hidden vector at the specified position:
    /// - `position = -1` (default): last token of each sequence
    /// - `position = 0`: first token (often [CLS] token)
    /// - `position > 0`: specific position
    pub fn extract_embeddings(&self, position: i32) -> Vec<Vec<f32>> {
        let states = self.states.lock().unwrap();
        states
            .iter()
            .map(|state| {
                let [batch_size, seq_len, hidden_size] = state.shape;
                let pos = if position < 0 {
                    seq_len.saturating_sub((-position) as usize)
                } else {
                    (position as usize).min(seq_len.saturating_sub(1))
                };

                // Extract embeddings for each batch element
                let mut embeddings = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let offset = (b * seq_len + pos) * hidden_size;
                    let end = offset + hidden_size;
                    if end <= state.data.len() {
                        embeddings.extend_from_slice(&state.data[offset..end]);
                    }
                }
                embeddings
            })
            .collect()
    }
}

impl InferenceHook for LastHiddenStateCollector {
    fn on_last_hidden_state(&self, hidden_states: &[f32], shape: [usize; 3], layer_idx: usize) {
        let mut states = self.states.lock().unwrap();

        // Respect max capacity
        if self.max_states > 0 && states.len() >= self.max_states {
            states.remove(0);
        }

        states.push(CollectedState {
            data: hidden_states.to_vec(),
            shape,
            layer_idx,
        });
    }

    fn name(&self) -> &str {
        "last_hidden_state_collector"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHook {
        call_count: std::sync::atomic::AtomicUsize,
        name: String,
        priority: i32,
    }

    impl TestHook {
        fn new(name: &str, priority: i32) -> Self {
            Self {
                call_count: std::sync::atomic::AtomicUsize::new(0),
                name: name.to_string(),
                priority,
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl InferenceHook for TestHook {
        fn on_last_hidden_state(&self, _: &[f32], _: [usize; 3], _: usize) {
            self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }
    }

    #[test]
    fn test_hook_manager_basic() {
        let mut manager = HookManager::new();
        assert!(manager.is_empty());

        let hook1 = Arc::new(TestHook::new("hook1", 0));
        let hook2 = Arc::new(TestHook::new("hook2", 0));

        manager.register(hook1.clone());
        manager.register(hook2.clone());

        assert_eq!(manager.len(), 2);
        assert!(!manager.is_empty());

        // Fire event
        manager.fire_last_hidden(&[1.0, 2.0, 3.0], [1, 1, 3], 11);

        assert_eq!(hook1.calls(), 1);
        assert_eq!(hook2.calls(), 1);

        // Fire again
        manager.fire_last_hidden(&[1.0, 2.0, 3.0], [1, 1, 3], 11);

        assert_eq!(hook1.calls(), 2);
        assert_eq!(hook2.calls(), 2);
    }

    #[test]
    fn test_hook_manager_priority() {
        let mut manager = HookManager::new();

        let hook_low = Arc::new(TestHook::new("low", 0));
        let hook_high = Arc::new(TestHook::new("high", 100));
        let hook_mid = Arc::new(TestHook::new("mid", 50));

        // Register in random order
        manager.register(hook_low);
        manager.register(hook_high);
        manager.register(hook_mid);

        // Check order by priority (descending)
        let names = manager.hook_names();
        assert_eq!(names, vec!["high", "mid", "low"]);
    }

    #[test]
    fn test_hook_manager_unregister() {
        let mut manager = HookManager::new();

        let hook1 = Arc::new(TestHook::new("hook1", 0));
        let hook2 = Arc::new(TestHook::new("hook2", 0));

        manager.register(hook1);
        manager.register(hook2);

        assert_eq!(manager.len(), 2);

        assert!(manager.unregister_by_name("hook1"));
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.hook_names(), vec!["hook2"]);

        assert!(!manager.unregister_by_name("nonexistent"));

        manager.unregister_all();
        assert!(manager.is_empty());
    }

    #[test]
    fn test_hook_manager_enabled() {
        let mut manager = HookManager::new();
        let hook = Arc::new(TestHook::new("test", 0));
        manager.register(hook.clone());

        // Enabled by default
        assert!(manager.is_enabled());
        manager.fire_last_hidden(&[1.0], [1, 1, 1], 0);
        assert_eq!(hook.calls(), 1);

        // Disable
        manager.set_enabled(false);
        assert!(!manager.is_enabled());
        manager.fire_last_hidden(&[1.0], [1, 1, 1], 0);
        assert_eq!(hook.calls(), 1); // No change

        // Re-enable
        manager.set_enabled(true);
        manager.fire_last_hidden(&[1.0], [1, 1, 1], 0);
        assert_eq!(hook.calls(), 2);
    }

    #[test]
    fn test_last_hidden_state_collector() {
        let collector = Arc::new(LastHiddenStateCollector::new(3));

        assert!(collector.is_empty());

        // Simulate collecting states
        collector.on_last_hidden_state(&[1.0, 2.0, 3.0, 4.0], [1, 2, 2], 11);
        collector.on_last_hidden_state(&[5.0, 6.0, 7.0, 8.0], [1, 2, 2], 11);

        assert_eq!(collector.len(), 2);

        let states = collector.get_states();
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(states[0].shape, [1, 2, 2]);

        // Test max capacity
        collector.on_last_hidden_state(&[9.0, 10.0, 11.0, 12.0], [1, 2, 2], 11);
        collector.on_last_hidden_state(&[13.0, 14.0, 15.0, 16.0], [1, 2, 2], 11);

        // Should only keep last 3
        assert_eq!(collector.len(), 3);
        let states = collector.get_states();
        assert_eq!(states[0].data, vec![5.0, 6.0, 7.0, 8.0]); // First one was dropped
    }

    #[test]
    fn test_extract_embeddings() {
        let collector = LastHiddenStateCollector::new(0);

        // Shape: batch=1, seq_len=3, hidden=2
        // Data: [[h0_0, h0_1], [h1_0, h1_1], [h2_0, h2_1]]
        collector.on_last_hidden_state(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 3, 2],
            11,
        );

        // Extract last token (position -1)
        let embeddings = collector.extract_embeddings(-1);
        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0], vec![5.0, 6.0]); // Last token [h2_0, h2_1]

        // Extract first token (position 0)
        let embeddings = collector.extract_embeddings(0);
        assert_eq!(embeddings[0], vec![1.0, 2.0]); // First token [h0_0, h0_1]

        // Extract middle token (position 1)
        let embeddings = collector.extract_embeddings(1);
        assert_eq!(embeddings[0], vec![3.0, 4.0]); // Middle token [h1_0, h1_1]
    }

    #[test]
    fn test_collector_clear() {
        let collector = LastHiddenStateCollector::new(0);

        collector.on_last_hidden_state(&[1.0, 2.0], [1, 1, 2], 0);
        collector.on_last_hidden_state(&[3.0, 4.0], [1, 1, 2], 0);

        assert_eq!(collector.len(), 2);

        collector.clear();
        assert!(collector.is_empty());
    }
}
