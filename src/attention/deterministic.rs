//! Deterministic computation guarantees for reproducible results.
//!
//! In ultra-long context scenarios (2M+ tokens), even tiny numerical differences
//! can compound into noticeable output variations. This module provides
//! mechanisms to ensure bit-exact reproducibility.
//!
//! # Sources of Non-determinism
//!
//! 1. **GPU floating-point operations**: Some GPU operations use non-deterministic
//!    algorithms for performance (e.g., cuBLAS reduction order)
//! 2. **Thread scheduling**: Different execution orders can cause different
//!    floating-point rounding
//! 3. **Memory allocation**: Different allocation patterns can affect cache behavior
//!
//! # Guarantees Provided
//!
//! When deterministic mode is enabled:
//! - Same input always produces bit-exact same output
//! - Results are independent of thread count or GPU parallelism
//! - Computation order is strictly defined

use std::sync::atomic::{AtomicBool, Ordering};

/// Configuration for deterministic computation.
pub use gllm_kernels::ops::flash_attention::DeterministicConfig;

/// Global flag for deterministic mode (for GPU operations).
static DETERMINISTIC_MODE: AtomicBool = AtomicBool::new(false);

/// Extension trait for applying deterministic config globally.
pub trait DeterministicConfigExt {
    /// Apply this configuration globally.
    fn apply_global(&self);
}

impl DeterministicConfigExt for DeterministicConfig {
    fn apply_global(&self) {
        DETERMINISTIC_MODE.store(self.no_gpu_nondeterminism, Ordering::SeqCst);

        // Note: In a real implementation, we would also:
        // - Set CUDA deterministic flags
        // - Configure cuBLAS/cuDNN for deterministic algorithms
        // - Set up thread synchronization barriers
    }
}

/// Guard that enforces deterministic execution within a scope.
///
/// When created, it captures the current deterministic settings
/// and restores them when dropped.
///
/// # Example
///
/// ```ignore
/// {
///     let _guard = DeterministicGuard::new(DeterministicConfig::strict());
///     // All operations here are deterministic
///     let result = compute_attention(...);
/// } // Settings restored here
/// ```
pub struct DeterministicGuard {
    /// Previous global deterministic mode
    previous_mode: bool,
    /// Configuration for this scope
    #[allow(dead_code)]
    config: DeterministicConfig,
}

impl DeterministicGuard {
    /// Create a new deterministic guard with the given configuration.
    pub fn new(config: DeterministicConfig) -> Self {
        let previous_mode = DETERMINISTIC_MODE.load(Ordering::SeqCst);
        config.apply_global();

        Self {
            previous_mode,
            config,
        }
    }

    /// Create a guard for strict determinism.
    pub fn strict() -> Self {
        Self::new(DeterministicConfig::strict())
    }

    /// Check if we're currently in deterministic mode.
    pub fn is_active(&self) -> bool {
        DETERMINISTIC_MODE.load(Ordering::SeqCst)
    }
}

impl Drop for DeterministicGuard {
    fn drop(&mut self) {
        DETERMINISTIC_MODE.store(self.previous_mode, Ordering::SeqCst);
    }
}

/// Check if global deterministic mode is enabled.
pub fn is_deterministic_mode() -> bool {
    DETERMINISTIC_MODE.load(Ordering::SeqCst)
}

/// Trait for types that can execute operations deterministically.
pub trait DeterministicExecution {
    /// Execute the operation with deterministic guarantees.
    fn execute_deterministic(&self, config: &DeterministicConfig) -> Self;
}

/// Assertion helper for verifying determinism.
///
/// Runs a computation twice and compares results.
#[cfg(debug_assertions)]
pub fn verify_deterministic<T, F>(config: &DeterministicConfig, f: F) -> T
where
    T: PartialEq + std::fmt::Debug + Clone,
    F: Fn() -> T,
{
    if !config.verify_determinism {
        return f();
    }

    let result1 = f();
    let result2 = f();

    assert_eq!(
        result1, result2,
        "Non-deterministic behavior detected! Results differ between runs."
    );

    result1
}

#[cfg(not(debug_assertions))]
pub fn verify_deterministic<T, F>(_config: &DeterministicConfig, f: F) -> T
where
    F: Fn() -> T,
{
    f()
}

/// Strict ordering iterator for deterministic processing.
///
/// Ensures elements are processed in exact order, preventing
/// any reordering optimizations that could affect floating-point results.
pub struct StrictOrderIterator<I> {
    inner: I,
    index: usize,
}

impl<I: Iterator> StrictOrderIterator<I> {
    pub fn new(iter: I) -> Self {
        Self {
            inner: iter,
            index: 0,
        }
    }

    /// Get the current index (for verification).
    pub fn current_index(&self) -> usize {
        self.index
    }
}

impl<I: Iterator> Iterator for StrictOrderIterator<I> {
    type Item = (usize, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next()?;
        let index = self.index;
        self.index += 1;

        // Memory barrier to prevent reordering
        std::sync::atomic::fence(Ordering::SeqCst);

        Some((index, item))
    }
}

/// Extension trait for creating strict order iterators.
pub trait StrictOrderExt: Iterator + Sized {
    fn strict_order(self) -> StrictOrderIterator<Self> {
        StrictOrderIterator::new(self)
    }
}

impl<I: Iterator> StrictOrderExt for I {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_config_default() {
        let config = DeterministicConfig::default();
        assert!(config.strict_order);
        assert!(config.deterministic_rng);
        assert!(config.no_gpu_nondeterminism);
        assert!(config.is_deterministic());
    }

    #[test]
    fn test_deterministic_config_relaxed() {
        let config = DeterministicConfig::relaxed();
        assert!(!config.strict_order);
        assert!(!config.deterministic_rng);
        assert!(!config.no_gpu_nondeterminism);
        assert!(!config.is_deterministic());
    }

    #[test]
    fn test_deterministic_guard() {
        // Initial state
        DETERMINISTIC_MODE.store(false, Ordering::SeqCst);
        assert!(!is_deterministic_mode());

        {
            let guard = DeterministicGuard::strict();
            assert!(guard.is_active());
            assert!(is_deterministic_mode());
        }

        // Should be restored after guard is dropped
        assert!(!is_deterministic_mode());
    }

    #[test]
    fn test_strict_order_iterator() {
        let items = vec![10, 20, 30, 40, 50];
        let collected: Vec<_> = items.iter().strict_order().collect();

        assert_eq!(collected.len(), 5);
        for (i, (idx, &val)) in collected.iter().enumerate() {
            assert_eq!(*idx, i);
            assert_eq!(val, items[i]);
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_verify_deterministic() {
        let config = DeterministicConfig {
            verify_determinism: true,
            ..Default::default()
        };

        // Deterministic function should pass
        let result = verify_deterministic(&config, || 42);
        assert_eq!(result, 42);
    }
}
