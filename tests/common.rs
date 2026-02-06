//! Common utilities for E2E tests.

use gllm::loader::Loader;
use std::path::Path;

/// Simple helper for test loaders.
///
/// Note: Actual model caching is handled internally by the HF Hub cache
/// in ~/.gllm/models/, so we don't need to maintain our own cache.
pub struct TestModelFiles;

impl TestModelFiles {
    /// Create a new test model files helper.
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }

    /// Get a loader for the given model alias.
    ///
    /// The alias should be a valid HuggingFace model ID (e.g. "Qwen/Qwen3-0.6B").
    pub fn loader(&self, alias: &str) -> Result<Loader, Box<dyn std::error::Error>> {
        Ok(Loader::from_hf(alias)?)
    }

    /// Get the cache directory (for compatibility).
    pub fn _cache_dir(&self) -> &Path {
        std::path::Path::new("~/.gllm/models")
    }
}

impl Default for TestModelFiles {
    fn default() -> Self {
        Self::new().expect("TestModelFiles::default")
    }
}
