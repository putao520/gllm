//! Common test utilities for gllm tests.

use std::path::PathBuf;
use gllm::loader::{Loader, LoaderConfig, ModelSource, Result};

/// Test model files helper for unit tests.
///
/// This provides a simple way to create loaders for test models
/// without requiring the full model files to be present.
pub struct TestModelFiles {
    /// Base directory for test models
    pub base_dir: PathBuf,
}

impl TestModelFiles {
    /// Create a new TestModelFiles instance.
    ///
    /// This will use the default cache directory (`~/.gllm/models/`).
    pub fn new() -> Result<Self> {
        let base_dir = dirs::home_dir()
            .map(|p| p.join(".gllm").join("models"))
            .ok_or_else(|| std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "home directory not available",
            ))?;

        Ok(Self { base_dir })
    }

    /// Create a loader for the given model alias.
    ///
    /// The alias is looked up in the model registry.
    pub fn loader(&self, _alias: &str) -> Result<Loader> {
        // For now, create a basic loader that uses the default config
        // The actual model files will be loaded from the cache
        let config = LoaderConfig {
            cache_dir: Some(self.base_dir.clone()),
            ..Default::default()
        };

        // Note: This creates a loader without a specific model
        // The caller is responsible for setting up the correct repo/alias
        Loader::from_hf_with_config("test", config)
            .or_else(|_| Loader::from_ms("test"))
    }

    /// Get the base directory for test models.
    pub fn base_dir(&self) -> &PathBuf {
        &self.base_dir
    }
}

impl Default for TestModelFiles {
    fn default() -> Self {
        Self::new().expect("failed to create TestModelFiles")
    }
}
