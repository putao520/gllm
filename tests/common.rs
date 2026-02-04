//! Common test utilities for gllm tests.

use std::path::PathBuf;
use gllm::loader::{Loader, LoaderConfig, Result};
use gllm::registry;

/// Test model files helper for unit tests.
///
/// This provides a simple way to create loaders for test models
/// using the model registry.
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
    /// The alias is looked up in the model registry, and the loader
    /// is configured to download from HuggingFace (with ModelScope fallback).
    pub fn loader(&self, alias: &str) -> Result<Loader> {
        // Look up the manifest to get the actual HF repo
        let manifest = registry::lookup(alias)
            .or_else(|| {
                // Try with common prefixes
                registry::lookup(&format!("qwen2.5-{}", alias))
                    .or_else(|| registry::lookup(&format!("qwen3-{}", alias)))
            });

        let repo = if let Some(m) = manifest {
            m.hf_repo
        } else {
            // Default to using alias as repo name
            alias
        };

        let config = LoaderConfig {
            cache_dir: Some(self.base_dir.clone()),
            ..Default::default()
        };

        // Try HuggingFace first, fallback to ModelScope
        Loader::from_hf_with_config(repo, config)
            .or_else(|_| Loader::from_ms(repo))
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
