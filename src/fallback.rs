//! Runtime fallback embedder that automatically switches from GPU to CPU on failure.
//!
//! This module provides `FallbackEmbedder` which wraps `EmbedderHandle` and adds:
//! - Automatic GPU→CPU fallback on OOM or other GPU errors
//! - Lazy CPU backend initialization (only when needed)
//! - Transparent API compatible with `EmbedderHandle`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use gllm::{FallbackEmbedder, Device};
//!
//! // Create with auto device selection (prefer GPU, fallback to CPU)
//! let embedder = FallbackEmbedder::new("bge-small-en").await?;
//!
//! // Embed text - automatically falls back to CPU if GPU fails
//! let vector = embedder.embed("Hello world").await?;
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::handle::EmbedderHandle;
use crate::types::{Device, Error, GraphCodeInput, Result};

#[cfg(feature = "tokio")]
use tokio::sync::Mutex;

#[cfg(not(feature = "tokio"))]
use std::sync::Mutex;

/// Maximum consecutive GPU failures before switching to CPU-only mode.
const MAX_GPU_FAILURES: usize = 3;

/// Embedder with automatic runtime fallback from GPU to CPU.
///
/// This wrapper provides transparent fallback handling:
/// 1. Starts with GPU (if available via Device::Auto)
/// 2. On GPU failure (OOM, driver error, etc.), retries with CPU
/// 3. After MAX_GPU_FAILURES, permanently switches to CPU mode
///
/// The CPU embedder is lazily initialized only when fallback is needed,
/// avoiding unnecessary resource usage when GPU works fine.
pub struct FallbackEmbedder {
    /// Primary embedder (GPU or Auto)
    primary: Arc<Mutex<EmbedderHandle>>,
    /// CPU fallback embedder (lazy initialized)
    fallback: Arc<Mutex<Option<EmbedderHandle>>>,
    /// Model name for creating fallback
    model_name: String,
    /// Whether GPU mode is disabled (after repeated failures)
    gpu_disabled: Arc<AtomicBool>,
    /// Consecutive GPU failure counter
    failure_count: Arc<AtomicUsize>,
}

impl FallbackEmbedder {
    /// Create a new fallback embedder with auto device selection.
    ///
    /// Prefers GPU, automatically falls back to CPU on failure.
    #[cfg(feature = "tokio")]
    pub async fn new(model: &str) -> Result<Self> {
        Self::new_with_device(model, Device::Auto).await
    }

    /// Create a fallback embedder with specified primary device.
    ///
    /// If primary device fails, will fall back to CPU.
    #[cfg(feature = "tokio")]
    pub async fn new_with_device(model: &str, device: Device) -> Result<Self> {
        let primary = EmbedderHandle::new_with_device(model, device).await?;

        Ok(Self {
            primary: Arc::new(Mutex::new(primary)),
            fallback: Arc::new(Mutex::new(None)),
            model_name: model.to_string(),
            gpu_disabled: Arc::new(AtomicBool::new(false)),
            failure_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Create a new fallback embedder (sync version).
    #[cfg(not(feature = "tokio"))]
    pub fn new(model: &str) -> Result<Self> {
        Self::new_with_device(model, Device::Auto)
    }

    /// Create a fallback embedder with specified primary device (sync version).
    #[cfg(not(feature = "tokio"))]
    pub fn new_with_device(model: &str, device: Device) -> Result<Self> {
        let primary = EmbedderHandle::new_with_device(model, device)?;

        Ok(Self {
            primary: Arc::new(Mutex::new(primary)),
            fallback: Arc::new(Mutex::new(None)),
            model_name: model.to_string(),
            gpu_disabled: Arc::new(AtomicBool::new(false)),
            failure_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Check if GPU mode is currently disabled.
    pub fn is_gpu_disabled(&self) -> bool {
        self.gpu_disabled.load(Ordering::Relaxed)
    }

    /// Get the current failure count.
    pub fn failure_count(&self) -> usize {
        self.failure_count.load(Ordering::Relaxed)
    }

    /// Reset failure count and re-enable GPU mode.
    ///
    /// Call this after GPU becomes available again (e.g., after freeing VRAM).
    pub fn reset_gpu(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        self.gpu_disabled.store(false, Ordering::Relaxed);
    }

    /// Get or create the CPU fallback embedder.
    #[cfg(feature = "tokio")]
    async fn get_or_create_fallback(&self) -> Result<()> {
        let mut guard = self.fallback.lock().await;
        if guard.is_none() {
            eprintln!("🔄 gllm: Initializing CPU fallback embedder...");
            let cpu_embedder = EmbedderHandle::new_with_device(&self.model_name, Device::Cpu).await?;
            eprintln!("✅ gllm: CPU fallback ready");
            *guard = Some(cpu_embedder);
        }
        Ok(())
    }

    #[cfg(not(feature = "tokio"))]
    fn get_or_create_fallback(&self) -> Result<()> {
        let mut guard = self.fallback.lock().map_err(|e| {
            Error::InternalError(format!("Failed to lock fallback mutex: {}", e))
        })?;
        if guard.is_none() {
            eprintln!("🔄 gllm: Initializing CPU fallback embedder...");
            let cpu_embedder = EmbedderHandle::new_with_device(&self.model_name, Device::Cpu)?;
            eprintln!("✅ gllm: CPU fallback ready");
            *guard = Some(cpu_embedder);
        }
        Ok(())
    }

    /// Embed a single text with automatic fallback.
    #[cfg(feature = "tokio")]
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // If GPU is disabled, use fallback directly
        if self.gpu_disabled.load(Ordering::Relaxed) {
            self.get_or_create_fallback().await?;
            let guard = self.fallback.lock().await;
            return guard.as_ref()
                .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                .embed(text).await;
        }

        // Try primary (GPU)
        let primary = self.primary.lock().await;
        match primary.embed(text).await {
            Ok(result) => {
                // Success - reset failure count
                self.failure_count.store(0, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) if e.is_oom() => {
                drop(primary); // Release lock before fallback
                self.handle_gpu_failure(&e).await?;

                // Retry with fallback
                let guard = self.fallback.lock().await;
                guard.as_ref()
                    .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                    .embed(text).await
            }
            Err(e) => Err(e),
        }
    }

    /// Embed multiple texts in batch with automatic fallback.
    #[cfg(feature = "tokio")]
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        // If GPU is disabled, use fallback directly
        if self.gpu_disabled.load(Ordering::Relaxed) {
            self.get_or_create_fallback().await?;
            let guard = self.fallback.lock().await;
            return guard.as_ref()
                .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                .embed_batch(texts).await;
        }

        // Try primary (GPU)
        let primary = self.primary.lock().await;
        match primary.embed_batch(texts).await {
            Ok(result) => {
                self.failure_count.store(0, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) if e.is_oom() => {
                drop(primary);
                self.handle_gpu_failure(&e).await?;

                let guard = self.fallback.lock().await;
                guard.as_ref()
                    .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                    .embed_batch(texts).await
            }
            Err(e) => Err(e),
        }
    }

    /// Embed graph code inputs with automatic fallback.
    #[cfg(feature = "tokio")]
    pub async fn embed_graph_batch(&self, inputs: Vec<GraphCodeInput>) -> Result<Vec<Vec<f32>>> {
        // If GPU is disabled, use fallback directly
        if self.gpu_disabled.load(Ordering::Relaxed) {
            self.get_or_create_fallback().await?;
            let guard = self.fallback.lock().await;
            return guard.as_ref()
                .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                .embed_graph_batch(inputs).await;
        }

        // Try primary (GPU)
        let primary = self.primary.lock().await;
        match primary.embed_graph_batch(inputs.clone()).await {
            Ok(result) => {
                self.failure_count.store(0, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) if e.is_oom() => {
                drop(primary);
                self.handle_gpu_failure(&e).await?;

                let guard = self.fallback.lock().await;
                guard.as_ref()
                    .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                    .embed_graph_batch(inputs).await
            }
            Err(e) => Err(e),
        }
    }

    /// Handle GPU failure: increment counter, possibly disable GPU, init fallback.
    #[cfg(feature = "tokio")]
    async fn handle_gpu_failure(&self, error: &Error) -> Result<()> {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        eprintln!(
            "⚠️ gllm: GPU failure {}/{}: {}",
            failures, MAX_GPU_FAILURES, error
        );

        if failures >= MAX_GPU_FAILURES {
            eprintln!("🚫 gllm: GPU disabled after {} failures, switching to CPU-only mode", failures);
            self.gpu_disabled.store(true, Ordering::Relaxed);
        }

        // Initialize fallback
        self.get_or_create_fallback().await
    }

    // ==================== Sync versions ====================

    #[cfg(not(feature = "tokio"))]
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        if self.gpu_disabled.load(Ordering::Relaxed) {
            self.get_or_create_fallback()?;
            let guard = self.fallback.lock().map_err(|e| {
                Error::InternalError(format!("Failed to lock fallback mutex: {}", e))
            })?;
            return guard.as_ref()
                .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                .embed(text);
        }

        let primary = self.primary.lock().map_err(|e| {
            Error::InternalError(format!("Failed to lock primary mutex: {}", e))
        })?;

        match primary.embed(text) {
            Ok(result) => {
                self.failure_count.store(0, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) if e.is_oom() => {
                drop(primary);
                self.handle_gpu_failure_sync(&e)?;

                let guard = self.fallback.lock().map_err(|e| {
                    Error::InternalError(format!("Failed to lock fallback mutex: {}", e))
                })?;
                guard.as_ref()
                    .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                    .embed(text)
            }
            Err(e) => Err(e),
        }
    }

    #[cfg(not(feature = "tokio"))]
    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if self.gpu_disabled.load(Ordering::Relaxed) {
            self.get_or_create_fallback()?;
            let guard = self.fallback.lock().map_err(|e| {
                Error::InternalError(format!("Failed to lock fallback mutex: {}", e))
            })?;
            return guard.as_ref()
                .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                .embed_batch(texts);
        }

        let primary = self.primary.lock().map_err(|e| {
            Error::InternalError(format!("Failed to lock primary mutex: {}", e))
        })?;

        match primary.embed_batch(texts) {
            Ok(result) => {
                self.failure_count.store(0, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) if e.is_oom() => {
                drop(primary);
                self.handle_gpu_failure_sync(&e)?;

                let guard = self.fallback.lock().map_err(|e| {
                    Error::InternalError(format!("Failed to lock fallback mutex: {}", e))
                })?;
                guard.as_ref()
                    .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                    .embed_batch(texts)
            }
            Err(e) => Err(e),
        }
    }

    #[cfg(not(feature = "tokio"))]
    pub fn embed_graph_batch(&self, inputs: Vec<GraphCodeInput>) -> Result<Vec<Vec<f32>>> {
        if self.gpu_disabled.load(Ordering::Relaxed) {
            self.get_or_create_fallback()?;
            let guard = self.fallback.lock().map_err(|e| {
                Error::InternalError(format!("Failed to lock fallback mutex: {}", e))
            })?;
            return guard.as_ref()
                .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                .embed_graph_batch(inputs);
        }

        let primary = self.primary.lock().map_err(|e| {
            Error::InternalError(format!("Failed to lock primary mutex: {}", e))
        })?;

        match primary.embed_graph_batch(inputs.clone()) {
            Ok(result) => {
                self.failure_count.store(0, Ordering::Relaxed);
                Ok(result)
            }
            Err(e) if e.is_oom() => {
                drop(primary);
                self.handle_gpu_failure_sync(&e)?;

                let guard = self.fallback.lock().map_err(|e| {
                    Error::InternalError(format!("Failed to lock fallback mutex: {}", e))
                })?;
                guard.as_ref()
                    .ok_or_else(|| Error::InternalError("Fallback not initialized".into()))?
                    .embed_graph_batch(inputs)
            }
            Err(e) => Err(e),
        }
    }

    #[cfg(not(feature = "tokio"))]
    fn handle_gpu_failure_sync(&self, error: &Error) -> Result<()> {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        eprintln!(
            "⚠️ gllm: GPU failure {}/{}: {}",
            failures, MAX_GPU_FAILURES, error
        );

        if failures >= MAX_GPU_FAILURES {
            eprintln!("🚫 gllm: GPU disabled after {} failures, switching to CPU-only mode", failures);
            self.gpu_disabled.store(true, Ordering::Relaxed);
        }

        self.get_or_create_fallback()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn test_fallback_embedder_creation() {
        // Just test creation - actual embedding tested in handle.rs
        let embedder = FallbackEmbedder::new("bge-small-en").await;
        assert!(embedder.is_ok());

        let embedder = embedder.unwrap();
        assert!(!embedder.is_gpu_disabled());
        assert_eq!(embedder.failure_count(), 0);
    }

    #[cfg(feature = "tokio")]
    #[tokio::test]
    async fn test_reset_gpu() {
        let embedder = FallbackEmbedder::new("bge-small-en").await.unwrap();

        // Simulate failures
        embedder.failure_count.store(5, Ordering::Relaxed);
        embedder.gpu_disabled.store(true, Ordering::Relaxed);

        assert!(embedder.is_gpu_disabled());
        assert_eq!(embedder.failure_count(), 5);

        // Reset
        embedder.reset_gpu();

        assert!(!embedder.is_gpu_disabled());
        assert_eq!(embedder.failure_count(), 0);
    }
}
