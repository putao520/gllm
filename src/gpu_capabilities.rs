//! GPU capabilities detection for optimal resource allocation.
//!
//! This module provides runtime detection of GPU capabilities to enable:
//! - Optimal batch size selection based on available VRAM
//! - Device type classification (Discrete, Integrated, Virtual, CPU)
//! - Intelligent resource allocation for embedding workloads
//!
//! ## Usage
//!
//! ```rust,ignore
//! use gllm::GpuCapabilities;
//!
//! // Detect GPU capabilities (cached after first call)
//! let caps = GpuCapabilities::detect();
//!
//! println!("GPU: {} ({:?})", caps.name, caps.gpu_type);
//! println!("VRAM: {}MB", caps.vram_mb);
//! println!("Recommended batch size: {}", caps.recommended_batch_size);
//! ```

use std::sync::OnceLock;

/// Cached GPU capabilities for the current system.
static GPU_CAPABILITIES: OnceLock<GpuCapabilities> = OnceLock::new();

/// GPU device type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuType {
    /// Dedicated GPU (NVIDIA, AMD discrete)
    Discrete,
    /// Integrated GPU (Intel UHD, AMD APU)
    Integrated,
    /// Virtual GPU (cloud instances)
    Virtual,
    /// CPU fallback (no GPU available)
    Cpu,
    /// Unknown GPU type
    Unknown,
}

/// GPU capabilities detected from the system.
#[derive(Debug, Clone)]
pub struct GpuCapabilities {
    /// GPU device type
    pub gpu_type: GpuType,
    /// GPU name (e.g., "NVIDIA GeForce RTX 3080")
    pub name: String,
    /// Estimated VRAM in MB (0 if unknown or CPU)
    pub vram_mb: u64,
    /// Recommended batch size based on capabilities
    pub recommended_batch_size: usize,
    /// Whether GPU is available and working
    pub gpu_available: bool,
    /// Backend type that will be used
    pub backend_name: &'static str,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        Self {
            gpu_type: GpuType::Cpu,
            name: "CPU".to_string(),
            vram_mb: 0,
            recommended_batch_size: 4, // Conservative CPU default
            gpu_available: false,
            backend_name: "ndarray",
        }
    }
}

impl GpuCapabilities {
    /// Detect GPU capabilities from the system.
    ///
    /// Results are cached after first call for performance.
    /// This function is thread-safe.
    pub fn detect() -> &'static GpuCapabilities {
        GPU_CAPABILITIES.get_or_init(|| {
            let caps = detect_gpu_capabilities_impl();
            log::info!(
                "gllm: GPU detected: {} ({:?}, {}MB VRAM, batch_size={})",
                caps.name,
                caps.gpu_type,
                caps.vram_mb,
                caps.recommended_batch_size
            );
            caps
        })
    }

    /// Get recommended batch size for embedding operations.
    ///
    /// The batch size is calculated based on:
    /// - Available VRAM (for GPU)
    /// - Device type (discrete vs integrated)
    /// - Model memory requirements (~500MB base + ~50MB per batch item)
    pub fn recommended_batch_size(&self) -> usize {
        self.recommended_batch_size
    }

    /// Check if GPU is available.
    pub fn is_gpu_available(&self) -> bool {
        self.gpu_available
    }

    /// Calculate batch size for a specific model's memory footprint.
    ///
    /// # Arguments
    /// * `base_memory_mb` - Model's base memory usage in MB
    /// * `per_item_mb` - Memory per batch item in MB
    /// * `headroom` - Safety headroom (0.0 - 1.0, e.g., 0.3 for 30%)
    pub fn calculate_batch_size(
        &self,
        base_memory_mb: u64,
        per_item_mb: u64,
        headroom: f64,
    ) -> usize {
        if self.vram_mb == 0 || self.gpu_type == GpuType::Cpu {
            return 4; // Conservative CPU default
        }

        let usable_vram = (self.vram_mb as f64 * (1.0 - headroom.clamp(0.0, 0.9))) as u64;
        let available_for_batch = usable_vram.saturating_sub(base_memory_mb);
        let batch_size = if per_item_mb > 0 {
            (available_for_batch / per_item_mb).max(1).min(128) as usize
        } else {
            self.recommended_batch_size
        };

        // Apply device-specific limits
        match self.gpu_type {
            GpuType::Integrated => batch_size.min(16), // Shared memory
            GpuType::Discrete | GpuType::Virtual => batch_size.min(128),
            GpuType::Cpu | GpuType::Unknown => batch_size.min(8),
        }
    }
}

/// Internal implementation of GPU detection.
///
/// CRITICAL: GPU detection uses `pollster::block_on()` which can deadlock
/// when called from a tokio runtime thread. To avoid this, we spawn a
/// dedicated OS thread for GPU detection.
fn detect_gpu_capabilities_impl() -> GpuCapabilities {
    // Try wgpu-based detection (only when wgpu-detect feature is enabled)
    #[cfg(feature = "wgpu-detect")]
    {
        // CRITICAL FIX: Run GPU detection in a dedicated thread to avoid
        // pollster::block_on() deadlock with tokio runtime.
        // pollster uses a simple spin-loop that blocks the current thread,
        // which can deadlock if called from within a tokio worker thread.
        let handle = std::thread::spawn(|| detect_wgpu_basic());

        match handle.join() {
            Ok(Ok(caps)) => return caps,
            Ok(Err(e)) => {
                log::debug!("gllm: GPU detection failed: {}", e);
            }
            Err(_) => {
                log::warn!("gllm: GPU detection thread panicked, using CPU fallback");
            }
        }
    }

    #[cfg(not(feature = "wgpu-detect"))]
    {
        log::debug!("gllm: GPU detection disabled (wgpu-detect feature not enabled)");
    }

    log::debug!("gllm: Using CPU defaults");
    GpuCapabilities::default()
}

/// Detect GPU capabilities using wgpu.
///
/// This function actually requests a wgpu Device to verify GPU availability.
/// This catches OOM errors that would otherwise panic during burn/cubecl initialization.
#[cfg(feature = "wgpu-detect")]
fn detect_wgpu_basic() -> Result<GpuCapabilities, String> {
    use wgpu::{DeviceType, Instance, InstanceDescriptor};

    // Create wgpu instance
    let instance = Instance::new(&InstanceDescriptor::default());

    // Request adapter (blocking)
    let adapter_future = instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    });

    let adapter = pollster::block_on(adapter_future)
        .map_err(|e| format!("No GPU adapter found: {}", e))?;

    let info = adapter.get_info();

    // Determine GPU type
    let gpu_type = match info.device_type {
        DeviceType::DiscreteGpu => GpuType::Discrete,
        DeviceType::IntegratedGpu => GpuType::Integrated,
        DeviceType::VirtualGpu => GpuType::Virtual,
        DeviceType::Cpu => GpuType::Cpu,
        DeviceType::Other => GpuType::Unknown,
    };

    // CRITICAL: Actually request a device to verify GPU is usable
    // This catches OOM errors that would otherwise panic in cubecl-wgpu's .unwrap()
    // The panic happens in burn/cubecl initialization, which can't be caught with
    // panic=abort in release builds.
    let device_result = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("gllm-gpu-probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        },
    ));

    let gpu_available = match device_result {
        Ok(_device) => {
            log::debug!("gllm: GPU device probe successful for {}", info.name);
            gpu_type != GpuType::Cpu
        }
        Err(e) => {
            log::warn!("gllm: GPU device probe failed for {}: {} - will use CPU fallback",
                info.name, e);
            false
        }
    };

    let vram_mb = 0;
    let recommended_batch_size = if gpu_available {
        match gpu_type {
            GpuType::Integrated => 8,
            GpuType::Discrete | GpuType::Virtual => 32,
            GpuType::Unknown | GpuType::Cpu => 4,
        }
    } else {
        4
    };

    let backend_name = if gpu_available {
        "wgpu"
    } else {
        "ndarray"
    };

    Ok(GpuCapabilities {
        gpu_type,
        name: info.name.clone(),
        vram_mb,
        recommended_batch_size,
        gpu_available,
        backend_name,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_capabilities_default() {
        let caps = GpuCapabilities::default();
        assert_eq!(caps.gpu_type, GpuType::Cpu);
        assert!(!caps.gpu_available);
        assert_eq!(caps.recommended_batch_size, 4);
    }

    #[test]
    fn test_calculate_batch_size_discrete() {
        let caps = GpuCapabilities {
            gpu_type: GpuType::Discrete,
            name: "Test GPU".to_string(),
            vram_mb: 8192,
            recommended_batch_size: 32,
            gpu_available: true,
            backend_name: "wgpu",
        };

        // Test with default embedding model footprint
        let batch = caps.calculate_batch_size(500, 50, 0.3);
        assert!(batch > 0 && batch <= 128);
    }

    #[test]
    fn test_calculate_batch_size_cpu() {
        let caps = GpuCapabilities::default();
        let batch = caps.calculate_batch_size(500, 50, 0.3);
        assert_eq!(batch, 4);
    }

}
