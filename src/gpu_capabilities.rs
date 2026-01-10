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

/// Calculate recommended batch size based on VRAM and device type.
fn calculate_default_batch_size(gpu_type: GpuType, vram_mb: u64) -> usize {
    match gpu_type {
        GpuType::Cpu => 4, // CPU: small batches to avoid memory pressure
        GpuType::Integrated => {
            // Integrated GPU shares system RAM, be conservative
            if vram_mb >= 2048 {
                8
            } else {
                4
            }
        }
        GpuType::Discrete | GpuType::Virtual => {
            // Discrete GPU: scale with VRAM
            // Base: 500MB, Per-item: ~50MB, Headroom: 30%
            let usable_vram = (vram_mb as f64 * 0.7) as u64;
            let available_for_batch = usable_vram.saturating_sub(500);
            let batch_size = (available_for_batch / 50).max(4).min(128) as usize;

            // Clamp to reasonable range based on VRAM
            match vram_mb {
                0..=2048 => batch_size.min(8),      // <=2GB: max 8
                2049..=4096 => batch_size.min(16),  // 2-4GB: max 16
                4097..=8192 => batch_size.min(32),  // 4-8GB: max 32
                8193..=16384 => batch_size.min(64), // 8-16GB: max 64
                _ => batch_size.min(128),           // >16GB: max 128
            }
        }
        GpuType::Unknown => 8, // Unknown: moderate default
    }
}

/// Internal implementation of GPU detection.
fn detect_gpu_capabilities_impl() -> GpuCapabilities {
    // Check if test mode is enabled (skip GPU detection)
    if std::env::var("GLLM_TEST_MODE").is_ok() {
        return GpuCapabilities::default();
    }

    // Try wgpu-based detection (only when wgpu-detect feature is enabled)
    #[cfg(feature = "wgpu-detect")]
    {
        match detect_wgpu_basic() {
            Ok(caps) => return caps,
            Err(e) => {
                log::debug!("gllm: GPU detection failed: {}", e);
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

    // Get VRAM estimate from device name
    let vram_mb = estimate_vram_from_name(&info.name);

    let recommended_batch_size = calculate_default_batch_size(gpu_type, vram_mb);

    let backend_name = match gpu_type {
        GpuType::Cpu => "ndarray",
        _ => "wgpu",
    };

    Ok(GpuCapabilities {
        gpu_type,
        name: info.name.clone(),
        vram_mb,
        recommended_batch_size,
        gpu_available: gpu_type != GpuType::Cpu,
        backend_name,
    })
}

/// Estimate VRAM from GPU name (heuristic).
/// Returns 0 if unknown.
fn estimate_vram_from_name(name: &str) -> u64 {
    let name_lower = name.to_lowercase();

    // NVIDIA RTX 40 series
    if name_lower.contains("4090") {
        return 24576;
    }
    if name_lower.contains("4080") {
        return 16384;
    }
    if name_lower.contains("4070 ti super") {
        return 16384;
    }
    if name_lower.contains("4070 ti") {
        return 12288;
    }
    if name_lower.contains("4070 super") {
        return 12288;
    }
    if name_lower.contains("4070") {
        return 12288;
    }
    if name_lower.contains("4060 ti") {
        return 8192;
    }
    if name_lower.contains("4060") {
        return 8192;
    }

    // NVIDIA RTX 30 series
    if name_lower.contains("3090 ti") {
        return 24576;
    }
    if name_lower.contains("3090") {
        return 24576;
    }
    if name_lower.contains("3080 ti") {
        return 12288;
    }
    if name_lower.contains("3080") {
        return 10240;
    }
    if name_lower.contains("3070 ti") {
        return 8192;
    }
    if name_lower.contains("3070") {
        return 8192;
    }
    if name_lower.contains("3060 ti") {
        return 8192;
    }
    if name_lower.contains("3060") {
        return 12288;
    }

    // NVIDIA RTX 20 series
    if name_lower.contains("2080 ti") {
        return 11264;
    }
    if name_lower.contains("2080 super") {
        return 8192;
    }
    if name_lower.contains("2080") {
        return 8192;
    }
    if name_lower.contains("2070 super") {
        return 8192;
    }
    if name_lower.contains("2070") {
        return 8192;
    }
    if name_lower.contains("2060 super") {
        return 8192;
    }
    if name_lower.contains("2060") {
        return 6144;
    }

    // NVIDIA GTX 16 series
    if name_lower.contains("1660 ti") {
        return 6144;
    }
    if name_lower.contains("1660 super") {
        return 6144;
    }
    if name_lower.contains("1660") {
        return 6144;
    }
    if name_lower.contains("1650 super") {
        return 4096;
    }
    if name_lower.contains("1650") {
        return 4096;
    }

    // NVIDIA GTX 10 series
    if name_lower.contains("1080 ti") {
        return 11264;
    }
    if name_lower.contains("1080") {
        return 8192;
    }
    if name_lower.contains("1070 ti") {
        return 8192;
    }
    if name_lower.contains("1070") {
        return 8192;
    }
    if name_lower.contains("1060") {
        return 6144;
    }
    if name_lower.contains("1050 ti") {
        return 4096;
    }
    if name_lower.contains("1050") {
        return 2048;
    }

    // NVIDIA RTX 50 series (upcoming)
    if name_lower.contains("5090") {
        return 32768;
    }
    if name_lower.contains("5080") {
        return 16384;
    }

    // AMD RX 7000 series
    if name_lower.contains("7900 xtx") {
        return 24576;
    }
    if name_lower.contains("7900 xt") {
        return 20480;
    }
    if name_lower.contains("7900 gre") {
        return 16384;
    }
    if name_lower.contains("7800 xt") {
        return 16384;
    }
    if name_lower.contains("7700 xt") {
        return 12288;
    }
    if name_lower.contains("7600 xt") {
        return 16384;
    }
    if name_lower.contains("7600") {
        return 8192;
    }

    // AMD RX 6000 series
    if name_lower.contains("6950 xt") {
        return 16384;
    }
    if name_lower.contains("6900 xt") {
        return 16384;
    }
    if name_lower.contains("6800 xt") {
        return 16384;
    }
    if name_lower.contains("6800") {
        return 16384;
    }
    if name_lower.contains("6750 xt") {
        return 12288;
    }
    if name_lower.contains("6700 xt") {
        return 12288;
    }
    if name_lower.contains("6700") {
        return 10240;
    }
    if name_lower.contains("6650 xt") {
        return 8192;
    }
    if name_lower.contains("6600 xt") {
        return 8192;
    }
    if name_lower.contains("6600") {
        return 8192;
    }

    // Intel Arc
    if name_lower.contains("a770") {
        return 16384;
    }
    if name_lower.contains("a750") {
        return 8192;
    }
    if name_lower.contains("a580") {
        return 8192;
    }
    if name_lower.contains("a380") {
        return 6144;
    }
    if name_lower.contains("a310") {
        return 4096;
    }

    // Intel integrated (approximate shared memory)
    if name_lower.contains("intel")
        && (name_lower.contains("uhd") || name_lower.contains("iris") || name_lower.contains("xe"))
    {
        return 2048; // Shared memory estimate
    }

    // AMD APU (approximate shared memory)
    if name_lower.contains("radeon graphics")
        || name_lower.contains("vega")
        || name_lower.contains("radeon 780m")
        || name_lower.contains("radeon 760m")
    {
        return 2048; // Shared memory estimate
    }

    // Apple Silicon (M-series)
    if name_lower.contains("apple m") {
        if name_lower.contains("m3 max") || name_lower.contains("m2 max") {
            return 32768; // Up to 96GB unified, assume 32GB usable for GPU
        }
        if name_lower.contains("m3 pro") || name_lower.contains("m2 pro") {
            return 16384;
        }
        return 8192; // Base M-series
    }

    // Unknown - return conservative default
    4096
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

    #[test]
    fn test_vram_estimation_nvidia() {
        assert_eq!(estimate_vram_from_name("NVIDIA GeForce RTX 4090"), 24576);
        assert_eq!(estimate_vram_from_name("NVIDIA GeForce RTX 3080"), 10240);
        assert_eq!(estimate_vram_from_name("NVIDIA GeForce GTX 1660 Ti"), 6144);
    }

    #[test]
    fn test_vram_estimation_amd() {
        assert_eq!(estimate_vram_from_name("AMD Radeon RX 7900 XTX"), 24576);
        assert_eq!(estimate_vram_from_name("AMD Radeon RX 6800 XT"), 16384);
    }

    #[test]
    fn test_vram_estimation_intel() {
        assert_eq!(estimate_vram_from_name("Intel Arc A770"), 16384);
        assert_eq!(estimate_vram_from_name("Intel UHD Graphics 770"), 2048);
    }

    #[test]
    fn test_default_batch_size_calculation() {
        assert_eq!(calculate_default_batch_size(GpuType::Cpu, 0), 4);
        assert_eq!(calculate_default_batch_size(GpuType::Integrated, 2048), 8);
        assert!(calculate_default_batch_size(GpuType::Discrete, 8192) <= 32);
        assert!(calculate_default_batch_size(GpuType::Discrete, 24576) <= 128);
    }
}
