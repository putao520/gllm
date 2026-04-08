use super::backend_trait::{self, Backend};
use super::Element;
use crate::engine::executor::{
    AttentionTopology, BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheConfig,
    KvCacheHandle, LogitsHandle, SamplingConfig,
};
use crate::scheduler::types::{PageId, PageState, StorageKey};

// ---------------------------------------------------------------------------
// CudaBackend<E> — CUDA backend with persistent device handle
// ---------------------------------------------------------------------------

/// GPU device info captured at detection time.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Device ordinal (0-based)
    pub ordinal: usize,
    /// SM version (e.g. 80 for sm_80 / A100)
    pub sm_version: u32,
    /// Number of streaming multiprocessors
    pub sm_count: u32,
    /// Total device memory in bytes
    pub total_memory: usize,
    /// Human-readable device name
    pub name: String,
}

/// CUDA backend with GPU hardware detection and persistent device context.
///
/// When the `cuda` feature is enabled, `CudaBackend::new(ordinal)` creates a
/// persistent `CudaDevice` (owns a CUDA context) and queries the GPU hardware
/// capability profile. The device handle is shared via `Arc` so clones reuse
/// the same context.
///
/// `CudaBackend::new(ordinal)` returns `None` if:
/// - `cuda` feature is not enabled
/// - libcuda.so.1 is not loadable (no NVIDIA driver)
/// - cuInit fails
/// - No CUDA devices are present
/// - The requested ordinal is out of range
pub struct CudaBackend<E: Element = f32> {
    /// Detected GPU device info
    device_info: GpuDeviceInfo,
    /// CPU-side device profile (used for CPU JIT execution in phase 1)
    cpu_profile: gllm_kernels::dispatch::DeviceProfile,
    /// Persistent CUDA device handle (owns context + driver)
    #[cfg(feature = "cuda")]
    pub(super) device: std::sync::Arc<gllm_kernels::gpu::cuda::CudaDevice>,
    /// GPU hardware capability profile for codegen decisions
    #[cfg(feature = "cuda")]
    pub(super) gpu_profile: gllm_kernels::gpu::GpuDeviceProfile,
    /// Cached PTX code from graph compilations, keyed by a description string.
    #[cfg(feature = "cuda")]
    compiled_ptx: std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>,
    /// Host-side swap storage for evicted KV cache pages.
    #[cfg(feature = "cuda")]
    pub(super) swap_store: super::gpu_compile::GpuSwapStore,
    /// KV cache metadata for swap offset computation.
    #[cfg(feature = "cuda")]
    pub(super) kv_meta: super::gpu_compile::GpuKvMetaStore,
    /// GPU 权重常驻缓存 (REQ-ARCH-005)
    #[cfg(feature = "cuda")]
    pub(super) weight_cache: std::sync::Arc<std::sync::Mutex<super::gpu_compile::GpuWeightCache>>,

    _marker: std::marker::PhantomData<E>,
}

// Manual Debug since Mutex<HashMap> and Arc<CudaDevice> don't derive Debug cleanly
impl<E: Element> std::fmt::Debug for CudaBackend<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(feature = "cuda")]
        let ptx_count = self
            .compiled_ptx
            .lock()
            .map(|m| m.len())
            .unwrap_or(0); // LEGAL: Debug fmt 锁失败时返回 0
        #[cfg(not(feature = "cuda"))]
        let ptx_count = 0usize;
        let mut s = f.debug_struct("CudaBackend");
        s.field("device_info", &self.device_info);
        #[cfg(feature = "cuda")]
        {
            s.field("device_name", &self.device_info.name);
            s.field("gpu_compute_units", &self.gpu_profile.compute_units);
        }
        s.field("compiled_ptx_count", &ptx_count);
        s.finish()
    }
}

// Manual Clone: Arc::clone shares the device handle, compiled_ptx cache starts fresh
impl<E: Element> Clone for CudaBackend<E> {
    fn clone(&self) -> Self {
        Self {
            device_info: self.device_info.clone(),
            cpu_profile: self.cpu_profile.clone(),
            #[cfg(feature = "cuda")]
            device: std::sync::Arc::clone(&self.device),
            #[cfg(feature = "cuda")]
            gpu_profile: self.gpu_profile.clone(),
            #[cfg(feature = "cuda")]
            compiled_ptx: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "cuda")]
            swap_store: self.swap_store.clone(),
            #[cfg(feature = "cuda")]
            kv_meta: self.kv_meta.clone(),
            #[cfg(feature = "cuda")]
            weight_cache: std::sync::Arc::clone(&self.weight_cache),

            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: Element> CudaBackend<E> {
    /// Attempt to create a CUDA backend on the given device ordinal.
    ///
    /// Returns `None` if:
    /// - `cuda` feature is not enabled at compile time
    /// - NVIDIA driver is not installed (libcuda.so.1 not found)
    /// - cuInit fails
    /// - No CUDA-capable devices are present
    /// - The requested ordinal exceeds available device count
    ///
    /// When `None` is returned, the upper layer (detect_backend) automatically
    /// falls back to CpuBackend.
    #[cfg(feature = "cuda")]
    pub fn new(device: usize) -> Option<Self> {
        // Step 1: Try to load the CUDA driver library
        let driver = match gllm_kernels::gpu::cuda::CudaDriver::load() {
            Ok(d) => d,
            Err(e) => {
                log::debug!("CUDA driver not available: {e}");
                return None;
            }
        };

        // Step 2: Initialize the driver
        if let Err(e) = driver.init() {
            log::debug!("CUDA driver init failed: {e}");
            return None;
        }

        // Step 3: Check device count
        let count = match driver.device_count() {
            Ok(c) => c,
            Err(e) => {
                log::debug!("CUDA device_count failed: {e}");
                return None;
            }
        };
        if count <= 0 || device >= count as usize {
            return None;
        }

        // Step 4: Create a persistent CudaDevice (owns context + driver handle)
        let driver = std::sync::Arc::new(driver);
        let cuda_device =
            gllm_kernels::gpu::cuda::CudaDevice::new(std::sync::Arc::clone(&driver), device as i32)
                .ok()?;

        // Step 5: Query GPU hardware capability profile
        let gpu_profile = cuda_device.gpu_profile().ok()?;

        // Step 6: Build GpuDeviceInfo from the CudaDevice properties
        use gllm_kernels::gpu::GpuDevice;
        let total_memory = cuda_device.total_memory();
        let sm_version = cuda_device.sm_version();
        let sm_count = gpu_profile.compute_units;

        let name = format!(
            "CUDA device {} (sm_{}, {} SMs, {} MB)",
            device,
            sm_version,
            sm_count,
            total_memory / (1024 * 1024)
        );

        eprintln!("[CudaBackend] Detected: {}", name);

        let device_info = GpuDeviceInfo {
            ordinal: device,
            sm_version,
            sm_count,
            total_memory,
            name,
        };

        // CPU profile for phase 1 JIT execution
        let cpu_profile = gllm_kernels::dispatch::DeviceProfile::detect();

        Some(Self {
            device_info,
            cpu_profile,
            device: std::sync::Arc::new(cuda_device),
            gpu_profile,
            compiled_ptx: std::sync::Mutex::new(std::collections::HashMap::new()),
            swap_store: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            kv_meta: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            weight_cache: std::sync::Arc::new(std::sync::Mutex::new(super::gpu_compile::GpuWeightCache::new())),

            _marker: std::marker::PhantomData,
        })
    }

    /// Attempt to create a CUDA backend (no-cuda feature: always returns None).
    #[cfg(not(feature = "cuda"))]
    pub fn new(_device: usize) -> Option<Self> {
        None
    }

    /// Returns the detected GPU device info.
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    /// Returns the SM version (e.g. 80 for A100).
    pub fn sm_version(&self) -> u32 {
        self.device_info.sm_version
    }

    /// Returns a reference to the persistent CUDA device handle.
    #[cfg(feature = "cuda")]
    pub fn device(&self) -> &gllm_kernels::gpu::cuda::CudaDevice {
        &self.device
    }

    /// Returns a reference to the GPU hardware capability profile.
    #[cfg(feature = "cuda")]
    pub fn gpu_profile(&self) -> &gllm_kernels::gpu::GpuDeviceProfile {
        &self.gpu_profile
    }
}

impl_gpu_backend! {
    backend = CudaBackend,
    cfg_pred = [feature = "cuda"],
    feature_label = "cuda",
    upload_err = Other,
    decoder_forward = cuda_decoder_forward,
    bert_forward = cuda_bert_encoder_forward,
}
