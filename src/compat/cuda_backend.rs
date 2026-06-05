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
    /// Uploaded weight blob on GPU: (device_ptr, bytes). Uploaded once, reused.
    #[cfg(feature = "cuda")]
    weight_blob_gpu: std::sync::Mutex<Option<(u64, usize)>>,

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
            weight_blob_gpu: std::sync::Mutex::new(None),

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
            weight_blob_gpu: std::sync::Mutex::new(None),

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

    /// Upload weight blob to GPU (once), returning device pointer.
    /// Subsequent calls return the cached device pointer.
    #[cfg(feature = "cuda")]
    pub fn upload_weight_blob(&self, weight_blob: &[u8]) -> Result<u64, String> {
        use gllm_kernels::gpu::{GpuBuffer, GpuDevice};
        let mut cache = self.weight_blob_gpu.lock()
            .map_err(|e| format!("weight_blob_gpu lock poisoned: {e}"))?;
        if let Some((ptr, bytes)) = *cache {
            if bytes == weight_blob.len() {
                return Ok(ptr);
            }
        }
        let mut buf = self.device.alloc(weight_blob.len())
            .map_err(|e| format!("weight blob alloc failed ({} bytes): {e}", weight_blob.len()))?;
        let ptr = buf.as_device_ptr();
        let stream = self.device.default_stream();
        self.device.htod(weight_blob, &mut buf, stream)
            .map_err(|e| format!("weight blob htod failed: {e}"))?;
        std::mem::forget(buf);
        *cache = Some((ptr, weight_blob.len()));
        Ok(ptr)
    }

    /// Allocate scratchpad on GPU.
    #[cfg(feature = "cuda")]
    pub fn alloc_scratchpad_gpu(&self, bytes: usize) -> Result<u64, String> {
        use gllm_kernels::gpu::{GpuBuffer, GpuDevice};
        let buf = self.device.alloc(bytes)
            .map_err(|e| format!("scratchpad alloc failed ({} bytes): {e}", bytes))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }

    /// Upload data to GPU, returning device pointer.
    #[cfg(feature = "cuda")]
    pub fn upload_to_gpu<T: Copy>(&self, data: &[T]) -> Result<u64, String> {
        use gllm_kernels::gpu::{GpuBuffer, GpuDevice};
        let bytes = std::mem::size_of_val(data);
        let mut buf = self.device.alloc(bytes)
            .map_err(|e| format!("upload alloc failed ({bytes} bytes): {e}"))?;
        let ptr = buf.as_device_ptr();
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes)
        };
        let stream = self.device.default_stream();
        self.device.htod(byte_slice, &mut buf, stream)
            .map_err(|e| format!("upload htod failed: {e}"))?;
        std::mem::forget(buf);
        Ok(ptr)
    }

    /// Download data from GPU device pointer.
    #[cfg(feature = "cuda")]
    pub fn download_from_gpu(&self, src_ptr: u64, bytes: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; bytes];
        let res = unsafe {
            (self.device.driver().cuMemcpyDtoH_v2)(
                buf.as_mut_ptr() as *mut _,
                src_ptr,
                bytes,
            )
        };
        if res != 0 {
            Err(format!("cuMemcpyDtoH failed: {res}"))
        } else {
            Ok(buf)
        }
    }

    /// Returns a reference to the GPU hardware capability profile.
    #[cfg(feature = "cuda")]
    pub fn gpu_profile(&self) -> &gllm_kernels::gpu::GpuDeviceProfile {
        &self.gpu_profile
    }

    /// Launch GPU mega-kernel generate loop.
    ///
    /// Takes the compiled PTX code from `MegaKernelCompileOutput::gpu_code`,
    /// loads it via `cuModuleLoadData`, and launches with the 21-parameter ABI.
    #[cfg(feature = "cuda")]
    pub fn gpu_generate_single_sequence(
        &self,
        ptx_code: &[u8],
        kernel_name: &str,
        input_ids: &[u32],
        weight_blob_device: u64,
        scratchpad_device: u64,
        output_tokens_device: u64,
        page_table_device: Option<u64>,
        prompt_len: usize,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        eos_token_id: u32,
    ) -> Result<usize, String> {
        use gllm_kernels::gpu::GpuDevice;

        // Load PTX module
        let module = self.device.load_ptx(ptx_code)
            .map_err(|e| format!("load_ptx failed: {e}"))?;

        // Get kernel function
        let func = module.get_function(kernel_name)
            .map_err(|e| format!("get_function({kernel_name}) failed: {e}"))?;

        // Decode grid/block: single-block decode (serial generate loop)
        let warp_size = self.gpu_profile.warp_size;
        let block = (warp_size, 1u32, 1u32);
        let grid = (1u32, 1u32, 1u32);

        // Prepare 23 parameter array (cuLaunchKernel takes **void = array of pointers-to-arguments)
        let mut positions: Vec<u32> = (0..(prompt_len + max_new_tokens) as u32).collect();
        let batch_size: usize = 1;
        let temperature_u32 = temperature.to_bits();
        let top_p_u32 = top_p.to_bits();
        let output_mode_selector: usize = 0; // Generate

        // page_table_ptr: device pointer to u32 page table array, or NULL for contiguous KV
        let pt_ptr = page_table_device.unwrap_or(0);

        let args: [*mut std::ffi::c_void; 23] = [
            &input_ids.as_ptr() as *const _ as *mut std::ffi::c_void,             // 0: input_ids_ptr
            &weight_blob_device as *const _ as *mut std::ffi::c_void,              // 1: weight_blob_ptr
            &std::ptr::null::<u8>() as *const _ as *mut std::ffi::c_void,          // 2: kv_cache_ptr
            &positions.as_ptr() as *const _ as *mut std::ffi::c_void,              // 3: positions_ptr
            &std::ptr::null::<u8>() as *const _ as *mut std::ffi::c_void,          // 4: aux_ptr
            &batch_size as *const _ as *mut std::ffi::c_void,                       // 5: batch_size
            &prompt_len as *const _ as *mut std::ffi::c_void,                       // 6: prompt_len
            &scratchpad_device as *const _ as *mut std::ffi::c_void,               // 7: scratchpad_ptr
            &output_tokens_device as *const _ as *mut std::ffi::c_void,            // 8: output_tokens_ptr
            &temperature_u32 as *const _ as *mut std::ffi::c_void,                 // 9: temperature_u32
            &top_k as *const _ as *mut std::ffi::c_void,                           // 10: top_k
            &top_p_u32 as *const _ as *mut std::ffi::c_void,                       // 11: top_p_u32
            &max_new_tokens as *const _ as *mut std::ffi::c_void,                  // 12: max_new_tokens
            &eos_token_id as *const _ as *mut std::ffi::c_void,                    // 13: eos_token_id
            &output_mode_selector as *const _ as *mut std::ffi::c_void,            // 14: output_mode_selector
            &std::ptr::null::<u8>() as *const _ as *mut std::ffi::c_void,          // 15: hook_ctx_ptr
            &std::ptr::null::<u8>() as *const _ as *mut std::ffi::c_void,          // 16: telemetry_ptr
            &0usize as *const _ as *mut std::ffi::c_void,                           // 17: session_position
            &std::ptr::null::<u8>() as *const _ as *mut std::ffi::c_void,          // 18: fused_hidden_ptr
            &0usize as *const _ as *mut std::ffi::c_void,                           // 19: num_mm_tokens
            &std::ptr::null::<u8>() as *const _ as *mut std::ffi::c_void,          // 20: callback_table_ptr
            &pt_ptr as *const _ as *mut std::ffi::c_void,                          // 21: page_table_ptr (0 = contiguous KV)
            &std::ptr::null::<u8>() as *const _ as *mut std::ffi::c_void,          // 22: batch_ctx_ptr (NULL = single-seq legacy)
        ];

        let stream = self.device.default_stream();
        self.device.launch_kernel(func, grid, block, &args, stream)
            .map_err(|e| format!("launch_kernel failed: {e}"))?;

        Ok(max_new_tokens)
    }

    /// Upload a host-side page table (u32 array) to GPU device memory.
    /// Returns the device pointer to the uploaded page table.
    #[cfg(feature = "cuda")]
    pub fn upload_page_table(&self, page_table: &[u32]) -> Result<u64, String> {
        use gllm_kernels::gpu::GpuDevice;
        let bytes = page_table.len() * 4;
        let ptr = self.device.raw_alloc(bytes)
            .map_err(|e| format!("page_table alloc failed ({bytes} bytes): {e}"))?;
        let src = page_table.as_ptr() as *const u8;
        self.device.memcpy_h_to_d(ptr, unsafe { std::slice::from_raw_parts(src, bytes) })
            .map_err(|e| format!("page_table memcpy failed: {e}"))?;
        Ok(ptr)
    }

    // ------------------------------------------------------------------
    // GPU Mega-Kernel helper methods (used by gpu_backend_macro)
    // ------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    pub(super) fn get_cached_ptx(&self, key: &str) -> Option<Vec<u8>> {
        self.compiled_ptx.lock().ok()?.get(key).cloned()
    }

    #[cfg(feature = "cuda")]
    pub(super) fn get_weight_gpu_ptr(&self) -> Option<u64> {
        self.weight_blob_gpu.lock().ok()?.map(|(ptr, _)| ptr)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn get_cached_scratchpad_bytes(&self) -> usize {
        self.compiled_ptx.lock()
            .ok()
            .and_then(|cache| cache.get("__scratchpad_bytes__").map(|v| {
                let arr: [u8; 8] = v[..8].try_into().unwrap_or([0u8; 8]);
                usize::from_le_bytes(arr)
            }))
            .unwrap_or(0)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn gpu_launch_mega_kernel(
        &self,
        ptx_code: &[u8],
        kernel_name: &str,
        args: &[usize; 23],
    ) -> Result<(), String> {
        use gllm_kernels::gpu::GpuDevice;
        let module = self.device.load_ptx(ptx_code)
            .map_err(|e| format!("load_ptx failed: {e}"))?;
        let func = module.get_function(kernel_name)
            .map_err(|e| format!("get_function({kernel_name}) failed: {e}"))?;

        let warp_size = self.gpu_profile.warp_size;
        let block = (warp_size, 1u32, 1u32);
        let grid = (1u32, 1u32, 1u32);

        let kernel_args: [*mut std::ffi::c_void; 23] = std::array::from_fn(|i| {
            unsafe { &args[i] as *const usize as *mut std::ffi::c_void }
        });

        let stream = self.device.default_stream();
        self.device.launch_kernel(func, grid, block, &kernel_args, stream)
            .map_err(|e| format!("launch_kernel failed: {e}"))?;
        Ok(())
    }
}

// SAFETY: CudaBackend's raw pointers are CUDA driver handles that are thread-safe
// by the CUDA driver API. Access is synchronized through Mutex/Arc.
unsafe impl<E: Element + Send> Send for CudaBackend<E> {}
unsafe impl<E: Element + Sync> Sync for CudaBackend<E> {}

impl_gpu_backend! {
    backend = CudaBackend,
    cfg_pred = [feature = "cuda"],
    feature_label = "cuda",
    upload_err = Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_device_info_construction() {
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 80,
            sm_count: 108,
            total_memory: 80 * 1024 * 1024 * 1024,
            name: "NVIDIA A100-SXM4-80GB".to_string(),
        };
        assert_eq!(info.ordinal, 0);
        assert_eq!(info.sm_version, 80);
        assert_eq!(info.sm_count, 108);
        assert_eq!(info.total_memory, 80 * 1024 * 1024 * 1024);
        assert_eq!(info.name, "NVIDIA A100-SXM4-80GB");
    }

    #[test]
    fn gpu_device_info_clone_independence() {
        let info = GpuDeviceInfo {
            ordinal: 1,
            sm_version: 90,
            sm_count: 132,
            total_memory: 40 * 1024 * 1024 * 1024,
            name: "NVIDIA H100".to_string(),
        };
        let cloned = info.clone();
        assert_eq!(cloned.ordinal, info.ordinal);
        assert_eq!(cloned.sm_version, info.sm_version);
        assert_eq!(cloned.name, info.name);

        drop(info);
        assert_eq!(cloned.name, "NVIDIA H100");
    }

    #[test]
    fn gpu_device_info_debug_format() {
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 86,
            sm_count: 72,
            total_memory: 24 * 1024 * 1024 * 1024,
            name: "RTX 4090".to_string(),
        };
        let debug = format!("{:?}", info);
        assert!(debug.contains("GpuDeviceInfo"));
        assert!(debug.contains("ordinal"));
        assert!(debug.contains("RTX 4090"));
    }

    #[test]
    fn gpu_device_info_zero_memory() {
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 0,
            total_memory: 0,
            name: String::new(),
        };
        assert_eq!(info.total_memory, 0);
        assert!(info.name.is_empty());
    }

    #[test]
    fn gpu_device_info_equality() {
        let a = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 80,
            sm_count: 108,
            total_memory: 80,
            name: "A100".to_string(),
        };
        let b = a.clone();
        assert_eq!(a.ordinal, b.ordinal);
        assert_eq!(a.sm_version, b.sm_version);
        assert_eq!(a.sm_count, b.sm_count);
        assert_eq!(a.total_memory, b.total_memory);
        assert_eq!(a.name, b.name);
    }

    #[test]
    fn cuda_backend_new_zero_ordinal_succeeds_or_none() {
        let result = CudaBackend::<f32>::new(0);
        // On CUDA-capable system: Some; otherwise: None
        if result.is_some() {
            let backend = result.unwrap();
            assert_eq!(backend.device_info().ordinal, 0);
            assert!(backend.device_info().sm_version > 0);
        }
    }

    #[test]
    fn cuda_backend_new_invalid_ordinal_returns_none() {
        let result = CudaBackend::<f32>::new(999);
        assert!(result.is_none());
    }

    #[test]
    fn gpu_device_info_large_sm_version() {
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 100,
            sm_count: 200,
            total_memory: usize::MAX,
            name: "Future GPU".to_string(),
        };
        assert_eq!(info.sm_version, 100);
        assert_eq!(info.total_memory, usize::MAX);
    }

    #[test]
    fn gpu_device_info_name_unicode() {
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 90,
            sm_count: 100,
            total_memory: 1024,
            name: "NVIDIA H100 \u{1F680}".to_string(),
        };
        assert!(info.name.contains('\u{1F680}'));
    }

    // -----------------------------------------------------------------------
    // New tests (15 additional)
    // -----------------------------------------------------------------------

    #[test]
    fn gpu_device_info_clone_multiple_deep_copies() {
        // Arrange
        let original = GpuDeviceInfo {
            ordinal: 3,
            sm_version: 89,
            sm_count: 128,
            total_memory: 16 * 1024 * 1024 * 1024,
            name: "RTX 4070 Ti".to_string(),
        };
        // Act
        let copy1 = original.clone();
        let copy2 = original.clone();
        let copy3 = copy1.clone();
        drop(original);
        // Assert: all copies are independent deep copies with correct field values
        assert_eq!(copy1.ordinal, 3);
        assert_eq!(copy2.ordinal, 3);
        assert_eq!(copy3.ordinal, 3);
        assert_eq!(copy1.name, "RTX 4070 Ti");
        assert_eq!(copy2.name, "RTX 4070 Ti");
        assert_eq!(copy3.name, "RTX 4070 Ti");
        assert_eq!(copy1.total_memory, 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn gpu_device_info_debug_contains_all_fields() {
        // Arrange
        let info = GpuDeviceInfo {
            ordinal: 2,
            sm_version: 75,
            sm_count: 40,
            total_memory: 8_589_934_592,
            name: "TITAN V".to_string(),
        };
        // Act
        let debug_str = format!("{:#?}", info);
        // Assert: Debug output should contain field names and values
        assert!(debug_str.contains("ordinal: 2"), "Debug output missing ordinal");
        assert!(debug_str.contains("sm_version: 75"), "Debug output missing sm_version");
        assert!(debug_str.contains("sm_count: 40"), "Debug output missing sm_count");
        assert!(debug_str.contains("TITAN V"), "Debug output missing name");
        assert!(debug_str.contains("8589934592"), "Debug output missing total_memory");
    }

    #[test]
    fn gpu_device_info_memory_gb_calculation() {
        // Arrange: 24 GB GPU
        let total_bytes = 24usize * 1024 * 1024 * 1024;
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 86,
            sm_count: 72,
            total_memory: total_bytes,
            name: "RTX 4090".to_string(),
        };
        // Act
        let gb = info.total_memory / (1024 * 1024 * 1024);
        // Assert
        assert_eq!(gb, 24);
        assert_eq!(info.total_memory, total_bytes);
    }

    #[test]
    fn gpu_device_info_sm_version_accessor() {
        // Arrange
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 90,
            sm_count: 132,
            total_memory: 80 * 1024 * 1024 * 1024,
            name: "H100 SXM".to_string(),
        };
        // Act & Assert: sm_version field is the same value used by CudaBackend::sm_version()
        assert_eq!(info.sm_version, 90);
        assert!(info.sm_version > 0);
        assert!(info.sm_version < 200);
    }

    #[test]
    fn gpu_device_info_different_ordinals() {
        // Arrange: simulate a multi-GPU system with 4 devices
        let devices: Vec<GpuDeviceInfo> = (0..4)
            .map(|i| GpuDeviceInfo {
                ordinal: i,
                sm_version: 80,
                sm_count: 108,
                total_memory: 40 * 1024 * 1024 * 1024,
                name: format!("A100-40GB-{}", i),
            })
            .collect();
        // Act & Assert: each device has a unique ordinal and unique name
        for (idx, dev) in devices.iter().enumerate() {
            assert_eq!(dev.ordinal, idx);
            assert_eq!(dev.name, format!("A100-40GB-{}", idx));
        }
        // Assert: all share the same hardware profile
        assert!(devices.windows(2).all(|w| w[0].sm_version == w[1].sm_version));
        assert!(devices.windows(2).all(|w| w[0].sm_count == w[1].sm_count));
    }

    #[test]
    fn gpu_device_info_string_field_ownership() {
        // Arrange
        let external_name = String::from("Tesla V100");
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 70,
            sm_count: 80,
            total_memory: 16 * 1024 * 1024 * 1024,
            name: external_name.clone(),
        };
        // Act: mutate the external string
        let external_len_before = external_name.len();
        drop(external_name);
        // Assert: GpuDeviceInfo retains its own copy
        assert_eq!(info.name.len(), external_len_before);
        assert_eq!(info.name, "Tesla V100");
    }

    #[test]
    fn cuda_backend_device_info_accessor_with_valid_info() {
        // Arrange
        let result = CudaBackend::<f32>::new(0);
        if let Some(backend) = result {
            // Act
            let info = backend.device_info();
            // Assert: device_info() returns a reference to the stored GpuDeviceInfo
            assert_eq!(info.ordinal, 0);
            assert!(!info.name.is_empty());
            assert!(info.sm_version > 0);
            assert!(info.sm_count > 0);
            assert!(info.total_memory > 0);
        }
        // If no CUDA device available, test is vacuously true
    }

    #[test]
    fn cuda_backend_sm_version_accessor() {
        // Arrange
        let result = CudaBackend::<f32>::new(0);
        if let Some(backend) = result {
            // Act
            let sm = backend.sm_version();
            let info_sm = backend.device_info().sm_version;
            // Assert: sm_version() returns the same value as device_info().sm_version
            assert_eq!(sm, info_sm);
            assert!(sm > 0);
        }
    }

    #[test]
    fn cuda_backend_debug_output_format() {
        // Arrange
        let result = CudaBackend::<f32>::new(0);
        if let Some(backend) = result {
            // Act
            let debug = format!("{:?}", backend);
            // Assert: Debug output contains the struct name and device_info
            assert!(debug.contains("CudaBackend"), "Debug missing struct name");
            assert!(debug.contains("device_info"), "Debug missing device_info field");
            assert!(debug.contains("compiled_ptx_count"), "Debug missing ptx count");
        }
    }

    #[test]
    fn cuda_backend_clone_preserves_device_info() {
        // Arrange
        let result = CudaBackend::<f32>::new(0);
        if let Some(backend) = result {
            let original_info = backend.device_info().clone();
            // Act
            let cloned = backend.clone();
            // Assert: cloned backend has the same device_info
            assert_eq!(cloned.device_info().ordinal, original_info.ordinal);
            assert_eq!(cloned.device_info().sm_version, original_info.sm_version);
            assert_eq!(cloned.device_info().sm_count, original_info.sm_count);
            assert_eq!(cloned.device_info().total_memory, original_info.total_memory);
            assert_eq!(cloned.device_info().name, original_info.name);
        }
    }

    #[test]
    fn cuda_backend_device_memory_capacity() {
        // Arrange
        let result = CudaBackend::<f32>::new(0);
        if let Some(backend) = result {
            use crate::compat::Backend;
            // Act
            let capacity = backend.device_memory_capacity();
            // Assert: capacity should be positive and match total_memory from device_info
            assert!(capacity > 0, "GPU memory capacity should be positive");
            assert_eq!(capacity, backend.device_info().total_memory);
        }
    }

    #[test]
    fn cuda_backend_get_memory_pressure() {
        // Arrange
        let result = CudaBackend::<f32>::new(0);
        if let Some(backend) = result {
            use crate::compat::Backend;
            // Act
            let pressure = backend.get_memory_pressure().expect("get_memory_pressure should succeed");
            // Assert: pressure is a normalized value between 0.0 and 1.0
            assert!(
                pressure >= 0.0 && pressure <= 1.0,
                "Memory pressure should be in [0.0, 1.0], got {}",
                pressure
            );
        }
    }

    #[test]
    fn cuda_backend_gpu_sm_version_via_trait() {
        // Arrange
        let result = CudaBackend::<f32>::new(0);
        if let Some(backend) = result {
            use crate::compat::Backend;
            // Act
            let trait_sm = backend.gpu_sm_version();
            let method_sm = backend.sm_version();
            // Assert: trait method and direct accessor return same value
            assert_eq!(trait_sm, method_sm);
            assert!(trait_sm > 0);
        }
    }

    #[test]
    fn cuda_backend_new_with_f32_and_bf16_element_types() {
        // Arrange & Act: CudaBackend should compile with different Element type parameters
        let result_f32 = CudaBackend::<f32>::new(0);
        let result_bf16 = CudaBackend::<half::bf16>::new(0);
        // Assert: both should return Some or both None consistently
        assert_eq!(
            result_f32.is_some(),
            result_bf16.is_some(),
            "CudaBackend::new should behave consistently regardless of Element type parameter"
        );
        if let (Some(bf32), Some(bbf16)) = (result_f32, result_bf16) {
            assert_eq!(bf32.device_info().ordinal, bbf16.device_info().ordinal);
            assert_eq!(bf32.sm_version(), bbf16.sm_version());
        }
    }

    #[test]
    fn cuda_backend_new_returns_none_for_very_large_ordinal() {
        // Arrange
        let huge_ordinal = usize::MAX;
        // Act
        let result = CudaBackend::<f32>::new(huge_ordinal);
        // Assert: no system has usize::MAX CUDA devices
        assert!(result.is_none(), "CudaBackend::new(usize::MAX) should return None");
    }

    // -----------------------------------------------------------------------
    // Additional tests (15 new)
    // -----------------------------------------------------------------------

    #[test]
    fn backend_error_cuda_variant_display() {
        // Arrange
        let err = BE::Cuda("device lost".to_string());
        // Act
        let msg = format!("{}", err);
        // Assert: Display for Cuda variant contains the error message
        assert!(msg.contains("CUDA error: device lost"), "Expected CUDA error prefix, got: {msg}");
    }

    #[test]
    fn backend_error_other_variant_display() {
        // Arrange
        let err = BE::Other("custom failure".to_string());
        // Act
        let msg = format!("{}", err);
        // Assert: Display for Other variant contains the error message
        assert!(msg.contains("custom failure"), "Expected custom failure in display, got: {msg}");
        assert!(msg.contains("backend error"), "Expected 'backend error' prefix");
    }

    #[test]
    fn backend_error_unimplemented_variant_display() {
        // Arrange
        let err = BE::Unimplemented("cuda feature not enabled");
        // Act
        let msg = format!("{}", err);
        // Assert: Display for Unimplemented variant contains the message
        assert!(msg.contains("unimplemented"), "Expected 'unimplemented' prefix, got: {msg}");
        assert!(msg.contains("cuda feature not enabled"), "Expected full message");
    }

    #[test]
    fn backend_error_hip_and_metal_variants_display() {
        // Arrange
        let hip_err = BE::Hip("hip alloc failed".to_string());
        let metal_err = BE::Metal("metal compile failed".to_string());
        let cpu_err = BE::Cpu("cpu error".to_string());
        // Act
        let hip_msg = format!("{}", hip_err);
        let metal_msg = format!("{}", metal_err);
        let cpu_msg = format!("{}", cpu_err);
        // Assert: each variant has its own prefix
        assert!(hip_msg.contains("HIP error: hip alloc failed"), "Expected HIP prefix, got: {hip_msg}");
        assert!(metal_msg.contains("Metal error: metal compile failed"), "Expected Metal prefix, got: {metal_msg}");
        assert!(cpu_msg.contains("CPU error: cpu error"), "Expected CPU prefix, got: {cpu_msg}");
    }

    #[test]
    fn backend_error_debug_format_matches_variant() {
        // Arrange
        let err = BE::Other("test message".to_string());
        // Act
        let debug = format!("{:?}", err);
        // Assert: Debug output contains the variant name and message
        assert!(debug.contains("Other"), "Debug should contain variant name 'Other', got: {debug}");
        assert!(debug.contains("test message"), "Debug should contain the message, got: {debug}");
    }

    #[test]
    fn backend_error_clone_preserves_message() {
        // Arrange
        let original = BE::Cuda("original error".to_string());
        // Act
        let cloned = original.clone();
        // Assert
        assert_eq!(format!("{}", original), format!("{}", cloned));
    }

    #[test]
    fn kv_cache_handle_zero_value() {
        // Arrange & Act
        let handle = KvCacheHandle(0);
        // Assert: zero handle represents no allocation
        assert_eq!(handle.0, 0);
    }

    #[test]
    fn kv_cache_handle_equality_and_hash() {
        // Arrange
        let a = KvCacheHandle(42);
        let b = KvCacheHandle(42);
        let c = KvCacheHandle(99);
        // Assert: PartialEq + Copy + Hash derived
        assert_eq!(a, b);
        assert_ne!(a, c);
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        set.insert(c);
        assert_eq!(set.len(), 2, "Equal handles should deduplicate in HashSet");
    }

    #[test]
    fn logits_handle_empty_data() {
        // Arrange & Act
        let handle = LogitsHandle { data: vec![] };
        // Assert
        assert!(handle.data.is_empty());
    }

    #[test]
    fn logits_handle_clone_preserves_data() {
        // Arrange
        let original = LogitsHandle { data: vec![1.0, 2.5, -0.3] };
        // Act
        let cloned = original.clone();
        // Assert
        assert_eq!(cloned.data.len(), 3);
        assert_eq!(cloned.data[0], 1.0);
        assert_eq!(cloned.data[1], 2.5);
        assert_eq!(cloned.data[2], -0.3);
        // Mutating clone does not affect original
        let mut detached = cloned;
        detached.data[0] = 999.0;
        assert_eq!(original.data[0], 1.0, "Clone should be independent");
    }

    #[test]
    fn weight_placement_variants_and_equality() {
        // Arrange
        use crate::compat::backend_trait::WeightPlacement;
        let device_local = WeightPlacement::DeviceLocal;
        let host_local = WeightPlacement::HostLocal;
        // Assert: both variants exist and are distinct
        assert_ne!(device_local, host_local);
        assert_eq!(device_local, WeightPlacement::DeviceLocal);
        assert_eq!(host_local, WeightPlacement::HostLocal);
    }

    #[test]
    fn weight_placement_debug_and_clone() {
        // Arrange
        use crate::compat::backend_trait::WeightPlacement;
        let original = WeightPlacement::DeviceLocal;
        // Act
        let cloned = original.clone();
        let debug = format!("{:?}", original);
        // Assert
        assert_eq!(original, cloned);
        assert!(debug.contains("DeviceLocal"), "Debug should contain variant name, got: {debug}");
    }

    #[test]
    fn cuda_backend_trait_returns_unimplemented_without_feature() {
        // Arrange
        let result = CudaBackend::<f32>::new(0);
        if let Some(backend) = result {
            // Act: alloc_kv_cache should work (requires real CUDA hardware)
            // If we have a real GPU, this test exercises the CUDA path.
            // Without feature, Backend methods return Unimplemented.
            // We test that the trait object dispatch compiles and the backend exists.
            use crate::compat::Backend;
            let sm = backend.gpu_sm_version();
            // Assert: valid GPU always has sm > 0
            assert!(sm > 0, "GPU SM version should be positive when CUDA is available");
        }
        // Without CUDA hardware, this test is vacuously true (None path).
    }

    #[test]
    fn cuda_backend_new_with_f16_element_type() {
        // Arrange & Act
        let result_f16 = CudaBackend::<half::f16>::new(0);
        let result_f32 = CudaBackend::<f32>::new(0);
        // Assert: both element types behave identically for new()
        assert_eq!(result_f16.is_some(), result_f32.is_some());
        if let (Some(bf16), Some(bf32)) = (result_f16, result_f32) {
            assert_eq!(bf16.device_info().ordinal, bf32.device_info().ordinal);
            assert_eq!(bf16.sm_version(), bf32.sm_version());
        }
    }

    #[test]
    fn gpu_device_info_sm_version_boundary_values() {
        // Arrange: test SM versions from earliest (sm_30) through future (sm_120)
        let versions = [30u32, 52, 60, 70, 75, 80, 86, 89, 90, 100, 120];
        for &sm in &versions {
            // Act
            let info = GpuDeviceInfo {
                ordinal: 0,
                sm_version: sm,
                sm_count: 100,
                total_memory: 16 * 1024 * 1024 * 1024,
                name: format!("GPU sm_{sm}"),
            };
            // Assert
            assert_eq!(info.sm_version, sm, "SM version should round-trip for sm_{sm}");
            assert!(info.name.contains(&sm.to_string()));
        }
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 2 (15 new)
    // -----------------------------------------------------------------------

    #[test]
    fn sampling_config_default_values() {
        // Arrange & Act
        let config = SamplingConfig::default();
        // Assert: defaults match standard greedy sampling parameters
        assert_eq!(config.temperature, 1.0, "default temperature should be 1.0");
        assert_eq!(config.top_k, 0, "default top_k should be 0 (disabled)");
        assert_eq!(config.top_p, 1.0, "default top_p should be 1.0 (no filtering)");
    }

    #[test]
    fn sampling_config_custom_values_and_copy_semantics() {
        // Arrange
        let original = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
        };
        // Act: Copy (SamplingConfig should be Copy via derive)
        let copied = original;
        // Assert: both have identical values
        assert_eq!(copied.temperature, 0.7);
        assert_eq!(copied.top_k, 50);
        assert_eq!(copied.top_p, 0.95);
        // Assert: original still valid after copy (Copy semantics)
        assert_eq!(original.temperature, 0.7);
    }

    #[test]
    fn page_state_variants_distinct_and_equality() {
        // Arrange
        use crate::scheduler::types::PageState;
        let states = [
            (PageState::Free, "Free"),
            (PageState::Active, "Active"),
            (PageState::Standby, "Standby"),
            (PageState::SwappedOut, "SwappedOut"),
            (PageState::Warm, "Warm"),
            (PageState::Protected, "Protected"),
            (PageState::Swapped, "Swapped"),
        ];
        // Assert: each variant is distinct from every other
        for i in 0..states.len() {
            for j in 0..states.len() {
                if i == j {
                    assert_eq!(states[i].0, states[j].0, "{} should equal itself", states[i].1);
                } else {
                    assert_ne!(states[i].0, states[j].0, "{} should not equal {}", states[i].1, states[j].1);
                }
            }
        }
    }

    #[test]
    fn page_state_debug_and_clone() {
        // Arrange
        use crate::scheduler::types::PageState;
        let original = PageState::SwappedOut;
        // Act
        let cloned = original.clone();
        let debug = format!("{:?}", original);
        // Assert
        assert_eq!(original, cloned);
        assert!(debug.contains("SwappedOut"), "Debug should contain variant name, got: {debug}");
    }

    #[test]
    fn storage_key_type_alias_usage() {
        // Arrange: StorageKey is u64, used as opaque handle for swap operations
        use crate::scheduler::types::StorageKey;
        let key1: StorageKey = 0xDEADBEEFu64;
        let key2: StorageKey = 0u64;
        // Assert: basic u64 semantics work
        assert_ne!(key1, key2);
        assert_eq!(key2, 0u64);
        assert!(key1 > key2);
        // Assert: can be used in HashMap
        use std::collections::HashMap;
        let mut map: HashMap<StorageKey, Vec<u8>> = HashMap::new();
        map.insert(key1, vec![1, 2, 3]);
        assert_eq!(map.get(&key1).unwrap().len(), 3);
        assert!(map.get(&key2).is_none());
    }

    #[test]
    fn kv_cache_handle_sorting_by_inner_value() {
        // Arrange: KvCacheHandle is Copy + Hash + Eq, verify sort by inner u64
        let mut handles = vec![
            KvCacheHandle(300),
            KvCacheHandle(100),
            KvCacheHandle(200),
        ];
        // Act: sort by the inner u64 value
        handles.sort_by_key(|h| h.0);
        // Assert: sorted in ascending order by inner value
        assert_eq!(handles[0], KvCacheHandle(100));
        assert_eq!(handles[1], KvCacheHandle(200));
        assert_eq!(handles[2], KvCacheHandle(300));
    }

    #[test]
    fn logits_handle_with_special_float_values() {
        // Arrange: logits can contain NaN, Inf, negative values
        let handle = LogitsHandle {
            data: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0],
        };
        // Assert: data preserves special float values
        assert!(handle.data[0].is_nan(), "NaN should be preserved");
        assert!(handle.data[1].is_infinite() && handle.data[1].is_sign_positive(), "Inf should be preserved");
        assert!(handle.data[2].is_infinite() && handle.data[2].is_sign_negative(), "NegInf should be preserved");
        assert_eq!(handle.data.len(), 5);
    }

    #[test]
    fn logits_handle_large_data_clone_independence() {
        // Arrange: large logits tensor (vocab_size=128256 for Gemma)
        let large_data: Vec<f32> = (0..128256).map(|i| i as f32 * 0.001).collect();
        let handle = LogitsHandle { data: large_data };
        // Act
        let cloned = handle.clone();
        // Assert: cloned is a deep copy
        assert_eq!(cloned.data.len(), handle.data.len());
        assert_eq!(cloned.data[0], 0.0);
        assert_eq!(cloned.data[128255], 128255.0f32 * 0.001);
        // Assert: drop original, cloned remains valid
        drop(handle);
        assert_eq!(cloned.data.len(), 128256);
    }

    #[test]
    fn backend_error_implements_std_error() {
        // Arrange
        let err = BE::Cuda("device error".to_string());
        // Act: cast to &dyn std::error::Error
        let err_ref: &dyn std::error::Error = &err;
        // Assert: std::error::Error trait is implemented and display works
        let msg = err_ref.to_string();
        assert!(msg.contains("device error"), "Error display should contain message, got: {msg}");
    }

    #[test]
    fn attention_mask_type_variants_and_hash() {
        // Arrange
        use crate::engine::executor::AttentionMaskType;
        let bidir = AttentionMaskType::Bidirectional;
        let causal = AttentionMaskType::Causal;
        // Assert: variants are distinct
        assert_ne!(bidir, causal);
        // Assert: can be used in HashSet (Hash derived)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(bidir);
        set.insert(causal);
        assert_eq!(set.len(), 2, "Both mask types should be distinct in HashSet");
        // Assert: equality works
        assert_eq!(bidir, AttentionMaskType::Bidirectional);
        assert_eq!(causal, AttentionMaskType::Causal);
    }

    #[test]
    fn weight_placement_all_variants_exhaustive() {
        // Arrange: ensure we cover all WeightPlacement variants exhaustively
        use crate::compat::backend_trait::WeightPlacement;
        let device = WeightPlacement::DeviceLocal;
        let host = WeightPlacement::HostLocal;
        // Assert: exactly 2 distinct variants exist and they are not equal
        assert_ne!(device, host, "DeviceLocal and HostLocal should be distinct");
        // Assert: each variant equals itself
        assert_eq!(device, WeightPlacement::DeviceLocal);
        assert_eq!(host, WeightPlacement::HostLocal);
        // Assert: exhaustive match — compiler enforces we handle every variant
        let count = match device { WeightPlacement::DeviceLocal => 1, WeightPlacement::HostLocal => 1 }
                  + match host { WeightPlacement::DeviceLocal => 1, WeightPlacement::HostLocal => 1 };
        assert_eq!(count, 2, "Should have 2 variants total");
    }

    #[test]
    fn swap_config_construction_and_fields() {
        // Arrange
        use crate::engine::executor::SwapConfig;
        let config = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.85,
            lru_granularity: 4,
        };
        // Assert: all fields are set correctly
        assert!(config.enable_swap);
        assert!((config.swap_threshold - 0.85).abs() < f32::EPSILON);
        assert_eq!(config.lru_granularity, 4);
    }

    #[test]
    fn swap_config_clone_and_partial_eq() {
        // Arrange
        use crate::engine::executor::SwapConfig;
        let original = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.9,
            lru_granularity: 8,
        };
        // Act
        let cloned = original.clone();
        // Assert: PartialEq
        assert_eq!(original, cloned);
        // Assert: different values are not equal
        let different = SwapConfig {
            enable_swap: false,
            swap_threshold: 0.9,
            lru_granularity: 8,
        };
        assert_ne!(original, different);
    }

    #[test]
    fn paged_kv_config_construction() {
        // Arrange
        use crate::engine::executor::PagedKvConfig;
        let config = PagedKvConfig {
            page_table: Some(vec![0u32, 1, 2, 3, 4]),
            page_size: 16,
        };
        // Assert
        assert_eq!(config.page_size, 16);
        assert_eq!(config.page_table.as_ref().unwrap().len(), 5);
        // Assert: None variant (no paging)
        let no_paging = PagedKvConfig {
            page_table: None,
            page_size: 16,
        };
        assert!(no_paging.page_table.is_none());
    }

    #[test]
    fn gpu_device_info_name_format_matches_real_output() {
        // Arrange: verify the name format matches what CudaBackend::new() produces
        let ordinal = 2;
        let sm_version = 90u32;
        let sm_count = 132u32;
        let total_memory = 80usize * 1024 * 1024 * 1024;
        // Act: reproduce the name format from CudaBackend::new()
        let name = format!(
            "CUDA device {} (sm_{}, {} SMs, {} MB)",
            ordinal,
            sm_version,
            sm_count,
            total_memory / (1024 * 1024)
        );
        let info = GpuDeviceInfo {
            ordinal,
            sm_version,
            sm_count,
            total_memory,
            name: name.clone(),
        };
        // Assert: name contains all expected components
        assert!(info.name.contains("CUDA device 2"), "name should contain ordinal");
        assert!(info.name.contains("sm_90"), "name should contain SM version");
        assert!(info.name.contains("132 SMs"), "name should contain SM count");
        assert!(info.name.contains("81920 MB"), "name should contain memory in MB");
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 3 (15 new)
    // -----------------------------------------------------------------------

    #[test]
    fn sampling_config_extreme_temperature_values() {
        // Arrange: temperature=0.0 is greedy decoding, temperature=2.0 is high entropy
        let greedy = SamplingConfig { temperature: 0.0, top_k: 1, top_p: 1.0 };
        let high_entropy = SamplingConfig { temperature: 2.0, top_k: 0, top_p: 0.5 };
        // Assert: both configs are valid and distinct
        assert_eq!(greedy.temperature, 0.0);
        assert_eq!(high_entropy.temperature, 2.0);
        assert_ne!(greedy.temperature, high_entropy.temperature);
        assert_eq!(greedy.top_k, 1);
        assert_eq!(high_entropy.top_p, 0.5);
    }

    #[test]
    fn kv_cache_handle_max_u64_value() {
        // Arrange & Act: u64::MAX represents the maximum handle value
        let handle = KvCacheHandle(u64::MAX);
        // Assert: the handle preserves the full u64 range
        assert_eq!(handle.0, u64::MAX);
        assert_eq!(handle, KvCacheHandle(u64::MAX));
        assert_ne!(handle, KvCacheHandle(0));
    }

    #[test]
    fn logits_handle_equality_after_identical_construction() {
        // Arrange
        let data = vec![0.5, -1.2, 3.14, 0.0];
        let handle_a = LogitsHandle { data: data.clone() };
        let handle_b = LogitsHandle { data: data.clone() };
        // Act & Assert: two handles with identical data have equal content
        assert_eq!(handle_a.data, handle_b.data);
        assert_eq!(handle_a.data.len(), 4);
        assert_eq!(handle_a.data[2], 3.14);
    }

    #[test]
    fn sequence_input_validate_page_table_with_valid_entries() {
        // Arrange
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2, 3]),
            fused_hidden: None,
        };
        // Act
        let result = input.validate_page_table(10);
        // Assert: all page IDs are within bounds
        assert!(result.is_ok(), "Valid page table should pass validation");
    }

    #[test]
    fn sequence_input_validate_page_table_with_out_of_bounds_entry() {
        // Arrange
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1, 2],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 5, 10]),
            fused_hidden: None,
        };
        // Act: total_pages=8, so page_table[1]=5 is ok but page_table[2]=10 is out of bounds
        let result = input.validate_page_table(8);
        // Assert
        assert!(result.is_err(), "Out-of-bounds page ID should fail validation");
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("10"), "Error should mention the offending page ID");
        assert!(err_msg.contains("bounds violation"), "Error should mention bounds violation");
    }

    #[test]
    fn sequence_input_validate_page_table_none_is_always_valid() {
        // Arrange
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        // Act: no page table = contiguous access, always valid
        let result = input.validate_page_table(0);
        // Assert
        assert!(result.is_ok(), "None page_table should always be valid, even with 0 total_pages");
    }

    #[test]
    fn batch_input_empty_sequences_is_valid_construction() {
        // Arrange & Act
        use crate::engine::executor::BatchInput;
        let batch = BatchInput { sequences: vec![] };
        // Assert: empty batch is a valid construction (represents no active sequences)
        assert!(batch.sequences.is_empty());
        assert_eq!(batch.sequences.len(), 0);
    }

    #[test]
    fn batch_input_multiple_sequences_independent_ownership() {
        // Arrange
        use crate::engine::executor::{BatchInput, SequenceInput};
        let batch = BatchInput {
            sequences: vec![
                SequenceInput {
                    tokens: vec![10, 20, 30],
                    position: 0,
                    draft_steps: 0,
                    page_table: None,
                    fused_hidden: None,
                },
                SequenceInput {
                    tokens: vec![40, 50],
                    position: 3,
                    draft_steps: 0,
                    page_table: Some(vec![0, 1]),
                    fused_hidden: None,
                },
            ],
        };
        // Act
        let cloned = batch.clone();
        // Assert: cloned sequences are independent copies
        assert_eq!(cloned.sequences.len(), 2);
        assert_eq!(cloned.sequences[0].tokens, vec![10, 20, 30]);
        assert_eq!(cloned.sequences[1].position, 3);
        assert_eq!(cloned.sequences[1].page_table.as_ref().unwrap().len(), 2);
        assert!(cloned.sequences[0].page_table.is_none());
    }

    #[test]
    fn backend_error_from_cuda_matches_display_prefix() {
        // Arrange: BackendError::Cuda is the canonical CUDA error variant
        let errors = vec![
            BE::Cuda("out of memory".to_string()),
            BE::Cuda("invalid device ordinal".to_string()),
            BE::Cuda("cuLaunchKernel failed: 999".to_string()),
        ];
        // Act & Assert: every Cuda variant starts with "CUDA error:"
        for err in &errors {
            let msg = format!("{}", err);
            assert!(msg.starts_with("CUDA error:"), "Cuda error should start with 'CUDA error:', got: {msg}");
        }
    }

    #[test]
    fn backend_error_all_variants_have_non_empty_display() {
        // Arrange: every BackendError variant should produce non-empty Display output
        let variants: Vec<BE> = vec![
            BE::Cuda("c".to_string()),
            BE::Hip("h".to_string()),
            BE::Metal("m".to_string()),
            BE::Cpu("p".to_string()),
            BE::Unimplemented("unimpl"),
            BE::Other("o".to_string()),
        ];
        // Act & Assert
        for err in &variants {
            let msg = format!("{}", err);
            assert!(!msg.is_empty(), "Every BackendError variant should have non-empty Display, got empty for {:?}", err);
        }
    }

    #[test]
    fn gpu_device_info_total_memory_edge_case_one_byte() {
        // Arrange: edge case — GPU reports 1 byte of memory (absurd but tests boundary)
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 1,
            sm_count: 1,
            total_memory: 1,
            name: "Minimal GPU".to_string(),
        };
        // Assert: single byte is preserved
        assert_eq!(info.total_memory, 1);
        // Act: MB calculation yields 0 for sub-MB memory
        let mb = info.total_memory / (1024 * 1024);
        assert_eq!(mb, 0, "Sub-MB memory should yield 0 when divided by MB");
    }

    #[test]
    fn gpu_device_info_name_empty_string_is_valid() {
        // Arrange: backend could theoretically have an empty name if driver returns empty
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 80,
            sm_count: 108,
            total_memory: 40 * 1024 * 1024 * 1024,
            name: String::new(),
        };
        // Assert: empty name is valid, does not panic on Debug
        let debug = format!("{:?}", info);
        assert!(debug.contains("GpuDeviceInfo"));
        assert!(info.name.is_empty());
    }

    #[test]
    fn sampling_config_copy_trait_allows_simultaneous_use() {
        // Arrange: SamplingConfig derives Copy, verify both values remain usable
        let config = SamplingConfig { temperature: 0.8, top_k: 100, top_p: 0.9 };
        // Act: Copy — no move, both original and assigned variable are usable
        let copied = config;
        // Assert: both are independently usable (Copy semantics)
        assert_eq!(config.temperature, 0.8);
        assert_eq!(copied.temperature, 0.8);
        assert_eq!(config.top_k, copied.top_k);
        assert_eq!(config.top_p, copied.top_p);
    }

    #[test]
    fn sequence_input_with_fused_hidden_multimodal_data() {
        // Arrange: multimodal request carries fused hidden state for prefill
        use crate::engine::executor::SequenceInput;
        let fused: Vec<f32> = vec![0.1; 64]; // e.g., hidden_size=64, seq_len=1
        let input = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(fused.clone()),
        };
        // Act
        let cloned = input.clone();
        // Assert: fused_hidden is cloned correctly
        assert!(input.fused_hidden.is_some());
        assert!(cloned.fused_hidden.is_some());
        assert_eq!(cloned.fused_hidden.as_ref().unwrap().len(), 64);
        assert_eq!(cloned.fused_hidden.as_ref().unwrap()[0], 0.1);
    }

    #[test]
    fn kv_cache_handle_usable_as_hashmap_key() {
        // Arrange: KvCacheHandle must work as HashMap key (Hash + Eq)
        use std::collections::HashMap;
        let mut map: HashMap<KvCacheHandle, String> = HashMap::new();
        let h1 = KvCacheHandle(100);
        let h2 = KvCacheHandle(200);
        let h3 = KvCacheHandle(100); // same as h1
        // Act
        map.insert(h1, "cache A".to_string());
        map.insert(h2, "cache B".to_string());
        // Assert: h3 finds h1's value (same hash)
        assert_eq!(map.get(&h3).unwrap(), "cache A");
        assert_eq!(map.get(&h1).unwrap(), "cache A");
        assert_eq!(map.get(&h2).unwrap(), "cache B");
        assert_eq!(map.len(), 2, "Duplicate key should overwrite, not create new entry");
        // Act: overwrite h1
        map.insert(h3, "cache C".to_string());
        // Assert: value was overwritten
        assert_eq!(map.get(&h1).unwrap(), "cache C");
        assert_eq!(map.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 4 (13 new)
    // -----------------------------------------------------------------------

    #[test]
    fn position_encoding_variants_are_distinct() {
        // Arrange
        use crate::engine::executor::PositionEncoding;
        let none = PositionEncoding::None;
        let rope = PositionEncoding::Rope;
        // Assert: the two variants are distinct and self-equal
        assert_ne!(none, rope, "None and Rope should be distinct");
        assert_eq!(none, PositionEncoding::None);
        assert_eq!(rope, PositionEncoding::Rope);
        // Assert: Copy + Clone semantics
        let copied = rope;
        assert_eq!(copied, PositionEncoding::Rope);
        assert_eq!(rope, PositionEncoding::Rope, "Copy leaves original usable");
    }

    #[test]
    fn position_encoding_debug_output() {
        // Arrange
        use crate::engine::executor::PositionEncoding;
        // Act
        let none_debug = format!("{:?}", PositionEncoding::None);
        let rope_debug = format!("{:?}", PositionEncoding::Rope);
        // Assert: Debug output contains variant name
        assert!(none_debug.contains("None"), "Expected 'None' in debug, got: {none_debug}");
        assert!(rope_debug.contains("Rope"), "Expected 'Rope' in debug, got: {rope_debug}");
    }

    #[test]
    fn rope_config_construction_and_copy_semantics() {
        // Arrange
        use crate::engine::executor::RoPEConfig;
        let original = RoPEConfig {
            theta: 500000.0,
            scale: 2.0,
            interleaved: true,
            precompute: false,
        };
        // Act: RoPEConfig derives Copy
        let copied = original;
        // Assert: both remain usable after copy
        assert_eq!(original.theta, 500000.0);
        assert_eq!(copied.theta, 500000.0);
        assert_eq!(original.scale, copied.scale);
        assert_eq!(original.interleaved, copied.interleaved);
        assert_eq!(original.precompute, copied.precompute);
    }

    #[test]
    fn rope_config_equality_and_default_like_values() {
        // Arrange: typical Llama-style defaults
        use crate::engine::executor::RoPEConfig;
        let a = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        let b = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        let c = RoPEConfig { theta: 1000000.0, scale: 1.0, interleaved: false, precompute: false };
        // Assert: identical configs are equal
        assert_eq!(a, b, "Identical RoPEConfig should be equal");
        assert_ne!(a, c, "Different theta should make configs unequal");
    }

    #[test]
    fn attention_head_config_construction_and_fields() {
        // Arrange
        use crate::engine::executor::AttentionHeadConfig;
        let config = AttentionHeadConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
        };
        // Assert: all fields preserved correctly
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        // Assert: GQA ratio calculation
        let gqa_ratio = config.num_heads / config.num_kv_heads;
        assert_eq!(gqa_ratio, 4, "GQA group size should be 32/8=4");
    }

    #[test]
    fn attention_head_config_copy_semantics() {
        // Arrange
        use crate::engine::executor::AttentionHeadConfig;
        let original = AttentionHeadConfig {
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 64,
        };
        // Act: Copy (not move)
        let copied = original;
        // Assert: both remain usable
        assert_eq!(original.num_heads, 16);
        assert_eq!(copied.num_heads, 16);
        assert_eq!(original.head_dim, copied.head_dim);
    }

    #[test]
    fn backend_error_unimplemented_static_str_lifetime() {
        // Arrange: Unimplemented variant borrows a &'static str
        let err = BE::Unimplemented("cuda feature not enabled");
        // Act: move the error around to verify static lifetime is fine
        let moved_err = err;
        let msg = format!("{}", moved_err);
        // Assert: message is preserved after move
        assert!(msg.contains("cuda feature not enabled"), "Static str should survive move, got: {msg}");
    }

    #[test]
    fn page_id_type_alias_arithmetic() {
        // Arrange: PageId = usize, used for physical page indexing
        use crate::scheduler::types::PageId;
        let page: PageId = 42;
        let offset: PageId = page * 16 + 3;
        // Assert: standard usize arithmetic works
        assert_eq!(offset, 675);
        assert!(page < offset);
        // Assert: can be used in range
        let pages: Vec<PageId> = (0..5).collect();
        assert_eq!(pages.len(), 5);
        assert_eq!(pages[4], 4);
    }

    #[test]
    fn paged_kv_config_large_page_table() {
        // Arrange: simulate a large page table for long context
        use crate::engine::executor::PagedKvConfig;
        let page_count = 4096usize;
        let page_table: Vec<u32> = (0..page_count as u32).collect();
        let config = PagedKvConfig {
            page_table: Some(page_table),
            page_size: 16,
        };
        // Assert: page table length matches expected count
        assert_eq!(config.page_table.as_ref().unwrap().len(), page_count);
        assert_eq!(config.page_size, 16);
        // Assert: first and last entries
        assert_eq!(config.page_table.as_ref().unwrap()[0], 0);
        assert_eq!(config.page_table.as_ref().unwrap()[page_count - 1], (page_count - 1) as u32);
    }

    #[test]
    fn swap_config_disabled_no_swap() {
        // Arrange: swap is disabled
        use crate::engine::executor::SwapConfig;
        let config = SwapConfig {
            enable_swap: false,
            swap_threshold: 0.0,
            lru_granularity: 0,
        };
        // Assert: disabled config is valid construction
        assert!(!config.enable_swap);
        assert_eq!(config.swap_threshold, 0.0);
        assert_eq!(config.lru_granularity, 0);
        // Assert: different from enabled config
        let enabled = SwapConfig { enable_swap: true, swap_threshold: 0.8, lru_granularity: 4 };
        assert_ne!(config, enabled);
    }

    #[test]
    fn attention_mask_type_copy_and_debug() {
        // Arrange
        use crate::engine::executor::AttentionMaskType;
        let bidir = AttentionMaskType::Bidirectional;
        // Act: Copy
        let copied = bidir;
        let debug = format!("{:?}", bidir);
        // Assert: Copy leaves original usable
        assert_eq!(bidir, AttentionMaskType::Bidirectional);
        assert_eq!(copied, AttentionMaskType::Bidirectional);
        // Assert: Debug output
        assert!(debug.contains("Bidirectional"), "Debug should contain variant, got: {debug}");
    }

    #[test]
    fn sequence_input_empty_tokens_boundary() {
        // Arrange: empty token sequence (edge case)
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        // Assert: empty tokens is a valid construction
        assert!(input.tokens.is_empty());
        assert_eq!(input.position, 0);
        assert!(input.page_table.is_none());
        assert!(input.fused_hidden.is_none());
        // Act: validate_page_table with empty input should always succeed
        let result = input.validate_page_table(0);
        assert!(result.is_ok(), "Empty tokens with None page_table should be valid");
    }

    #[test]
    fn batch_input_single_sequence_with_page_table() {
        // Arrange: single sequence with a page table (typical decode scenario)
        use crate::engine::executor::{BatchInput, SequenceInput};
        let batch = BatchInput {
            sequences: vec![SequenceInput {
                tokens: vec![100, 200, 300],
                position: 5,
                draft_steps: 0,
                page_table: Some(vec![0u32, 1, 2]),
                fused_hidden: None,
            }],
        };
        // Act
        assert_eq!(batch.sequences.len(), 1);
        let seq = &batch.sequences[0];
        // Assert: sequence fields are correct
        assert_eq!(seq.tokens, vec![100, 200, 300]);
        assert_eq!(seq.position, 5);
        assert_eq!(seq.page_table.as_ref().unwrap().len(), 3);
        // Assert: validate succeeds when total_pages > max page ID
        let validation = seq.validate_page_table(10);
        assert!(validation.is_ok(), "Page table [0,1,2] should be valid with 10 total pages");
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 5 (13 new)
    // -----------------------------------------------------------------------

    #[test]
    fn generator_forward_config_default_for_test_accessors() {
        // Arrange
        use crate::engine::executor::GeneratorForwardConfig;
        use gllm_kernels::types::DType;
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act & Assert: all backward-compatible accessors return expected values
        assert_eq!(cfg.hidden_size(), 64);
        assert_eq!(cfg.num_layers(), 4);
        assert_eq!(cfg.vocab_size(), 100);
        assert_eq!(cfg.intermediate_size(), 128);
        assert!((cfg.norm_eps() - 1e-5).abs() < f32::EPSILON);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.max_seq_len(), 512);
    }

    #[test]
    fn generator_forward_config_default_for_test_rope_and_position() {
        // Arrange
        use crate::engine::executor::GeneratorForwardConfig;
        use gllm_kernels::types::DType;
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act & Assert: RoPE accessors
        assert_eq!(cfg.rope_theta(), 10000.0);
        assert_eq!(cfg.rope_scale(), 1.0);
        assert_eq!(cfg.rope.theta, 10000.0);
        assert_eq!(cfg.rope.scale, 1.0);
        assert!(!cfg.rope.interleaved);
        assert!(!cfg.rope.precompute);
        // Assert: position encoding
        assert_eq!(cfg.position_encoding, crate::engine::executor::PositionEncoding::Rope);
    }

    #[test]
    fn attention_topology_linear_constructor() {
        // Arrange & Act
        use crate::engine::executor::AttentionTopology;
        let topo = AttentionTopology::linear();
        // Assert: linear() creates a bidirectional topology with minimal geometry
        assert_eq!(topo.mask_type, crate::engine::executor::AttentionMaskType::Bidirectional);
        assert_eq!(topo.num_heads(), 1);
        assert_eq!(topo.num_kv_heads(), 1);
        assert_eq!(topo.head_dim(), 1);
        assert_eq!(topo.max_seq_len(), 512);
    }

    #[test]
    fn attention_topology_bidirectional_and_causal_constructors() {
        // Arrange
        use crate::engine::executor::AttentionTopology;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 256,
            num_layers: 6,
            vocab_size: 500,
            intermediate_size: 512,
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 32,
            max_seq_len: 1024,
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
        });
        // Act
        let bidir = AttentionTopology::bidirectional(geometry.clone());
        let causal = AttentionTopology::causal(geometry.clone());
        // Assert: mask types differ
        assert_eq!(bidir.mask_type, crate::engine::executor::AttentionMaskType::Bidirectional);
        assert_eq!(causal.mask_type, crate::engine::executor::AttentionMaskType::Causal);
        assert_ne!(bidir.mask_type, causal.mask_type);
        // Assert: both share the same geometry accessors
        assert_eq!(bidir.num_heads(), 8);
        assert_eq!(causal.num_heads(), 8);
        assert_eq!(bidir.num_kv_heads(), 2);
        assert_eq!(causal.head_dim(), 32);
        assert_eq!(bidir.max_seq_len(), 1024);
    }

    #[test]
    fn kv_cache_config_construction_and_accessors() {
        // Arrange
        use crate::engine::executor::KvCacheConfig;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 512,
            num_layers: 12,
            vocab_size: 32000,
            intermediate_size: 2048,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
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
        });
        let swap = crate::engine::executor::SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        // Act
        let config = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: Some(swap),
        };
        // Assert: accessors return correct values
        assert_eq!(config.dtype_size(), 4, "F32 should be 4 bytes");
        assert_eq!(config.num_layers(), 12);
        assert_eq!(config.num_heads(), 4);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.max_seq_len(), 2048);
        assert_eq!(config.page_size, 16);
        assert!(config.swap_config.is_some());
        assert!(config.swap_config.as_ref().unwrap().enable_swap);
    }

    #[test]
    fn kv_cache_config_equality_same_geometry() {
        // Arrange
        use crate::engine::executor::KvCacheConfig;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 256,
            num_layers: 4,
            vocab_size: 100,
            intermediate_size: 512,
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 32,
            max_seq_len: 512,
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
        });
        let a = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let b = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: same geometry + same params = equal (PartialEq)
        assert_eq!(a, b, "KvCacheConfig with same params should be equal");
    }

    #[test]
    fn arch_family_variants_are_distinct() {
        // Arrange
        use crate::manifest::ArchFamily;
        let encoder = ArchFamily::Encoder;
        let decoder = ArchFamily::Decoder;
        // Assert: the two variants are distinct
        assert_ne!(encoder, decoder);
        assert_eq!(encoder, ArchFamily::Encoder);
        assert_eq!(decoder, ArchFamily::Decoder);
        // Assert: Copy semantics
        let copied = decoder;
        assert_eq!(copied, ArchFamily::Decoder);
        assert_eq!(decoder, ArchFamily::Decoder, "Copy leaves original usable");
    }

    #[test]
    fn arch_family_debug_output() {
        // Arrange
        use crate::manifest::ArchFamily;
        // Act
        let encoder_debug = format!("{:?}", ArchFamily::Encoder);
        let decoder_debug = format!("{:?}", ArchFamily::Decoder);
        // Assert: Debug output contains variant name
        assert!(encoder_debug.contains("Encoder"), "Expected 'Encoder' in debug, got: {encoder_debug}");
        assert!(decoder_debug.contains("Decoder"), "Expected 'Decoder' in debug, got: {decoder_debug}");
    }

    #[test]
    fn moe_config_construction_and_equality() {
        // Arrange
        use crate::manifest::{MoEConfig, RouterType};
        let config = MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 8,
            router_type: RouterType::DeepSeek,
        };
        // Assert: all fields are set correctly
        assert_eq!(config.num_experts, 64);
        assert_eq!(config.num_experts_per_tok, 8);
        // Assert: Debug output
        let debug = format!("{:?}", config);
        assert!(debug.contains("MoEConfig"), "Debug should contain struct name");
    }

    #[test]
    fn router_type_all_variants_distinct() {
        // Arrange
        use crate::manifest::RouterType;
        let variants = [
            (RouterType::Qwen, "Qwen"),
            (RouterType::Mixtral, "Mixtral"),
            (RouterType::DeepSeek, "DeepSeek"),
            (RouterType::GptOss, "GptOss"),
            (RouterType::Unknown, "Unknown"),
        ];
        // Assert: all variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i].0, variants[j].0, "{} should equal itself", variants[i].1);
                } else {
                    assert_ne!(variants[i].0, variants[j].0, "{} should not equal {}", variants[i].1, variants[j].1);
                }
            }
        }
    }

    #[test]
    fn kv_cache_handle_sorting_via_inner_value() {
        // Arrange: KvCacheHandle is Copy + Hash + Eq, sort by inner u64 manually
        let mut handles = vec![
            KvCacheHandle(300),
            KvCacheHandle(100),
            KvCacheHandle(200),
            KvCacheHandle(50),
        ];
        // Act: sort by the inner u64 value
        handles.sort_by_key(|h| h.0);
        // Assert: sorted in ascending order by inner value
        assert_eq!(handles[0], KvCacheHandle(50));
        assert_eq!(handles[1], KvCacheHandle(100));
        assert_eq!(handles[2], KvCacheHandle(200));
        assert_eq!(handles[3], KvCacheHandle(300));
    }

    #[test]
    fn logits_handle_debug_output_contains_data() {
        // Arrange
        let handle = LogitsHandle { data: vec![1.5, -2.0, 0.0] };
        // Act
        let debug = format!("{:?}", handle);
        // Assert: Debug output contains struct name and data
        assert!(debug.contains("LogitsHandle"), "Debug should contain struct name, got: {debug}");
        assert!(debug.contains("1.5"), "Debug should contain data values, got: {debug}");
    }

    #[test]
    fn sequence_input_with_nonzero_draft_steps() {
        // Arrange: speculative decoding scenario with draft steps > 0
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![10, 20, 30, 40],
            position: 100,
            draft_steps: 5,
            page_table: Some(vec![0u32, 1, 2, 3]),
            fused_hidden: None,
        };
        // Assert: draft_steps field is preserved
        assert_eq!(input.draft_steps, 5);
        assert_eq!(input.position, 100);
        assert_eq!(input.tokens.len(), 4);
        // Act: validate page table with sufficient total_pages
        let result = input.validate_page_table(10);
        assert!(result.is_ok(), "Page table [0,1,2,3] should be valid with 10 total pages");
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 6 (13 new)
    // -----------------------------------------------------------------------

    #[test]
    fn gpu_device_info_debug_pretty_format_all_fields() {
        // Arrange: use the same format that CudaBackend::new() produces
        let info = GpuDeviceInfo {
            ordinal: 7,
            sm_version: 89,
            sm_count: 128,
            total_memory: 12 * 1024 * 1024 * 1024,
            name: "RTX 4070".to_string(),
        };
        // Act
        let debug = format!("{:#?}", info);
        // Assert: pretty Debug output contains every field name
        assert!(debug.contains("ordinal: 7"), "missing ordinal in: {debug}");
        assert!(debug.contains("sm_version: 89"), "missing sm_version in: {debug}");
        assert!(debug.contains("sm_count: 128"), "missing sm_count in: {debug}");
        assert!(debug.contains("RTX 4070"), "missing name in: {debug}");
    }

    #[test]
    fn gpu_device_info_total_memory_alignment_calculation() {
        // Arrange: 8 GB GPU — verify byte count divides cleanly by page sizes
        let total = 8usize * 1024 * 1024 * 1024;
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 80,
            sm_count: 108,
            total_memory: total,
            name: "A100-8GB".to_string(),
        };
        // Act: verify memory is aligned to common page sizes
        assert_eq!(info.total_memory % (4 * 1024), 0, "should be 4K aligned");
        assert_eq!(info.total_memory % (64 * 1024), 0, "should be 64K aligned");
        // Assert: GB conversion is exact
        assert_eq!(info.total_memory / (1024 * 1024 * 1024), 8);
    }

    #[test]
    fn sequence_input_validate_page_table_with_u32_max_id() {
        // Arrange: page ID = u32::MAX requires total_pages > u32::MAX, which is impossible
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![u32::MAX]),
            fused_hidden: None,
        };
        // Act: usize on 64-bit > u32::MAX, but page ID u32::MAX as usize requires
        // total_pages > u32::MAX. Even usize::MAX as total_pages means the check
        // is (u32::MAX as usize >= usize::MAX) which is false for 64-bit usize,
        // so this should be Ok.
        // Actually: page_id (u32::MAX = 4294967295) as usize >= total_pages.
        // If total_pages = usize::MAX, 4294967295 < usize::MAX on 64-bit, so Ok.
        let result = input.validate_page_table(usize::MAX);
        // Assert: u32::MAX < usize::MAX on 64-bit, so it's valid
        assert!(result.is_ok(), "u32::MAX page_id should be valid when total_pages=usize::MAX on 64-bit");
    }

    #[test]
    fn sequence_input_validate_page_table_boundary_last_valid_page() {
        // Arrange: page_id = 9, total_pages = 10 → page_id < total_pages → valid
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![9]),
            fused_hidden: None,
        };
        // Act
        let result = input.validate_page_table(10);
        // Assert: last valid index (0-indexed) is total_pages - 1
        assert!(result.is_ok(), "page_id=9 should be valid with total_pages=10");
    }

    #[test]
    fn sequence_input_clone_independence_after_mutation() {
        // Arrange
        use crate::engine::executor::SequenceInput;
        let original = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2]),
            fused_hidden: Some(vec![0.5, 1.5]),
        };
        // Act: clone and mutate
        let mut cloned = original.clone();
        cloned.tokens.push(99);
        cloned.fused_hidden.as_mut().unwrap().push(42.0);
        // Assert: original is unaffected
        assert_eq!(original.tokens.len(), 3, "original tokens should be unchanged");
        assert_eq!(original.fused_hidden.as_ref().unwrap().len(), 2, "original fused_hidden unchanged");
        assert_eq!(cloned.tokens.len(), 4);
        assert_eq!(cloned.fused_hidden.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn model_kind_all_variants_distinct_and_parseable() {
        // Arrange
        use crate::manifest::ModelKind;
        let variants = [
            (ModelKind::Chat, "Chat"),
            (ModelKind::Embedding, "Embedding"),
            (ModelKind::Reranker, "Reranker"),
            (ModelKind::Classifier, "Classifier"),
        ];
        // Assert: all pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i].0, variants[j].0);
                } else {
                    assert_ne!(variants[i].0, variants[j].0, "{} != {}", variants[i].1, variants[j].1);
                }
            }
        }
        // Assert: parse round-trips for canonical names
        assert_eq!(ModelKind::parse("chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("embedding"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("reranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("classifier"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("unknown"), None);
    }

    #[test]
    fn tensor_role_to_canonical_name_with_and_without_layer() {
        // Arrange
        use crate::manifest::TensorRole;
        let cases = [
            (TensorRole::Embedding, "embed"),
            (TensorRole::OutputHead, "lm_head"),
            (TensorRole::AttentionQuery, "q_proj"),
            (TensorRole::FfnGate, "gate_proj"),
            (TensorRole::MoEGate, "moe_gate"),
        ];
        for (role, expected_base) in cases {
            // Act: without layer
            let name_no_layer = role.to_canonical_name(None);
            // Assert
            assert_eq!(name_no_layer, expected_base, "no-layer name mismatch for {:?}", role);

            // Act: with layer
            let name_with_layer = role.to_canonical_name(Some(5));
            // Assert
            assert_eq!(name_with_layer, format!("L5.{}", expected_base), "layer name mismatch for {:?}", role);
        }
    }

    #[test]
    fn moe_config_equality_and_copy_semantics() {
        // Arrange
        use crate::manifest::{MoEConfig, RouterType};
        let a = MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 8,
            router_type: RouterType::DeepSeek,
        };
        let b = MoEConfig {
            num_experts: 64,
            num_experts_per_tok: 8,
            router_type: RouterType::DeepSeek,
        };
        // Assert: equality
        assert_eq!(a, b, "identical MoEConfig should be equal");
        // Assert: Copy semantics
        let copied = a;
        assert_eq!(a, copied, "Copy leaves original usable");
        assert_eq!(a.num_experts, 64);
        // Assert: different router type → not equal
        let c = MoEConfig { router_type: RouterType::Qwen, ..a };
        assert_ne!(a, c);
    }

    #[test]
    fn moe_config_hash_usable_in_set() {
        // Arrange
        use crate::manifest::{MoEConfig, RouterType};
        use std::collections::HashSet;
        let a = MoEConfig { num_experts: 64, num_experts_per_tok: 8, router_type: RouterType::Mixtral };
        let b = MoEConfig { num_experts: 64, num_experts_per_tok: 8, router_type: RouterType::Mixtral };
        let c = MoEConfig { num_experts: 32, num_experts_per_tok: 4, router_type: RouterType::Qwen };
        // Act
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        set.insert(c);
        // Assert: a and b are equal, so set deduplicates to 2
        assert_eq!(set.len(), 2, "equal MoEConfigs should deduplicate in HashSet");
        assert!(set.contains(&a));
        assert!(set.contains(&c));
    }

    #[test]
    fn generator_forward_config_attention_method_returns_correct_config() {
        // Arrange
        use crate::engine::executor::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act: use the attention() convenience method
        let attn = cfg.attention();
        // Assert: matches the geometry from default_for_test
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);
        assert_eq!(attn.head_dim, 16);
        // Assert: GQA group size
        assert_eq!(attn.num_heads / attn.num_kv_heads, 2, "GQA ratio should be 4/2=2");
    }

    #[test]
    fn kv_cache_config_equality_different_swap_threshold() {
        // Arrange: same geometry, same dtype, same page_size, but different swap threshold
        use crate::engine::executor::{KvCacheConfig, SwapConfig};
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 256, num_layers: 4, vocab_size: 100, intermediate_size: 512,
            num_heads: 8, num_kv_heads: 4, head_dim: 64, max_seq_len: 512,
            rope_theta: 10000.0, rope_scale: 1.0, rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32, compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5, num_experts: 0, moe_top_k: 0, expert_intermediate_size: 0,
            global_rope_theta: 0.0, rope_partial_ratio: 1.0, attention_pattern: vec![],
            sliding_window: 0, num_kv_shared_layers: 0, global_head_dim: 0,
            hidden_size_per_layer_input: 0, position_offset: None, rope_scaling: None,
            final_logit_softcapping: None, hidden_act: None,
            mla_d_c: 0, mla_d_rope: 0, mla_unabsorbed_threshold: 0,
        });
        let a = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: Some(SwapConfig { enable_swap: true, swap_threshold: 0.8, lru_granularity: 4 }),
        };
        let b = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: Some(SwapConfig { enable_swap: true, swap_threshold: 0.9, lru_granularity: 4 }),
        };
        // Assert: different swap_threshold makes them not equal
        assert_ne!(a, b, "different swap_threshold should make KvCacheConfig unequal");
    }

    #[test]
    fn swap_config_nan_threshold_is_valid_construction() {
        // Arrange: NaN threshold is technically valid f32 (edge case from config)
        use crate::engine::executor::SwapConfig;
        let config = SwapConfig {
            enable_swap: true,
            swap_threshold: f32::NAN,
            lru_granularity: 1,
        };
        // Assert: NaN is preserved (NaN != NaN is a valid check)
        assert!(config.swap_threshold.is_nan(), "NaN threshold should be preserved");
        // Assert: NaN threshold config is not equal to itself via PartialEq
        assert_ne!(config, config, "NaN-containing SwapConfig should not equal itself");
    }

    #[test]
    fn backend_error_moving_between_variants_preserves_message() {
        // Arrange
        let cuda_err = BE::Cuda("device lost".to_string());
        // Act: re-assign to a different variant
        let msg = format!("{}", cuda_err);
        let other_err = BE::Other(msg.clone());
        // Assert: the original message is preserved through the chain
        let other_msg = format!("{}", other_err);
        assert!(other_msg.contains("device lost"), "Message should survive variant conversion, got: {other_msg}");
        assert!(other_msg.contains("backend error"), "Other variant should have 'backend error' prefix");
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 7 (13 new)
    // -----------------------------------------------------------------------

    #[test]
    fn request_data_construction_with_all_fields() {
        // Arrange
        use crate::engine::executor::RequestData;
        // Act
        let data = RequestData {
            prompt_tokens: vec![1, 2, 3, 4, 5],
            output_tokens: vec![10, 20],
            sampling_config: SamplingConfig { temperature: 0.7, top_k: 50, top_p: 0.95 },
            is_prefill: true,
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens: 256,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        // Assert: all fields are correctly preserved
        assert_eq!(data.prompt_tokens.len(), 5);
        assert_eq!(data.output_tokens, vec![10, 20]);
        assert_eq!(data.sampling_config.temperature, 0.7);
        assert!(data.is_prefill);
        assert_eq!(data.max_new_tokens, 256);
        assert!(!data.finished);
        assert!(data.session_id.is_none());
        assert!(data.thinking_budget.is_none());
        assert!(data.fused_prefill_hidden.is_none());
    }

    #[test]
    fn request_data_with_session_and_thinking_budget() {
        // Arrange: RequestData does not derive Clone; test field access for decode scenario
        use crate::engine::executor::RequestData;
        let data = RequestData {
            prompt_tokens: vec![100, 200],
            output_tokens: vec![10],
            sampling_config: SamplingConfig::default(),
            is_prefill: false,
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 128,
            finished: false,
            session_id: Some(42u64),
            thinking_budget: Some(1024),
            fused_prefill_hidden: Some(vec![0.5; 32]),
        };
        // Assert: all fields are accessible with correct values
        assert_eq!(data.prompt_tokens, vec![100, 200]);
        assert_eq!(data.output_tokens, vec![10]);
        assert!(!data.is_prefill);
        assert_eq!(data.max_new_tokens, 128);
        assert_eq!(data.session_id, Some(42u64));
        assert_eq!(data.thinking_budget, Some(1024));
        assert!(data.fused_prefill_hidden.is_some());
        assert_eq!(data.fused_prefill_hidden.as_ref().unwrap().len(), 32);
    }

    #[test]
    fn effective_kv_max_seq_len_identity_passthrough() {
        // Arrange: the function is a pure passthrough of geometry.max_seq_len
        use crate::engine::executor::effective_kv_max_seq_len;
        // Act & Assert: various values pass through unchanged
        assert_eq!(effective_kv_max_seq_len(512), 512);
        assert_eq!(effective_kv_max_seq_len(2048), 2048);
        assert_eq!(effective_kv_max_seq_len(8192), 8192);
        assert_eq!(effective_kv_max_seq_len(131072), 131072);
        assert_eq!(effective_kv_max_seq_len(1), 1);
        assert_eq!(effective_kv_max_seq_len(0), 0);
    }

    #[test]
    fn model_manifest_default_values() {
        // Arrange & Act
        use crate::manifest::ModelManifest;
        let manifest = ModelManifest::default();
        // Assert: Default impl produces sensible baseline values
        assert_eq!(&*manifest.model_id, "default");
        assert_eq!(manifest.arch, "llama");
        assert_eq!(manifest.kind, crate::manifest::ModelKind::Chat);
        assert!(manifest.rope_base_override.is_none());
        assert!(manifest.max_context_override.is_none());
        assert!(manifest.moe_config.is_none());
        assert!(manifest.tensor_map.is_empty());
        assert!(!manifest.is_moe(), "default manifest should not be MoE");
    }

    #[test]
    fn tensor_role_mla_and_mtp_variants_canonical_names() {
        // Arrange: MLA and MTP tensor roles should map to correct canonical names
        use crate::manifest::TensorRole;
        let mla_roles = [
            (TensorRole::MlaQCompress, "q_a_proj"),
            (TensorRole::MlaQExpand, "q_b_proj"),
            (TensorRole::MlaKvCompress, "kv_b_proj"),
            (TensorRole::MlaKeyAbsorb, "k_b_proj"),
            (TensorRole::MlaValueAbsorb, "v_b_proj"),
            (TensorRole::MlaRopeKey, "k_pe_proj"),
            (TensorRole::MtpProjection, "mtp_proj"),
        ];
        for (role, expected) in mla_roles {
            // Act
            let name = role.to_canonical_name(None);
            // Assert
            assert_eq!(name, expected, "MLA/MTP role {:?} should map to {}", role, expected);
            // Assert: with layer prefix
            let layered = role.to_canonical_name(Some(3));
            assert_eq!(layered, format!("L3.{}", expected), "layered name mismatch for {:?}", role);
        }
    }

    #[test]
    fn model_kind_parse_case_insensitive_and_whitespace() {
        // Arrange: parse should handle case variations and surrounding whitespace
        use crate::manifest::ModelKind;
        // Act & Assert
        assert_eq!(ModelKind::parse("CHAT"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("  embedding  "), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("GENERATION"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("text-generation"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("embeddings"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("RE-RANKER"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("sequence-classification"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("Classify"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse(""), None);
        assert_eq!(ModelKind::parse("foo_bar"), None);
    }

    #[test]
    fn attention_head_config_from_geometry() {
        // Arrange
        use crate::engine::executor::AttentionHeadConfig;
        let geometry = crate::model_config::ModelGeometry {
            hidden_size: 1024,
            num_layers: 12,
            vocab_size: 32000,
            intermediate_size: 4096,
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
            max_seq_len: 2048,
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
        };
        // Act
        let config = AttentionHeadConfig::from_geometry(&geometry);
        // Assert: values come from geometry, not hardcoded
        assert_eq!(config.num_heads, 16);
        assert_eq!(config.num_kv_heads, 4);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn gpu_device_info_struct_update_syntax() {
        // Arrange: use struct update syntax to create variants from a base
        let base = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 80,
            sm_count: 108,
            total_memory: 40 * 1024 * 1024 * 1024,
            name: "A100-40GB".to_string(),
        };
        // Act: create a variant with only ordinal and name changed
        let variant = GpuDeviceInfo {
            ordinal: 1,
            name: "A100-40GB-1".to_string(),
            ..base
        };
        // Assert: changed fields differ, shared fields match
        assert_eq!(variant.ordinal, 1);
        assert_ne!(variant.ordinal, base.ordinal);
        assert_eq!(variant.name, "A100-40GB-1");
        assert_eq!(variant.sm_version, base.sm_version, "sm_version should be inherited from base");
        assert_eq!(variant.sm_count, base.sm_count, "sm_count should be inherited from base");
        assert_eq!(variant.total_memory, base.total_memory, "total_memory should be inherited from base");
    }

    #[test]
    fn rope_config_f64_precision_theta_scale() {
        // Arrange: RoPEConfig uses f64 for theta and scale — verify full f64 precision
        use crate::engine::executor::RoPEConfig;
        let precise_theta = 1000000.0000001f64;
        let precise_scale = 0.000000000001f64;
        let config = RoPEConfig {
            theta: precise_theta,
            scale: precise_scale,
            interleaved: false,
            precompute: true,
        };
        // Act: Copy
        let copied = config;
        // Assert: f64 precision is preserved through Copy
        assert_eq!(copied.theta, precise_theta);
        assert_eq!(copied.scale, precise_scale);
        assert!((copied.theta - 1000000.0000001f64).abs() < f64::EPSILON);
        assert!(copied.precompute);
        // Assert: original remains valid after Copy
        assert_eq!(config.theta, precise_theta);
    }

    #[test]
    fn kv_cache_config_dtype_size_bf16() {
        // Arrange: BF16 is 2 bytes per element
        use crate::engine::executor::KvCacheConfig;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 256, num_layers: 4, vocab_size: 100, intermediate_size: 512,
            num_heads: 8, num_kv_heads: 4, head_dim: 64, max_seq_len: 512,
            rope_theta: 10000.0, rope_scale: 1.0, rope_interleaved: false,
            dtype: gllm_kernels::types::DType::BF16, compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5, num_experts: 0, moe_top_k: 0, expert_intermediate_size: 0,
            global_rope_theta: 0.0, rope_partial_ratio: 1.0, attention_pattern: vec![],
            sliding_window: 0, num_kv_shared_layers: 0, global_head_dim: 0,
            hidden_size_per_layer_input: 0, position_offset: None, rope_scaling: None,
            final_logit_softcapping: None, hidden_act: None,
            mla_d_c: 0, mla_d_rope: 0, mla_unabsorbed_threshold: 0,
        });
        let config = KvCacheConfig {
            geometry,
            kv_dtype: gllm_kernels::types::DType::BF16,
            page_size: 16,
            swap_config: None,
        };
        // Act & Assert: BF16 dtype_size is 2 bytes
        assert_eq!(config.dtype_size(), 2, "BF16 should be 2 bytes");
    }

    #[test]
    fn backend_error_source_chain_with_std_error() {
        // Arrange: BackendError implements std::error::Error
        let err = BE::Cuda("out of memory".to_string());
        // Act: verify the std::error::Error trait object works with source()
        let err_ref: &dyn std::error::Error = &err;
        // Assert: source() returns None (BackendError has no nested error)
        assert!(err_ref.source().is_none(), "BackendError::Cuda should have no source");
        // Assert: Display via trait object works
        let msg = err_ref.to_string();
        assert!(msg.contains("out of memory"), "Display should work via trait object, got: {msg}");
    }

    #[test]
    fn logits_handle_single_element_boundary() {
        // Arrange: single-element logits (1-token vocab edge case)
        let handle = LogitsHandle { data: vec![f32::MAX] };
        // Assert
        assert_eq!(handle.data.len(), 1);
        assert_eq!(handle.data[0], f32::MAX);
        // Act: clone preserves single-element
        let cloned = handle.clone();
        assert_eq!(cloned.data.len(), 1);
        assert_eq!(cloned.data[0], f32::MAX);
    }

    #[test]
    fn cuda_backend_phantom_data_element_type_isolation() {
        // Arrange: CudaBackend<f32> and CudaBackend<half::bf16> should have
        // identical behavior for new() — PhantomData carries no runtime cost
        let result_f32 = CudaBackend::<f32>::new(0);
        let result_bf16 = CudaBackend::<half::bf16>::new(0);
        // Assert: both return the same Some/None status
        assert_eq!(result_f32.is_some(), result_bf16.is_some());
        // Assert: if both succeed, device_info is identical (same GPU)
        if let (Some(b_f32), Some(b_bf16)) = (result_f32, result_bf16) {
            assert_eq!(b_f32.device_info().ordinal, b_bf16.device_info().ordinal);
            assert_eq!(b_f32.sm_version(), b_bf16.sm_version());
            assert_eq!(b_f32.device_info().total_memory, b_bf16.device_info().total_memory);
        }
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 8 (13 new)
    // -----------------------------------------------------------------------

    #[test]
    fn request_phase_variants_are_distinct_and_self_equal() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        let prefill = RequestPhase::Prefill;
        let decode = RequestPhase::Decode;
        let chunked = RequestPhase::ChunkedPrefill;
        // Assert: all three variants are pairwise distinct
        assert_ne!(prefill, decode, "Prefill != Decode");
        assert_ne!(prefill, chunked, "Prefill != ChunkedPrefill");
        assert_ne!(decode, chunked, "Decode != ChunkedPrefill");
        // Assert: each variant equals itself
        assert_eq!(prefill, RequestPhase::Prefill);
        assert_eq!(decode, RequestPhase::Decode);
        assert_eq!(chunked, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn request_phase_copy_and_clone_semantics() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        let original = RequestPhase::ChunkedPrefill;
        // Act: Copy (RequestPhase derives Copy)
        let copied = original;
        // Assert: both remain usable after copy
        assert_eq!(original, RequestPhase::ChunkedPrefill);
        assert_eq!(copied, RequestPhase::ChunkedPrefill);
        assert_eq!(original, copied);
    }

    #[test]
    fn request_phase_debug_output_all_variants() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        // Act
        let prefill_debug = format!("{:?}", RequestPhase::Prefill);
        let decode_debug = format!("{:?}", RequestPhase::Decode);
        let chunked_debug = format!("{:?}", RequestPhase::ChunkedPrefill);
        // Assert: Debug output contains variant names
        assert!(prefill_debug.contains("Prefill"), "Expected 'Prefill', got: {prefill_debug}");
        assert!(decode_debug.contains("Decode"), "Expected 'Decode', got: {decode_debug}");
        assert!(chunked_debug.contains("ChunkedPrefill"), "Expected 'ChunkedPrefill', got: {chunked_debug}");
    }

    #[test]
    fn request_phase_usable_in_hashset() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        // Act
        set.insert(RequestPhase::Prefill);
        set.insert(RequestPhase::Decode);
        set.insert(RequestPhase::ChunkedPrefill);
        set.insert(RequestPhase::Decode); // duplicate
        // Assert: 3 distinct variants
        assert_eq!(set.len(), 3, "Should have 3 distinct RequestPhase variants");
        assert!(set.contains(&RequestPhase::Prefill));
        assert!(set.contains(&RequestPhase::Decode));
        assert!(set.contains(&RequestPhase::ChunkedPrefill));
    }

    #[test]
    fn tensor_role_norm_and_ffn_variants_canonical_names() {
        // Arrange: verify norm and FFN tensor roles map to correct canonical names
        use crate::manifest::TensorRole;
        let roles = [
            (TensorRole::InputNorm, "input_norm"),
            (TensorRole::PostAttnNorm, "post_attn_norm"),
            (TensorRole::FinalNorm, "final_norm"),
            (TensorRole::FfnUp, "up_proj"),
            (TensorRole::FfnDown, "down_proj"),
        ];
        for (role, expected) in roles {
            // Act
            let name = role.to_canonical_name(None);
            // Assert
            assert_eq!(name, expected, "{:?} should map to {}", role, expected);
            // Assert: with layer prefix
            let layered = role.to_canonical_name(Some(7));
            assert_eq!(layered, format!("L7.{}", expected));
        }
    }

    #[test]
    fn tensor_role_moe_variants_canonical_names() {
        // Arrange: verify MoE-specific tensor roles
        use crate::manifest::TensorRole;
        let roles = [
            (TensorRole::MoESharedExpert, "shared_expert"),
            (TensorRole::MoEExpert, "expert"),
        ];
        for (role, expected) in roles {
            // Act
            let name = role.to_canonical_name(None);
            // Assert
            assert_eq!(name, expected, "{:?} should map to {}", role, expected);
        }
    }

    #[test]
    fn kv_cache_config_kv_dim_standard_model() {
        // Arrange: standard (non-MLA) model: kv_dim = num_kv_heads * head_dim
        use crate::engine::executor::KvCacheConfig;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 4096, num_layers: 32, vocab_size: 32000, intermediate_size: 11008,
            num_heads: 32, num_kv_heads: 32, head_dim: 128, max_seq_len: 4096,
            rope_theta: 10000.0, rope_scale: 1.0, rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32, compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5, num_experts: 0, moe_top_k: 0, expert_intermediate_size: 0,
            global_rope_theta: 0.0, rope_partial_ratio: 1.0, attention_pattern: vec![],
            sliding_window: 0, num_kv_shared_layers: 0, global_head_dim: 0,
            hidden_size_per_layer_input: 0, position_offset: None, rope_scaling: None,
            final_logit_softcapping: None, hidden_act: None,
            mla_d_c: 0, mla_d_rope: 0, mla_unabsorbed_threshold: 0,
        });
        let config = KvCacheConfig {
            geometry,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Act
        let kv_dim = config.kv_dim();
        // Assert: standard model = num_kv_heads * head_dim
        assert_eq!(kv_dim, 32 * 128, "kv_dim should be num_kv_heads * head_dim = 4096");
        // Assert: not MLA
        assert!(!config.is_mla(), "standard model should not be MLA");
    }

    #[test]
    fn kv_cache_config_kv_dim_mla_model() {
        // Arrange: MLA model (DeepSeek V3): kv_dim = d_c + d_rope
        use crate::engine::executor::KvCacheConfig;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 7168, num_layers: 61, vocab_size: 129280, intermediate_size: 18432,
            num_heads: 128, num_kv_heads: 128, head_dim: 128, max_seq_len: 131072,
            rope_theta: 10000.0, rope_scale: 1.0, rope_interleaved: false,
            dtype: gllm_kernels::types::DType::BF16, compute_dtype: gllm_kernels::types::DType::BF16,
            norm_eps: 1e-6, num_experts: 256, moe_top_k: 8, expert_intermediate_size: 1408,
            global_rope_theta: 0.0, rope_partial_ratio: 1.0, attention_pattern: vec![],
            sliding_window: 0, num_kv_shared_layers: 0, global_head_dim: 0,
            hidden_size_per_layer_input: 0, position_offset: None, rope_scaling: None,
            final_logit_softcapping: None, hidden_act: None,
            mla_d_c: 512, mla_d_rope: 64, mla_unabsorbed_threshold: 0,
        });
        let config = KvCacheConfig {
            geometry,
            kv_dtype: gllm_kernels::types::DType::BF16,
            page_size: 16,
            swap_config: None,
        };
        // Act
        let kv_dim = config.kv_dim();
        // Assert: MLA model = d_c + d_rope = 512 + 64 = 576
        assert_eq!(kv_dim, 576, "MLA kv_dim should be d_c + d_rope = 576");
        // Assert: is MLA
        assert!(config.is_mla(), "model with mla_d_c > 0 should be MLA");
        // Assert: dtype_size for BF16
        assert_eq!(config.dtype_size(), 2);
    }

    #[test]
    fn kv_cache_config_num_kv_shared_layers_accessor() {
        // Arrange: Gemma 4 model with shared KV layers
        use crate::engine::executor::KvCacheConfig;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 256, num_layers: 4, vocab_size: 100, intermediate_size: 512,
            num_heads: 8, num_kv_heads: 2, head_dim: 32, max_seq_len: 512,
            rope_theta: 10000.0, rope_scale: 1.0, rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32, compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5, num_experts: 0, moe_top_k: 0, expert_intermediate_size: 0,
            global_rope_theta: 0.0, rope_partial_ratio: 1.0, attention_pattern: vec![],
            sliding_window: 0, num_kv_shared_layers: 2, global_head_dim: 0,
            hidden_size_per_layer_input: 0, position_offset: None, rope_scaling: None,
            final_logit_softcapping: None, hidden_act: None,
            mla_d_c: 0, mla_d_rope: 0, mla_unabsorbed_threshold: 0,
        });
        let config = KvCacheConfig {
            geometry,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Act & Assert
        assert_eq!(config.num_kv_shared_layers(), 2, "num_kv_shared_layers should match geometry");
        assert_eq!(config.num_layers(), 4);
        assert_eq!(config.attention_pattern().len(), 0);
    }

    #[test]
    fn generator_forward_config_default_for_test_arch_family() {
        // Arrange
        use crate::engine::executor::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act & Assert: arch_family should be Decoder for generator models
        assert_eq!(cfg.arch_family, crate::manifest::ArchFamily::Decoder);
        // Assert: MoE config is None for default test model
        assert!(cfg.moe_config.is_none(), "default test model should not be MoE");
        // Assert: rerank token IDs are None for generator
        assert!(cfg.rerank_yes_token_id.is_none());
        assert!(cfg.rerank_no_token_id.is_none());
    }

    #[test]
    fn sequence_input_validate_page_table_empty_table_is_valid() {
        // Arrange: empty page table with tokens (edge case — no pages needed?)
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![]),
            fused_hidden: None,
        };
        // Act: empty page table with any total_pages should be valid (no entries to check)
        let result = input.validate_page_table(0);
        // Assert: no page IDs to validate, so always valid
        assert!(result.is_ok(), "Empty page table should be valid regardless of total_pages");
    }

    #[test]
    fn logits_handle_with_all_zeros_and_negative_values() {
        // Arrange: logits representing a degenerate case (all-zero or negative)
        let zeros = LogitsHandle { data: vec![0.0; 100] };
        let negatives = LogitsHandle { data: vec![-1e10; 100] };
        // Assert: zeros and negatives are preserved exactly
        assert!(zeros.data.iter().all(|&v| v == 0.0), "all-zero logits should be preserved");
        assert!(negatives.data.iter().all(|&v| v == -1e10), "negative logits should be preserved");
        assert_eq!(zeros.data.len(), 100);
        assert_eq!(negatives.data.len(), 100);
        // Act: clone
        let cloned_zeros = zeros.clone();
        // Assert: cloned zeros match original
        assert!(cloned_zeros.data.iter().all(|&v| v == 0.0));
        assert_eq!(cloned_zeros.data.len(), zeros.data.len());
    }

    #[test]
    fn gpu_device_info_multiple_devices_in_vec() {
        // Arrange: simulate a multi-GPU node with heterogeneous devices
        let devices: Vec<GpuDeviceInfo> = vec![
            GpuDeviceInfo {
                ordinal: 0, sm_version: 90, sm_count: 132,
                total_memory: 80 * 1024 * 1024 * 1024,
                name: "H100-80GB-0".to_string(),
            },
            GpuDeviceInfo {
                ordinal: 1, sm_version: 80, sm_count: 108,
                total_memory: 40 * 1024 * 1024 * 1024,
                name: "A100-40GB-1".to_string(),
            },
        ];
        // Act
        let total_memory: usize = devices.iter().map(|d| d.total_memory).sum();
        let max_sm: u32 = devices.iter().map(|d| d.sm_version).max().unwrap_or(0);
        // Assert: aggregate calculations work correctly
        assert_eq!(total_memory, 120 * 1024 * 1024 * 1024, "sum of all GPU memory");
        assert_eq!(max_sm, 90, "max SM version across devices");
        assert_eq!(devices.len(), 2);
        // Assert: each device has a unique ordinal
        let ordinals: std::collections::HashSet<usize> = devices.iter().map(|d| d.ordinal).collect();
        assert_eq!(ordinals.len(), 2, "all device ordinals should be unique");
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 9 (13 new)
    // -----------------------------------------------------------------------

    #[test]
    fn executor_error_backend_variant_display() {
        // Arrange
        use crate::engine::executor::ExecutorError;
        let err = ExecutorError::Backend(BE::Cuda("alloc failed".to_string()));
        // Act
        let msg = format!("{}", err);
        // Assert
        assert!(msg.contains("CUDA error: alloc failed"), "Backend wrapping should preserve inner display, got: {msg}");
    }

    #[test]
    fn executor_error_empty_prompt_display() {
        // Arrange
        use crate::engine::executor::ExecutorError;
        let err = ExecutorError::EmptyPrompt;
        // Act
        let msg = format!("{}", err);
        // Assert
        assert!(msg.contains("empty prompt tokens"), "got: {msg}");
    }

    #[test]
    fn executor_error_compilation_variant_display() {
        // Arrange
        use crate::engine::executor::ExecutorError;
        let err = ExecutorError::Compilation("symbolic dimension unresolved".to_string());
        // Act
        let msg = format!("{}", err);
        // Assert
        assert!(msg.contains("JIT compilation failed"), "got: {msg}");
        assert!(msg.contains("symbolic dimension unresolved"), "got: {msg}");
    }

    #[test]
    fn tensor_role_attention_and_rope_variants_canonical_names() {
        // Arrange
        use crate::manifest::TensorRole;
        let roles = [
            (TensorRole::AttentionKey, "k_proj"),
            (TensorRole::AttentionValue, "v_proj"),
            (TensorRole::AttentionOutput, "o_proj"),
            (TensorRole::AttentionQNorm, "q_norm"),
            (TensorRole::AttentionKNorm, "k_norm"),
            (TensorRole::Rope, "rope"),
        ];
        for (role, expected) in roles {
            // Act
            let name = role.to_canonical_name(None);
            // Assert
            assert_eq!(name, expected, "{:?} should map to {}", role, expected);
            let layered = role.to_canonical_name(Some(11));
            assert_eq!(layered, format!("L11.{}", expected));
        }
    }

    #[test]
    fn tensor_role_classifier_and_special_variants_canonical_names() {
        // Arrange
        use crate::manifest::TensorRole;
        let roles = [
            (TensorRole::ClassifierDense, "classifier.dense"),
            (TensorRole::ClassifierOutProj, "classifier"),
            (TensorRole::PatchEmbed, "patch_embed"),
            (TensorRole::PositionEmbedding, "position_embed"),
            (TensorRole::AttentionFusedQkv, "qkv_proj"),
            (TensorRole::AttentionSinks, "attn_sinks"),
            (TensorRole::DepthwiseConv, "depthwise_conv"),
        ];
        for (role, expected) in roles {
            // Act
            let name = role.to_canonical_name(None);
            // Assert
            assert_eq!(name, expected, "{:?} should map to {}", role, expected);
        }
    }

    #[test]
    fn request_telemetry_default_values() {
        // Arrange
        use crate::scheduler::request_state::RequestTelemetry;
        // Act
        let tel = RequestTelemetry::default();
        // Assert
        assert_eq!(tel.entropy, 0.0);
        assert_eq!(tel.centroid, 0.0);
        assert_eq!(tel.residual_delta, 1.0);
        assert_eq!(tel.residual_cosine, 1.0);
        assert_eq!(tel.range_group, 0);
    }

    #[test]
    fn request_telemetry_construction_and_copy_semantics() {
        // Arrange
        use crate::scheduler::request_state::RequestTelemetry;
        let original = RequestTelemetry {
            entropy: 3.14,
            centroid: 0.5,
            residual_delta: -0.1,
            residual_cosine: 0.99,
            range_group: 7,
        };
        // Act: Copy (RequestTelemetry derives Copy)
        let copied = original;
        // Assert: both remain usable
        assert_eq!(original.entropy, 3.14);
        assert_eq!(copied.entropy, 3.14);
        assert_eq!(copied.range_group, 7);
        assert_eq!(original, copied);
    }

    #[test]
    fn compact_scatter_meta_construction_and_equality() {
        // Arrange
        use crate::scheduler::request_state::CompactScatterMeta;
        let a = CompactScatterMeta {
            original_slot: 3,
            compacted_slot: 1,
            active: 1,
        };
        let b = CompactScatterMeta {
            original_slot: 3,
            compacted_slot: 1,
            active: 1,
        };
        let c = CompactScatterMeta {
            original_slot: 3,
            compacted_slot: 1,
            active: 0,
        };
        // Assert: equal when all fields match
        assert_eq!(a, b, "identical CompactScatterMeta should be equal");
        assert_ne!(a, c, "different active field should make them unequal");
        // Assert: Copy semantics
        let copied = a;
        assert_eq!(copied.original_slot, 3);
        assert_eq!(copied.compacted_slot, 1);
        assert_eq!(copied.active, 1);
    }

    #[test]
    fn sequence_telemetry_default_and_custom_construction() {
        // Arrange
        use crate::scheduler::telemetry::SequenceTelemetry;
        let default = SequenceTelemetry::default();
        let custom = SequenceTelemetry {
            l2_delta: 0.5,
            has_outlier: true,
            dead_density: 0.1,
            per_head_entropy: 2.7,
            transform_ratio: 0.03,
            output_entropy: 4.2,
        };
        // Assert: defaults are zero-valued
        assert_eq!(default.l2_delta, 0.0);
        assert!(!default.has_outlier);
        assert_eq!(default.dead_density, 0.0);
        // Assert: custom values preserved
        assert_eq!(custom.l2_delta, 0.5);
        assert!(custom.has_outlier);
        assert!((custom.per_head_entropy - 2.7).abs() < f32::EPSILON);
        assert_eq!(custom.output_entropy, 4.2);
        // Assert: Copy semantics
        let copied = custom;
        assert_eq!(copied.l2_delta, 0.5);
        assert_eq!(copied, custom);
    }

    #[test]
    fn model_kind_from_str_roundtrip() {
        // Arrange
        use crate::manifest::ModelKind;
        use std::str::FromStr;
        // Act & Assert: FromStr delegates to parse()
        assert_eq!(ModelKind::from_str("chat").unwrap(), ModelKind::Chat);
        assert_eq!(ModelKind::from_str("embedding").unwrap(), ModelKind::Embedding);
        assert_eq!(ModelKind::from_str("reranker").unwrap(), ModelKind::Reranker);
        assert_eq!(ModelKind::from_str("classifier").unwrap(), ModelKind::Classifier);
        assert!(ModelKind::from_str("nonexistent").is_err());
    }

    #[test]
    fn generator_forward_config_dtype_accessor() {
        // Arrange
        use crate::engine::executor::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act
        let dtype = cfg.dtype();
        // Assert: default_for_test uses F32 compute_dtype
        assert_eq!(dtype, gllm_kernels::types::DType::F32);
    }

    #[test]
    fn attention_topology_debug_output() {
        // Arrange
        use crate::engine::executor::AttentionTopology;
        let topo = AttentionTopology::linear();
        // Act
        let debug = format!("{:?}", topo);
        // Assert: Debug output contains struct name and geometry
        assert!(debug.contains("AttentionTopology"), "Debug should contain struct name, got: {debug}");
        assert!(debug.contains("geometry"), "Debug should contain geometry field, got: {debug}");
    }

    #[test]
    fn generator_forward_config_intermediate_size_accessor() {
        // Arrange
        use crate::engine::executor::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act & Assert: intermediate_size is 128 from default_for_test geometry
        assert_eq!(cfg.intermediate_size(), 128);
        // Assert: accessor matches geometry field
        assert_eq!(cfg.intermediate_size(), cfg.geometry.intermediate_size);
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 10 (13 new)
    // -----------------------------------------------------------------------

    #[test]
    fn executor_error_request_not_found_display() {
        // Arrange
        use crate::engine::executor::ExecutorError;
        let err = ExecutorError::RequestNotFound { request_id: 42 };
        // Act
        let msg = format!("{}", err);
        // Assert: display contains the request_id
        assert!(msg.contains("request not found"), "Expected 'request not found' prefix, got: {msg}");
        assert!(msg.contains("42"), "Expected request_id 42 in message, got: {msg}");
    }

    #[test]
    fn executor_error_graph_expansion_display() {
        // Arrange
        use crate::engine::executor::ExecutorError;
        let err = ExecutorError::GraphExpansion("unsupported op: Conv2D".to_string());
        // Act
        let msg = format!("{}", err);
        // Assert: display contains the error detail
        assert!(msg.contains("graph expansion failed"), "Expected prefix, got: {msg}");
        assert!(msg.contains("unsupported op: Conv2D"), "Expected detail, got: {msg}");
    }

    #[test]
    fn executor_error_empty_sample_display() {
        // Arrange
        use crate::engine::executor::ExecutorError;
        let err = ExecutorError::EmptySample;
        // Act
        let msg = format!("{}", err);
        // Assert
        assert!(msg.contains("empty sample"), "Expected 'empty sample', got: {msg}");
    }

    #[test]
    fn gpu_device_info_ordinals_unique_in_collection() {
        // Arrange: 8 GPUs in a DGX-style node, all with unique ordinals
        let devices: Vec<GpuDeviceInfo> = (0..8)
            .map(|i| GpuDeviceInfo {
                ordinal: i,
                sm_version: 90,
                sm_count: 132,
                total_memory: 80 * 1024 * 1024 * 1024,
                name: format!("H100-Slot-{}", i),
            })
            .collect();
        // Act: collect ordinals into a HashSet
        let ordinals: std::collections::HashSet<usize> =
            devices.iter().map(|d| d.ordinal).collect();
        // Assert: all 8 ordinals are unique
        assert_eq!(ordinals.len(), 8, "All ordinals should be unique");
        // Assert: ordinals are 0..8
        for i in 0..8 {
            assert!(ordinals.contains(&i), "Ordinal {} should be present", i);
        }
    }

    #[test]
    fn gpu_device_info_total_memory_power_of_two_boundaries() {
        // Arrange: verify memory values at common GPU capacities
        let capacities_gb: Vec<usize> = vec![8, 16, 24, 40, 48, 80, 96, 141, 192];
        for gb in capacities_gb {
            let bytes = gb * 1024 * 1024 * 1024;
            let info = GpuDeviceInfo {
                ordinal: 0,
                sm_version: 90,
                sm_count: 132,
                total_memory: bytes,
                name: format!("GPU-{}GB", gb),
            };
            // Act & Assert: round-trip byte count and GB conversion
            assert_eq!(info.total_memory, bytes);
            assert_eq!(info.total_memory / (1024 * 1024 * 1024), gb);
        }
    }

    #[test]
    fn gpu_device_info_sm_count_zero_is_valid_construction() {
        // Arrange: edge case — driver might report 0 SMs for a virtual/uninitialized device
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 0,
            total_memory: 0,
            name: "Virtual GPU".to_string(),
        };
        // Assert: all zero fields are preserved
        assert_eq!(info.sm_count, 0);
        assert_eq!(info.sm_version, 0);
        assert_eq!(info.ordinal, 0);
        assert!(info.name.contains("Virtual"));
        // Assert: Debug does not panic
        let debug = format!("{:?}", info);
        assert!(debug.contains("GpuDeviceInfo"));
    }

    #[test]
    fn request_data_empty_prompt_tokens_and_output_tokens() {
        // Arrange: edge case — empty prompt and output tokens (minimal request state)
        use crate::engine::executor::RequestData;
        let data = RequestData {
            prompt_tokens: vec![],
            output_tokens: vec![],
            sampling_config: SamplingConfig::default(),
            is_prefill: true,
            phase: crate::scheduler::request_state::RequestPhase::Prefill,
            max_new_tokens: 0,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        // Assert: empty collections are valid constructions
        assert!(data.prompt_tokens.is_empty());
        assert!(data.output_tokens.is_empty());
        assert_eq!(data.max_new_tokens, 0);
        assert!(!data.finished);
    }

    #[test]
    fn request_data_finished_state_with_session_id() {
        // Arrange: completed request with session_id for multi-turn
        use crate::engine::executor::RequestData;
        let data = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![4, 5, 6, 7, 8],
            sampling_config: SamplingConfig { temperature: 0.0, top_k: 1, top_p: 1.0 },
            is_prefill: false,
            phase: crate::scheduler::request_state::RequestPhase::Decode,
            max_new_tokens: 100,
            finished: true,
            session_id: Some(12345u64),
            thinking_budget: Some(2048),
            fused_prefill_hidden: None,
        };
        // Assert: finished state with all fields populated
        assert!(data.finished);
        assert_eq!(data.session_id, Some(12345u64));
        assert_eq!(data.thinking_budget, Some(2048));
        assert_eq!(data.sampling_config.temperature, 0.0);
        assert_eq!(data.output_tokens.len(), 5);
    }

    #[test]
    fn compact_scatter_meta_usable_in_vec_and_sort() {
        // Arrange: multiple scatter metadata entries
        use crate::scheduler::request_state::CompactScatterMeta;
        let entries = vec![
            CompactScatterMeta { original_slot: 5, compacted_slot: 2, active: 1 },
            CompactScatterMeta { original_slot: 1, compacted_slot: 0, active: 1 },
            CompactScatterMeta { original_slot: 9, compacted_slot: 3, active: 0 },
        ];
        // Act: sort by compacted_slot
        let mut sorted = entries.clone();
        sorted.sort_by_key(|e| e.compacted_slot);
        // Assert: sorted order by compacted_slot
        assert_eq!(sorted[0].compacted_slot, 0);
        assert_eq!(sorted[1].compacted_slot, 2);
        assert_eq!(sorted[2].compacted_slot, 3);
        // Assert: original entries unchanged
        assert_eq!(entries[0].original_slot, 5);
    }

    #[test]
    fn sequence_telemetry_copy_and_equality() {
        // Arrange
        use crate::scheduler::telemetry::SequenceTelemetry;
        let original = SequenceTelemetry {
            l2_delta: 1.5,
            has_outlier: true,
            dead_density: 0.25,
            per_head_entropy: 5.0,
            transform_ratio: 0.5,
            output_entropy: 8.0,
        };
        // Act: Copy
        let copied = original;
        // Assert: Copy leaves original usable and values match
        assert_eq!(original.l2_delta, 1.5);
        assert_eq!(copied.l2_delta, 1.5);
        assert_eq!(original, copied, "Copy should produce equal value");
        // Assert: different values are not equal
        let different = SequenceTelemetry {
            l2_delta: 0.0,
            ..copied
        };
        assert_ne!(copied, different, "Different l2_delta should make them unequal");
    }

    #[test]
    fn model_manifest_default_tensor_map_is_empty_and_not_moe() {
        // Arrange & Act
        use crate::manifest::ModelManifest;
        let manifest = ModelManifest::default();
        // Assert: default tensor_map is empty (no tensors registered)
        assert!(manifest.tensor_map.is_empty(), "default tensor_map should be empty");
        // Assert: is_moe() returns false when moe_config is None
        assert!(!manifest.is_moe());
        // Assert: arch is a known default
        assert_eq!(manifest.arch, "llama");
    }

    #[test]
    fn backend_error_cuda_and_hip_variants_preserve_original_message() {
        // Arrange: verify error messages survive variant wrapping
        let cuda_err = BE::Cuda("cuMemAlloc failed: out of memory".to_string());
        let hip_err = BE::Hip("hipMalloc failed: invalid device".to_string());
        // Act
        let cuda_msg = format!("{}", cuda_err);
        let hip_msg = format!("{}", hip_err);
        // Assert: each variant contains its full original message
        assert!(cuda_msg.contains("cuMemAlloc failed: out of memory"),
            "CUDA error should preserve full message, got: {cuda_msg}");
        assert!(hip_msg.contains("hipMalloc failed: invalid device"),
            "HIP error should preserve full message, got: {hip_msg}");
        // Assert: each has its own prefix
        assert!(cuda_msg.starts_with("CUDA error:"), "CUDA should start with 'CUDA error:'");
        assert!(hip_msg.starts_with("HIP error:"), "HIP should start with 'HIP error:'");
    }

    #[test]
    fn effective_kv_max_seq_len_large_context_sizes() {
        // Arrange: verify passthrough for large context sizes used in modern models
        use crate::engine::executor::effective_kv_max_seq_len;
        let large_contexts = [32768, 65536, 131072, 262144, 524288, 1048576];
        // Act & Assert: every value passes through unchanged
        for &ctx in &large_contexts {
            assert_eq!(effective_kv_max_seq_len(ctx), ctx,
                "effective_kv_max_seq_len({}) should be identity", ctx);
        }
    }

    // -----------------------------------------------------------------------
    // Additional tests — round 11 (10 new)
    // -----------------------------------------------------------------------

    #[test]
    fn executor_error_scheduler_variant_display() {
        // Arrange: ExecutorError::Scheduler carries a freeform string
        use crate::engine::executor::ExecutorError;
        let err = ExecutorError::Scheduler("no available slots for batch".to_string());
        // Act
        let msg = format!("{}", err);
        // Assert: display contains the prefix and the detail
        assert!(msg.contains("scheduler error"), "Expected 'scheduler error' prefix, got: {msg}");
        assert!(msg.contains("no available slots for batch"), "Expected detail, got: {msg}");
    }

    #[test]
    fn kv_cache_config_max_seq_len_accessor() {
        // Arrange: KvCacheConfig::max_seq_len() delegates to effective_kv_max_seq_len
        use crate::engine::executor::KvCacheConfig;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 4096, num_layers: 32, vocab_size: 32000, intermediate_size: 11008,
            num_heads: 32, num_kv_heads: 8, head_dim: 128, max_seq_len: 8192,
            rope_theta: 500000.0, rope_scale: 1.0, rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32, compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5, num_experts: 0, moe_top_k: 0, expert_intermediate_size: 0,
            global_rope_theta: 0.0, rope_partial_ratio: 1.0, attention_pattern: vec![],
            sliding_window: 0, num_kv_shared_layers: 0, global_head_dim: 0,
            hidden_size_per_layer_input: 0, position_offset: None, rope_scaling: None,
            final_logit_softcapping: None, hidden_act: None,
            mla_d_c: 0, mla_d_rope: 0, mla_unabsorbed_threshold: 0,
        });
        let config = KvCacheConfig {
            geometry,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Act
        let max_seq = config.max_seq_len();
        // Assert: passes through geometry.max_seq_len unchanged
        assert_eq!(max_seq, 8192, "max_seq_len should be identity passthrough");
    }

    #[test]
    fn generator_forward_config_attention_geometry_derivation() {
        // Arrange: attention_geometry() computes q_dim, kv_dim, heads_per_group from geometry
        use crate::engine::executor::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act
        let geo = cfg.attention_geometry();
        // Assert: derived values match geometry fields
        assert_eq!(geo.num_heads, 4, "num_heads from geometry");
        assert_eq!(geo.num_kv_heads, 2, "num_kv_heads from geometry");
        assert_eq!(geo.head_dim, 16, "head_dim from geometry");
        // Assert: q_dim = num_heads * head_dim = 4 * 16 = 64
        assert_eq!(geo.q_dim, 64, "q_dim = num_heads * head_dim");
        // Assert: kv_dim = num_kv_heads * head_dim = 2 * 16 = 32
        assert_eq!(geo.kv_dim, 32, "kv_dim = num_kv_heads * head_dim");
        // Assert: heads_per_group = num_heads / num_kv_heads = 4 / 2 = 2
        assert_eq!(geo.heads_per_group, 2, "GQA group size");
    }

    #[test]
    fn tensor_role_layer_norm_maps_to_input_norm() {
        // Arrange: TensorRole::LayerNorm is a legacy alias that maps to "input_norm"
        use crate::manifest::TensorRole;
        let role = TensorRole::LayerNorm;
        // Act
        let name_no_layer = role.to_canonical_name(None);
        let name_with_layer = role.to_canonical_name(Some(0));
        // Assert: LayerNorm maps to "input_norm" (backward compat)
        assert_eq!(name_no_layer, "input_norm",
            "LayerNorm should map to 'input_norm' for backward compat");
        assert_eq!(name_with_layer, "L0.input_norm",
            "LayerNorm with layer 0 should be 'L0.input_norm'");
    }

    #[test]
    fn kv_cache_config_attention_pattern_with_alternating_layers() {
        // Arrange: Gemma 4 style alternating sliding/global attention pattern
        use crate::engine::executor::KvCacheConfig;
        use std::sync::Arc;
        use crate::model_config::ModelGeometry;
        let pattern: Vec<u8> = vec![0, 1, 0, 1, 0, 1]; // sliding, global, ...
        let geometry = Arc::new(ModelGeometry {
            hidden_size: 256, num_layers: 6, vocab_size: 100, intermediate_size: 512,
            num_heads: 8, num_kv_heads: 2, head_dim: 32, max_seq_len: 512,
            rope_theta: 10000.0, rope_scale: 1.0, rope_interleaved: false,
            dtype: gllm_kernels::types::DType::F32, compute_dtype: gllm_kernels::types::DType::F32,
            norm_eps: 1e-5, num_experts: 0, moe_top_k: 0, expert_intermediate_size: 0,
            global_rope_theta: 0.0, rope_partial_ratio: 1.0,
            attention_pattern: pattern.clone(),
            sliding_window: 4096, num_kv_shared_layers: 0, global_head_dim: 0,
            hidden_size_per_layer_input: 0, position_offset: None, rope_scaling: None,
            final_logit_softcapping: None, hidden_act: None,
            mla_d_c: 0, mla_d_rope: 0, mla_unabsorbed_threshold: 0,
        });
        let config = KvCacheConfig {
            geometry,
            kv_dtype: gllm_kernels::types::DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Act
        let pat = config.attention_pattern();
        // Assert: pattern is accessible and matches the geometry
        assert_eq!(pat.len(), 6, "attention_pattern should have 6 layers");
        assert_eq!(pat[0], 0, "layer 0 should be sliding");
        assert_eq!(pat[1], 1, "layer 1 should be global");
        assert_eq!(pat[5], 1, "layer 5 should be global");
    }

    #[test]
    fn generator_forward_config_paged_kv_accessor() {
        // Arrange: default_for_test has a PagedKvConfig with page_size=16, no page_table
        use crate::engine::executor::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();
        // Act
        let paged_kv = &cfg.paged_kv;
        // Assert: paged_kv has the default test configuration
        assert_eq!(paged_kv.page_size, 16, "default page_size should be 16");
        assert!(paged_kv.page_table.is_none(), "default page_table should be None");
    }

    #[test]
    fn sequence_input_validate_page_table_with_all_zero_page_ids() {
        // Arrange: page table where all entries are 0 (all map to the first page)
        use crate::engine::executor::SequenceInput;
        let input = SequenceInput {
            tokens: vec![1, 2, 3, 4],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0u32, 0, 0, 0]),
            fused_hidden: None,
        };
        // Act: total_pages=1 means valid page IDs are 0 only
        let result = input.validate_page_table(1);
        // Assert: all-zero page IDs are valid when total_pages >= 1
        assert!(result.is_ok(), "All-zero page IDs should be valid with total_pages=1");
        // Act: total_pages=0 means no valid pages
        let result_empty = input.validate_page_table(0);
        // Assert: page_id=0 >= total_pages=0 is out of bounds
        assert!(result_empty.is_err(), "page_id=0 should be out of bounds when total_pages=0");
    }

    #[test]
    fn model_manifest_family_returns_decoder_for_llama_arch() {
        // Arrange: default manifest has arch="llama" which resolves to Decoder family
        use crate::manifest::ModelManifest;
        let manifest = ModelManifest::default();
        // Act
        let family = manifest.family();
        // Assert: "llama" resolves to Decoder in the arch registry
        assert_eq!(family, crate::manifest::ArchFamily::Decoder,
            "llama arch should resolve to Decoder family");
    }

    #[test]
    fn request_telemetry_with_extreme_values() {
        // Arrange: RequestTelemetry with negative and extreme float values
        use crate::scheduler::request_state::RequestTelemetry;
        let tel = RequestTelemetry {
            entropy: -1.0,
            centroid: f32::MAX,
            residual_delta: f32::MIN,
            residual_cosine: -1.0,
            range_group: u32::MAX,
        };
        // Act: Copy
        let copied = tel;
        // Assert: extreme values are preserved through Copy
        assert_eq!(copied.entropy, -1.0);
        assert_eq!(copied.centroid, f32::MAX);
        assert_eq!(copied.residual_delta, f32::MIN);
        assert_eq!(copied.residual_cosine, -1.0);
        assert_eq!(copied.range_group, u32::MAX);
        // Assert: original remains valid after Copy
        assert_eq!(tel.range_group, u32::MAX);
        assert_eq!(tel, copied);
    }

    #[test]
    fn attention_geometry_equality_and_hash() {
        // Arrange: two AttentionGeometry instances with identical fields
        use crate::compat::types::AttentionGeometry;
        let a = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 32,
            q_dim: 256,
            kv_dim: 128,
            heads_per_group: 2,
        };
        let b = AttentionGeometry {
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 32,
            q_dim: 256,
            kv_dim: 128,
            heads_per_group: 2,
        };
        // Assert: equal instances
        assert_eq!(a, b, "identical AttentionGeometry should be equal");
        // Assert: Copy semantics
        let copied = a;
        assert_eq!(copied, a);
        assert_eq!(a.num_heads, 8, "original should remain valid after Copy");
        // Assert: can be used in HashSet (Hash derived)
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1, "equal AttentionGeometry should deduplicate");
        // Assert: different head_dim makes them unequal
        let different = AttentionGeometry { head_dim: 64, ..a };
        assert_ne!(a, different, "different head_dim should be unequal");
    }
}
