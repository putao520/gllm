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
            // Size changed (model reload?) — re-upload
        }
        let buf = self.device.alloc(weight_blob.len())
            .map_err(|e| format!("weight blob alloc failed ({} bytes): {e}", weight_blob.len()))?;
        let ptr = buf.as_device_ptr();
        self.device.htod(weight_blob, ptr)
            .map_err(|e| format!("weight blob htod failed: {e}"))?;
        std::mem::forget(buf); // leak the buffer — lifetime bound to CudaBackend
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
        let buf = self.device.alloc(bytes)
            .map_err(|e| format!("upload alloc failed ({bytes} bytes): {e}"))?;
        let ptr = buf.as_device_ptr();
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes)
        };
        self.device.htod(byte_slice, ptr)
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

        // Prepare 21 parameter array (cuLaunchKernel takes **void = array of pointers-to-arguments)
        let mut positions: Vec<u32> = (0..(prompt_len + max_new_tokens) as u32).collect();
        let batch_size: usize = 1;
        let temperature_u32 = temperature.to_bits();
        let top_p_u32 = top_p.to_bits();
        let output_mode_selector: usize = 0; // Generate

        let args: [*mut std::ffi::c_void; 21] = [
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
        ];

        let stream = self.device.default_stream();
        self.device.launch_kernel(func, grid, block, &args, stream)
            .map_err(|e| format!("launch_kernel failed: {e}"))?;

        Ok(max_new_tokens)
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
            .and_then(|cache| cache.get("__scratchpad_bytes__")
                .and_then(|v| usize::from_le_bytes(v[..8].try_into().ok()?)))
            .unwrap_or(0)
    }

    #[cfg(feature = "cuda")]
    pub(super) fn gpu_launch_mega_kernel(
        &self,
        ptx_code: &[u8],
        kernel_name: &str,
        args: &[usize; 22],
    ) -> Result<(), String> {
        use gllm_kernels::gpu::GpuDevice;
        let module = self.device.load_ptx(ptx_code)
            .map_err(|e| format!("load_ptx failed: {e}"))?;
        let func = module.get_function(kernel_name)
            .map_err(|e| format!("get_function({kernel_name}) failed: {e}"))?;

        let warp_size = self.gpu_profile.warp_size;
        let block = (warp_size, 1u32, 1u32);
        let grid = (1u32, 1u32, 1u32);

        let kernel_args: [*mut std::ffi::c_void; 21] = std::array::from_fn(|i| {
            unsafe { &args[i] as *const usize as *mut std::ffi::c_void }
        });

        let stream = self.device.default_stream();
        self.device.launch_kernel(func, grid, block, &kernel_args, stream)
            .map_err(|e| format!("launch_kernel failed: {e}"))?;
        Ok(())
    }
}

impl_gpu_backend! {
    backend = CudaBackend,
    cfg_pred = [feature = "cuda"],
    feature_label = "cuda",
    upload_err = Other,
}
