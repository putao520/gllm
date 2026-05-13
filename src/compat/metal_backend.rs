use super::backend_trait::{self, Backend};
use super::Element;
use crate::engine::executor::{
    AttentionTopology, BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheConfig,
    KvCacheHandle, LogitsHandle, SamplingConfig,
};
use crate::scheduler::types::{PageId, PageState, StorageKey};

// ---------------------------------------------------------------------------
// MetalBackend<E> — Apple Metal GPU backend
// ---------------------------------------------------------------------------

pub struct MetalBackend<E: Element = f32> {
    device_name: String,
    total_memory: usize,
    cpu_profile: gllm_kernels::dispatch::DeviceProfile,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    device: std::sync::Arc<gllm_kernels::gpu::metal::MetalDevice>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    gpu_profile: gllm_kernels::gpu::GpuDeviceProfile,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) compiled_ptx: std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) swap_store: super::gpu_compile::GpuSwapStore,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) kv_meta: super::gpu_compile::GpuKvMetaStore,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) paged_kv_meta: super::gpu_compile::GpuPagedKvMetaStore,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) weight_blob_gpu: std::sync::Mutex<Option<(u64, usize)>>,

    _marker: std::marker::PhantomData<E>,
}

impl<E: Element> std::fmt::Debug for MetalBackend<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device_name", &self.device_name)
            .field("total_memory", &self.total_memory)
            .finish()
    }
}

impl<E: Element> Clone for MetalBackend<E> {
    fn clone(&self) -> Self {
        Self {
            device_name: self.device_name.clone(),
            total_memory: self.total_memory,
            cpu_profile: self.cpu_profile.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            device: self.device.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            gpu_profile: self.gpu_profile.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            compiled_ptx: std::sync::Mutex::new(
                self.compiled_ptx.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            ),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            swap_store: self.swap_store.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            kv_meta: self.kv_meta.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            paged_kv_meta: self.paged_kv_meta.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            weight_blob_gpu: std::sync::Mutex::new(None),

            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: Element> MetalBackend<E> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new(device: usize) -> Option<Self> {
        if device != 0 { return None; }

        let metal_device = gllm_kernels::gpu::metal::MetalDevice::new().ok()?;
        let device_name = metal_device.name().to_string();

        use gllm_kernels::gpu::GpuDevice;
        let total_memory = metal_device.total_memory();
        let cpu_profile = gllm_kernels::dispatch::DeviceProfile::detect();

        use gllm_kernels::compiler::codegen::emitter::Platform;
        let gpu_profile = gllm_kernels::gpu::GpuDeviceProfile {
            platform: Platform::Metal { gpu_family: 9 },
            compute_units: 10,
            shared_mem_per_block: 32768,
            max_registers_per_thread: 0,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 1024],
            max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
            total_memory,
            memory_bandwidth_gbs: 200.0,
            peak_gflops_f32: 5000.0,
            peak_gflops_f16: 10000.0,
            has_matrix_unit: true,
            clock_mhz: 1000,
            isv: gllm_kernels::gpu::GpuIsvCapabilities::default(),
        };

        eprintln!("[MetalBackend] Detected: {}", device_name);

        Some(Self {
            device_name,
            total_memory,
            cpu_profile,
            device: std::sync::Arc::new(metal_device),
            gpu_profile,
            compiled_ptx: std::sync::Mutex::new(std::collections::HashMap::new()),
            swap_store: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            kv_meta: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            paged_kv_meta: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            weight_blob_gpu: std::sync::Mutex::new(None),

            _marker: std::marker::PhantomData,
        })
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn new(_device: usize) -> Option<Self> {
        None
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn device(&self) -> &gllm_kernels::gpu::metal::MetalDevice {
        &self.device
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn gpu_profile(&self) -> &gllm_kernels::gpu::GpuDeviceProfile {
        &self.gpu_profile
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn device_info(&self) -> super::cuda_backend::GpuDeviceInfo {
        super::cuda_backend::GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0, // Metal uses GPU family, not SM version
            sm_count: self.gpu_profile.compute_units,
            total_memory: self.total_memory,
            name: self.device_name.clone(),
        }
    }

    // ------------------------------------------------------------------
    // GPU Mega-Kernel helper methods (used by gpu_backend_macro)
    // ------------------------------------------------------------------

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn upload_weight_blob(&self, weight_blob: &[u8]) -> Result<u64, String> {
        use gllm_kernels::gpu::GpuBuffer;
        let mut cache = self.weight_blob_gpu.lock()
            .map_err(|e| format!("weight_blob_gpu lock poisoned: {e}"))?;
        if let Some((ptr, bytes)) = *cache {
            if bytes == weight_blob.len() { return Ok(ptr); }
        }
        let buf = self.device.alloc(weight_blob.len())
            .map_err(|e| format!("weight blob alloc failed: {e}"))?;
        let ptr = buf.as_device_ptr();
        self.device.htod_raw(weight_blob, ptr)
            .map_err(|e| format!("weight blob htod failed: {e}"))?;
        std::mem::forget(buf);
        *cache = Some((ptr, weight_blob.len()));
        Ok(ptr)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn alloc_scratchpad_gpu(&self, bytes: usize) -> Result<u64, String> {
        use gllm_kernels::gpu::GpuBuffer;
        let buf = self.device.alloc(bytes)
            .map_err(|e| format!("scratchpad alloc failed: {e}"))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn upload_to_gpu<T: Copy>(&self, data: &[T]) -> Result<u64, String> {
        use gllm_kernels::gpu::GpuBuffer;
        let bytes = std::mem::size_of_val(data);
        let buf = self.device.alloc(bytes)
            .map_err(|e| format!("upload alloc failed ({bytes} bytes): {e}"))?;
        let ptr = buf.as_device_ptr();
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes)
        };
        self.device.htod_raw(byte_slice, ptr)
            .map_err(|e| format!("upload htod failed: {e}"))?;
        std::mem::forget(buf);
        Ok(ptr)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn download_from_gpu(&self, src_ptr: u64, bytes: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; bytes];
        self.device.dtoh_raw(src_ptr, &mut buf)
            .map_err(|e| format!("download dtoh failed: {e}"))?;
        Ok(buf)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn get_cached_ptx(&self, key: &str) -> Option<Vec<u8>> {
        self.compiled_ptx.lock().ok()?.get(key).cloned()
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn get_weight_gpu_ptr(&self) -> Option<u64> {
        self.weight_blob_gpu.lock().ok()?.map(|(ptr, _)| ptr)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn get_cached_scratchpad_bytes(&self) -> usize {
        self.compiled_ptx.lock()
            .ok()
            .and_then(|cache| cache.get("__scratchpad_bytes__")
                .and_then(|v| usize::from_le_bytes(v[..8].try_into().ok()?)))
            .unwrap_or(0)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn gpu_launch_mega_kernel(
        &self,
        gpu_code: &[u8],
        kernel_name: &str,
        args: &[usize; 22],
    ) -> Result<(), String> {
        use gllm_kernels::gpu::GpuDevice;
        let library = self.device.load_library_data(gpu_code)
            .map_err(|e| format!("load_library_data failed: {e}"))?;

        let threadgroup_size = self.gpu_profile.warp_size;
        let buffers: Vec<(u64, usize)> = args.iter()
            .map(|&a| (a as u64, 0))
            .collect();

        let stream = self.device.default_stream();
        self.device.launch_compute(
            library,
            kernel_name,
            &buffers,
            (1, 1, 1),
            (threadgroup_size, 1, 1),
            stream,
        ).map_err(|e| format!("launch_compute failed: {e}"))?;
        Ok(())
    }
}

impl_gpu_backend! {
    backend = MetalBackend,
    cfg_pred = [all(target_os = "macos", feature = "metal")],
    feature_label = "metal",
    upload_err = Metal,
}
