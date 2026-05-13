use super::backend_trait::{self, Backend};
use super::cuda_backend::GpuDeviceInfo;
use super::Element;
use crate::engine::executor::{
    AttentionTopology, BackendError as BE, BatchInput, GeneratorForwardConfig, KvCacheConfig,
    KvCacheHandle, LogitsHandle, SamplingConfig,
};
use crate::scheduler::types::{PageId, PageState, StorageKey};

// ---------------------------------------------------------------------------
// HipBackend<E> — AMD ROCm backend with persistent device handle
// ---------------------------------------------------------------------------

pub struct HipBackend<E: Element = f32> {
    device_info: GpuDeviceInfo,
    cpu_profile: gllm_kernels::dispatch::DeviceProfile,
    #[cfg(feature = "hip")]
    pub(super) device: std::sync::Arc<gllm_kernels::gpu::hip::HipDevice>,
    #[cfg(feature = "hip")]
    pub(super) gpu_profile: gllm_kernels::gpu::GpuDeviceProfile,
    #[cfg(feature = "hip")]
    pub(super) compiled_ptx: std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>,
    #[cfg(feature = "hip")]
    pub(super) swap_store: super::gpu_compile::GpuSwapStore,
    #[cfg(feature = "hip")]
    pub(super) kv_meta: super::gpu_compile::GpuKvMetaStore,
    #[cfg(feature = "hip")]
    pub(super) paged_kv_meta: super::gpu_compile::GpuPagedKvMetaStore,
    #[cfg(feature = "hip")]
    pub(super) weight_blob_gpu: std::sync::Mutex<Option<(u64, usize)>>,
    _marker: std::marker::PhantomData<E>,
}

impl<E: Element> std::fmt::Debug for HipBackend<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HipBackend")
            .field("device_info", &self.device_info)
            .finish()
    }
}

impl<E: Element> Clone for HipBackend<E> {
    fn clone(&self) -> Self {
        Self {
            device_info: self.device_info.clone(),
            cpu_profile: self.cpu_profile.clone(),
            #[cfg(feature = "hip")]
            device: self.device.clone(),
            #[cfg(feature = "hip")]
            gpu_profile: self.gpu_profile.clone(),
            #[cfg(feature = "hip")]
            compiled_ptx: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "hip")]
            swap_store: self.swap_store.clone(),
            #[cfg(feature = "hip")]
            kv_meta: self.kv_meta.clone(),
            #[cfg(feature = "hip")]
            paged_kv_meta: self.paged_kv_meta.clone(),
            #[cfg(feature = "hip")]
            weight_blob_gpu: std::sync::Mutex::new(None),

            _marker: std::marker::PhantomData,
        }
    }
}

impl<E: Element> HipBackend<E> {
    #[cfg(feature = "hip")]
    pub fn new(device: usize) -> Option<Self> {
        use gllm_kernels::gpu::GpuDevice;

        let driver = match gllm_kernels::gpu::hip::HipDriver::load() {
            Ok(d) => std::sync::Arc::new(d),
            Err(e) => {
                log::debug!("HIP driver not available: {e}");
                return None;
            }
        };
        if let Err(e) = driver.init() {
            log::debug!("HIP driver init failed: {e}");
            return None;
        }
        let hip_device = gllm_kernels::gpu::hip::HipDevice::new(
            std::sync::Arc::clone(&driver),
            device as i32,
        ).ok()?;

        let gpu_profile = hip_device.gpu_profile().ok()?;

        let device_info = GpuDeviceInfo {
            ordinal: device,
            sm_version: 0, // AMD uses GCN arch, not SM version
            sm_count: gpu_profile.compute_units,
            total_memory: hip_device.total_memory(),
            name: hip_device.name().to_owned(),
        };

        eprintln!("[HipBackend] Detected: {}", device_info.name);

        let cpu_profile = gllm_kernels::dispatch::DeviceProfile::detect();

        Some(Self {
            device_info,
            cpu_profile,
            device: std::sync::Arc::new(hip_device),
            gpu_profile,
            compiled_ptx: std::sync::Mutex::new(std::collections::HashMap::new()),
            swap_store: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            kv_meta: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            paged_kv_meta: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            weight_blob_gpu: std::sync::Mutex::new(None),

            _marker: std::marker::PhantomData,
        })
    }

    #[cfg(not(feature = "hip"))]
    pub fn new(_device: usize) -> Option<Self> {
        None
    }

    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }

    #[cfg(feature = "hip")]
    pub fn device(&self) -> &gllm_kernels::gpu::hip::HipDevice {
        &self.device
    }

    #[cfg(feature = "hip")]
    pub fn gpu_profile(&self) -> &gllm_kernels::gpu::GpuDeviceProfile {
        &self.gpu_profile
    }

    // ------------------------------------------------------------------
    // GPU Mega-Kernel helper methods (used by gpu_backend_macro)
    // ------------------------------------------------------------------

    #[cfg(feature = "hip")]
    pub(super) fn upload_weight_blob(&self, weight_blob: &[u8]) -> Result<u64, String> {
        use gllm_kernels::gpu::{GpuBuffer, GpuDevice};
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

    #[cfg(feature = "hip")]
    pub(super) fn alloc_scratchpad_gpu(&self, bytes: usize) -> Result<u64, String> {
        use gllm_kernels::gpu::{GpuBuffer, GpuDevice};
        let buf = self.device.alloc(bytes)
            .map_err(|e| format!("scratchpad alloc failed: {e}"))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }

    #[cfg(feature = "hip")]
    pub(super) fn upload_to_gpu<T: Copy>(&self, data: &[T]) -> Result<u64, String> {
        use gllm_kernels::gpu::{GpuBuffer, GpuDevice};
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

    #[cfg(feature = "hip")]
    pub(super) fn download_from_gpu(&self, src_ptr: u64, bytes: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; bytes];
        self.device.dtoh_raw(src_ptr, &mut buf)
            .map_err(|e| format!("download dtoh failed: {e}"))?;
        Ok(buf)
    }

    #[cfg(feature = "hip")]
    pub(super) fn get_cached_ptx(&self, key: &str) -> Option<Vec<u8>> {
        self.compiled_ptx.lock().ok()?.get(key).cloned()
    }

    #[cfg(feature = "hip")]
    pub(super) fn get_weight_gpu_ptr(&self) -> Option<u64> {
        self.weight_blob_gpu.lock().ok()?.map(|(ptr, _)| ptr)
    }

    #[cfg(feature = "hip")]
    pub(super) fn get_cached_scratchpad_bytes(&self) -> usize {
        self.compiled_ptx.lock()
            .ok()
            .and_then(|cache| cache.get("__scratchpad_bytes__")
                .and_then(|v| usize::from_le_bytes(v[..8].try_into().ok()?)))
            .unwrap_or(0)
    }

    #[cfg(feature = "hip")]
    pub(super) fn gpu_launch_mega_kernel(
        &self,
        gpu_code: &[u8],
        kernel_name: &str,
        args: &[usize; 22],
    ) -> Result<(), String> {
        use gllm_kernels::gpu::GpuDevice;
        let module = self.device.load_hsaco(gpu_code)
            .map_err(|e| format!("load_hsaco failed: {e}"))?;
        let func = module.get_function(kernel_name)
            .map_err(|e| format!("get_function({kernel_name}) failed: {e}"))?;

        let wave_size = self.gpu_profile.warp_size;
        let block = (wave_size, 1u32, 1u32);
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
    backend = HipBackend,
    cfg_pred = [feature = "hip"],
    feature_label = "hip",
    upload_err = Other,
}
