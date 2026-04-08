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
    compiled_hsaco: std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>,
    #[cfg(feature = "hip")]
    pub(super) swap_store: super::gpu_compile::GpuSwapStore,
    #[cfg(feature = "hip")]
    pub(super) kv_meta: super::gpu_compile::GpuKvMetaStore,
    #[cfg(feature = "hip")]
    pub(super) paged_kv_meta: super::gpu_compile::GpuPagedKvMetaStore,
    #[cfg(feature = "hip")]
    pub(super) weight_cache: std::sync::Mutex<super::gpu_compile::HipWeightCache>,
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
            compiled_hsaco: std::sync::Mutex::new(std::collections::HashMap::new()),
            #[cfg(feature = "hip")]
            swap_store: self.swap_store.clone(),
            #[cfg(feature = "hip")]
            kv_meta: self.kv_meta.clone(),
            #[cfg(feature = "hip")]
            paged_kv_meta: self.paged_kv_meta.clone(),
            #[cfg(feature = "hip")]
            weight_cache: std::sync::Mutex::new(super::gpu_compile::HipWeightCache::new()),

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
            compiled_hsaco: std::sync::Mutex::new(std::collections::HashMap::new()),
            swap_store: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            kv_meta: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            paged_kv_meta: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            weight_cache: std::sync::Mutex::new(super::gpu_compile::HipWeightCache::new()),

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
}

impl_gpu_backend! {
    backend = HipBackend,
    cfg_pred = [feature = "hip"],
    feature_label = "hip",
    upload_err = Other,
    decoder_forward = hip_decoder_forward,
    bert_forward = hip_bert_encoder_forward,
}
