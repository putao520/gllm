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

impl<E: Element> Backend<E> for HipBackend<E> {
    type Tensor = Vec<E>;

    fn alloc_kv_cache(&self, config: &KvCacheConfig) -> Result<KvCacheHandle, BE> {
        #[cfg(feature = "hip")]
        {
            super::gpu_helpers::gpu_alloc_kv_cache(
                unsafe { &*(self as *const HipBackend<E> as *const HipBackend<f32>) },
                config,
            )
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = config;
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }

    fn batch_forward_gpu_pure(
        &self,
        input: &BatchInput,
        topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        kv_caches: &mut [KvCacheHandle],
        config: &GeneratorForwardConfig,
    ) -> Result<(Vec<LogitsHandle>, f32, Vec<crate::scheduler::SequenceTelemetry>), BE> {
        #[cfg(feature = "hip")]
        {
            super::gpu_compile::hip_decoder_forward(self, input, topology, weights, kv_caches, config)
                .map(|(logits, telemetry)| {
                    (logits, 0.0, telemetry)
                })
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = (input, topology, weights, kv_caches, config);
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }

    fn sample_from_tensor(
        &self,
        logits: &LogitsHandle,
        topology: &AttentionTopology,
        vocab_size: usize,
        sampling: &SamplingConfig,
    ) -> Result<Vec<u32>, BE> {
        #[cfg(feature = "hip")]
        {
            super::gpu_helpers::gpu_sample_from_tensor(logits, topology, vocab_size, sampling)
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = (logits, topology, vocab_size, sampling);
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }

    fn embedding_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        #[cfg(feature = "hip")]
        {
            super::gpu_compile::hip_bert_encoder_forward(self, tokens, weights, config)
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = (tokens, weights, config);
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }

    fn rerank_forward_gpu_pure(
        &self,
        tokens: &[u32],
        _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        #[cfg(feature = "hip")]
        {
            super::gpu_compile::hip_bert_encoder_forward(self, tokens, weights, config)
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = (tokens, weights, config);
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }

    fn get_memory_pressure(&self) -> Result<f32, BE> {
        #[cfg(feature = "hip")]
        {
            use gllm_kernels::gpu::GpuDevice;
            let total = self.device.total_memory();
            if total == 0 { return Ok(0.0); }
            let free = self.device.free_memory();
            Ok(1.0 - (free as f32 / total as f32))
        }
        #[cfg(not(feature = "hip"))]
        { Ok(0.0) }
    }

    fn swap_out_pages(&self, handle: &mut KvCacheHandle, mappings: &[(PageId, StorageKey)]) -> Result<(), BE> {
        #[cfg(feature = "hip")]
        {
            super::gpu_helpers::gpu_swap_out_pages(
                unsafe { &*(self as *const HipBackend<E> as *const HipBackend<f32>) },
                handle,
                mappings,
            )
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = (handle, mappings);
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }

    fn swap_in_pages(&self, handle: &mut KvCacheHandle, mappings: &[(PageId, StorageKey)]) -> Result<(), BE> {
        #[cfg(feature = "hip")]
        {
            super::gpu_helpers::gpu_swap_in_pages(
                unsafe { &*(self as *const HipBackend<E> as *const HipBackend<f32>) },
                handle,
                mappings,
            )
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = (handle, mappings);
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }

    fn get_page_states(&self, handle: &KvCacheHandle) -> Result<Vec<(PageId, PageState)>, BE> {
        #[cfg(feature = "hip")]
        {
            super::gpu_helpers::gpu_get_page_states(
                unsafe { &*(self as *const HipBackend<E> as *const HipBackend<f32>) },
                handle,
            )
        }
        #[cfg(not(feature = "hip"))]
        {
            let _ = handle;
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }



    fn upload_weights(&self, _data: &[E]) -> Result<Self::Tensor, BE> {
        #[cfg(feature = "hip")]
        {
            use gllm_kernels::gpu::GpuDevice;
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<E>(),
                )
            };
            let mut buf = self.device.alloc(bytes.len()).map_err(|e| {
                BE::Other(format!("GPU alloc failed: {e}"))
            })?;
            let stream = self.device.default_stream();
            self.device.htod(bytes, &mut buf, stream).map_err(|e| {
                BE::Other(format!("GPU htod failed: {e}"))
            })?;
            Ok(data.to_vec())
        }
        #[cfg(not(feature = "hip"))]
        {
            Err(BE::Unimplemented("hip feature not enabled"))
        }
    }
}
