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
    compiled_msl: std::sync::Mutex<std::collections::HashMap<String, Vec<u8>>>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) swap_store: super::gpu_compile::GpuSwapStore,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) kv_meta: super::gpu_compile::GpuKvMetaStore,
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
            compiled_msl: std::sync::Mutex::new(
                self.compiled_msl.lock().unwrap_or_else(|e| e.into_inner()).clone(),
            ),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            swap_store: self.swap_store.clone(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            kv_meta: self.kv_meta.clone(),
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
            compiled_msl: std::sync::Mutex::new(std::collections::HashMap::new()),
            swap_store: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            kv_meta: std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
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
}

impl<E: Element> Backend<E> for MetalBackend<E> {
    type Tensor = Vec<E>;

    fn alloc_kv_cache(&self, config: &KvCacheConfig) -> Result<KvCacheHandle, BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            super::gpu_helpers::gpu_alloc_kv_cache(
                unsafe { &*(self as *const MetalBackend<E> as *const MetalBackend<f32>) },
                config,
            )
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = config;
            Err(BE::Unimplemented("metal feature not enabled"))
        }
    }

    fn batch_forward_gpu_pure(
        &self, input: &BatchInput, topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        kv_caches: &mut [KvCacheHandle], config: &GeneratorForwardConfig,
    ) -> Result<(Vec<LogitsHandle>, f32), BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            super::gpu_compile::metal_decoder_forward(self, input, topology, weights, kv_caches, config)
                .map(|logits| (logits, 0.0))
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (input, topology, weights, kv_caches, config);
            Err(BE::Unimplemented("metal feature not enabled"))
        }
    }

    fn sample_from_tensor(
        &self, logits: &LogitsHandle, topology: &AttentionTopology,
        vocab_size: usize, sampling: &SamplingConfig,
    ) -> Result<Vec<u32>, BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            super::gpu_helpers::gpu_sample_from_tensor(logits, topology, vocab_size, sampling)
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (logits, topology, vocab_size, sampling);
            Err(BE::Unimplemented("metal feature not enabled"))
        }
    }

    fn embedding_forward_gpu_pure(
        &self, tokens: &[u32], _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            super::gpu_compile::metal_bert_encoder_forward(self, tokens, weights, config)
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (tokens, weights, config);
            Err(BE::Unimplemented("metal feature not enabled"))
        }
    }

    fn rerank_forward_gpu_pure(
        &self, tokens: &[u32], _topology: &AttentionTopology,
        weights: &dyn backend_trait::TensorLookup<E, Self>,
        config: &GeneratorForwardConfig,
    ) -> Result<Vec<f32>, BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            super::gpu_compile::metal_bert_encoder_forward(self, tokens, weights, config)
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (tokens, weights, config);
            Err(BE::Unimplemented("metal feature not enabled"))
        }
    }

    fn get_memory_pressure(&self) -> Result<f32, BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            use gllm_kernels::gpu::GpuDevice;
            let total = self.device.total_memory();
            let free = self.device.free_memory();
            if total == 0 { return Ok(0.0); }
            Ok(1.0 - (free as f32 / total as f32))
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        { Ok(0.0) }
    }

    fn swap_out_pages(&self, handle: &mut KvCacheHandle, mappings: &[(PageId, StorageKey)]) -> Result<(), BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            super::gpu_helpers::gpu_swap_out_pages(
                unsafe { &*(self as *const MetalBackend<E> as *const MetalBackend<f32>) },
                handle,
                mappings,
            )
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (handle, mappings);
            Err(BE::Unimplemented("metal feature not enabled"))
        }
    }

    fn swap_in_pages(&self, handle: &mut KvCacheHandle, mappings: &[(PageId, StorageKey)]) -> Result<(), BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            super::gpu_helpers::gpu_swap_in_pages(
                unsafe { &*(self as *const MetalBackend<E> as *const MetalBackend<f32>) },
                handle,
                mappings,
            )
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (handle, mappings);
            Err(BE::Unimplemented("metal feature not enabled"))
        }
    }

    fn get_page_states(&self, handle: &KvCacheHandle) -> Result<Vec<(PageId, PageState)>, BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            super::gpu_helpers::gpu_get_page_states(
                unsafe { &*(self as *const MetalBackend<E> as *const MetalBackend<f32>) },
                handle,
            )
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = handle;
            Err(BE::Unimplemented("metal feature not enabled"))
        }
    }

    fn upload_weights(&self, data: &[E]) -> Result<Self::Tensor, BE> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            use gllm_kernels::gpu::GpuDevice;
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<E>(),
                )
            };
            let mut buf = self.device.alloc(bytes.len()).map_err(|e| {
                BE::Metal(format!("GPU alloc failed: {e}"))
            })?;
            let stream = self.device.default_stream();
            self.device.htod(bytes, &mut buf, stream).map_err(|e| {
                BE::Metal(format!("GPU htod failed: {e}"))
            })?;
            Ok(data.to_vec())
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            Ok(data.to_vec())
        }
    }
}
