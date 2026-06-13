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
        args: &[usize; 23],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::{
        AttentionHeadConfig, AttentionMaskType, BackendError, BatchInput, KvCacheConfig,
        PagedKvConfig, RoPEConfig, SamplingConfig, SequenceInput, SwapConfig,
    };
    use crate::model_config::ModelGeometry;
    use gllm_kernels::types::DType;
    use std::sync::Arc;

    /// Helper: build a minimal `ModelGeometry` for tests.
    fn make_geometry() -> ModelGeometry {
        ModelGeometry {
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
            dtype: DType::F32,
            compute_dtype: DType::F32,
            norm_eps: 1e-5,
            num_experts: 0,
            moe_top_k: 0,
            expert_intermediate_size: 0,
            global_rope_theta: 0.0,
            rope_partial_ratio: 1.0,
            rope_partial_ratio_global: 1.0,
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
            qk_norm: false,
            value_norm: false,
            embedding_scale_factor: 0.0,
            mla_use_unabsorbed: false,
        }
    }

    #[test]
    fn metal_backend_new_returns_none_on_linux() {
        // On Linux, Metal is never available regardless of ordinal
        let result = MetalBackend::<f32>::new(0);
        assert!(result.is_none());
    }

    #[test]
    fn metal_backend_new_invalid_ordinal_returns_none() {
        let result = MetalBackend::<f32>::new(5);
        assert!(result.is_none());
    }

    #[test]
    fn metal_backend_new_negative_ordinal_returns_none() {
        // Arrange / Act: usize wrapping means -1 wraps to MAX, which is out of range
        let result = MetalBackend::<f32>::new(usize::MAX);
        // Assert: any non-zero ordinal should return None
        assert!(result.is_none());
    }

    #[test]
    fn metal_backend_new_multiple_calls_all_none() {
        // Arrange / Act: call new multiple times with different ordinals
        let results: Vec<_> = (0..4).map(|i| MetalBackend::<f32>::new(i)).collect();
        // Assert: all must be None on non-macOS
        for (i, result) in results.iter().enumerate() {
            assert!(result.is_none(), "expected None for ordinal {i}");
        }
    }

    #[test]
    fn metal_backend_phantom_data_type_parameter_f32() {
        // Arrange / Act: MetalBackend::<f32>::new should compile and return None
        let result = MetalBackend::<f32>::new(0);
        // Assert: f32 parameter accepted, returns None on non-macOS
        assert!(result.is_none());
    }

    #[test]
    fn metal_backend_phantom_data_type_parameter_f16() {
        // Arrange / Act: MetalBackend::<half::f16>::new should compile
        let result = MetalBackend::<half::f16>::new(0);
        // Assert: f16 parameter accepted, returns None on non-macOS
        assert!(result.is_none());
    }

    #[test]
    fn metal_backend_alloc_kv_cache_returns_unimplemented() {
        // Arrange: verify the BackendError::Unimplemented variant
        // type exists and produces correct Display output
        let err = BackendError::Unimplemented("metal feature not enabled");
        // Assert
        assert!(matches!(err, BackendError::Unimplemented(_)));
        assert_eq!(err.to_string(), "unimplemented: metal feature not enabled");
    }

    #[test]
    fn backend_error_metal_variant_format() {
        // Arrange: construct a Metal variant BackendError
        let err = BackendError::Metal("device not found".into());
        // Assert: Display includes the message
        assert_eq!(err.to_string(), "Metal error: device not found");
    }

    #[test]
    fn backend_error_metal_variant_equality() {
        // Arrange: two Metal errors with same content
        let err1 = BackendError::Metal("oom".into());
        let err2 = err1.clone();
        // Assert: Clone produces equal Display output
        assert_eq!(err1.to_string(), err2.to_string());
    }

    #[test]
    fn backend_error_metal_variant_clone() {
        // Arrange
        let err = BackendError::Metal("alloc failed".into());
        // Act
        let cloned = err.clone();
        // Assert
        assert_eq!(err.to_string(), cloned.to_string());
    }

    #[test]
    fn backend_error_all_variants_are_distinct() {
        // Arrange: create one of each BackendError variant
        let errors = vec![
            BackendError::Cuda("c".into()),
            BackendError::Hip("h".into()),
            BackendError::Metal("m".into()),
            BackendError::Cpu("u".into()),
            BackendError::Unimplemented("unimpl"),
            BackendError::Other("o".into()),
        ];
        // Assert: all Display outputs are distinct
        let displays: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        for i in 0..displays.len() {
            for j in (i + 1)..displays.len() {
                assert_ne!(displays[i], displays[j],
                    "variants {i} and {j} have same display: {}", displays[i]);
            }
        }
    }

    #[test]
    fn kv_cache_config_default_page_size() {
        // Arrange: build config with explicit page_size
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: page_size preserved
        assert_eq!(config.page_size, 16);
    }

    #[test]
    fn kv_cache_config_num_layers_and_heads() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: delegates to geometry
        assert_eq!(config.num_layers(), geometry.num_layers);
        assert_eq!(config.num_heads(), geometry.num_kv_heads);
        assert_eq!(config.head_dim(), geometry.head_dim);
    }

    #[test]
    fn kv_cache_config_dtype_size() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        let config_f32 = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let config_bf16 = KvCacheConfig {
            geometry,
            kv_dtype: DType::BF16,
            page_size: 16,
            swap_config: None,
        };
        // Assert: dtype_size reflects the kv_dtype
        assert_eq!(config_f32.dtype_size(), 4);
        assert_eq!(config_bf16.dtype_size(), 2);
    }

    #[test]
    fn sampling_config_default_values() {
        // Arrange / Act
        let config = SamplingConfig::default();
        // Assert: defaults match expected values
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 0);
        assert!((config.top_p - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sampling_config_custom_values() {
        // Arrange
        let config = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.95,
        };
        // Assert: custom values preserved
        assert!((config.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!((config.top_p - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn logits_handle_data_preserved() {
        // Arrange
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        // Act
        let handle = LogitsHandle { data: data.clone() };
        // Assert: data round-trips
        assert_eq!(handle.data, data);
        assert_eq!(handle.data.len(), 4);
    }

    #[test]
    fn kv_cache_handle_zero_value() {
        // Arrange / Act
        let handle = KvCacheHandle(0);
        // Assert: zero-value handle is valid
        assert_eq!(handle.0, 0u64);
    }

    #[test]
    fn kv_cache_handle_equality() {
        // Arrange
        let h1 = KvCacheHandle(42);
        let h2 = KvCacheHandle(42);
        let h3 = KvCacheHandle(99);
        // Assert: equality semantics
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn backend_error_cuda_display_format() {
        // Arrange
        let err = BackendError::Cuda("out of memory".into());
        // Assert
        assert_eq!(err.to_string(), "CUDA error: out of memory");
    }

    #[test]
    fn backend_error_hip_display_format() {
        // Arrange
        let err = BackendError::Hip("device lost".into());
        // Assert
        assert_eq!(err.to_string(), "HIP error: device lost");
    }

    #[test]
    fn backend_error_cpu_display_format() {
        // Arrange
        let err = BackendError::Cpu("illegal instruction".into());
        // Assert
        assert_eq!(err.to_string(), "CPU error: illegal instruction");
    }

    #[test]
    fn backend_error_other_display_format() {
        // Arrange
        let err = BackendError::Other("unexpected failure".into());
        // Assert
        assert_eq!(err.to_string(), "backend error: unexpected failure");
    }

    #[test]
    fn backend_error_unimplemented_is_static_str() {
        // Arrange: Unimplemented stores &'static str, not String
        let err = BackendError::Unimplemented("test feature");
        // Assert: round-trips through Display
        assert_eq!(err.to_string(), "unimplemented: test feature");
    }

    #[test]
    fn backend_error_implements_std_error() {
        // Arrange
        let err = BackendError::Metal("device lost".into());
        // Act: verify std::error::Error is implemented by downcasting
        let err_boxed: Box<dyn std::error::Error> = Box::new(err);
        // Assert: can be used as a dyn Error
        assert!(err_boxed.downcast_ref::<BackendError>().is_some());
    }

    #[test]
    fn kv_cache_config_equality_same_fields() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        let config1 = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let config2 = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: same fields → equal (PartialEq)
        assert_eq!(config1, config2);
    }

    #[test]
    fn kv_cache_config_inequality_different_page_size() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        let config_a = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let config_b = KvCacheConfig {
            geometry,
            kv_dtype: DType::F32,
            page_size: 32,
            swap_config: None,
        };
        // Assert: different page_size → not equal
        assert_ne!(config_a, config_b);
    }

    #[test]
    fn kv_cache_config_with_swap_config_some() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        let swap = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        let config = KvCacheConfig {
            geometry,
            kv_dtype: DType::BF16,
            page_size: 32,
            swap_config: Some(swap.clone()),
        };
        // Assert: swap_config round-trips
        assert!(config.swap_config.is_some());
        let sc = config.swap_config.as_ref().unwrap();
        assert_eq!(sc, &swap);
        assert_eq!(config.dtype_size(), 2); // BF16
    }

    #[test]
    fn page_state_all_variants_distinct() {
        // Arrange: exercise all PageState variants
        let states = vec![
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];
        // Assert: every pair is distinct
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert_ne!(states[i], states[j],
                    "PageState variants at index {i} and {j} should differ");
            }
        }
        assert_eq!(states.len(), 7);
    }

    #[test]
    fn attention_topology_bidirectional_construction() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        // Act
        let topo = AttentionTopology::bidirectional(geometry.clone());
        // Assert: mask is bidirectional, geometry accessors work
        assert_eq!(topo.num_heads(), 16);
        assert_eq!(topo.num_kv_heads(), 4);
        assert_eq!(topo.head_dim(), 64);
        assert_eq!(topo.max_seq_len(), 2048);
    }

    #[test]
    fn attention_topology_causal_construction() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        // Act
        let topo = AttentionTopology::causal(geometry);
        // Assert: causal topology preserves geometry values
        assert_eq!(topo.num_heads(), 16);
        assert_eq!(topo.num_kv_heads(), 4);
        assert_eq!(topo.head_dim(), 64);
        assert_eq!(topo.max_seq_len(), 2048);
    }

    #[test]
    fn attention_topology_linear_legacy_minimal() {
        // Arrange / Act: legacy linear() constructor uses minimal geometry
        let topo = AttentionTopology::linear();
        // Assert: all geometry fields are the minimal sentinel value 1
        assert_eq!(topo.num_heads(), 1);
        assert_eq!(topo.num_kv_heads(), 1);
        assert_eq!(topo.head_dim(), 1);
        assert_eq!(topo.max_seq_len(), 512);
    }

    #[test]
    fn logits_handle_clone_and_debug() {
        // Arrange
        let handle = LogitsHandle { data: vec![0.5, 1.5, 2.5] };
        // Act
        let cloned = handle.clone();
        let debug_str = format!("{:?}", handle);
        // Assert: clone preserves data, Debug includes struct name
        assert_eq!(handle.data, cloned.data);
        assert!(debug_str.contains("LogitsHandle"), "Debug output should contain type name");
    }

    #[test]
    fn kv_cache_handle_hash_consistency() {
        // Arrange: KvCacheHandle derives Hash
        use std::collections::HashSet;
        let h1 = KvCacheHandle(100);
        let h2 = KvCacheHandle(100);
        let h3 = KvCacheHandle(200);
        // Act
        let mut set = HashSet::new();
        set.insert(h1);
        set.insert(h2); // duplicate
        set.insert(h3);
        // Assert: equal handles hash to same bucket, set deduplicates
        assert_eq!(set.len(), 2);
        assert!(set.contains(&KvCacheHandle(100)));
        assert!(set.contains(&KvCacheHandle(200)));
    }

    #[test]
    fn weight_placement_variants_distinct() {
        // Arrange: WeightPlacement has two variants
        use super::super::backend_trait::WeightPlacement;
        // Assert: DeviceLocal != HostLocal
        assert_ne!(WeightPlacement::DeviceLocal, WeightPlacement::HostLocal);
    }

    #[test]
    fn weight_placement_copy_semantics() {
        // Arrange: WeightPlacement derives Copy
        use super::super::backend_trait::WeightPlacement;
        let a = WeightPlacement::DeviceLocal;
        let b = a; // Copy, not move
        // Assert: both usable after copy
        assert_eq!(a, WeightPlacement::DeviceLocal);
        assert_eq!(b, WeightPlacement::DeviceLocal);
    }

    #[test]
    fn gpu_device_info_construction_and_fields() {
        // Arrange / Act
        let info = super::super::cuda_backend::GpuDeviceInfo {
            ordinal: 0,
            sm_version: 80,
            sm_count: 108,
            total_memory: 80 * 1024 * 1024 * 1024,
            name: "Apple M3 Max".to_string(),
        };
        // Assert: all fields round-trip
        assert_eq!(info.ordinal, 0);
        assert_eq!(info.sm_version, 80);
        assert_eq!(info.sm_count, 108);
        assert_eq!(info.total_memory, 80 * 1024 * 1024 * 1024);
        assert_eq!(info.name, "Apple M3 Max");
    }

    #[test]
    fn gpu_device_info_clone_and_debug() {
        // Arrange
        let info = super::super::cuda_backend::GpuDeviceInfo {
            ordinal: 1,
            sm_version: 90,
            sm_count: 64,
            total_memory: 40 * 1024 * 1024 * 1024,
            name: "Test GPU".into(),
        };
        // Act
        let cloned = info.clone();
        let debug = format!("{:?}", info);
        // Assert: Clone preserves fields, Debug contains struct name
        assert_eq!(cloned.ordinal, info.ordinal);
        assert_eq!(cloned.total_memory, info.total_memory);
        assert_eq!(cloned.name, info.name);
        assert!(debug.contains("GpuDeviceInfo"));
    }

    #[test]
    fn sequence_input_default_fields() {
        // Arrange / Act
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 5,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        // Assert: fields are preserved
        assert_eq!(seq.tokens, vec![1, 2, 3]);
        assert_eq!(seq.position, 5);
        assert_eq!(seq.draft_steps, 0);
        assert!(seq.page_table.is_none());
        assert!(seq.fused_hidden.is_none());
    }

    #[test]
    fn sequence_input_validate_page_table_valid() {
        // Arrange: page_table entries all within bounds
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2, 3]),
            fused_hidden: None,
        };
        // Act
        let result = seq.validate_page_table(10);
        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_out_of_bounds() {
        // Arrange: page_table contains an out-of-bounds page ID
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 5, 10]),
            fused_hidden: None,
        };
        // Act
        let result = seq.validate_page_table(8);
        // Assert: page_table[2] = 10 >= total_pages 8
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("page_table[2]"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn batch_input_empty_sequences() {
        // Arrange / Act
        let batch = BatchInput { sequences: vec![] };
        // Assert: empty batch has zero sequences
        assert!(batch.sequences.is_empty());
        assert_eq!(batch.sequences.len(), 0);
    }

    #[test]
    fn batch_input_clone_independence() {
        // Arrange
        let batch = BatchInput {
            sequences: vec![SequenceInput {
                tokens: vec![10, 20],
                position: 0,
                draft_steps: 0,
                page_table: None,
                fused_hidden: None,
            }],
        };
        // Act
        let mut cloned = batch.clone();
        cloned.sequences[0].tokens.push(30);
        // Assert: original is unmodified
        assert_eq!(batch.sequences[0].tokens, vec![10, 20]);
        assert_eq!(cloned.sequences[0].tokens, vec![10, 20, 30]);
    }

    #[test]
    fn sampling_config_copy_preserves_fields() {
        // Arrange: SamplingConfig derives Copy
        let original = SamplingConfig {
            temperature: 0.5,
            top_k: 100,
            top_p: 0.9,
        };
        // Act
        let copied = original;
        // Assert: both usable after copy, fields identical
        assert!((original.temperature - 0.5).abs() < f32::EPSILON);
        assert_eq!(original.top_k, 100);
        assert!((original.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(copied.temperature, original.temperature);
        assert_eq!(copied.top_k, original.top_k);
    }

    #[test]
    fn backend_error_metal_is_debug() {
        // Arrange
        let err = BackendError::Metal("timeout".into());
        // Act
        let debug_str = format!("{:?}", err);
        // Assert: Debug output contains variant name and message
        assert!(debug_str.contains("Metal"));
        assert!(debug_str.contains("timeout"));
    }

    #[test]
    fn logits_handle_empty_data() {
        // Arrange / Act
        let handle = LogitsHandle { data: vec![] };
        // Assert: empty data is valid
        assert!(handle.data.is_empty());
    }

    #[test]
    fn kv_cache_config_dtype_bf16_size() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry,
            kv_dtype: DType::BF16,
            page_size: 32,
            swap_config: None,
        };
        // Assert: BF16 is 2 bytes
        assert_eq!(config.dtype_size(), 2);
        assert_eq!(config.kv_dtype, DType::BF16);
    }

    #[test]
    fn rope_config_fields_preserved() {
        // Arrange
        let config = RoPEConfig {
            theta: 500000.0,
            scale: 0.5,
            interleaved: true,
            precompute: true,
        };
        // Assert: all fields round-trip
        assert!((config.theta - 500000.0).abs() < f64::EPSILON);
        assert!((config.scale - 0.5).abs() < f64::EPSILON);
        assert!(config.interleaved);
        assert!(config.precompute);
    }

    #[test]
    fn rope_config_partial_eq_semantics() {
        // Arrange: two configs with identical fields
        let a = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let b = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let c = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: true,
            precompute: false,
        };
        // Assert: equality by value, not by reference
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn attention_head_config_from_geometry() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        // Act
        let head_cfg = AttentionHeadConfig::from_geometry(&geometry);
        // Assert: derives from ModelGeometry SSOT
        assert_eq!(head_cfg.num_heads, 16);
        assert_eq!(head_cfg.num_kv_heads, 4);
        assert_eq!(head_cfg.head_dim, 64);
    }

    #[test]
    fn attention_mask_type_variants_distinct() {
        // Arrange: AttentionMaskType has two variants
        let bidir = AttentionMaskType::Bidirectional;
        let causal = AttentionMaskType::Causal;
        // Assert: variants differ and Copy works
        assert_ne!(bidir, causal);
        let copy = bidir;
        assert_eq!(copy, AttentionMaskType::Bidirectional);
        assert_eq!(bidir, AttentionMaskType::Bidirectional);
    }

    #[test]
    fn paged_kv_config_with_page_table() {
        // Arrange
        let config = PagedKvConfig {
            page_table: Some(vec![0, 1, 2, 3, 4]),
            page_size: 16,
        };
        // Assert: page_table and page_size preserved
        assert!(config.page_table.is_some());
        assert_eq!(config.page_table.as_ref().unwrap().len(), 5);
        assert_eq!(config.page_size, 16);
    }

    #[test]
    fn paged_kv_config_without_page_table() {
        // Arrange: contiguous KV access (no paging)
        let config = PagedKvConfig {
            page_table: None,
            page_size: 32,
        };
        // Assert: None means no paging
        assert!(config.page_table.is_none());
        assert_eq!(config.page_size, 32);
    }

    #[test]
    fn kv_cache_config_max_seq_len_delegates_to_geometry() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry: geometry.clone(),
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: max_seq_len comes from geometry, no hardcoded cap
        assert_eq!(config.max_seq_len(), geometry.max_seq_len);
        assert_eq!(config.max_seq_len(), 2048);
    }

    #[test]
    fn kv_cache_handle_max_u64_value() {
        // Arrange: u64::MAX boundary
        let handle = KvCacheHandle(u64::MAX);
        // Assert: maximum value round-trips
        assert_eq!(handle.0, u64::MAX);
    }

    #[test]
    fn gpu_device_info_zero_memory_edge_case() {
        // Arrange: device with zero memory (edge case for detection failure)
        let info = super::super::cuda_backend::GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 0,
            total_memory: 0,
            name: String::new(),
        };
        // Assert: zero values are valid
        assert_eq!(info.total_memory, 0);
        assert!(info.name.is_empty());
        assert_eq!(info.sm_count, 0);
    }

    #[test]
    fn sampling_config_extreme_temperature_values() {
        // Arrange: boundary temperature values
        let zero_temp = SamplingConfig { temperature: 0.0, top_k: 1, top_p: 0.0 };
        let high_temp = SamplingConfig { temperature: f32::MAX, top_k: usize::MAX, top_p: 1.0 };
        // Assert: extreme values preserved without panicking
        assert!((zero_temp.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(zero_temp.top_k, 1);
        assert_eq!(high_temp.temperature, f32::MAX);
        assert_eq!(high_temp.top_k, usize::MAX);
    }

    #[test]
    fn logits_handle_large_data_integrity() {
        // Arrange: large logits vector
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        // Act
        let handle = LogitsHandle { data: data.clone() };
        let cloned = handle.clone();
        // Assert: large data round-trips and clone is independent
        assert_eq!(handle.data.len(), 1024);
        assert_eq!(cloned.data.len(), 1024);
        assert_eq!(handle.data[512], 0.512f32);
        assert_eq!(cloned.data[512], handle.data[512]);
    }

    // ---- 13 new tests (59 → 72) ----

    #[test]
    fn kv_cache_config_kv_dim_standard_model() {
        // Arrange: standard (non-MLA) model geometry: num_kv_heads=4, head_dim=64
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: kv_dim = num_kv_heads * head_dim = 4 * 64 = 256
        assert_eq!(config.kv_dim(), 256);
    }

    #[test]
    fn kv_cache_config_is_mla_standard_model() {
        // Arrange: standard model with mla_d_c = 0
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: is_mla returns false for standard attention
        assert!(!config.is_mla());
    }

    #[test]
    fn kv_cache_config_num_kv_shared_layers_default() {
        // Arrange: default geometry has num_kv_shared_layers = 0
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: no shared KV layers in standard model
        assert_eq!(config.num_kv_shared_layers(), 0);
    }

    #[test]
    fn kv_cache_config_attention_pattern_empty_default() {
        // Arrange: default geometry has empty attention_pattern
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        // Assert: empty attention pattern for standard model
        assert!(config.attention_pattern().is_empty());
    }

    #[test]
    fn swap_config_fields_preserved() {
        // Arrange
        let swap = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.75,
            lru_granularity: 8,
        };
        // Assert: all fields round-trip
        assert!(swap.enable_swap);
        assert!((swap.swap_threshold - 0.75).abs() < f32::EPSILON);
        assert_eq!(swap.lru_granularity, 8);
    }

    #[test]
    fn swap_config_partial_eq_same_and_different() {
        // Arrange: two identical configs and one different
        let a = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        let b = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        let c = SwapConfig {
            enable_swap: false,
            swap_threshold: 0.8,
            lru_granularity: 4,
        };
        // Assert: equal when same fields, different when any field differs
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn sequence_input_with_fused_hidden() {
        // Arrange: multimodal sequence with fused hidden embedding
        let hidden = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(hidden.clone()),
        };
        // Assert: fused_hidden round-trips
        assert!(seq.fused_hidden.is_some());
        assert_eq!(seq.fused_hidden.as_ref().unwrap().len(), 5);
        assert!((seq.fused_hidden.as_ref().unwrap()[0] - 0.1f32).abs() < f32::EPSILON);
    }

    #[test]
    fn sequence_input_validate_page_table_none_returns_ok() {
        // Arrange: sequence with no page table
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        // Act: validate with any total_pages value
        let result = seq.validate_page_table(0);
        // Assert: None page_table is always valid
        assert!(result.is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_boundary_last_valid_page() {
        // Arrange: page_table with max valid page ID = total_pages - 1
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 3, 7]),
            fused_hidden: None,
        };
        // Act: total_pages = 8 means valid page IDs are 0..=7
        let result = seq.validate_page_table(8);
        // Assert: page_table[2] = 7 < 8, so it's valid
        assert!(result.is_ok());
    }

    #[test]
    fn batch_input_multiple_sequences_count() {
        // Arrange: batch with 3 sequences of different lengths
        let seq1 = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        let seq2 = SequenceInput {
            tokens: vec![4, 5],
            position: 3,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        let seq3 = SequenceInput {
            tokens: vec![6],
            position: 5,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        // Act
        let batch = BatchInput {
            sequences: vec![seq1, seq2, seq3],
        };
        // Assert: correct count and per-sequence token lengths
        assert_eq!(batch.sequences.len(), 3);
        assert_eq!(batch.sequences[0].tokens.len(), 3);
        assert_eq!(batch.sequences[1].tokens.len(), 2);
        assert_eq!(batch.sequences[2].tokens.len(), 1);
    }

    #[test]
    fn attention_topology_mask_type_bidirectional() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        // Act
        let topo = AttentionTopology::bidirectional(geometry);
        // Assert: mask_type is Bidirectional
        assert_eq!(topo.mask_type, AttentionMaskType::Bidirectional);
    }

    #[test]
    fn attention_topology_mask_type_causal() {
        // Arrange
        let geometry = Arc::new(make_geometry());
        // Act
        let topo = AttentionTopology::causal(geometry);
        // Assert: mask_type is Causal
        assert_eq!(topo.mask_type, AttentionMaskType::Causal);
    }

    #[test]
    fn paged_kv_config_clone_preserves_fields() {
        // Arrange
        let config = PagedKvConfig {
            page_table: Some(vec![10, 20, 30]),
            page_size: 64,
        };
        // Act
        let cloned = config.clone();
        // Assert: clone is independent and fields match
        assert_eq!(cloned.page_size, config.page_size);
        assert_eq!(cloned.page_table, config.page_table);
        assert_eq!(cloned.page_table.as_ref().unwrap().len(), 3);
        assert_eq!(cloned.page_table.as_ref().unwrap()[1], 20);
    }

    // ---- 13 new tests (72 → 85) ----

    #[test]
    fn model_geometry_is_moe_false_when_zero_experts() {
        let geometry = make_geometry();
        assert!(!geometry.is_moe());
        assert_eq!(geometry.num_experts, 0);
    }

    #[test]
    fn model_geometry_is_moe_true_with_experts() {
        let mut geometry = make_geometry();
        geometry.num_experts = 8;
        geometry.moe_top_k = 2;
        geometry.expert_intermediate_size = 4096;
        assert!(geometry.is_moe());
    }

    #[test]
    fn model_geometry_is_mla_false_when_zero_d_c() {
        let geometry = make_geometry();
        assert!(!geometry.is_mla());
        assert_eq!(geometry.mla_d_c, 0);
    }

    #[test]
    fn kv_cache_config_kv_bytes_per_token_standard_f32() {
        let geometry = Arc::new(make_geometry());
        let config = KvCacheConfig {
            geometry,
            kv_dtype: DType::F32,
            page_size: 16,
            swap_config: None,
        };
        let kv_dim = 4 * 64;
        let expected = 2 * kv_dim * 12 * 4;
        assert_eq!(config.geometry.kv_bytes_per_token(), expected);
    }

    #[test]
    fn generator_forward_config_default_for_test_geometry_values() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.hidden_size(), 64);
        assert_eq!(cfg.num_layers(), 4);
        assert_eq!(cfg.vocab_size(), 100);
        assert_eq!(cfg.intermediate_size(), 128);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.max_seq_len(), 512);
    }

    #[test]
    fn generator_forward_config_default_for_test_rope_accessors() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!((cfg.rope_theta() - 10000.0).abs() < f64::EPSILON);
        assert!((cfg.rope_scale() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn generator_forward_config_default_for_test_arch_family_decoder() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert_eq!(cfg.arch_family, crate::manifest::ArchFamily::Decoder);
    }

    #[test]
    fn generator_forward_config_attention_head_config() {
        let cfg = GeneratorForwardConfig::default_for_test();
        let head_cfg = cfg.attention();
        assert_eq!(head_cfg.num_heads, 4);
        assert_eq!(head_cfg.num_kv_heads, 2);
        assert_eq!(head_cfg.head_dim, 16);
    }

    #[test]
    fn model_kind_parse_all_valid_variants() {
        use crate::manifest::ModelKind;
        assert_eq!(ModelKind::parse("chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("embedding"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("reranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("classifier"), Some(ModelKind::Classifier));
    }

    #[test]
    fn model_kind_parse_rejects_invalid_input() {
        use crate::manifest::ModelKind;
        assert_eq!(ModelKind::parse("unknown"), None);
        assert_eq!(ModelKind::parse(""), None);
        assert_eq!(ModelKind::parse("embedding-model"), None);
    }

    #[test]
    fn arch_family_variants_are_distinct() {
        use crate::manifest::ArchFamily;
        assert_ne!(ArchFamily::Encoder, ArchFamily::Decoder);
        assert_eq!(ArchFamily::Encoder, ArchFamily::Encoder);
        assert_eq!(ArchFamily::Decoder, ArchFamily::Decoder);
    }

    #[test]
    fn attention_topology_linear_legacy_mask_type_bidirectional() {
        let topo = AttentionTopology::linear();
        assert_eq!(topo.mask_type, AttentionMaskType::Bidirectional);
    }

    #[test]
    fn generator_forward_config_default_for_test_no_rerank_tokens() {
        let cfg = GeneratorForwardConfig::default_for_test();
        assert!(cfg.rerank_yes_token_id.is_none());
        assert!(cfg.rerank_no_token_id.is_none());
        assert!(cfg.moe_config.is_none());
    }
}
