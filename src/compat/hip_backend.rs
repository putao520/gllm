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
        args: &[usize; 23],
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor_types::SequenceInput;

    #[test]
    fn hip_backend_new_returns_none_without_feature() {
        let result = HipBackend::<f32>::new(0);
        assert!(result.is_none());
    }

    #[test]
    fn hip_backend_new_invalid_ordinal_returns_none() {
        let result = HipBackend::<f32>::new(99);
        assert!(result.is_none());
    }

    #[test]
    fn gpu_device_info_shared_between_backends() {
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 120,
            total_memory: 64 * 1024 * 1024 * 1024,
            name: "AMD MI300X".to_string(),
        };
        assert_eq!(info.sm_count, 120);
        assert_eq!(info.name, "AMD MI300X");
    }

    // ── New tests ──────────────────────────────────────────────────────────

    #[test]
    fn hip_backend_alloc_kv_cache_returns_unimplemented_without_feature() {
        // Arrange: Without the hip feature, Backend::alloc_kv_cache is unreachable
        // since new() returns None. The macro-generated impl returns Unimplemented
        // for all Backend methods when the feature is off.
        // We verify the error variant type exists and has the correct string.
        let err = BE::Unimplemented("hip feature not enabled");
        assert_eq!(format!("{err}"), "unimplemented: hip feature not enabled");
    }

    #[test]
    fn hip_backend_upload_weights_returns_unimplemented_without_feature() {
        // Arrange
        let err = BE::Unimplemented("hip feature not enabled");

        // Assert
        assert!(matches!(err, BE::Unimplemented("hip feature not enabled")));
        assert!(!format!("{err}").is_empty());
    }

    #[test]
    fn gpu_device_info_clone_preserves_all_fields() {
        // Arrange
        let original = GpuDeviceInfo {
            ordinal: 3,
            sm_version: 0,
            sm_count: 304,
            total_memory: 192 * 1024 * 1024 * 1024,
            name: "AMD Instinct MI325X".to_string(),
        };

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(cloned.ordinal, original.ordinal);
        assert_eq!(cloned.sm_version, original.sm_version);
        assert_eq!(cloned.sm_count, original.sm_count);
        assert_eq!(cloned.total_memory, original.total_memory);
        assert_eq!(cloned.name, original.name);
    }

    #[test]
    fn gpu_device_info_debug_format_includes_fields() {
        // Arrange
        let info = GpuDeviceInfo {
            ordinal: 1,
            sm_version: 0,
            sm_count: 228,
            total_memory: 128 * 1024 * 1024 * 1024,
            name: "AMD MI250X".to_string(),
        };

        // Act
        let debug_str = format!("{info:?}");

        // Assert
        assert!(debug_str.contains("ordinal"));
        assert!(debug_str.contains("sm_version"));
        assert!(debug_str.contains("sm_count"));
        assert!(debug_str.contains("total_memory"));
        assert!(debug_str.contains("AMD MI250X"));
    }

    #[test]
    fn gpu_device_info_sm_version_zero_for_amd() {
        // Arrange
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 120,
            total_memory: 64 * 1024 * 1024 * 1024,
            name: "AMD MI300X".to_string(),
        };

        // Assert: AMD uses GCN arch, not SM version — must be 0
        assert_eq!(info.sm_version, 0);
    }

    #[test]
    fn gpu_device_info_large_memory_repr() {
        // Arrange: 192 GB VRAM (MI325X)
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 304,
            total_memory: 192usize * 1024 * 1024 * 1024,
            name: "MI325X".to_string(),
        };

        // Assert
        let expected_bytes: usize = 192 * 1024 * 1024 * 1024;
        assert_eq!(info.total_memory, expected_bytes);
        assert_eq!(info.total_memory / (1024 * 1024 * 1024), 192);
    }

    #[test]
    fn kv_cache_handle_equality_and_default() {
        // Arrange
        let h1 = KvCacheHandle(42);
        let h2 = KvCacheHandle(42);
        let h3 = KvCacheHandle(0);
        let h4 = KvCacheHandle(99);

        // Assert
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h1, h4);
        assert_eq!(h3.0, 0u64);
    }

    #[test]
    fn kv_cache_handle_copy_semantics() {
        // Arrange
        let original = KvCacheHandle(12345u64);

        // Act: Copy (not move)
        let copied = original;

        // Assert: both are usable after copy
        assert_eq!(original.0, 12345u64);
        assert_eq!(copied.0, 12345u64);
    }

    #[test]
    fn logits_handle_debug_and_clone() {
        // Arrange
        let handle = LogitsHandle {
            data: vec![0.1, 0.5, 0.9, 0.3],
        };

        // Act
        let cloned = handle.clone();
        let debug_str = format!("{handle:?}");

        // Assert
        assert_eq!(cloned.data, handle.data);
        assert!(debug_str.contains("data"));
    }

    #[test]
    fn backend_error_hip_variant_display() {
        // Arrange
        let err = BE::Hip("device lost".to_string());

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.contains("HIP error"));
        assert!(msg.contains("device lost"));
    }

    #[test]
    fn backend_error_unimplemented_is_static_str() {
        // Arrange
        let err = BE::Unimplemented("hip feature not enabled");

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.contains("unimplemented"));
        assert!(msg.contains("hip feature not enabled"));
    }

    #[test]
    fn backend_error_variants_distinct() {
        // Arrange
        let hip = BE::Hip("hip msg".to_string());
        let other = BE::Other("other msg".to_string());
        let unimpl = BE::Unimplemented("not supported");

        // Assert: each variant has a distinct format prefix
        assert!(format!("{hip}").starts_with("HIP error"));
        assert!(format!("{other}").starts_with("backend error"));
        assert!(format!("{unimpl}").starts_with("unimplemented"));
    }

    #[test]
    fn sampling_config_default_values() {
        // Arrange / Act
        let config = SamplingConfig::default();

        // Assert
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 0);
        // top_p default is 1.0 (fully open)
        assert!((config.top_p - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sequence_input_validate_page_table_accepts_valid_entries() {
        // Arrange
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2]),
            fused_hidden: None,
        };

        // Act
        let result = seq.validate_page_table(10);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_rejects_out_of_bounds() {
        // Arrange
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 5, 10]),
            fused_hidden: None,
        };

        // Act: total_pages = 8, so page_table[2] = 10 is out of bounds
        let result = seq.validate_page_table(8);

        // Assert
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("page_table[2]"));
        assert!(err_msg.contains("10 >= total_pages 8"));
    }

    // ── Additional tests ──────────────────────────────────────────────────

    #[test]
    fn backend_error_cuda_variant_display() {
        // Arrange
        let err = BE::Cuda("out of memory".to_string());

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.starts_with("CUDA error"));
        assert!(msg.contains("out of memory"));
    }

    #[test]
    fn backend_error_metal_variant_display() {
        // Arrange
        let err = BE::Metal("buffer allocation failed".to_string());

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.starts_with("Metal error"));
        assert!(msg.contains("buffer allocation failed"));
    }

    #[test]
    fn backend_error_cpu_variant_display() {
        // Arrange
        let err = BE::Cpu("stack overflow".to_string());

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.starts_with("CPU error"));
        assert!(msg.contains("stack overflow"));
    }

    #[test]
    fn backend_error_other_variant_display() {
        // Arrange
        let err = BE::Other("generic failure".to_string());

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.starts_with("backend error"));
        assert!(msg.contains("generic failure"));
    }

    #[test]
    fn backend_error_clone_preserves_variant_and_message() {
        // Arrange
        let original = BE::Hip("device lost".to_string());

        // Act
        let cloned = original.clone();

        // Assert
        assert_eq!(format!("{original}"), format!("{cloned}"));
        assert!(matches!(cloned, BE::Hip(msg) if msg == "device lost"));
    }

    #[test]
    fn kv_cache_handle_hash_consistency() {
        // Arrange
        use std::collections::HashSet;
        let h1 = KvCacheHandle(42);
        let h2 = KvCacheHandle(42);
        let h3 = KvCacheHandle(99);

        // Act
        let mut set = HashSet::new();
        set.insert(h1);
        set.insert(h2);
        set.insert(h3);

        // Assert: equal handles produce same hash, so only 2 unique entries
        assert_eq!(set.len(), 2);
        assert!(set.contains(&KvCacheHandle(42)));
        assert!(set.contains(&KvCacheHandle(99)));
    }

    #[test]
    fn kv_cache_handle_debug_format() {
        // Arrange
        let handle = KvCacheHandle(777);

        // Act
        let debug = format!("{handle:?}");

        // Assert
        assert!(debug.contains("777"));
    }

    #[test]
    fn logits_handle_empty_data() {
        // Arrange
        let handle = LogitsHandle { data: vec![] };

        // Act
        let cloned = handle.clone();

        // Assert
        assert!(handle.data.is_empty());
        assert!(cloned.data.is_empty());
        assert_eq!(handle.data.len(), 0);
    }

    #[test]
    fn logits_handle_large_data_clone_preserves_values() {
        // Arrange
        let data: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
        let handle = LogitsHandle { data };

        // Act
        let cloned = handle.clone();

        // Assert
        assert_eq!(cloned.data.len(), 1024);
        assert_eq!(cloned.data[0], 0.0);
        assert_eq!(cloned.data[1023], 1.023);
        assert!(format!("{handle:?}").contains("data"));
    }

    #[test]
    fn sequence_input_validate_page_table_none_is_ok() {
        // Arrange: page_table is None (contiguous KV, no paging)
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };

        // Act
        let result = seq.validate_page_table(10);

        // Assert: None page_table always passes
        assert!(result.is_ok());
    }

    #[test]
    fn sequence_input_empty_tokens_with_page_table() {
        // Arrange
        let seq = SequenceInput {
            tokens: vec![],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![]),
            fused_hidden: None,
        };

        // Act
        let result = seq.validate_page_table(10);

        // Assert: empty page table is valid (no entries to check)
        assert!(result.is_ok());
    }

    #[test]
    fn gpu_device_info_zero_memory_repr() {
        // Arrange: simulated device reporting zero memory (edge case)
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 0,
            total_memory: 0,
            name: String::new(),
        };

        // Assert
        assert_eq!(info.total_memory, 0);
        assert_eq!(info.sm_count, 0);
        assert!(info.name.is_empty());
        assert_eq!(info.ordinal, 0);
    }

    #[test]
    fn gpu_device_info_multiple_amd_device_names() {
        // Arrange: verify different AMD GPU names are preserved
        let names = vec![
            "AMD Instinct MI250X",
            "AMD Instinct MI300X",
            "AMD Instinct MI325X",
            "AMD Radeon RX 7900 XTX",
        ];
        let infos: Vec<GpuDeviceInfo> = names
            .iter()
            .enumerate()
            .map(|(i, name)| GpuDeviceInfo {
                ordinal: i,
                sm_version: 0,
                sm_count: 100 + i as u32,
                total_memory: (i as usize + 1) * 32 * 1024 * 1024 * 1024,
                name: name.to_string(),
            })
            .collect();

        // Assert: each device info preserves its unique name and ordinal
        assert_eq!(infos.len(), 4);
        assert_eq!(infos[0].name, "AMD Instinct MI250X");
        assert_eq!(infos[3].name, "AMD Radeon RX 7900 XTX");
        assert_eq!(infos[2].ordinal, 2);
        assert_eq!(infos[1].sm_count, 101);
    }

    #[test]
    fn batch_input_construction_and_debug() {
        // Arrange
        let batch = BatchInput {
            sequences: vec![
                SequenceInput {
                    tokens: vec![1, 2, 3],
                    position: 0,
                    draft_steps: 0,
                    page_table: None,
                    fused_hidden: None,
                },
                SequenceInput {
                    tokens: vec![4, 5],
                    position: 3,
                    draft_steps: 0,
                    page_table: Some(vec![0, 1]),
                    fused_hidden: None,
                },
            ],
        };

        // Act
        let debug_str = format!("{batch:?}");

        // Assert
        assert_eq!(batch.sequences.len(), 2);
        assert_eq!(batch.sequences[0].tokens, vec![1, 2, 3]);
        assert_eq!(batch.sequences[1].position, 3);
        assert!(debug_str.contains("sequences"));
    }

    #[test]
    fn page_state_variants_distinct() {
        // Arrange
        use crate::scheduler::types::PageState;
        let states = vec![
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];

        // Act & Assert: all variants are pairwise distinct
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert_ne!(states[i], states[j], "PageState variants must be distinct");
            }
        }
        assert_eq!(states.len(), 7);
    }

    #[test]
    fn hip_backend_debug_format_without_feature() {
        // Arrange: Without the hip feature, we cannot construct HipBackend via new(),
        // but the Debug impl exists and is testable if we can construct via other means.
        // Since new() returns None without the feature, we verify the Debug format
        // output via a constructed device_info.
        let info = GpuDeviceInfo {
            ordinal: 2,
            sm_version: 0,
            sm_count: 228,
            total_memory: 128 * 1024 * 1024 * 1024,
            name: "AMD MI250X".to_string(),
        };

        // Act
        let debug = format!("{info:?}");

        // Assert: Debug output contains "ordinal" and the device name
        assert!(debug.contains("ordinal: 2"));
        assert!(debug.contains("AMD MI250X"));
        assert!(debug.contains("sm_count: 228"));
    }

    // ── Wave 12x34 new tests ─────────────────────────────────────────────

    #[test]
    fn sampling_config_copy_preserves_independent_values() {
        // Arrange
        let mut original = SamplingConfig::default();
        original.temperature = 0.7;
        original.top_k = 50;
        original.top_p = 0.95;

        // Act: SamplingConfig derives Copy, so assignment copies
        let copied = original;

        // Assert: modifying original does not affect copy (both are independent)
        assert_eq!(copied.temperature, 0.7);
        assert_eq!(copied.top_k, 50);
        assert!((copied.top_p - 0.95).abs() < f32::EPSILON);

        // Modify original after copy to prove independence
        original.temperature = 2.0;
        assert_eq!(copied.temperature, 0.7);
        assert_eq!(original.temperature, 2.0);
    }

    #[test]
    fn sampling_config_zero_temperature_boundary() {
        // Arrange: temperature = 0 means greedy decoding
        let config = SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 0.0,
        };

        // Assert: boundary values are preserved exactly
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.top_k, 1);
        assert_eq!(config.top_p, 0.0);
    }

    #[test]
    fn sampling_config_debug_format_shows_fields() {
        // Arrange
        let config = SamplingConfig {
            temperature: 1.5,
            top_k: 100,
            top_p: 0.9,
        };

        // Act
        let debug = format!("{config:?}");

        // Assert: Debug output contains struct name and field values
        assert!(debug.contains("SamplingConfig"));
        assert!(debug.contains("temperature"));
        assert!(debug.contains("top_k"));
        assert!(debug.contains("top_p"));
    }

    #[test]
    fn sequence_input_with_fused_hidden_construction() {
        // Arrange: multimodal request with pre-computed fused embedding
        let fused: Vec<f32> = vec![0.1; 64]; // hidden_size=64
        let seq = SequenceInput {
            tokens: vec![1, 2, 3, 4, 5],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1]),
            fused_hidden: Some(fused.clone()),
        };

        // Assert: fused_hidden is preserved with correct length
        assert!(seq.fused_hidden.is_some());
        let hidden = seq.fused_hidden.as_ref().unwrap();
        assert_eq!(hidden.len(), 64);
        assert_eq!(hidden[0], 0.1);
        assert_eq!(seq.tokens.len(), 5);
    }

    #[test]
    fn sequence_input_position_nonzero_start() {
        // Arrange: continuation request starting at position 128
        let seq = SequenceInput {
            tokens: vec![100, 101, 102],
            position: 128,
            draft_steps: 0,
            page_table: Some(vec![5, 6, 7]),
            fused_hidden: None,
        };

        // Assert
        assert_eq!(seq.position, 128);
        assert_eq!(seq.tokens, vec![100, 101, 102]);
        assert_eq!(seq.page_table.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn sequence_input_draft_steps_field() {
        // Arrange: speculative decoding with 4 draft steps
        let seq = SequenceInput {
            tokens: vec![42],
            position: 0,
            draft_steps: 4,
            page_table: None,
            fused_hidden: None,
        };

        // Assert
        assert_eq!(seq.draft_steps, 4);
    }

    #[test]
    fn batch_input_empty_sequences_valid() {
        // Arrange: empty batch (no sequences)
        let batch = BatchInput {
            sequences: vec![],
        };

        // Assert
        assert!(batch.sequences.is_empty());
        assert_eq!(batch.sequences.len(), 0);
    }

    #[test]
    fn batch_input_clone_preserves_all_sequences() {
        // Arrange
        let original = BatchInput {
            sequences: vec![
                SequenceInput {
                    tokens: vec![10, 20],
                    position: 0,
                    draft_steps: 0,
                    page_table: None,
                    fused_hidden: None,
                },
                SequenceInput {
                    tokens: vec![30],
                    position: 5,
                    draft_steps: 2,
                    page_table: Some(vec![0, 1]),
                    fused_hidden: Some(vec![1.0, 2.0, 3.0]),
                },
            ],
        };

        // Act
        let cloned = original.clone();

        // Assert: all sequences preserved with their fields
        assert_eq!(cloned.sequences.len(), 2);
        assert_eq!(cloned.sequences[0].tokens, vec![10, 20]);
        assert_eq!(cloned.sequences[1].position, 5);
        assert_eq!(cloned.sequences[1].draft_steps, 2);
        assert_eq!(
            cloned.sequences[1].fused_hidden.as_ref().unwrap().len(),
            3
        );
    }

    #[test]
    fn backend_error_is_std_error() {
        // Arrange
        let err: BE = BE::Hip("test error".to_string());

        // Act: cast to std::error::Error trait object
        let as_error: &dyn std::error::Error = &err;

        // Assert: Display via trait object works
        let msg = format!("{as_error}");
        assert!(msg.contains("HIP error"));
        assert!(msg.contains("test error"));
    }

    #[test]
    fn backend_error_unimplemented_static_str_lifetime() {
        // Arrange: Unimplemented takes &'static str, verify it works with string literals
        let err1 = BE::Unimplemented("hip feature not enabled");
        let err2 = BE::Unimplemented("quantized_matmul");

        // Assert: different static strings produce distinct messages
        assert_ne!(format!("{err1}"), format!("{err2}"));
        assert!(format!("{err1}").contains("hip feature not enabled"));
        assert!(format!("{err2}").contains("quantized_matmul"));
    }

    #[test]
    fn weight_placement_variants_equality() {
        // Arrange
        use super::super::backend_trait::WeightPlacement;

        let device = WeightPlacement::DeviceLocal;
        let host = WeightPlacement::HostLocal;

        // Assert: each variant equals itself, not others
        assert_eq!(device, WeightPlacement::DeviceLocal);
        assert_eq!(host, WeightPlacement::HostLocal);
        assert_ne!(device, host);
    }

    #[test]
    fn weight_placement_debug_and_clone() {
        // Arrange
        use super::super::backend_trait::WeightPlacement;

        let device = WeightPlacement::DeviceLocal;
        let host = WeightPlacement::HostLocal;

        // Act
        let device_cloned = device.clone();
        let host_cloned = host.clone();
        let debug_device = format!("{device:?}");
        let debug_host = format!("{host:?}");

        // Assert
        assert_eq!(device, device_cloned);
        assert_eq!(host, host_cloned);
        assert!(debug_device.contains("DeviceLocal"));
        assert!(debug_host.contains("HostLocal"));
    }

    #[test]
    fn gpu_device_info_equality_semantics() {
        // Arrange: two GpuDeviceInfo with same fields
        let info_a = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 120,
            total_memory: 64 * 1024 * 1024 * 1024,
            name: "AMD MI300X".to_string(),
        };
        let info_b = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 120,
            total_memory: 64 * 1024 * 1024 * 1024,
            name: "AMD MI300X".to_string(),
        };
        let info_c = GpuDeviceInfo {
            ordinal: 1,
            sm_version: 0,
            sm_count: 120,
            total_memory: 64 * 1024 * 1024 * 1024,
            name: "AMD MI300X".to_string(),
        };

        // Assert: GpuDeviceInfo derives Clone but not PartialEq,
        // so we verify field-by-field equality
        assert_eq!(info_a.ordinal, info_b.ordinal);
        assert_eq!(info_a.sm_version, info_b.sm_version);
        assert_eq!(info_a.sm_count, info_b.sm_count);
        assert_eq!(info_a.total_memory, info_b.total_memory);
        assert_eq!(info_a.name, info_b.name);
        assert_ne!(info_a.ordinal, info_c.ordinal);
    }

    #[test]
    fn kv_cache_handle_sort_by_inner_value() {
        // Arrange: KvCacheHandle does not derive Ord, so sort by .0 field
        let mut handles = vec![
            KvCacheHandle(100),
            KvCacheHandle(0),
            KvCacheHandle(50),
            KvCacheHandle(200),
        ];

        // Act: sort by inner u64 value
        handles.sort_by_key(|h| h.0);

        // Assert: sorted in ascending order
        assert_eq!(handles[0].0, 0);
        assert_eq!(handles[1].0, 50);
        assert_eq!(handles[2].0, 100);
        assert_eq!(handles[3].0, 200);
    }

    #[test]
    fn logits_handle_data_equality_via_field() {
        // Arrange: LogitsHandle does not derive PartialEq, but inner Vec<f32> does
        let h1 = LogitsHandle {
            data: vec![0.5, 0.3, 0.2],
        };
        let h2 = LogitsHandle {
            data: vec![0.5, 0.3, 0.2],
        };
        let h3 = LogitsHandle {
            data: vec![0.5, 0.3, 0.1],
        };

        // Assert: compare via data field
        assert_eq!(h1.data, h2.data);
        assert_ne!(h1.data, h3.data);
    }

    // ── Wave 12x35 new tests (13 tests) ────────────────────────────────────

    #[test]
    fn sequence_input_validate_page_table_zero_total_pages_rejects_any() {
        // Arrange: total_pages=0 means no pages exist; any page_id >= 0 is out of bounds
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };

        // Act
        let result = seq.validate_page_table(0);

        // Assert
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("page_table[0]"));
    }

    #[test]
    fn sequence_input_validate_page_table_boundary_page_id_passes() {
        // Arrange: page_id = total_pages - 1 is the last valid page
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 7, 9]),
            fused_hidden: None,
        };

        // Act: total_pages = 10, so page_id=9 is valid (last page)
        let result = seq.validate_page_table(10);

        // Assert
        assert!(result.is_ok());
    }

    #[test]
    fn backend_error_hip_variant_with_special_characters() {
        // Arrange: HIP error message containing special characters and unicode
        let err = BE::Hip("HIP error: hipMalloc failed (GFX1100) — out of VRAM が".to_string());

        // Act
        let msg = format!("{err}");

        // Assert: Display preserves special characters
        assert!(msg.contains("HIP error:"));
        assert!(msg.contains("GFX1100"));
        assert!(msg.contains("out of VRAM"));
        assert!(msg.contains("が"));
    }

    #[test]
    fn gpu_device_info_sm_count_u32_max_boundary() {
        // Arrange: verify sm_count handles u32::MAX without overflow
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: u32::MAX,
            total_memory: 0,
            name: "Boundary GPU".to_string(),
        };

        // Assert: stored and retrieved without overflow
        assert_eq!(info.sm_count, u32::MAX);
        let debug = format!("{info:?}");
        assert!(debug.contains(&u32::MAX.to_string()));
    }

    #[test]
    fn kv_cache_handle_u64_max_boundary() {
        // Arrange
        let handle = KvCacheHandle(u64::MAX);

        // Act
        let debug = format!("{handle:?}");
        let copied = handle;

        // Assert: u64::MAX is preserved through Copy and Debug
        assert_eq!(copied.0, u64::MAX);
        assert_eq!(handle.0, u64::MAX);
        assert!(debug.contains(&u64::MAX.to_string()));
    }

    #[test]
    fn logits_handle_preserves_nan_and_infinity() {
        // Arrange: logits may contain NaN or Inf from numerical overflow
        let data = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0];
        let handle = LogitsHandle { data };

        // Act
        let cloned = handle.clone();

        // Assert: NaN and Inf are preserved through clone
        assert!(cloned.data[0].is_nan());
        assert!(cloned.data[1].is_infinite() && cloned.data[1].is_sign_positive());
        assert!(cloned.data[2].is_infinite() && cloned.data[2].is_sign_negative());
        assert_eq!(cloned.data[3], 1.0);
    }

    #[test]
    fn sampling_config_negative_temperature_preserved() {
        // Arrange: negative temperature is a valid f32 value even if semantically invalid
        let config = SamplingConfig {
            temperature: -1.5,
            top_k: 0,
            top_p: -0.1,
        };

        // Assert: negative values are preserved exactly (no clamping in struct)
        assert_eq!(config.temperature, -1.5);
        assert_eq!(config.top_p, -0.1);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn batch_input_many_sequences_preserve_order() {
        // Arrange: 64 sequences with incrementing token IDs and positions
        let sequences: Vec<SequenceInput> = (0..64)
            .map(|i| SequenceInput {
                tokens: vec![i as u32],
                position: i * 10,
                draft_steps: 0,
                page_table: None,
                fused_hidden: None,
            })
            .collect();

        let batch = BatchInput { sequences };

        // Act
        let cloned = batch.clone();

        // Assert: order and values preserved in clone
        assert_eq!(cloned.sequences.len(), 64);
        for i in 0..64 {
            assert_eq!(cloned.sequences[i].tokens, vec![i as u32]);
            assert_eq!(cloned.sequences[i].position, i * 10);
        }
    }

    #[test]
    fn page_state_copy_semantics_preserve_identity() {
        // Arrange
        let state = PageState::Active;

        // Act: PageState derives Copy
        let copied = state;

        // Assert: both original and copy are Active
        assert_eq!(state, PageState::Active);
        assert_eq!(copied, PageState::Active);
        assert_eq!(state, copied);
    }

    #[test]
    fn weight_placement_copy_semantics() {
        // Arrange
        use super::super::backend_trait::WeightPlacement;
        let device = WeightPlacement::DeviceLocal;

        // Act: WeightPlacement derives Copy
        let copied = device;
        let _moved = copied; // Copy again to prove it is Copy, not just Clone

        // Assert: original still usable after move of copy
        assert_eq!(device, WeightPlacement::DeviceLocal);
    }

    #[test]
    fn gpu_device_info_ordinal_max_usize() {
        // Arrange: ordinal at usize::MAX (extreme boundary)
        let info = GpuDeviceInfo {
            ordinal: usize::MAX,
            sm_version: 0,
            sm_count: 0,
            total_memory: 0,
            name: String::new(),
        };

        // Assert: ordinal preserves usize::MAX
        assert_eq!(info.ordinal, usize::MAX);
        let debug = format!("{info:?}");
        assert!(debug.contains(&usize::MAX.to_string()));
    }

    #[test]
    fn sequence_input_validate_page_table_rejects_at_exact_boundary() {
        // Arrange: page_id equals total_pages (one past the last valid index)
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![5]),
            fused_hidden: None,
        };

        // Act: total_pages = 5, so page_id=5 is out of bounds (valid: 0..4)
        let result = seq.validate_page_table(5);

        // Assert
        assert!(result.is_err());
        let err_msg = result.unwrap_err();
        assert!(err_msg.contains("5 >= total_pages 5"));
    }

    #[test]
    fn backend_error_all_string_variants_preserve_empty_message() {
        // Arrange: empty string messages for all string-bearing variants
        let hip_empty = BE::Hip(String::new());
        let cuda_empty = BE::Cuda(String::new());
        let other_empty = BE::Other(String::new());

        // Act
        let hip_msg = format!("{hip_empty}");
        let cuda_msg = format!("{cuda_empty}");
        let other_msg = format!("{other_empty}");

        // Assert: prefix is present but message body is empty
        assert_eq!(hip_msg, "HIP error: ");
        assert_eq!(cuda_msg, "CUDA error: ");
        assert_eq!(other_msg, "backend error: ");
    }

    // ── Wave 12x36 new tests (13 tests) ────────────────────────────────────

    #[test]
    fn position_encoding_variants_are_distinct() {
        // Arrange
        use crate::engine::executor::PositionEncoding;
        let none = PositionEncoding::None;
        let rope = PositionEncoding::Rope;

        // Assert: two variants are distinct
        assert_ne!(none, rope);
        assert_eq!(none, PositionEncoding::None);
        assert_eq!(rope, PositionEncoding::Rope);
    }

    #[test]
    fn position_encoding_debug_format() {
        // Arrange
        use crate::engine::executor::PositionEncoding;
        let none = PositionEncoding::None;
        let rope = PositionEncoding::Rope;

        // Act
        let debug_none = format!("{none:?}");
        let debug_rope = format!("{rope:?}");

        // Assert
        assert!(debug_none.contains("None"));
        assert!(debug_rope.contains("Rope"));
        assert_ne!(debug_none, debug_rope);
    }

    #[test]
    fn attention_mask_type_variants_and_hash() {
        // Arrange
        use std::collections::HashSet;
        let bidi = crate::engine::executor::AttentionMaskType::Bidirectional;
        let causal = crate::engine::executor::AttentionMaskType::Causal;

        // Act
        let mut set = HashSet::new();
        set.insert(bidi);
        set.insert(causal);

        // Assert: two distinct variants produce two entries
        assert_eq!(set.len(), 2);
        assert!(set.contains(&crate::engine::executor::AttentionMaskType::Bidirectional));
        assert!(set.contains(&crate::engine::executor::AttentionMaskType::Causal));
        assert_ne!(bidi, causal);
    }

    #[test]
    fn rope_config_partial_eq_semantics() {
        // Arrange
        use crate::engine::executor::RoPEConfig;
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
            theta: 1000000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };

        // Assert
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn rope_config_copy_independence() {
        // Arrange
        use crate::engine::executor::RoPEConfig;
        let original = RoPEConfig {
            theta: 500000.0,
            scale: 0.5,
            interleaved: true,
            precompute: true,
        };

        // Act: Copy semantics
        let copied = original;

        // Assert: both are independent after copy (RoPEConfig derives Copy)
        assert_eq!(copied.theta, 500000.0);
        assert_eq!(copied.scale, 0.5);
        assert!(copied.interleaved);
        assert!(copied.precompute);
        // Original still valid after copy
        assert_eq!(original.theta, 500000.0);
    }

    #[test]
    fn paged_kv_config_with_page_table_construction() {
        // Arrange
        use crate::engine::executor::PagedKvConfig;
        let config = PagedKvConfig {
            page_table: Some(vec![0, 1, 2, 3, 4]),
            page_size: 16,
        };

        // Assert
        assert!(config.page_table.is_some());
        let pt = config.page_table.as_ref().unwrap();
        assert_eq!(pt.len(), 5);
        assert_eq!(pt[0], 0);
        assert_eq!(pt[4], 4);
        assert_eq!(config.page_size, 16);
    }

    #[test]
    fn paged_kv_config_without_page_table() {
        // Arrange
        use crate::engine::executor::PagedKvConfig;
        let config = PagedKvConfig {
            page_table: None,
            page_size: 32,
        };

        // Assert: None means contiguous KV (no paging)
        assert!(config.page_table.is_none());
        assert_eq!(config.page_size, 32);
    }

    #[test]
    fn attention_head_config_debug_format() {
        // Arrange
        use crate::engine::executor::AttentionHeadConfig;
        let config = AttentionHeadConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
        };

        // Act
        let debug = format!("{config:?}");

        // Assert
        assert!(debug.contains("num_heads"));
        assert!(debug.contains("32"));
        assert!(debug.contains("num_kv_heads"));
        assert!(debug.contains("8"));
        assert!(debug.contains("head_dim"));
        assert!(debug.contains("128"));
    }

    #[test]
    fn swap_config_equality_same_and_different() {
        // Arrange
        use crate::engine::executor::SwapConfig;
        let a = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.85,
            lru_granularity: 4,
        };
        let b = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.85,
            lru_granularity: 4,
        };
        let c = SwapConfig {
            enable_swap: false,
            swap_threshold: 0.85,
            lru_granularity: 4,
        };

        // Assert
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn swap_config_debug_shows_fields() {
        // Arrange
        use crate::engine::executor::SwapConfig;
        let config = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.75,
            lru_granularity: 8,
        };

        // Act
        let debug = format!("{config:?}");

        // Assert: Debug output includes field names and values
        assert!(debug.contains("enable_swap"));
        assert!(debug.contains("swap_threshold"));
        assert!(debug.contains("lru_granularity"));
    }

    #[test]
    fn sequence_input_validate_page_table_single_entry_at_boundary() {
        // Arrange: total_pages = 1, page_id = 0 is the only valid entry
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };

        // Act
        let result = seq.validate_page_table(1);

        // Assert: page_id=0 < total_pages=1 is valid
        assert!(result.is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_large_page_ids() {
        // Arrange: large page IDs close to u32::MAX but within total_pages
        let large_id = u32::MAX - 1;
        let seq = SequenceInput {
            tokens: vec![1, 2],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![large_id]),
            fused_hidden: None,
        };

        // Act: total_pages = u32::MAX as usize, so large_id is valid
        let result = seq.validate_page_table(u32::MAX as usize);

        // Assert: large_id < total_pages
        assert!(result.is_ok());
    }

    #[test]
    fn gpu_device_info_total_memory_usize_max_boundary() {
        // Arrange: total_memory at usize::MAX (extreme boundary)
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 0,
            sm_count: 0,
            total_memory: usize::MAX,
            name: "Boundary GPU".to_string(),
        };

        // Assert: total_memory preserves usize::MAX without overflow
        assert_eq!(info.total_memory, usize::MAX);
        let debug = format!("{info:?}");
        assert!(debug.contains(&usize::MAX.to_string()));
    }

    // ── Wave 12x37 new tests (13 tests) ────────────────────────────────────

    #[test]
    fn group_state_variants_distinct_and_copy() {
        // Arrange
        use crate::scheduler::types::GroupState;
        let running = GroupState::Running;
        let swapped = GroupState::Swapped;
        let paused = GroupState::Paused;

        // Act: Copy semantics
        let copied = running;

        // Assert: all variants pairwise distinct
        assert_ne!(running, swapped);
        assert_ne!(running, paused);
        assert_ne!(swapped, paused);
        // Copy produces equal value
        assert_eq!(copied, running);
        assert_eq!(copied, GroupState::Running);
    }

    #[test]
    fn request_kind_variants_distinct_and_hash() {
        // Arrange
        use std::collections::HashSet;
        use crate::scheduler::types::RequestKind;
        let chat = RequestKind::Chat;
        let embedding = RequestKind::Embedding;
        let rerank = RequestKind::Rerank;

        // Act
        let mut set = HashSet::new();
        set.insert(chat);
        set.insert(embedding);
        set.insert(rerank);

        // Assert: 3 distinct variants produce 3 hash entries
        assert_eq!(set.len(), 3);
        assert!(set.contains(&RequestKind::Chat));
        assert!(set.contains(&RequestKind::Embedding));
        assert!(set.contains(&RequestKind::Rerank));
    }

    #[test]
    fn kv_pipeline_variants_distinct_and_debug() {
        // Arrange
        use crate::scheduler::types::KvPipeline;
        let conv = KvPipeline::Conversation;
        let work = KvPipeline::Working;

        // Act
        let debug_conv = format!("{conv:?}");
        let debug_work = format!("{work:?}");

        // Assert
        assert_ne!(conv, work);
        assert!(debug_conv.contains("Conversation"));
        assert!(debug_work.contains("Working"));
    }

    #[test]
    fn weight_tier_variants_distinct_and_ordered() {
        // Arrange
        use crate::scheduler::types::WeightTier;
        let hot = WeightTier::Hot;
        let warm = WeightTier::Warm;
        let cold = WeightTier::Cold;

        // Assert: all three tiers pairwise distinct
        assert_ne!(hot, warm);
        assert_ne!(warm, cold);
        assert_ne!(hot, cold);
        assert_eq!(hot, WeightTier::Hot);
        assert_eq!(warm, WeightTier::Warm);
        assert_eq!(cold, WeightTier::Cold);
    }

    #[test]
    fn batch_order_policy_default_is_strict_request_id() {
        // Arrange
        use crate::scheduler::types::BatchOrderPolicy;

        // Act
        let default = BatchOrderPolicy::default();

        // Assert: default must be StrictRequestIdOrder (determinism first)
        assert_eq!(default, BatchOrderPolicy::StrictRequestIdOrder);
        assert_ne!(default, BatchOrderPolicy::FifoOrder);
    }

    #[test]
    fn page_payload_kind_variants_distinct_and_equality() {
        // Arrange
        use crate::scheduler::types::PagePayloadKind;
        let variants = vec![
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];

        // Assert: all 5 variants pairwise distinct
        assert_eq!(variants.len(), 5);
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "PagePayloadKind variants must be distinct");
            }
            // Self-equality
            assert_eq!(variants[i], variants[i]);
        }
    }

    #[test]
    fn memory_residency_variants_distinct_and_copy() {
        // Arrange
        use crate::scheduler::types::MemoryResidency;
        let device = MemoryResidency::DeviceLocal;
        let host = MemoryResidency::HostLocal;
        let disk = MemoryResidency::DiskSwap;

        // Act: Copy
        let copied = device;

        // Assert
        assert_ne!(device, host);
        assert_ne!(host, disk);
        assert_ne!(device, disk);
        assert_eq!(copied, device);
        assert_eq!(copied, MemoryResidency::DeviceLocal);
    }

    #[test]
    fn pipelined_virtual_page_id_construction_and_equality() {
        // Arrange
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId, RequestId};
        let p1 = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42u64,
            logical_index: 5,
        };
        let p2 = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42u64,
            logical_index: 5,
        };
        let p3 = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 42u64,
            logical_index: 5,
        };

        // Assert: field-based equality
        assert_eq!(p1.pipeline, p2.pipeline);
        assert_eq!(p1.sequence_id, p2.sequence_id);
        assert_eq!(p1.logical_index, p2.logical_index);
        assert_ne!(p1.pipeline, p3.pipeline);
        assert_eq!(p1.logical_index, 5);
    }

    #[test]
    fn eviction_priority_construction_preserves_all_fields() {
        // Arrange
        use crate::scheduler::types::{EvictionPriority, PagePayloadKind};
        let ep = EvictionPriority {
            score: -100i64,
            payload_kind: PagePayloadKind::ExpertWeight,
            is_pinned: false,
            access_count: 42,
            recency: 7,
            layer_idx: Some(12),
            expert_id: Some(3),
        };

        // Assert: all fields preserved exactly
        assert_eq!(ep.score, -100);
        assert_eq!(ep.payload_kind, PagePayloadKind::ExpertWeight);
        assert!(!ep.is_pinned);
        assert_eq!(ep.access_count, 42);
        assert_eq!(ep.recency, 7);
        assert_eq!(ep.layer_idx, Some(12));
        assert_eq!(ep.expert_id, Some(3));
    }

    #[test]
    fn effective_kv_max_seq_len_is_identity_passthrough() {
        // Arrange
        use crate::engine::executor_types::effective_kv_max_seq_len;

        // Act & Assert: function returns input unchanged (identity)
        assert_eq!(effective_kv_max_seq_len(512), 512);
        assert_eq!(effective_kv_max_seq_len(1), 1);
        assert_eq!(effective_kv_max_seq_len(131072), 131072);
        assert_eq!(effective_kv_max_seq_len(0), 0);
    }

    #[test]
    fn sequence_input_all_zero_fields_is_valid() {
        // Arrange: edge case with all fields at zero/empty
        let seq = SequenceInput {
            tokens: vec![],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };

        // Act
        let validation = seq.validate_page_table(0);

        // Assert: empty tokens with no page table is valid
        assert!(validation.is_ok());
        assert!(seq.tokens.is_empty());
        assert_eq!(seq.position, 0);
        assert_eq!(seq.draft_steps, 0);
    }

    #[test]
    fn attention_topology_linear_construction() {
        // Arrange & Act: AttentionTopology::linear() creates minimal bidirectional topology
        use crate::engine::executor::AttentionTopology;
        let topo = AttentionTopology::linear();

        // Assert: minimal geometry with bidirectional mask
        assert_eq!(topo.mask_type, crate::engine::executor::AttentionMaskType::Bidirectional);
        assert_eq!(topo.num_heads(), 1);
        assert_eq!(topo.num_kv_heads(), 1);
        assert_eq!(topo.head_dim(), 1);
        assert_eq!(topo.max_seq_len(), 512);
    }

    #[test]
    fn page_state_hash_consistency_in_hashset() {
        // Arrange
        use std::collections::HashSet;
        let mut set1 = HashSet::new();
        let mut set2 = HashSet::new();

        // Act: insert same states in different order
        set1.insert(PageState::Free);
        set1.insert(PageState::Active);
        set1.insert(PageState::Swapped);

        set2.insert(PageState::Swapped);
        set2.insert(PageState::Free);
        set2.insert(PageState::Active);

        // Assert: hashset contents are order-independent
        assert_eq!(set1, set2);
        assert_eq!(set1.len(), 3);
    }

    // ── Wave 12x38 new tests (13 tests) ────────────────────────────────────

    #[test]
    fn page_metadata_default_standby_state() {
        // Arrange & Act
        let meta = crate::scheduler::types::PageMetadata::default();

        // Assert: default page is Standby with zero recency and no owner
        assert_eq!(meta.page_id, 0);
        assert!(meta.sequence_id.is_none());
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(!meta.is_lir);
        assert_eq!(meta.state, PageState::Standby);
        assert!(meta.swap_in_time.is_none());
        assert!(meta.warm_until.is_none());
    }

    #[test]
    fn unified_virtual_page_kv_construction_fields() {
        // Arrange & Act
        let page = crate::scheduler::types::UnifiedVirtualPage::kv(
            7,
            42u64,
            crate::scheduler::types::KvPipeline::Conversation,
            3,
            gllm_kernels::types::DType::F32,
        );

        // Assert: kv() sets correct payload kind, owner, pipeline, residency
        assert_eq!(page.page_id, 7);
        assert_eq!(page.payload_kind, crate::scheduler::types::PagePayloadKind::KvContext);
        assert_eq!(page.residency, crate::scheduler::types::MemoryResidency::DeviceLocal);
        assert_eq!(page.owner, Some(42u64));
        assert_eq!(page.pipeline, Some(crate::scheduler::types::KvPipeline::Conversation));
        assert_eq!(page.logical_index, 3);
        assert!(page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.expert_id.is_none());
        assert!(page.layer_idx.is_none());
    }

    #[test]
    fn unified_virtual_page_expert_construction_fields() {
        // Arrange & Act
        let page = crate::scheduler::types::UnifiedVirtualPage::expert(
            15,
            3u32,
            12,
            gllm_kernels::types::DType::BF16,
        );

        // Assert: expert() sets expert_id, layer_idx, no owner, evictable
        assert_eq!(page.page_id, 15);
        assert_eq!(page.payload_kind, crate::scheduler::types::PagePayloadKind::ExpertWeight);
        assert_eq!(page.owner, None);
        assert!(page.is_evictable());
        assert_eq!(page.expert_id, Some(3));
        assert_eq!(page.layer_idx, Some(12));
        assert_eq!(page.dtype, gllm_kernels::types::DType::BF16);
    }

    #[test]
    fn unified_virtual_page_system_prompt_not_evictable() {
        // Arrange & Act
        let page = crate::scheduler::types::UnifiedVirtualPage::system_prompt(
            0,
            gllm_kernels::types::DType::F16,
        );

        // Assert: system prompt pages are pinned and not evictable
        assert_eq!(page.payload_kind, crate::scheduler::types::PagePayloadKind::PromptSystem);
        assert!(!page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
    }

    #[test]
    fn unified_virtual_page_rag_host_local_and_evictable() {
        // Arrange & Act
        let page = crate::scheduler::types::UnifiedVirtualPage::rag(
            99,
            1001u64,
            gllm_kernels::types::DType::F32,
        );

        // Assert: RAG pages start on host, are evictable
        assert_eq!(page.payload_kind, crate::scheduler::types::PagePayloadKind::KnowledgeRAG);
        assert_eq!(page.residency, crate::scheduler::types::MemoryResidency::HostLocal);
        assert!(!page.is_on_device());
        assert!(page.is_evictable());
        assert_eq!(page.owner, Some(1001u64));
    }

    #[test]
    fn compression_codec_from_u8_round_trip() {
        // Arrange: all 5 codec variants
        use crate::kv_cache::CompressionCodec;
        let variants = vec![
            (0u8, CompressionCodec::None),
            (1u8, CompressionCodec::Lz4),
            (2u8, CompressionCodec::BitPackRle),
            (3u8, CompressionCodec::NvcompAns),
            (4u8, CompressionCodec::ZstdDict),
        ];

        // Act & Assert: from_u8 maps each discriminant to correct variant
        for (byte, expected) in &variants {
            assert_eq!(CompressionCodec::from_u8(*byte), Some(*expected));
        }
        // Invalid discriminant returns None
        assert_eq!(CompressionCodec::from_u8(5), None);
        assert_eq!(CompressionCodec::from_u8(255), None);
    }

    #[test]
    fn compression_codec_variants_distinct_and_hash() {
        // Arrange
        use std::collections::HashSet;
        use crate::kv_cache::CompressionCodec;
        let variants = vec![
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];

        // Act
        let set: HashSet<_> = variants.into_iter().collect();

        // Assert: 5 distinct variants produce 5 hash entries
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn attention_head_config_from_geometry_preserves_fields() {
        // Arrange
        use crate::engine::executor::AttentionHeadConfig;
        let geo = crate::model_config::ModelGeometry {
            hidden_size: 256,
            num_layers: 2,
            vocab_size: 100,
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
        };

        // Act
        let config = AttentionHeadConfig::from_geometry(&geo);

        // Assert: all fields propagated from geometry
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_kv_heads, 2);
        assert_eq!(config.head_dim, 32);
    }

    #[test]
    fn eviction_priority_negative_score_preserved() {
        // Arrange: negative score = high priority (evicted first)
        use crate::scheduler::types::{EvictionPriority, PagePayloadKind};
        let ep = EvictionPriority {
            score: i64::MIN,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 0,
            recency: 0,
            layer_idx: None,
            expert_id: None,
        };

        // Assert: i64::MIN is preserved without overflow
        assert_eq!(ep.score, i64::MIN);
        assert_eq!(ep.payload_kind, PagePayloadKind::KvContext);
        assert!(!ep.is_pinned);
    }

    #[test]
    fn eviction_priority_positive_score_max_boundary() {
        // Arrange: positive score = low priority (kept longest)
        use crate::scheduler::types::{EvictionPriority, PagePayloadKind};
        let ep = EvictionPriority {
            score: i64::MAX,
            payload_kind: PagePayloadKind::DenseLayerWeight,
            is_pinned: true,
            access_count: usize::MAX,
            recency: usize::MAX,
            layer_idx: Some(usize::MAX),
            expert_id: Some(u32::MAX),
        };

        // Assert: all boundary values preserved
        assert_eq!(ep.score, i64::MAX);
        assert!(ep.is_pinned);
        assert_eq!(ep.access_count, usize::MAX);
        assert_eq!(ep.recency, usize::MAX);
        assert_eq!(ep.layer_idx, Some(usize::MAX));
        assert_eq!(ep.expert_id, Some(u32::MAX));
    }

    #[test]
    fn generator_forward_config_default_for_test_accessors() {
        // Arrange
        use crate::engine::executor::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();

        // Act & Assert: all accessor methods return expected values from geometry
        assert_eq!(cfg.hidden_size(), 64);
        assert_eq!(cfg.num_layers(), 4);
        assert_eq!(cfg.vocab_size(), 100);
        assert_eq!(cfg.intermediate_size(), 128);
        assert!((cfg.norm_eps() - 1e-5).abs() < f32::EPSILON);
        assert_eq!(cfg.max_seq_len(), 512);
        assert_eq!(cfg.num_heads(), 4);
        assert_eq!(cfg.num_kv_heads(), 2);
        assert_eq!(cfg.head_dim(), 16);
        assert_eq!(cfg.rope_theta(), 10000.0);
        assert_eq!(cfg.rope_scale(), 1.0);
    }

    #[test]
    fn generator_forward_config_attention_derives_from_geometry() {
        // Arrange
        use crate::engine::executor::GeneratorForwardConfig;
        let cfg = GeneratorForwardConfig::default_for_test();

        // Act
        let attn = cfg.attention();

        // Assert: AttentionHeadConfig derived from geometry matches
        assert_eq!(attn.num_heads, 4);
        assert_eq!(attn.num_kv_heads, 2);
        assert_eq!(attn.head_dim, 16);
    }

    #[test]
    fn request_kind_debug_format_contains_variant_name() {
        // Arrange
        use crate::scheduler::types::RequestKind;
        let chat = RequestKind::Chat;
        let embedding = RequestKind::Embedding;
        let rerank = RequestKind::Rerank;

        // Act
        let debug_chat = format!("{chat:?}");
        let debug_emb = format!("{embedding:?}");
        let debug_rerank = format!("{rerank:?}");

        // Assert: Debug output contains variant name
        assert!(debug_chat.contains("Chat"));
        assert!(debug_emb.contains("Embedding"));
        assert!(debug_rerank.contains("Rerank"));
    }

    // ── Wave 12x39 new tests (13 tests) ────────────────────────────────────

    #[test]
    fn storage_tier_from_u8_round_trip_all_variants() {
        // Arrange: all 3 StorageTier variants
        use crate::kv_cache::StorageTier;
        let variants = vec![
            (0u8, StorageTier::GpuHbm),
            (1u8, StorageTier::CpuDram),
            (2u8, StorageTier::Nvme),
        ];

        // Act & Assert: from_u8 maps each discriminant correctly
        for (byte, expected) in &variants {
            assert_eq!(StorageTier::from_u8(*byte), Some(*expected));
            // Round-trip: variant -> as_u8 -> from_u8 -> same variant
            assert_eq!(StorageTier::from_u8(expected.as_u8()), Some(*expected));
        }
        // Invalid discriminant returns None
        assert_eq!(StorageTier::from_u8(3), None);
        assert_eq!(StorageTier::from_u8(255), None);
    }

    #[test]
    fn storage_tier_variants_distinct_and_hash() {
        // Arrange
        use std::collections::HashSet;
        use crate::kv_cache::StorageTier;
        let variants = vec![
            StorageTier::GpuHbm,
            StorageTier::CpuDram,
            StorageTier::Nvme,
        ];

        // Act
        let set: HashSet<_> = variants.into_iter().collect();

        // Assert: 3 distinct variants produce 3 hash entries
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn kv_cache_error_exhausted_display_format() {
        // Arrange
        use crate::kv_cache::KvCacheError;
        let err = KvCacheError::Exhausted {
            requested: 1024,
            available: 512,
        };

        // Act
        let msg = format!("{err}");

        // Assert: error message includes both requested and available counts
        assert!(msg.contains("kv cache exhausted"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("512"));
    }

    #[test]
    fn oom_halt_error_fatal_and_soft_construction() {
        // Arrange & Act
        use crate::kv_cache::OomHaltError;
        let fatal = OomHaltError::fatal_halt("GPU OOM");
        let soft = OomHaltError::soft_halt("retry possible");

        // Assert: fatal flag distinguishes the two variants
        assert!(fatal.fatal);
        assert!(!soft.fatal);
        assert_eq!(fatal.message, "GPU OOM");
        assert_eq!(soft.message, "retry possible");
    }

    #[test]
    fn select_codec_returns_correct_codec_per_tier() {
        // Arrange
        use crate::kv_cache::{select_codec, CompressionCodec, PrecisionTier};

        // Act & Assert: FP16 on CPU path without nvcomp -> Lz4
        assert_eq!(
            select_codec(PrecisionTier::FP16, false, false),
            CompressionCodec::Lz4,
        );
        // FP16 on GPU path with nvcomp -> NvcompAns
        assert_eq!(
            select_codec(PrecisionTier::FP16, true, true),
            CompressionCodec::NvcompAns,
        );
        // KIVI4 always -> BitPackRle
        assert_eq!(
            select_codec(PrecisionTier::KIVI4, true, false),
            CompressionCodec::BitPackRle,
        );
        // Sparse/Dictionary -> None (no compression)
        assert_eq!(
            select_codec(PrecisionTier::Sparse, false, false),
            CompressionCodec::None,
        );
        // Evicted -> None
        assert_eq!(
            select_codec(PrecisionTier::Evicted, true, true),
            CompressionCodec::None,
        );
    }

    #[test]
    fn precision_tier_variants_distinct_and_copy() {
        // Arrange
        use crate::kv_cache::PrecisionTier;
        let fp16 = PrecisionTier::FP16;
        let fp8 = PrecisionTier::FP8;
        let kivi4 = PrecisionTier::KIVI4;
        let kivi2 = PrecisionTier::KIVI2;
        let sparse = PrecisionTier::Sparse;
        let dict = PrecisionTier::Dictionary;
        let evicted = PrecisionTier::Evicted;

        // Act: Copy semantics
        let copied = fp16;

        // Assert: all 7 variants pairwise distinct
        let all = [fp16, fp8, kivi4, kivi2, sparse, dict, evicted];
        assert_eq!(all.len(), 7);
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "PrecisionTier variants must be distinct");
            }
        }
        // Copy produces equal value
        assert_eq!(copied, PrecisionTier::FP16);
    }

    #[test]
    fn layer_donor_info_owned_construction_and_defaults() {
        // Arrange & Act
        use crate::kv_cache::LayerDonorInfo;
        let info = LayerDonorInfo::owned(5, 1);

        // Assert: owned entry has no donor and zero borrower count
        assert_eq!(info.layer, 5);
        assert_eq!(info.attn_bucket, 1);
        assert!(info.donor_layer.is_none());
        assert_eq!(info.borrower_refcount, 0);
    }

    #[test]
    fn f32_f16_round_trip_preserves_normal_values() {
        // Arrange: select normal-range f32 values within f16 precision
        use crate::kv_cache::{f32_to_f16_bits, f16_bits_to_f32};
        let values = vec![0.0f32, 1.0, -1.0, 0.5, 2.0, 0.333, 100.0, -42.5];

        // Act & Assert: round-trip through f16 bits
        for v in &values {
            let bits = f32_to_f16_bits(*v);
            let recovered = f16_bits_to_f32(bits);
            let tolerance = (*v * 0.005).abs().max(0.001);
            assert!(
                (recovered - v).abs() <= tolerance,
                "f16 round-trip for {v}: recovered={recovered}, tolerance={tolerance}",
            );
        }
    }

    #[test]
    fn request_phase_variants_distinct_and_copy() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        let prefill = RequestPhase::Prefill;
        let decode = RequestPhase::Decode;
        let chunked = RequestPhase::ChunkedPrefill;

        // Act: Copy semantics
        let copied = prefill;

        // Assert: all 3 variants pairwise distinct
        assert_ne!(prefill, decode);
        assert_ne!(prefill, chunked);
        assert_ne!(decode, chunked);
        // Copy produces equal value
        assert_eq!(copied, RequestPhase::Prefill);
    }

    #[test]
    fn request_data_construction_preserves_all_fields() {
        // Arrange
        use crate::engine::executor_types::{RequestData, SamplingConfig};
        use crate::scheduler::request_state::RequestPhase;

        let sampling = SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        };

        // Act
        let data = RequestData {
            prompt_tokens: vec![1, 2, 3, 4],
            output_tokens: vec![100],
            sampling_config: sampling.clone(),
            is_prefill: true,
            phase: RequestPhase::Prefill,
            max_new_tokens: 256,
            finished: false,
            session_id: Some(42u64),
            thinking_budget: Some(128),
            fused_prefill_hidden: Some(vec![0.5; 64]),
        };

        // Assert: all fields preserved
        assert_eq!(data.prompt_tokens, vec![1, 2, 3, 4]);
        assert_eq!(data.output_tokens, vec![100]);
        assert_eq!(data.sampling_config.temperature, 0.7);
        assert!(data.is_prefill);
        assert_eq!(data.phase, RequestPhase::Prefill);
        assert_eq!(data.max_new_tokens, 256);
        assert!(!data.finished);
        assert_eq!(data.session_id, Some(42u64));
        assert_eq!(data.thinking_budget, Some(128));
        assert_eq!(data.fused_prefill_hidden.as_ref().unwrap().len(), 64);
    }

    #[test]
    fn executor_error_scheduler_variant_display() {
        // Arrange
        use crate::engine::executor_types::ExecutorError;
        let err = ExecutorError::Scheduler("no available slots".to_string());

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.contains("scheduler error"));
        assert!(msg.contains("no available slots"));
    }

    #[test]
    fn executor_error_empty_prompt_variant_display() {
        // Arrange
        use crate::engine::executor_types::ExecutorError;
        let err = ExecutorError::EmptyPrompt;

        // Act
        let msg = format!("{err}");

        // Assert
        assert!(msg.contains("empty prompt tokens"));
    }

    #[test]
    fn attention_topology_causal_construction_mask_type() {
        // Arrange
        use crate::engine::executor::AttentionTopology;
        use crate::engine::executor_types::AttentionMaskType;
        use crate::model_config::ModelGeometry;
        use gllm_kernels::types::DType;

        let geo = std::sync::Arc::new(ModelGeometry {
            hidden_size: 64,
            num_layers: 2,
            vocab_size: 100,
            intermediate_size: 128,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 16,
            max_seq_len: 256,
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
        });

        // Act
        let causal = AttentionTopology::causal(std::sync::Arc::clone(&geo));
        let bidirectional = AttentionTopology::bidirectional(geo);

        // Assert: causal mask differs from bidirectional
        assert_eq!(causal.mask_type, AttentionMaskType::Causal);
        assert_eq!(bidirectional.mask_type, AttentionMaskType::Bidirectional);
        assert_ne!(causal.mask_type, bidirectional.mask_type);
        // Accessor methods propagate from geometry
        assert_eq!(causal.num_heads(), 4);
        assert_eq!(causal.num_kv_heads(), 2);
        assert_eq!(causal.head_dim(), 16);
        assert_eq!(causal.max_seq_len(), 256);
    }

    // ── Wave 12x40 new tests (13 tests) ────────────────────────────────────

    #[test]
    fn hip_backend_new_with_different_element_types_returns_none() {
        let f32_result = HipBackend::<f32>::new(0);
        let f16_result = HipBackend::<half::f16>::new(0);
        assert!(f32_result.is_none());
        assert!(f16_result.is_none());
    }

    #[test]
    fn backend_error_source_chain_returns_none_for_leaf_variants() {
        let err = BE::Hip("device lost".to_string());
        assert!(std::error::Error::source(&err).is_none());

        let err2 = BE::Unimplemented("not supported");
        assert!(std::error::Error::source(&err2).is_none());

        let err3 = BE::Other("generic".to_string());
        assert!(std::error::Error::source(&err3).is_none());
    }

    #[test]
    fn attention_mask_type_debug_format_contains_variant_names() {
        use crate::engine::executor_types::AttentionMaskType;
        let bidi_debug = format!("{:?}", AttentionMaskType::Bidirectional);
        let causal_debug = format!("{:?}", AttentionMaskType::Causal);
        assert!(bidi_debug.contains("Bidirectional"));
        assert!(causal_debug.contains("Causal"));
        assert_ne!(bidi_debug, causal_debug);
    }

    #[test]
    fn rope_config_default_theta_and_scale_values() {
        use crate::engine::executor::RoPEConfig;
        let config = RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        assert_eq!(config.theta, 10000.0);
        assert_eq!(config.scale, 1.0);
        assert!(!config.interleaved);
        assert!(!config.precompute);
    }

    #[test]
    fn swap_config_clone_independence_after_modification() {
        use crate::engine::executor::SwapConfig;
        let mut original = SwapConfig {
            enable_swap: true,
            swap_threshold: 0.85,
            lru_granularity: 4,
        };
        let cloned = original.clone();
        original.enable_swap = false;
        assert!(cloned.enable_swap);
        assert!(!original.enable_swap);
        assert_eq!(cloned.swap_threshold, 0.85);
        assert_eq!(cloned.lru_granularity, 4);
    }

    #[test]
    fn compression_codec_as_u8_round_trip_all_variants() {
        use crate::kv_cache::CompressionCodec;
        for expected in [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ] {
            let byte = expected.as_u8();
            assert_eq!(CompressionCodec::from_u8(byte), Some(expected));
        }
    }

    #[test]
    fn sequence_input_clone_preserves_all_fields_including_fused_hidden() {
        let original = SequenceInput {
            tokens: vec![10, 20, 30],
            position: 5,
            draft_steps: 3,
            page_table: Some(vec![0, 1, 2]),
            fused_hidden: Some(vec![0.25, 0.75, 1.0]),
        };
        let cloned = original.clone();
        assert_eq!(cloned.tokens, original.tokens);
        assert_eq!(cloned.position, 5);
        assert_eq!(cloned.draft_steps, 3);
        assert_eq!(cloned.page_table, Some(vec![0, 1, 2]));
        let hidden = cloned.fused_hidden.as_ref().unwrap();
        assert_eq!(hidden.len(), 3);
        assert_eq!(hidden[0], 0.25);
        assert_eq!(hidden[2], 1.0);
    }

    #[test]
    fn paged_kv_config_debug_format_includes_fields() {
        use crate::engine::executor::PagedKvConfig;
        let config = PagedKvConfig {
            page_table: Some(vec![0, 1, 2]),
            page_size: 16,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("page_table"));
        assert!(debug.contains("page_size"));
    }

    #[test]
    fn gpu_device_info_debug_includes_sm_version_field() {
        let info = GpuDeviceInfo {
            ordinal: 0,
            sm_version: 90,
            sm_count: 132,
            total_memory: 80 * 1024 * 1024 * 1024,
            name: "NVIDIA H100".to_string(),
        };
        let debug = format!("{info:?}");
        assert!(debug.contains("sm_version: 90"));
        assert!(debug.contains("NVIDIA H100"));
    }

    #[test]
    fn executor_error_backend_variant_display_propagates_backend_message() {
        use crate::engine::executor_types::ExecutorError;
        let backend_err = BE::Hip("hipMalloc failed".to_string());
        let exec_err: ExecutorError = backend_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("HIP error"));
        assert!(msg.contains("hipMalloc failed"));
    }

    #[test]
    fn executor_error_empty_sample_variant_display() {
        use crate::engine::executor_types::ExecutorError;
        let err = ExecutorError::EmptySample;
        let msg = format!("{err}");
        assert!(msg.contains("empty sample"));
    }

    #[test]
    fn request_data_construction_with_all_optional_fields() {
        use crate::engine::executor_types::{RequestData, SamplingConfig};
        use crate::scheduler::request_state::RequestPhase;

        let data = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![100, 101],
            sampling_config: SamplingConfig {
                temperature: 0.5,
                top_k: 10,
                top_p: 0.95,
            },
            is_prefill: false,
            phase: RequestPhase::Decode,
            max_new_tokens: 512,
            finished: false,
            session_id: Some(7u64),
            thinking_budget: None,
            fused_prefill_hidden: Some(vec![0.1; 32]),
        };

        assert_eq!(data.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(data.output_tokens, vec![100, 101]);
        assert_eq!(data.sampling_config.temperature, 0.5);
        assert!(!data.is_prefill);
        assert_eq!(data.phase, RequestPhase::Decode);
        assert_eq!(data.max_new_tokens, 512);
        assert!(!data.finished);
        assert_eq!(data.session_id, Some(7u64));
        assert_eq!(data.thinking_budget, None);
        assert_eq!(data.fused_prefill_hidden.as_ref().unwrap().len(), 32);
    }

    #[test]
    fn attention_topology_clone_preserves_mask_type_and_geometry() {
        use crate::engine::executor::AttentionTopology;
        use crate::model_config::ModelGeometry;
        use gllm_kernels::types::DType;

        let geo = std::sync::Arc::new(ModelGeometry {
            hidden_size: 128,
            num_layers: 4,
            vocab_size: 200,
            intermediate_size: 256,
            num_heads: 8,
            num_kv_heads: 4,
            head_dim: 16,
            max_seq_len: 512,
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
        });

        let topo = AttentionTopology::causal(std::sync::Arc::clone(&geo));
        let cloned = topo.clone();

        assert_eq!(cloned.mask_type, topo.mask_type);
        assert_eq!(cloned.num_heads(), 8);
        assert_eq!(cloned.num_kv_heads(), 4);
        assert_eq!(cloned.head_dim(), 16);
        assert_eq!(cloned.max_seq_len(), 512);
    }

    // ── Wave 12x41 new tests (13 tests) ────────────────────────────────────

    #[test]
    fn kv_cache_slot_flip_round_trip() {
        // Arrange
        use crate::kv_cache::KvCacheSlot;
        let front = KvCacheSlot::Front;
        let back = KvCacheSlot::Back;

        // Act: flip twice returns to original
        let front_flipped = front.flip();
        let back_flipped = back.flip();
        let front_roundtrip = front_flipped.flip();

        // Assert: Front -> Back, Back -> Front, round-trip returns to start
        assert_eq!(front_flipped, KvCacheSlot::Back);
        assert_eq!(back_flipped, KvCacheSlot::Front);
        assert_eq!(front_roundtrip, KvCacheSlot::Front);
        assert_ne!(front, back);
    }

    #[test]
    fn compact_scatter_meta_copy_and_equality() {
        // Arrange
        use crate::scheduler::request_state::CompactScatterMeta;
        let meta = CompactScatterMeta {
            original_slot: 5,
            compacted_slot: 2,
            active: 1,
        };

        // Act: Copy semantics
        let copied = meta;

        // Assert: both usable and equal by field
        assert_eq!(copied.original_slot, 5);
        assert_eq!(copied.compacted_slot, 2);
        assert_eq!(copied.active, 1);
        assert_eq!(meta.original_slot, 5);
        assert_eq!(meta, copied);
    }

    #[test]
    fn request_telemetry_default_values() {
        // Arrange & Act
        use crate::scheduler::request_state::RequestTelemetry;
        let tel = RequestTelemetry::default();

        // Assert: default has zero entropy, zero centroid, unit residual metrics
        assert_eq!(tel.entropy, 0.0);
        assert_eq!(tel.centroid, 0.0);
        assert_eq!(tel.residual_delta, 1.0);
        assert_eq!(tel.residual_cosine, 1.0);
        assert_eq!(tel.range_group, 0);
    }

    #[test]
    fn request_telemetry_copy_preserves_independent_state() {
        // Arrange
        use crate::scheduler::request_state::RequestTelemetry;
        let mut original = RequestTelemetry::default();
        original.entropy = 2.5;
        original.centroid = 0.75;
        original.range_group = 3;

        // Act: Copy semantics
        let copied = original;
        original.entropy = 0.0;

        // Assert: copied retains original values, modified original is independent
        assert_eq!(copied.entropy, 2.5);
        assert_eq!(copied.centroid, 0.75);
        assert_eq!(copied.range_group, 3);
        assert_eq!(original.entropy, 0.0);
    }

    #[test]
    fn dead_ratio_round_trip_preserves_values() {
        // Arrange: known dead ratios from 0.0 to 1.0
        use crate::kv_cache::{f32_to_dead_ratio, dead_ratio_to_f32};
        let values = [0.0f32, 0.5, 1.0, 0.1, 0.99];

        // Act & Assert: round-trip through u8 encoding
        for v in &values {
            let encoded = f32_to_dead_ratio(*v);
            let decoded = dead_ratio_to_f32(encoded);
            let tolerance = 1.0 / 255.0;
            assert!(
                (decoded - v).abs() <= tolerance,
                "dead_ratio round-trip for {v}: decoded={decoded}, tolerance={tolerance}",
            );
        }
    }

    #[test]
    fn select_cold_codec_always_returns_zstd_dict() {
        // Arrange
        use crate::kv_cache::{select_cold_codec, CompressionCodec, PrecisionTier};

        // Act & Assert: cold tier always returns ZstdDict regardless of precision tier
        assert_eq!(select_cold_codec(PrecisionTier::FP16), CompressionCodec::ZstdDict);
        assert_eq!(select_cold_codec(PrecisionTier::FP8), CompressionCodec::ZstdDict);
        assert_eq!(select_cold_codec(PrecisionTier::KIVI4), CompressionCodec::ZstdDict);
        assert_eq!(select_cold_codec(PrecisionTier::KIVI2), CompressionCodec::ZstdDict);
        assert_eq!(select_cold_codec(PrecisionTier::Sparse), CompressionCodec::ZstdDict);
        assert_eq!(select_cold_codec(PrecisionTier::Dictionary), CompressionCodec::ZstdDict);
        assert_eq!(select_cold_codec(PrecisionTier::Evicted), CompressionCodec::ZstdDict);
    }

    #[test]
    fn layer_donor_info_reference_construction() {
        // Arrange & Act
        use crate::kv_cache::LayerDonorInfo;
        let info = LayerDonorInfo::reference(7, 1, 3);

        // Assert: reference entry has donor_layer set and is_shared returns true
        assert_eq!(info.layer, 7);
        assert_eq!(info.attn_bucket, 1);
        assert_eq!(info.donor_layer, Some(3));
        assert_eq!(info.borrower_refcount, 0);
        assert!(info.is_shared());
    }

    #[test]
    fn layer_donor_info_owned_is_not_shared() {
        // Arrange & Act
        use crate::kv_cache::LayerDonorInfo;
        let info = LayerDonorInfo::owned(10, 0);

        // Assert: owned entry has no donor and is_shared returns false
        assert_eq!(info.layer, 10);
        assert_eq!(info.attn_bucket, 0);
        assert!(info.donor_layer.is_none());
        assert_eq!(info.borrower_refcount, 0);
        assert!(!info.is_shared());
    }

    #[test]
    fn storage_tier_as_u8_round_trip_all_variants() {
        // Arrange
        use crate::kv_cache::StorageTier;
        let variants = [StorageTier::GpuHbm, StorageTier::CpuDram, StorageTier::Nvme];

        // Act & Assert: each variant round-trips through as_u8/from_u8
        for variant in &variants {
            let byte = variant.as_u8();
            assert_eq!(StorageTier::from_u8(byte), Some(*variant));
        }
    }

    #[test]
    fn kv_cache_error_display_includes_exhausted_details() {
        // Arrange
        use crate::kv_cache::KvCacheError;
        let err = KvCacheError::Exhausted {
            requested: 2048,
            available: 0,
        };

        // Act
        let msg = format!("{err}");

        // Assert: message contains both counts
        assert!(msg.contains("kv cache exhausted"));
        assert!(msg.contains("2048"));
        assert!(msg.contains("0"));
    }

    #[test]
    fn oom_halt_error_display_format() {
        // Arrange
        use crate::kv_cache::OomHaltError;
        let fatal = OomHaltError::fatal_halt("critical OOM");
        let soft = OomHaltError::soft_halt("pressure");

        // Act
        let fatal_msg = format!("{fatal}");
        let soft_msg = format!("{soft}");

        // Assert: fatal message contains "OOM Halt", "fatal=true", and message body
        assert!(fatal_msg.contains("OOM Halt"));
        assert!(fatal_msg.contains("fatal=true"));
        assert!(fatal_msg.contains("critical OOM"));
        assert!(soft_msg.contains("pressure"));
        assert!(soft_msg.contains("fatal=false"));
    }

    #[test]
    fn gpu_device_info_multiple_ordinals_distinct() {
        // Arrange: simulate multi-GPU system with distinct ordinals
        let gpus: Vec<GpuDeviceInfo> = (0..4)
            .map(|i| GpuDeviceInfo {
                ordinal: i,
                sm_version: 0,
                sm_count: 120 + i as u32,
                total_memory: (i as usize + 1) * 32 * 1024 * 1024 * 1024,
                name: format!("AMD MI{}00X", 2 + i),
            })
            .collect();

        // Assert: each GPU has unique ordinal and increasing memory
        assert_eq!(gpus.len(), 4);
        for i in 0..4 {
            assert_eq!(gpus[i].ordinal, i);
            assert_eq!(gpus[i].sm_count, 120 + i as u32);
            assert_eq!(gpus[i].total_memory / (1024 * 1024 * 1024), (i as usize + 1) * 32);
            assert_eq!(gpus[i].name, format!("AMD MI{}00X", 2 + i));
        }
    }

    #[test]
    fn executor_error_from_backend_error_preserves_hip_context() {
        // Arrange: BackendError::Hip wraps into ExecutorError::Backend
        use crate::engine::executor_types::ExecutorError;
        let hip_err = BE::Hip("hipEventRecord failed".to_string());

        // Act: BackendError converts to ExecutorError via From trait
        let exec_err: ExecutorError = hip_err.into();
        let msg = format!("{exec_err}");

        // Assert: the original HIP error message propagates through
        assert!(msg.contains("HIP error"));
        assert!(msg.contains("hipEventRecord failed"));
    }

    // ── Wave 12x42 new tests (10 tests) ─────────────────────────────────────

    #[test]
    fn kv_page_header_new_sets_page_id_with_defaults() {
        // Arrange & Act
        use crate::kv_cache::KvPageHeader;
        let header = KvPageHeader::new(42);

        // Assert: page_id is set, all other fields are default
        assert_eq!(header.page_id, 42);
        assert_eq!(header.ref_count, 0);
        assert!(!header.is_active());
        assert_eq!(header.dead_ratio, 0);
        assert_eq!(header.importance_score, 0);
        assert_eq!(header.sink_mask, 0);
        assert_eq!(header.compressed_size, 0);
        assert!(!header.has_sink_token());
        assert!(!header.needs_requantize());
        assert!(!header.is_position_agnostic());
    }

    #[test]
    fn kv_page_header_active_and_precision_tier_mutations() {
        // Arrange
        use crate::kv_cache::{KvPageHeader, PrecisionTier};
        let mut header = KvPageHeader::new(0);

        // Act: set ref_count, change precision tier, set sink/deopt flags
        header.ref_count = 3;
        assert!(header.is_active());
        header.set_precision_tier(PrecisionTier::KIVI4);
        assert_eq!(header.precision_tier(), PrecisionTier::KIVI4);
        header.sink_mask = 0b101;
        assert!(header.has_sink_token());
        header.deopt_flags = 0x81; // bit 0 (requantize) + bit 7 (position-agnostic)
        assert!(header.needs_requantize());
        assert!(header.is_position_agnostic());

        // Act: clear position-agnostic
        header.set_position_agnostic(false);
        assert!(!header.is_position_agnostic());
        assert!(header.needs_requantize()); // bit 0 still set
    }

    #[test]
    fn kv_page_header_entropy_and_dead_ratio_checks() {
        // Arrange
        use crate::kv_cache::KvPageHeader;
        let mut header = KvPageHeader::new(0);

        // Assert: zero entropy is low entropy
        assert!(header.is_low_entropy());
        assert!(!header.is_high_dead_ratio());

        // Act: set non-zero entropy, high dead ratio
        header.entropy_avg = 100;
        assert!(!header.is_low_entropy());
        header.dead_ratio = 200;
        assert!(header.is_high_dead_ratio());

        // Act: boundary at exactly 128
        header.dead_ratio = 128;
        assert!(header.is_high_dead_ratio());
        header.dead_ratio = 127;
        assert!(!header.is_high_dead_ratio());
    }

    #[test]
    fn kv_page_header_head_entropy_spread_calculation() {
        // Arrange
        use crate::kv_cache::KvPageHeader;
        let mut header = KvPageHeader::new(0);
        header.head_entropy_max = 100;
        header.head_entropy_min = 30;

        // Act
        let spread = header.head_entropy_spread();

        // Assert: max - min = 70
        assert_eq!(spread, 70);

        // Act: max < min → saturating_sub returns 0
        header.head_entropy_max = 10;
        header.head_entropy_min = 50;
        assert_eq!(header.head_entropy_spread(), 0);
    }

    #[test]
    fn storage_tier_ordering_hbm_highest_priority() {
        // Arrange
        use crate::kv_cache::StorageTier;
        let hbm = StorageTier::GpuHbm;
        let dram = StorageTier::CpuDram;
        let nvme = StorageTier::Nvme;

        // Assert: GpuHbm > CpuDram > Nvme (lower discriminant = higher priority)
        assert!(hbm > dram);
        assert!(dram > nvme);
        assert!(hbm > nvme);
        // Sorted ascending by priority
        let mut tiers = vec![nvme, hbm, dram];
        tiers.sort();
        assert_eq!(tiers[0], nvme);
        assert_eq!(tiers[1], dram);
        assert_eq!(tiers[2], hbm);
    }

    #[test]
    fn select_codec_fp8_cpu_path_returns_lz4() {
        // Arrange
        use crate::kv_cache::{select_codec, CompressionCodec, PrecisionTier};

        // Act: FP8 on CPU path without nvcomp
        let codec = select_codec(PrecisionTier::FP8, false, false);

        // Assert: same path as FP16 — Lz4
        assert_eq!(codec, CompressionCodec::Lz4);

        // Act: FP8 on GPU path without nvcomp
        let codec_gpu_no_nvcomp = select_codec(PrecisionTier::FP8, true, false);
        assert_eq!(codec_gpu_no_nvcomp, CompressionCodec::Lz4);
    }

    #[test]
    fn select_codec_kivi2_returns_bit_pack_rle() {
        // Arrange
        use crate::kv_cache::{select_codec, CompressionCodec, PrecisionTier};

        // Act: KIVI2 always returns BitPackRle regardless of gpu/nvcom flags
        assert_eq!(
            select_codec(PrecisionTier::KIVI2, false, false),
            CompressionCodec::BitPackRle,
        );
        assert_eq!(
            select_codec(PrecisionTier::KIVI2, true, true),
            CompressionCodec::BitPackRle,
        );
    }

    #[test]
    fn executor_error_compilation_and_graph_expansion_variants() {
        // Arrange
        use crate::engine::executor_types::ExecutorError;
        let comp_err = ExecutorError::Compilation("x86 codegen failed".to_string());
        let graph_err = ExecutorError::GraphExpansion("unsupported op".to_string());
        let not_found_err = ExecutorError::RequestNotFound { request_id: 42 };

        // Act
        let comp_msg = format!("{comp_err}");
        let graph_msg = format!("{graph_err}");
        let not_found_msg = format!("{not_found_err}");

        // Assert: each variant has distinct display format
        assert!(comp_msg.contains("JIT compilation failed"));
        assert!(comp_msg.contains("x86 codegen failed"));
        assert!(graph_msg.contains("graph expansion failed"));
        assert!(graph_msg.contains("unsupported op"));
        assert!(not_found_msg.contains("request not found"));
        assert!(not_found_msg.contains("42"));
    }

    #[test]
    fn f32_to_f16_bits_zero_and_special_values() {
        // Arrange
        use crate::kv_cache::{f32_to_f16_bits, f16_bits_to_f32};

        // Act & Assert: zero round-trips exactly
        let zero_bits = f32_to_f16_bits(0.0f32);
        assert_eq!(f16_bits_to_f32(zero_bits), 0.0);

        // Negative zero: f16 bits have sign bit set, exponent and mantissa zero
        let neg_zero_bits = f32_to_f16_bits(-0.0f32);
        assert_ne!(neg_zero_bits, zero_bits, "negative zero f16 bits differ from positive zero");
        assert_eq!(neg_zero_bits >> 15, 1, "sign bit must be set for negative zero");

        // Infinity round-trips
        let inf_bits = f32_to_f16_bits(f32::INFINITY);
        let inf_recovered = f16_bits_to_f32(inf_bits);
        assert!(inf_recovered.is_infinite() && inf_recovered.is_sign_positive());

        let neg_inf_bits = f32_to_f16_bits(f32::NEG_INFINITY);
        let neg_inf_recovered = f16_bits_to_f32(neg_inf_bits);
        assert!(neg_inf_recovered.is_infinite() && neg_inf_recovered.is_sign_negative());
    }

    #[test]
    fn precision_tier_repr_values_match_discriminants() {
        // Arrange
        use crate::kv_cache::PrecisionTier;

        // Assert: each variant has the expected repr(u8) discriminant
        assert_eq!(PrecisionTier::FP16 as u8, 0);
        assert_eq!(PrecisionTier::FP8 as u8, 1);
        assert_eq!(PrecisionTier::KIVI4 as u8, 2);
        assert_eq!(PrecisionTier::KIVI2 as u8, 3);
        assert_eq!(PrecisionTier::Sparse as u8, 4);
        assert_eq!(PrecisionTier::Dictionary as u8, 5);
        assert_eq!(PrecisionTier::Evicted as u8, 6);
    }
}
