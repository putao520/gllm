//! Generic GPU helper functions that eliminate CUDA/HIP/Metal triplication.
//!
//! All functions are parameterized over `B: Backend<E>` so a single implementation
//! serves all three GPU backends. The logic is byte-for-byte identical to the
//! original per-backend versions — only the type parameter differs.

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use super::backend_trait;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::engine::executor::BackendError as BE;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::engine::executor::KvCacheConfig;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::engine::executor::KvCacheHandle;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::scheduler::types::{PageId, PageState, StorageKey};

// ---------------------------------------------------------------------------
// GpuBackendOps — trait abstracting device/error/store differences
// ---------------------------------------------------------------------------

/// Trait that abstracts the 3 backend-specific differences:
/// 1. Device access (for alloc/htod/dtoh)
/// 2. Error variant construction (BE::Cuda/Hip/Metal)
/// 3. Swap store and KV metadata store access
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) trait GpuBackendOps {
    /// Raw device→host copy: src_ptr is a device pointer, dst is host slice.
    fn raw_dtoh(&self, src_ptr: u64, dst: &mut [u8]) -> Result<(), BE>;
    /// Raw host→device copy: src is host slice, dst_ptr is a device pointer.
    fn raw_htod(&self, src: &[u8], dst_ptr: u64) -> Result<(), BE>;
    /// Allocate device memory, return device pointer.
    fn raw_alloc(&self, bytes: usize) -> Result<u64, BE>;
    fn gpu_error(&self, msg: String) -> BE;
    fn swap_store(&self) -> &super::gpu_compile::GpuSwapStore;
    fn kv_meta(&self) -> &super::gpu_compile::GpuKvMetaStore;
}

#[cfg(feature = "cuda")]
impl GpuBackendOps for super::cuda_backend::CudaBackend<f32> {
    fn raw_dtoh(&self, src_ptr: u64, dst: &mut [u8]) -> Result<(), BE> {
        let res = unsafe {
            (self.device.driver().cuMemcpyDtoH_v2)(
                dst.as_mut_ptr() as *mut _,
                src_ptr,
                dst.len(),
            )
        };
        if res != 0 { Err(BE::Cuda(format!("cuMemcpyDtoH failed: {res}"))) } else { Ok(()) }
    }
    fn raw_htod(&self, src: &[u8], dst_ptr: u64) -> Result<(), BE> {
        let res = unsafe {
            (self.device.driver().cuMemcpyHtoD_v2)(
                dst_ptr,
                src.as_ptr() as *const _,
                src.len(),
            )
        };
        if res != 0 { Err(BE::Cuda(format!("cuMemcpyHtoD failed: {res}"))) } else { Ok(()) }
    }
    fn raw_alloc(&self, bytes: usize) -> Result<u64, BE> {
        use gllm_kernels::gpu::GpuDevice;
        use gllm_kernels::gpu::GpuBuffer;
        let buf = self.device.alloc(bytes).map_err(|e| BE::Cuda(format!("alloc failed: {e}")))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }
    fn gpu_error(&self, msg: String) -> BE { BE::Cuda(msg) }
    fn swap_store(&self) -> &super::gpu_compile::GpuSwapStore { &self.swap_store }
    fn kv_meta(&self) -> &super::gpu_compile::GpuKvMetaStore { &self.kv_meta }
}

#[cfg(feature = "hip")]
impl GpuBackendOps for super::hip_backend::HipBackend<f32> {
    fn raw_dtoh(&self, src_ptr: u64, dst: &mut [u8]) -> Result<(), BE> {
        self.device.dtoh_raw(src_ptr, dst)
            .map_err(|e| BE::Hip(format!("dtoh_raw failed: {e}")))
    }
    fn raw_htod(&self, src: &[u8], dst_ptr: u64) -> Result<(), BE> {
        self.device.htod_raw(src, dst_ptr)
            .map_err(|e| BE::Hip(format!("htod_raw failed: {e}")))
    }
    fn raw_alloc(&self, bytes: usize) -> Result<u64, BE> {
        use gllm_kernels::gpu::{GpuBuffer, GpuDevice};
        let buf = self.device.alloc(bytes).map_err(|e| BE::Hip(format!("alloc failed: {e}")))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }
    fn gpu_error(&self, msg: String) -> BE { BE::Hip(msg) }
    fn swap_store(&self) -> &super::gpu_compile::GpuSwapStore { &self.swap_store }
    fn kv_meta(&self) -> &super::gpu_compile::GpuKvMetaStore { &self.kv_meta }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl GpuBackendOps for super::metal_backend::MetalBackend<f32> {
    fn raw_dtoh(&self, src_ptr: u64, dst: &mut [u8]) -> Result<(), BE> {
        self.device().dtoh_raw(src_ptr, dst)
            .map_err(|e| BE::Metal(format!("dtoh_raw failed: {e}")))
    }
    fn raw_htod(&self, src: &[u8], dst_ptr: u64) -> Result<(), BE> {
        self.device().htod_raw(src, dst_ptr)
            .map_err(|e| BE::Metal(format!("htod_raw failed: {e}")))
    }
    fn raw_alloc(&self, bytes: usize) -> Result<u64, BE> {
        use gllm_kernels::gpu::GpuBuffer;
        let buf = self.device().alloc(bytes).map_err(|e| BE::Metal(format!("alloc failed: {e}")))?;
        let ptr = buf.as_device_ptr();
        std::mem::forget(buf);
        Ok(ptr)
    }
    fn gpu_error(&self, msg: String) -> BE { BE::Metal(msg) }
    fn swap_store(&self) -> &super::gpu_compile::GpuSwapStore { &self.swap_store }
    fn kv_meta(&self) -> &super::gpu_compile::GpuKvMetaStore { &self.kv_meta }
}

// ---------------------------------------------------------------------------
// Generic GPU ops using GpuBackendOps trait
// ---------------------------------------------------------------------------

/// GPU alloc_kv_cache: allocate GPU buffer for KV cache and register metadata.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_alloc_kv_cache(
    backend: &dyn GpuBackendOps,
    config: &KvCacheConfig,
) -> Result<KvCacheHandle, BE> {
    use gllm_kernels::gpu::GpuBuffer;

    let total_bytes = config.num_layers() * 2
        * config.num_heads() * config.max_seq_len() * config.head_dim() * config.dtype_size();

    let ptr = backend.raw_alloc(total_bytes)
        .map_err(|e| backend.gpu_error(format!("KV cache alloc failed ({} bytes): {e}", total_bytes)))?;

    let meta = super::gpu_compile::GpuKvCacheMeta::from_config(config, ptr);
    backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?
        .insert(ptr, meta);

    Ok(KvCacheHandle(ptr))
}

/// GPU sample_from_tensor: download logits to CPU and sample.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_sample_from_tensor(
    logits: &crate::engine::executor::LogitsHandle,
    _topology: &crate::engine::executor::AttentionTopology,
    vocab_size: usize,
    sampling: &crate::engine::executor::SamplingConfig,
) -> Result<Vec<u32>, BE> {
    super::gpu_compile::sample_logits_cpu(&logits.data, vocab_size, sampling)
}

/// GPU swap_out_pages: download page data from GPU to host swap store.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_swap_out_pages(
    backend: &dyn GpuBackendOps,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    let meta_store = backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| backend.gpu_error(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size();
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size();
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size();
    let total_page_bytes = meta.num_layers * meta.num_kv_heads * page_slice_bytes * 2;

    let mut swap_store = backend.swap_store().lock()
        .map_err(|e| backend.gpu_error(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(backend.gpu_error(format!(
                "swap_out: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size();
        let mut host_buf = vec![0u8; total_page_bytes];
        let mut dst_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let src_offset = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size()) as u64;
                    let src_ptr = half_base + src_offset;

                    backend.raw_dtoh(src_ptr, &mut host_buf[dst_offset..dst_offset + actual_slice_bytes])
                        .map_err(|e| backend.gpu_error(format!("swap_out dtoh failed: {e}")))?;

                    dst_offset += page_slice_bytes;
                }
            }
        }

        swap_store.insert(storage_key, host_buf);
    }

    Ok(())
}

/// GPU swap_in_pages: upload page data from host swap store to GPU.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_swap_in_pages(
    backend: &dyn GpuBackendOps,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    let meta_store = backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| backend.gpu_error(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size();
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size();
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size();

    let mut swap_store = backend.swap_store().lock()
        .map_err(|e| backend.gpu_error(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let host_buf = swap_store.remove(&storage_key)
            .ok_or_else(|| backend.gpu_error(format!("swap_in: storage key {} not found in swap store", storage_key)))?;

        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(backend.gpu_error(format!(
                "swap_in: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size();
        let mut src_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let dst_offset_gpu = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size()) as u64;
                    let dst_ptr = half_base + dst_offset_gpu;

                    backend.raw_htod(&host_buf[src_offset..src_offset + actual_slice_bytes], dst_ptr)
                        .map_err(|e| backend.gpu_error(format!("swap_in htod failed: {e}")))?;

                    src_offset += page_slice_bytes;
                }
            }
        }
    }

    Ok(())
}

/// GPU get_page_states: return page states based on metadata.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn gpu_get_page_states(
    backend: &dyn GpuBackendOps,
    handle: &KvCacheHandle,
) -> Result<Vec<(PageId, PageState)>, BE> {
    let meta_store = backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| backend.gpu_error(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let active = meta.active_pages();
    let total = meta.total_pages();
    let mut states = Vec::with_capacity(total);
    for page_id in 0..total {
        let state = if page_id < active { PageState::Active } else { PageState::Free };
        states.push((page_id, state));
    }
    Ok(states)
}

// ---------------------------------------------------------------------------
// GPU Mega-Kernel Launch Helpers
// ---------------------------------------------------------------------------

/// Build the 22-parameter ABI array for mega-kernel launch.
///
/// All pointer arguments must already be device pointers.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn build_mega_kernel_args(
    input_ids_gpu: u64,
    weight_blob_gpu: u64,
    kv_cache_gpu: u64,
    positions_gpu: u64,
    seq_lens_ptr: u64,
    batch_size: usize,
    seq_len: usize,
    scratchpad_gpu: u64,
    output_buf_gpu: u64,
    temperature_bits: usize,
    top_k: usize,
    top_p_bits: usize,
    max_new_tokens: usize,
    eos_token_id: usize,
    output_mode_selector: usize,
    hook_ctx_ptr: u64,
    telemetry_ptr: u64,
    session_position: usize,
    fused_hidden_ptr: u64,
    num_mm_tokens: usize,
    callback_table_ptr: u64,
    page_table_ptr_gpu: u64,
) -> [usize; 22] {
    [
        input_ids_gpu as usize,
        weight_blob_gpu as usize,
        kv_cache_gpu as usize,
        positions_gpu as usize,
        seq_lens_ptr as usize,
        batch_size,
        seq_len,
        scratchpad_gpu as usize,
        output_buf_gpu as usize,
        temperature_bits,
        top_k,
        top_p_bits,
        max_new_tokens,
        eos_token_id,
        output_mode_selector,
        hook_ctx_ptr as usize,
        telemetry_ptr as usize,
        session_position,
        fused_hidden_ptr as usize,
        num_mm_tokens,
        callback_table_ptr as usize,
        page_table_ptr_gpu as usize,
    ]
}

/// Extract f32 array from raw bytes (for DtoH results).
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) fn bytes_to_f32_vec(data: &[u8]) -> Vec<f32> {
    let count = data.len() / 4;
    let mut result = Vec::with_capacity(count);
    for i in 0..count {
        let bytes = [data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]];
        result.push(f32::from_le_bytes(bytes));
    }
    result
}
