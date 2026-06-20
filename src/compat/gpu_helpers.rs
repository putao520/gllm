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
    // Use GpuKvCacheMeta::from_config to compute the correct total_bytes
    // (handles MLA vs standard layout internally).
    let dummy_ptr = 0u64;
    let meta = super::gpu_compile::GpuKvCacheMeta::from_config(config, dummy_ptr);
    let total_bytes = meta.total_bytes;

    let ptr = backend.raw_alloc(total_bytes)
        .map_err(|e| backend.gpu_error(format!("KV cache alloc failed ({} bytes): {e}", total_bytes)))?;

    // Recreate meta with the actual device pointer.
    let meta = super::gpu_compile::GpuKvCacheMeta::from_config(config, ptr);
    backend.kv_meta().lock()
        .map_err(|e| backend.gpu_error(format!("kv_meta lock poisoned: {e}")))?
        .insert(ptr, meta);

    Ok(KvCacheHandle(ptr))
}

/// GPU sample_from_tensor: download logits to CPU and sample.
/// Note(GPU-RESIDENT-SAMPLING): mega-kernel 路径已内嵌采样管线（Argmax/Softmax/TopK/TopP/Multinomial），
/// 此函数仅用于 non-mega-kernel 的 batched step 路径。未来应替换为独立 GPU 采样 kernel。
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
        let total_page_bytes = meta.page_bytes();

        if meta.layout == super::KvLayoutStrategy::MlaCompressed {
            // MLA: [layers, seq, d_c+d_rope] — single contiguous copy per layer
            let layer_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size();
            let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size();
            let mut host_buf = vec![0u8; total_page_bytes];
            let mut dst_offset = 0usize;
            for layer in 0..meta.num_layers {
                let src_offset = (layer * layer_stride + token_start * meta.head_dim * meta.dtype_size()) as u64;
                backend.raw_dtoh(handle.0 + src_offset, &mut host_buf[dst_offset..dst_offset + actual_slice_bytes])
                    .map_err(|e| backend.gpu_error(format!("swap_out dtoh failed: {e}")))?;
                dst_offset += meta.page_size * meta.head_dim * meta.dtype_size();
            }
            swap_store.insert(storage_key, host_buf);
        } else {
            // Standard: [K_all_layers | V_all_layers], each: [layer][head][seq][dim]
            let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size();
            let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size();
            let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size();
            let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size();
            let mut host_buf = vec![0u8; total_page_bytes];
            let mut dst_offset = 0usize;

            for kv_half in 0..2usize {
                let half_base = handle.0 + (kv_half * half_bytes) as u64;
                for layer in 0..meta.num_layers {
                    for head in 0..meta.num_kv_heads {
                        let src_offset = ((layer * meta.num_kv_heads + head) * head_stride
                            + token_start * meta.head_dim * meta.dtype_size()) as u64;
                        backend.raw_dtoh(half_base + src_offset, &mut host_buf[dst_offset..dst_offset + actual_slice_bytes])
                            .map_err(|e| backend.gpu_error(format!("swap_out dtoh failed: {e}")))?;
                        dst_offset += page_slice_bytes;
                    }
                }
            }
            swap_store.insert(storage_key, host_buf);
        }
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

        if meta.layout == super::KvLayoutStrategy::MlaCompressed {
            // MLA: [layers, seq, d_c+d_rope]
            let layer_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size();
            let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size();
            let mut src_offset = 0usize;
            for layer in 0..meta.num_layers {
                let dst_offset_gpu = (layer * layer_stride + token_start * meta.head_dim * meta.dtype_size()) as u64;
                backend.raw_htod(&host_buf[src_offset..src_offset + actual_slice_bytes], handle.0 + dst_offset_gpu)
                    .map_err(|e| backend.gpu_error(format!("swap_in htod failed: {e}")))?;
                src_offset += meta.page_size * meta.head_dim * meta.dtype_size();
            }
        } else {
            // Standard: [K_all_layers | V_all_layers]
            let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size();
            let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size();
            let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size();
            let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size();
            let mut src_offset = 0usize;

            for kv_half in 0..2usize {
                let half_base = handle.0 + (kv_half * half_bytes) as u64;
                for layer in 0..meta.num_layers {
                    for head in 0..meta.num_kv_heads {
                        let dst_offset_gpu = ((layer * meta.num_kv_heads + head) * head_stride
                            + token_start * meta.head_dim * meta.dtype_size()) as u64;
                        backend.raw_htod(&host_buf[src_offset..src_offset + actual_slice_bytes], half_base + dst_offset_gpu)
                            .map_err(|e| backend.gpu_error(format!("swap_in htod failed: {e}")))?;
                        src_offset += page_slice_bytes;
                    }
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

/// Build a KernelContext for mega-kernel launch via single-pointer ABI (R2).
///
/// All pointer arguments must already be device pointers.
/// Returns a populated KernelContext that can be passed as `ctx: *const u8`.
/// Uses the unified `KernelContext::build()` shared with CPU path.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn build_kernel_context(
    weight_blob_gpu: u64,
    kv_cache_gpu: u64,
    output_buf_gpu: u64,
    hook_ctx_ptr: u64,
    kv_page_table_ptr: u64,
    kv_page_size: u32,
    kv_num_layers: u32,
    kv_num_heads: u32,
    kv_head_dim: u32,
    telemetry_ptr: u64,
    telemetry_flags: u32,
    callback_table_ptr: u64,
    scratch_buffer_gpu: u64,
    batch_ctx_ptr: u64,
) -> (super::super::engine::mega_kernel::KernelContext, Box<usize>) {
    super::super::engine::mega_kernel::KernelContext::build(
        weight_blob_gpu as *const u8,
        kv_cache_gpu as *mut u8,
        output_buf_gpu as *mut u8,
        hook_ctx_ptr as *mut u8,
        0, // seq_len — set per-invocation by caller
        std::ptr::null(), // rope_freqs_ptr — not used on GPU path
        kv_page_table_ptr as *const u32,
        std::ptr::null(), // batch_meta_ptr — not used on GPU path
        kv_page_size,
        kv_num_layers,
        kv_num_heads,
        kv_head_dim,
        telemetry_ptr as *mut u8,
        telemetry_flags,
        std::ptr::null(), // business_config_ptr — not used on GPU path
        std::ptr::null(), // weight_offsets_ptr — not used on GPU path
        0,                // weight_offsets_len
        callback_table_ptr as *const u64,
        scratch_buffer_gpu as *mut u8,
        batch_ctx_ptr as *const u8,
        std::ptr::null(), // weight_page_table_ptr — not used on GPU path
        std::ptr::null(), // weight_page_fault_cb_ptr — not used on GPU path
        0,                // weight_page_inject_flags — disabled
        std::ptr::null(), // kv_page_header_ptr — no KV decompress on GPU path
        0,                // decompress_inject_flags — disabled
    )
}

/// Build the 22-parameter ABI array for mega-kernel launch (legacy GPU path).
///
/// All pointer arguments must already be device pointers.
/// R1 transitional: retained for GPU backends that still use the 22-param kernel launch.
/// Internally delegates to GpuKernelLaunchConfig for unified parameter handling (REQ-KERNELS-GPU-001).
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
#[allow(clippy::too_many_arguments)]
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
    hook_ctx_ptr: u64,
    telemetry_ptr: u64,
    session_position: usize,
    fused_hidden_ptr: u64,
    num_mm_tokens: usize,
    callback_table_ptr: u64,
    page_table_ptr_gpu: u64,
    batch_ctx_ptr: u64,
) -> [usize; 22] {
    let config = super::gpu_compile::GpuKernelLaunchConfig {
        input_ids_gpu,
        weight_blob_gpu,
        kv_cache_gpu,
        positions_gpu,
        aux_ptr: seq_lens_ptr,
        batch_size,
        seq_len,
        scratchpad_gpu,
        output_buf_gpu,
        temperature_bits,
        top_k,
        top_p_bits,
        max_new_tokens,
        eos_token_id,
        hook_ctx_ptr,
        telemetry_ptr,
        session_position,
        fused_hidden_ptr,
        num_mm_tokens,
        callback_table_ptr,
        page_table_ptr: page_table_ptr_gpu,
        batch_ctx_ptr,
    };
    config.to_mega_kernel_args()
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

// ---------------------------------------------------------------------------
// GpuPagedKvPool — GPU device memory pool for paged KV cache (SPEC 18 REQ-PA-006)
// ---------------------------------------------------------------------------

/// Physical memory pool for paged KV cache on GPU device memory.
///
/// Mirrors CPU `PagedKvPool` but uses a device pointer instead of `Vec<u8>`.
/// Page allocation tracking uses a free-list bitmap managed on the host side.
///
/// Memory layout per page (standard MHA):
/// `page_stride = num_layers * 2 * num_kv_heads * page_size * head_dim * elem_bytes`
///
/// Memory layout per page (MLA):
/// `page_stride = num_layers * page_size * kv_dim * elem_bytes`
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub struct GpuPagedKvPool {
    /// Device pointer to the pool's base address.
    device_ptr: u64,
    /// Bytes per page.
    page_stride: usize,
    /// Total number of physical pages in the pool.
    num_pages: usize,
    /// Tokens per page.
    page_size: usize,
    /// Number of transformer layers.
    num_layers: usize,
    /// Number of KV heads (GQA). MLA: original value (unused for sizing).
    num_kv_heads: usize,
    /// Dimension per attention head. MLA: unused for sizing.
    head_dim: usize,
    /// Effective KV dimension per token per layer.
    kv_dim: usize,
    /// Bytes per element (4 for F32, 2 for F16/BF16).
    elem_bytes: usize,
    /// KV cache layout strategy (topology-derived, not bool).
    layout: super::KvLayoutStrategy,
    /// Free page bitmap: true = free, false = allocated.
    free_pages: Vec<bool>,
}

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl GpuPagedKvPool {
    /// Create a new GPU paged KV pool by allocating device memory.
    pub fn new<B: GpuBackendOps>(
        backend: &B,
        num_pages: usize,
        page_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        elem_bytes: usize,
        layout: super::KvLayoutStrategy,
    ) -> Result<Self, BE> {
        let page_stride = match layout {
            super::KvLayoutStrategy::MlaCompressed =>
                num_layers * page_size * kv_dim * elem_bytes,
            super::KvLayoutStrategy::Standard =>
                num_layers * 2 * num_kv_heads * page_size * head_dim * elem_bytes,
        };
        let total_bytes = num_pages * page_stride;
        let device_ptr = backend.raw_alloc(total_bytes).map_err(|e| {
            backend.gpu_error(format!("GpuPagedKvPool alloc failed ({total_bytes} bytes): {e}"))
        })?;
        Ok(Self {
            device_ptr,
            page_stride,
            num_pages,
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim,
            elem_bytes,
            layout,
            free_pages: vec![true; num_pages],
        })
    }

    /// Returns the device pointer to the pool's base address.
    pub fn device_ptr(&self) -> u64 {
        self.device_ptr
    }

    /// Returns the page stride in bytes.
    pub fn page_stride(&self) -> usize {
        self.page_stride
    }

    /// Returns the total number of physical pages.
    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    /// Returns the page size (tokens per page).
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// Returns the total pool size in bytes.
    pub fn total_bytes(&self) -> usize {
        self.num_pages * self.page_stride
    }

    /// Allocate a free page, returns its page_id or None if pool is full.
    pub fn alloc_page(&mut self) -> Option<u32> {
        for (i, free) in self.free_pages.iter_mut().enumerate() {
            if *free {
                *free = false;
                return Some(i as u32);
            }
        }
        None
    }

    /// Free a previously allocated page.
    pub fn free_page(&mut self, page_id: u32) {
        if (page_id as usize) < self.num_pages {
            self.free_pages[page_id as usize] = true;
        }
    }

    /// Returns the number of free pages.
    pub fn num_free(&self) -> usize {
        self.free_pages.iter().filter(|&&f| f).count()
    }

    /// Compute the device byte offset for a specific page, layer, K/V, head, and token.
    pub fn offset_of(
        &self,
        page_id: u32,
        layer: usize,
        is_value: bool,
        kv_head: usize,
        token_in_page: usize,
    ) -> usize {
        if self.layout == super::KvLayoutStrategy::MlaCompressed {
            let layer_stride = self.page_size * self.kv_dim * self.elem_bytes;
            page_id as usize * self.page_stride
                + layer * layer_stride
                + token_in_page * self.kv_dim * self.elem_bytes
        } else {
            let kv_half_offset = if is_value {
                self.num_kv_heads * self.page_size * self.head_dim * self.elem_bytes
            } else {
                0
            };
            page_id as usize * self.page_stride
                + layer * 2 * self.num_kv_heads * self.page_size * self.head_dim * self.elem_bytes
                + kv_half_offset
                + kv_head * self.page_size * self.head_dim * self.elem_bytes
                + token_in_page * self.head_dim * self.elem_bytes
        }
    }
}

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl std::fmt::Debug for GpuPagedKvPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuPagedKvPool")
            .field("device_ptr", &format!("0x{:016x}", self.device_ptr))
            .field("page_stride", &self.page_stride)
            .field("num_pages", &self.num_pages)
            .field("page_size", &self.page_size)
            .field("num_layers", &self.num_layers)
            .field("kv_dim", &self.kv_dim)
            .field("layout", &self.layout)
            .field("num_free", &self.num_free())
            .finish()
    }
}
