#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use gllm_kernels::types::DType;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use super::backend_trait;
#[cfg(feature = "cuda")]
use super::cuda_backend::CudaBackend;
#[cfg(feature = "hip")]
use super::hip_backend::HipBackend;
#[cfg(any(feature = "cuda", feature = "hip"))]
use gllm_kernels::gpu::GpuBuffer;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use super::Element;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::engine::executor::{BackendError as BE, GeneratorForwardConfig, KvCacheConfig, SamplingConfig};
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::scheduler::types::StorageKey;

/// Metadata for a GPU-resident KV cache buffer, needed for swap operations.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
#[derive(Debug, Clone)]
pub(super) struct GpuKvCacheMeta {
    /// Device pointer to the KV cache buffer.
    pub device_ptr: u64,
    /// Total size in bytes.
    pub total_bytes: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads (GQA heads). MLA: unused (0).
    pub num_kv_heads: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Head dimension. MLA: this is d_c + d_rope (compressed KV dim).
    pub head_dim: usize,
    /// Data type of KV cache elements.
    pub kv_dtype: DType,
    /// Tokens per page.
    pub page_size: usize,
    /// Current sequence length (tokens written so far).
    pub seq_len: usize,
    /// Whether this is an MLA model (affects layout: no K/V split, no per-head dimension).
    pub is_mla: bool,
}
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl GpuKvCacheMeta {
    pub fn dtype_size(&self) -> usize { self.kv_dtype.size_bytes() }

    pub fn from_config(config: &KvCacheConfig, device_ptr: u64) -> Self {
        let kv_dtype = DType::F32;
        let is_mla = config.is_mla();
        let kv_dim = config.kv_dim();
        let total_bytes = if is_mla {
            config.num_layers() * config.max_seq_len() * kv_dim * config.dtype_size()
        } else {
            config.num_layers() * 2 * config.num_heads() * config.max_seq_len()
                * config.head_dim() * config.dtype_size()
        };
        Self {
            device_ptr,
            total_bytes,
            num_layers: config.num_layers(),
            num_kv_heads: config.num_heads(),
            max_seq_len: config.max_seq_len(),
            head_dim: if is_mla { kv_dim } else { config.head_dim() },
            kv_dtype,
            page_size: config.page_size,
            seq_len: 0,
            is_mla,
        }
    }

    /// Bytes per page across all layers.
    /// Standard: layers × 2 (K+V) × kv_heads × page_size × head_dim × dtype.
    /// MLA: layers × page_size × (d_c + d_rope) × dtype (single compressed vector).
    pub fn page_bytes(&self) -> usize {
        if self.is_mla {
            self.num_layers * self.page_size * self.head_dim * self.dtype_size()
        } else {
            self.num_layers * 2 * self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size()
        }
    }

    /// Byte offset of a page within the KV cache buffer.
    pub fn page_offset_for_head_layer(&self, page_id: usize) -> usize {
        page_id * self.page_size * self.head_dim * self.dtype_size()
    }

    /// Number of pages currently active.
    pub fn active_pages(&self) -> usize {
        if self.page_size == 0 { return 0; }
        (self.seq_len + self.page_size - 1) / self.page_size
    }

    /// Total number of pages this buffer can hold.
    pub fn total_pages(&self) -> usize {
        if self.page_size == 0 { return 0; }
        (self.max_seq_len + self.page_size - 1) / self.page_size
    }
}
/// Metadata for a GPU-resident paged KV cache buffer.
///
/// Layout: page pool with indirect addressing via page table.
/// Standard: each physical page: `[layer][K+V][head][page_size][head_dim]`
/// MLA: each physical page: `[layer][page_size][kv_dim]`
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
#[derive(Debug, Clone)]
pub(super) struct GpuPagedKvMeta {
    /// Device pointer to the page pool buffer.
    pub pool_ptr: u64,
    /// Total size of page pool in bytes.
    pub pool_bytes: usize,
    /// Number of physical pages in the pool.
    pub num_physical_pages: usize,
    /// Tokens per page.
    pub page_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of KV heads. MLA: original value (unused for sizing).
    pub num_kv_heads: usize,
    /// Head dimension. MLA: unused for sizing.
    pub head_dim: usize,
    /// Effective KV dimension per token per layer.
    /// Standard: num_kv_heads * head_dim. MLA: d_c + d_rope.
    pub kv_dim: usize,
    /// Data type of KV cache elements.
    pub kv_dtype: DType,
    /// Bytes per physical page.
    /// Standard: layers * 2 * kv_heads * page_size * head_dim * dtype.
    /// MLA: layers * page_size * kv_dim * dtype.
    pub page_stride: usize,
    /// Whether this is an MLA model.
    pub is_mla: bool,
}
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl GpuPagedKvMeta {
    pub fn dtype_size(&self) -> usize { self.kv_dtype.size_bytes() }

    pub fn new(
        pool_ptr: u64,
        num_physical_pages: usize,
        page_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        kv_dim: usize,
        kv_dtype: DType,
        is_mla: bool,
    ) -> Self {
        let dtype_size = kv_dtype.size_bytes();
        let page_stride = if is_mla {
            num_layers * page_size * kv_dim * dtype_size
        } else {
            num_layers * 2 * num_kv_heads * page_size * head_dim * dtype_size
        };
        let pool_bytes = num_physical_pages * page_stride;
        Self {
            pool_ptr,
            pool_bytes,
            num_physical_pages,
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dim,
            kv_dtype,
            page_stride,
            is_mla,
        }
    }

    /// Byte offset within a physical page for a specific layer, K or V half, head, and token.
    ///
    /// Standard layout within page: `[layer][kv_half][head][token][dim]`
    /// - kv_half: 0 = K, 1 = V
    ///
    /// MLA layout within page: `[layer][token][kv_dim]`
    /// - kv_half and head are ignored
    pub fn offset_in_page(&self, layer: usize, kv_half: usize, head: usize, token: usize) -> usize {
        if self.is_mla {
            let layer_stride = self.page_size * self.kv_dim * self.dtype_size();
            layer * layer_stride + token * self.kv_dim * self.dtype_size()
        } else {
            let layer_stride = 2 * self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size();
            let half_stride = self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size();
            let head_stride = self.page_size * self.head_dim * self.dtype_size();
            let token_stride = self.head_dim * self.dtype_size();
            layer * layer_stride + kv_half * half_stride + head * head_stride + token * token_stride
        }
    }

    /// Device pointer to a specific location within the page pool.
    pub fn ptr_at(&self, phys_page: usize, layer: usize, kv_half: usize, head: usize, token: usize) -> u64 {
        self.pool_ptr + (phys_page * self.page_stride + self.offset_in_page(layer, kv_half, head, token)) as u64
    }
}
/// Type aliases for GPU swap stores.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) type GpuSwapStore = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<StorageKey, Vec<u8>>>>;

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) type GpuKvMetaStore = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u64, GpuKvCacheMeta>>>;

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) type GpuPagedKvMetaStore = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u64, GpuPagedKvMeta>>>;

/// CPU-side sampling from logits: temperature -> top-k -> top-p -> softmax -> multinomial.
///
/// GPU 后端将 logits DtoH 后调用此函数。采样逻辑与 CPU 后端共享 `compat::sampling::sample_logits_row`，
/// 确保 temperature / top_k / top_p 在所有后端产生一致的随机采样行为（T==0 → argmax,
/// T>0 → multinomial）。禁止静默降级为 argmax。
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) fn sample_logits_cpu(
    logits: &[f32],
    vocab_size: usize,
    sampling: &SamplingConfig,
) -> Result<Vec<u32>, crate::engine::executor::BackendError> {
    use crate::engine::executor::BackendError as BE;

    if logits.is_empty() {
        return Err(BE::Other("sample_logits_cpu: empty logits".into()));
    }
    if vocab_size == 0 {
        return Err(BE::Other("sample_logits_cpu: vocab_size == 0".into()));
    }
    if logits.len() % vocab_size != 0 {
        return Err(BE::Other(format!(
            "sample_logits_cpu: logits len {} not divisible by vocab_size {}",
            logits.len(),
            vocab_size
        )));
    }

    let rows = logits.len() / vocab_size;
    let mut result = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * vocab_size;
        let row = &logits[start..start + vocab_size];
        result.push(crate::compat::sampling::sample_logits_row(row, sampling)?);
    }
    Ok(result)
}
