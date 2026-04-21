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
    /// Number of KV heads (GQA heads).
    pub num_kv_heads: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Data type of KV cache elements.
    pub kv_dtype: DType,
    /// Tokens per page.
    pub page_size: usize,
    /// Current sequence length (tokens written so far).
    pub seq_len: usize,
}
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl GpuKvCacheMeta {
    pub fn dtype_size(&self) -> usize { self.kv_dtype.size_bytes() }

    pub fn from_config(config: &KvCacheConfig, device_ptr: u64) -> Self {
        let kv_dtype = DType::F32; // KvCacheConfig.dtype_size carries raw bytes; derive DType
        let total_bytes = config.num_layers() * 2
            * config.num_heads() * config.max_seq_len() * config.head_dim() * config.dtype_size();
        Self {
            device_ptr,
            total_bytes,
            num_layers: config.num_layers(),
            num_kv_heads: config.num_heads(),
            max_seq_len: config.max_seq_len(),
            head_dim: config.head_dim(),
            kv_dtype,
            page_size: config.page_size,
            seq_len: 0,
        }
    }

    /// Bytes per page across all layers/heads (K+V combined).
    pub fn page_bytes(&self) -> usize {
        self.num_layers * 2 * self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size()
    }

    /// Byte offset of a page within the KV cache buffer.
    /// Layout: [K_all_layers | V_all_layers], each sub-block: [layer][head][seq][dim]
    /// Page offset within each K or V block:
    ///   page_id * page_size * head_dim * dtype_size  (per head per layer)
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
/// Each physical page: `[layer][K+V][head][page_size][head_dim]`
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
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
    /// Data type of KV cache elements.
    pub kv_dtype: DType,
    /// Bytes per physical page (all layers, K+V, all heads).
    pub page_stride: usize,
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
        kv_dtype: DType,
    ) -> Self {
        let dtype_size = kv_dtype.size_bytes();
        let page_stride = num_layers * 2 * num_kv_heads * page_size * head_dim * dtype_size;
        let pool_bytes = num_physical_pages * page_stride;
        Self {
            pool_ptr,
            pool_bytes,
            num_physical_pages,
            page_size,
            num_layers,
            num_kv_heads,
            head_dim,
            kv_dtype,
            page_stride,
        }
    }

    /// Byte offset within a physical page for a specific layer, K or V half, head, and token.
    ///
    /// Layout within page: `[layer][kv_half][head][token][dim]`
    /// - kv_half: 0 = K, 1 = V
    pub fn offset_in_page(&self, layer: usize, kv_half: usize, head: usize, token: usize) -> usize {
        let layer_stride = 2 * self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size();
        let half_stride = self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size();
        let head_stride = self.page_size * self.head_dim * self.dtype_size();
        let token_stride = self.head_dim * self.dtype_size();
        layer * layer_stride + kv_half * half_stride + head * head_stride + token * token_stride
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

// ---------------------------------------------------------------------------
// GPU weight resident cache (REQ-ARCH-005)
// ---------------------------------------------------------------------------

/// GPU 权重常驻缓存。
/// 首次 incremental forward 时一次性上传所有层权重到 GPU，后续 step DtoD 复制。
/// ARCH-GPU-DATAPATH §8.9.3
#[cfg(feature = "cuda")]
pub(super) struct GpuWeightCache {
    /// per-layer 权重 GPU buffer: layer_idx → (tensor_name → device_ptr)
    layers: Vec<std::collections::HashMap<String, u64>>,
    /// lm_head weight GPU buffer (device_ptr, None if not yet uploaded)
    lm_head_ptr: Option<u64>,
    /// 总显存占用（字节）
    pub total_bytes: usize,
    /// 是否已初始化
    initialized: bool,
    /// 原始 CUDA buffer 持有（防止提前释放）
    _buffers: Vec<Vec<gllm_kernels::gpu::cuda::CudaBuffer>>,
    /// lm_head buffer 持有
    _lm_head_buf: Option<gllm_kernels::gpu::cuda::CudaBuffer>,
}
#[cfg(feature = "cuda")]
impl GpuWeightCache {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            lm_head_ptr: None,
            total_bytes: 0,
            initialized: false,
            _buffers: Vec::new(),
            _lm_head_buf: None,
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// 获取某层某权重的 device_ptr（None 表示未缓存）
    pub fn get(&self, layer: usize, name: &str) -> Option<u64> {
        self.layers.get(layer)?.get(name).copied()
    }

    /// 获取 lm_head weight 的 device_ptr
    pub fn get_lm_head(&self) -> Option<u64> {
        self.lm_head_ptr
    }

    /// 上传 lm_head weight 到 GPU（首次调用时）
    pub fn init_lm_head(
        &mut self,
        data: &[u8],
        device: &gllm_kernels::gpu::cuda::CudaDevice,
        stream: &gllm_kernels::gpu::cuda::device::CudaStream,
    ) -> Result<(), BE> {
        use gllm_kernels::gpu::GpuDevice;
        let mut buf = device.alloc(data.len())
            .map_err(|e| BE::Cuda(format!("lm_head cache alloc: {e}")))?;
        device.htod(data, &mut buf, stream)
            .map_err(|e| BE::Cuda(format!("lm_head cache htod: {e}")))?;
        self.total_bytes += data.len();
        self.lm_head_ptr = Some(buf.as_device_ptr());
        self._lm_head_buf = Some(buf);
        Ok(())
    }

    /// 初始化缓存：上传所有层权重到 GPU
    pub fn init(
        &mut self,
        all_layer_weights: &[std::collections::HashMap<String, Vec<u8>>],
        device: &gllm_kernels::gpu::cuda::CudaDevice,
        stream: &gllm_kernels::gpu::cuda::device::CudaStream,
    ) -> Result<(), BE> {
        use gllm_kernels::gpu::GpuDevice;

        self.layers = Vec::with_capacity(all_layer_weights.len());
        self._buffers = Vec::with_capacity(all_layer_weights.len());

        for layer_weights in all_layer_weights {
            let mut layer_map = std::collections::HashMap::new();
            let mut layer_bufs = Vec::new();

            for (name, data) in layer_weights {
                let mut buf = device.alloc(data.len())
                    .map_err(|e| BE::Cuda(format!("weight cache alloc {name}: {e}")))?;
                device.htod(data, &mut buf, stream)
                    .map_err(|e| BE::Cuda(format!("weight cache htod {name}: {e}")))?;
                layer_map.insert(name.clone(), buf.as_device_ptr());
                self.total_bytes += data.len();
                layer_bufs.push(buf);
            }

            self.layers.push(layer_map);
            self._buffers.push(layer_bufs);
        }

        self.initialized = true;
        Ok(())
    }
}
/// GPU 权重常驻缓存 (HIP)。
/// 首次 incremental forward 时一次性上传所有层权重到 GPU，后续 step DtoD 复制。
#[cfg(feature = "hip")]
pub(super) struct HipWeightCache {
    layers: Vec<std::collections::HashMap<String, u64>>,
    lm_head_ptr: Option<u64>,
    pub total_bytes: usize,
    initialized: bool,
    _buffers: Vec<Vec<gllm_kernels::gpu::hip::HipBuffer>>,
    _lm_head_buf: Option<gllm_kernels::gpu::hip::HipBuffer>,
}
#[cfg(feature = "hip")]
impl HipWeightCache {
    pub fn new() -> Self {
        Self { layers: Vec::new(), lm_head_ptr: None, total_bytes: 0, initialized: false, _buffers: Vec::new(), _lm_head_buf: None }
    }

    pub fn is_initialized(&self) -> bool { self.initialized }

    pub fn get(&self, layer: usize, name: &str) -> Option<u64> {
        self.layers.get(layer)?.get(name).copied()
    }

    pub fn get_lm_head(&self) -> Option<u64> { self.lm_head_ptr }

    pub fn init_lm_head(
        &mut self,
        data: &[u8],
        device: &gllm_kernels::gpu::hip::HipDevice,
        stream: &gllm_kernels::gpu::hip::HipGpuStream,
    ) -> Result<(), BE> {
        use gllm_kernels::gpu::GpuDevice;
        let mut buf = device.alloc(data.len())
            .map_err(|e| BE::Hip(format!("lm_head cache alloc: {e}")))?;
        device.htod(data, &mut buf, stream)
            .map_err(|e| BE::Hip(format!("lm_head cache htod: {e}")))?;
        self.total_bytes += data.len();
        self.lm_head_ptr = Some(buf.as_device_ptr());
        self._lm_head_buf = Some(buf);
        Ok(())
    }

    pub fn init(
        &mut self,
        all_layer_weights: &[std::collections::HashMap<String, Vec<u8>>],
        device: &gllm_kernels::gpu::hip::HipDevice,
        stream: &gllm_kernels::gpu::hip::HipGpuStream,
    ) -> Result<(), BE> {
        use gllm_kernels::gpu::GpuDevice;
        self.layers = Vec::with_capacity(all_layer_weights.len());
        self._buffers = Vec::with_capacity(all_layer_weights.len());
        for layer_weights in all_layer_weights {
            let mut layer_map = std::collections::HashMap::new();
            let mut layer_bufs = Vec::new();
            for (name, data) in layer_weights {
                let mut buf = device.alloc(data.len())
                    .map_err(|e| BE::Hip(format!("weight cache alloc {name}: {e}")))?;
                device.htod(data, &mut buf, stream)
                    .map_err(|e| BE::Hip(format!("weight cache htod {name}: {e}")))?;
                layer_map.insert(name.clone(), buf.as_device_ptr());
                self.total_bytes += data.len();
                layer_bufs.push(buf);
            }
            self.layers.push(layer_map);
            self._buffers.push(layer_bufs);
        }
        self.initialized = true;
        Ok(())
    }
}
/// GPU 权重常驻缓存 (Metal)。
/// 首次 incremental forward 时一次性上传所有层权重到 GPU，后续 step DtoD 复制。
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) struct MetalWeightCache {
    layers: Vec<std::collections::HashMap<String, u64>>,
    lm_head_buf: Option<gllm_kernels::gpu::metal::MetalBuffer>,
    pub total_bytes: usize,
    initialized: bool,
    _buffers: Vec<Vec<gllm_kernels::gpu::metal::MetalBuffer>>,
}
#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalWeightCache {
    pub fn new() -> Self {
        Self { layers: Vec::new(), lm_head_buf: None, total_bytes: 0, initialized: false, _buffers: Vec::new() }
    }

    pub fn is_initialized(&self) -> bool { self.initialized }

    pub fn get(&self, layer: usize, name: &str) -> Option<u64> {
        self.layers.get(layer)?.get(name).copied()
    }

    /// Get a reference to the cached MetalBuffer for dtod operations.
    pub fn get_buf(&self, layer: usize, name: &str) -> Option<&gllm_kernels::gpu::metal::MetalBuffer> {
        let ptr = self.layers.get(layer)?.get(name)?;
        for buf in &self._buffers[layer] {
            if buf.as_device_ptr() == *ptr {
                return Some(buf);
            }
        }
        None
    }

    pub fn get_lm_head_buf(&self) -> Option<&gllm_kernels::gpu::metal::MetalBuffer> {
        self.lm_head_buf.as_ref()
    }

    pub fn init_lm_head(
        &mut self,
        data: &[u8],
        device: &gllm_kernels::gpu::metal::MetalDevice,
        stream: &gllm_kernels::gpu::metal::MetalStream,
    ) -> Result<(), BE> {
        use gllm_kernels::gpu::GpuDevice;
        let mut buf = device.alloc(data.len())
            .map_err(|e| BE::Metal(format!("lm_head cache alloc: {e}")))?;
        device.htod(data, &mut buf, stream)
            .map_err(|e| BE::Metal(format!("lm_head cache htod: {e}")))?;
        self.total_bytes += data.len();
        self.lm_head_buf = Some(buf);
        Ok(())
    }

    pub fn init(
        &mut self,
        all_layer_weights: &[std::collections::HashMap<String, Vec<u8>>],
        device: &gllm_kernels::gpu::metal::MetalDevice,
        stream: &gllm_kernels::gpu::metal::MetalStream,
    ) -> Result<(), BE> {
        use gllm_kernels::gpu::GpuDevice;
        self.layers = Vec::with_capacity(all_layer_weights.len());
        self._buffers = Vec::with_capacity(all_layer_weights.len());
        for layer_weights in all_layer_weights {
            let mut layer_map = std::collections::HashMap::new();
            let mut layer_bufs = Vec::new();
            for (name, data) in layer_weights {
                let mut buf = device.alloc(data.len())
                    .map_err(|e| BE::Metal(format!("weight cache alloc {name}: {e}")))?;
                device.htod(data, &mut buf, stream)
                    .map_err(|e| BE::Metal(format!("weight cache htod {name}: {e}")))?;
                layer_map.insert(name.clone(), buf.as_device_ptr());
                self.total_bytes += data.len();
                layer_bufs.push(buf);
            }
            self.layers.push(layer_map);
            self._buffers.push(layer_bufs);
        }
        self.initialized = true;
        Ok(())
    }
}
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
