#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use super::backend_trait;
#[cfg(feature = "cuda")]
use super::cuda_backend::CudaBackend;
#[cfg(feature = "hip")]
use super::hip_backend::HipBackend;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use super::weight_helpers::transpose_f32;
#[cfg(any(feature = "cuda", feature = "hip"))]
use super::bert_forward::build_mean_pool_graph;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use super::Element;
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::engine::executor::{BackendError as BE, GeneratorForwardConfig};
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::engine::executor::{
    BatchInput, AttentionTopology, KvCacheConfig, KvCacheHandle, LogitsHandle, SamplingConfig,
};
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
use crate::scheduler::types::{PageId, PageState, StorageKey};


// ---------------------------------------------------------------------------
// GPU KV cache metadata for swap support
// ---------------------------------------------------------------------------

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
    /// Bytes per element (dtype_size).
    pub dtype_size: usize,
    /// Tokens per page.
    pub page_size: usize,
    /// Current sequence length (tokens written so far).
    pub seq_len: usize,
}

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl GpuKvCacheMeta {
    pub fn from_config(config: &KvCacheConfig, device_ptr: u64) -> Self {
        let total_bytes = config.num_layers * 2
            * config.num_heads * config.max_seq_len * config.head_dim * config.dtype_size;
        Self {
            device_ptr,
            total_bytes,
            num_layers: config.num_layers,
            num_kv_heads: config.num_heads,
            max_seq_len: config.max_seq_len,
            head_dim: config.head_dim,
            dtype_size: config.dtype_size,
            page_size: config.page_size,
            seq_len: 0,
        }
    }

    /// Bytes per page across all layers/heads (K+V combined).
    pub fn page_bytes(&self) -> usize {
        self.num_layers * 2 * self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size
    }

    /// Byte offset of a page within the KV cache buffer.
    /// Layout: [K_all_layers | V_all_layers], each sub-block: [layer][head][seq][dim]
    /// Page offset within each K or V block:
    ///   page_id * page_size * head_dim * dtype_size  (per head per layer)
    pub fn page_offset_for_head_layer(&self, page_id: usize) -> usize {
        page_id * self.page_size * self.head_dim * self.dtype_size
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
    /// Bytes per element.
    pub dtype_size: usize,
    /// Bytes per physical page (all layers, K+V, all heads).
    pub page_stride: usize,
}

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl GpuPagedKvMeta {
    pub fn new(
        pool_ptr: u64,
        num_physical_pages: usize,
        page_size: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        dtype_size: usize,
    ) -> Self {
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
            dtype_size,
            page_stride,
        }
    }

    /// Byte offset within a physical page for a specific layer, K or V half, head, and token.
    ///
    /// Layout within page: `[layer][kv_half][head][token][dim]`
    /// - kv_half: 0 = K, 1 = V
    pub fn offset_in_page(&self, layer: usize, kv_half: usize, head: usize, token: usize) -> usize {
        let layer_stride = 2 * self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size;
        let half_stride = self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size;
        let head_stride = self.page_size * self.head_dim * self.dtype_size;
        let token_stride = self.head_dim * self.dtype_size;
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
    /// 总显存占用（字节）
    pub total_bytes: usize,
    /// 是否已初始化
    initialized: bool,
    /// 原始 CUDA buffer 持有（防止提前释放）
    _buffers: Vec<Vec<gllm_kernels::gpu::cuda::CudaBuffer>>,
}

#[cfg(feature = "cuda")]
impl GpuWeightCache {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            total_bytes: 0,
            initialized: false,
            _buffers: Vec::new(),
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// 获取某层某权重的 device_ptr（None 表示未缓存）
    pub fn get(&self, layer: usize, name: &str) -> Option<u64> {
        self.layers.get(layer)?.get(name).copied()
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
    pub total_bytes: usize,
    initialized: bool,
    _buffers: Vec<Vec<gllm_kernels::gpu::hip::HipBuffer>>,
}

#[cfg(feature = "hip")]
impl HipWeightCache {
    pub fn new() -> Self {
        Self { layers: Vec::new(), total_bytes: 0, initialized: false, _buffers: Vec::new() }
    }

    pub fn is_initialized(&self) -> bool { self.initialized }

    pub fn get(&self, layer: usize, name: &str) -> Option<u64> {
        self.layers.get(layer)?.get(name).copied()
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
    pub total_bytes: usize,
    initialized: bool,
    _buffers: Vec<Vec<gllm_kernels::gpu::metal::MetalBuffer>>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalWeightCache {
    pub fn new() -> Self {
        Self { layers: Vec::new(), total_bytes: 0, initialized: false, _buffers: Vec::new() }
    }

    pub fn is_initialized(&self) -> bool { self.initialized }

    pub fn get(&self, layer: usize, name: &str) -> Option<u64> {
        self.layers.get(layer)?.get(name).copied()
    }

    /// Get a reference to the cached MetalBuffer for dtod operations.
    pub fn get_buf(&self, layer: usize, name: &str) -> Option<&gllm_kernels::gpu::metal::MetalBuffer> {
        let ptr = self.layers.get(layer)?.get(name)?;
        // Find the buffer in _buffers that matches this ptr
        for buf in &self._buffers[layer] {
            if buf.as_device_ptr() == *ptr {
                return Some(buf);
            }
        }
        None
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

// ---------------------------------------------------------------------------
// Shared KV cache write helper (used by all GPU backends)
// ---------------------------------------------------------------------------

/// Write k_rope and v_proj host bytes into the GPU KV cache for a single layer.
///
/// `k_host` / `v_host`: raw bytes `[seq_len, kv_dim]` in f32 layout (from dtoh of GPU tensors).
/// Repacks per-head `[seq_len, head_dim]` and writes via `htod_raw` into the correct KV cache offsets.
/// `write_start`: the seq position at which to start writing.
/// Write k/v host bytes into the GPU KV cache using raw CUDA memcpy.
#[cfg(feature = "cuda")]
fn gpu_write_kv_cache(
    k_host: &[u8],
    v_host: &[u8],
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    handle: &KvCacheHandle,
    layer: usize,
    half_bytes: usize,
    write_start: usize,
    head_stride: usize,
    dtype_size: usize,
    device: &gllm_kernels::gpu::cuda::CudaDevice,
    _stream: &gllm_kernels::gpu::cuda::device::CudaStream,
) -> Result<(), BE> {
    for head in 0..num_kv_heads {
        let dst_k = handle.0
            + ((layer * num_kv_heads + head) * head_stride
                + write_start * head_dim * dtype_size) as u64;
        let mut k_packed = vec![0u8; seq_len * head_dim * dtype_size];
        for s in 0..seq_len {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let dst_off = s * head_dim * dtype_size;
            k_packed[dst_off..dst_off + head_dim * dtype_size]
                .copy_from_slice(&k_host[src_off..src_off + head_dim * dtype_size]);
        }
        let res = unsafe {
            (device.driver().cuMemcpyHtoD_v2)(
                dst_k,
                k_packed.as_ptr() as *const _,
                k_packed.len(),
            )
        };
        if res != 0 {
            return Err(BE::Other(format!("htod KV cache K failed: CUDA error {res}")));
        }

        let dst_v = handle.0 + half_bytes as u64
            + ((layer * num_kv_heads + head) * head_stride
                + write_start * head_dim * dtype_size) as u64;
        let mut v_packed = vec![0u8; seq_len * head_dim * dtype_size];
        for s in 0..seq_len {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let dst_off = s * head_dim * dtype_size;
            v_packed[dst_off..dst_off + head_dim * dtype_size]
                .copy_from_slice(&v_host[src_off..src_off + head_dim * dtype_size]);
        }
        let res = unsafe {
            (device.driver().cuMemcpyHtoD_v2)(
                dst_v,
                v_packed.as_ptr() as *const _,
                v_packed.len(),
            )
        };
        if res != 0 {
            return Err(BE::Other(format!("htod KV cache V failed: CUDA error {res}")));
        }
    }
    Ok(())
}

/// Write k/v host bytes into the HIP KV cache using raw HIP memcpy.
#[cfg(feature = "hip")]
fn gpu_write_kv_cache_hip(
    k_host: &[u8],
    v_host: &[u8],
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    handle: &KvCacheHandle,
    layer: usize,
    half_bytes: usize,
    write_start: usize,
    head_stride: usize,
    dtype_size: usize,
    device: &gllm_kernels::gpu::hip::HipDevice,
    _stream: &gllm_kernels::gpu::hip::HipGpuStream,
) -> Result<(), BE> {
    for head in 0..num_kv_heads {
        let dst_k = handle.0
            + ((layer * num_kv_heads + head) * head_stride
                + write_start * head_dim * dtype_size) as u64;
        let mut k_packed = vec![0u8; seq_len * head_dim * dtype_size];
        for s in 0..seq_len {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let dst_off = s * head_dim * dtype_size;
            k_packed[dst_off..dst_off + head_dim * dtype_size]
                .copy_from_slice(&k_host[src_off..src_off + head_dim * dtype_size]);
        }
        device.htod_raw(&k_packed, dst_k)
            .map_err(|e| BE::Other(format!("htod KV cache K failed: {e}")))?;

        let dst_v = handle.0 + half_bytes as u64
            + ((layer * num_kv_heads + head) * head_stride
                + write_start * head_dim * dtype_size) as u64;
        let mut v_packed = vec![0u8; seq_len * head_dim * dtype_size];
        for s in 0..seq_len {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let dst_off = s * head_dim * dtype_size;
            v_packed[dst_off..dst_off + head_dim * dtype_size]
                .copy_from_slice(&v_host[src_off..src_off + head_dim * dtype_size]);
        }
        device.htod_raw(&v_packed, dst_v)
            .map_err(|e| BE::Other(format!("htod KV cache V failed: {e}")))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GPU KV cache helpers for native GPU attention (zero CPU round-trip)
// ---------------------------------------------------------------------------

/// Compute device pointers to a specific layer's K and V slices within the GPU KV cache.
///
/// Layout: `[K_all_layers | V_all_layers]`, each sub-block `[layer][head][seq][dim]`.
/// Returns `(k_layer_ptr, v_layer_ptr)` pointing to `[num_kv_heads * max_seq * head_dim]` for the layer.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
fn kv_cache_layer_ptrs(
    handle: &KvCacheHandle,
    layer: usize,
    half_bytes: usize,
    num_kv_heads: usize,
    head_stride: usize,
) -> (u64, u64) {
    let layer_offset = (layer * num_kv_heads * head_stride) as u64;
    let k_ptr = handle.0 + layer_offset;
    let v_ptr = handle.0 + half_bytes as u64 + layer_offset;
    (k_ptr, v_ptr)
}

/// Write k_rope/v_proj from GPU device pointers into the GPU KV cache via DtoD copy (CUDA).
///
/// Unlike `gpu_write_kv_cache` which takes host bytes, this operates entirely on-device.
/// Source layout: `[seq_len, kv_dim]` (interleaved heads).
/// Destination layout: per-head `[max_seq, head_dim]` within the KV cache.
#[cfg(feature = "cuda")]
fn gpu_write_kv_cache_device(
    k_device_ptr: u64,
    v_device_ptr: u64,
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    handle: &KvCacheHandle,
    layer: usize,
    half_bytes: usize,
    write_start: usize,
    head_stride: usize,
    dtype_size: usize,
    device: &gllm_kernels::gpu::cuda::CudaDevice,
    stream: &gllm_kernels::gpu::cuda::device::CudaStream,
    gpu_profile: &gllm_kernels::gpu::GpuDeviceProfile,
    sm_version: u32,
) -> Result<(), BE> {
    // Fast path: single KV head (MQA) — one DtoD per K/V
    if num_kv_heads == 1 {
        let copy_bytes = seq_len * head_dim * dtype_size;
        let dst_k = handle.0
            + (layer * num_kv_heads * head_stride + write_start * head_dim * dtype_size) as u64;
        let dst_v = handle.0 + half_bytes as u64
            + (layer * num_kv_heads * head_stride + write_start * head_dim * dtype_size) as u64;
        let res = unsafe { (device.driver().cuMemcpyDtoD_v2)(dst_k, k_device_ptr, copy_bytes) };
        if res != 0 { return Err(BE::Other(format!("DtoD KV cache K failed: CUDA error {res}"))); }
        let res = unsafe { (device.driver().cuMemcpyDtoD_v2)(dst_v, v_device_ptr, copy_bytes) };
        if res != 0 { return Err(BE::Other(format!("DtoD KV cache V failed: CUDA error {res}"))); }
        return Ok(());
    }

    // GQA multi-head path: use scatter kernel (one launch replaces O(heads×seq) DtoD calls)
    let layer_offset = layer * num_kv_heads * head_stride;
    let half_offset = half_bytes;
    let scatter_graph = build_kv_scatter_graph(
        seq_len, num_kv_heads, head_dim, kv_dim,
        write_start, layer_offset, half_offset, head_stride, dtype_size,
    );
    let (_mod, entries) = cuda_compile_graph(device, gpu_profile, sm_version, &scatter_graph)?;

    // tensor map: inputs[0]=k_src, inputs[1]=v_src, inputs[2]=kv_cache
    let mut tensor_ptrs = std::collections::HashMap::new();
    for (idx, tmeta) in scatter_graph.tensors.iter().enumerate() {
        let tid = gllm_kernels::compiler::TensorId(idx as u32);
        match tmeta.name.as_str() {
            "k_src"    => { tensor_ptrs.insert(tid, k_device_ptr); }
            "v_src"    => { tensor_ptrs.insert(tid, v_device_ptr); }
            "kv_cache" => { tensor_ptrs.insert(tid, handle.0); }
            _          => {}
        }
    }
    cuda_launch_graph(device, stream, &entries, &tensor_ptrs, &scatter_graph)?;
    Ok(())
}

/// Build a CompilerGraph containing a single KvScatterWrite op.
///
/// Inputs: k_src[seq_len, kv_dim], v_src[seq_len, kv_dim], kv_cache (opaque ptr)
/// The op encodes all layout parameters; no output tensor is needed (writes in-place).
#[cfg(feature = "cuda")]
fn build_kv_scatter_graph(
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    write_start: usize,
    layer_offset: usize,
    half_offset: usize,
    head_stride: usize,
    dtype_size: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let k_src = g.add_tensor_concrete("k_src", &[seq_len, kv_dim], DType::F32);
    let v_src = g.add_tensor_concrete("v_src", &[seq_len, kv_dim], DType::F32);
    // kv_cache is an opaque byte buffer — represent as 1-element placeholder
    let kv_cache = g.add_tensor_concrete("kv_cache", &[1], DType::F32);
    g.inputs = vec![k_src, v_src, kv_cache];

    // dummy output (scatter writes in-place; we need at least one output for graph validity)
    let out = g.add_tensor_concrete("scatter_out", &[1], DType::F32);
    g.add_op(
        OpKind::KvScatterWrite {
            seq_len,
            num_kv_heads,
            head_dim,
            kv_dim,
            write_start,
            layer_offset,
            half_offset,
            head_stride,
            dtype_size,
        },
        vec![k_src, v_src, kv_cache],
        vec![out],
        "kv_scatter_write",
    );
    g.outputs = vec![out];
    g
}

/// Write k_rope/v_proj from GPU device pointers into the GPU KV cache via DtoD copy (HIP).
#[cfg(feature = "hip")]
fn gpu_write_kv_cache_device_hip(
    k_device_ptr: u64,
    v_device_ptr: u64,
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    handle: &KvCacheHandle,
    layer: usize,
    half_bytes: usize,
    write_start: usize,
    head_stride: usize,
    dtype_size: usize,
    device: &gllm_kernels::gpu::hip::HipDevice,
    _stream: &gllm_kernels::gpu::hip::HipGpuStream,
) -> Result<(), BE> {
    // Fast path: single KV head (MQA) — one DtoD per K/V
    if num_kv_heads == 1 {
        let copy_bytes = seq_len * head_dim * dtype_size;
        let dst_k = handle.0
            + (layer * num_kv_heads * head_stride + write_start * head_dim * dtype_size) as u64;
        let dst_v = handle.0 + half_bytes as u64
            + (layer * num_kv_heads * head_stride + write_start * head_dim * dtype_size) as u64;
        let res = unsafe { (device.driver().hipMemcpyDtoD)(dst_k as _, k_device_ptr as _, copy_bytes) };
        if res != 0 { return Err(BE::Other(format!("DtoD KV cache K failed: HIP error {res}"))); }
        let res = unsafe { (device.driver().hipMemcpyDtoD)(dst_v as _, v_device_ptr as _, copy_bytes) };
        if res != 0 { return Err(BE::Other(format!("DtoD KV cache V failed: HIP error {res}"))); }
        return Ok(());
    }

    for head in 0..num_kv_heads {
        let dst_k = handle.0
            + ((layer * num_kv_heads + head) * head_stride
                + write_start * head_dim * dtype_size) as u64;
        let dst_v = handle.0 + half_bytes as u64
            + ((layer * num_kv_heads + head) * head_stride
                + write_start * head_dim * dtype_size) as u64;

        for s in 0..seq_len {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let dst_off_s = (s * head_dim * dtype_size) as u64;
            let copy_bytes = head_dim * dtype_size;

            // K: DtoD via hipMemcpyDtoD
            let res = unsafe {
                (device.driver().hipMemcpyDtoD)(
                    (dst_k + dst_off_s) as _,
                    (k_device_ptr + src_off as u64) as _,
                    copy_bytes,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD KV cache K failed: HIP error {res}")));
            }

            // V: DtoD via hipMemcpyDtoD
            let res = unsafe {
                (device.driver().hipMemcpyDtoD)(
                    (dst_v + dst_off_s) as _,
                    (v_device_ptr + src_off as u64) as _,
                    copy_bytes,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD KV cache V failed: HIP error {res}")));
            }
        }
    }
    Ok(())
}

/// Write k_rope/v_proj into the GPU KV cache via a single scatter kernel launch (HIP).
///
/// MQA fast path (num_kv_heads == 1) preserved as 2 DtoD calls.
#[cfg(feature = "hip")]
fn gpu_write_kv_cache_scatter_hip(
    k_device_ptr: u64,
    v_device_ptr: u64,
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    handle: &KvCacheHandle,
    layer: usize,
    half_bytes: usize,
    write_start: usize,
    head_stride: usize,
    dtype_size: usize,
    device: &gllm_kernels::gpu::hip::HipDevice,
    stream: &gllm_kernels::gpu::hip::HipGpuStream,
    gpu_profile: &gllm_kernels::gpu::GpuDeviceProfile,
    gfx_arch: u32,
) -> Result<(), BE> {
    // MQA fast path: single head — 2 DtoD calls cheaper than kernel launch overhead
    if num_kv_heads == 1 {
        let copy_bytes = seq_len * head_dim * dtype_size;
        let dst_k = handle.0
            + (layer * head_stride + write_start * head_dim * dtype_size) as u64;
        let dst_v = handle.0 + half_bytes as u64
            + (layer * head_stride + write_start * head_dim * dtype_size) as u64;
        let res = unsafe { (device.driver().hipMemcpyDtoD)(dst_k as _, k_device_ptr as _, copy_bytes) };
        if res != 0 { return Err(BE::Other(format!("DtoD KV scatter K (MQA) failed: HIP error {res}"))); }
        let res = unsafe { (device.driver().hipMemcpyDtoD)(dst_v as _, v_device_ptr as _, copy_bytes) };
        if res != 0 { return Err(BE::Other(format!("DtoD KV scatter V (MQA) failed: HIP error {res}"))); }
        return Ok(());
    }

    // GQA path: one scatter kernel launch
    let layer_offset = layer * num_kv_heads * head_stride;
    let half_offset = half_bytes;
    let scatter_graph = build_kv_scatter_graph_hip(
        seq_len, num_kv_heads, head_dim, kv_dim,
        write_start, layer_offset, half_offset, head_stride, dtype_size,
    );
    let (_mod, entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &scatter_graph)?;

    let mut tensor_ptrs = std::collections::HashMap::new();
    for (idx, tmeta) in scatter_graph.tensors.iter().enumerate() {
        let tid = gllm_kernels::compiler::TensorId(idx as u32);
        match tmeta.name.as_str() {
            "k_src"    => { tensor_ptrs.insert(tid, k_device_ptr); }
            "v_src"    => { tensor_ptrs.insert(tid, v_device_ptr); }
            "kv_cache" => { tensor_ptrs.insert(tid, handle.0); }
            _          => {}
        }
    }
    // dummy output ptr (scatter writes in-place)
    for (idx, tmeta) in scatter_graph.tensors.iter().enumerate() {
        let tid = gllm_kernels::compiler::TensorId(idx as u32);
        if tmeta.name == "scatter_out" {
            tensor_ptrs.insert(tid, handle.0);
        }
    }

    hip_launch_graph(device, stream, &entries, &tensor_ptrs, &scatter_graph)?;
    Ok(())
}

/// Build a CompilerGraph containing a single KvScatterWrite op (HIP variant).
#[cfg(feature = "hip")]
fn build_kv_scatter_graph_hip(
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    write_start: usize,
    layer_offset: usize,
    half_offset: usize,
    head_stride: usize,
    dtype_size: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let k_src = g.add_tensor_concrete("k_src", &[seq_len, kv_dim], DType::U8);
    let v_src = g.add_tensor_concrete("v_src", &[seq_len, kv_dim], DType::U8);
    let kv_cache = g.add_tensor_concrete("kv_cache", &[1], DType::U8);
    g.inputs = vec![k_src, v_src, kv_cache];
    let out = g.add_tensor_concrete("scatter_out", &[1], DType::U8);
    g.add_op(
        OpKind::KvScatterWrite {
            seq_len, num_kv_heads, head_dim, kv_dim,
            write_start, layer_offset, half_offset, head_stride, dtype_size,
        },
        vec![k_src, v_src, kv_cache],
        vec![out],
        "kv_scatter_write",
    );
    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for GPU-native cached GQA attention.
///
/// Inputs: q_rope[seq_len, q_dim], k_cache[total_seq, kv_dim], v_cache[total_seq, kv_dim]
/// Output: attn_out[seq_len, q_dim]
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
fn build_gpu_cached_attention_graph(
    seq_len: usize,
    total_seq: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::compiler::graph::AttentionStrategy;

    let mut g = CompilerGraph::new();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q = g.add_tensor_concrete("q_rope", &[seq_len, q_dim], dtype);
    let k = g.add_tensor_concrete("k_cache", &[total_seq, kv_dim], dtype);
    let v = g.add_tensor_concrete("v_cache", &[total_seq, kv_dim], dtype);
    g.inputs = vec![q, k, v];

    let out = g.add_tensor_concrete("attn_out", &[seq_len, q_dim], dtype);
    g.add_op(
        OpKind::CachedGQA {
            seq_len,
            total_seq,
            num_heads,
            num_kv_heads,
            head_dim,
            strategy: AttentionStrategy::Naive,
            kv_dtype: dtype,
        },
        vec![q, k, v],
        vec![out],
        "gpu_cached_gqa",
    );

    g.outputs = vec![out];
    g
}

/// Build a CompilerGraph for GPU paged cached GQA attention.
///
/// Uses `AttentionStrategy::Paged` which dispatches to `build_paged_attention_kernel`.
/// Inputs: q_rope[seq_len, q_dim], page_table (implicit via kernel params), kv_cache (page pool)
/// Output: attn_out[seq_len, q_dim]
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
fn build_gpu_paged_attention_graph(
    seq_len: usize,
    total_seq: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::compiler::graph::AttentionStrategy;

    let mut g = CompilerGraph::new();
    let q_dim = num_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    let q = g.add_tensor_concrete("q_rope", &[seq_len, q_dim], dtype);
    // page_table and kv_cache are passed as raw pointers via kernel params,
    // but we need placeholder tensors for the graph structure
    let pt = g.add_tensor_concrete("page_table", &[total_seq / page_size.max(1) + 1], gllm_kernels::types::DType::F32);
    let kv = g.add_tensor_concrete("kv_cache", &[1], dtype); // placeholder, actual size is page pool
    g.inputs = vec![q, pt, kv];

    let out = g.add_tensor_concrete("attn_out", &[seq_len, q_dim], dtype);
    g.add_op(
        OpKind::CachedGQA {
            seq_len,
            total_seq,
            num_heads,
            num_kv_heads,
            head_dim,
            strategy: AttentionStrategy::Paged { page_size },
            kv_dtype: dtype,
        },
        vec![q, pt, kv],
        vec![out],
        "gpu_paged_gqa",
    );

    g.outputs = vec![out];
    g
}

/// Write k_rope/v_proj into paged GPU KV cache via DtoD (HIP).
#[cfg(feature = "hip")]
fn gpu_write_kv_paged_hip(
    k_device_ptr: u64,
    v_device_ptr: u64,
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    meta: &GpuPagedKvMeta,
    layer: usize,
    page_table: &[u32],
    write_start: usize,
    device: &gllm_kernels::gpu::hip::HipDevice,
) -> Result<(), BE> {
    let dtype_size = meta.dtype_size;
    let page_size = meta.page_size;

    for s in 0..seq_len {
        let global_token = write_start + s;
        let logical_page = global_token / page_size;
        let token_in_page = global_token % page_size;

        if logical_page >= page_table.len() {
            return Err(BE::Other(format!(
                "gpu_write_kv_paged_hip: token {} maps to logical page {} but page_table has {} entries",
                global_token, logical_page, page_table.len()
            )));
        }
        let phys_page = page_table[logical_page] as usize;

        for head in 0..num_kv_heads {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let copy_bytes = head_dim * dtype_size;

            let dst_k = meta.ptr_at(phys_page, layer, 0, head, token_in_page);
            let res = unsafe {
                (device.driver().hipMemcpyDtoD)(
                    dst_k as _,
                    (k_device_ptr + src_off as u64) as _,
                    copy_bytes,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD paged KV K failed: HIP error {res}")));
            }

            let dst_v = meta.ptr_at(phys_page, layer, 1, head, token_in_page);
            let res = unsafe {
                (device.driver().hipMemcpyDtoD)(
                    dst_v as _,
                    (v_device_ptr + src_off as u64) as _,
                    copy_bytes,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD paged KV V failed: HIP error {res}")));
            }
        }
    }
    Ok(())
}

/// Allocate a paged KV cache on GPU and return metadata (HIP).
#[cfg(feature = "hip")]
fn gpu_alloc_paged_kv_cache_hip(
    device: &gllm_kernels::gpu::hip::HipDevice,
    num_physical_pages: usize,
    page_size: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype_size: usize,
) -> Result<GpuPagedKvMeta, BE> {
    use gllm_kernels::gpu::GpuDevice;
    let meta = GpuPagedKvMeta::new(0, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, dtype_size);
    let buf = device.alloc(meta.pool_bytes)
        .map_err(|e| BE::Hip(format!("paged KV cache alloc failed ({} bytes): {e}", meta.pool_bytes)))?;
    let ptr = buf.as_device_ptr();
    std::mem::forget(buf);
    Ok(GpuPagedKvMeta::new(ptr, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, dtype_size))
}

/// Upload a page table to GPU as a u32 array (HIP).
#[cfg(feature = "hip")]
fn gpu_upload_page_table_hip(
    device: &gllm_kernels::gpu::hip::HipDevice,
    stream: &gllm_kernels::gpu::hip::HipGpuStream,
    page_table: &[u32],
) -> Result<gllm_kernels::gpu::hip::HipBuffer, BE> {
    use gllm_kernels::gpu::GpuDevice;
    let bytes = unsafe {
        std::slice::from_raw_parts(page_table.as_ptr() as *const u8, page_table.len() * 4)
    };
    let mut buf = device.alloc(bytes.len())
        .map_err(|e| BE::Hip(format!("page table alloc failed: {e}")))?;
    device.htod(bytes, &mut buf, stream)
        .map_err(|e| BE::Hip(format!("page table upload failed: {e}")))?;
    Ok(buf)
}

/// Write k_rope/v_proj into paged Metal KV cache via shared memory ptr::copy.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_write_kv_paged(
    k_buf_ptr: u64,
    v_buf_ptr: u64,
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    meta: &GpuPagedKvMeta,
    layer: usize,
    page_table: &[u32],
    write_start: usize,
) -> Result<(), BE> {
    let dtype_size = meta.dtype_size;
    let page_size = meta.page_size;
    let k_src = k_buf_ptr as *const u8;
    let v_src = v_buf_ptr as *const u8;

    for s in 0..seq_len {
        let global_token = write_start + s;
        let logical_page = global_token / page_size;
        let token_in_page = global_token % page_size;

        if logical_page >= page_table.len() {
            return Err(BE::Other(format!(
                "metal_write_kv_paged: token {} maps to logical page {} but page_table has {} entries",
                global_token, logical_page, page_table.len()
            )));
        }
        let phys_page = page_table[logical_page] as usize;

        for head in 0..num_kv_heads {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let copy_bytes = head_dim * dtype_size;

            let dst_k = meta.ptr_at(phys_page, layer, 0, head, token_in_page) as *mut u8;
            let dst_v = meta.ptr_at(phys_page, layer, 1, head, token_in_page) as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(k_src.add(src_off), dst_k, copy_bytes);
                std::ptr::copy_nonoverlapping(v_src.add(src_off), dst_v, copy_bytes);
            }
        }
    }
    Ok(())
}

/// Allocate a paged KV cache (Metal) — shared memory, direct ptr access.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_alloc_paged_kv_cache(
    device: &gllm_kernels::gpu::metal::MetalDevice,
    num_physical_pages: usize,
    page_size: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype_size: usize,
) -> Result<GpuPagedKvMeta, BE> {
    use gllm_kernels::gpu::GpuDevice;
    let meta = GpuPagedKvMeta::new(0, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, dtype_size);
    let buf = device.alloc(meta.pool_bytes)
        .map_err(|e| BE::Metal(format!("paged KV cache alloc failed ({} bytes): {e}", meta.pool_bytes)))?;
    let ptr = buf.as_device_ptr();
    std::mem::forget(buf);
    Ok(GpuPagedKvMeta::new(ptr, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, dtype_size))
}

/// Write k_rope/v_proj into paged GPU KV cache via DtoD (CUDA).
///
/// Writes new tokens into the correct physical pages based on the page table.
/// Source: k_device_ptr/v_device_ptr `[seq_len, kv_dim]` (interleaved heads)
/// Destination: paged KV cache at the correct physical page + token offset
#[cfg(feature = "cuda")]
fn gpu_write_kv_paged(
    k_device_ptr: u64,
    v_device_ptr: u64,
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    meta: &GpuPagedKvMeta,
    layer: usize,
    page_table: &[u32],
    write_start: usize,
    device: &gllm_kernels::gpu::cuda::CudaDevice,
) -> Result<(), BE> {
    let dtype_size = meta.dtype_size;
    let page_size = meta.page_size;

    for s in 0..seq_len {
        let global_token = write_start + s;
        let logical_page = global_token / page_size;
        let token_in_page = global_token % page_size;

        if logical_page >= page_table.len() {
            return Err(BE::Other(format!(
                "gpu_write_kv_paged: token {} maps to logical page {} but page_table has {} entries",
                global_token, logical_page, page_table.len()
            )));
        }
        let phys_page = page_table[logical_page] as usize;

        for head in 0..num_kv_heads {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let copy_bytes = head_dim * dtype_size;

            // K: DtoD into paged cache
            let dst_k = meta.ptr_at(phys_page, layer, 0, head, token_in_page);
            let res = unsafe {
                (device.driver().cuMemcpyDtoD_v2)(
                    dst_k,
                    k_device_ptr + src_off as u64,
                    copy_bytes,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD paged KV K failed: CUDA error {res}")));
            }

            // V: DtoD into paged cache
            let dst_v = meta.ptr_at(phys_page, layer, 1, head, token_in_page);
            let res = unsafe {
                (device.driver().cuMemcpyDtoD_v2)(
                    dst_v,
                    v_device_ptr + src_off as u64,
                    copy_bytes,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD paged KV V failed: CUDA error {res}")));
            }
        }
    }
    Ok(())
}

/// Allocate a paged KV cache on GPU and return metadata.
#[cfg(feature = "cuda")]
fn gpu_alloc_paged_kv_cache(
    device: &gllm_kernels::gpu::cuda::CudaDevice,
    num_physical_pages: usize,
    page_size: usize,
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype_size: usize,
) -> Result<GpuPagedKvMeta, BE> {
    use gllm_kernels::gpu::GpuDevice;
    let meta = GpuPagedKvMeta::new(0, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, dtype_size);
    let buf = device.alloc(meta.pool_bytes)
        .map_err(|e| BE::Cuda(format!("paged KV cache alloc failed ({} bytes): {e}", meta.pool_bytes)))?;
    let ptr = buf.as_device_ptr();
    std::mem::forget(buf);
    Ok(GpuPagedKvMeta::new(ptr, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, dtype_size))
}

/// Upload a page table (logical→physical mapping) to GPU as a u32 array.
#[cfg(feature = "cuda")]
fn gpu_upload_page_table(
    device: &gllm_kernels::gpu::cuda::CudaDevice,
    stream: &gllm_kernels::gpu::cuda::device::CudaStream,
    page_table: &[u32],
) -> Result<gllm_kernels::gpu::cuda::CudaBuffer, BE> {
    use gllm_kernels::gpu::GpuDevice;
    let bytes = unsafe {
        std::slice::from_raw_parts(page_table.as_ptr() as *const u8, page_table.len() * 4)
    };
    let mut buf = device.alloc(bytes.len())
        .map_err(|e| BE::Cuda(format!("page table alloc failed: {e}")))?;
    device.htod(bytes, &mut buf, stream)
        .map_err(|e| BE::Cuda(format!("page table upload failed: {e}")))?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// GPU compilation & kernel launch helpers (CUDA)
// ---------------------------------------------------------------------------

/// Compile a CompilerGraph to PTX source via the full JIT pipeline.
///
/// Pipeline: CompilerGraph → ScalarOpRegistry → SemanticDAG → FusionPlan → PTX
///
/// Returns (null-terminated PTX bytes, FusionPlan).
#[cfg(feature = "cuda")]
pub(crate) fn compile_graph_to_ptx(
    sm_version: u32,
    graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<(Vec<u8>, gllm_kernels::compiler::FusionPlan), BE> {
    use gllm_kernels::compiler::{
        ScalarOpRegistry, SemanticDAG,
        fusion,
        codegen::gpu_ir::{GpuDialect, PtxDialect, gpu_emit_plan},
    };

    let registry = ScalarOpRegistry::with_defaults();
    let dag = SemanticDAG::from_graph(graph, &registry);
    let profile = gllm_kernels::dispatch::DeviceProfile::detect();
    let plan = fusion::fuse_with_dag_prebuilt(graph, &dag, &profile);

    // Extract dtype from graph ops (Gemm/GemmBias/CachedGQA carry dtype, default F32)
    let graph_dtype = graph.ops.iter()
        .find_map(|op| match &op.kind {
            gllm_kernels::compiler::OpKind::Gemm { dtype, .. }
            | gllm_kernels::compiler::OpKind::GemmBias { dtype, .. } => Some(*dtype),
            gllm_kernels::compiler::OpKind::CachedGQA { kv_dtype, .. } => Some(*kv_dtype),
            _ => None,
        })
        .unwrap_or(gllm_kernels::types::DType::F32);

    let dialect = PtxDialect::with_dtype(sm_version, 1024, 49152, graph_dtype);
    let mut ptx = String::new();
    dialect.emit_header(&mut ptx);
    gpu_emit_plan(&dialect, &mut ptx, &plan, graph, Some(&registry), None)
        .map_err(|e| BE::Other(format!("PTX emission failed: {e}")))?;

    let mut ptx_bytes = ptx.into_bytes();
    ptx_bytes.push(0); // null terminator for cuModuleLoadData

    Ok((ptx_bytes, plan))
}

/// Determine the LaunchConfig for a fusion group based on its anchor op.
#[cfg(any(feature = "cuda", feature = "hip"))]
pub(crate) fn launch_config_for_op(
    profile: &gllm_kernels::gpu::GpuDeviceProfile,
    op_kind: &gllm_kernels::compiler::OpKind,
) -> gllm_kernels::gpu::LaunchConfig {
    use gllm_kernels::compiler::OpKind;
    match op_kind {
        OpKind::Add | OpKind::Mul | OpKind::Silu | OpKind::Gelu
        | OpKind::Residual | OpKind::SwiGlu | OpKind::GeGlu => {
            // Scale grid to saturate compute units
            let default_n = (profile.compute_units * profile.max_threads_per_block) as usize;
            profile.launch_config_1d(default_n.max(65536))
        }
        OpKind::RoPE { head_dim, .. } => {
            let n = *head_dim * 1;
            profile.launch_config_1d(n.max(64))
        }
        OpKind::LayerNorm { .. } | OpKind::RmsNorm { .. } => {
            let block = profile.max_threads_per_block.min(256);
            profile.launch_config_row_wise(profile.max_threads_per_block as usize, block)
        }
        OpKind::Softmax => {
            let block = profile.max_threads_per_block.min(256);
            profile.launch_config_row_wise(profile.max_threads_per_block as usize, block)
        }
        OpKind::Gemm { m, n, dtype, .. } | OpKind::GemmBias { m, n, dtype, .. } => {
            // Tile size derived from warp_size: sqrt(32)=5→8, sqrt(64)=8
            let tile = ((profile.warp_size as f32).sqrt() as u32).max(8).min(32);
            let grid_x = ((*n as u32) + tile - 1) / tile;
            let grid_y = ((*m as u32) + tile - 1) / tile;
            let elem_bytes = dtype.size_bytes() as u32;
            let shared = (2 * tile * tile * elem_bytes).min(profile.shared_mem_per_block);
            gllm_kernels::gpu::LaunchConfig {
                grid_dim: [grid_x * grid_y, 1, 1],
                block_dim: [tile * tile, 1, 1],
                shared_mem_bytes: shared,
            }
        }
        OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
            let block = (*head_dim as u32).next_power_of_two()
                .min(profile.max_threads_per_block);
            gllm_kernels::gpu::LaunchConfig {
                grid_dim: [(*seq_len as u32) * (*num_heads as u32), 1, 1],
                block_dim: [block, 1, 1],
                shared_mem_bytes: ((*seq_len + 2) as u32) * 4,
            }
        }
        OpKind::CachedGQA { seq_len, total_seq, num_heads, head_dim, strategy, .. } => {
            use gllm_kernels::compiler::graph::AttentionStrategy;
            match strategy {
                AttentionStrategy::Paged { page_size } => {
                    // Paged attention: one block per head, shared mem for K+V page tile
                    let block = (*head_dim as u32).next_power_of_two()
                        .min(profile.max_threads_per_block);
                    let elem_bytes = 4u32; // shared mem scores always f32
                    let smem = (2 * (*page_size as u32) * (*head_dim as u32) * elem_bytes)
                        .min(profile.shared_mem_per_block);
                    gllm_kernels::gpu::LaunchConfig {
                        grid_dim: [*num_heads as u32, 1, 1],
                        block_dim: [block, 1, 1],
                        shared_mem_bytes: smem,
                    }
                }
                _ => {
                    // Naive/FlashV2/SlidingWindow
                    let block = (*head_dim as u32).next_power_of_two()
                        .min(profile.max_threads_per_block);
                    gllm_kernels::gpu::LaunchConfig {
                        grid_dim: [(*seq_len as u32) * (*num_heads as u32), 1, 1],
                        block_dim: [block, 1, 1],
                        shared_mem_bytes: ((*total_seq + 2) as u32) * 4,
                    }
                }
            }
        }
        OpKind::MeanPool { hidden, .. } => {
            profile.launch_config_1d(*hidden)
        }
        OpKind::L2Normalize { hidden } => {
            profile.launch_config_1d(*hidden)
        }
        OpKind::Reshape { .. } | OpKind::Transpose { .. } => {
            profile.launch_config_1d(1) // metadata NOP
        }
        OpKind::KvScatterWrite { num_kv_heads, seq_len, head_dim, dtype_size, .. } => {
            let block = ((*head_dim * *dtype_size) as u32).min(profile.max_threads_per_block);
            gllm_kernels::gpu::LaunchConfig {
                grid_dim: [*num_kv_heads as u32, *seq_len as u32, 1],
                block_dim: [block, 1, 1],
                shared_mem_bytes: 0,
            }
        }
        _ => profile.launch_config_1d(profile.max_threads_per_block as usize),
    }
}

/// Build the kernel launch parameters for a specific op type.
///
/// Maps graph tensor device pointers and op-specific scalars to the kernel's
/// parameter array, following the conventions in kernel_builder.rs:
/// - Elementwise: (input, output, N)
/// - BinaryElementwise: (A, B, output, N)
/// - GEMM: (A, B, C, M, N, K)
/// - GemmBias: (A, B, C, M, N, K, bias)
/// - MHA: (Q, K, V, output, seq_len, num_heads, head_dim)
/// - NormLike: (input, output, N, [weight], [bias])
/// - MeanPool: (input, output)
#[cfg(any(feature = "cuda", feature = "hip"))]
pub(crate) fn build_kernel_params(
    op_kind: &gllm_kernels::compiler::OpKind,
    input_ptrs: &[u64],
    output_ptr: u64,
) -> Result<Vec<u64>, BE> {
    use gllm_kernels::compiler::OpKind;

    match op_kind {
        OpKind::GemmBias { m, n, k, .. } => {
            Ok(vec![
                input_ptrs[0], input_ptrs[1], output_ptr,
                *m as u64, *n as u64, *k as u64,
                input_ptrs[2], // bias
            ])
        }
        OpKind::Gemm { m, n, k, .. } => {
            Ok(vec![
                input_ptrs[0], input_ptrs[1], output_ptr,
                *m as u64, *n as u64, *k as u64,
            ])
        }
        OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
            Ok(vec![
                input_ptrs[0], input_ptrs[1], input_ptrs[2], output_ptr,
                *seq_len as u64, *num_heads as u64, *head_dim as u64,
            ])
        }
        OpKind::CachedGQA { total_seq, num_heads, head_dim, strategy, .. } => {
            use gllm_kernels::compiler::graph::AttentionStrategy;
            match strategy {
                AttentionStrategy::Paged { .. } => {
                    // Paged kernel ABI: (Q, page_table, kv_cache, output, num_pages, seq_len)
                    let num_pages = if let Some(ps) = match strategy {
                        AttentionStrategy::Paged { page_size } => Some(*page_size),
                        _ => None,
                    } { (*total_seq + ps - 1) / ps } else { 0 };
                    Ok(vec![
                        input_ptrs[0], input_ptrs[1], input_ptrs[2], output_ptr,
                        num_pages as u64, *total_seq as u64,
                    ])
                }
                _ => {
                    // Naive/FlashV2/SlidingWindow: (Q, K, V, output, total_seq, num_heads, head_dim)
                    Ok(vec![
                        input_ptrs[0], input_ptrs[1], input_ptrs[2], output_ptr,
                        *total_seq as u64, *num_heads as u64, *head_dim as u64,
                    ])
                }
            }
        }
        OpKind::LayerNorm { .. } => {
            let mut params = vec![input_ptrs[0], output_ptr];
            params.push(0); // placeholder for N
            if input_ptrs.len() > 1 { params.push(input_ptrs[1]); }
            if input_ptrs.len() > 2 { params.push(input_ptrs[2]); }
            Ok(params)
        }
        OpKind::RmsNorm { .. } => {
            let mut params = vec![input_ptrs[0], output_ptr];
            params.push(0); // placeholder for N
            if input_ptrs.len() > 1 { params.push(input_ptrs[1]); }
            Ok(params)
        }
        OpKind::Residual | OpKind::Add | OpKind::Mul => {
            Ok(vec![input_ptrs[0], input_ptrs[1], output_ptr, 0])
        }
        OpKind::Silu | OpKind::Gelu => {
            Ok(vec![input_ptrs[0], output_ptr, 0])
        }
        OpKind::SwiGlu | OpKind::GeGlu => {
            Ok(vec![input_ptrs[0], input_ptrs[1], output_ptr, 0])
        }
        OpKind::RoPE { head_dim, .. } => {
            Ok(vec![input_ptrs[0], output_ptr, *head_dim as u64, 0])
        }
        OpKind::MeanPool { .. } => {
            Ok(vec![input_ptrs[0], output_ptr])
        }
        OpKind::Softmax => {
            // Kernel: (input, output, N) — N placeholder filled by caller
            Ok(vec![input_ptrs[0], output_ptr, 0])
        }
        OpKind::L2Normalize { hidden } => {
            // Kernel: (input, output, hidden)
            Ok(vec![input_ptrs[0], output_ptr, *hidden as u64])
        }
        OpKind::Reshape { .. } | OpKind::Transpose { .. } => {
            // Metadata-only ops (NOP) — CLAUDE.md allows
            Ok(vec![])
        }
        OpKind::KvScatterWrite {
            layer_offset, half_offset, head_stride, write_start,
            kv_dim, head_dim, dtype_size, ..
        } => {
            // ABI: (k_src, v_src, kv_cache, layer_offset, half_offset, head_stride,
            //        write_start, kv_dim, head_dim, dtype_size)
            // inputs[0]=k_src, inputs[1]=v_src, inputs[2]=kv_cache ptr
            Ok(vec![
                input_ptrs[0], input_ptrs[1], input_ptrs[2],
                *layer_offset as u64, *half_offset as u64, *head_stride as u64,
                *write_start as u64, *kv_dim as u64, *head_dim as u64, *dtype_size as u64,
            ])
        }
        _ => Err(BE::Other(format!(
            "build_kernel_params: unhandled OpKind {:?}", op_kind
        ))),
    }
}

/// GPU kernel execution session for a compiled graph.
///
/// Holds the loaded CUDA module and per-kernel metadata needed for
/// launching the full graph in sequence.
#[cfg(feature = "cuda")]
pub(crate) struct GpuKernelEntry {
    pub(crate) func: gllm_kernels::gpu::cuda::driver::CUfunction,
    pub(crate) config: gllm_kernels::gpu::LaunchConfig,
    pub(crate) op_kind: gllm_kernels::compiler::OpKind,
    /// Input tensor IDs in the CompilerGraph.
    pub(crate) input_tids: Vec<gllm_kernels::compiler::TensorId>,
    /// Output tensor ID in the CompilerGraph.
    pub(crate) output_tid: gllm_kernels::compiler::TensorId,
}

/// Compile a CompilerGraph to a loaded CUDA module with kernel metadata.
///
/// Returns the module and a list of kernel entries ready for launch.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_compile_graph(
    device: &gllm_kernels::gpu::cuda::CudaDevice,
    gpu_profile: &gllm_kernels::gpu::GpuDeviceProfile,
    sm_version: u32,
    graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<(gllm_kernels::gpu::cuda::CudaModule, Vec<GpuKernelEntry>), BE> {
    use gllm_kernels::compiler::OpKind;

    // Step 1: Compile to PTX
    let (ptx_bytes, plan) = compile_graph_to_ptx(sm_version, graph)?;

    // Step 2: Load PTX module
    let module = device.load_ptx(&ptx_bytes)
        .map_err(|e| BE::Other(format!("cuModuleLoadData failed: {e}")))?;

    // Step 3: Extract kernel entries from fusion plan
    let mut entries = Vec::new();
    for group in &plan.groups {
        let anchor = graph.op(group.anchor)
            .ok_or_else(|| BE::Other(format!("Missing anchor op {:?}", group.anchor)))?;

        if matches!(anchor.kind, OpKind::Reshape { .. } | OpKind::Transpose { .. }) {
            continue;
        }

        let name = format!("group_{}", group.id);
        let func = module.get_function(&name)
            .map_err(|e| BE::Other(format!("cuModuleGetFunction({name}) failed: {e}")))?;
        let config = launch_config_for_op(gpu_profile, &anchor.kind);

        // Determine input/output tensor IDs:
        // For EpilogueInjection, inputs come from anchor, output from last epilogue op.
        let input_tids = anchor.inputs.clone();
        let output_tid = if !group.epilogue.is_empty() {
            let last_epi = *group.epilogue.last()
                .ok_or_else(|| BE::Other("empty epilogue in fusion group".into()))?;
            let last_op = graph.op(last_epi)
                .ok_or_else(|| BE::Other(format!("missing epilogue op {:?}", last_epi)))?;
            last_op.outputs[0]
        } else {
            anchor.outputs[0]
        };

        entries.push(GpuKernelEntry {
            func,
            config,
            op_kind: anchor.kind.clone(),
            input_tids,
            output_tid,
        });
    }

    Ok((module, entries))
}

/// Launch all kernels for a compiled GPU graph.
///
/// `tensor_buffers` maps CompilerGraph TensorId → GPU device pointer.
/// Kernels are launched in sequence (matching the fusion plan's topological order).
#[cfg(feature = "cuda")]
pub(crate) fn cuda_launch_graph(
    device: &gllm_kernels::gpu::cuda::CudaDevice,
    stream: &gllm_kernels::gpu::cuda::CudaStream,
    entries: &[GpuKernelEntry],
    tensor_ptrs: &std::collections::HashMap<gllm_kernels::compiler::TensorId, u64>,
    graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<(), BE> {
    use std::ffi::c_void;
    use gllm_kernels::compiler::OpKind;

    for entry in entries {
        let input_ptrs: Vec<u64> = entry.input_tids.iter()
            .map(|tid| tensor_ptrs.get(tid).copied().ok_or_else(|| {
                BE::Cuda(format!("missing tensor pointer for input {:?}", tid))
            }))
            .collect::<Result<Vec<_>, _>>()?;
        let output_ptr = tensor_ptrs.get(&entry.output_tid).copied().ok_or_else(|| {
            BE::Cuda(format!("missing tensor pointer for output {:?}", entry.output_tid))
        })?;

        // Build typed params based on op kind
        let mut raw_params = build_kernel_params(&entry.op_kind, &input_ptrs, output_ptr)?;

        // Fill in N placeholder for ops that need it
        match &entry.op_kind {
            OpKind::LayerNorm { .. } | OpKind::RmsNorm { .. } => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for LayerNorm/RmsNorm input {:?}", entry.input_tids[0])))?;
                let n = input_meta.shape.last()
                    .and_then(|d| d.as_concrete())
                    .unwrap_or(1);
                raw_params[2] = n as u64;
            }
            OpKind::Residual | OpKind::Add | OpKind::Mul => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for binary op input {:?}", entry.input_tids[0])))?;
                let n = input_meta.concrete_numel();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::Silu | OpKind::Gelu => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for activation input {:?}", entry.input_tids[0])))?;
                let n = input_meta.concrete_numel();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::SwiGlu | OpKind::GeGlu => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for SwiGlu/GeGlu input {:?}", entry.input_tids[0])))?;
                let n = input_meta.concrete_numel();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::RoPE { .. } => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for RoPE input {:?}", entry.input_tids[0])))?;
                let seq_len = input_meta.shape[0].as_concrete().unwrap_or(1);
                let last = raw_params.len() - 1;
                raw_params[last] = seq_len as u64;
            }
            OpKind::Softmax => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for Softmax input {:?}", entry.input_tids[0])))?;
                let n = input_meta.concrete_numel();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            _ => {}
        }

        // Convert to c_void param array for cuLaunchKernel
        let mut param_ptrs: Vec<*mut c_void> = raw_params.iter_mut()
            .map(|p| p as *mut u64 as *mut c_void)
            .collect();

        let cfg = &entry.config;
        device.launch_kernel(
            entry.func,
            (cfg.grid_dim[0], cfg.grid_dim[1], cfg.grid_dim[2]),
            (cfg.block_dim[0], cfg.block_dim[1], cfg.block_dim[2]),
            &param_ptrs,
            stream,
        ).map_err(|e| BE::Other(format!(
            "Kernel launch failed for {:?}: {e}", entry.op_kind
        )))?;
    }

    Ok(())
}

/// Full BERT encoder forward on CUDA GPU.
///
/// Mirrors `bert_encoder_forward` but executes on GPU:
/// 1. Token embedding lookup (CPU) → upload to GPU
/// 2. Per-layer: compile graph → PTX → load → launch kernels
/// 3. Mean pooling → download result
#[cfg(feature = "cuda")]
pub(super) fn cuda_bert_encoder_forward<E: Element>(
    backend: &CudaBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CudaBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    use gllm_kernels::gpu::{GpuDevice, GpuBuffer};
    use gllm_kernels::compiler::TensorId;
    use super::bert_forward::build_bert_layer_graph;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("BERT encoder only supports f32 element type".into()));
    }

    let device = &*backend.device;
    let stream = device.default_stream();
    let gpu_profile = &backend.gpu_profile;
    let sm_version = backend.sm_version();

    // SM < 70: GPU kernel_builder emits C-like code for non-GEMM ops (LayerNorm, MHA, Softmax)
    // which is invalid PTX. Fall back to CPU path via OOM error signal.
    if sm_version < 70 {
        return Err(BE::Cuda(format!(
            "GPU forward requires SM >= 70 for full kernel support (got SM {}). \
             Falling back to CPU. alloc failed",
            sm_version
        )));
    }

    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let transpose_w = super::gpu_helpers::needs_weight_transpose_gpu(weights);

    // ── Step (a-d): Embedding lookup + LayerNorm (CPU, same as CPU path) ──
    let mut hidden_state = super::gpu_helpers::embed_tokens_gpu(tokens, weights, backend, config)?;

    // ── Compile BERT layer graph to PTX (once, reused across layers) ──
    let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps, comp_dtype);
    let (_module, kernel_entries) = cuda_compile_graph(
        device, gpu_profile, sm_version, &graph,
    )?;

    // Upload hidden_state to GPU once before layer loop
    let elem_bytes = comp_dtype.size_bytes();
    let hidden_bytes = seq_len * hidden * elem_bytes;
    let mut gpu_hidden = device.alloc(hidden_bytes)
        .map_err(|e| BE::Other(format!("GPU alloc gpu_hidden failed: {e}")))?;
    {
        let bytes = &hidden_state[..hidden_bytes];
        device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Other(format!("htod initial hidden: {e}")))?;
    }

    // ── Pre-load all layer weights (CPU side, once) ──
    let all_bert_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
        .map(|layer| super::gpu_helpers::load_bert_layer_weights_gpu_typed(
            weights, backend, layer, seq_len, hidden, inter, transpose_w, comp_dtype,
        ))
        .collect::<Result<Vec<_>, _>>()?;

    // ── Per-layer GPU execution ──
    for layer in 0..num_layers {
        let layer_weights = &all_bert_weights[layer];

        let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
        let mut tensor_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

        for (idx, meta) in graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let size_bytes = meta.concrete_numel() * elem_bytes;
            let mut buf = device.alloc(size_bytes)
                .map_err(|e| BE::Other(format!("GPU alloc failed for {}: {e}", meta.name)))?;

            if meta.name == "input" {
                // DtoD from gpu_hidden (no CPU round-trip)
                let res = unsafe {
                    (device.driver().cuMemcpyDtoD_v2)(
                        buf.as_device_ptr(),
                        gpu_hidden.as_device_ptr(),
                        hidden_bytes_typed,
                    )
                };
                if res != 0 {
                    return Err(BE::Other(format!("DtoD hidden→input failed: CUDA error {res}")));
                }
            } else if let Some(data) = layer_weights.get(&meta.name) {
                device.htod(data, &mut buf, stream)
                    .map_err(|e| BE::Other(format!("htod {} failed: {e}", meta.name)))?;
            }

            tensor_ptrs.insert(tid, buf.as_device_ptr());
            gpu_buffers.push((tid, buf));
        }

        cuda_launch_graph(device, stream, &kernel_entries, &tensor_ptrs, &graph)?;
        device.sync()
            .map_err(|e| BE::Other(format!("GPU sync failed: {e}")))?;

        // Update gpu_hidden: DtoD from output (stays on GPU)
        let output_tid = graph.outputs[0];
        let output_ptr = gpu_buffers.iter().find(|(tid, _)| *tid == output_tid).map(|(_, b)| b.as_device_ptr())
            .ok_or_else(|| BE::Other("output buffer not found".into()))?;
        let res = unsafe {
            (device.driver().cuMemcpyDtoD_v2)(
                gpu_hidden.as_device_ptr(),
                output_ptr,
                hidden_bytes_typed,
            )
        };
        if res != 0 {
            return Err(BE::Other(format!("DtoD output→gpu_hidden failed: CUDA error {res}")));
        }
    }

    // ── Mean pooling (GPU) — gpu_hidden already on device, no CPU round-trip ──
    let pool_graph = build_mean_pool_graph(seq_len, hidden, super::jit_helpers::computation_dtype_from_config(config));
    let (_pool_module, pool_entries) = cuda_compile_graph(
        device, gpu_profile, sm_version, &pool_graph,
    )?;

    // Output buffer
    let elem_bytes = comp_dtype.size_bytes();
    let output_bytes = hidden * elem_bytes;
    let gpu_output = device.alloc(output_bytes)
        .map_err(|e| BE::Other(format!("GPU alloc pool output: {e}")))?;

    let mut pool_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
    pool_ptrs.insert(pool_graph.inputs[0], gpu_hidden.as_device_ptr());
    pool_ptrs.insert(pool_graph.outputs[0], gpu_output.as_device_ptr());

    cuda_launch_graph(device, stream, &pool_entries, &pool_ptrs, &pool_graph)?;
    device.sync().map_err(|e| BE::Other(format!("GPU sync pool: {e}")))?;

    let mut pooled_bytes = vec![0u8; output_bytes];
    device.dtoh(&gpu_output, &mut pooled_bytes, stream)
        .map_err(|e| BE::Other(format!("dtoh pool output: {e}")))?;

    // Convert to f32 slice without unnecessary to_vec() copy
    let pooled: Vec<f32> = unsafe {
        std::slice::from_raw_parts(pooled_bytes.as_ptr() as *const f32, hidden)
    }.to_vec();

    Ok(pooled)
}

/// Full decoder (generator) forward pass on CUDA GPU.
///
/// 1. Token embedding lookup (CPU)
/// 2. Per-layer: compile decoder layer graph -> PTX -> load -> launch kernels
/// 3. lm_head projection -> logits
/// 4. Return LogitsHandle per sequence
#[cfg(feature = "cuda")]
pub(super) fn cuda_decoder_forward<E: Element>(
    backend: &CudaBackend<E>,
    input: &BatchInput,
    _topology: &AttentionTopology,
    weights: &dyn backend_trait::TensorLookup<E, CudaBackend<E>>,
    kv_caches: &mut [KvCacheHandle],
    config: &GeneratorForwardConfig,
) -> Result<Vec<LogitsHandle>, BE> {
    use gllm_kernels::gpu::{GpuDevice, GpuBuffer};
    use gllm_kernels::compiler::TensorId;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder forward only supports f32 element type".into()));
    }

    let device = &*backend.device;
    let stream = device.default_stream();
    let gpu_profile = &backend.gpu_profile;
    let sm_version = backend.sm_version();

    // SM < 70: GPU kernel_builder emits C-like code for non-GEMM ops (RmsNorm, MHA, SwiGLU)
    // which is invalid PTX. Fall back to CPU path via OOM error signal.
    if sm_version < 70 {
        return Err(BE::Cuda(format!(
            "GPU forward requires SM >= 70 for full kernel support (got SM {}). \
             Falling back to CPU. alloc failed",
            sm_version
        )));
    }

    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let vocab_size = config.vocab_size;

    // Flatten all sequences into a single batch dimension
    let total_tokens: usize = input.sequences.iter().map(|s| s.tokens.len()).sum();
    let seq_len = total_tokens;

    // Embedding lookup (CPU, dtype-adaptive)
    let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let elem_bytes = comp_dtype.size_bytes();
    let word_emb = super::gpu_helpers::get_typed_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("token_embedding.weight", Some("token_embd.weight")),
        comp_dtype)?;
    let hidden_bytes = hidden * elem_bytes;
    let vocab = word_emb.len() / hidden_bytes;
    let mut hidden_state = vec![0u8; seq_len * hidden_bytes];
    let mut pos = 0;
    for seq in &input.sequences {
        for &tok in &seq.tokens {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
            }
            hidden_state[pos * hidden_bytes..(pos + 1) * hidden_bytes]
                .copy_from_slice(&word_emb[v * hidden_bytes..(v + 1) * hidden_bytes]);
            pos += 1;
        }
    }

    // Detect incremental decode: position > 0 and KV cache has data
    let rope_theta = config.rope_theta;
    let is_incremental = if let Some(handle) = kv_caches.first() {
        let meta_store = backend.kv_meta.lock()
            .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
        meta_store.get(&handle.0)
            .map(|m| m.seq_len > 0)
            .unwrap_or(false)
            && input.sequences.iter().all(|s| s.position > 0)
    } else { false };

    if is_incremental {
        // ── Incremental decode path (GPU-native, zero CPU round-trip) ──
        // Optimizations: hidden_state stays on GPU across layers, buffers pre-allocated,
        // syncs minimized (1 per layer instead of 3).
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
        let proj_graph = build_projection_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta, comp_dtype);
        let (_proj_mod, proj_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &proj_graph)?;
        let post_graph = build_post_attention_graph(seq_len, hidden, num_heads, head_dim, inter, eps, comp_dtype);
        let (_post_mod, post_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &post_graph)?;

        let handle = kv_caches.first()
            .ok_or_else(|| BE::Cuda("no KV cache handles provided".into()))?;
        let (cached_seq_len, half_bytes, _total_kv_floats, max_seq_len, head_stride, kv_dtype_size) = {
            let ms = backend.kv_meta.lock()
                .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
            let m = ms.get(&handle.0)
                .ok_or_else(|| BE::Cuda(format!("KV cache handle {} not found", handle.0)))?;
            let hb = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim * m.dtype_size;
            let tkf = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim;
            let hs = m.max_seq_len * m.head_dim * m.dtype_size;
            (m.seq_len, hb, tkf, m.max_seq_len, hs, m.dtype_size)
        };
        let total_seq = cached_seq_len + seq_len;

        // Compile GPU cached attention kernel (once, shared across all layers)
        // Choose paged or dense attention based on config
        let use_paged = config.paged_kv_page_table.is_some();
        let page_size = config.paged_kv_page_size;
        let attn_graph = if use_paged {
            build_gpu_paged_attention_graph(
                seq_len, total_seq, num_heads, num_kv_heads, head_dim, page_size, comp_dtype,
            )
        } else {
            build_gpu_cached_attention_graph(
                seq_len, total_seq, num_heads, num_kv_heads, head_dim, comp_dtype,
            )
        };
        let (_attn_mod, attn_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &attn_graph)?;

        // Paged KV: allocate paged cache + upload page table (once before layer loop)
        let paged_meta = if use_paged {
            let pt = config.paged_kv_page_table.as_ref().unwrap();
            let num_physical_pages = pt.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
            let meta = gpu_alloc_paged_kv_cache(
                device, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, kv_dtype_size,
            )?;
            Some((meta, pt.clone()))
        } else {
            None
        };

        // ── Pre-allocate GPU buffers (reused across all layers) ──
        let elem_bytes = comp_dtype.size_bytes();
        // hidden_state is f32 from embedding lookup; GPU buffers use comp_dtype
        let hidden_bytes = seq_len * hidden * elem_bytes;
        let attn_out_bytes = seq_len * q_dim * elem_bytes;

        // hidden_state GPU buffer — stays on GPU across layers
        let mut gpu_hidden = device.alloc(hidden_bytes)
            .map_err(|e| BE::Cuda(format!("GPU alloc gpu_hidden failed: {e}")))?;
        {
            let bytes = &hidden_state[..hidden_bytes];
            device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Cuda(format!("htod initial hidden: {e}")))?;
        }

        // Pre-allocate proj/attn/post buffers
        let mut proj_bufs: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
        for (idx, tmeta) in proj_graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let size_bytes = tmeta.concrete_numel() * elem_bytes;
            let buf = device.alloc(size_bytes)
                .map_err(|e| BE::Cuda(format!("GPU alloc proj {} failed: {e}", tmeta.name)))?;
            proj_bufs.push((tid, buf));
        }
        let attn_out_buf = device.alloc(attn_out_bytes)
            .map_err(|e| BE::Cuda(format!("GPU alloc attn_out failed: {e}")))?;
        let mut post_bufs: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
        for (idx, tmeta) in post_graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let size_bytes = tmeta.concrete_numel() * elem_bytes;
            let buf = device.alloc(size_bytes)
                .map_err(|e| BE::Cuda(format!("GPU alloc post {} failed: {e}", tmeta.name)))?;
            post_bufs.push((tid, buf));
        }

        // ── Pre-load all layer weights (CPU side, once) ──
        let all_layer_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
            .map(|layer| super::gpu_helpers::load_decoder_layer_weights_gpu_typed(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter, comp_dtype,
            ))
            .collect::<Result<Vec<_>, _>>()?;

        // ── GPU 权重常驻缓存 (REQ-ARCH-005): 首次 forward 上传所有层权重 ──
        {
            let mut wc = backend.weight_cache.lock()
                .map_err(|e| BE::Cuda(format!("weight_cache lock: {e}")))?;
            if !wc.is_initialized() {
                wc.init(&all_layer_weights, device, stream)?;
                log::debug!("[GpuWeightCache] initialized: {} bytes", wc.total_bytes);
            }
        }

        for layer in 0..num_layers {
            let layer_weights = &all_layer_weights[layer];

            // ── GPU: projection graph (RmsNorm → Q/K/V → RoPE) ──
            let mut proj_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            for (idx, tmeta) in proj_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let (_, ref mut buf) = proj_bufs[idx];
                if tmeta.name == "input" {
                    // DtoD from gpu_hidden (no CPU round-trip)
                    let res = unsafe {
                        (device.driver().cuMemcpyDtoD_v2)(
                            buf.as_device_ptr(),
                            gpu_hidden.as_device_ptr(),
                            hidden_bytes_typed,
                        )
                    };
                    if res != 0 {
                        return Err(BE::Other(format!("DtoD hidden→proj input failed: CUDA error {res}")));
                    }
                } else {
                    // 优先从 GPU 权重缓存 DtoD 复制，避免每 step htod
                    let wc = backend.weight_cache.lock()
                        .map_err(|e| BE::Cuda(format!("weight_cache lock: {e}")))?;
                    if let Some(cached_ptr) = wc.get(layer, &tmeta.name) {
                        let copy_bytes = tmeta.concrete_numel() * elem_bytes;
                        let res = unsafe {
                            (device.driver().cuMemcpyDtoD_v2)(
                                buf.as_device_ptr(),
                                cached_ptr,
                                copy_bytes,
                            )
                        };
                        if res != 0 {
                            return Err(BE::Cuda(format!("DtoD weight cache proj {} failed: CUDA error {res}", tmeta.name)));
                        }
                    } else if let Some(data) = layer_weights.get(&tmeta.name) {
                        device.htod(data, buf, stream).map_err(|e| BE::Cuda(format!("htod proj {}: {e}", tmeta.name)))?;
                    }
                }
                proj_ptrs.insert(tid, buf.as_device_ptr());
            }
            cuda_launch_graph(device, stream, &proj_entries, &proj_ptrs, &proj_graph)?;
            // No sync needed here — DtoD KV write and attention launch are on same stream (ordered)

            // ── GPU: write k_rope/v_proj into KV cache (DtoD, zero CPU copy) ──
            let k_tid = proj_graph.outputs[1];
            let v_tid = proj_graph.outputs[2];
            let k_ptr = proj_bufs.iter().find(|(t, _)| *t == k_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Cuda(format!("proj buffer for k_tid {:?} not found", k_tid)))?;
            let v_ptr = proj_bufs.iter().find(|(t, _)| *t == v_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Cuda(format!("proj buffer for v_tid {:?} not found", v_tid)))?;

            if let Some((ref paged_meta, ref page_table)) = paged_meta {
                // Paged KV: scatter write into physical pages
                gpu_write_kv_paged(
                    k_ptr, v_ptr, kv_dim, seq_len, num_kv_heads, head_dim,
                    paged_meta, layer, page_table, cached_seq_len, device,
                )?;
            } else {
                // Dense KV: contiguous write
                gpu_write_kv_cache_device(
                    k_ptr, v_ptr, kv_dim, seq_len, num_kv_heads, head_dim,
                    handle, layer, half_bytes, cached_seq_len, head_stride, kv_dtype_size, device, stream,
                    gpu_profile, sm_version,
                )?;
            }

            if layer == 0 {
                let mut ms = backend.kv_meta.lock()
                    .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
                if let Some(m) = ms.get_mut(&handle.0) {
                    m.seq_len = (m.seq_len + seq_len).min(m.max_seq_len);
                }
            }

            // ── GPU: cached attention (q stays on GPU, reads KV cache directly) ──
            let q_tid = proj_graph.outputs[0];
            let q_ptr = proj_bufs.iter().find(|(t, _)| *t == q_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Cuda(format!("proj buffer for q_tid {:?} not found", q_tid)))?;

            let mut attn_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            if let Some((ref paged_meta, ref page_table)) = paged_meta {
                // Paged attention: upload page table, pass pool pointer
                let pt_buf = gpu_upload_page_table(device, stream, page_table)?;
                for (idx, tmeta) in attn_graph.tensors.iter().enumerate() {
                    let tid = TensorId(idx as u32);
                    match tmeta.name.as_str() {
                        "q_rope" => { attn_ptrs.insert(tid, q_ptr); }
                        "page_table" => { attn_ptrs.insert(tid, pt_buf.as_device_ptr()); }
                        "kv_cache" => { attn_ptrs.insert(tid, paged_meta.pool_ptr); }
                        "attn_out" => { attn_ptrs.insert(tid, attn_out_buf.as_device_ptr()); }
                        _ => {}
                    }
                }
            } else {
                // Dense attention: pass layer K/V pointers directly
                let (k_layer_ptr, v_layer_ptr) = kv_cache_layer_ptrs(handle, layer, half_bytes, num_kv_heads, head_stride);
                for (idx, tmeta) in attn_graph.tensors.iter().enumerate() {
                    let tid = TensorId(idx as u32);
                    match tmeta.name.as_str() {
                        "q_rope" => { attn_ptrs.insert(tid, q_ptr); }
                        "k_cache" => { attn_ptrs.insert(tid, k_layer_ptr); }
                        "v_cache" => { attn_ptrs.insert(tid, v_layer_ptr); }
                        "attn_out" => { attn_ptrs.insert(tid, attn_out_buf.as_device_ptr()); }
                        _ => {}
                    }
                }
            }
            cuda_launch_graph(device, stream, &attn_entries, &attn_ptrs, &attn_graph)?;
            // No sync — post graph launch is on same stream

            // ── GPU: post-attention graph (O Gemm → Residual → FFN → Residual) ──
            let mut post_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            for (idx, tmeta) in post_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let (_, ref mut buf) = post_bufs[idx];
                if tmeta.name == "input" {
                    // DtoD from gpu_hidden
                    let res = unsafe {
                        (device.driver().cuMemcpyDtoD_v2)(
                            buf.as_device_ptr(),
                            gpu_hidden.as_device_ptr(),
                            hidden_bytes,
                        )
                    };
                    if res != 0 {
                        return Err(BE::Other(format!("DtoD hidden→post input failed: CUDA error {res}")));
                    }
                } else if tmeta.name == "attn_out" {
                    // attn_out already on GPU — DtoD copy from attn_out_buf
                    let res = unsafe {
                        (device.driver().cuMemcpyDtoD_v2)(
                            buf.as_device_ptr(),
                            attn_out_buf.as_device_ptr(),
                            attn_out_bytes,
                        )
                    };
                    if res != 0 {
                        return Err(BE::Other(format!("DtoD attn_out→post failed: CUDA error {res}")));
                    }
                } else {
                    // 优先从 GPU 权重缓存 DtoD 复制
                    let wc = backend.weight_cache.lock()
                        .map_err(|e| BE::Cuda(format!("weight_cache lock: {e}")))?;
                    if let Some(cached_ptr) = wc.get(layer, &tmeta.name) {
                        let copy_bytes = tmeta.concrete_numel() * elem_bytes;
                        let res = unsafe {
                            (device.driver().cuMemcpyDtoD_v2)(
                                buf.as_device_ptr(),
                                cached_ptr,
                                copy_bytes,
                            )
                        };
                        if res != 0 {
                            return Err(BE::Cuda(format!("DtoD weight cache post {} failed: CUDA error {res}", tmeta.name)));
                        }
                    } else if let Some(data) = layer_weights.get(&tmeta.name) {
                        device.htod(data, buf, stream).map_err(|e| BE::Cuda(format!("htod post {}: {e}", tmeta.name)))?;
                    }
                }
                post_ptrs.insert(tid, buf.as_device_ptr());
            }
            cuda_launch_graph(device, stream, &post_entries, &post_ptrs, &post_graph)?;
            // Single sync per layer — ensures post graph completes before next layer reads output
            device.sync().map_err(|e| BE::Cuda(format!("GPU sync layer {layer}: {e}")))?;

            // ── Update gpu_hidden: DtoD from post output (stays on GPU) ──
            let output_tid = post_graph.outputs[0];
            let output_ptr = post_bufs.iter().find(|(tid, _)| *tid == output_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Cuda("post output buffer not found".into()))?;
            let res = unsafe {
                (device.driver().cuMemcpyDtoD_v2)(
                    gpu_hidden.as_device_ptr(),
                    output_ptr,
                    hidden_bytes_typed,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD post output→gpu_hidden failed: CUDA error {res}")));
            }
        }

        // ── Final: download hidden_state from GPU (once, after all layers) ──
        let mut output_host = vec![0u8; hidden_bytes_typed];
        device.dtoh(&gpu_hidden, &mut output_host, stream)
            .map_err(|e| BE::Cuda(format!("dtoh final hidden: {e}")))?;
        hidden_state.copy_from_slice(&output_host);
    } else {
        // ── Prefill path (GPU-resident hidden_state) ──
        let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
        let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta, comp_dtype);
        let (_module, kernel_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &graph)?;

        // Upload hidden_state to GPU once before layer loop
        let hidden_bytes_typed = seq_len * hidden * elem_bytes;
        let elem_bytes = comp_dtype.size_bytes();
        let mut gpu_hidden = device.alloc(hidden_bytes_typed)
            .map_err(|e| BE::Cuda(format!("GPU alloc gpu_hidden failed: {e}")))?;
        {
            let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr(), hidden_bytes_typed) };
            device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Cuda(format!("htod initial hidden: {e}")))?;
        }

        // ── Pre-load all layer weights (CPU side, once) ──
        let all_layer_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
            .map(|layer| super::gpu_helpers::load_decoder_layer_weights_gpu_typed(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter, comp_dtype,
            ))
            .collect::<Result<Vec<_>, _>>()?;

        for layer in 0..num_layers {
            let layer_weights = &all_layer_weights[layer];

            let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
            let mut tensor_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

            for (idx, meta) in graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let size_bytes = meta.concrete_numel() * elem_bytes;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Cuda(format!("GPU alloc failed for {}: {e}", meta.name)))?;

                if meta.name == "input" {
                    // DtoD from gpu_hidden (no CPU round-trip)
                    let res = unsafe {
                        (device.driver().cuMemcpyDtoD_v2)(
                            buf.as_device_ptr(),
                            gpu_hidden.as_device_ptr(),
                            hidden_bytes_typed,
                        )
                    };
                    if res != 0 {
                        return Err(BE::Other(format!("DtoD hidden→input failed: CUDA error {res}")));
                    }
                } else if let Some(data) = layer_weights.get(&meta.name) {
                    device.htod(data, &mut buf, stream)
                        .map_err(|e| BE::Cuda(format!("htod {} failed: {e}", meta.name)))?;
                }

                tensor_ptrs.insert(tid, buf.as_device_ptr());
                gpu_buffers.push((tid, buf));
            }

            cuda_launch_graph(device, stream, &kernel_entries, &tensor_ptrs, &graph)?;
            device.sync().map_err(|e| BE::Cuda(format!("GPU sync failed: {e}")))?;

            // ── Write K/V to KV cache ──
            if let Some(handle) = kv_caches.first() {
                let kv_dim = num_kv_heads * head_dim;

                let k_tid = graph.tensors.iter().enumerate()
                    .find(|(_, m)| m.name == "k_rope")
                    .map(|(i, _)| TensorId(i as u32))
                    .ok_or_else(|| BE::Cuda("k_rope tensor not found in graph".into()))?;
                let v_tid = graph.tensors.iter().enumerate()
                    .find(|(_, m)| m.name == "v_proj")
                    .map(|(i, _)| TensorId(i as u32))
                    .ok_or_else(|| BE::Cuda("v_proj tensor not found in graph".into()))?;

                let k_ptr = gpu_buffers.iter().find(|(t, _)| *t == k_tid).map(|(_, b)| b.as_device_ptr())
                    .ok_or_else(|| BE::Cuda("k_rope buffer not found".into()))?;
                let v_ptr = gpu_buffers.iter().find(|(t, _)| *t == v_tid).map(|(_, b)| b.as_device_ptr())
                    .ok_or_else(|| BE::Cuda("v_proj buffer not found".into()))?;

                let meta_store = backend.kv_meta.lock()
                    .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
                let meta = meta_store.get(&handle.0)
                    .ok_or_else(|| BE::Cuda(format!("KV cache handle {} not found", handle.0)))?;

                let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let write_start = if layer == 0 { meta.seq_len } else { meta.seq_len.saturating_sub(seq_len) };
                let kv_ds = meta.dtype_size;

                drop(meta_store);

                // DtoD: write K/V directly from GPU buffers to KV cache (no host bounce)
                gpu_write_kv_cache_device(
                    k_ptr, v_ptr, kv_dim, seq_len, num_kv_heads, head_dim,
                    handle, layer, half_bytes, write_start, head_stride, kv_ds, device, stream,
                    gpu_profile, sm_version,
                )?;

                if layer == 0 {
                    let mut meta_store = backend.kv_meta.lock()
                        .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
                    if let Some(meta) = meta_store.get_mut(&handle.0) {
                        meta.seq_len = (meta.seq_len + seq_len).min(meta.max_seq_len);
                    }
                }
            }

            // Update gpu_hidden: DtoD from output (stays on GPU)
            let output_tid = graph.outputs[0];
            let output_ptr = gpu_buffers.iter().find(|(tid, _)| *tid == output_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Cuda("output buffer not found".into()))?;
            let res = unsafe {
                (device.driver().cuMemcpyDtoD_v2)(
                    gpu_hidden.as_device_ptr(),
                    output_ptr,
                    hidden_bytes_typed,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD output→gpu_hidden failed: CUDA error {res}")));
            }
        }

        // Final: download hidden_state from GPU (once)
        {
            let mut output_host = vec![0u8; hidden_bytes_typed];
            device.dtoh(&gpu_hidden, &mut output_host, stream)
                .map_err(|e| BE::Cuda(format!("dtoh final hidden: {e}")))?;
            hidden_state.copy_from_slice(&output_host);
        }
    }

    // ── lm_head projection (GPU) ──
    let lm_comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let lm_elem_bytes = lm_comp_dtype.size_bytes();
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size, lm_comp_dtype);
    let (_lm_module, lm_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &lm_graph)?;

    let lm_head_w = super::gpu_helpers::get_typed_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("lm_head.weight", Some("output.weight")),
        lm_comp_dtype)?;

    let mut gpu_buffers_lm: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
    let mut lm_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

    for (idx, meta) in lm_graph.tensors.iter().enumerate() {
        let tid = TensorId(idx as u32);
        let n_elements = meta.concrete_numel();
        let size_bytes = n_elements * lm_elem_bytes;
        let mut buf = device.alloc(size_bytes)
            .map_err(|e| BE::Cuda(format!("GPU alloc lm_head {} failed: {e}", meta.name)))?;

        if meta.name == "input" {
            device.htod(&hidden_state, &mut buf, stream)
                .map_err(|e| BE::Cuda(format!("htod lm_head input: {e}")))?;
        } else if meta.name == "w_lm" {
            device.htod(&lm_head_w, &mut buf, stream)
                .map_err(|e| BE::Cuda(format!("htod lm_head weight: {e}")))?;
        }

        lm_ptrs.insert(tid, buf.as_device_ptr());
        gpu_buffers_lm.push((tid, buf));
    }

    cuda_launch_graph(device, stream, &lm_entries, &lm_ptrs, &lm_graph)?;
    device.sync().map_err(|e| BE::Cuda(format!("GPU sync lm_head: {e}")))?;

    // Download logits
    let logits_tid = lm_graph.outputs[0];
    let logits_buf = gpu_buffers_lm.iter()
        .find(|(tid, _)| *tid == logits_tid)
        .map(|(_, buf)| buf)
        .ok_or_else(|| BE::Cuda("logits buffer not found".into()))?;
    let logits_bytes = seq_len * vocab_size * lm_elem_bytes;
    let mut logits_host = vec![0u8; logits_bytes];
    device.dtoh(logits_buf, &mut logits_host, stream)
        .map_err(|e| BE::Cuda(format!("dtoh logits: {e}")))?;

    let logits_f32 = super::jit_helpers::typed_bytes_to_f32(&logits_host, lm_comp_dtype);

    // Split logits per sequence
    let mut results = Vec::new();
    let mut offset = 0;
    for seq in &input.sequences {
        let tok_count = seq.tokens.len();
        let last_pos = offset + tok_count - 1;
        let seq_logits = logits_f32[last_pos * vocab_size..(last_pos + 1) * vocab_size].to_vec();
        results.push(LogitsHandle { data: seq_logits });
        offset += tok_count;
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// GPU compilation & kernel launch helpers (HIP / ROCm)
// ---------------------------------------------------------------------------

/// Compile a CompilerGraph to HIP C++ source via the full JIT pipeline.
///
/// Pipeline: CompilerGraph → ScalarOpRegistry → SemanticDAG → FusionPlan → HIP C++
///
/// Returns (HIP C++ source string, FusionPlan).
#[cfg(feature = "hip")]
pub(crate) fn compile_graph_to_hip(
    gfx_arch: u32,
    graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<(String, gllm_kernels::compiler::FusionPlan), BE> {
    use gllm_kernels::compiler::{
        ScalarOpRegistry, SemanticDAG,
        fusion,
        codegen::gpu_ir::{GpuDialect, HipDialect, gpu_emit_plan},
    };

    let registry = ScalarOpRegistry::with_defaults();
    let dag = SemanticDAG::from_graph(graph, &registry);
    let profile = gllm_kernels::dispatch::DeviceProfile::detect();
    let plan = fusion::fuse_with_dag_prebuilt(graph, &dag, &profile);

    // Extract dtype from graph ops
    let graph_dtype = graph.ops.iter()
        .find_map(|op| match &op.kind {
            gllm_kernels::compiler::OpKind::Gemm { dtype, .. }
            | gllm_kernels::compiler::OpKind::GemmBias { dtype, .. } => Some(*dtype),
            gllm_kernels::compiler::OpKind::CachedGQA { kv_dtype, .. } => Some(*kv_dtype),
            _ => None,
        })
        .unwrap_or(gllm_kernels::types::DType::F32);

    let dialect = HipDialect::with_dtype(gfx_arch, 1024, 65536, graph_dtype);
    let mut hip_src = String::new();
    dialect.emit_header(&mut hip_src);
    gpu_emit_plan(&dialect, &mut hip_src, &plan, graph, Some(&registry))
        .map_err(|e| BE::Other(format!("HIP emission failed: {e}")))?;

    Ok((hip_src, plan))
}

/// HIP kernel entry — mirrors GpuKernelEntry but uses HIP function handle.
#[cfg(feature = "hip")]
pub(crate) struct HipKernelEntry {
    pub(crate) func: gllm_kernels::gpu::hip::driver::HipFunction,
    pub(crate) config: gllm_kernels::gpu::LaunchConfig,
    pub(crate) op_kind: gllm_kernels::compiler::OpKind,
    pub(crate) input_tids: Vec<gllm_kernels::compiler::TensorId>,
    pub(crate) output_tid: gllm_kernels::compiler::TensorId,
}

/// Compile a CompilerGraph to a loaded HIP module with kernel metadata.
///
/// Pipeline: CompilerGraph → HIP C++ → hiprtc → load module → extract kernels.
#[cfg(feature = "hip")]
pub(crate) fn hip_compile_graph(
    device: &gllm_kernels::gpu::hip::HipDevice,
    gpu_profile: &gllm_kernels::gpu::GpuDeviceProfile,
    gfx_arch: u32,
    graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<(gllm_kernels::gpu::hip::HipModule, Vec<HipKernelEntry>), BE> {
    use gllm_kernels::compiler::OpKind;

    let (hip_src, plan) = compile_graph_to_hip(gfx_arch, graph)?;

    let code_bytes = device.compile_hip_source(&hip_src, "gllm_kernel.hip")
        .map_err(|e| BE::Other(format!("hiprtc compile failed: {e}")))?;

    let module = device.load_module(&code_bytes)
        .map_err(|e| BE::Other(format!("hipModuleLoadData failed: {e}")))?;

    let mut entries = Vec::new();
    for group in &plan.groups {
        let anchor = graph.op(group.anchor)
            .ok_or_else(|| BE::Other(format!("Missing anchor op {:?}", group.anchor)))?;

        if matches!(anchor.kind, OpKind::Reshape { .. } | OpKind::Transpose { .. }) {
            continue;
        }

        let name = format!("group_{}", group.id);
        let func = module.get_function(&name)
            .map_err(|e| BE::Other(format!("hipModuleGetFunction({name}) failed: {e}")))?;
        let config = launch_config_for_op(gpu_profile, &anchor.kind);

        let input_tids = anchor.inputs.clone();
        let output_tid = if !group.epilogue.is_empty() {
            let last_epi = *group.epilogue.last()
                .ok_or_else(|| BE::Other("empty epilogue in fusion group".into()))?;
            let last_op = graph.op(last_epi)
                .ok_or_else(|| BE::Other(format!("missing epilogue op {:?}", last_epi)))?;
            last_op.outputs[0]
        } else {
            anchor.outputs[0]
        };

        entries.push(HipKernelEntry {
            func,
            config,
            op_kind: anchor.kind.clone(),
            input_tids,
            output_tid,
        });
    }

    Ok((module, entries))
}

/// Launch all kernels for a compiled HIP graph.
#[cfg(feature = "hip")]
pub(crate) fn hip_launch_graph(
    device: &gllm_kernels::gpu::hip::HipDevice,
    stream: &gllm_kernels::gpu::hip::HipGpuStream,
    entries: &[HipKernelEntry],
    tensor_ptrs: &std::collections::HashMap<gllm_kernels::compiler::TensorId, u64>,
    graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<(), BE> {
    use std::ffi::c_void;
    use gllm_kernels::compiler::OpKind;

    for entry in entries {
        let input_ptrs: Vec<u64> = entry.input_tids.iter()
            .map(|tid| tensor_ptrs.get(tid).copied().ok_or_else(|| {
                BE::Cuda(format!("missing tensor pointer for input {:?}", tid))
            }))
            .collect::<Result<Vec<_>, _>>()?;
        let output_ptr = tensor_ptrs.get(&entry.output_tid).copied().ok_or_else(|| {
            BE::Cuda(format!("missing tensor pointer for output {:?}", entry.output_tid))
        })?;

        let mut raw_params = build_kernel_params(&entry.op_kind, &input_ptrs, output_ptr)?;

        match &entry.op_kind {
            OpKind::LayerNorm { .. } | OpKind::RmsNorm { .. } => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for LayerNorm/RmsNorm input {:?}", entry.input_tids[0])))?;
                let n = input_meta.shape.last()
                    .and_then(|d| d.as_concrete())
                    .unwrap_or(1);
                raw_params[2] = n as u64;
            }
            OpKind::Residual | OpKind::Add | OpKind::Mul => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for binary op input {:?}", entry.input_tids[0])))?;
                let n = input_meta.concrete_numel();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::Silu | OpKind::Gelu => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for activation input {:?}", entry.input_tids[0])))?;
                let n = input_meta.concrete_numel();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::SwiGlu | OpKind::GeGlu => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for SwiGlu/GeGlu input {:?}", entry.input_tids[0])))?;
                let n = input_meta.concrete_numel();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::RoPE { .. } => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for RoPE input {:?}", entry.input_tids[0])))?;
                let seq_len = input_meta.shape[0].as_concrete().unwrap_or(1);
                let last = raw_params.len() - 1;
                raw_params[last] = seq_len as u64;
            }
            OpKind::Softmax => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .ok_or_else(|| BE::Other(format!("tensor meta for Softmax input {:?}", entry.input_tids[0])))?;
                let n = input_meta.concrete_numel();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            _ => {}
        }

        let mut param_ptrs: Vec<*mut c_void> = raw_params.iter_mut()
            .map(|p| p as *mut u64 as *mut c_void)
            .collect();

        let cfg = &entry.config;
        device.launch_kernel(entry.func, &entry.config, stream, &mut param_ptrs)
            .map_err(|e| BE::Other(format!(
                "HIP kernel launch failed for {:?}: {e}", entry.op_kind
            )))?;
    }

    Ok(())
}

/// Full BERT encoder forward on HIP GPU.
///
/// Mirrors `cuda_bert_encoder_forward` but uses HIP APIs.
#[cfg(feature = "hip")]
pub(super) fn hip_bert_encoder_forward<E: Element>(
    backend: &HipBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, HipBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    use gllm_kernels::gpu::{GpuDevice, GpuBuffer};
    use gllm_kernels::compiler::TensorId;
    use super::bert_forward::build_bert_layer_graph;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("BERT encoder only supports f32 element type".into()));
    }

    let device = &*backend.device;
    let stream = device.default_stream();
    let gpu_profile = &backend.gpu_profile;
    let gfx_arch = device.gfx_arch();

    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let transpose_w = super::gpu_helpers::needs_weight_transpose_gpu(weights);

    let mut hidden_state = super::gpu_helpers::embed_tokens_gpu(tokens, weights, backend, config)?;

    let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let elem_bytes = comp_dtype.size_bytes();
    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps, comp_dtype);
    let (_module, kernel_entries) = hip_compile_graph(
        device, gpu_profile, gfx_arch, &graph,
    )?;

    let elem_bytes = comp_dtype.size_bytes();
    let hidden_bytes = seq_len * hidden * elem_bytes;
    let mut gpu_hidden = device.alloc(hidden_bytes)
        .map_err(|e| BE::Other(format!("GPU alloc gpu_hidden failed: {e}")))?;
    {
        let bytes = &hidden_state[..hidden_bytes];
        device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Other(format!("htod initial hidden: {e}")))?;
    }

    // ── Pre-load all layer weights (CPU side, once) ──
    let all_bert_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
        .map(|layer| super::gpu_helpers::load_bert_layer_weights_gpu_typed(
            weights, backend, layer, seq_len, hidden, inter, transpose_w, comp_dtype,
        ))
        .collect::<Result<Vec<_>, _>>()?;

    for layer in 0..num_layers {
        let layer_weights = &all_bert_weights[layer];

        let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
        let mut tensor_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

        for (idx, meta) in graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let size_bytes = meta.concrete_numel() * elem_bytes;
            let mut buf = device.alloc(size_bytes)
                .map_err(|e| BE::Other(format!("GPU alloc failed for {}: {e}", meta.name)))?;

            if meta.name == "input" {
                let res = unsafe {
                    (device.driver().hipMemcpyDtoD)(
                        buf.as_device_ptr() as _,
                        gpu_hidden.as_device_ptr() as _,
                        hidden_bytes_typed,
                    )
                };
                if res != 0 {
                    return Err(BE::Other(format!("DtoD hidden→input failed: HIP error {res}")));
                }
            } else if let Some(data) = layer_weights.get(&meta.name) {
                device.htod(data, &mut buf, stream)
                    .map_err(|e| BE::Other(format!("htod {} failed: {e}", meta.name)))?;
            }

            tensor_ptrs.insert(tid, buf.as_device_ptr());
            gpu_buffers.push((tid, buf));
        }

        hip_launch_graph(device, stream, &kernel_entries, &tensor_ptrs, &graph)?;
        device.sync()
            .map_err(|e| BE::Other(format!("GPU sync failed: {e}")))?;

        let output_tid = graph.outputs[0];
        let output_ptr = gpu_buffers.iter().find(|(tid, _)| *tid == output_tid).map(|(_, b)| b.as_device_ptr())
            .ok_or_else(|| BE::Other("output buffer not found".into()))?;
        let res = unsafe {
            (device.driver().hipMemcpyDtoD)(
                gpu_hidden.as_device_ptr() as _,
                output_ptr as _,
                hidden_bytes_typed,
            )
        };
        if res != 0 {
            return Err(BE::Other(format!("DtoD output→gpu_hidden failed: HIP error {res}")));
        }
    }

    // ── Mean pooling (GPU) — gpu_hidden already on device, no CPU round-trip ──
    let pool_graph = build_mean_pool_graph(seq_len, hidden, super::jit_helpers::computation_dtype_from_config(config));
    let (_pool_module, pool_entries) = hip_compile_graph(
        device, gpu_profile, gfx_arch, &pool_graph,
    )?;

    let output_bytes = hidden * elem_bytes;
    let gpu_output = device.alloc(output_bytes)
        .map_err(|e| BE::Other(format!("GPU alloc pool output: {e}")))?;

    let mut pool_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
    pool_ptrs.insert(pool_graph.inputs[0], gpu_hidden.as_device_ptr());
    pool_ptrs.insert(pool_graph.outputs[0], gpu_output.as_device_ptr());

    hip_launch_graph(device, stream, &pool_entries, &pool_ptrs, &pool_graph)?;
    device.sync().map_err(|e| BE::Other(format!("GPU sync pool: {e}")))?;

    let mut pooled_bytes = vec![0u8; output_bytes];
    device.dtoh(&gpu_output, &mut pooled_bytes, stream)
        .map_err(|e| BE::Other(format!("dtoh pool output: {e}")))?;

    let pooled = super::jit_helpers::typed_bytes_to_f32(&pooled_bytes, comp_dtype);

    Ok(pooled)
}

// ===========================================================================
// Metal GPU compilation & kernel launch helpers
// ===========================================================================

/// Metal KV cache 直写：利用 shared memory 直接通过指针写入，消除 dtoh→Vec→htod 三步路径。
/// Metal buffer 的 as_device_ptr() 返回 CPU 可见指针（[buffer contents]），无需 dtoh/htod。
#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_write_kv_direct(
    k_buf_ptr: u64,
    v_buf_ptr: u64,
    kv_cache_handle: &KvCacheHandle,
    kv_dim: usize,
    seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    layer: usize,
    half_bytes: usize,
    write_start: usize,
    head_stride: usize,
    dtype_size: usize,
) -> Result<(), BE> {
    let k_src = k_buf_ptr as *const u8;
    let v_src = v_buf_ptr as *const u8;
    let kv_dst = kv_cache_handle.0 as *mut u8;

    for head in 0..num_kv_heads {
        let dst_k_base = unsafe {
            kv_dst.add(
                (layer * num_kv_heads + head) * head_stride
                    + write_start * head_dim * dtype_size,
            )
        };
        let dst_v_base = unsafe {
            kv_dst.add(
                half_bytes
                    + (layer * num_kv_heads + head) * head_stride
                    + write_start * head_dim * dtype_size,
            )
        };

        for s in 0..seq_len {
            let src_off = (s * kv_dim + head * head_dim) * dtype_size;
            let dst_off = s * head_dim * dtype_size;
            let copy_bytes = head_dim * dtype_size;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    k_src.add(src_off),
                    dst_k_base.add(dst_off),
                    copy_bytes,
                );
                std::ptr::copy_nonoverlapping(
                    v_src.add(src_off),
                    dst_v_base.add(dst_off),
                    copy_bytes,
                );
            }
        }
    }
    Ok(())
}

/// Compile a CompilerGraph to MSL source via the full JIT pipeline.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn compile_graph_to_msl(
    gpu_family: u32,
    graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<(String, gllm_kernels::compiler::FusionPlan), BE> {
    use gllm_kernels::compiler::{
        ScalarOpRegistry, SemanticDAG,
        fusion,
        codegen::gpu_ir::{GpuDialect, MslDialect, gpu_emit_plan},
    };

    let registry = ScalarOpRegistry::with_defaults();
    let dag = SemanticDAG::from_graph(graph, &registry);
    let profile = gllm_kernels::dispatch::DeviceProfile::detect();
    let plan = fusion::fuse_with_dag_prebuilt(graph, &dag, &profile);

    // Extract dtype from graph ops (same pattern as PTX)
    let graph_dtype = graph.ops.iter()
        .find_map(|op| match &op.kind {
            gllm_kernels::compiler::OpKind::Gemm { dtype, .. }
            | gllm_kernels::compiler::OpKind::GemmBias { dtype, .. } => Some(*dtype),
            gllm_kernels::compiler::OpKind::CachedGQA { kv_dtype, .. } => Some(*kv_dtype),
            _ => None,
        })
        .unwrap_or(gllm_kernels::types::DType::F32);

    let dialect = MslDialect::with_dtype(gpu_family, 1024, 32768, graph_dtype);
    let mut msl = String::new();
    dialect.emit_header(&mut msl);
    gpu_emit_plan(&dialect, &mut msl, &plan, graph, Some(&registry))
        .map_err(|e| BE::Other(format!("MSL emission failed: {e}")))?;

    Ok((msl, plan))
}

/// Kernel entry for a Metal compute pipeline.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) struct MetalKernelEntry {
    pub(crate) pipeline: objc_runtime::Id,
    pub(crate) config: gllm_kernels::gpu::LaunchConfig,
    pub(crate) op_kind: gllm_kernels::compiler::OpKind,
    pub(crate) input_tids: Vec<gllm_kernels::compiler::TensorId>,
    pub(crate) output_tid: gllm_kernels::compiler::TensorId,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
use gllm_kernels::gpu::metal::objc_runtime;

/// Compile a CompilerGraph to Metal pipelines.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn metal_compile_graph(
    device: &gllm_kernels::gpu::metal::MetalDevice,
    gpu_profile: &gllm_kernels::gpu::GpuDeviceProfile,
    gpu_family: u32,
    graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<Vec<MetalKernelEntry>, BE> {
    use gllm_kernels::compiler::OpKind;

    let (msl_source, plan) = compile_graph_to_msl(gpu_family, graph)?;

    // Compile MSL to library
    let library = device.compile_msl(&msl_source)
        .map_err(|e| BE::Metal(format!("MSL compilation failed: {e}")))?;

    let mut entries = Vec::new();
    for group in &plan.groups {
        let anchor = graph.op(group.anchor)
            .ok_or_else(|| BE::Metal(format!("Missing anchor op {:?}", group.anchor)))?;

        if matches!(anchor.kind, OpKind::Reshape { .. } | OpKind::Transpose { .. }) {
            continue;
        }

        let name = format!("group_{}", group.id);
        let func = device.get_function(library, &name)
            .map_err(|e| BE::Metal(format!("get_function({name}) failed: {e}")))?;
        let pipeline = device.create_compute_pipeline(func)
            .map_err(|e| BE::Metal(format!("create_pipeline({name}) failed: {e}")))?;
        let config = launch_config_for_op(gpu_profile, &anchor.kind);

        let input_tids = anchor.inputs.clone();
        let output_tid = if !group.epilogue.is_empty() {
            let last_epi = *group.epilogue.last()
                .ok_or_else(|| BE::Metal("empty epilogue in fusion group".into()))?;
            let last_op = graph.op(last_epi)
                .ok_or_else(|| BE::Metal(format!("missing epilogue op {:?}", last_epi)))?;
            last_op.outputs[0]
        } else {
            anchor.outputs[0]
        };

        entries.push(MetalKernelEntry {
            pipeline,
            config,
            op_kind: anchor.kind.clone(),
            input_tids,
            output_tid,
        });
    }

    Ok(entries)
}

/// Launch all Metal kernels for a compiled graph.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) fn metal_launch_graph(
    device: &gllm_kernels::gpu::metal::MetalDevice,
    stream: &gllm_kernels::gpu::metal::MetalStream,
    entries: &[MetalKernelEntry],
    tensor_bufs: &std::collections::HashMap<gllm_kernels::compiler::TensorId, gllm_kernels::gpu::metal::MetalBuffer>,
    _graph: &gllm_kernels::compiler::CompilerGraph,
) -> Result<(), BE> {
    for entry in entries {
        // Collect buffer references for this kernel
        let mut bufs: Vec<&gllm_kernels::gpu::metal::MetalBuffer> = Vec::new();
        for tid in &entry.input_tids {
            let buf = tensor_bufs.get(tid)
                .ok_or_else(|| BE::Metal(format!("input buffer {:?} not found", tid)))?;
            bufs.push(buf);
        }
        let out_buf = tensor_bufs.get(&entry.output_tid)
            .ok_or_else(|| BE::Metal(format!("output buffer {:?} not found", entry.output_tid)))?;
        bufs.push(out_buf);

        let grid = [entry.config.grid_dim[0], entry.config.grid_dim[1], entry.config.grid_dim[2]];
        let tg = [entry.config.block_dim[0], entry.config.block_dim[1], entry.config.block_dim[2]];

        device.dispatch_compute(stream, entry.pipeline, grid, tg, &bufs)
            .map_err(|e| BE::Metal(format!("dispatch failed for {:?}: {e}", entry.op_kind)))?;
    }
    Ok(())
}

/// Full BERT encoder forward on Metal GPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn metal_bert_encoder_forward<E: Element>(
    backend: &super::metal_backend::MetalBackend<E>,
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, super::metal_backend::MetalBackend<E>>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    use gllm_kernels::gpu::{GpuDevice, GpuBuffer};
    use gllm_kernels::compiler::TensorId;
    use super::bert_forward::build_bert_layer_graph;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("BERT encoder only supports f32 element type".into()));
    }

    let device = backend.device();
    let stream = device.default_stream();
    let gpu_profile = backend.gpu_profile();

    use gllm_kernels::compiler::codegen::emitter::Platform;
    let gpu_family = match gpu_profile.platform {
        Platform::Metal { gpu_family } => gpu_family,
        other => {
            return Err(GpuCompileError::Unsupported(format!(
                "expected Metal platform for Metal backend, got {:?}", other
            )));
        }
    };

    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let transpose_w = super::gpu_helpers::needs_weight_transpose_gpu(weights);

    // Embedding lookup on CPU
    let mut hidden_state = super::gpu_helpers::embed_tokens_gpu(tokens, weights, backend, config)?;

    let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let elem_bytes = comp_dtype.size_bytes();

    // Compile BERT layer graph to Metal pipelines (once)
    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps, comp_dtype);
    let kernel_entries = metal_compile_graph(device, gpu_profile, gpu_family, &graph)?;

    let elem_bytes = comp_dtype.size_bytes();
    let hidden_bytes = seq_len * hidden * elem_bytes;
    let mut gpu_hidden = device.alloc(hidden_bytes)
        .map_err(|e| BE::Metal(format!("alloc gpu_hidden failed: {e}")))?;
    {
        let bytes = &hidden_state[..hidden_bytes];
        device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Metal(format!("htod initial hidden: {e}")))?;
    }

    // ── Pre-load all layer weights (CPU side, once) ──
    let all_bert_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
        .map(|layer| super::gpu_helpers::load_bert_layer_weights_gpu_typed(
            weights, backend, layer, seq_len, hidden, inter, transpose_w, comp_dtype,
        ))
        .collect::<Result<Vec<_>, _>>()?;

    // Per-layer GPU execution
    for layer in 0..num_layers {
        let layer_weights = &all_bert_weights[layer];

        let mut gpu_buffers: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
            std::collections::HashMap::new();

        for (idx, meta) in graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let size_bytes = meta.concrete_numel() * elem_bytes;
            let mut buf = device.alloc(size_bytes)
                .map_err(|e| BE::Metal(format!("alloc failed for {}: {e}", meta.name)))?;

            if meta.name == "input" {
                device.dtod(&gpu_hidden, &mut buf, stream)
                    .map_err(|e| BE::Metal(format!("dtod hidden→input: {e}")))?;
            } else if let Some(data) = layer_weights.get(&meta.name) {
                device.htod(data, &mut buf, stream)
                    .map_err(|e| BE::Metal(format!("htod {} failed: {e}", meta.name)))?;
            }

            gpu_buffers.insert(tid, buf);
        }

        metal_launch_graph(device, stream, &kernel_entries, &gpu_buffers, &graph)?;
        device.sync().map_err(|e| BE::Metal(format!("sync failed: {e}")))?;

        let output_tid = graph.outputs[0];
        let output_buf = gpu_buffers.get(&output_tid)
            .ok_or_else(|| BE::Metal("output buffer not found".into()))?;
        device.dtod(output_buf, &mut gpu_hidden, stream)
            .map_err(|e| BE::Metal(format!("dtod output→gpu_hidden: {e}")))?;
    }

    // ── Mean pooling (GPU) — gpu_hidden already on device, no CPU round-trip ──
    let pool_graph = super::bert_forward::build_mean_pool_graph(seq_len, hidden, comp_dtype);
    let pool_entries = metal_compile_graph(device, gpu_profile, gpu_family, &pool_graph)?;

    let gpu_output = device.alloc(hidden * elem_bytes)
        .map_err(|e| BE::Metal(format!("alloc pool output: {e}")))?;

    let mut pool_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
        std::collections::HashMap::new();
    pool_bufs.insert(pool_graph.inputs[0], gpu_hidden);
    pool_bufs.insert(pool_graph.outputs[0], gpu_output);

    metal_launch_graph(device, stream, &pool_entries, &pool_bufs, &pool_graph)?;
    device.sync().map_err(|e| BE::Metal(format!("sync pool: {e}")))?;

    let output_buf = pool_bufs.get(&pool_graph.outputs[0])
        .ok_or_else(|| BE::Metal("pool output buffer not found".into()))?;
    let mut pooled_bytes = vec![0u8; hidden * elem_bytes];
    device.dtoh(output_buf, &mut pooled_bytes, stream)
        .map_err(|e| BE::Metal(format!("dtoh pool output: {e}")))?;

    let pooled = super::jit_helpers::typed_bytes_to_f32(&pooled_bytes, comp_dtype);

    Ok(pooled)
}

// ===========================================================================
// Decoder (generator) forward pass — shared decoder layer graph + per-backend
// ===========================================================================

/// Build a CompilerGraph for a single decoder layer (causal attention + SwiGLU FFN).
///
/// The graph is hardware-agnostic; each backend compiles it to its own ISA.
/// Layout:
///   input[seq, hidden]
///   -> RmsNorm -> Q/K/V Gemm -> MultiHeadAttention -> Out Gemm -> Residual
///   -> RmsNorm -> Gate Gemm -> Up Gemm -> SwiGLU -> Down Gemm -> Residual
///   -> output[seq, hidden]
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) fn build_decoder_layer_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    rope_theta: f64,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_h = num_kv_heads * head_dim;

    // ── Graph inputs ──
    let input = g.add_tensor_concrete("input", &[s, h], dt);

    // Attention weights (q_dim may differ from hidden for Qwen3 etc.)
    let attn_norm_w = g.add_tensor_concrete("attn_norm_w", &[h], dt);
    let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_h], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_h], dt);
    let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);

    // FFN weights
    let ffn_norm_w = g.add_tensor_concrete("ffn_norm_w", &[h], dt);
    let w_gate = g.add_tensor_concrete("w_gate", &[h, inter], dt);
    let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
    let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);

    g.inputs = vec![input];

    // ── Pre-attention RmsNorm ──
    let attn_normed = g.add_tensor_concrete("attn_normed", &[s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![input, attn_norm_w],
        vec![attn_normed],
        "attn_rms_norm",
    );

    // ── Q/K/V projections ──
    let q_proj = g.add_tensor_concrete("q_proj", &[s, q_dim], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: q_dim, k: h, dtype: dt },
        vec![attn_normed, w_q],
        vec![q_proj],
        "q_proj",
    );

    let k_proj = g.add_tensor_concrete("k_proj", &[s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h, dtype: dt },
        vec![attn_normed, w_k],
        vec![k_proj],
        "k_proj",
    );

    let v_proj = g.add_tensor_concrete("v_proj", &[s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h, dtype: dt },
        vec![attn_normed, w_v],
        vec![v_proj],
        "v_proj",
    );

    // ── RoPE on Q and K ──
    let q_rope = g.add_tensor_concrete("q_rope", &[s, q_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![q_proj],
        vec![q_rope],
        "rope_q",
    );

    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_h], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![k_proj],
        vec![k_rope],
        "rope_k",
    );

    // ── Causal MultiHeadAttention ──
    let attn_out = g.add_tensor_concrete("attn_out", &[s, q_dim], dt);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len: s,
            num_heads,
            head_dim,
        },
        vec![q_rope, k_rope, v_proj],
        vec![attn_out],
        "causal_attention",
    );

    // ── Output projection ──
    let o_proj = g.add_tensor_concrete("o_proj", &[s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: q_dim, dtype: dt },
        vec![attn_out, w_o],
        vec![o_proj],
        "o_proj",
    );

    // ── Attention residual ──
    let attn_residual = g.add_tensor_concrete("attn_residual", &[s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![input, o_proj],
        vec![attn_residual],
        "attn_residual",
    );

    // ── Pre-FFN RmsNorm ──
    let ffn_normed = g.add_tensor_concrete("ffn_normed", &[s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![attn_residual, ffn_norm_w],
        vec![ffn_normed],
        "ffn_rms_norm",
    );

    // ── Gate + Up projections ──
    let gate = g.add_tensor_concrete("gate", &[s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h, dtype: dt },
        vec![ffn_normed, w_gate],
        vec![gate],
        "gate_proj",
    );

    let up = g.add_tensor_concrete("up", &[s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h, dtype: dt },
        vec![ffn_normed, w_up],
        vec![up],
        "up_proj",
    );

    // ── SwiGLU: silu(gate) * up ──
    let swiglu_out = g.add_tensor_concrete("swiglu_out", &[s, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate, up], vec![swiglu_out], "swiglu");

    // ── Down projection ──
    let down = g.add_tensor_concrete("down", &[s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: inter, dtype: dt },
        vec![swiglu_out, w_down],
        vec![down],
        "down_proj",
    );

    // ── FFN residual ──
    let output = g.add_tensor_concrete("output", &[s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![attn_residual, down],
        vec![output],
        "ffn_residual",
    );

    g.outputs = vec![output];
    g
}

/// Build a projection sub-graph for incremental decode:
/// input[seq, hidden] → RmsNorm → Q/K/V Gemm → RoPE → outputs: [q_rope, k_rope, v_proj]
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) fn build_projection_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f64,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_h = num_kv_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], dt);
    let attn_norm_w = g.add_tensor_concrete("attn_norm_w", &[h], dt);
    let w_q = g.add_tensor_concrete("w_q", &[h, q_dim], dt);
    let w_k = g.add_tensor_concrete("w_k", &[h, kv_h], dt);
    let w_v = g.add_tensor_concrete("w_v", &[h, kv_h], dt);

    g.inputs = vec![input];

    let attn_normed = g.add_tensor_concrete("attn_normed", &[s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![input, attn_norm_w],
        vec![attn_normed],
        "attn_rms_norm",
    );

    let q_proj = g.add_tensor_concrete("q_proj", &[s, q_dim], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: q_dim, k: h, dtype: dt },
        vec![attn_normed, w_q],
        vec![q_proj],
        "q_proj",
    );

    let k_proj = g.add_tensor_concrete("k_proj", &[s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h, dtype: dt },
        vec![attn_normed, w_k],
        vec![k_proj],
        "k_proj",
    );

    let v_proj = g.add_tensor_concrete("v_proj", &[s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h, dtype: dt },
        vec![attn_normed, w_v],
        vec![v_proj],
        "v_proj",
    );

    let q_rope = g.add_tensor_concrete("q_rope", &[s, q_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![q_proj],
        vec![q_rope],
        "rope_q",
    );

    let k_rope = g.add_tensor_concrete("k_rope", &[s, kv_h], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![k_proj],
        vec![k_rope],
        "rope_k",
    );

    g.outputs = vec![q_rope, k_rope, v_proj];
    g
}

/// Build a post-attention sub-graph for incremental decode:
/// (input[seq, hidden], attn_out[seq, q_dim]) → O Gemm → Residual → RmsNorm → FFN → Residual → output
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) fn build_post_attention_graph(
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    head_dim: usize,
    inter: usize,
    eps: f32,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;

    let input = g.add_tensor_concrete("input", &[s, h], dt);
    let attn_out = g.add_tensor_concrete("attn_out", &[s, q_dim], dt);
    let w_o = g.add_tensor_concrete("w_o", &[q_dim, h], dt);
    let ffn_norm_w = g.add_tensor_concrete("ffn_norm_w", &[h], dt);
    let w_gate = g.add_tensor_concrete("w_gate", &[h, inter], dt);
    let w_up = g.add_tensor_concrete("w_up", &[h, inter], dt);
    let w_down = g.add_tensor_concrete("w_down", &[inter, h], dt);

    g.inputs = vec![input, attn_out];

    let o_proj = g.add_tensor_concrete("o_proj", &[s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: q_dim, dtype: dt },
        vec![attn_out, w_o],
        vec![o_proj],
        "o_proj",
    );

    let attn_residual = g.add_tensor_concrete("attn_residual", &[s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![input, o_proj],
        vec![attn_residual],
        "attn_residual",
    );

    let ffn_normed = g.add_tensor_concrete("ffn_normed", &[s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![attn_residual, ffn_norm_w],
        vec![ffn_normed],
        "ffn_rms_norm",
    );

    let gate = g.add_tensor_concrete("gate", &[s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h, dtype: dt },
        vec![ffn_normed, w_gate],
        vec![gate],
        "gate_proj",
    );

    let up = g.add_tensor_concrete("up", &[s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h, dtype: dt },
        vec![ffn_normed, w_up],
        vec![up],
        "up_proj",
    );

    let swiglu_out = g.add_tensor_concrete("swiglu_out", &[s, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate, up], vec![swiglu_out], "swiglu");

    let down = g.add_tensor_concrete("down", &[s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: inter, dtype: dt },
        vec![swiglu_out, w_down],
        vec![down],
        "down_proj",
    );

    let output = g.add_tensor_concrete("output", &[s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![attn_residual, down],
        vec![output],
        "ffn_residual",
    );

    g.outputs = vec![output];
    g
}

/// Build a CompilerGraph for the final lm_head projection: hidden -> vocab logits.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) fn build_lm_head_graph(
    seq_len: usize,
    hidden: usize,
    vocab_size: usize,
    dtype: gllm_kernels::types::DType,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};

    let mut g = CompilerGraph::new();
    let dt = dtype;

    let input = g.add_tensor_concrete("input", &[seq_len, hidden], dt);
    let w_lm = g.add_tensor_concrete("w_lm", &[hidden, vocab_size], dt);
    g.inputs = vec![input, w_lm];

    let logits = g.add_tensor_concrete("logits", &[seq_len, vocab_size], dt);
    g.add_op(
        OpKind::Gemm { m: seq_len, n: vocab_size, k: hidden, dtype: dt },
        vec![input, w_lm],
        vec![logits],
        "lm_head",
    );

    g.outputs = vec![logits];
    g
}

/// CPU-side sampling from logits: temperature -> top-k -> top-p -> softmax -> sample.
///
/// Shared across all GPU backends (logits are downloaded to CPU before sampling).
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) fn sample_logits_cpu(
    logits: &[f32],
    vocab_size: usize,
    sampling: &SamplingConfig,
) -> Vec<u32> {
    let seq_logits_count = logits.len() / vocab_size;
    let mut result = Vec::with_capacity(seq_logits_count.max(1));

    for seq_idx in 0..seq_logits_count.max(1) {
        let start = seq_idx * vocab_size;
        let end = (start + vocab_size).min(logits.len());
        let row = &logits[start..end];

        // Apply temperature
        let temperature = if sampling.temperature <= 0.0 { 1e-8 } else { sampling.temperature };
        let mut scored: Vec<(usize, f32)> = row.iter().enumerate()
            .map(|(i, &v)| (i, v / temperature))
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Top-k filtering
        if sampling.top_k > 0 && sampling.top_k < scored.len() {
            scored.truncate(sampling.top_k);
        }

        // Top-p (nucleus) filtering
        if sampling.top_p < 1.0 && sampling.top_p > 0.0 {
            let max_val = scored[0].1;
            let mut cumulative = 0.0f32;
            let mut cutoff = scored.len();
            let exp_sum: f32 = scored.iter().map(|(_, s)| (*s - max_val).exp()).sum();
            for (i, (_, s)) in scored.iter().enumerate() {
                cumulative += (*s - max_val).exp() / exp_sum;
                if cumulative >= sampling.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            scored.truncate(cutoff);
        }

        // Softmax + argmax (greedy / deterministic sampling)
        let max_val = scored[0].1;
        let exp_vals: Vec<f32> = scored.iter().map(|(_, s)| (*s - max_val).exp()).collect();
        let exp_sum: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|e| e / exp_sum).collect();

        let best_idx = probs.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        result.push(scored[best_idx].0 as u32);
    }

    result
}

// ===========================================================================
// HIP decoder forward pass
// ===========================================================================

/// Full decoder (generator) forward pass on HIP GPU.
///
/// 1. Per-layer: compile decoder layer graph -> HIP C++ -> hiprtc -> launch
/// 2. lm_head projection -> logits
/// 3. Return LogitsHandle per sequence
#[cfg(feature = "hip")]
pub(super) fn hip_decoder_forward<E: Element>(
    backend: &HipBackend<E>,
    input: &BatchInput,
    _topology: &AttentionTopology,
    weights: &dyn backend_trait::TensorLookup<E, HipBackend<E>>,
    kv_caches: &mut [KvCacheHandle],
    config: &GeneratorForwardConfig,
) -> Result<Vec<LogitsHandle>, BE> {
    use gllm_kernels::gpu::{GpuDevice, GpuBuffer};
    use gllm_kernels::compiler::TensorId;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder forward only supports f32 element type".into()));
    }

    let device = &*backend.device;
    let stream = device.default_stream();
    let gpu_profile = &backend.gpu_profile;
    let gfx_arch = device.gfx_arch();

    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let vocab_size = config.vocab_size;

    // Flatten all sequences into a single batch dimension
    let total_tokens: usize = input.sequences.iter().map(|s| s.tokens.len()).sum();
    let seq_len = total_tokens;

    // Embedding lookup (CPU, dtype-adaptive)
    let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let elem_bytes = comp_dtype.size_bytes();
    let word_emb = super::gpu_helpers::get_typed_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("token_embedding.weight", Some("token_embd.weight")),
        comp_dtype)?;
    let hidden_bytes = hidden * elem_bytes;
    let vocab = word_emb.len() / hidden_bytes;
    let mut hidden_state = vec![0u8; seq_len * hidden_bytes];
    let mut pos = 0;
    for seq in &input.sequences {
        for &tok in &seq.tokens {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
            }
            hidden_state[pos * hidden_bytes..(pos + 1) * hidden_bytes]
                .copy_from_slice(&word_emb[v * hidden_bytes..(v + 1) * hidden_bytes]);
            pos += 1;
        }
    }

    // Detect incremental decode: position > 0 and KV cache has data
    let rope_theta = config.rope_theta;
    let is_incremental = if let Some(handle) = kv_caches.first() {
        let meta_store = backend.kv_meta.lock()
            .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
        meta_store.get(&handle.0)
            .map(|m| m.seq_len > 0)
            .unwrap_or(false)
            && input.sequences.iter().all(|s| s.position > 0)
    } else { false };

    if is_incremental {
        // ── Incremental decode path (GPU-native, zero CPU round-trip) ──
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
        let proj_graph = build_projection_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta, comp_dtype);
        let (_proj_mod, proj_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &proj_graph)?;
        let post_graph = build_post_attention_graph(seq_len, hidden, num_heads, head_dim, inter, eps, comp_dtype);
        let (_post_mod, post_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &post_graph)?;

        let handle = kv_caches.first()
            .ok_or_else(|| BE::Hip("no KV cache handles provided".into()))?;
        let (cached_seq_len, half_bytes, _total_kv_floats, max_seq_len, head_stride, kv_dtype_size) = {
            let ms = backend.kv_meta.lock()
                .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
            let m = ms.get(&handle.0)
                .ok_or_else(|| BE::Hip(format!("KV cache handle {} not found", handle.0)))?;
            let hb = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim * m.dtype_size;
            let tkf = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim;
            let hs = m.max_seq_len * m.head_dim * m.dtype_size;
            (m.seq_len, hb, tkf, m.max_seq_len, hs, m.dtype_size)
        };
        let total_seq = cached_seq_len + seq_len;

        let use_paged = config.paged_kv_page_table.is_some();
        let page_size = config.paged_kv_page_size;
        let attn_graph = if use_paged {
            build_gpu_paged_attention_graph(
                seq_len, total_seq, num_heads, num_kv_heads, head_dim, page_size, comp_dtype,
            )
        } else {
            build_gpu_cached_attention_graph(
                seq_len, total_seq, num_heads, num_kv_heads, head_dim, comp_dtype,
            )
        };
        let (_attn_mod, attn_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &attn_graph)?;

        // Paged KV: allocate paged cache + upload page table (once before layer loop)
        let paged_meta = if use_paged {
            let pt = config.paged_kv_page_table.as_ref().unwrap();
            let num_physical_pages = pt.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
            let meta = gpu_alloc_paged_kv_cache_hip(
                device, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, kv_dtype_size,
            )?;
            Some((meta, pt.clone()))
        } else {
            None
        };

        // ── Pre-allocate GPU buffers (reused across all layers) ──
        let elem_bytes = comp_dtype.size_bytes();
        let hidden_bytes_typed = seq_len * hidden * elem_bytes;
        let hidden_bytes = seq_len * hidden * elem_bytes;
        let attn_out_bytes = seq_len * q_dim * elem_bytes;

        let mut gpu_hidden = device.alloc(hidden_bytes_typed)
            .map_err(|e| BE::Hip(format!("GPU alloc gpu_hidden failed: {e}")))?;
        {
            let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr(), hidden_bytes_typed) };
            device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Hip(format!("htod initial hidden: {e}")))?;
        }

        let mut proj_bufs: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
        for (idx, tmeta) in proj_graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let buf = device.alloc(tmeta.concrete_numel() * elem_bytes)
                .map_err(|e| BE::Hip(format!("GPU alloc proj {} failed: {e}", tmeta.name)))?;
            proj_bufs.push((tid, buf));
        }
        let attn_out_buf = device.alloc(attn_out_bytes)
            .map_err(|e| BE::Hip(format!("GPU alloc attn_out failed: {e}")))?;
        let mut post_bufs: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
        for (idx, tmeta) in post_graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let buf = device.alloc(tmeta.concrete_numel() * elem_bytes)
                .map_err(|e| BE::Hip(format!("GPU alloc post {} failed: {e}", tmeta.name)))?;
            post_bufs.push((tid, buf));
        }

        // ── Pre-load all layer weights (CPU side, once) ──
        let all_layer_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
            .map(|layer| super::gpu_helpers::load_decoder_layer_weights_gpu_typed(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter, comp_dtype,
            ))
            .collect::<Result<Vec<_>, _>>()?;

        // ── Init HIP weight cache (once, first incremental step) ──
        {
            let mut wc = backend.weight_cache.lock()
                .map_err(|e| BE::Hip(format!("weight_cache lock: {e}")))?;
            if !wc.is_initialized() {
                wc.init(&all_layer_weights, device, stream)?;
                log::debug!("[HipWeightCache] initialized: {} bytes", wc.total_bytes);
            }
        }

        for layer in 0..num_layers {
            let layer_weights = &all_layer_weights[layer];

            // ── GPU: projection graph ──
            let mut proj_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            for (idx, tmeta) in proj_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let (_, ref mut buf) = proj_bufs[idx];
                if tmeta.name == "input" {
                    let res = unsafe {
                        (device.driver().hipMemcpyDtoD)(
                            buf.as_device_ptr() as _,
                            gpu_hidden.as_device_ptr() as _,
                            hidden_bytes_typed,
                        )
                    };
                    if res != 0 {
                        return Err(BE::Other(format!("DtoD hidden→proj input failed: HIP error {res}")));
                    }
                } else {
                    let wc = backend.weight_cache.lock()
                        .map_err(|e| BE::Hip(format!("weight_cache lock: {e}")))?;
                    if let Some(cached_ptr) = wc.get(layer, &tmeta.name) {
                        let copy_bytes = tmeta.concrete_numel() * elem_bytes;
                        let res = unsafe {
                            (device.driver().hipMemcpyDtoD)(
                                buf.as_device_ptr() as _,
                                cached_ptr as _,
                                copy_bytes,
                            )
                        };
                        if res != 0 {
                            return Err(BE::Hip(format!("DtoD weight cache proj {} failed: HIP error {res}", tmeta.name)));
                        }
                    } else if let Some(data) = layer_weights.get(&tmeta.name) {
                        device.htod(data, buf, stream).map_err(|e| BE::Hip(format!("htod proj {}: {e}", tmeta.name)))?;
                    }
                }
                proj_ptrs.insert(tid, buf.as_device_ptr());
            }
            hip_launch_graph(device, stream, &proj_entries, &proj_ptrs, &proj_graph)?;

            // ── GPU: write k_rope/v_proj into KV cache (DtoD) ──
            let k_tid = proj_graph.outputs[1];
            let v_tid = proj_graph.outputs[2];
            let k_ptr = proj_bufs.iter().find(|(t, _)| *t == k_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Hip(format!("proj buffer for k_tid {:?} not found", k_tid)))?;
            let v_ptr = proj_bufs.iter().find(|(t, _)| *t == v_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Hip(format!("proj buffer for v_tid {:?} not found", v_tid)))?;

            if let Some((ref paged_meta, ref page_table)) = paged_meta {
                gpu_write_kv_paged_hip(
                    k_ptr, v_ptr, kv_dim, seq_len, num_kv_heads, head_dim,
                    paged_meta, layer, page_table, cached_seq_len, device,
                )?;
            } else {
                gpu_write_kv_cache_scatter_hip(
                    k_ptr, v_ptr, kv_dim, seq_len, num_kv_heads, head_dim,
                    handle, layer, half_bytes, cached_seq_len, head_stride, kv_dtype_size, device, stream,
                    gpu_profile, gfx_arch,
                )?;
            }

            if layer == 0 {
                let mut ms = backend.kv_meta.lock()
                    .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
                if let Some(m) = ms.get_mut(&handle.0) {
                    m.seq_len = (m.seq_len + seq_len).min(m.max_seq_len);
                }
            }

            // ── GPU: cached attention ──
            let q_tid = proj_graph.outputs[0];
            let q_ptr = proj_bufs.iter().find(|(t, _)| *t == q_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Hip(format!("proj buffer for q_tid {:?} not found", q_tid)))?;

            let mut attn_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            if let Some((ref paged_meta, ref page_table)) = paged_meta {
                let pt_buf = gpu_upload_page_table_hip(device, stream, page_table)?;
                for (idx, tmeta) in attn_graph.tensors.iter().enumerate() {
                    let tid = TensorId(idx as u32);
                    match tmeta.name.as_str() {
                        "q_rope" => { attn_ptrs.insert(tid, q_ptr); }
                        "page_table" => { attn_ptrs.insert(tid, pt_buf.as_device_ptr()); }
                        "kv_cache" => { attn_ptrs.insert(tid, paged_meta.pool_ptr); }
                        "attn_out" => { attn_ptrs.insert(tid, attn_out_buf.as_device_ptr()); }
                        _ => {}
                    }
                }
            } else {
                let (k_layer_ptr, v_layer_ptr) = kv_cache_layer_ptrs(handle, layer, half_bytes, num_kv_heads, head_stride);
                for (idx, tmeta) in attn_graph.tensors.iter().enumerate() {
                    let tid = TensorId(idx as u32);
                    match tmeta.name.as_str() {
                        "q_rope" => { attn_ptrs.insert(tid, q_ptr); }
                        "k_cache" => { attn_ptrs.insert(tid, k_layer_ptr); }
                        "v_cache" => { attn_ptrs.insert(tid, v_layer_ptr); }
                        "attn_out" => { attn_ptrs.insert(tid, attn_out_buf.as_device_ptr()); }
                        _ => {}
                    }
                }
            }
            hip_launch_graph(device, stream, &attn_entries, &attn_ptrs, &attn_graph)?;

            // ── GPU: post-attention graph ──
            let mut post_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            for (idx, tmeta) in post_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let (_, ref mut buf) = post_bufs[idx];
                if tmeta.name == "input" {
                    let res = unsafe {
                        (device.driver().hipMemcpyDtoD)(
                            buf.as_device_ptr() as _,
                            gpu_hidden.as_device_ptr() as _,
                            hidden_bytes,
                        )
                    };
                    if res != 0 {
                        return Err(BE::Other(format!("DtoD hidden→post input failed: HIP error {res}")));
                    }
                } else if tmeta.name == "attn_out" {
                    let res = unsafe {
                        (device.driver().hipMemcpyDtoD)(
                            buf.as_device_ptr() as _,
                            attn_out_buf.as_device_ptr() as _,
                            attn_out_bytes,
                        )
                    };
                    if res != 0 {
                        return Err(BE::Other(format!("DtoD attn_out→post failed: HIP error {res}")));
                    }
                } else {
                    let wc = backend.weight_cache.lock()
                        .map_err(|e| BE::Hip(format!("weight_cache lock: {e}")))?;
                    if let Some(cached_ptr) = wc.get(layer, &tmeta.name) {
                        let copy_bytes = tmeta.concrete_numel() * elem_bytes;
                        let res = unsafe {
                            (device.driver().hipMemcpyDtoD)(
                                buf.as_device_ptr() as _,
                                cached_ptr as _,
                                copy_bytes,
                            )
                        };
                        if res != 0 {
                            return Err(BE::Hip(format!("DtoD weight cache post {} failed: HIP error {res}", tmeta.name)));
                        }
                    } else if let Some(data) = layer_weights.get(&tmeta.name) {
                        device.htod(data, buf, stream).map_err(|e| BE::Hip(format!("htod post {}: {e}", tmeta.name)))?;
                    }
                }
                post_ptrs.insert(tid, buf.as_device_ptr());
            }
            hip_launch_graph(device, stream, &post_entries, &post_ptrs, &post_graph)?;
            device.sync().map_err(|e| BE::Hip(format!("GPU sync layer {layer}: {e}")))?;

            // Update gpu_hidden: DtoD from post output
            let output_tid = post_graph.outputs[0];
            let output_ptr = post_bufs.iter().find(|(tid, _)| *tid == output_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Hip("post output buffer not found".into()))?;
            let res = unsafe {
                (device.driver().hipMemcpyDtoD)(
                    gpu_hidden.as_device_ptr() as _,
                    output_ptr as _,
                    hidden_bytes_typed,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD post output→gpu_hidden failed: HIP error {res}")));
            }
        }

        // Final: download hidden_state from GPU (once)
        let mut output_host = vec![0u8; hidden_bytes_typed];
        device.dtoh(&gpu_hidden, &mut output_host, stream)
            .map_err(|e| BE::Hip(format!("dtoh final hidden: {e}")))?;
        hidden_state.copy_from_slice(&output_host);
    } else {
        // ── Prefill path (GPU-resident hidden_state) ──
        let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
        let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta, comp_dtype);
        let (_module, kernel_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &graph)?;

        let hidden_bytes_typed = seq_len * hidden * elem_bytes;
        let elem_bytes = comp_dtype.size_bytes();
        let mut gpu_hidden = device.alloc(hidden_bytes_typed)
            .map_err(|e| BE::Hip(format!("GPU alloc gpu_hidden failed: {e}")))?;
        {
            let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr(), hidden_bytes_typed) };
            device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Hip(format!("htod initial hidden: {e}")))?;
        }

        // ── Pre-load all layer weights (CPU side, once) ──
        let all_layer_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
            .map(|layer| super::gpu_helpers::load_decoder_layer_weights_gpu_typed(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter, comp_dtype,
            ))
            .collect::<Result<Vec<_>, _>>()?;

        for layer in 0..num_layers {
            let layer_weights = &all_layer_weights[layer];

            let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
            let mut tensor_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

            for (idx, meta) in graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let size_bytes = meta.concrete_numel() * elem_bytes;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Hip(format!("GPU alloc failed for {}: {e}", meta.name)))?;

                if meta.name == "input" {
                    let res = unsafe {
                        (device.driver().hipMemcpyDtoD)(
                            buf.as_device_ptr() as _,
                            gpu_hidden.as_device_ptr() as _,
                            hidden_bytes_typed,
                        )
                    };
                    if res != 0 {
                        return Err(BE::Other(format!("DtoD hidden→input failed: HIP error {res}")));
                    }
                } else if let Some(data) = layer_weights.get(&meta.name) {
                    device.htod(data, &mut buf, stream)
                        .map_err(|e| BE::Hip(format!("htod {} failed: {e}", meta.name)))?;
                }

                tensor_ptrs.insert(tid, buf.as_device_ptr());
                gpu_buffers.push((tid, buf));
            }

            hip_launch_graph(device, stream, &kernel_entries, &tensor_ptrs, &graph)?;
            device.sync().map_err(|e| BE::Hip(format!("GPU sync failed: {e}")))?;

            // ── Write K/V to KV cache ──
            if let Some(handle) = kv_caches.first() {
                let kv_dim = num_kv_heads * head_dim;

                let k_tid = graph.tensors.iter().enumerate()
                    .find(|(_, m)| m.name == "k_rope")
                    .map(|(i, _)| TensorId(i as u32))
                    .ok_or_else(|| BE::Hip("k_rope tensor not found in graph".into()))?;
                let v_tid = graph.tensors.iter().enumerate()
                    .find(|(_, m)| m.name == "v_proj")
                    .map(|(i, _)| TensorId(i as u32))
                    .ok_or_else(|| BE::Hip("v_proj tensor not found in graph".into()))?;

                let k_ptr = gpu_buffers.iter().find(|(t, _)| *t == k_tid).map(|(_, b)| b.as_device_ptr())
                    .ok_or_else(|| BE::Hip("k_rope buffer not found".into()))?;
                let v_ptr = gpu_buffers.iter().find(|(t, _)| *t == v_tid).map(|(_, b)| b.as_device_ptr())
                    .ok_or_else(|| BE::Hip("v_proj buffer not found".into()))?;

                let meta_store = backend.kv_meta.lock()
                    .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
                let meta = meta_store.get(&handle.0)
                    .ok_or_else(|| BE::Hip(format!("KV cache handle {} not found", handle.0)))?;

                let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let write_start = if layer == 0 { meta.seq_len } else { meta.seq_len.saturating_sub(seq_len) };
                let kv_ds = meta.dtype_size;

                drop(meta_store);

                gpu_write_kv_cache_scatter_hip(
                    k_ptr, v_ptr, kv_dim, seq_len, num_kv_heads, head_dim,
                    handle, layer, half_bytes, write_start, head_stride, kv_ds, device, stream,
                    gpu_profile, gfx_arch,
                )?;

                if layer == 0 {
                    let mut meta_store = backend.kv_meta.lock()
                        .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
                    if let Some(meta) = meta_store.get_mut(&handle.0) {
                        meta.seq_len = (meta.seq_len + seq_len).min(meta.max_seq_len);
                    }
                }
            }

            // Update gpu_hidden: DtoD from output
            let output_tid = graph.outputs[0];
            let output_ptr = gpu_buffers.iter().find(|(tid, _)| *tid == output_tid).map(|(_, b)| b.as_device_ptr())
                .ok_or_else(|| BE::Hip("output buffer not found".into()))?;
            let res = unsafe {
                (device.driver().hipMemcpyDtoD)(
                    gpu_hidden.as_device_ptr() as _,
                    output_ptr as _,
                    hidden_bytes_typed,
                )
            };
            if res != 0 {
                return Err(BE::Other(format!("DtoD output→gpu_hidden failed: HIP error {res}")));
            }
        }

        // Final: download hidden_state from GPU (once)
        {
            let mut output_host = vec![0u8; hidden_bytes_typed];
            device.dtoh(&gpu_hidden, &mut output_host, stream)
                .map_err(|e| BE::Hip(format!("dtoh final hidden: {e}")))?;
            hidden_state.copy_from_slice(&output_host);
        }
    }

    // ── lm_head projection (GPU) ──
    let lm_comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let lm_elem_bytes = lm_comp_dtype.size_bytes();
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size, lm_comp_dtype);
    let (_lm_module, lm_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &lm_graph)?;

    let lm_head_w = super::gpu_helpers::get_typed_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("lm_head.weight", Some("output.weight")),
        lm_comp_dtype)?;

    let mut gpu_buffers_lm: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
    let mut lm_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

    for (idx, meta) in lm_graph.tensors.iter().enumerate() {
        let tid = TensorId(idx as u32);
        let n_elements = meta.concrete_numel();
        let size_bytes = n_elements * lm_elem_bytes;
        let mut buf = device.alloc(size_bytes)
            .map_err(|e| BE::Hip(format!("GPU alloc lm_head {} failed: {e}", meta.name)))?;

        if meta.name == "input" {
            device.htod(&hidden_state, &mut buf, stream)
                .map_err(|e| BE::Hip(format!("htod lm_head input: {e}")))?;
        } else if meta.name == "w_lm" {
            device.htod(&lm_head_w, &mut buf, stream)
                .map_err(|e| BE::Hip(format!("htod lm_head weight: {e}")))?;
        }

        lm_ptrs.insert(tid, buf.as_device_ptr());
        gpu_buffers_lm.push((tid, buf));
    }

    hip_launch_graph(device, stream, &lm_entries, &lm_ptrs, &lm_graph)?;
    device.sync().map_err(|e| BE::Hip(format!("GPU sync lm_head: {e}")))?;

    // Download logits
    let logits_tid = lm_graph.outputs[0];
    let logits_buf = gpu_buffers_lm.iter()
        .find(|(tid, _)| *tid == logits_tid)
        .map(|(_, buf)| buf)
        .ok_or_else(|| BE::Hip("logits buffer not found".into()))?;
    let logits_bytes = seq_len * vocab_size * lm_elem_bytes;
    let mut logits_host = vec![0u8; logits_bytes];
    device.dtoh(logits_buf, &mut logits_host, stream)
        .map_err(|e| BE::Hip(format!("dtoh logits: {e}")))?;

    let logits_f32 = super::jit_helpers::typed_bytes_to_f32(&logits_host, lm_comp_dtype);

    // Split logits per sequence
    let mut results = Vec::new();
    let mut offset = 0;
    for seq in &input.sequences {
        let tok_count = seq.tokens.len();
        let last_pos = offset + tok_count - 1;
        let seq_logits = logits_f32[last_pos * vocab_size..(last_pos + 1) * vocab_size].to_vec();
        results.push(LogitsHandle { data: seq_logits });
        offset += tok_count;
    }

    Ok(results)
}

// ===========================================================================
// Metal decoder forward pass
// ===========================================================================

/// Full decoder (generator) forward pass on Metal GPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn metal_decoder_forward<E: Element>(
    backend: &super::metal_backend::MetalBackend<E>,
    input: &BatchInput,
    _topology: &AttentionTopology,
    weights: &dyn backend_trait::TensorLookup<E, super::metal_backend::MetalBackend<E>>,
    kv_caches: &mut [KvCacheHandle],
    config: &GeneratorForwardConfig,
) -> Result<Vec<LogitsHandle>, BE> {
    use gllm_kernels::gpu::{GpuDevice, GpuBuffer};
    use gllm_kernels::compiler::TensorId;
    use gllm_kernels::compiler::codegen::emitter::Platform;

    if std::any::TypeId::of::<E>() != std::any::TypeId::of::<f32>() {
        return Err(BE::Other("Decoder forward only supports f32 element type".into()));
    }

    let device = backend.device();
    let stream = device.default_stream();
    let gpu_profile = backend.gpu_profile();
    let gpu_family = match gpu_profile.platform {
        Platform::Metal { gpu_family } => gpu_family,
        other => {
            return Err(BE::Other(format!(
                "expected Metal platform for Metal backend, got {:?}", other
            )));
        }
    };

    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let vocab_size = config.vocab_size;

    let total_tokens: usize = input.sequences.iter().map(|s| s.tokens.len()).sum();
    let seq_len = total_tokens;

    // Embedding lookup (CPU, dtype-adaptive)
    let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let elem_bytes = comp_dtype.size_bytes();
    let word_emb = super::gpu_helpers::get_typed_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("token_embedding.weight", Some("token_embd.weight")),
        comp_dtype)?;
    let hidden_bytes = hidden * elem_bytes;
    let vocab = word_emb.len() / hidden_bytes;
    let mut hidden_state = vec![0u8; seq_len * hidden_bytes];
    let mut pos = 0;
    for seq in &input.sequences {
        for &tok in &seq.tokens {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
            }
            hidden_state[pos * hidden_bytes..(pos + 1) * hidden_bytes]
                .copy_from_slice(&word_emb[v * hidden_bytes..(v + 1) * hidden_bytes]);
            pos += 1;
        }
    }

    // Detect incremental decode: position > 0 and KV cache has data
    let rope_theta = config.rope_theta;
    let is_incremental = if let Some(handle) = kv_caches.first() {
        let meta_store = backend.kv_meta.lock()
            .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
        meta_store.get(&handle.0)
            .map(|m| m.seq_len > 0)
            .unwrap_or(false)
            && input.sequences.iter().all(|s| s.position > 0)
    } else { false };

    if is_incremental {
        // ── Incremental decode path (GPU-native, buffer reuse, hidden stays on GPU) ──
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
        let proj_graph = build_projection_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta, comp_dtype);
        let proj_entries = metal_compile_graph(device, gpu_profile, gpu_family, &proj_graph)?;
        let post_graph = build_post_attention_graph(seq_len, hidden, num_heads, head_dim, inter, eps, comp_dtype);
        let post_entries = metal_compile_graph(device, gpu_profile, gpu_family, &post_graph)?;

        let handle = kv_caches.first()
            .ok_or_else(|| BE::Metal("no KV cache handles provided".into()))?;
        let (cached_seq_len, half_bytes, _total_kv_floats, max_seq_len, head_stride, kv_dtype_size) = {
            let ms = backend.kv_meta.lock()
                .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
            let m = ms.get(&handle.0)
                .ok_or_else(|| BE::Metal(format!("KV cache handle {} not found", handle.0)))?;
            let hb = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim * m.dtype_size;
            let tkf = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim;
            let hs = m.max_seq_len * m.head_dim * m.dtype_size;
            (m.seq_len, hb, tkf, m.max_seq_len, hs, m.dtype_size)
        };
        let total_seq = cached_seq_len + seq_len;

        let use_paged = config.paged_kv_page_table.is_some();
        let page_size = config.paged_kv_page_size;
        let attn_graph = if use_paged {
            build_gpu_paged_attention_graph(
                seq_len, total_seq, num_heads, num_kv_heads, head_dim, page_size, comp_dtype,
            )
        } else {
            build_gpu_cached_attention_graph(
                seq_len, total_seq, num_heads, num_kv_heads, head_dim, comp_dtype,
            )
        };
        let attn_entries = metal_compile_graph(device, gpu_profile, gpu_family, &attn_graph)?;

        // Paged KV: allocate paged cache (once before layer loop)
        let paged_meta = if use_paged {
            let pt = config.paged_kv_page_table.as_ref().unwrap();
            let num_physical_pages = pt.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
            let meta = metal_alloc_paged_kv_cache(
                device, num_physical_pages, page_size, num_layers, num_kv_heads, head_dim, kv_dtype_size,
            )?;
            Some((meta, pt.clone()))
        } else {
            None
        };

        // ── Pre-allocate: hidden_state GPU buffer (stays on GPU across layers) ──
        let elem_bytes = comp_dtype.size_bytes();
        let hidden_bytes_typed = seq_len * hidden * elem_bytes;
        let attn_out_bytes = seq_len * q_dim * elem_bytes;
        let kv_bytes = seq_len * kv_dim * elem_bytes;
        let layer_kv_bytes = num_kv_heads * head_stride;

        let mut gpu_hidden = device.alloc(hidden_bytes_typed)
            .map_err(|e| BE::Metal(format!("alloc gpu_hidden failed: {e}")))?;
        {
            let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr(), hidden_bytes_typed) };
            device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Metal(format!("htod initial hidden: {e}")))?;
        }

        // ── Pre-load all layer weights (CPU side, once) ──
        let all_layer_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
            .map(|layer| super::gpu_helpers::load_decoder_layer_weights_gpu_typed(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter, comp_dtype,
            ))
            .collect::<Result<Vec<_>, _>>()?;

        // ── Init Metal weight cache (once, first incremental step) ──
        {
            let mut wc = backend.weight_cache.lock()
                .map_err(|e| BE::Metal(format!("weight_cache lock: {e}")))?;
            if !wc.is_initialized() {
                wc.init(&all_layer_weights, device, stream)?;
                log::debug!("[MetalWeightCache] initialized: {} bytes", wc.total_bytes);
            }
        }

        for layer in 0..num_layers {
            let layer_weights = &all_layer_weights[layer];

            // ── GPU: projection graph ──
            let mut proj_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
                std::collections::HashMap::new();
            for (idx, tmeta) in proj_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let size_bytes = tmeta.concrete_numel() * elem_bytes;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Metal(format!("alloc proj {} failed: {e}", tmeta.name)))?;
                if tmeta.name == "input" {
                    device.dtod(&gpu_hidden, &mut buf, stream)
                        .map_err(|e| BE::Metal(format!("dtod hidden→proj input: {e}")))?;
                } else {
                    let wc = backend.weight_cache.lock()
                        .map_err(|e| BE::Metal(format!("weight_cache lock: {e}")))?;
                    if let Some(cached_buf) = wc.get_buf(layer, &tmeta.name) {
                        device.dtod(cached_buf, &mut buf, stream)
                            .map_err(|e| BE::Metal(format!("dtod weight cache proj {}: {e}", tmeta.name)))?;
                    } else if let Some(data) = layer_weights.get(&tmeta.name) {
                        device.htod(data, &mut buf, stream).map_err(|e| BE::Metal(format!("htod proj {}: {e}", tmeta.name)))?;
                    }
                }
                proj_bufs.insert(tid, buf);
            }
            metal_launch_graph(device, stream, &proj_entries, &proj_bufs, &proj_graph)?;
            device.sync().map_err(|e| BE::Metal(format!("sync proj: {e}")))?;

            // ── Write k_rope/v_proj into KV cache ──
            let k_tid = proj_graph.outputs[1];
            let v_tid = proj_graph.outputs[2];

            let k_buf = proj_bufs.get(&k_tid)
                .ok_or_else(|| BE::Metal(format!("proj buffer for k_tid {:?} not found", k_tid)))?;
            let v_buf = proj_bufs.get(&v_tid)
                .ok_or_else(|| BE::Metal(format!("proj buffer for v_tid {:?} not found", v_tid)))?;

            if let Some((ref paged_meta, ref page_table)) = paged_meta {
                metal_write_kv_paged(
                    k_buf.as_device_ptr(), v_buf.as_device_ptr(),
                    kv_dim, seq_len, num_kv_heads, head_dim,
                    paged_meta, layer, page_table, cached_seq_len,
                )?;
            } else {
                metal_write_kv_direct(
                    k_buf.as_device_ptr(), v_buf.as_device_ptr(),
                    handle, kv_dim, seq_len, num_kv_heads, head_dim,
                    layer, half_bytes, cached_seq_len, head_stride, kv_dtype_size,
                )?;
            }

            if layer == 0 {
                let mut ms = backend.kv_meta.lock()
                    .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
                if let Some(m) = ms.get_mut(&handle.0) {
                    m.seq_len = (m.seq_len + seq_len).min(m.max_seq_len);
                }
            }

            // ── GPU: cached attention ──
            let q_tid = proj_graph.outputs[0];

            // Build attention buffers
            let mut attn_metal_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
                std::collections::HashMap::new();
            if let Some((ref paged_meta, ref page_table)) = paged_meta {
                // Paged attention: pass pool pointer and page table via shared memory
                for (idx, tmeta) in attn_graph.tensors.iter().enumerate() {
                    let tid = TensorId(idx as u32);
                    let size_bytes = tmeta.concrete_numel() * elem_bytes;
                    let mut buf = device.alloc(size_bytes)
                        .map_err(|e| BE::Metal(format!("alloc attn {} failed: {e}", tmeta.name)))?;
                    match tmeta.name.as_str() {
                        "q_rope" => {
                            let q_buf = proj_bufs.get(&q_tid)
                                .ok_or_else(|| BE::Metal(format!("proj buffer for q_tid {:?} not found", q_tid)))?;
                            device.dtod(q_buf, &mut buf, stream)
                                .map_err(|e| BE::Metal(format!("dtod q_rope for attn: {e}")))?;
                        }
                        "page_table" => {
                            let pt_bytes = unsafe {
                                std::slice::from_raw_parts(page_table.as_ptr() as *const u8, page_table.len() * 4)
                            };
                            device.htod(pt_bytes, &mut buf, stream)
                                .map_err(|e| BE::Metal(format!("htod page_table: {e}")))?;
                        }
                        "kv_cache" => {
                            // placeholder — actual pool pointer passed via raw ptr
                            // Metal shared memory: set buf to point at pool
                            // We pass pool_ptr directly by uploading a 1-element placeholder
                        }
                        _ => {}
                    }
                    attn_metal_bufs.insert(tid, buf);
                }
                // Override kv_cache tensor ptr to pool_ptr
                for (idx, tmeta) in attn_graph.tensors.iter().enumerate() {
                    if tmeta.name == "kv_cache" {
                        let tid = TensorId(idx as u32);
                        // Metal shared memory: pool_ptr is a CPU-accessible pointer
                        // Replace the allocated buf with one pointing at pool
                        let pool_bytes = unsafe {
                            std::slice::from_raw_parts(paged_meta.pool_ptr as *const u8, paged_meta.pool_bytes)
                        };
                        let mut pool_buf = device.alloc(paged_meta.pool_bytes)
                            .map_err(|e| BE::Metal(format!("alloc kv_cache pool buf failed: {e}")))?;
                        device.htod(pool_bytes, &mut pool_buf, stream)
                            .map_err(|e| BE::Metal(format!("htod kv_cache pool: {e}")))?;
                        attn_metal_bufs.insert(tid, pool_buf);
                        break;
                    }
                }
            } else {
                let (k_layer_ptr, v_layer_ptr) = kv_cache_layer_ptrs(handle, layer, half_bytes, num_kv_heads, head_stride);
                // Metal uses shared memory — read KV cache layer directly via pointer
                let k_layer_host = unsafe {
                    std::slice::from_raw_parts(k_layer_ptr as *const u8, layer_kv_bytes)
                }.to_vec();
                let v_layer_host = unsafe {
                    std::slice::from_raw_parts(v_layer_ptr as *const u8, layer_kv_bytes)
                }.to_vec();

                for (idx, tmeta) in attn_graph.tensors.iter().enumerate() {
                    let tid = TensorId(idx as u32);
                    let size_bytes = tmeta.concrete_numel() * elem_bytes;
                    let mut buf = device.alloc(size_bytes)
                        .map_err(|e| BE::Metal(format!("alloc attn {} failed: {e}", tmeta.name)))?;
                    match tmeta.name.as_str() {
                        "q_rope" => {
                            let q_buf = proj_bufs.get(&q_tid)
                                .ok_or_else(|| BE::Metal(format!("proj buffer for q_tid {:?} not found", q_tid)))?;
                            device.dtod(q_buf, &mut buf, stream)
                                .map_err(|e| BE::Metal(format!("dtod q_rope for attn: {e}")))?;
                        }
                        "k_cache" => {
                            device.htod(&k_layer_host, &mut buf, stream)
                                .map_err(|e| BE::Metal(format!("htod k_cache layer: {e}")))?;
                        }
                        "v_cache" => {
                            device.htod(&v_layer_host, &mut buf, stream)
                                .map_err(|e| BE::Metal(format!("htod v_cache layer: {e}")))?;
                        }
                        _ => {}
                    }
                    attn_metal_bufs.insert(tid, buf);
                }
            }
            metal_launch_graph(device, stream, &attn_entries, &attn_metal_bufs, &attn_graph)?;
            device.sync().map_err(|e| BE::Metal(format!("sync attn: {e}")))?;

            // Get attn_out for post graph
            let attn_out_tid = attn_graph.outputs[0];
            let attn_out_buf = attn_metal_bufs.remove(&attn_out_tid)
                .ok_or_else(|| BE::Metal("attn_out buffer not found".into()))?;

            // ── GPU: post-attention graph ──
            let mut post_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
                std::collections::HashMap::new();
            for (idx, tmeta) in post_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let size_bytes = tmeta.concrete_numel() * elem_bytes;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Metal(format!("alloc post {} failed: {e}", tmeta.name)))?;
                if tmeta.name == "input" {
                    // DtoD from gpu_hidden
                    device.dtod(&gpu_hidden, &mut buf, stream)
                        .map_err(|e| BE::Metal(format!("dtod hidden→post input: {e}")))?;
                } else if tmeta.name == "attn_out" {
                    // DtoD from attn_out_buf
                    device.dtod(&attn_out_buf, &mut buf, stream)
                        .map_err(|e| BE::Metal(format!("dtod attn_out→post: {e}")))?;
                } else {
                    let wc = backend.weight_cache.lock()
                        .map_err(|e| BE::Metal(format!("weight_cache lock: {e}")))?;
                    if let Some(cached_buf) = wc.get_buf(layer, &tmeta.name) {
                        device.dtod(cached_buf, &mut buf, stream)
                            .map_err(|e| BE::Metal(format!("dtod weight cache post {}: {e}", tmeta.name)))?;
                    } else if let Some(data) = layer_weights.get(&tmeta.name) {
                        device.htod(data, &mut buf, stream).map_err(|e| BE::Metal(format!("htod post {}: {e}", tmeta.name)))?;
                    }
                }
                post_bufs.insert(tid, buf);
            }
            metal_launch_graph(device, stream, &post_entries, &post_bufs, &post_graph)?;
            device.sync().map_err(|e| BE::Metal(format!("sync layer {layer}: {e}")))?;

            // Update gpu_hidden: DtoD from post output (stays on GPU)
            let output_tid = post_graph.outputs[0];
            let output_buf = post_bufs.get(&output_tid)
                .ok_or_else(|| BE::Metal("post output buffer not found".into()))?;
            device.dtod(output_buf, &mut gpu_hidden, stream)
                .map_err(|e| BE::Metal(format!("dtod post output→gpu_hidden: {e}")))?;
        }

        // Final: download hidden_state from GPU (once)
        let mut output_host = vec![0u8; hidden_bytes_typed];
        device.dtoh(&gpu_hidden, &mut output_host, stream)
            .map_err(|e| BE::Metal(format!("dtoh final hidden: {e}")))?;
        hidden_state.copy_from_slice(&output_host);
    } else {
        // ── Prefill path (GPU-resident hidden_state) ──
        let comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
        let elem_bytes = comp_dtype.size_bytes();
        let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta, comp_dtype);
        let kernel_entries = metal_compile_graph(device, gpu_profile, gpu_family, &graph)?;

        let hidden_bytes_typed = seq_len * hidden * elem_bytes;
        let mut gpu_hidden = device.alloc(hidden_bytes_typed)
            .map_err(|e| BE::Metal(format!("alloc gpu_hidden failed: {e}")))?;
        {
            let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr(), hidden_bytes_typed) };
            device.htod(bytes, &mut gpu_hidden, stream).map_err(|e| BE::Metal(format!("htod initial hidden: {e}")))?;
        }

        // ── Pre-load all layer weights (CPU side, once) ──
        let all_layer_weights: Vec<std::collections::HashMap<String, Vec<u8>>> = (0..num_layers)
            .map(|layer| super::gpu_helpers::load_decoder_layer_weights_gpu_typed(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter, comp_dtype,
            ))
            .collect::<Result<Vec<_>, _>>()?;

        for layer in 0..num_layers {
            let layer_weights = &all_layer_weights[layer];

            let mut gpu_buffers: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
                std::collections::HashMap::new();

            for (idx, meta) in graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let size_bytes = meta.concrete_numel() * elem_bytes;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Metal(format!("alloc failed for {}: {e}", meta.name)))?;

                if meta.name == "input" {
                    device.dtod(&gpu_hidden, &mut buf, stream)
                        .map_err(|e| BE::Metal(format!("dtod hidden→input: {e}")))?;
                } else if let Some(data) = layer_weights.get(&meta.name) {
                    device.htod(data, &mut buf, stream)
                        .map_err(|e| BE::Metal(format!("htod {} failed: {e}", meta.name)))?;
                }

                gpu_buffers.insert(tid, buf);
            }

            metal_launch_graph(device, stream, &kernel_entries, &gpu_buffers, &graph)?;
            device.sync().map_err(|e| BE::Metal(format!("sync failed: {e}")))?;

            // ── Write K/V to KV cache (Metal 直写：shared memory 指针，无 dtoh/htod) ──
            if let Some(handle) = kv_caches.first() {
                let kv_dim = num_kv_heads * head_dim;

                let k_tid = graph.tensors.iter().enumerate()
                    .find(|(_, m)| m.name == "k_rope")
                    .map(|(i, _)| TensorId(i as u32))
                    .ok_or_else(|| BE::Metal("k_rope tensor not found in graph".into()))?;
                let v_tid = graph.tensors.iter().enumerate()
                    .find(|(_, m)| m.name == "v_proj")
                    .map(|(i, _)| TensorId(i as u32))
                    .ok_or_else(|| BE::Metal("v_proj tensor not found in graph".into()))?;

                let k_buf = gpu_buffers.get(&k_tid)
                    .ok_or_else(|| BE::Metal("k_rope buffer not found".into()))?;
                let v_buf = gpu_buffers.get(&v_tid)
                    .ok_or_else(|| BE::Metal("v_proj buffer not found".into()))?;

                let meta_store = backend.kv_meta.lock()
                    .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
                let meta = meta_store.get(&handle.0)
                    .ok_or_else(|| BE::Metal(format!("KV cache handle {} not found", handle.0)))?;

                let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let write_start = if layer == 0 { meta.seq_len } else { meta.seq_len.saturating_sub(seq_len) };
                let kv_ds = meta.dtype_size;

                drop(meta_store);

                metal_write_kv_direct(
                    k_buf.as_device_ptr(), v_buf.as_device_ptr(),
                    handle, kv_dim, seq_len, num_kv_heads, head_dim,
                    layer, half_bytes, write_start, head_stride, kv_ds,
                )?;

                if layer == 0 {
                    let mut meta_store = backend.kv_meta.lock()
                        .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
                    if let Some(meta) = meta_store.get_mut(&handle.0) {
                        meta.seq_len = (meta.seq_len + seq_len).min(meta.max_seq_len);
                    }
                }
            }

            // Update gpu_hidden: DtoD from output
            let output_tid = graph.outputs[0];
            let output_buf = gpu_buffers.get(&output_tid)
                .ok_or_else(|| BE::Metal("output buffer not found".into()))?;
            device.dtod(output_buf, &mut gpu_hidden, stream)
                .map_err(|e| BE::Metal(format!("dtod output→gpu_hidden: {e}")))?;
        }

        // Final: download hidden_state from GPU (once)
        {
            let mut output_host = vec![0u8; hidden_bytes_typed];
            device.dtoh(&gpu_hidden, &mut output_host, stream)
                .map_err(|e| BE::Metal(format!("dtoh final hidden: {e}")))?;
            hidden_state.copy_from_slice(&output_host);
        }
    }

    // ── lm_head projection (GPU) ──
    let lm_comp_dtype = super::jit_helpers::computation_dtype_from_config(config);
    let lm_elem_bytes = lm_comp_dtype.size_bytes();
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size, lm_comp_dtype);
    let lm_entries = metal_compile_graph(device, gpu_profile, gpu_family, &lm_graph)?;

    let lm_head_w = super::gpu_helpers::get_typed_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("lm_head.weight", Some("output.weight")),
        lm_comp_dtype)?;

    let mut lm_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
        std::collections::HashMap::new();

    for (idx, meta) in lm_graph.tensors.iter().enumerate() {
        let tid = TensorId(idx as u32);
        let n_elements = meta.concrete_numel();
        let size_bytes = n_elements * lm_elem_bytes;
        let mut buf = device.alloc(size_bytes)
            .map_err(|e| BE::Metal(format!("alloc lm_head {} failed: {e}", meta.name)))?;

        if meta.name == "input" {
            device.htod(&hidden_state, &mut buf, stream)
                .map_err(|e| BE::Metal(format!("htod lm_head input: {e}")))?;
        } else if meta.name == "w_lm" {
            device.htod(&lm_head_w, &mut buf, stream)
                .map_err(|e| BE::Metal(format!("htod lm_head weight: {e}")))?;
        }

        lm_bufs.insert(tid, buf);
    }

    metal_launch_graph(device, stream, &lm_entries, &lm_bufs, &lm_graph)?;
    device.sync().map_err(|e| BE::Metal(format!("sync lm_head: {e}")))?;

    let logits_tid = lm_graph.outputs[0];
    let logits_buf = lm_bufs.get(&logits_tid)
        .ok_or_else(|| BE::Metal("logits buffer not found".into()))?;
    let logits_bytes = seq_len * vocab_size * lm_elem_bytes;
    let mut logits_host = vec![0u8; logits_bytes];
    device.dtoh(logits_buf, &mut logits_host, stream)
        .map_err(|e| BE::Metal(format!("dtoh logits: {e}")))?;

    let logits_f32 = super::jit_helpers::typed_bytes_to_f32(&logits_host, lm_comp_dtype);

    // Split logits per sequence
    let mut results = Vec::new();
    let mut offset = 0;
    for seq in &input.sequences {
        let tok_count = seq.tokens.len();
        let last_pos = offset + tok_count - 1;
        let seq_logits = logits_f32[last_pos * vocab_size..(last_pos + 1) * vocab_size].to_vec();
        results.push(LogitsHandle { data: seq_logits });
        offset += tok_count;
    }

    Ok(results)
}
