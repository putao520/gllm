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

use super::KvLayoutStrategy;

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
    /// KV cache layout strategy (topology-derived, not bool).
    pub layout: KvLayoutStrategy,
}
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl GpuKvCacheMeta {
    pub fn dtype_size(&self) -> usize { self.kv_dtype.size_bytes() }

    pub fn from_config(config: &KvCacheConfig, device_ptr: u64) -> Self {
        let kv_dtype = config.kv_dtype;
        let layout = if config.is_mla() { KvLayoutStrategy::MlaCompressed } else { KvLayoutStrategy::Standard };
        let kv_dim = config.kv_dim();
        let total_bytes = match layout {
            KvLayoutStrategy::MlaCompressed =>
                config.num_layers() * config.max_seq_len() * kv_dim * config.dtype_size(),
            KvLayoutStrategy::Standard =>
                config.num_layers() * 2 * config.num_heads() * config.max_seq_len()
                    * config.head_dim() * config.dtype_size(),
        };
        Self {
            device_ptr,
            total_bytes,
            num_layers: config.num_layers(),
            num_kv_heads: config.num_heads(),
            max_seq_len: config.max_seq_len(),
            head_dim: match layout {
                KvLayoutStrategy::MlaCompressed => kv_dim,
                KvLayoutStrategy::Standard => config.head_dim(),
            },
            kv_dtype,
            page_size: config.page_size,
            seq_len: 0,
            layout,
        }
    }

    /// Bytes per page across all layers.
    /// Standard: layers × 2 (K+V) × kv_heads × page_size × head_dim × dtype.
    /// MLA: layers × page_size × (d_c + d_rope) × dtype (single compressed vector).
    pub fn page_bytes(&self) -> usize {
        match self.layout {
            KvLayoutStrategy::MlaCompressed =>
                self.num_layers * self.page_size * self.head_dim * self.dtype_size(),
            KvLayoutStrategy::Standard =>
                self.num_layers * 2 * self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size(),
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
    /// KV cache layout strategy (topology-derived, not bool).
    pub layout: KvLayoutStrategy,
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
        layout: KvLayoutStrategy,
    ) -> Self {
        let dtype_size = kv_dtype.size_bytes();
        let page_stride = match layout {
            KvLayoutStrategy::MlaCompressed => num_layers * page_size * kv_dim * dtype_size,
            KvLayoutStrategy::Standard => num_layers * 2 * num_kv_heads * page_size * head_dim * dtype_size,
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
            layout,
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
        match self.layout {
            KvLayoutStrategy::MlaCompressed => {
                let layer_stride = self.page_size * self.kv_dim * self.dtype_size();
                layer * layer_stride + token * self.kv_dim * self.dtype_size()
            }
            KvLayoutStrategy::Standard => {
                let layer_stride = 2 * self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size();
                let half_stride = self.num_kv_heads * self.page_size * self.head_dim * self.dtype_size();
                let head_stride = self.page_size * self.head_dim * self.dtype_size();
                let token_stride = self.head_dim * self.dtype_size();
                layer * layer_stride + kv_half * half_stride + head * head_stride + token * token_stride
            }
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

// ---------------------------------------------------------------------------
// GPU Kernel Launch Configuration (REQ-KERNELS-GPU-001)
// ---------------------------------------------------------------------------

/// Unified configuration for GPU mega-kernel launch.
///
/// All GPU backends (CUDA/HIP/Metal) use the same parameter layout.
/// This struct centralizes parameter construction and validation,
/// eliminating per-backend duplication.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
#[derive(Debug, Clone)]
pub(crate) struct GpuKernelLaunchConfig {
    /// Device pointer to input token IDs.
    pub input_ids_gpu: u64,
    /// Device pointer to weight blob.
    pub weight_blob_gpu: u64,
    /// Device pointer to KV cache buffer (0 = no KV cache).
    pub kv_cache_gpu: u64,
    /// Device pointer to position IDs.
    pub positions_gpu: u64,
    /// Device pointer to auxiliary data (seq_lens, etc.).
    pub aux_ptr: u64,
    /// Batch size.
    pub batch_size: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Device pointer to scratchpad buffer.
    pub scratchpad_gpu: u64,
    /// Device pointer to output buffer.
    pub output_buf_gpu: u64,
    /// Temperature as IEEE 754 bits.
    pub temperature_bits: usize,
    /// Top-k sampling parameter.
    pub top_k: usize,
    /// Top-p as IEEE 754 bits.
    pub top_p_bits: usize,
    /// Maximum new tokens to generate.
    pub max_new_tokens: usize,
    /// End-of-sequence token ID.
    pub eos_token_id: usize,
    /// Device pointer to hook context.
    pub hook_ctx_ptr: u64,
    /// Device pointer to telemetry buffer.
    pub telemetry_ptr: u64,
    /// Session position offset.
    pub session_position: usize,
    /// Device pointer to fused hidden state (multimodal).
    pub fused_hidden_ptr: u64,
    /// Number of multimodal tokens.
    pub num_mm_tokens: usize,
    /// Device pointer to callback table.
    pub callback_table_ptr: u64,
    /// Device pointer to page table (0 = contiguous KV).
    pub page_table_ptr: u64,
    /// Device pointer to batch context.
    pub batch_ctx_ptr: u64,
}

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
impl GpuKernelLaunchConfig {
    /// Validate the launch configuration before kernel invocation.
    ///
    /// Returns `Err` if any invariant is violated. This prevents
    /// launching a GPU kernel with invalid parameters that would
    /// cause undefined behavior on the device.
    pub fn validate(&self) -> Result<(), BE> {
        if self.input_ids_gpu == 0 {
            return Err(BE::Other("GpuKernelLaunchConfig: input_ids_gpu is NULL".into()));
        }
        if self.weight_blob_gpu == 0 {
            return Err(BE::Other("GpuKernelLaunchConfig: weight_blob_gpu is NULL".into()));
        }
        if self.scratchpad_gpu == 0 {
            return Err(BE::Other("GpuKernelLaunchConfig: scratchpad_gpu is NULL".into()));
        }
        if self.batch_size == 0 {
            return Err(BE::Other("GpuKernelLaunchConfig: batch_size is 0".into()));
        }
        if self.seq_len == 0 {
            return Err(BE::Other("GpuKernelLaunchConfig: seq_len is 0".into()));
        }
        Ok(())
    }

    /// Build the 22-parameter ABI array for mega-kernel launch.
    ///
    /// The ABI order matches the CPU mega-kernel ABI (SPEC/40).
    /// All pointer values must be valid device pointers.
    pub fn to_mega_kernel_args(&self) -> [usize; 22] {
        [
            self.input_ids_gpu as usize,
            self.weight_blob_gpu as usize,
            self.kv_cache_gpu as usize,
            self.positions_gpu as usize,
            self.aux_ptr as usize,
            self.batch_size,
            self.seq_len,
            self.scratchpad_gpu as usize,
            self.output_buf_gpu as usize,
            self.temperature_bits,
            self.top_k,
            self.top_p_bits,
            self.max_new_tokens,
            self.eos_token_id,
            self.hook_ctx_ptr as usize,
            self.telemetry_ptr as usize,
            self.session_position,
            self.fused_hidden_ptr as usize,
            self.num_mm_tokens,
            self.callback_table_ptr as usize,
            self.page_table_ptr as usize,
            self.batch_ctx_ptr as usize,
        ]
    }
}

// ---------------------------------------------------------------------------
// GPU/CPU Numerical Alignment Verification (REQ-KERNELS-GPU-001)
// ---------------------------------------------------------------------------

/// Default tolerance threshold for GPU/CPU numerical alignment.
/// GPU kernels use mixed-precision (BF16/TF16 tensor cores) while
/// CPU uses F32, so some divergence is expected.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) const GPU_CPU_ALIGN_TOLERANCE: f32 = 1e-3;

/// Verify numerical alignment between GPU and CPU outputs.
///
/// Compares two f32 slices element-by-element, checking that the maximum
/// relative error is within the specified tolerance. Returns the maximum
/// relative error observed.
///
/// Returns `Err` if:
/// - Slices have different lengths
/// - Any element has relative error exceeding tolerance
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) fn verify_gpu_cpu_alignment(
    gpu_output: &[f32],
    cpu_output: &[f32],
    tolerance: f32,
) -> Result<f32, BE> {
    if gpu_output.len() != cpu_output.len() {
        return Err(BE::Other(format!(
            "GPU/CPU output length mismatch: GPU={} CPU={}",
            gpu_output.len(),
            cpu_output.len()
        )));
    }

    let mut max_rel_error: f32 = 0.0;
    let mut mismatch_count: usize = 0;
    let mut first_mismatch_idx: Option<usize> = None;

    for (i, (&gpu_val, &cpu_val)) in gpu_output.iter().zip(cpu_output.iter()).enumerate() {
        if cpu_val == 0.0 && gpu_val == 0.0 {
            continue;
        }
        if cpu_val == 0.0 {
            let abs_err = (gpu_val - cpu_val).abs();
            if abs_err > tolerance && first_mismatch_idx.is_none() {
                first_mismatch_idx = Some(i);
                mismatch_count += 1;
            }
            max_rel_error = max_rel_error.max(abs_err);
            continue;
        }
        let rel_err = ((gpu_val - cpu_val) / cpu_val).abs();
        max_rel_error = max_rel_error.max(rel_err);
        if rel_err > tolerance && first_mismatch_idx.is_none() {
            first_mismatch_idx = Some(i);
            mismatch_count += 1;
        }
    }

    if max_rel_error > tolerance {
        if let Some(idx) = first_mismatch_idx {
            return Err(BE::Other(format!(
                "GPU/CPU numerical alignment failed: max_rel_error={:.6} > tolerance={:.6}, \
                 first mismatch at index {} (GPU={:.6} CPU={:.6}), total mismatches={}",
                max_rel_error, tolerance, idx,
                gpu_output.get(idx).copied().unwrap_or(0.0),
                cpu_output.get(idx).copied().unwrap_or(0.0),
                mismatch_count
            )));
        }
    }

    Ok(max_rel_error)
}