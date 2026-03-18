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

/// Type aliases for GPU swap stores.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) type GpuSwapStore = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<StorageKey, Vec<u8>>>>;

#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(super) type GpuKvMetaStore = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<u64, GpuKvCacheMeta>>>;

// ---------------------------------------------------------------------------
// Shared KV cache write helper (used by all GPU backends)
// ---------------------------------------------------------------------------

/// Write k_rope and v_proj host bytes into the GPU KV cache for a single layer.
///
/// `k_host` / `v_host`: raw bytes `[seq_len, kv_dim]` in f32 layout (from dtoh of GPU tensors).
/// Repacks per-head `[seq_len, head_dim]` and writes via `htod_raw` into the correct KV cache offsets.
/// `write_start`: the seq position at which to start writing.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
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
    device: &dyn gllm_kernels::gpu::GpuDevice,
    stream: u64,
) -> Result<(), BE> {
    let dtype_size = 4; // f32
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
        device.htod_raw(&k_packed, dst_k, stream)
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
        device.htod_raw(&v_packed, dst_v, stream)
            .map_err(|e| BE::Other(format!("htod KV cache V failed: {e}")))?;
    }
    Ok(())
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

    let dialect = PtxDialect::new(sm_version);
    let mut ptx = String::new();
    dialect.emit_header(&mut ptx);
    gpu_emit_plan(&dialect, &mut ptx, &plan, graph, Some(&registry))
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
        OpKind::RoPE { head_dim, seq_len, .. } => {
            // Use actual seq_len when available
            let n = *head_dim * (*seq_len).max(1);
            profile.launch_config_1d(n)
        }
        OpKind::LayerNorm { hidden_size, .. } | OpKind::RmsNorm { hidden_size, .. } => {
            let block = (*hidden_size as u32).next_power_of_two()
                .min(profile.max_threads_per_block);
            profile.launch_config_row_wise(1024, block)
        }
        OpKind::Softmax => {
            let block = profile.max_threads_per_block.min(256);
            profile.launch_config_row_wise(1024, block)
        }
        OpKind::Gemm { m, n, .. } | OpKind::GemmBias { m, n, .. } => {
            // Tile size derived from warp_size: sqrt(32)=5→8, sqrt(64)=8
            let tile = ((profile.warp_size as f32).sqrt() as u32).max(8).min(32);
            let grid_x = ((*n as u32) + tile - 1) / tile;
            let grid_y = ((*m as u32) + tile - 1) / tile;
            let shared = (2 * tile * tile * 4).min(profile.shared_mem_per_block);
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
        OpKind::MeanPool { hidden, .. } => {
            profile.launch_config_1d(*hidden)
        }
        OpKind::L2Normalize { hidden } => {
            profile.launch_config_1d(*hidden)
        }
        OpKind::Reshape { .. } | OpKind::Transpose { .. } => {
            profile.launch_config_1d(1) // metadata NOP
        }
        _ => profile.launch_config_1d(1024),
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
            let last_epi = *group.epilogue.last().unwrap();
            let last_op = graph.op(last_epi).unwrap();
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
                BE::Gpu(format!("missing tensor pointer for input {:?}", tid))
            }))
            .collect::<Result<Vec<_>, _>>()?;
        let output_ptr = tensor_ptrs.get(&entry.output_tid).copied().ok_or_else(|| {
            BE::Gpu(format!("missing tensor pointer for output {:?}", entry.output_tid))
        })?;

        // Build typed params based on op kind
        let mut raw_params = build_kernel_params(&entry.op_kind, &input_ptrs, output_ptr)?;

        // Fill in N placeholder for ops that need it
        match &entry.op_kind {
            OpKind::LayerNorm { .. } | OpKind::RmsNorm { .. } => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for LayerNorm/RmsNorm input");
                let n = *input_meta.shape.last().unwrap_or(&1);
                raw_params[2] = n as u64;
            }
            OpKind::Residual | OpKind::Add | OpKind::Mul => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for binary op input");
                let n: usize = input_meta.shape.iter().product();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::Silu | OpKind::Gelu => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for activation input");
                let n: usize = input_meta.shape.iter().product();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::SwiGlu | OpKind::GeGlu => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for SwiGlu/GeGlu input");
                let n: usize = input_meta.shape.iter().product();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::RoPE { .. } => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for RoPE input");
                let seq_len = input_meta.shape[0];
                let last = raw_params.len() - 1;
                raw_params[last] = seq_len as u64;
            }
            OpKind::Softmax => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for Softmax input");
                let n: usize = input_meta.shape.iter().product();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            _ => {}
        }

        // Convert to c_void param array for cuLaunchKernel
        let mut param_ptrs: Vec<*mut c_void> = raw_params.iter_mut()
            .map(|p| p as *mut u64 as *mut c_void)
            .collect();

        device.launch_kernel(entry.func, &entry.config, stream, &mut param_ptrs)
            .map_err(|e| BE::Other(format!(
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
    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
    let (_module, kernel_entries) = cuda_compile_graph(
        device, gpu_profile, sm_version, &graph,
    )?;

    // ── Per-layer GPU execution ──
    for layer in 0..num_layers {
        // Load all weight tensors for this layer (CPU side)
        let layer_weights = super::gpu_helpers::load_bert_layer_weights_gpu(
            weights, backend, layer, seq_len, hidden, inter, transpose_w,
        )?;

        // Allocate GPU buffers for all graph tensors.
        // We keep CudaBuffer objects alive (for dtoh) and build a separate
        // u64 pointer map for kernel launch.
        let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
        let mut tensor_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

        for (idx, meta) in graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let n_elements: usize = meta.shape.iter().product();
            let size_bytes = n_elements * 4; // f32
            let mut buf = device.alloc(size_bytes)
                .map_err(|e| BE::Other(format!("GPU alloc failed for {}: {e}", meta.name)))?;

            // Upload data for input and weight tensors
            if meta.name == "input" {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        hidden_state.as_ptr() as *const u8,
                        hidden_state.len() * 4,
                    )
                };
                device.htod(bytes, &mut buf, stream)
                    .map_err(|e| BE::Other(format!("htod input failed: {e}")))?;
            } else if let Some(data) = layer_weights.get(&meta.name) {
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                device.htod(bytes, &mut buf, stream)
                    .map_err(|e| BE::Other(format!("htod {} failed: {e}", meta.name)))?;
            }

            tensor_ptrs.insert(tid, buf.as_device_ptr());
            gpu_buffers.push((tid, buf));
        }

        // Launch all kernels
        cuda_launch_graph(device, stream, &kernel_entries, &tensor_ptrs, &graph)?;

        // Synchronize
        device.sync()
            .map_err(|e| BE::Other(format!("GPU sync failed: {e}")))?;

        // Find the output buffer and download
        let output_tid = graph.outputs[0];
        let output_buf = gpu_buffers.iter()
            .find(|(tid, _)| *tid == output_tid)
            .map(|(_, buf)| buf)
            .ok_or_else(|| BE::Other("output buffer not found".into()))?;
        let output_bytes = seq_len * hidden * 4;
        let mut output_host = vec![0u8; output_bytes];
        device.dtoh(output_buf, &mut output_host, stream)
            .map_err(|e| BE::Other(format!("dtoh output failed: {e}")))?;

        // Copy result back to hidden_state for next layer
        let output_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
        };
        hidden_state.copy_from_slice(output_f32);

        // GPU buffers freed when gpu_buffers drops
    }

    // ── Mean pooling (GPU) ──
    let pool_graph = build_mean_pool_graph(seq_len, hidden);
    let (_pool_module, pool_entries) = cuda_compile_graph(
        device, gpu_profile, sm_version, &pool_graph,
    )?;

    // Input buffer
    let input_bytes = seq_len * hidden * 4;
    let mut gpu_input = device.alloc(input_bytes)
        .map_err(|e| BE::Other(format!("GPU alloc pool input: {e}")))?;
    let hs_bytes = unsafe {
        std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, input_bytes)
    };
    device.htod(hs_bytes, &mut gpu_input, stream)
        .map_err(|e| BE::Other(format!("htod pool input: {e}")))?;

    // Output buffer
    let output_bytes = hidden * 4;
    let gpu_output = device.alloc(output_bytes)
        .map_err(|e| BE::Other(format!("GPU alloc pool output: {e}")))?;

    let mut pool_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
    pool_ptrs.insert(pool_graph.inputs[0], gpu_input.as_device_ptr());
    pool_ptrs.insert(pool_graph.outputs[0], gpu_output.as_device_ptr());

    cuda_launch_graph(device, stream, &pool_entries, &pool_ptrs, &pool_graph)?;
    device.sync().map_err(|e| BE::Other(format!("GPU sync pool: {e}")))?;

    let mut pooled_bytes = vec![0u8; output_bytes];
    device.dtoh(&gpu_output, &mut pooled_bytes, stream)
        .map_err(|e| BE::Other(format!("dtoh pool output: {e}")))?;

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

    // Embedding lookup (CPU)
    let word_emb = super::gpu_helpers::get_f32_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("token_embedding.weight", Some("token_embd.weight")))?;
    let vocab = word_emb.len() / hidden;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    let mut pos = 0;
    for seq in &input.sequences {
        for &tok in &seq.tokens {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
            }
            hidden_state[pos * hidden..(pos + 1) * hidden]
                .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
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
        // ── Incremental decode path ──
        // GPU projection (RmsNorm → Q/K/V Gemm → RoPE) + CPU cached attention + GPU post-attention
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let proj_graph = build_projection_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta);
        let (_proj_mod, proj_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &proj_graph)?;
        let post_graph = build_post_attention_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
        let (_post_mod, post_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &post_graph)?;

        let handle = kv_caches.first().unwrap();
        let (cached_seq_len, half_bytes, total_kv_floats, max_seq_len, head_stride) = {
            let ms = backend.kv_meta.lock()
                .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
            let m = ms.get(&handle.0)
                .ok_or_else(|| BE::Cuda(format!("KV cache handle {} not found", handle.0)))?;
            let hb = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim * m.dtype_size;
            let tkf = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim;
            let hs = m.max_seq_len * m.head_dim * m.dtype_size;
            (m.seq_len, hb, tkf, m.max_seq_len, hs)
        };
        let total_seq = cached_seq_len + seq_len;
        let positions: Vec<u32> = input.sequences.iter()
            .flat_map(|s| {
                let start = s.position as u32;
                (0..s.tokens.len() as u32).map(move |i| start + i)
            })
            .collect();

        for layer in 0..num_layers {
            let layer_weights = super::gpu_helpers::load_decoder_layer_weights_gpu(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter,
            )?;

            // ── GPU: projection graph (RmsNorm → Q/K/V → RoPE) ──
            let mut proj_bufs: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
            let mut proj_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            for (idx, tmeta) in proj_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = tmeta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Cuda(format!("GPU alloc proj {} failed: {e}", tmeta.name)))?;
                if tmeta.name == "input" {
                    let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Cuda(format!("htod proj input: {e}")))?;
                } else if let Some(data) = layer_weights.get(&tmeta.name) {
                    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Cuda(format!("htod proj {}: {e}", tmeta.name)))?;
                }
                proj_ptrs.insert(tid, buf.as_device_ptr());
                proj_bufs.push((tid, buf));
            }
            cuda_launch_graph(device, stream, &proj_entries, &proj_ptrs, &proj_graph)?;
            device.sync().map_err(|e| BE::Cuda(format!("GPU sync proj: {e}")))?;

            // Download q_rope, k_rope, v_proj from GPU
            let q_tid = proj_graph.outputs[0]; // q_rope
            let k_tid = proj_graph.outputs[1]; // k_rope
            let v_tid = proj_graph.outputs[2]; // v_proj

            let q_bytes = seq_len * q_dim * 4;
            let q_buf = proj_bufs.iter().find(|(t, _)| *t == q_tid).map(|(_, b)| b).unwrap();
            let mut q_host = vec![0u8; q_bytes];
            device.dtoh(q_buf, &mut q_host, stream).map_err(|e| BE::Cuda(format!("dtoh q_rope: {e}")))?;
            let q_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(q_host.as_ptr() as *const f32, seq_len * q_dim) }.to_vec();

            let kv_bytes = seq_len * kv_dim * 4;
            let k_buf = proj_bufs.iter().find(|(t, _)| *t == k_tid).map(|(_, b)| b).unwrap();
            let mut k_host = vec![0u8; kv_bytes];
            device.dtoh(k_buf, &mut k_host, stream).map_err(|e| BE::Cuda(format!("dtoh k_rope: {e}")))?;

            let v_buf = proj_bufs.iter().find(|(t, _)| *t == v_tid).map(|(_, b)| b).unwrap();
            let mut v_host = vec![0u8; kv_bytes];
            device.dtoh(v_buf, &mut v_host, stream).map_err(|e| BE::Cuda(format!("dtoh v_proj: {e}")))?;

            // Write k_rope/v_proj into GPU KV cache
            gpu_write_kv_cache(
                &k_host, &v_host, kv_dim, seq_len, num_kv_heads, head_dim,
                handle, layer, half_bytes, cached_seq_len, head_stride, device, stream,
            )?;

            if layer == 0 {
                let mut ms = backend.kv_meta.lock()
                    .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
                if let Some(m) = ms.get_mut(&handle.0) {
                    m.seq_len = (m.seq_len + seq_len).min(m.max_seq_len);
                }
            }

            // Download full KV cache to CPU for cached attention
            let (kv_cache_k, kv_cache_v) = download_kv_cache_to_host(handle, half_bytes, total_kv_floats, device, stream)?;

            // CPU cached attention
            let attn_out = cpu_cached_attention(
                &q_f32, &kv_cache_k, &kv_cache_v, &positions,
                layer, total_seq, seq_len, num_heads, num_kv_heads, head_dim, max_seq_len,
            );

            // ── GPU: post-attention graph (O Gemm → Residual → FFN → Residual) ──
            let mut post_bufs: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
            let mut post_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            for (idx, tmeta) in post_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = tmeta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Cuda(format!("GPU alloc post {} failed: {e}", tmeta.name)))?;
                if tmeta.name == "input" {
                    let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Cuda(format!("htod post input: {e}")))?;
                } else if tmeta.name == "attn_out" {
                    let bytes = unsafe { std::slice::from_raw_parts(attn_out.as_ptr() as *const u8, attn_out.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Cuda(format!("htod post attn_out: {e}")))?;
                } else if let Some(data) = layer_weights.get(&tmeta.name) {
                    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Cuda(format!("htod post {}: {e}", tmeta.name)))?;
                }
                post_ptrs.insert(tid, buf.as_device_ptr());
                post_bufs.push((tid, buf));
            }
            cuda_launch_graph(device, stream, &post_entries, &post_ptrs, &post_graph)?;
            device.sync().map_err(|e| BE::Cuda(format!("GPU sync post: {e}")))?;

            // Download output
            let output_tid = post_graph.outputs[0];
            let output_buf = post_bufs.iter().find(|(tid, _)| *tid == output_tid).map(|(_, buf)| buf)
                .ok_or_else(|| BE::Cuda("post output buffer not found".into()))?;
            let mut output_host = vec![0u8; seq_len * hidden * 4];
            device.dtoh(output_buf, &mut output_host, stream)
                .map_err(|e| BE::Cuda(format!("dtoh post output: {e}")))?;
            let output_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
            };
            hidden_state.copy_from_slice(output_f32);
        }
    } else {
        // ── Prefill path (existing) ──
        let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta);
        let (_module, kernel_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &graph)?;

        for layer in 0..num_layers {
            let layer_weights = super::gpu_helpers::load_decoder_layer_weights_gpu(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter,
            )?;

            let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
            let mut tensor_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

            for (idx, meta) in graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = meta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Cuda(format!("GPU alloc failed for {}: {e}", meta.name)))?;

                if meta.name == "input" {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4)
                    };
                    device.htod(bytes, &mut buf, stream)
                        .map_err(|e| BE::Cuda(format!("htod input failed: {e}")))?;
                } else if let Some(data) = layer_weights.get(&meta.name) {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                    };
                    device.htod(bytes, &mut buf, stream)
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

                let kv_bytes = seq_len * kv_dim * 4;
                let k_buf = gpu_buffers.iter().find(|(t, _)| *t == k_tid).map(|(_, b)| b)
                    .ok_or_else(|| BE::Cuda("k_rope buffer not found".into()))?;
                let mut k_host = vec![0u8; kv_bytes];
                device.dtoh(k_buf, &mut k_host, stream)
                    .map_err(|e| BE::Cuda(format!("dtoh k_rope failed: {e}")))?;

                let v_buf = gpu_buffers.iter().find(|(t, _)| *t == v_tid).map(|(_, b)| b)
                    .ok_or_else(|| BE::Cuda("v_proj buffer not found".into()))?;
                let mut v_host = vec![0u8; kv_bytes];
                device.dtoh(v_buf, &mut v_host, stream)
                    .map_err(|e| BE::Cuda(format!("dtoh v_proj failed: {e}")))?;

                let meta_store = backend.kv_meta.lock()
                    .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
                let meta = meta_store.get(&handle.0)
                    .ok_or_else(|| BE::Cuda(format!("KV cache handle {} not found", handle.0)))?;

                let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let write_start = if layer == 0 { meta.seq_len } else { meta.seq_len.saturating_sub(seq_len) };

                drop(meta_store);

                gpu_write_kv_cache(
                    &k_host, &v_host, kv_dim, seq_len, num_kv_heads, head_dim,
                    handle, layer, half_bytes, write_start, head_stride, device, stream,
                )?;

                if layer == 0 {
                    let mut meta_store = backend.kv_meta.lock()
                        .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
                    if let Some(meta) = meta_store.get_mut(&handle.0) {
                        meta.seq_len = (meta.seq_len + seq_len).min(meta.max_seq_len);
                    }
                }
            }

            // Download output
            let output_tid = graph.outputs[0];
            let output_buf = gpu_buffers.iter()
                .find(|(tid, _)| *tid == output_tid)
                .map(|(_, buf)| buf)
                .ok_or_else(|| BE::Cuda("output buffer not found".into()))?;
            let output_bytes = seq_len * hidden * 4;
            let mut output_host = vec![0u8; output_bytes];
            device.dtoh(output_buf, &mut output_host, stream)
                .map_err(|e| BE::Cuda(format!("dtoh output failed: {e}")))?;

            let output_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
            };
            hidden_state.copy_from_slice(output_f32);
        }
    }

    // ── lm_head projection (GPU) ──
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size);
    let (_lm_module, lm_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &lm_graph)?;

    let lm_head_w = super::gpu_helpers::get_f32_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("lm_head.weight", Some("output.weight")))?;

    let mut gpu_buffers_lm: Vec<(TensorId, gllm_kernels::gpu::cuda::CudaBuffer)> = Vec::new();
    let mut lm_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

    for (idx, meta) in lm_graph.tensors.iter().enumerate() {
        let tid = TensorId(idx as u32);
        let n_elements: usize = meta.shape.iter().product();
        let size_bytes = n_elements * 4;
        let mut buf = device.alloc(size_bytes)
            .map_err(|e| BE::Cuda(format!("GPU alloc lm_head {} failed: {e}", meta.name)))?;

        if meta.name == "input" {
            let bytes = unsafe {
                std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4)
            };
            device.htod(bytes, &mut buf, stream)
                .map_err(|e| BE::Cuda(format!("htod lm_head input: {e}")))?;
        } else if meta.name == "w_lm" {
            let bytes = unsafe {
                std::slice::from_raw_parts(lm_head_w.as_ptr() as *const u8, lm_head_w.len() * 4)
            };
            device.htod(bytes, &mut buf, stream)
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
    let logits_bytes = seq_len * vocab_size * 4;
    let mut logits_host = vec![0u8; logits_bytes];
    device.dtoh(logits_buf, &mut logits_host, stream)
        .map_err(|e| BE::Cuda(format!("dtoh logits: {e}")))?;

    let logits_f32: Vec<f32> = unsafe {
        std::slice::from_raw_parts(logits_host.as_ptr() as *const f32, seq_len * vocab_size)
    }.to_vec();

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

    let dialect = HipDialect { gfx_arch };
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
            let last_epi = *group.epilogue.last().unwrap();
            let last_op = graph.op(last_epi).unwrap();
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
                BE::Gpu(format!("missing tensor pointer for input {:?}", tid))
            }))
            .collect::<Result<Vec<_>, _>>()?;
        let output_ptr = tensor_ptrs.get(&entry.output_tid).copied().ok_or_else(|| {
            BE::Gpu(format!("missing tensor pointer for output {:?}", entry.output_tid))
        })?;

        let mut raw_params = build_kernel_params(&entry.op_kind, &input_ptrs, output_ptr)?;

        match &entry.op_kind {
            OpKind::LayerNorm { .. } | OpKind::RmsNorm { .. } => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for LayerNorm/RmsNorm input");
                let n = *input_meta.shape.last().unwrap_or(&1);
                raw_params[2] = n as u64;
            }
            OpKind::Residual | OpKind::Add | OpKind::Mul => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for binary op input");
                let n: usize = input_meta.shape.iter().product();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::Silu | OpKind::Gelu => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for activation input");
                let n: usize = input_meta.shape.iter().product();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::SwiGlu | OpKind::GeGlu => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for SwiGlu/GeGlu input");
                let n: usize = input_meta.shape.iter().product();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            OpKind::RoPE { .. } => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for RoPE input");
                let seq_len = input_meta.shape[0];
                let last = raw_params.len() - 1;
                raw_params[last] = seq_len as u64;
            }
            OpKind::Softmax => {
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for Softmax input");
                let n: usize = input_meta.shape.iter().product();
                let last = raw_params.len() - 1;
                raw_params[last] = n as u64;
            }
            _ => {}
        }

        let mut param_ptrs: Vec<*mut c_void> = raw_params.iter_mut()
            .map(|p| p as *mut u64 as *mut c_void)
            .collect();

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

    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
    let (_module, kernel_entries) = hip_compile_graph(
        device, gpu_profile, gfx_arch, &graph,
    )?;

    for layer in 0..num_layers {
        let layer_weights = super::gpu_helpers::load_bert_layer_weights_gpu(
            weights, backend, layer, seq_len, hidden, inter, transpose_w,
        )?;

        let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
        let mut tensor_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

        for (idx, meta) in graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let n_elements: usize = meta.shape.iter().product();
            let size_bytes = n_elements * 4;
            let mut buf = device.alloc(size_bytes)
                .map_err(|e| BE::Other(format!("GPU alloc failed for {}: {e}", meta.name)))?;

            if meta.name == "input" {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        hidden_state.as_ptr() as *const u8,
                        hidden_state.len() * 4,
                    )
                };
                device.htod(bytes, &mut buf, stream)
                    .map_err(|e| BE::Other(format!("htod input failed: {e}")))?;
            } else if let Some(data) = layer_weights.get(&meta.name) {
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                device.htod(bytes, &mut buf, stream)
                    .map_err(|e| BE::Other(format!("htod {} failed: {e}", meta.name)))?;
            }

            tensor_ptrs.insert(tid, buf.as_device_ptr());
            gpu_buffers.push((tid, buf));
        }

        hip_launch_graph(device, stream, &kernel_entries, &tensor_ptrs, &graph)?;

        device.sync()
            .map_err(|e| BE::Other(format!("GPU sync failed: {e}")))?;

        let output_tid = graph.outputs[0];
        let output_buf = gpu_buffers.iter()
            .find(|(tid, _)| *tid == output_tid)
            .map(|(_, buf)| buf)
            .ok_or_else(|| BE::Other("output buffer not found".into()))?;
        let output_bytes = seq_len * hidden * 4;
        let mut output_host = vec![0u8; output_bytes];
        device.dtoh(output_buf, &mut output_host, stream)
            .map_err(|e| BE::Other(format!("dtoh output failed: {e}")))?;

        let output_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
        };
        hidden_state.copy_from_slice(output_f32);
    }

    // Mean pooling (GPU)
    let pool_graph = build_mean_pool_graph(seq_len, hidden);
    let (_pool_module, pool_entries) = hip_compile_graph(
        device, gpu_profile, gfx_arch, &pool_graph,
    )?;

    let input_bytes = seq_len * hidden * 4;
    let mut gpu_input = device.alloc(input_bytes)
        .map_err(|e| BE::Other(format!("GPU alloc pool input: {e}")))?;
    let hs_bytes = unsafe {
        std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, input_bytes)
    };
    device.htod(hs_bytes, &mut gpu_input, stream)
        .map_err(|e| BE::Other(format!("htod pool input: {e}")))?;

    let output_bytes = hidden * 4;
    let gpu_output = device.alloc(output_bytes)
        .map_err(|e| BE::Other(format!("GPU alloc pool output: {e}")))?;

    let mut pool_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
    pool_ptrs.insert(pool_graph.inputs[0], gpu_input.as_device_ptr());
    pool_ptrs.insert(pool_graph.outputs[0], gpu_output.as_device_ptr());

    hip_launch_graph(device, stream, &pool_entries, &pool_ptrs, &pool_graph)?;
    device.sync().map_err(|e| BE::Other(format!("GPU sync pool: {e}")))?;

    let mut pooled_bytes = vec![0u8; output_bytes];
    device.dtoh(&gpu_output, &mut pooled_bytes, stream)
        .map_err(|e| BE::Other(format!("dtoh pool output: {e}")))?;

    let pooled: Vec<f32> = unsafe {
        std::slice::from_raw_parts(pooled_bytes.as_ptr() as *const f32, hidden)
    }.to_vec();

    Ok(pooled)
}

// ===========================================================================
// Metal GPU compilation & kernel launch helpers
// ===========================================================================

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

    let dialect = MslDialect::new(gpu_family);
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
            let last_epi = *group.epilogue.last().unwrap();
            let last_op = graph.op(last_epi).unwrap();
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

    // Compile BERT layer graph to Metal pipelines (once)
    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
    let kernel_entries = metal_compile_graph(device, gpu_profile, gpu_family, &graph)?;

    // Per-layer GPU execution
    for layer in 0..num_layers {
        let layer_weights = super::gpu_helpers::load_bert_layer_weights_gpu(
            weights, backend, layer, seq_len, hidden, inter, transpose_w,
        )?;

        // Allocate Metal buffers
        let mut gpu_buffers: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
            std::collections::HashMap::new();

        for (idx, meta) in graph.tensors.iter().enumerate() {
            let tid = TensorId(idx as u32);
            let n_elements: usize = meta.shape.iter().product();
            let size_bytes = n_elements * 4;
            let mut buf = device.alloc(size_bytes)
                .map_err(|e| BE::Metal(format!("alloc failed for {}: {e}", meta.name)))?;

            if meta.name == "input" {
                let bytes = unsafe {
                    std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4)
                };
                device.htod(bytes, &mut buf, stream)
                    .map_err(|e| BE::Metal(format!("htod input failed: {e}")))?;
            } else if let Some(data) = layer_weights.get(&meta.name) {
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                };
                device.htod(bytes, &mut buf, stream)
                    .map_err(|e| BE::Metal(format!("htod {} failed: {e}", meta.name)))?;
            }

            gpu_buffers.insert(tid, buf);
        }

        metal_launch_graph(device, stream, &kernel_entries, &gpu_buffers, &graph)?;
        device.sync().map_err(|e| BE::Metal(format!("sync failed: {e}")))?;

        // Download output
        let output_tid = graph.outputs[0];
        let output_buf = gpu_buffers.get(&output_tid)
            .ok_or_else(|| BE::Metal("output buffer not found".into()))?;
        let output_bytes = seq_len * hidden * 4;
        let mut output_host = vec![0u8; output_bytes];
        device.dtoh(output_buf, &mut output_host, stream)
            .map_err(|e| BE::Metal(format!("dtoh output failed: {e}")))?;

        let output_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
        };
        hidden_state.copy_from_slice(output_f32);
    }

    // Mean pooling
    let pool_graph = super::bert_forward::build_mean_pool_graph(seq_len, hidden);
    let pool_entries = metal_compile_graph(device, gpu_profile, gpu_family, &pool_graph)?;

    let input_bytes = seq_len * hidden * 4;
    let mut gpu_input = device.alloc(input_bytes)
        .map_err(|e| BE::Metal(format!("alloc pool input: {e}")))?;
    let hs_bytes = unsafe {
        std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, input_bytes)
    };
    device.htod(hs_bytes, &mut gpu_input, stream)
        .map_err(|e| BE::Metal(format!("htod pool input: {e}")))?;

    let gpu_output = device.alloc(hidden * 4)
        .map_err(|e| BE::Metal(format!("alloc pool output: {e}")))?;

    let mut pool_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
        std::collections::HashMap::new();
    pool_bufs.insert(pool_graph.inputs[0], gpu_input);
    pool_bufs.insert(pool_graph.outputs[0], gpu_output);

    metal_launch_graph(device, stream, &pool_entries, &pool_bufs, &pool_graph)?;
    device.sync().map_err(|e| BE::Metal(format!("sync pool: {e}")))?;

    let output_buf = pool_bufs.get(&pool_graph.outputs[0]).unwrap();
    let mut pooled_bytes = vec![0u8; hidden * 4];
    device.dtoh(output_buf, &mut pooled_bytes, stream)
        .map_err(|e| BE::Metal(format!("dtoh pool output: {e}")))?;

    let pooled: Vec<f32> = unsafe {
        std::slice::from_raw_parts(pooled_bytes.as_ptr() as *const f32, hidden)
    }.to_vec();

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
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_h = num_kv_heads * head_dim;

    // ── Graph inputs ──
    let input = g.add_tensor("input", vec![s, h], dt);

    // Attention weights (q_dim may differ from hidden for Qwen3 etc.)
    let attn_norm_w = g.add_tensor("attn_norm_w", vec![h], dt);
    let w_q = g.add_tensor("w_q", vec![h, q_dim], dt);
    let w_k = g.add_tensor("w_k", vec![h, kv_h], dt);
    let w_v = g.add_tensor("w_v", vec![h, kv_h], dt);
    let w_o = g.add_tensor("w_o", vec![q_dim, h], dt);

    // FFN weights
    let ffn_norm_w = g.add_tensor("ffn_norm_w", vec![h], dt);
    let w_gate = g.add_tensor("w_gate", vec![h, inter], dt);
    let w_up = g.add_tensor("w_up", vec![h, inter], dt);
    let w_down = g.add_tensor("w_down", vec![inter, h], dt);

    g.inputs = vec![input];

    // ── Pre-attention RmsNorm ──
    let attn_normed = g.add_tensor("attn_normed", vec![s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![input, attn_norm_w],
        vec![attn_normed],
        "attn_rms_norm",
    );

    // ── Q/K/V projections ──
    let q_proj = g.add_tensor("q_proj", vec![s, q_dim], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: q_dim, k: h, dtype: dt },
        vec![attn_normed, w_q],
        vec![q_proj],
        "q_proj",
    );

    let k_proj = g.add_tensor("k_proj", vec![s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h, dtype: dt },
        vec![attn_normed, w_k],
        vec![k_proj],
        "k_proj",
    );

    let v_proj = g.add_tensor("v_proj", vec![s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h, dtype: dt },
        vec![attn_normed, w_v],
        vec![v_proj],
        "v_proj",
    );

    // ── RoPE on Q and K ──
    let q_rope = g.add_tensor("q_rope", vec![s, q_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![q_proj],
        vec![q_rope],
        "rope_q",
    );

    let k_rope = g.add_tensor("k_rope", vec![s, kv_h], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![k_proj],
        vec![k_rope],
        "rope_k",
    );

    // ── Causal MultiHeadAttention ──
    let attn_out = g.add_tensor("attn_out", vec![s, q_dim], dt);
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
    let o_proj = g.add_tensor("o_proj", vec![s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: q_dim, dtype: dt },
        vec![attn_out, w_o],
        vec![o_proj],
        "o_proj",
    );

    // ── Attention residual ──
    let attn_residual = g.add_tensor("attn_residual", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![input, o_proj],
        vec![attn_residual],
        "attn_residual",
    );

    // ── Pre-FFN RmsNorm ──
    let ffn_normed = g.add_tensor("ffn_normed", vec![s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![attn_residual, ffn_norm_w],
        vec![ffn_normed],
        "ffn_rms_norm",
    );

    // ── Gate + Up projections ──
    let gate = g.add_tensor("gate", vec![s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h, dtype: dt },
        vec![ffn_normed, w_gate],
        vec![gate],
        "gate_proj",
    );

    let up = g.add_tensor("up", vec![s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h, dtype: dt },
        vec![ffn_normed, w_up],
        vec![up],
        "up_proj",
    );

    // ── SwiGLU: silu(gate) * up ──
    let swiglu_out = g.add_tensor("swiglu_out", vec![s, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate, up], vec![swiglu_out], "swiglu");

    // ── Down projection ──
    let down = g.add_tensor("down", vec![s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: inter, dtype: dt },
        vec![swiglu_out, w_down],
        vec![down],
        "down_proj",
    );

    // ── FFN residual ──
    let output = g.add_tensor("output", vec![s, h], dt);
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
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;
    let kv_h = num_kv_heads * head_dim;

    let input = g.add_tensor("input", vec![s, h], dt);
    let attn_norm_w = g.add_tensor("attn_norm_w", vec![h], dt);
    let w_q = g.add_tensor("w_q", vec![h, q_dim], dt);
    let w_k = g.add_tensor("w_k", vec![h, kv_h], dt);
    let w_v = g.add_tensor("w_v", vec![h, kv_h], dt);

    g.inputs = vec![input];

    let attn_normed = g.add_tensor("attn_normed", vec![s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![input, attn_norm_w],
        vec![attn_normed],
        "attn_rms_norm",
    );

    let q_proj = g.add_tensor("q_proj", vec![s, q_dim], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: q_dim, k: h, dtype: dt },
        vec![attn_normed, w_q],
        vec![q_proj],
        "q_proj",
    );

    let k_proj = g.add_tensor("k_proj", vec![s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h, dtype: dt },
        vec![attn_normed, w_k],
        vec![k_proj],
        "k_proj",
    );

    let v_proj = g.add_tensor("v_proj", vec![s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h, dtype: dt },
        vec![attn_normed, w_v],
        vec![v_proj],
        "v_proj",
    );

    let q_rope = g.add_tensor("q_rope", vec![s, q_dim], dt);
    g.add_op(
        OpKind::RoPE { head_dim, theta: rope_theta },
        vec![q_proj],
        vec![q_rope],
        "rope_q",
    );

    let k_rope = g.add_tensor("k_rope", vec![s, kv_h], dt);
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
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;
    let s = seq_len;
    let h = hidden;
    let q_dim = num_heads * head_dim;

    let input = g.add_tensor("input", vec![s, h], dt);
    let attn_out = g.add_tensor("attn_out", vec![s, q_dim], dt);
    let w_o = g.add_tensor("w_o", vec![q_dim, h], dt);
    let ffn_norm_w = g.add_tensor("ffn_norm_w", vec![h], dt);
    let w_gate = g.add_tensor("w_gate", vec![h, inter], dt);
    let w_up = g.add_tensor("w_up", vec![h, inter], dt);
    let w_down = g.add_tensor("w_down", vec![inter, h], dt);

    g.inputs = vec![input, attn_out];

    let o_proj = g.add_tensor("o_proj", vec![s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: q_dim, dtype: dt },
        vec![attn_out, w_o],
        vec![o_proj],
        "o_proj",
    );

    let attn_residual = g.add_tensor("attn_residual", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![input, o_proj],
        vec![attn_residual],
        "attn_residual",
    );

    let ffn_normed = g.add_tensor("ffn_normed", vec![s, h], dt);
    g.add_op(
        OpKind::RmsNorm { eps },
        vec![attn_residual, ffn_norm_w],
        vec![ffn_normed],
        "ffn_rms_norm",
    );

    let gate = g.add_tensor("gate", vec![s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h, dtype: dt },
        vec![ffn_normed, w_gate],
        vec![gate],
        "gate_proj",
    );

    let up = g.add_tensor("up", vec![s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h, dtype: dt },
        vec![ffn_normed, w_up],
        vec![up],
        "up_proj",
    );

    let swiglu_out = g.add_tensor("swiglu_out", vec![s, inter], dt);
    g.add_op(OpKind::SwiGlu, vec![gate, up], vec![swiglu_out], "swiglu");

    let down = g.add_tensor("down", vec![s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: inter, dtype: dt },
        vec![swiglu_out, w_down],
        vec![down],
        "down_proj",
    );

    let output = g.add_tensor("output", vec![s, h], dt);
    g.add_op(
        OpKind::Residual,
        vec![attn_residual, down],
        vec![output],
        "ffn_residual",
    );

    g.outputs = vec![output];
    g
}

/// CPU-side causal attention using cached K/V from GPU KV cache.
///
/// q_rope: [seq_len, q_dim] — RoPE'd Q for new tokens
/// kv_cache_k: [num_layers * num_kv_heads * max_seq_len * head_dim] — full K cache (flat)
/// kv_cache_v: same layout — full V cache
/// Returns attn_out: [seq_len, q_dim]
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
fn cpu_cached_attention(
    q_rope: &[f32],
    kv_cache_k: &[f32],
    kv_cache_v: &[f32],
    positions: &[u32],
    layer: usize,
    total_seq: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
) -> Vec<f32> {
    let q_dim = num_heads * head_dim;
    let heads_per_group = num_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut attn_out = vec![0.0f32; seq_len * q_dim];

    for h in 0..num_heads {
        let kv_h = h / heads_per_group;
        let cache_base = (layer * num_kv_heads + kv_h) * max_seq_len * head_dim;

        for s in 0..seq_len {
            let q_offset = s * q_dim + h * head_dim;

            let mut scores = vec![0.0f32; total_seq];
            for t in 0..total_seq {
                let k_offset = cache_base + t * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_rope[q_offset + d] * kv_cache_k[k_offset + d];
                }
                scores[t] = dot * scale;
            }

            let cur_pos = positions[s] as usize;
            for t in 0..total_seq {
                if t > cur_pos {
                    scores[t] = f32::NEG_INFINITY;
                }
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for t in 0..total_seq {
                scores[t] = (scores[t] - max_score).exp();
                sum += scores[t];
            }
            if sum > 0.0 {
                for t in 0..total_seq {
                    scores[t] /= sum;
                }
            }

            let out_offset = s * q_dim + h * head_dim;
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for t in 0..total_seq {
                    let v_offset = cache_base + t * head_dim;
                    val += scores[t] * kv_cache_v[v_offset + d];
                }
                attn_out[out_offset + d] = val;
            }
        }
    }

    attn_out
}

/// Download K/V from GPU KV cache to host flat arrays.
/// Returns (k_cache_flat, v_cache_flat) each [num_layers * num_kv_heads * max_seq_len * head_dim].
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
fn download_kv_cache_to_host(
    handle: &KvCacheHandle,
    half_bytes: usize,
    total_kv_floats: usize,
    device: &dyn gllm_kernels::gpu::GpuDevice,
    stream: u64,
) -> Result<(Vec<f32>, Vec<f32>), BE> {
    let mut k_host_bytes = vec![0u8; half_bytes];
    device.dtoh_raw(handle.0, &mut k_host_bytes, stream)
        .map_err(|e| BE::Other(format!("dtoh KV cache K failed: {e}")))?;
    let k_flat: Vec<f32> = unsafe {
        std::slice::from_raw_parts(k_host_bytes.as_ptr() as *const f32, total_kv_floats)
    }.to_vec();

    let mut v_host_bytes = vec![0u8; half_bytes];
    device.dtoh_raw(handle.0 + half_bytes as u64, &mut v_host_bytes, stream)
        .map_err(|e| BE::Other(format!("dtoh KV cache V failed: {e}")))?;
    let v_flat: Vec<f32> = unsafe {
        std::slice::from_raw_parts(v_host_bytes.as_ptr() as *const f32, total_kv_floats)
    }.to_vec();

    Ok((k_flat, v_flat))
}

/// Build a CompilerGraph for the final lm_head projection: hidden -> vocab logits.
#[cfg(any(feature = "cuda", feature = "hip", all(target_os = "macos", feature = "metal")))]
pub(crate) fn build_lm_head_graph(
    seq_len: usize,
    hidden: usize,
    vocab_size: usize,
) -> gllm_kernels::compiler::CompilerGraph {
    use gllm_kernels::compiler::{CompilerGraph, OpKind};
    use gllm_kernels::types::DType;

    let mut g = CompilerGraph::new();
    let dt = DType::F32;

    let input = g.add_tensor("input", vec![seq_len, hidden], dt);
    let w_lm = g.add_tensor("w_lm", vec![hidden, vocab_size], dt);
    g.inputs = vec![input, w_lm];

    let logits = g.add_tensor("logits", vec![seq_len, vocab_size], dt);
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

    // Embedding lookup (CPU)
    let word_emb = super::gpu_helpers::get_f32_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("token_embedding.weight", Some("token_embd.weight")))?;
    let vocab = word_emb.len() / hidden;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    let mut pos = 0;
    for seq in &input.sequences {
        for &tok in &seq.tokens {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
            }
            hidden_state[pos * hidden..(pos + 1) * hidden]
                .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
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
        // ── Incremental decode path ──
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let proj_graph = build_projection_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta);
        let (_proj_mod, proj_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &proj_graph)?;
        let post_graph = build_post_attention_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
        let (_post_mod, post_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &post_graph)?;

        let handle = kv_caches.first().unwrap();
        let (cached_seq_len, half_bytes, total_kv_floats, max_seq_len, head_stride) = {
            let ms = backend.kv_meta.lock()
                .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
            let m = ms.get(&handle.0)
                .ok_or_else(|| BE::Hip(format!("KV cache handle {} not found", handle.0)))?;
            let hb = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim * m.dtype_size;
            let tkf = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim;
            let hs = m.max_seq_len * m.head_dim * m.dtype_size;
            (m.seq_len, hb, tkf, m.max_seq_len, hs)
        };
        let total_seq = cached_seq_len + seq_len;
        let positions: Vec<u32> = input.sequences.iter()
            .flat_map(|s| {
                let start = s.position as u32;
                (0..s.tokens.len() as u32).map(move |i| start + i)
            })
            .collect();

        for layer in 0..num_layers {
            let layer_weights = super::gpu_helpers::load_decoder_layer_weights_gpu(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter,
            )?;

            // ── GPU: projection graph ──
            let mut proj_bufs: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
            let mut proj_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            for (idx, tmeta) in proj_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = tmeta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Hip(format!("GPU alloc proj {} failed: {e}", tmeta.name)))?;
                if tmeta.name == "input" {
                    let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Hip(format!("htod proj input: {e}")))?;
                } else if let Some(data) = layer_weights.get(&tmeta.name) {
                    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Hip(format!("htod proj {}: {e}", tmeta.name)))?;
                }
                proj_ptrs.insert(tid, buf.as_device_ptr());
                proj_bufs.push((tid, buf));
            }
            hip_launch_graph(device, stream, &proj_entries, &proj_ptrs, &proj_graph)?;
            device.sync().map_err(|e| BE::Hip(format!("GPU sync proj: {e}")))?;

            // Download q_rope, k_rope, v_proj
            let q_tid = proj_graph.outputs[0];
            let k_tid = proj_graph.outputs[1];
            let v_tid = proj_graph.outputs[2];

            let q_bytes = seq_len * q_dim * 4;
            let q_buf = proj_bufs.iter().find(|(t, _)| *t == q_tid).map(|(_, b)| b).unwrap();
            let mut q_host = vec![0u8; q_bytes];
            device.dtoh(q_buf, &mut q_host, stream).map_err(|e| BE::Hip(format!("dtoh q_rope: {e}")))?;
            let q_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(q_host.as_ptr() as *const f32, seq_len * q_dim) }.to_vec();

            let kv_bytes = seq_len * kv_dim * 4;
            let k_buf = proj_bufs.iter().find(|(t, _)| *t == k_tid).map(|(_, b)| b).unwrap();
            let mut k_host = vec![0u8; kv_bytes];
            device.dtoh(k_buf, &mut k_host, stream).map_err(|e| BE::Hip(format!("dtoh k_rope: {e}")))?;

            let v_buf = proj_bufs.iter().find(|(t, _)| *t == v_tid).map(|(_, b)| b).unwrap();
            let mut v_host = vec![0u8; kv_bytes];
            device.dtoh(v_buf, &mut v_host, stream).map_err(|e| BE::Hip(format!("dtoh v_proj: {e}")))?;

            // Write k_rope/v_proj into GPU KV cache
            gpu_write_kv_cache(
                &k_host, &v_host, kv_dim, seq_len, num_kv_heads, head_dim,
                handle, layer, half_bytes, cached_seq_len, head_stride, device, stream,
            )?;

            if layer == 0 {
                let mut ms = backend.kv_meta.lock()
                    .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
                if let Some(m) = ms.get_mut(&handle.0) {
                    m.seq_len = (m.seq_len + seq_len).min(m.max_seq_len);
                }
            }

            // Download full KV cache for CPU attention
            let (kv_cache_k, kv_cache_v) = download_kv_cache_to_host(handle, half_bytes, total_kv_floats, device, stream)?;

            // CPU cached attention
            let attn_out = cpu_cached_attention(
                &q_f32, &kv_cache_k, &kv_cache_v, &positions,
                layer, total_seq, seq_len, num_heads, num_kv_heads, head_dim, max_seq_len,
            );

            // ── GPU: post-attention graph ──
            let mut post_bufs: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
            let mut post_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();
            for (idx, tmeta) in post_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = tmeta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Hip(format!("GPU alloc post {} failed: {e}", tmeta.name)))?;
                if tmeta.name == "input" {
                    let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Hip(format!("htod post input: {e}")))?;
                } else if tmeta.name == "attn_out" {
                    let bytes = unsafe { std::slice::from_raw_parts(attn_out.as_ptr() as *const u8, attn_out.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Hip(format!("htod post attn_out: {e}")))?;
                } else if let Some(data) = layer_weights.get(&tmeta.name) {
                    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Hip(format!("htod post {}: {e}", tmeta.name)))?;
                }
                post_ptrs.insert(tid, buf.as_device_ptr());
                post_bufs.push((tid, buf));
            }
            hip_launch_graph(device, stream, &post_entries, &post_ptrs, &post_graph)?;
            device.sync().map_err(|e| BE::Hip(format!("GPU sync post: {e}")))?;

            let output_tid = post_graph.outputs[0];
            let output_buf = post_bufs.iter().find(|(tid, _)| *tid == output_tid).map(|(_, buf)| buf)
                .ok_or_else(|| BE::Hip("post output buffer not found".into()))?;
            let mut output_host = vec![0u8; seq_len * hidden * 4];
            device.dtoh(output_buf, &mut output_host, stream)
                .map_err(|e| BE::Hip(format!("dtoh post output: {e}")))?;
            let output_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
            };
            hidden_state.copy_from_slice(output_f32);
        }
    } else {
        // ── Prefill path (existing) ──
        let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta);
        let (_module, kernel_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &graph)?;

        for layer in 0..num_layers {
            let layer_weights = super::gpu_helpers::load_decoder_layer_weights_gpu(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter,
            )?;

            let mut gpu_buffers: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
            let mut tensor_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

            for (idx, meta) in graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = meta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Hip(format!("GPU alloc failed for {}: {e}", meta.name)))?;

                if meta.name == "input" {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4)
                    };
                    device.htod(bytes, &mut buf, stream)
                        .map_err(|e| BE::Hip(format!("htod input failed: {e}")))?;
                } else if let Some(data) = layer_weights.get(&meta.name) {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                    };
                    device.htod(bytes, &mut buf, stream)
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

                let kv_bytes = seq_len * kv_dim * 4;
                let k_buf = gpu_buffers.iter().find(|(t, _)| *t == k_tid).map(|(_, b)| b)
                    .ok_or_else(|| BE::Hip("k_rope buffer not found".into()))?;
                let mut k_host = vec![0u8; kv_bytes];
                device.dtoh(k_buf, &mut k_host, stream)
                    .map_err(|e| BE::Hip(format!("dtoh k_rope failed: {e}")))?;

                let v_buf = gpu_buffers.iter().find(|(t, _)| *t == v_tid).map(|(_, b)| b)
                    .ok_or_else(|| BE::Hip("v_proj buffer not found".into()))?;
                let mut v_host = vec![0u8; kv_bytes];
                device.dtoh(v_buf, &mut v_host, stream)
                    .map_err(|e| BE::Hip(format!("dtoh v_proj failed: {e}")))?;

                let meta_store = backend.kv_meta.lock()
                    .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
                let meta = meta_store.get(&handle.0)
                    .ok_or_else(|| BE::Hip(format!("KV cache handle {} not found", handle.0)))?;

                let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let write_start = if layer == 0 { meta.seq_len } else { meta.seq_len.saturating_sub(seq_len) };

                drop(meta_store);

                gpu_write_kv_cache(
                    &k_host, &v_host, kv_dim, seq_len, num_kv_heads, head_dim,
                    handle, layer, half_bytes, write_start, head_stride, device, stream,
                )?;

                if layer == 0 {
                    let mut meta_store = backend.kv_meta.lock()
                        .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
                    if let Some(meta) = meta_store.get_mut(&handle.0) {
                        meta.seq_len = (meta.seq_len + seq_len).min(meta.max_seq_len);
                    }
                }
            }

            // Download output
            let output_tid = graph.outputs[0];
            let output_buf = gpu_buffers.iter()
                .find(|(tid, _)| *tid == output_tid)
                .map(|(_, buf)| buf)
                .ok_or_else(|| BE::Hip("output buffer not found".into()))?;
            let output_bytes = seq_len * hidden * 4;
            let mut output_host = vec![0u8; output_bytes];
            device.dtoh(output_buf, &mut output_host, stream)
                .map_err(|e| BE::Hip(format!("dtoh output failed: {e}")))?;

            let output_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
            };
            hidden_state.copy_from_slice(output_f32);
        }
    }

    // ── lm_head projection (GPU) ──
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size);
    let (_lm_module, lm_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &lm_graph)?;

    let lm_head_w = super::gpu_helpers::get_f32_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("lm_head.weight", Some("output.weight")))?;

    let mut gpu_buffers_lm: Vec<(TensorId, gllm_kernels::gpu::hip::HipBuffer)> = Vec::new();
    let mut lm_ptrs: std::collections::HashMap<TensorId, u64> = std::collections::HashMap::new();

    for (idx, meta) in lm_graph.tensors.iter().enumerate() {
        let tid = TensorId(idx as u32);
        let n_elements: usize = meta.shape.iter().product();
        let size_bytes = n_elements * 4;
        let mut buf = device.alloc(size_bytes)
            .map_err(|e| BE::Hip(format!("GPU alloc lm_head {} failed: {e}", meta.name)))?;

        if meta.name == "input" {
            let bytes = unsafe {
                std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4)
            };
            device.htod(bytes, &mut buf, stream)
                .map_err(|e| BE::Hip(format!("htod lm_head input: {e}")))?;
        } else if meta.name == "w_lm" {
            let bytes = unsafe {
                std::slice::from_raw_parts(lm_head_w.as_ptr() as *const u8, lm_head_w.len() * 4)
            };
            device.htod(bytes, &mut buf, stream)
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
    let logits_bytes = seq_len * vocab_size * 4;
    let mut logits_host = vec![0u8; logits_bytes];
    device.dtoh(logits_buf, &mut logits_host, stream)
        .map_err(|e| BE::Hip(format!("dtoh logits: {e}")))?;

    let logits_f32: Vec<f32> = unsafe {
        std::slice::from_raw_parts(logits_host.as_ptr() as *const f32, seq_len * vocab_size)
    }.to_vec();

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

    // Embedding lookup (CPU)
    let word_emb = super::gpu_helpers::get_f32_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("token_embedding.weight", Some("token_embd.weight")))?;
    let vocab = word_emb.len() / hidden;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    let mut pos = 0;
    for seq in &input.sequences {
        for &tok in &seq.tokens {
            let v = tok as usize;
            if v >= vocab {
                return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
            }
            hidden_state[pos * hidden..(pos + 1) * hidden]
                .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
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
        // ── Incremental decode path ──
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = num_heads * head_dim;
        let proj_graph = build_projection_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, eps, rope_theta);
        let proj_entries = metal_compile_graph(device, gpu_profile, gpu_family, &proj_graph)?;
        let post_graph = build_post_attention_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
        let post_entries = metal_compile_graph(device, gpu_profile, gpu_family, &post_graph)?;

        let handle = kv_caches.first().unwrap();
        let (cached_seq_len, half_bytes, total_kv_floats, max_seq_len, head_stride) = {
            let ms = backend.kv_meta.lock()
                .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
            let m = ms.get(&handle.0)
                .ok_or_else(|| BE::Metal(format!("KV cache handle {} not found", handle.0)))?;
            let hb = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim * m.dtype_size;
            let tkf = m.num_layers * m.num_kv_heads * m.max_seq_len * m.head_dim;
            let hs = m.max_seq_len * m.head_dim * m.dtype_size;
            (m.seq_len, hb, tkf, m.max_seq_len, hs)
        };
        let total_seq = cached_seq_len + seq_len;
        let positions: Vec<u32> = input.sequences.iter()
            .flat_map(|s| {
                let start = s.position as u32;
                (0..s.tokens.len() as u32).map(move |i| start + i)
            })
            .collect();

        for layer in 0..num_layers {
            let layer_weights = super::gpu_helpers::load_decoder_layer_weights_gpu(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter,
            )?;

            // ── GPU: projection graph ──
            let mut proj_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
                std::collections::HashMap::new();
            for (idx, tmeta) in proj_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = tmeta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Metal(format!("alloc proj {} failed: {e}", tmeta.name)))?;
                if tmeta.name == "input" {
                    let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Metal(format!("htod proj input: {e}")))?;
                } else if let Some(data) = layer_weights.get(&tmeta.name) {
                    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Metal(format!("htod proj {}: {e}", tmeta.name)))?;
                }
                proj_bufs.insert(tid, buf);
            }
            metal_launch_graph(device, stream, &proj_entries, &proj_bufs, &proj_graph)?;
            device.sync().map_err(|e| BE::Metal(format!("sync proj: {e}")))?;

            // Download q_rope, k_rope, v_proj
            let q_tid = proj_graph.outputs[0];
            let k_tid = proj_graph.outputs[1];
            let v_tid = proj_graph.outputs[2];

            let q_bytes = seq_len * q_dim * 4;
            let q_buf = proj_bufs.get(&q_tid).unwrap();
            let mut q_host = vec![0u8; q_bytes];
            device.dtoh(q_buf, &mut q_host, stream).map_err(|e| BE::Metal(format!("dtoh q_rope: {e}")))?;
            let q_f32: Vec<f32> = unsafe { std::slice::from_raw_parts(q_host.as_ptr() as *const f32, seq_len * q_dim) }.to_vec();

            let kv_bytes = seq_len * kv_dim * 4;
            let k_buf = proj_bufs.get(&k_tid).unwrap();
            let mut k_host = vec![0u8; kv_bytes];
            device.dtoh(k_buf, &mut k_host, stream).map_err(|e| BE::Metal(format!("dtoh k_rope: {e}")))?;

            let v_buf = proj_bufs.get(&v_tid).unwrap();
            let mut v_host = vec![0u8; kv_bytes];
            device.dtoh(v_buf, &mut v_host, stream).map_err(|e| BE::Metal(format!("dtoh v_proj: {e}")))?;

            // Write k_rope/v_proj into GPU KV cache
            gpu_write_kv_cache(
                &k_host, &v_host, kv_dim, seq_len, num_kv_heads, head_dim,
                handle, layer, half_bytes, cached_seq_len, head_stride, device, stream,
            )?;

            if layer == 0 {
                let mut ms = backend.kv_meta.lock()
                    .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
                if let Some(m) = ms.get_mut(&handle.0) {
                    m.seq_len = (m.seq_len + seq_len).min(m.max_seq_len);
                }
            }

            // Download full KV cache for CPU attention
            let (kv_cache_k, kv_cache_v) = download_kv_cache_to_host(handle, half_bytes, total_kv_floats, device, stream)?;

            // CPU cached attention
            let attn_out = cpu_cached_attention(
                &q_f32, &kv_cache_k, &kv_cache_v, &positions,
                layer, total_seq, seq_len, num_heads, num_kv_heads, head_dim, max_seq_len,
            );

            // ── GPU: post-attention graph ──
            let mut post_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
                std::collections::HashMap::new();
            for (idx, tmeta) in post_graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = tmeta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Metal(format!("alloc post {} failed: {e}", tmeta.name)))?;
                if tmeta.name == "input" {
                    let bytes = unsafe { std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Metal(format!("htod post input: {e}")))?;
                } else if tmeta.name == "attn_out" {
                    let bytes = unsafe { std::slice::from_raw_parts(attn_out.as_ptr() as *const u8, attn_out.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Metal(format!("htod post attn_out: {e}")))?;
                } else if let Some(data) = layer_weights.get(&tmeta.name) {
                    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
                    device.htod(bytes, &mut buf, stream).map_err(|e| BE::Metal(format!("htod post {}: {e}", tmeta.name)))?;
                }
                post_bufs.insert(tid, buf);
            }
            metal_launch_graph(device, stream, &post_entries, &post_bufs, &post_graph)?;
            device.sync().map_err(|e| BE::Metal(format!("sync post: {e}")))?;

            let output_tid = post_graph.outputs[0];
            let output_buf = post_bufs.get(&output_tid)
                .ok_or_else(|| BE::Metal("post output buffer not found".into()))?;
            let mut output_host = vec![0u8; seq_len * hidden * 4];
            device.dtoh(output_buf, &mut output_host, stream)
                .map_err(|e| BE::Metal(format!("dtoh post output: {e}")))?;
            let output_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
            };
            hidden_state.copy_from_slice(output_f32);
        }
    } else {
        // ── Prefill path (existing) ──
        let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps, rope_theta);
        let kernel_entries = metal_compile_graph(device, gpu_profile, gpu_family, &graph)?;

        for layer in 0..num_layers {
            let layer_weights = super::gpu_helpers::load_decoder_layer_weights_gpu(
                weights, backend, layer, hidden, num_heads * head_dim, num_kv_heads * head_dim, inter,
            )?;

            let mut gpu_buffers: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
                std::collections::HashMap::new();

            for (idx, meta) in graph.tensors.iter().enumerate() {
                let tid = TensorId(idx as u32);
                let n_elements: usize = meta.shape.iter().product();
                let size_bytes = n_elements * 4;
                let mut buf = device.alloc(size_bytes)
                    .map_err(|e| BE::Metal(format!("alloc failed for {}: {e}", meta.name)))?;

                if meta.name == "input" {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4)
                    };
                    device.htod(bytes, &mut buf, stream)
                        .map_err(|e| BE::Metal(format!("htod input failed: {e}")))?;
                } else if let Some(data) = layer_weights.get(&meta.name) {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
                    };
                    device.htod(bytes, &mut buf, stream)
                        .map_err(|e| BE::Metal(format!("htod {} failed: {e}", meta.name)))?;
                }

                gpu_buffers.insert(tid, buf);
            }

            metal_launch_graph(device, stream, &kernel_entries, &gpu_buffers, &graph)?;
            device.sync().map_err(|e| BE::Metal(format!("sync failed: {e}")))?;

            // ── Write K/V to KV cache ──
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

                let kv_bytes = seq_len * kv_dim * 4;
                let k_buf = gpu_buffers.get(&k_tid)
                    .ok_or_else(|| BE::Metal("k_rope buffer not found".into()))?;
                let mut k_host = vec![0u8; kv_bytes];
                device.dtoh(k_buf, &mut k_host, stream)
                    .map_err(|e| BE::Metal(format!("dtoh k_rope failed: {e}")))?;

                let v_buf = gpu_buffers.get(&v_tid)
                    .ok_or_else(|| BE::Metal("v_proj buffer not found".into()))?;
                let mut v_host = vec![0u8; kv_bytes];
                device.dtoh(v_buf, &mut v_host, stream)
                    .map_err(|e| BE::Metal(format!("dtoh v_proj failed: {e}")))?;

                let meta_store = backend.kv_meta.lock()
                    .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
                let meta = meta_store.get(&handle.0)
                    .ok_or_else(|| BE::Metal(format!("KV cache handle {} not found", handle.0)))?;

                let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
                let write_start = if layer == 0 { meta.seq_len } else { meta.seq_len.saturating_sub(seq_len) };

                drop(meta_store);

                gpu_write_kv_cache(
                    &k_host, &v_host, kv_dim, seq_len, num_kv_heads, head_dim,
                    handle, layer, half_bytes, write_start, head_stride, device, stream,
                )?;

                if layer == 0 {
                    let mut meta_store = backend.kv_meta.lock()
                        .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
                    if let Some(meta) = meta_store.get_mut(&handle.0) {
                        meta.seq_len = (meta.seq_len + seq_len).min(meta.max_seq_len);
                    }
                }
            }

            let output_tid = graph.outputs[0];
            let output_buf = gpu_buffers.get(&output_tid)
                .ok_or_else(|| BE::Metal("output buffer not found".into()))?;
            let output_bytes = seq_len * hidden * 4;
            let mut output_host = vec![0u8; output_bytes];
            device.dtoh(output_buf, &mut output_host, stream)
                .map_err(|e| BE::Metal(format!("dtoh output failed: {e}")))?;

            let output_f32: &[f32] = unsafe {
                std::slice::from_raw_parts(output_host.as_ptr() as *const f32, seq_len * hidden)
            };
            hidden_state.copy_from_slice(output_f32);
        }
    }

    // ── lm_head projection (GPU) ──
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size);
    let lm_entries = metal_compile_graph(device, gpu_profile, gpu_family, &lm_graph)?;

    let lm_head_w = super::gpu_helpers::get_f32_data_gpu(weights, backend,
        &crate::weight_names::embedding_aliases("lm_head.weight", Some("output.weight")))?;

    let mut lm_bufs: std::collections::HashMap<TensorId, gllm_kernels::gpu::metal::MetalBuffer> =
        std::collections::HashMap::new();

    for (idx, meta) in lm_graph.tensors.iter().enumerate() {
        let tid = TensorId(idx as u32);
        let n_elements: usize = meta.shape.iter().product();
        let size_bytes = n_elements * 4;
        let mut buf = device.alloc(size_bytes)
            .map_err(|e| BE::Metal(format!("alloc lm_head {} failed: {e}", meta.name)))?;

        if meta.name == "input" {
            let bytes = unsafe {
                std::slice::from_raw_parts(hidden_state.as_ptr() as *const u8, hidden_state.len() * 4)
            };
            device.htod(bytes, &mut buf, stream)
                .map_err(|e| BE::Metal(format!("htod lm_head input: {e}")))?;
        } else if meta.name == "w_lm" {
            let bytes = unsafe {
                std::slice::from_raw_parts(lm_head_w.as_ptr() as *const u8, lm_head_w.len() * 4)
            };
            device.htod(bytes, &mut buf, stream)
                .map_err(|e| BE::Metal(format!("htod lm_head weight: {e}")))?;
        }

        lm_bufs.insert(tid, buf);
    }

    metal_launch_graph(device, stream, &lm_entries, &lm_bufs, &lm_graph)?;
    device.sync().map_err(|e| BE::Metal(format!("sync lm_head: {e}")))?;

    let logits_tid = lm_graph.outputs[0];
    let logits_buf = lm_bufs.get(&logits_tid)
        .ok_or_else(|| BE::Metal("logits buffer not found".into()))?;
    let logits_bytes = seq_len * vocab_size * 4;
    let mut logits_host = vec![0u8; logits_bytes];
    device.dtoh(logits_buf, &mut logits_host, stream)
        .map_err(|e| BE::Metal(format!("dtoh logits: {e}")))?;

    let logits_f32: Vec<f32> = unsafe {
        std::slice::from_raw_parts(logits_host.as_ptr() as *const f32, seq_len * vocab_size)
    }.to_vec();

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
