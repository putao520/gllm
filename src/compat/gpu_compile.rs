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
            profile.launch_config_1d(65536)
        }
        OpKind::RoPE { head_dim, .. } => {
            // One thread per (position, dimension_pair) — covers all heads
            profile.launch_config_1d(*head_dim * 1024)
        }
        OpKind::LayerNorm { .. } | OpKind::RmsNorm { .. } => {
            profile.launch_config_row_wise(1024, 256)
        }
        OpKind::Softmax => {
            profile.launch_config_row_wise(1024, 256)
        }
        OpKind::Gemm { m, n, .. } | OpKind::GemmBias { m, n, .. } => {
            let tile: u32 = 16;
            let grid_x = ((*n as u32) + tile - 1) / tile;
            let grid_y = ((*m as u32) + tile - 1) / tile;
            gllm_kernels::gpu::LaunchConfig {
                grid_dim: [grid_x * grid_y, 1, 1],
                block_dim: [tile * tile, 1, 1],
                shared_mem_bytes: 2 * tile * tile * 4,
            }
        }
        OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
            let block = (*head_dim as u32).next_power_of_two().min(256);
            gllm_kernels::gpu::LaunchConfig {
                grid_dim: [(*seq_len as u32) * (*num_heads as u32), 1, 1],
                block_dim: [block, 1, 1],
                shared_mem_bytes: ((*seq_len + 2) as u32) * 4,
            }
        }
        OpKind::MeanPool { hidden, .. } => {
            profile.launch_config_1d(*hidden)
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
) -> Vec<u64> {
    use gllm_kernels::compiler::OpKind;

    match op_kind {
        OpKind::GemmBias { m, n, k } => {
            // Kernel: (A, B, C, M, N, K, bias)
            vec![
                input_ptrs[0], // A
                input_ptrs[1], // B
                output_ptr,    // C
                *m as u64, *n as u64, *k as u64,
                input_ptrs[2], // bias
            ]
        }
        OpKind::Gemm { m, n, k } => {
            // Kernel: (A, B, C, M, N, K)
            vec![
                input_ptrs[0], // A
                input_ptrs[1], // B
                output_ptr,    // C
                *m as u64, *n as u64, *k as u64,
            ]
        }
        OpKind::MultiHeadAttention { seq_len, num_heads, head_dim } => {
            // Kernel: (Q, K, V, output, seq_len, num_heads, head_dim)
            vec![
                input_ptrs[0], // Q
                input_ptrs[1], // K
                input_ptrs[2], // V
                output_ptr,
                *seq_len as u64, *num_heads as u64, *head_dim as u64,
            ]
        }
        OpKind::LayerNorm { .. } => {
            // Kernel: (input, output, N, weight, bias)
            let mut params = vec![input_ptrs[0], output_ptr];
            // N will be added by caller based on tensor shape
            params.push(0); // placeholder for N
            if input_ptrs.len() > 1 { params.push(input_ptrs[1]); } // weight
            if input_ptrs.len() > 2 { params.push(input_ptrs[2]); } // bias
            params
        }
        OpKind::RmsNorm { .. } => {
            // Kernel: (input, output, N, weight)
            let mut params = vec![input_ptrs[0], output_ptr];
            params.push(0); // placeholder for N
            if input_ptrs.len() > 1 { params.push(input_ptrs[1]); } // weight
            params
        }
        OpKind::Residual | OpKind::Add | OpKind::Mul => {
            // Kernel: (A, B, output, N)
            vec![input_ptrs[0], input_ptrs[1], output_ptr, 0] // N placeholder
        }
        OpKind::Silu | OpKind::Gelu => {
            // Kernel: (input, output, N)
            vec![input_ptrs[0], output_ptr, 0] // N placeholder
        }
        OpKind::SwiGlu | OpKind::GeGlu => {
            // Kernel: (gate, up, output, N)
            vec![input_ptrs[0], input_ptrs[1], output_ptr, 0] // N placeholder
        }
        OpKind::RoPE { head_dim, .. } => {
            // Kernel: (input, output, head_dim, seq_len)
            // head_dim is baked in; seq_len will be filled by caller
            vec![input_ptrs[0], output_ptr, *head_dim as u64, 0] // seq_len placeholder
        }
        OpKind::MeanPool { .. } => {
            // Kernel: (input, output) — dims baked in
            vec![input_ptrs[0], output_ptr]
        }
        _ => vec![],
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
            .map(|tid| tensor_ptrs.get(tid).copied().unwrap_or(0))
            .collect();
        let output_ptr = tensor_ptrs.get(&entry.output_tid).copied().unwrap_or(0);

        // Build typed params based on op kind
        let mut raw_params = build_kernel_params(&entry.op_kind, &input_ptrs, output_ptr);

        // Fill in N placeholder for ops that need it
        match &entry.op_kind {
            OpKind::LayerNorm { .. } | OpKind::RmsNorm { .. } => {
                // N = last dimension of input tensor (hidden size)
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for LayerNorm/RmsNorm input");
                let n = *input_meta.shape.last().unwrap_or(&1);
                raw_params[2] = n as u64; // N is at index 2
            }
            OpKind::Residual | OpKind::Add | OpKind::Mul => {
                // N = total f32 elements
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
                // seq_len placeholder at last position
                let input_meta = graph.tensor(entry.input_tids[0])
                    .expect("tensor meta for RoPE input");
                let seq_len = input_meta.shape[0];
                let last = raw_params.len() - 1;
                raw_params[last] = seq_len as u64;
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
    let transpose_w = needs_weight_transpose_cuda(weights);

    // ── Step (a-d): Embedding lookup + LayerNorm (CPU, same as CPU path) ──
    let mut hidden_state = embed_tokens_cpu(tokens, weights, backend, config)?;

    // ── Compile BERT layer graph to PTX (once, reused across layers) ──
    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
    let (_module, kernel_entries) = cuda_compile_graph(
        device, gpu_profile, sm_version, &graph,
    )?;

    // ── Per-layer GPU execution ──
    for layer in 0..num_layers {
        // Load all weight tensors for this layer (CPU side)
        let layer_weights = load_bert_layer_weights_cuda(
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

/// Embedding lookup and LayerNorm on CPU (shared between CPU and GPU paths).
#[cfg(feature = "cuda")]
fn embed_tokens_cpu<E: Element>(
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, CudaBackend<E>>,
    backend: &CudaBackend<E>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    use gllm_kernels::Kernels;
    let kern = gllm_kernels::backend::CpuKernels::<f32>::new();
    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let eps = config.norm_eps;

    // Word embeddings
    let word_emb = get_f32_data_cuda(weights, backend,
        &crate::weight_names::embedding_aliases("word_embeddings.weight", Some("token_embd.weight")))?;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    let vocab = word_emb.len() / hidden;
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= vocab {
            return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
        }
        hidden_state[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
    }

    // Position embeddings
    let pos_emb = get_f32_data_cuda(weights, backend,
        &crate::weight_names::embedding_aliases("position_embeddings.weight", Some("position_embd.weight")))?;
    let max_pos = pos_emb.len() / hidden;
    for s in 0..seq_len {
        if s >= max_pos { break; }
        for i in 0..hidden {
            hidden_state[s * hidden + i] += pos_emb[s * hidden + i];
        }
    }

    // Token type embeddings (type 0)
    let tt_emb = get_f32_data_cuda(weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")))?;
    if tt_emb.len() >= hidden {
        for s in 0..seq_len {
            for i in 0..hidden {
                hidden_state[s * hidden + i] += tt_emb[i];
            }
        }
    }

    // Embedding LayerNorm
    let emb_ln_w = get_f32_data_cuda(weights, backend,
        &crate::weight_names::embedding_aliases("LayerNorm.weight", Some("token_embd_norm.weight")))?;
    let emb_ln_b = get_bias_data_cuda(weights,
        &crate::weight_names::embedding_aliases("LayerNorm.bias", Some("token_embd_norm.bias")), hidden);
    let mut normed = vec![0.0f32; hidden];
    for s in 0..seq_len {
        let row = &hidden_state[s * hidden..(s + 1) * hidden];
        kern.layer_norm(row, &emb_ln_w, &emb_ln_b, &mut normed, eps);
        hidden_state[s * hidden..(s + 1) * hidden].copy_from_slice(&normed);
    }

    Ok(hidden_state)
}

/// Load all weight tensors for a single BERT encoder layer (CUDA variant).
#[cfg(feature = "cuda")]
fn load_bert_layer_weights_cuda<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CudaBackend<E>>,
    backend: &CudaBackend<E>,
    layer: usize,
    _seq_len: usize,
    hidden: usize,
    inter: usize,
    transpose: bool,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_cuda(weights, backend, $aliases)?);
        };
    }
    macro_rules! load_bias {
        ($graph_name:expr, $aliases:expr, $size:expr) => {
            m.insert($graph_name.to_string(), get_bias_data_cuda(weights, $aliases, $size));
        };
    }

    load!("w_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.weight", Some("attn_q.weight")));
    load_bias!("b_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.bias", Some("attn_q.bias")), hidden);
    load!("w_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.weight", Some("attn_k.weight")));
    load_bias!("b_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.bias", Some("attn_k.bias")), hidden);
    load!("w_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.weight", Some("attn_v.weight")));
    load_bias!("b_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.bias", Some("attn_v.bias")), hidden);
    load!("w_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.weight", Some("attn_output.weight")));
    load_bias!("b_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.bias", Some("attn_output.bias")), hidden);
    load!("ln1_w", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.weight", Some("attn_output_norm.weight")));
    load_bias!("ln1_b", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.bias", Some("attn_output_norm.bias")), hidden);
    load!("w_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.weight", Some("ffn_up.weight")));
    load_bias!("b_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.bias", Some("ffn_up.bias")), inter);
    load!("w_down", &crate::weight_names::layer_aliases(layer, "output.dense.weight", Some("ffn_down.weight")));
    load_bias!("b_down", &crate::weight_names::layer_aliases(layer, "output.dense.bias", Some("ffn_down.bias")), hidden);
    load!("ln2_w", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.weight", Some("layer_output_norm.weight")));
    load_bias!("ln2_b", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.bias", Some("layer_output_norm.bias")), hidden);

    // Apply weight transposition if needed (SafeTensors stores [out, in])
    if transpose {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", hidden, hidden);
        t("w_k", hidden, hidden);
        t("w_v", hidden, hidden);
        t("w_o", hidden, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

/// Get f32 data from CudaBackend weights (mirrors get_f32_data for CpuBackend).
#[cfg(feature = "cuda")]
fn get_f32_data_cuda<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CudaBackend<E>>,
    _backend: &CudaBackend<E>,
    aliases: &[impl AsRef<str>],
) -> Result<Vec<f32>, BE> {
    // Try quantized path first
    for name in aliases {
        if let Some(qt) = weights.get_quantized(name.as_ref()) {
            let n_elements = qt.shape.iter().product::<usize>();
            let mut out = vec![0.0f32; n_elements];
            let kern = gllm_kernels::backend::CpuKernels::<E>::new();
            use gllm_kernels::Kernels;
            let blk_elems = qt.quant_type.block_size();
            let blk_bytes = qt.quant_type.block_bytes();
            for (blk_in, blk_out) in qt.data.chunks_exact(blk_bytes).zip(out.chunks_exact_mut(blk_elems)) {
                match qt.quant_type {
                    gllm_kernels::quant::QuantType::Q4_0 => kern.dequant_q4_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4_1 => kern.dequant_q4_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_0 => kern.dequant_q8_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_1 => kern.dequant_q8_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_0 => kern.dequant_q5_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_1 => kern.dequant_q5_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q2K => kern.dequant_q2_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q3K => kern.dequant_q3_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4K => kern.dequant_q4_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5K => kern.dequant_q5_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q6K => kern.dequant_q6_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8K => kern.dequant_q8_k(blk_in, blk_out),
                    _ => {}
                }
            }
            return Ok(out);
        }
    }

    // Fall back to f32 tensor
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    tensor.as_ptr() as *const f32,
                    tensor.len() * std::mem::size_of::<E>() / 4,
                )
            };
            return Ok(slice.to_vec());
        }
    }

    let name_strs: Vec<&str> = aliases.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("Weight not found: {:?}", name_strs)))
}

/// Get bias data from CudaBackend weights (zeros if not found).
#[cfg(feature = "cuda")]
fn get_bias_data_cuda<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CudaBackend<E>>,
    aliases: &[impl AsRef<str>],
    size: usize,
) -> Vec<f32> {
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    tensor.as_ptr() as *const f32,
                    tensor.len() * std::mem::size_of::<E>() / 4,
                )
            };
            return slice.to_vec();
        }
    }
    vec![0.0f32; size]
}

/// Check if weights need transposition (SafeTensors stores [out, in]).
#[cfg(feature = "cuda")]
fn needs_weight_transpose_cuda<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CudaBackend<E>>,
) -> bool {
    crate::weight_names::has_any_embedding_weight(|n| weights.get_tensor(n).is_some())
}

// ===========================================================================
// CUDA decoder forward pass
// ===========================================================================

/// CUDA alloc_kv_cache: allocate GPU buffer for KV cache and register metadata.
#[cfg(feature = "cuda")]
pub(super) fn cuda_alloc_kv_cache(
    backend: &CudaBackend<f32>,
    config: &KvCacheConfig,
) -> Result<KvCacheHandle, BE> {
    use gllm_kernels::gpu::{GpuDevice, GpuBuffer};

    let total_bytes = config.num_layers * 2
        * config.num_heads * config.max_seq_len * config.head_dim * config.dtype_size;

    let buf = backend.device.alloc(total_bytes)
        .map_err(|e| BE::Cuda(format!("KV cache alloc failed ({} bytes): {e}", total_bytes)))?;

    let ptr = buf.as_device_ptr();
    std::mem::forget(buf); // ownership transferred to handle

    // Register metadata for swap support
    let meta = GpuKvCacheMeta::from_config(config, ptr);
    backend.kv_meta.lock()
        .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?
        .insert(ptr, meta);

    Ok(KvCacheHandle(ptr))
}

/// CUDA sample_from_tensor: download logits to CPU and sample.
#[cfg(feature = "cuda")]
pub(super) fn cuda_sample_from_tensor(
    logits: &LogitsHandle,
    _topology: &AttentionTopology,
    vocab_size: usize,
    sampling: &SamplingConfig,
) -> Result<Vec<u32>, BE> {
    Ok(sample_logits_cpu(&logits.data, vocab_size, sampling))
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
    _kv_caches: &mut [KvCacheHandle],
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
    let word_emb = get_f32_data_cuda(weights, backend,
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

    // Compile decoder layer graph (once, reused across layers)
    let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps);
    let (_module, kernel_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &graph)?;

    // Per-layer GPU execution
    for layer in 0..num_layers {
        let layer_weights = load_decoder_layer_weights_cuda(
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

    // ── lm_head projection (GPU) ──
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size);
    let (_lm_module, lm_entries) = cuda_compile_graph(device, gpu_profile, sm_version, &lm_graph)?;

    let lm_head_w = get_f32_data_cuda(weights, backend,
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

/// Load decoder layer weights for CUDA backend.
#[cfg(feature = "cuda")]
fn load_decoder_layer_weights_cuda<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, CudaBackend<E>>,
    backend: &CudaBackend<E>,
    layer: usize,
    hidden: usize,
    q_dim: usize,
    kv_hidden: usize,
    inter: usize,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_cuda(weights, backend, $aliases)?);
        };
    }

    load!("attn_norm_w", &crate::weight_names::layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")));
    load!("w_q", &crate::weight_names::layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")));
    load!("w_k", &crate::weight_names::layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")));
    load!("w_v", &crate::weight_names::layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")));
    load!("w_o", &crate::weight_names::layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")));
    load!("ffn_norm_w", &crate::weight_names::layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")));
    load!("w_gate", &crate::weight_names::layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")));
    load!("w_up", &crate::weight_names::layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")));
    load!("w_down", &crate::weight_names::layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")));

    if needs_weight_transpose_cuda(weights) {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", q_dim, hidden);
        t("w_k", kv_hidden, hidden);
        t("w_v", kv_hidden, hidden);
        t("w_o", hidden, q_dim);
        t("w_gate", inter, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
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
            .map(|tid| tensor_ptrs.get(tid).copied().unwrap_or(0))
            .collect();
        let output_ptr = tensor_ptrs.get(&entry.output_tid).copied().unwrap_or(0);

        let mut raw_params = build_kernel_params(&entry.op_kind, &input_ptrs, output_ptr);

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
    let transpose_w = needs_weight_transpose_hip(weights);

    let mut hidden_state = embed_tokens_cpu_hip(tokens, weights, backend, config)?;

    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
    let (_module, kernel_entries) = hip_compile_graph(
        device, gpu_profile, gfx_arch, &graph,
    )?;

    for layer in 0..num_layers {
        let layer_weights = load_bert_layer_weights_hip(
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

/// Embedding lookup and LayerNorm on CPU (HIP variant).
#[cfg(feature = "hip")]
fn embed_tokens_cpu_hip<E: Element>(
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, HipBackend<E>>,
    backend: &HipBackend<E>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    use gllm_kernels::Kernels;
    let kern = gllm_kernels::backend::CpuKernels::<f32>::new();
    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let eps = config.norm_eps;

    let word_emb = get_f32_data_hip(weights, backend,
        &crate::weight_names::embedding_aliases("word_embeddings.weight", Some("token_embd.weight")))?;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    let vocab = word_emb.len() / hidden;
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= vocab {
            return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
        }
        hidden_state[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
    }

    let pos_emb = get_f32_data_hip(weights, backend,
        &crate::weight_names::embedding_aliases("position_embeddings.weight", Some("position_embd.weight")))?;
    let max_pos = pos_emb.len() / hidden;
    for s in 0..seq_len {
        if s >= max_pos { break; }
        for i in 0..hidden {
            hidden_state[s * hidden + i] += pos_emb[s * hidden + i];
        }
    }

    let tt_emb = get_f32_data_hip(weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")))?;
    if tt_emb.len() >= hidden {
        for s in 0..seq_len {
            for i in 0..hidden {
                hidden_state[s * hidden + i] += tt_emb[i];
            }
        }
    }

    let emb_ln_w = get_f32_data_hip(weights, backend,
        &crate::weight_names::embedding_aliases("LayerNorm.weight", Some("token_embd_norm.weight")))?;
    let emb_ln_b = get_bias_data_hip(weights,
        &crate::weight_names::embedding_aliases("LayerNorm.bias", Some("token_embd_norm.bias")), hidden);
    let mut normed = vec![0.0f32; hidden];
    for s in 0..seq_len {
        let row = &hidden_state[s * hidden..(s + 1) * hidden];
        kern.layer_norm(row, &emb_ln_w, &emb_ln_b, &mut normed, eps);
        hidden_state[s * hidden..(s + 1) * hidden].copy_from_slice(&normed);
    }

    Ok(hidden_state)
}

/// Load all weight tensors for a single BERT encoder layer (HIP variant).
#[cfg(feature = "hip")]
fn load_bert_layer_weights_hip<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, HipBackend<E>>,
    backend: &HipBackend<E>,
    layer: usize,
    _seq_len: usize,
    hidden: usize,
    inter: usize,
    transpose: bool,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_hip(weights, backend, $aliases)?);
        };
    }
    macro_rules! load_bias {
        ($graph_name:expr, $aliases:expr, $size:expr) => {
            m.insert($graph_name.to_string(), get_bias_data_hip(weights, $aliases, $size));
        };
    }

    load!("w_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.weight", Some("attn_q.weight")));
    load_bias!("b_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.bias", Some("attn_q.bias")), hidden);
    load!("w_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.weight", Some("attn_k.weight")));
    load_bias!("b_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.bias", Some("attn_k.bias")), hidden);
    load!("w_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.weight", Some("attn_v.weight")));
    load_bias!("b_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.bias", Some("attn_v.bias")), hidden);
    load!("w_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.weight", Some("attn_output.weight")));
    load_bias!("b_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.bias", Some("attn_output.bias")), hidden);
    load!("ln1_w", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.weight", Some("attn_output_norm.weight")));
    load_bias!("ln1_b", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.bias", Some("attn_output_norm.bias")), hidden);
    load!("w_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.weight", Some("ffn_up.weight")));
    load_bias!("b_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.bias", Some("ffn_up.bias")), inter);
    load!("w_down", &crate::weight_names::layer_aliases(layer, "output.dense.weight", Some("ffn_down.weight")));
    load_bias!("b_down", &crate::weight_names::layer_aliases(layer, "output.dense.bias", Some("ffn_down.bias")), hidden);
    load!("ln2_w", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.weight", Some("layer_output_norm.weight")));
    load_bias!("ln2_b", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.bias", Some("layer_output_norm.bias")), hidden);

    if transpose {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", hidden, hidden);
        t("w_k", hidden, hidden);
        t("w_v", hidden, hidden);
        t("w_o", hidden, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

/// Get f32 data from HipBackend weights.
#[cfg(feature = "hip")]
fn get_f32_data_hip<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, HipBackend<E>>,
    _backend: &HipBackend<E>,
    aliases: &[impl AsRef<str>],
) -> Result<Vec<f32>, BE> {
    for name in aliases {
        if let Some(qt) = weights.get_quantized(name.as_ref()) {
            let n_elements = qt.shape.iter().product::<usize>();
            let mut out = vec![0.0f32; n_elements];
            let kern = gllm_kernels::backend::CpuKernels::<E>::new();
            use gllm_kernels::Kernels;
            let blk_elems = qt.quant_type.block_size();
            let blk_bytes = qt.quant_type.block_bytes();
            for (blk_in, blk_out) in qt.data.chunks_exact(blk_bytes).zip(out.chunks_exact_mut(blk_elems)) {
                match qt.quant_type {
                    gllm_kernels::quant::QuantType::Q4_0 => kern.dequant_q4_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4_1 => kern.dequant_q4_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_0 => kern.dequant_q8_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_1 => kern.dequant_q8_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_0 => kern.dequant_q5_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_1 => kern.dequant_q5_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q2K => kern.dequant_q2_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q3K => kern.dequant_q3_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4K => kern.dequant_q4_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5K => kern.dequant_q5_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q6K => kern.dequant_q6_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8K => kern.dequant_q8_k(blk_in, blk_out),
                    _ => {}
                }
            }
            return Ok(out);
        }
    }

    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    tensor.as_ptr() as *const f32,
                    tensor.len() * std::mem::size_of::<E>() / 4,
                )
            };
            return Ok(slice.to_vec());
        }
    }

    let name_strs: Vec<&str> = aliases.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("Weight not found: {:?}", name_strs)))
}

/// Get bias data from HipBackend weights (zeros if not found).
#[cfg(feature = "hip")]
fn get_bias_data_hip<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, HipBackend<E>>,
    aliases: &[impl AsRef<str>],
    size: usize,
) -> Vec<f32> {
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    tensor.as_ptr() as *const f32,
                    tensor.len() * std::mem::size_of::<E>() / 4,
                )
            };
            return slice.to_vec();
        }
    }
    vec![0.0f32; size]
}

/// Check if weights need transposition (HIP variant).
#[cfg(feature = "hip")]
fn needs_weight_transpose_hip<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, HipBackend<E>>,
) -> bool {
    crate::weight_names::has_any_embedding_weight(|n| weights.get_tensor(n).is_some())
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
        _ => 9, // default to Apple9
    };

    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;
    let inter = config.intermediate_size;
    let eps = config.norm_eps;
    let num_layers = config.num_layers;
    let transpose_w = needs_weight_transpose_metal(weights);

    // Embedding lookup on CPU
    let mut hidden_state = embed_tokens_cpu_metal(tokens, weights, backend, config)?;

    // Compile BERT layer graph to Metal pipelines (once)
    let graph = build_bert_layer_graph(seq_len, hidden, num_heads, head_dim, inter, eps);
    let kernel_entries = metal_compile_graph(device, gpu_profile, gpu_family, &graph)?;

    // Per-layer GPU execution
    for layer in 0..num_layers {
        let layer_weights = load_bert_layer_weights_metal(
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

/// Embedding lookup for Metal backend (CPU side).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn embed_tokens_cpu_metal<E: Element>(
    tokens: &[u32],
    weights: &dyn backend_trait::TensorLookup<E, super::metal_backend::MetalBackend<E>>,
    backend: &super::metal_backend::MetalBackend<E>,
    config: &GeneratorForwardConfig,
) -> Result<Vec<f32>, BE> {
    use gllm_kernels::Kernels;
    let kern = gllm_kernels::backend::CpuKernels::<f32>::new();
    let seq_len = tokens.len();
    let hidden = config.hidden_size;
    let eps = config.norm_eps;

    let word_emb = get_f32_data_metal(weights, backend,
        &crate::weight_names::embedding_aliases("word_embeddings.weight", Some("token_embd.weight")))?;
    let mut hidden_state = vec![0.0f32; seq_len * hidden];
    let vocab = word_emb.len() / hidden;
    for (s, &tok) in tokens.iter().enumerate() {
        let v = tok as usize;
        if v >= vocab {
            return Err(BE::Other(format!("token id {} out of range (vocab {})", tok, vocab)));
        }
        hidden_state[s * hidden..(s + 1) * hidden]
            .copy_from_slice(&word_emb[v * hidden..(v + 1) * hidden]);
    }

    let pos_emb = get_f32_data_metal(weights, backend,
        &crate::weight_names::embedding_aliases("position_embeddings.weight", Some("position_embd.weight")))?;
    let max_pos = pos_emb.len() / hidden;
    for s in 0..seq_len {
        if s >= max_pos { break; }
        for i in 0..hidden {
            hidden_state[s * hidden + i] += pos_emb[s * hidden + i];
        }
    }

    let tt_emb = get_f32_data_metal(weights, backend,
        &crate::weight_names::embedding_aliases("token_type_embeddings.weight", Some("token_types.weight")))?;
    if tt_emb.len() >= hidden {
        for s in 0..seq_len {
            for i in 0..hidden {
                hidden_state[s * hidden + i] += tt_emb[i];
            }
        }
    }

    let emb_ln_w = get_f32_data_metal(weights, backend,
        &crate::weight_names::embedding_aliases("LayerNorm.weight", Some("token_embd_norm.weight")))?;
    let emb_ln_b = get_bias_data_metal(weights,
        &crate::weight_names::embedding_aliases("LayerNorm.bias", Some("token_embd_norm.bias")), hidden);
    let mut normed = vec![0.0f32; hidden];
    for s in 0..seq_len {
        let row = &hidden_state[s * hidden..(s + 1) * hidden];
        kern.layer_norm(row, &emb_ln_w, &emb_ln_b, &mut normed, eps);
        hidden_state[s * hidden..(s + 1) * hidden].copy_from_slice(&normed);
    }

    Ok(hidden_state)
}

/// Load BERT layer weights for Metal backend.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn load_bert_layer_weights_metal<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, super::metal_backend::MetalBackend<E>>,
    backend: &super::metal_backend::MetalBackend<E>,
    layer: usize,
    _seq_len: usize,
    hidden: usize,
    inter: usize,
    transpose: bool,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_metal(weights, backend, $aliases)?);
        };
    }
    macro_rules! load_bias {
        ($graph_name:expr, $aliases:expr, $size:expr) => {
            m.insert($graph_name.to_string(), get_bias_data_metal(weights, $aliases, $size));
        };
    }

    load!("w_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.weight", Some("attn_q.weight")));
    load_bias!("b_q", &crate::weight_names::layer_aliases(layer, "attention.self.query.bias", Some("attn_q.bias")), hidden);
    load!("w_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.weight", Some("attn_k.weight")));
    load_bias!("b_k", &crate::weight_names::layer_aliases(layer, "attention.self.key.bias", Some("attn_k.bias")), hidden);
    load!("w_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.weight", Some("attn_v.weight")));
    load_bias!("b_v", &crate::weight_names::layer_aliases(layer, "attention.self.value.bias", Some("attn_v.bias")), hidden);
    load!("w_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.weight", Some("attn_output.weight")));
    load_bias!("b_o", &crate::weight_names::layer_aliases(layer, "attention.output.dense.bias", Some("attn_output.bias")), hidden);
    load!("ln1_w", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.weight", Some("attn_output_norm.weight")));
    load_bias!("ln1_b", &crate::weight_names::layer_aliases(layer, "attention.output.LayerNorm.bias", Some("attn_output_norm.bias")), hidden);
    load!("w_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.weight", Some("ffn_up.weight")));
    load_bias!("b_up", &crate::weight_names::layer_aliases(layer, "intermediate.dense.bias", Some("ffn_up.bias")), inter);
    load!("w_down", &crate::weight_names::layer_aliases(layer, "output.dense.weight", Some("ffn_down.weight")));
    load_bias!("b_down", &crate::weight_names::layer_aliases(layer, "output.dense.bias", Some("ffn_down.bias")), hidden);
    load!("ln2_w", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.weight", Some("layer_output_norm.weight")));
    load_bias!("ln2_b", &crate::weight_names::layer_aliases(layer, "output.LayerNorm.bias", Some("layer_output_norm.bias")), hidden);

    if transpose {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = super::weight_helpers::transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", hidden, hidden);
        t("w_k", hidden, hidden);
        t("w_v", hidden, hidden);
        t("w_o", hidden, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

/// Get f32 data from Metal backend weights.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn get_f32_data_metal<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, super::metal_backend::MetalBackend<E>>,
    _backend: &super::metal_backend::MetalBackend<E>,
    aliases: &[impl AsRef<str>],
) -> Result<Vec<f32>, BE> {
    for name in aliases {
        if let Some(qt) = weights.get_quantized(name.as_ref()) {
            let n_elements = qt.shape.iter().product::<usize>();
            let mut out = vec![0.0f32; n_elements];
            let kern = gllm_kernels::backend::CpuKernels::<E>::new();
            use gllm_kernels::Kernels;
            let blk_elems = qt.quant_type.block_size();
            let blk_bytes = qt.quant_type.block_bytes();
            for (blk_in, blk_out) in qt.data.chunks_exact(blk_bytes).zip(out.chunks_exact_mut(blk_elems)) {
                match qt.quant_type {
                    gllm_kernels::quant::QuantType::Q4_0 => kern.dequant_q4_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4_1 => kern.dequant_q4_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_0 => kern.dequant_q8_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8_1 => kern.dequant_q8_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_0 => kern.dequant_q5_0(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5_1 => kern.dequant_q5_1(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q2K => kern.dequant_q2_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q3K => kern.dequant_q3_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q4K => kern.dequant_q4_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q5K => kern.dequant_q5_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q6K => kern.dequant_q6_k(blk_in, blk_out),
                    gllm_kernels::quant::QuantType::Q8K => kern.dequant_q8_k(blk_in, blk_out),
                    _ => {}
                }
            }
            return Ok(out);
        }
    }
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    tensor.as_ptr() as *const f32,
                    tensor.len() * std::mem::size_of::<E>() / 4,
                )
            };
            return Ok(slice.to_vec());
        }
    }
    let name_strs: Vec<&str> = aliases.iter().map(|s| s.as_ref()).collect();
    Err(BE::Other(format!("Weight not found: {:?}", name_strs)))
}

/// Get bias data from Metal backend weights (zeros if not found).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn get_bias_data_metal<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, super::metal_backend::MetalBackend<E>>,
    aliases: &[impl AsRef<str>],
    size: usize,
) -> Vec<f32> {
    for name in aliases {
        if let Some(tensor) = weights.get_tensor(name.as_ref()) {
            let slice = unsafe {
                std::slice::from_raw_parts(
                    tensor.as_ptr() as *const f32,
                    tensor.len() * std::mem::size_of::<E>() / 4,
                )
            };
            return slice.to_vec();
        }
    }
    vec![0.0f32; size]
}

/// Check if weights need transposition (Metal variant).
#[cfg(all(target_os = "macos", feature = "metal"))]
fn needs_weight_transpose_metal<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, super::metal_backend::MetalBackend<E>>,
) -> bool {
    crate::weight_names::has_any_embedding_weight(|n| weights.get_tensor(n).is_some())
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
        OpKind::Gemm { m: s, n: q_dim, k: h },
        vec![attn_normed, w_q],
        vec![q_proj],
        "q_proj",
    );

    let k_proj = g.add_tensor("k_proj", vec![s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h },
        vec![attn_normed, w_k],
        vec![k_proj],
        "k_proj",
    );

    let v_proj = g.add_tensor("v_proj", vec![s, kv_h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: kv_h, k: h },
        vec![attn_normed, w_v],
        vec![v_proj],
        "v_proj",
    );

    // ── Causal MultiHeadAttention ──
    let attn_out = g.add_tensor("attn_out", vec![s, q_dim], dt);
    g.add_op(
        OpKind::MultiHeadAttention {
            seq_len: s,
            num_heads,
            head_dim,
        },
        vec![q_proj, k_proj, v_proj],
        vec![attn_out],
        "causal_attention",
    );

    // ── Output projection ──
    let o_proj = g.add_tensor("o_proj", vec![s, h], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: h, k: q_dim },
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
        OpKind::Gemm { m: s, n: inter, k: h },
        vec![ffn_normed, w_gate],
        vec![gate],
        "gate_proj",
    );

    let up = g.add_tensor("up", vec![s, inter], dt);
    g.add_op(
        OpKind::Gemm { m: s, n: inter, k: h },
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
        OpKind::Gemm { m: s, n: h, k: inter },
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
        OpKind::Gemm { m: seq_len, n: vocab_size, k: hidden },
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
    _kv_caches: &mut [KvCacheHandle],
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
    let word_emb = get_f32_data_hip(weights, backend,
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

    // Compile decoder layer graph (once, reused across layers)
    let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps);
    let (_module, kernel_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &graph)?;

    // Per-layer GPU execution
    for layer in 0..num_layers {
        let layer_weights = load_decoder_layer_weights_hip(
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

    // ── lm_head projection (GPU) ──
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size);
    let (_lm_module, lm_entries) = hip_compile_graph(device, gpu_profile, gfx_arch, &lm_graph)?;

    let lm_head_w = get_f32_data_hip(weights, backend,
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

/// Load decoder layer weights for HIP backend.
#[cfg(feature = "hip")]
fn load_decoder_layer_weights_hip<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, HipBackend<E>>,
    backend: &HipBackend<E>,
    layer: usize,
    hidden: usize,
    q_dim: usize,
    kv_hidden: usize,
    inter: usize,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_hip(weights, backend, $aliases)?);
        };
    }

    load!("attn_norm_w", &crate::weight_names::layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")));
    load!("w_q", &crate::weight_names::layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")));
    load!("w_k", &crate::weight_names::layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")));
    load!("w_v", &crate::weight_names::layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")));
    load!("w_o", &crate::weight_names::layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")));
    load!("ffn_norm_w", &crate::weight_names::layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")));
    load!("w_gate", &crate::weight_names::layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")));
    load!("w_up", &crate::weight_names::layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")));
    load!("w_down", &crate::weight_names::layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")));

    if needs_weight_transpose_hip(weights) {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", q_dim, hidden);
        t("w_k", kv_hidden, hidden);
        t("w_v", kv_hidden, hidden);
        t("w_o", hidden, q_dim);
        t("w_gate", inter, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

/// HIP alloc_kv_cache: allocate GPU buffer for KV cache and register metadata.
#[cfg(feature = "hip")]
pub(super) fn hip_alloc_kv_cache(
    backend: &HipBackend<f32>,
    config: &KvCacheConfig,
) -> Result<KvCacheHandle, BE> {
    use gllm_kernels::gpu::{GpuDevice, GpuBuffer};

    let total_bytes = config.num_layers * 2
        * config.num_heads * config.max_seq_len * config.head_dim * config.dtype_size;

    let buf = backend.device.alloc(total_bytes)
        .map_err(|e| BE::Hip(format!("KV cache alloc failed ({} bytes): {e}", total_bytes)))?;

    let ptr = buf.as_device_ptr();
    std::mem::forget(buf);

    // Register metadata for swap support
    let meta = GpuKvCacheMeta::from_config(config, ptr);
    backend.kv_meta.lock()
        .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?
        .insert(ptr, meta);

    Ok(KvCacheHandle(ptr))
}

/// HIP sample_from_tensor: download logits to CPU and sample.
#[cfg(feature = "hip")]
pub(super) fn hip_sample_from_tensor(
    logits: &LogitsHandle,
    _topology: &AttentionTopology,
    vocab_size: usize,
    sampling: &SamplingConfig,
) -> Result<Vec<u32>, BE> {
    Ok(sample_logits_cpu(&logits.data, vocab_size, sampling))
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
    _kv_caches: &mut [KvCacheHandle],
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
        _ => 9,
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
    let word_emb = get_f32_data_metal(weights, backend,
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

    // Compile decoder layer graph (once)
    let graph = build_decoder_layer_graph(seq_len, hidden, num_heads, num_kv_heads, head_dim, inter, eps);
    let kernel_entries = metal_compile_graph(device, gpu_profile, gpu_family, &graph)?;

    // Per-layer GPU execution
    for layer in 0..num_layers {
        let layer_weights = load_decoder_layer_weights_metal(
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

    // ── lm_head projection (GPU) ──
    let lm_graph = build_lm_head_graph(seq_len, hidden, vocab_size);
    let lm_entries = metal_compile_graph(device, gpu_profile, gpu_family, &lm_graph)?;

    let lm_head_w = get_f32_data_metal(weights, backend,
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

/// Load decoder layer weights for Metal backend.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn load_decoder_layer_weights_metal<E: Element>(
    weights: &dyn backend_trait::TensorLookup<E, super::metal_backend::MetalBackend<E>>,
    backend: &super::metal_backend::MetalBackend<E>,
    layer: usize,
    hidden: usize,
    q_dim: usize,
    kv_hidden: usize,
    inter: usize,
) -> Result<std::collections::HashMap<String, Vec<f32>>, BE> {
    let mut m = std::collections::HashMap::new();

    macro_rules! load {
        ($graph_name:expr, $aliases:expr) => {
            m.insert($graph_name.to_string(), get_f32_data_metal(weights, backend, $aliases)?);
        };
    }

    load!("attn_norm_w", &crate::weight_names::layer_aliases(layer, "input_layernorm.weight", Some("attn_norm.weight")));
    load!("w_q", &crate::weight_names::layer_aliases(layer, "self_attn.q_proj.weight", Some("attn_q.weight")));
    load!("w_k", &crate::weight_names::layer_aliases(layer, "self_attn.k_proj.weight", Some("attn_k.weight")));
    load!("w_v", &crate::weight_names::layer_aliases(layer, "self_attn.v_proj.weight", Some("attn_v.weight")));
    load!("w_o", &crate::weight_names::layer_aliases(layer, "self_attn.o_proj.weight", Some("attn_output.weight")));
    load!("ffn_norm_w", &crate::weight_names::layer_aliases(layer, "post_attention_layernorm.weight", Some("ffn_norm.weight")));
    load!("w_gate", &crate::weight_names::layer_aliases(layer, "mlp.gate_proj.weight", Some("ffn_gate.weight")));
    load!("w_up", &crate::weight_names::layer_aliases(layer, "mlp.up_proj.weight", Some("ffn_up.weight")));
    load!("w_down", &crate::weight_names::layer_aliases(layer, "mlp.down_proj.weight", Some("ffn_down.weight")));

    if needs_weight_transpose_metal(weights) {
        let mut t = |name: &str, rows: usize, cols: usize| {
            if let Some(data) = m.get(name) {
                let transposed = super::weight_helpers::transpose_f32(data, rows, cols);
                m.insert(name.to_string(), transposed);
            }
        };
        t("w_q", q_dim, hidden);
        t("w_k", kv_hidden, hidden);
        t("w_v", kv_hidden, hidden);
        t("w_o", hidden, q_dim);
        t("w_gate", inter, hidden);
        t("w_up", inter, hidden);
        t("w_down", hidden, inter);
    }

    Ok(m)
}

/// Metal alloc_kv_cache: allocate Metal buffer for KV cache and register metadata.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn metal_alloc_kv_cache(
    backend: &super::metal_backend::MetalBackend<f32>,
    config: &KvCacheConfig,
) -> Result<KvCacheHandle, BE> {
    use gllm_kernels::gpu::GpuDevice;

    let total_bytes = config.num_layers * 2
        * config.num_heads * config.max_seq_len * config.head_dim * config.dtype_size;

    let buf = backend.device().alloc(total_bytes)
        .map_err(|e| BE::Metal(format!("KV cache alloc failed ({} bytes): {e}", total_bytes)))?;

    let ptr = buf.as_device_ptr();
    std::mem::forget(buf);

    // Register metadata for swap support
    let meta = GpuKvCacheMeta::from_config(config, ptr);
    backend.kv_meta.lock()
        .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?
        .insert(ptr, meta);

    Ok(KvCacheHandle(ptr))
}

/// Metal sample_from_tensor: download logits to CPU and sample.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn metal_sample_from_tensor(
    logits: &LogitsHandle,
    _topology: &AttentionTopology,
    vocab_size: usize,
    sampling: &SamplingConfig,
) -> Result<Vec<u32>, BE> {
    Ok(sample_logits_cpu(&logits.data, vocab_size, sampling))
}

// ===========================================================================
// GPU swap helpers (CUDA / HIP / Metal)
// ===========================================================================
//
// The GPU KV cache is a contiguous buffer: [K_all_layers | V_all_layers].
// Each half has layout: [layer][head][seq_pos][head_dim].
// A "page" spans `page_size` consecutive seq positions.
//
// swap_out: for each (page_id, storage_key):
//   1. Compute byte offset for each (layer, head) slice within K and V blocks.
//   2. dtoh to download page data from GPU to host.
//   3. Store in swap_store keyed by storage_key.
//
// swap_in: reverse of swap_out (htod from host to GPU).

/// CUDA swap_out_pages: download page data from GPU to host swap store.
#[cfg(feature = "cuda")]
pub(super) fn cuda_swap_out_pages(
    backend: &CudaBackend<f32>,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    use gllm_kernels::gpu::GpuDevice;

    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Cuda(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let device = &*backend.device;
    let stream = device.default_stream();

    // Each half (K or V) size in bytes
    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
    // Bytes per page slice (one head, one layer)
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size;
    // Stride between consecutive seq positions for one (layer, head): max_seq_len * head_dim * dtype_size
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
    // Total bytes per page across all layers/heads (K+V)
    let total_page_bytes = meta.num_layers * meta.num_kv_heads * page_slice_bytes * 2;

    let mut swap_store = backend.swap_store.lock()
        .map_err(|e| BE::Cuda(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(BE::Cuda(format!(
                "swap_out: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size;
        let mut host_buf = vec![0u8; total_page_bytes];
        let mut dst_offset = 0usize;

        // Download K pages then V pages
        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let src_offset = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size) as u64;
                    let src_ptr = half_base + src_offset;

                    device.dtoh_raw(src_ptr, &mut host_buf[dst_offset..dst_offset + actual_slice_bytes], stream)
                        .map_err(|e| BE::Cuda(format!("swap_out dtoh failed: {e}")))?;

                    dst_offset += page_slice_bytes;
                }
            }
        }

        swap_store.insert(storage_key, host_buf);
    }

    Ok(())
}

/// CUDA swap_in_pages: upload page data from host swap store to GPU.
#[cfg(feature = "cuda")]
pub(super) fn cuda_swap_in_pages(
    backend: &CudaBackend<f32>,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    use gllm_kernels::gpu::GpuDevice;

    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Cuda(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let device = &*backend.device;
    let stream = device.default_stream();

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size;
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;

    let mut swap_store = backend.swap_store.lock()
        .map_err(|e| BE::Cuda(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let host_buf = swap_store.remove(&storage_key)
            .ok_or_else(|| BE::Cuda(format!("swap_in: storage key {} not found in swap store", storage_key)))?;

        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(BE::Cuda(format!(
                "swap_in: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size;
        let mut src_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let dst_offset_gpu = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size) as u64;
                    let dst_ptr = half_base + dst_offset_gpu;

                    device.htod_raw(&host_buf[src_offset..src_offset + actual_slice_bytes], dst_ptr, stream)
                        .map_err(|e| BE::Cuda(format!("swap_in htod failed: {e}")))?;

                    src_offset += page_slice_bytes;
                }
            }
        }
    }

    Ok(())
}

/// CUDA get_page_states: return page states based on metadata.
#[cfg(feature = "cuda")]
pub(super) fn cuda_get_page_states(
    backend: &CudaBackend<f32>,
    handle: &KvCacheHandle,
) -> Result<Vec<(PageId, PageState)>, BE> {
    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Cuda(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Cuda(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let active = meta.active_pages();
    let total = meta.total_pages();
    let mut states = Vec::with_capacity(total);
    for page_id in 0..total {
        let state = if page_id < active { PageState::Active } else { PageState::Free };
        states.push((page_id, state));
    }
    Ok(states)
}

/// HIP swap_out_pages: download page data from GPU to host swap store.
#[cfg(feature = "hip")]
pub(super) fn hip_swap_out_pages(
    backend: &HipBackend<f32>,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    use gllm_kernels::gpu::GpuDevice;

    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Hip(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let device = &*backend.device;
    let stream = device.default_stream();

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size;
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let total_page_bytes = meta.num_layers * meta.num_kv_heads * page_slice_bytes * 2;

    let mut swap_store = backend.swap_store.lock()
        .map_err(|e| BE::Hip(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(BE::Hip(format!(
                "swap_out: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size;
        let mut host_buf = vec![0u8; total_page_bytes];
        let mut dst_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let src_offset = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size) as u64;
                    let src_ptr = half_base + src_offset;

                    device.dtoh_raw(src_ptr, &mut host_buf[dst_offset..dst_offset + actual_slice_bytes], stream)
                        .map_err(|e| BE::Hip(format!("swap_out dtoh failed: {e}")))?;

                    dst_offset += page_slice_bytes;
                }
            }
        }

        swap_store.insert(storage_key, host_buf);
    }

    Ok(())
}

/// HIP swap_in_pages: upload page data from host swap store to GPU.
#[cfg(feature = "hip")]
pub(super) fn hip_swap_in_pages(
    backend: &HipBackend<f32>,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    use gllm_kernels::gpu::GpuDevice;

    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Hip(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let device = &*backend.device;
    let stream = device.default_stream();

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size;
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;

    let mut swap_store = backend.swap_store.lock()
        .map_err(|e| BE::Hip(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let host_buf = swap_store.remove(&storage_key)
            .ok_or_else(|| BE::Hip(format!("swap_in: storage key {} not found in swap store", storage_key)))?;

        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(BE::Hip(format!(
                "swap_in: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size;
        let mut src_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let dst_offset_gpu = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size) as u64;
                    let dst_ptr = half_base + dst_offset_gpu;

                    device.htod_raw(&host_buf[src_offset..src_offset + actual_slice_bytes], dst_ptr, stream)
                        .map_err(|e| BE::Hip(format!("swap_in htod failed: {e}")))?;

                    src_offset += page_slice_bytes;
                }
            }
        }
    }

    Ok(())
}

/// HIP get_page_states: return page states based on metadata.
#[cfg(feature = "hip")]
pub(super) fn hip_get_page_states(
    backend: &HipBackend<f32>,
    handle: &KvCacheHandle,
) -> Result<Vec<(PageId, PageState)>, BE> {
    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Hip(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Hip(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let active = meta.active_pages();
    let total = meta.total_pages();
    let mut states = Vec::with_capacity(total);
    for page_id in 0..total {
        let state = if page_id < active { PageState::Active } else { PageState::Free };
        states.push((page_id, state));
    }
    Ok(states)
}

/// Metal swap_out_pages: download page data from GPU to host swap store.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn metal_swap_out_pages(
    backend: &super::metal_backend::MetalBackend<f32>,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    use gllm_kernels::gpu::GpuDevice;

    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Metal(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let device = backend.device();
    let stream = device.default_stream();

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size;
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let total_page_bytes = meta.num_layers * meta.num_kv_heads * page_slice_bytes * 2;

    let mut swap_store = backend.swap_store.lock()
        .map_err(|e| BE::Metal(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(BE::Metal(format!(
                "swap_out: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size;
        let mut host_buf = vec![0u8; total_page_bytes];
        let mut dst_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let src_offset = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size) as u64;
                    let src_ptr = half_base + src_offset;

                    device.dtoh_raw(src_ptr, &mut host_buf[dst_offset..dst_offset + actual_slice_bytes], stream)
                        .map_err(|e| BE::Metal(format!("swap_out dtoh failed: {e}")))?;

                    dst_offset += page_slice_bytes;
                }
            }
        }

        swap_store.insert(storage_key, host_buf);
    }

    Ok(())
}

/// Metal swap_in_pages: upload page data from host swap store to GPU.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn metal_swap_in_pages(
    backend: &super::metal_backend::MetalBackend<f32>,
    handle: &KvCacheHandle,
    mappings: &[(PageId, StorageKey)],
) -> Result<(), BE> {
    if mappings.is_empty() { return Ok(()); }

    use gllm_kernels::gpu::GpuDevice;

    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Metal(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let device = backend.device();
    let stream = device.default_stream();

    let half_bytes = meta.num_layers * meta.num_kv_heads * meta.max_seq_len * meta.head_dim * meta.dtype_size;
    let page_slice_bytes = meta.page_size * meta.head_dim * meta.dtype_size;
    let head_stride = meta.max_seq_len * meta.head_dim * meta.dtype_size;

    let mut swap_store = backend.swap_store.lock()
        .map_err(|e| BE::Metal(format!("swap_store lock poisoned: {e}")))?;

    for &(page_id, storage_key) in mappings {
        let host_buf = swap_store.remove(&storage_key)
            .ok_or_else(|| BE::Metal(format!("swap_in: storage key {} not found in swap store", storage_key)))?;

        let token_start = page_id * meta.page_size;
        if token_start >= meta.max_seq_len {
            return Err(BE::Metal(format!(
                "swap_in: page {} starts at token {} beyond max_seq_len {}",
                page_id, token_start, meta.max_seq_len
            )));
        }

        let actual_tokens = meta.page_size.min(meta.max_seq_len - token_start);
        let actual_slice_bytes = actual_tokens * meta.head_dim * meta.dtype_size;
        let mut src_offset = 0usize;

        for kv_half in 0..2usize {
            let half_base = handle.0 + (kv_half * half_bytes) as u64;
            for layer in 0..meta.num_layers {
                for head in 0..meta.num_kv_heads {
                    let dst_offset_gpu = ((layer * meta.num_kv_heads + head) * head_stride
                        + token_start * meta.head_dim * meta.dtype_size) as u64;
                    let dst_ptr = half_base + dst_offset_gpu;

                    device.htod_raw(&host_buf[src_offset..src_offset + actual_slice_bytes], dst_ptr, stream)
                        .map_err(|e| BE::Metal(format!("swap_in htod failed: {e}")))?;

                    src_offset += page_slice_bytes;
                }
            }
        }
    }

    Ok(())
}

/// Metal get_page_states: return page states based on metadata.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(super) fn metal_get_page_states(
    backend: &super::metal_backend::MetalBackend<f32>,
    handle: &KvCacheHandle,
) -> Result<Vec<(PageId, PageState)>, BE> {
    let meta_store = backend.kv_meta.lock()
        .map_err(|e| BE::Metal(format!("kv_meta lock poisoned: {e}")))?;
    let meta = meta_store.get(&handle.0)
        .ok_or_else(|| BE::Metal(format!("KV cache handle {} not found in metadata", handle.0)))?;

    let active = meta.active_pages();
    let total = meta.total_pages();
    let mut states = Vec::with_capacity(total);
    for page_id in 0..total {
        let state = if page_id < active { PageState::Active } else { PageState::Free };
        states.push((page_id, state));
    }
    Ok(states)
}
