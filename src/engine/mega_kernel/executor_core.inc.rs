/// Mega-Kernel 执行器 (§9.1)
///
/// 唯一推理路径: 编译 → 单次 CALL。
/// 编译在模型加载时完成，推理时零 Rust 开销。

// ARCH-UNIFIED-EXEC 阶段2: 单次编译产物元数据 — CPU/GPU 统一承载 MegaKernelCompiled 构造所需字段。
//
// CPU 路径: 所有字段从 MegaKernelCompileOutput 真实提取。
// GPU 路径: 共享字段(rope_cache/logits_scratch_offset/vocab_size/hidden)从 GpuMegaKernelOutput 提取;
//   CPU 专属字段(buffer_layout/tensor_sources/source_map)用零布局/empty/None 填充 —
//   GPU launcher 返 Err(阶段3B 注入真闭包), runtime_scratchpad_bytes 永不到达,
//   这些字段在 GPU 路径为类型占位(非功能性死字段)。零布局 BufferLayout 是 codebase 已有模式
//   (gllm-kernels src/compiler/mod.rs:1350 test fixture)。
struct CompileMeta {
    /// 运行时缓冲布局(activation ping/pong, logits, sampling workspace) — CPU JIT 专属。
    buffer_layout: gllm_kernels::compiler::BufferLayout,
    /// Intermediate tensor 源(TensorId → TensorPtrSource) — CPU JIT 专属, GPU 路径为 empty。
    tensor_sources: std::collections::HashMap<
        gllm_kernels::compiler::graph::TensorId,
        gllm_kernels::compiler::buffer_alloc::TensorPtrSource,
    >,
    /// JIT source map(VmInstr → 机器码偏移 → Op 标签) — 仅 debug_jit=true 时生成, GPU 路径为 None。
    source_map: Option<gllm_kernels::compiler::codegen::vm::debug_map::JitSourceMap>,
    /// VmInstr → 机器码字节偏移区间映射 (BCE-20260724-PLAN-C-RESIDUAL-BREAK)。
    /// 仅 X86_64 + debug 路径生成 (finalize_with_diag)；GPU/非 debug 路径为 None。
    /// @trace REQ-DUMP-003 [entity:ENT-COMPILER-GRAPH] VmInstr offset map 透传
    vm_instr_map: Option<gllm_kernels::compiler::codegen::vm::debug_map::VmInstrOffsetMap>,
    /// const_pool / data_tables 布局审计 (BCE-20260724-PLAN-C-RESIDUAL-BREAK)。
    /// 仅 X86_64 + debug 路径生成；GPU/非 debug 路径为 None。
    /// @trace REQ-DUMP-003 [entity:ENT-COMPILER-GRAPH] const_pool 审计透传
    const_pool_audit: Option<gllm_kernels::compiler::codegen::vm::debug_map::ConstPoolAudit>,
    /// RoPE cos/sin 表需求(caller 必须在每次调用前填充 scratchpad) — CPU/GPU 共享。
    rope_cache: Option<gllm_kernels::compiler::codegen::RopeCacheRequirement>,
    /// Logits 区域在 scratchpad 中的偏移 — CPU/GPU 共享。
    logits_scratch_offset: usize,
    /// vocab_size — logits 每行元素数 — CPU/GPU 共享。
    vocab_size: usize,
    /// hidden_dim — SG scratchpad 需要 — CPU/GPU 共享。
    hidden: usize,
}

/// Anonymous mmap allocation with inaccessible pages on both sides.
///
/// The JIT receives the interior page-aligned pointer while accesses just beyond
/// either end fault at the offending instruction instead of corrupting allocator
/// metadata. The mapping is intentionally local to this diagnostic execution path.
struct MmapGuardedBuffer {
    ptr: *mut u8,
    len: usize,
    mapped_ptr: *mut libc::c_void,
    mapped_len: usize,
}

impl MmapGuardedBuffer {
    fn new(usable_len: usize) -> Result<Self, MegaKernelError> {
        if usable_len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                mapped_ptr: std::ptr::null_mut(),
                mapped_len: 0,
            });
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err(MegaKernelError::Execution("invalid system page size".into()));
        }
        let page_size = page_size as usize;
        let aligned_len = usable_len
            .checked_add(page_size - 1)
            .ok_or_else(|| MegaKernelError::Execution("guarded buffer size overflow".into()))?
            / page_size
            * page_size;
        let mapped_len = aligned_len
            .checked_add(page_size)
            .and_then(|len| len.checked_add(page_size))
            .ok_or_else(|| MegaKernelError::Execution("guarded mapping size overflow".into()))?;
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(MegaKernelError::Execution(format!(
                "mmap guarded buffer failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        let map_ptr = mapping as *mut u8;
        let front_guard = mapping;
        let back_guard = unsafe { map_ptr.add(page_size + aligned_len) as *mut libc::c_void };
        if unsafe { libc::mprotect(front_guard, page_size, libc::PROT_NONE) } != 0
            || unsafe { libc::mprotect(back_guard, page_size, libc::PROT_NONE) } != 0
        {
            let error = std::io::Error::last_os_error();
            unsafe { libc::munmap(mapping, mapped_len); }
            return Err(MegaKernelError::Execution(format!(
                "mprotect guarded buffer failed: {error}"
            )));
        }
        Ok(Self {
            ptr: unsafe { map_ptr.add(page_size) },
            len: usable_len,
            mapped_ptr: mapping,
            mapped_len,
        })
    }

    #[inline]
    fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }

    #[inline]
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr points to the writable interior mapping and len never crosses
        // either PROT_NONE guard page.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl std::ops::Deref for MmapGuardedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: ptr points to the writable interior mapping and len never crosses
        // either PROT_NONE guard page.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl std::ops::DerefMut for MmapGuardedBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for MmapGuardedBuffer {
    fn drop(&mut self) {
        if self.mapped_len != 0 {
            // SAFETY: mapped_ptr/mapped_len are the exact successful mmap pair.
            unsafe { libc::munmap(self.mapped_ptr, self.mapped_len); }
        }
    }
}

pub struct MegaKernelExecutor {
    /// Mega-kernel 编译产物 (唯一路径 — SPEC/39 统一架构)
    mega_compiled: Option<MegaKernelCompiled>,
    /// Runtime-owned paged-KV header array address and length.
    /// The owner (Executor::kv.paged_kv_pool) keeps the allocation alive; atomics
    /// let the call path publish the address without changing the public generate ABI.
    kv_page_headers_ptr: std::sync::atomic::AtomicUsize,
    kv_page_headers_len: std::sync::atomic::AtomicUsize,
    /// 模型配置
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    /// EOS token ID — 从 ModelConfig 读取，传给 JIT 停止条件
    eos_token_id: u32,
    /// §19 KV-OPT-009: Mega-Kernel Variant 矩阵 (按 PrecisionTier 编译独立 Variant)
    variant_registry: crate::jit::variant_registry::VariantRegistry,
    /// Weight page JIT injection config (REQ-WP-008).
    pub weight_page_inject: WeightPageJitConfig,
    /// KV page decompress injection config (REQ-COMP11).
    pub decompress_inject: KvPageDecompressConfig,
    /// Weight page table for fault recovery (REQ-WP-009).
    weight_page_table: std::sync::Mutex<Option<crate::scheduler::fault_recovery::WeightPageTable>>,
    /// Fault recovery handler (REQ-WP-009).
    fault_handler: std::sync::Mutex<Option<crate::scheduler::fault_recovery::FaultRecoveryHandler>>,
    /// VmInstr → 机器码字节偏移区间映射 (BCE-20260724-PLAN-C-RESIDUAL-BREAK)。
    /// 仅 X86_64 + debug 路径生成；非 debug 编译为 None (生产零开销)。
    /// @trace REQ-DUMP-003 [entity:ENT-COMPILER-GRAPH] VmInstr offset map 持有
    vm_instr_map: Option<gllm_kernels::compiler::codegen::vm::debug_map::VmInstrOffsetMap>,
    /// const_pool / data_tables 布局审计 (BCE-20260724-PLAN-C-RESIDUAL-BREAK)。
    /// 仅 X86_64 + debug 路径生成；非 debug 编译为 None (生产零开销)。
    /// @trace REQ-DUMP-003 [entity:ENT-COMPILER-GRAPH] const_pool 审计持有
    const_pool_audit: Option<gllm_kernels::compiler::codegen::vm::debug_map::ConstPoolAudit>,
}

// SAFETY: MegaKernelExecutor contains JIT-compiled function pointers and weight blobs
// that are thread-safe — the compiled code is immutable after construction and all
// mutable state is synchronized through interior mutability (Mutex/Atomic).
unsafe impl Send for MegaKernelExecutor {}
unsafe impl Sync for MegaKernelExecutor {}

impl MegaKernelExecutor {
    /// 从 auto-derived CompilerGraph 编译 mega-kernel（含 Argmax 或无 Argmax 的图均适用）。
    ///
    /// Graph 由 `auto_graph::build_compiler_graph()` 从 tensor names + shapes 生成。
    /// 所有模型 geometry 从 graph 自动派生，外部只需传不可派生的字段。
    pub fn compile_from_auto_graph(
        graph: gllm_kernels::compiler::graph::CompilerGraph,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        raw_floats: &std::collections::HashMap<String, crate::loader::RawFloatTensor>,
        name_map: &crate::loader::name_map::TensorNameMap,
        max_seq_len: usize,
        eos_token_id: u32,
        business_config: gllm_kernels::compiler::BusinessConfig,
        hetero_config: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig>,
        gpu_sm_version: Option<u32>,
        // ARCH-UNIFIED-EXEC 阶段3B-1: GPU launcher factory.
        // When `target = Gpu`, this factory is invoked with (ptx, kernel_name)
        // to build the real launcher closure (captures backend, calls
        // `gpu_launch_mega_kernel`). None = CPU target or backend without GPU.
        // Architect sessionId 5d98f4f4: closure must be built inside backend
        // module (`pub(super)` visibility), so we accept a factory not backend.
        gpu_launcher_builder: Option<&dyn Fn(Vec<u8>, String) -> Result<
            std::sync::Arc<dyn Fn(&MegaKernelArgs) -> Result<(), MegaKernelError> + Send + Sync>,
            MegaKernelError,
        >>,
    ) -> Result<Self, MegaKernelError> {
        // Derive all geometry from graph — CompilerGraph is the SSOT.
        let geometry =
            gllm_kernels::compiler::graph_geometry::GraphDerivedGeometry::from_graph(&graph, &gllm_kernels::dispatch::device_profile::DeviceProfile::detect())
                .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        // Build slim config with only non-derivable fields.
        // SPEC/39: mtp_config 已迁移到图拓扑（Op::MtpDraft），不再从 BusinessConfig 读取。
        // ARCH-JIT-DATA-YIELDS: mtp_depth derived from topology, not graph.ops.iter().find_map.
        let topology = gllm_kernels::compiler::codegen::vm::topology::GraphTopologyAnalysis::analyze(&graph);
        let mtp_depth = topology.mtp_config.map(|c| c.depth).unwrap_or(0);
        // SPEC/39: BusinessConfig no longer nested in CompileConfig.
        // debug_jit promoted to CompileConfig top level — the only business
        // parameter the compiler reads directly.
        // ARCH-UNIFIED-EXEC 阶段2: target 推导 — 最强硬件优先。
        // 单次编译: target = if gpu_sm_version.is_some() { Gpu } else { Cpu }
        // 套娃双编译根源(硬编码 Cpu + 单独 GPU 编译块)已删除。
        let target = if let Some(sm) = gpu_sm_version {
            gllm_kernels::compiler::mega_kernel_abi::CompileTarget::Gpu { sm_version: sm }
        } else {
            gllm_kernels::compiler::mega_kernel_abi::CompileTarget::Cpu
        };
        let config = gllm_kernels::compiler::mega_kernel_abi::CompileConfig {
            max_seq_len,
            debug_jit: business_config.debug_jit,
            hetero: hetero_config.clone(),
            target,
        };

        let hetero_layout = hetero_config.as_ref().map(|hc| {
            gllm_kernels::compiler::mega_kernel_abi::HeteroKernelWeightLayout::from_geometry_and_config(
                &geometry, hc,
            )
        });

        // Derive persistent KV storage dtype from the K projection tensor before
        // moving the graph into the compiler. The compiler's attention lowering
        // uses this same TensorMeta (lower_op.inc.rs:1503), so allocation and
        // emitted MemCopy strides share one source of truth. Graphs without a
        // persistent cache never consume kv_dtype; use the compute dtype only
        // as the inert value carried by the compiled metadata.
        let kv_dtype = graph.ops.iter()
            .find_map(|op| {
                let is_cached_attention = matches!(
                    &op.op,
                    gllm_kernels::compiler::graph::Op::MultiHeadAttention(spec)
                        if spec.kv_source == gllm_kernels::compiler::graph::KvSource::FromCache
                );
                if !is_cached_attention {
                    return None;
                }
                op.inputs.get(1)
                    .and_then(|&tid| graph.tensor(tid))
                    .map(|tensor| tensor.dtype)
            })
            .unwrap_or(geometry.compute_dtype);

        // Pre-resolve weight layout before moving graph into compiler.
        // Also save layer_loop_config for weight packing (needed even without GPU).
        let layer_loop_cfg = graph.layer_loop_config.clone();
        let hetero_loop_cfg = graph.hetero_layer_loop_config.clone();
        // Mixed-quant per-layer dtype config (Q5_K_M etc.): saved here because `graph`
        // is moved into compile() below, and pack_weights_from_graph needs the
        // non-linear offset_table to place each layer at its actual dtype size.
        let mixed_quant_loop_cfg = graph.mixed_quant_layer_loop_config.clone();
        // ARCH-UNIFIED-EXEC 阶段2: graph_for_gpu 克隆已删除 — 单次编译,
        // target 决定编译 CPU 还是 GPU, 不再克隆两份图编两次。
        let weight_layout = graph.weight_layout();
        // ARCH-BLOB-YIELDS-WEIGHT: preserve per-tensor dtype from graph tensors.
        // dtype is taken from the graph TensorMeta (the source of truth) so the blob
        // reader can decode each tensor by its actual storage dtype.
        let mut named_offsets: Vec<(String, usize, gllm_kernels::types::DType)> = weight_layout
            .offsets
            .iter()
            .filter_map(|&(tid, offset)| {
                graph
                    .tensors
                    .get(tid.0 as usize)
                    .map(|t| (t.name.clone(), offset, t.dtype))
            })
            .collect();
        // BCE-20260629-006: clone graph tensor name map before graph is moved into compile()
        // 用于后续 compile 返回 tensor_sources 时补全 intermediate tensor 的 named_offsets
        let tensor_names: std::collections::HashMap<gllm_kernels::compiler::graph::TensorId, (String, gllm_kernels::types::DType)> = graph
            .tensors
            .iter()
            .map(|t| (t.id, (t.name.clone(), t.dtype)))
            .collect();
        // Expand total_weight_bytes for layer loop: the layout only contains 1 copy of
        // per-layer weights (L0.*), but packing replicates to num_layers copies.
        // Mixed-quant (Q5_K_M etc.) uses a non-linear offset_table — see
        // compute_total_weight_bytes for the per-mode expansion math.
        let total_weight_bytes = compute_total_weight_bytes(
            &weight_layout,
            layer_loop_cfg.as_ref(),
            hetero_loop_cfg.as_ref(),
            mixed_quant_loop_cfg.as_ref(),
        );
        // Default graph compiles with kv_load_mode=None (Direct).
        // KIVI4 variant compiles with kv_load_mode=Kivi4 for compressed KV load.
        let graph_kivi4 = {
            let mut g = graph.clone();
            g.kv_load_mode = Some(gllm_kernels::compiler::codegen::vm::instr::KvLoadMode::Kivi4);
            g
        };

        let hetero_layout_for_kivi4 = hetero_layout.clone();
        // SPEC/39: 从图拓扑推导 norm residual 约定，替代硬编码。
        // Gemma 1/2/3: has_embedding_scale=true, has_qk_norm=false → (1+weight) residual
        // Gemma 4+: has_qk_norm=true → standard RMSNorm, no residual
        // All other models: no embedding_scale → no residual
        let has_gemma_norm_residual = graph.embedding_scale.is_some() && !topology.has_qk_norm;
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let output = compiler
            .compile(graph, &config, hetero_layout)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        // ARCH-UNIFIED-EXEC 阶段2: 单次编译 — match CompileOutput 构造 CompiledExecutable + CompileMeta。
        // CPU 路径: 提取 layer_code + entry_fn + 完整布局元数据(buffer_layout/tensor_sources/source_map)。
        // GPU 路径: 提取 PTX 字节码 + placeholder launcher(返 Err) — 真闭包阶段3B 在
        //   compile_and_upload_mega 注入(那里有 backend)。GPU 共享字段(rope_cache/logits_scratch_offset/
        //   vocab_size/hidden)从 GpuMegaKernelOutput 提取; CPU 专属字段(buffer_layout/source_map/
        //   tensor_sources)用零布局/None/empty 填充 — GPU launcher 返 Err, 这些字段在 GPU 路径永不读取。
        let (executable, meta): (CompiledExecutable, CompileMeta) = match output {
            gllm_kernels::compiler::CompileOutput::Cpu(c) => {
                let gllm_kernels::compiler::MegaKernelCompileOutput {
                    layer_code: exec_code,
                    buffer_layout,
                    tensor_sources,
                    source_map,
                    vm_instr_map,
                    const_pool_audit,
                    rope_cache,
                    logits_scratch_offset,
                    vocab_size,
                    hidden,
                    num_layers: _,
                    weight_layout: _,
                    total_scratchpad_bytes: _,
                    hetero_layout: _,
                } = c;
                // entry_point_as_mega_kernel 借用 &exec_code 返回 fn 指针(已 transmute 拷贝),
                // 不持借用 → 可先取 entry_fn 再 move exec_code 进 CompiledExecutable::Cpu。
                let entry_fn = unsafe { exec_code.entry_point_as_mega_kernel() };
                let executable = CompiledExecutable::Cpu {
                    code: exec_code,
                    entry_fn,
                };
                let meta = CompileMeta {
                    buffer_layout,
                    tensor_sources,
                    source_map,
                    vm_instr_map,
                    const_pool_audit,
                    rope_cache,
                    logits_scratch_offset,
                    vocab_size,
                    hidden,
                };
                (executable, meta)
            }
            gllm_kernels::compiler::CompileOutput::Gpu(g) => {
                // ARCH-UNIFIED-EXEC 阶段3B-1: 真 launcher 注入.
                // 工厂 (gpu_launcher_builder) 由 compile_and_upload_mega 构造,
                // 闭包内调 backend.build_mega_launcher(ptx, kernel_name) → 真 cuLaunchKernel.
                // Architect sessionId 5d98f4f4: 闭包必须在 backend 模块内构造 (pub(super)),
                // 故 compile_from_auto_graph 收工厂而非 backend.
                let gllm_kernels::compiler::GpuMegaKernelOutput {
                    gpu_code,
                    rope_cache,
                    logits_scratch_offset,
                    vocab_size,
                    hidden,
                    total_scratchpad_bytes: _,
                    num_layers: _,
                } = g;
                let kernel_name = "mega_kernel".to_string();
                let launcher = if let Some(builder) = gpu_launcher_builder {
                    builder(gpu_code.clone(), kernel_name.clone())?
                } else {
                    return Err(MegaKernelError::Execution(
                        "GPU target compiled but no launcher builder provided (backend has no GPU capability?)".to_string(),
                    ));
                };
                let executable = CompiledExecutable::Gpu {
                    ptx: gpu_code,
                    kernel_name,
                    launcher,
                };
                // GPU 路径: GpuMegaKernelOutput 不携带 buffer_layout/tensor_sources/source_map
                // (CPU JIT 专属字段)。用零布局 + None + empty 填充 — GPU launcher 返 Err,
                // runtime_scratchpad_bytes 永不到达, 这些字段在 GPU 路径为类型占位(非功能性死字段)。
                // 零布局 BufferLayout 是 codebase 已有模式(gllm-kernels mod.rs:1350 test fixture)。
                // 阶段3B 接真 launcher 时若需 GPU scratchpad 管理, 再扩 GpuMegaKernelOutput。
                let buffer_layout = gllm_kernels::compiler::mega_kernel_abi::BufferLayout {
                    activation_a_offset: 0,
                    activation_b_offset: 0,
                    activation_bytes: 0,
                    logits_offset: 0,
                    logits_bytes: 0,
                    sampling_workspace_offset: 0,
                    sampling_workspace_bytes: 0,
                    sg_detect_offset: 0,
                    sg_knowledge_offset: 0,
                    sg_data_bytes: 0,
                    layer_capture_offset: 0,
                    layer_capture_stride: 0,
                    layer_capture_bytes: 0,
                    total_scratchpad_bytes: 0,
                };
                let meta = CompileMeta {
                    buffer_layout,
                    tensor_sources: std::collections::HashMap::new(),
                    source_map: None,
                    // GPU 路径无 VmInstr offset map / const_pool 审计 (CPU JIT 专属)。
                    vm_instr_map: None,
                    const_pool_audit: None,
                    rope_cache,
                    logits_scratch_offset,
                    vocab_size,
                    hidden,
                };
                (executable, meta)
            }
        };

        // BCE-20260629-006: 追加 intermediate tensor offsets（供 DIAG harness 动态查询）
        // CPU 专属: GpuMegaKernelOutput 无 tensor_sources, GPU 路径 tensor_sources 为 empty,
        // 循环不迭代 → 自动跳过增强。无需额外条件分支。
        // 对于 tensor_sources 中的 ActivationPing/Pong，直接用 buffer_layout 的 offset
        // 因为 VAM 把 embedding 映射为 ActivationPing 但 Resolver 已强制覆盖为 Intermediate
        // 所以从 meta.tensor_sources 提取即可。对于 ActivationPing（被 VAM 覆盖的），
        // 用 meta.buffer_layout.activation_a_offset 而不是 0。
        // 先收集 tensor_sources 中的所有 Intermediate 映射
        let mut inter_map: std::collections::HashMap<gllm_kernels::compiler::graph::TensorId, usize> = std::collections::HashMap::new();
        for (&tid, src) in &meta.tensor_sources {
            if let gllm_kernels::compiler::buffer_alloc::TensorPtrSource::Intermediate { offset } = src {
                inter_map.insert(tid, *offset);
            }
        }
        // 构建 named_offsets：先用 weight_layout 的，再补上 intermediate（有 Intermediate 映射的）
        for (&tid, src) in &meta.tensor_sources {
            if let Some((name, dt)) = tensor_names.get(&tid) {
                if !named_offsets.iter().any(|(n, _, _)| n == name) {
                    // 优先用 inter_map 中的 offset（Resolver 覆盖后的值）
                    let offset = inter_map.get(&tid).copied().unwrap_or_else(|| {
                        match src {
                            gllm_kernels::compiler::buffer_alloc::TensorPtrSource::Intermediate { offset } => *offset,
                            gllm_kernels::compiler::buffer_alloc::TensorPtrSource::ActivationPing => meta.buffer_layout.activation_a_offset,
                            gllm_kernels::compiler::buffer_alloc::TensorPtrSource::ActivationPong => meta.buffer_layout.activation_b_offset,
                            gllm_kernels::compiler::buffer_alloc::TensorPtrSource::Output { offset } => *offset,
                            gllm_kernels::compiler::buffer_alloc::TensorPtrSource::Weight { offset } => *offset,
                            gllm_kernels::compiler::buffer_alloc::TensorPtrSource::Activation => 0,
                        }
                    });
                    named_offsets.push((name.clone(), offset, *dt));
                }
            }
        }

        // Ring-Buffer 逐层捕获: 注册 layer_capture 基址到 named_offsets (诊断 harness 可查).
        // diagnostic_tensor_offset("layer_capture") 返回 capture 区起点;
        // 第 N 层输出 = layer_capture_offset + N * layer_capture_stride.
        // feature 关时 layer_capture_bytes=0, 不注册 (生产零开销).
        if meta.buffer_layout.layer_capture_bytes > 0 {
            named_offsets.push((
                "layer_capture".to_string(),
                meta.buffer_layout.layer_capture_offset,
                gllm_kernels::types::DType::F32,
            ));
            eprintln!("[RING-BUF] layer_capture registered: offset={} stride={} bytes={}",
                meta.buffer_layout.layer_capture_offset,
                meta.buffer_layout.layer_capture_stride,
                meta.buffer_layout.layer_capture_bytes);
        }

        // §19 KV-OPT-009: Compile KIVI4 variant for compressed KV attention.
        // CPU 专属: KIVI4 variant 走 CPU JIT 编译(compile + expect_cpu),
        // GPU 路径(meta.tensor_sources 为 empty)不产 KIVI4 variant — 阶段3B 后再考虑 GPU KIVI4。
        // TEMP: disable KIVI4 compilation for GGUF models (no raw_floats) —
        // RegAllocator on 28-layer N-layer takes 15min per compilation.
        // Will re-enable after RegAllocator optimization.
        let kivi4_exec = if !meta.tensor_sources.is_empty() && !raw_floats.is_empty() {
            compiler
                .compile(graph_kivi4, &config, hetero_layout_for_kivi4)
                .ok()
                .map(|o| o.expect_cpu())
        } else {
            None
        };

        // Pack weights using pre-resolved named offsets — unified with 无 Argmax 的图路径。
        let weight_blob = pack_weights_from_graph(
            &named_offsets,
            total_weight_bytes,
            weight_ptrs,
            weight_sizes,
            raw_floats,
            name_map,
            layer_loop_cfg.as_ref(),
            hetero_loop_cfg.as_ref(),
            mixed_quant_loop_cfg.as_ref(),
            geometry.num_layers,
            has_gemma_norm_residual,
        );

        // ARCH-UNIFIED-EXEC 阶段2: 套娃双编译块已删除 — 不再克隆 graph_for_gpu 单独编 GPU。
        // 单次编译按 target 选 CPU/GPU; gpu_sm_version 已在 target 推导处消费。
        let _ = gpu_sm_version;

        let mtp_depth_extracted = mtp_depth;
        let mega_compiled = MegaKernelCompiled {
            named_offsets,
            buffer_layout: meta.buffer_layout,
            logits_scratch_offset: meta.logits_scratch_offset,
            weight_blob,
            executable,
            target,
            rope_cache: meta.rope_cache,
            scratchpad_base_bytes: meta.logits_scratch_offset,
            vocab_size: meta.vocab_size,
            hidden: meta.hidden,
            compute_dtype: geometry.compute_dtype,
            kv_dtype,

            source_map: meta.source_map,
            num_kv_heads: geometry.num_kv_heads,
            head_dim: geometry.head_dim,
            max_seq_len: max_seq_len.min(gllm_kernels::compiler::buffer_alloc::ALLOC_SEQ_CAP),
            mtp_depth: mtp_depth_extracted,
        };

        Ok(Self {
            mega_compiled: Some(mega_compiled),
            kv_page_headers_ptr: std::sync::atomic::AtomicUsize::new(0),
            kv_page_headers_len: std::sync::atomic::AtomicUsize::new(0),
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden,
            vocab_size: geometry.vocab_size,
            eos_token_id,
            variant_registry: {
                let mut registry = crate::jit::variant_registry::VariantRegistry::new();
                // Register default Direct variant (the compiled mega-kernel itself)
                let default_key = crate::jit::variant_registry::VariantRegistry::derive_key(
                    "default", None, false, None, false, 64, None, None, None,
                );
                let default_variant = crate::jit::variant_registry::CompiledVariant {
                    code: Vec::new(),                  // The actual code lives in mega_compiled
                    instruction_footprint_bytes: 8192, // estimated
                    mechanisms: vec![crate::jit::variant_registry::MechanismId::Dense],
                    section: crate::jit::variant_registry::CodeSection::Hot,
                    key: default_key.clone(),
                };
                let _ = registry.register(default_variant);

                // §19 KV-OPT-009: Register KIVI4 variant (compiled with Kivi4 kv_load_mode).
                if let Some(ref kivi4) = kivi4_exec {
                    let kivi4_key = crate::jit::variant_registry::VariantRegistry::derive_key(
                        "default", None, false, None, false, 64, None, Some("KIVI4".to_string()), None,
                    );
                    let kivi4_variant = crate::jit::variant_registry::CompiledVariant {
                        code: kivi4.layer_code.code_bytes().to_vec(),
                        instruction_footprint_bytes: kivi4.layer_code.code_bytes().len(),
                        mechanisms: vec![crate::jit::variant_registry::MechanismId::KiviQuant],
                        section: crate::jit::variant_registry::CodeSection::Hot,
                        key: kivi4_key,
                    };
                    let _ = registry.register(kivi4_variant);
                    log::info!("[mega] KV-OPT-009: KIVI4 variant compiled ({} bytes)",
                        kivi4.layer_code.code_bytes().len());
                }

                registry
            },
            weight_page_inject: WeightPageJitConfig::default(),
            decompress_inject: KvPageDecompressConfig::default(),
            weight_page_table: std::sync::Mutex::new(None),
            fault_handler: std::sync::Mutex::new(None),
            // BCE-20260724-PLAN-C-RESIDUAL-BREAK: 透传诊断 map 到 executor 供 dump_offset_map。
            // 非 debug 编译时 meta.vm_instr_map / meta.const_pool_audit 均为 None (生产零开销)。
            vm_instr_map: meta.vm_instr_map,
            const_pool_audit: meta.const_pool_audit,
        })
    }

    /// Publish the long-lived paged-KV header array used by subsequent calls.
    // @trace REQ-KV-OPT-004
    pub fn set_kv_page_headers(&self, headers: &[crate::kv_cache::KvPageHeader]) {
        self.kv_page_headers_ptr
            .store(headers.as_ptr() as usize, std::sync::atomic::Ordering::Release);
        self.kv_page_headers_len
            .store(headers.len(), std::sync::atomic::Ordering::Release);
    }

    /// Returns total scratchpad bytes needed for execution.
    ///
    /// INVARIANT: `mega_compiled` is always `Some` after successful construction.
    /// PSC-1 root cause: returning 0 here (via `unwrap_or(0)`) silently hides an
    /// invariant violation and leads to a zero-sized scratchpad allocation, causing
    /// a heap-buffer-overflow during JIT execution. Fail loudly instead.
    pub fn total_scratchpad_bytes(&self) -> usize {
        self.mega_compiled
            .as_ref()
            .map(|m| m.scratchpad_base_bytes)
            .expect("total_scratchpad_bytes: mega_compiled must be Some — executor constructed without compiling mega-kernel (invariant violation)")
    }

    /// Set weight page table and fault handler for explicit fault recovery (REQ-WP-009).
    pub fn set_weight_page_table(
        &self,
        table: crate::scheduler::fault_recovery::WeightPageTable,
        handler: crate::scheduler::fault_recovery::FaultRecoveryHandler,
    ) {
        *self.weight_page_table.lock().expect("weight_page_table Mutex poisoned — previous holder panicked") = Some(table);
        *self.fault_handler.lock().expect("fault_handler Mutex poisoned — previous holder panicked") = Some(handler);
    }

    /// REQ-WP-009: Ensure all weight pages for the current step are in GpuHbm (Tier::L1).
    ///
    /// Fast path: no weight page table registered → zero overhead (just a Mutex lock + None check).
    /// Slow path: iterates layers 0..num_layers, checks each page's tier via `page_tier()`,
    /// and calls `recover_fault()` for any page not in L1.
    fn ensure_weight_pages_resident(&self) {
        use crate::scheduler::fault_recovery::PageFault;
        use crate::scheduler::memory_manager::{GlobalMemoryManager, Tier};

        // Phase 1: Collect faults (immutable borrow of table)
        let faults: Vec<PageFault> = {
            let table_lock = self.weight_page_table.lock().expect("weight_page_table Mutex poisoned in ensure_weight_pages_resident — previous holder panicked");
            let Some(table) = table_lock.as_ref() else { return };
            let mut faults = Vec::new();
            for layer_idx in 0..self.num_layers {
                let Some(pages) = table.get_layer_pages(layer_idx) else { continue };
                for &pid in pages {
                    let tier = table.page_tier(pid).unwrap_or(Tier::L3);
                    if tier != Tier::L1 {
                        faults.push(PageFault {
                            page_id: pid,
                            current_tier: tier,
                            target_tier: Tier::L1,
                            fault_time: std::time::Instant::now(),
                            expert_key: None,
                            dense_layer_idx: Some(layer_idx),
                        });
                    }
                }
            }
            faults
        };

        // Fast path: all pages already in L1
        if faults.is_empty() {
            return;
        }

        // Phase 2: Recover faults (mutable borrow of table + handler)
        let mut table_lock = self.weight_page_table.lock().expect("weight_page_table Mutex poisoned in ensure_weight_pages_resident (phase 2) — previous holder panicked");
        let table = table_lock.as_mut().unwrap();
        let mut handler_lock = self.fault_handler.lock().expect("fault_handler Mutex poisoned in ensure_weight_pages_resident — previous holder panicked");
        let handler = handler_lock.as_mut().unwrap();

        for fault in faults {
            let mut gmm = GlobalMemoryManager::new_with_capacities(0, 0, 0);
            if let Err(e) = handler.recover_fault(&fault, &mut gmm, table) {
                log::warn!("[WP-009] fault recovery failed for page {:?}: {}", fault.page_id, e);
            }
        }
    }

    /// 单序列 mega-kernel 生成。
    ///
    /// ARCH-RUST-IS-CODEGEN: 一次 CALL 完成。
    /// JIT mega-kernel 内部执行完整的 generate loop:
    ///   LoopBegin → embed → N 层 → logits-producer → Argmax → StoreToken → CheckStopCondition → LoopEnd
    /// Rust 只做：(1) 准备输入 (2) 预填 RoPE 表 (3) 一次 CALL (4) 读 output_tokens
    pub fn generate_single_sequence(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        hook_ctx_ptr: *const u8,
        callback_table_ptr: *const u8,
        page_table: Option<&[u32]>,
        pool_base: *const u8,
        session_position: usize,
        fused_hidden: Option<&[f32]>,
        num_mm_tokens: usize,
    ) -> Result<Vec<u32>, MegaKernelError> {
        // REQ-KV-OPT-010: no position-agnostic range when called without it.
        self.generate_single_sequence_inner(
            prompt_tokens,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            hook_ctx_ptr,
            callback_table_ptr,
            page_table,
            pool_base,
            session_position,
            fused_hidden,
            num_mm_tokens,
            None,
        )
    }

    /// Inner implementation with optional position-agnostic range.
    ///
    /// When `position_agnostic_range` is `Some((start, end))`, RoPE is set to
    /// identity (cos=1, sin=0) for positions `[start, end)`. This implements
    /// REQ-KV-OPT-010 CacheSlide: system prompt pages skip position encoding
    /// to enable cross-request KV reuse.
    fn generate_single_sequence_inner(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        hook_ctx_ptr: *const u8,
        callback_table_ptr: *const u8,
        page_table: Option<&[u32]>,
        pool_base: *const u8,
        session_position: usize,
        fused_hidden: Option<&[f32]>,
        num_mm_tokens: usize,
        position_agnostic_range: Option<(usize, usize)>,
    ) -> Result<Vec<u32>, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        let prompt_len = prompt_tokens.len();
        let max_total = prompt_len + max_new_tokens;
        let mtp_depth = mega.mtp_depth;

        let mut input_ids = vec![0u32; max_total];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..max_total as u32).collect();
        // Output buffer: [0..max_new_tokens) = main tokens,
        // [max_new_tokens..max_new_tokens + max_new_tokens * mtp_depth) = MTP candidates.
        let output_size = max_new_tokens * (1 + mtp_depth);
        let mut output_tokens = vec![0u32; output_size];
        let mut scratchpad = MmapGuardedBuffer::new(
            mega.runtime_scratchpad_bytes(max_total)
                .map_err(MegaKernelError::Execution)?,
        )?;

        // Pre-fill RoPE cos/sin table for all positions [0..max_total).
        if let Some(ref rc) = mega.rope_cache {
            // Primary cache
            let rope_elems = max_total * rc.head_dim;
            let rope_bytes = rope_elems * std::mem::size_of::<f32>();
            if rc.cache_offset + rope_bytes <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice,
                    &positions[..max_total],
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling,
                );
            }
            // Secondary cache (for heterogeneous models with 2 head_dim values)
            if let Some(ref sec) = rc.secondary_cache {
                let sec_elems = max_total * sec.head_dim;
                let sec_bytes = sec_elems * std::mem::size_of::<f32>();
                if sec.cache_offset + sec_bytes <= scratchpad.len() {
                    let sec_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            scratchpad[sec.cache_offset..].as_mut_ptr() as *mut f32,
                            sec_elems,
                        )
                    };
                    gllm_kernels::compiler::fill_cos_sin_table_partial(
                        sec_slice,
                        &positions[..max_total],
                        sec.head_dim,
                        sec.theta,
                        sec.partial,
                        sec.rope_scaling,
                    );
                }
            }
        }

        // REQ-KV-OPT-010: Position-agnostic RoPE (CacheSlide).
        // For system prompt pages marked as position-agnostic, zero the rotation
        // by setting cos=1 and sin=0 for those positions in the RoPE cache.
        // This allows cross-request KV reuse without position encoding mismatch.
        if let Some((start, end)) = position_agnostic_range {
            let end = end.min(max_total);
            if start < end {
                if let Some(ref rc) = mega.rope_cache {
                    let half = rc.head_dim / 2;
                    // Primary cache
                    let rope_elems = max_total * rc.head_dim;
                    if rc.cache_offset + rope_elems * std::mem::size_of::<f32>() <= scratchpad.len() {
                        let rope_slice = unsafe {
                            std::slice::from_raw_parts_mut(
                                scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                                rope_elems,
                            )
                        };
                        for pos in start..end {
                            let row_start = pos * rc.head_dim;
                            // cos half: set to 1.0 (identity rotation)
                            for i in 0..half {
                                rope_slice[row_start + i] = 1.0;
                            }
                            // sin half: set to 0.0 (no rotation)
                            for i in half..rc.head_dim {
                                rope_slice[row_start + i] = 0.0;
                            }
                        }
                    }
                    // Secondary cache (heterogeneous models)
                    if let Some(ref sec) = rc.secondary_cache {
                        let sec_half = sec.head_dim / 2;
                        let sec_elems = max_total * sec.head_dim;
                        if sec.cache_offset + sec_elems * std::mem::size_of::<f32>() <= scratchpad.len() {
                            let sec_slice = unsafe {
                                std::slice::from_raw_parts_mut(
                                    scratchpad[sec.cache_offset..].as_mut_ptr() as *mut f32,
                                    sec_elems,
                                )
                            };
                            for pos in start..end {
                                let row_start = pos * sec.head_dim;
                                for i in 0..sec_half {
                                    sec_slice[row_start + i] = 1.0;
                                }
                                for i in sec_half..sec.head_dim {
                                    sec_slice[row_start + i] = 0.0;
                                }
                            }
                        }
                    }
                }
                log::debug!(
                    "[mega] REQ-KV-OPT-010: position-agnostic RoPE for positions [{}, {})",
                    start,
                    end,
                );
            }
        }

        let page_table_ptr = page_table.map_or(std::ptr::null(), |pt| pt.as_ptr());
        // Allocate KV cache buffer for contiguous KV attention.
        // The mega-kernel writes K/V data here after each layer's GEMM and reads
        // from it during attention, enabling the model to attend to all previous tokens.
        let kv_cache_bytes = mega.kv_cache_bytes(self.num_layers);
        let kv_cache = MmapGuardedBuffer::new(kv_cache_bytes)?;
        if std::env::var_os("GLLM_DEBUG_RESOURCE").is_some() {
            eprintln!(
                "[GUARD-ALLOC] sequence scratch={:p} len={} kv={:p} len={}",
                scratchpad.as_mut_ptr(),
                scratchpad.len(),
                kv_cache.as_mut_ptr(),
                kv_cache.len(),
            );
        }
        // @trace REQ-KV-OPT-004
        // The Executor publishes its long-lived header array before every call.
        // Keep a call-local fallback only for direct MegaKernelExecutor users.
        let header_page_size = mega.max_seq_len.max(1);
        let header_pages_per_layer = max_total.div_ceil(header_page_size);
        let header_count = self
            .num_layers
            .checked_mul(header_pages_per_layer)
            .ok_or_else(|| MegaKernelError::Execution("KV page-header count overflow".into()))?;
        let local_headers: Vec<crate::kv_cache::KvPageHeader> = (0..header_count)
            .map(|page_id| crate::kv_cache::KvPageHeader::new(page_id as u32))
            .collect();
        let configured_headers = self
            .kv_page_headers_ptr
            .load(std::sync::atomic::Ordering::Acquire);
        let header_ptr = if configured_headers != 0 {
            configured_headers as *const u8
        } else {
            local_headers.as_ptr() as *const u8
        };
        let effective_pool_base = if !kv_cache.is_empty() {
            kv_cache.as_mut_ptr() as *const u8
        } else {
            pool_base
        };

        // REQ-WP-009: Verify weight pages are in GpuHbm (L1) before CALL.
        // Fast path: no weight page table registered → zero overhead.
        self.ensure_weight_pages_resident();

        let generated_count = unsafe {
            // R1: Build KernelContext for single-pointer ABI transition.
            // Parameters are organized into the flat struct; legacy ABI args
            // are extracted from it at the call site.
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.kv_cache_ptr = effective_pool_base as *mut u8;
            ctx.hook_ctx_ptr = hook_ctx_ptr as *mut u8;
            ctx.callback_table_ptr = callback_table_ptr as *const u64;
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();
            ctx.batch_ctx_ptr = std::ptr::null();
            ctx.telemetry_ptr = std::ptr::null_mut();
            ctx.kv_page_header_ptr = header_ptr;
            ctx.kv_page_size = header_page_size as u32;
            ctx.kv_num_layers = self.num_layers as u32;
            ctx.kv_num_heads = mega.num_kv_heads as u32;
            ctx.kv_head_dim = mega.head_dim as u32;

            // REQ-COMP11: Wire KV page decompress injection.
            // When decompress_inject is enabled, the JIT reads KvPageHeader.codec
            // via kv_page_header_ptr before each KV page access. If codec != None,
            // the JIT invokes the corresponding decompress callback (Lz4/BitPackRle/Nvcomp)
            // registered in the callback table before reading the page data.
            if self.decompress_inject.enabled {
                ctx.decompress_inject_flags = 1; // bit 0 = enabled
            }

            // Save MXCSR before JIT call — JIT may modify FP exception masks
            let mut mxcsr_saved: u32 = 0;
            std::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr_saved, options(nostack));
            // NaN-TRAP: Enable Invalid Operation exception (bit 0 = IE unmask)
            // When GLLM_NAN_TRAP=1, any NaN-producing FP op triggers SIGFPE,
            // allowing precise identification of the first NaN generation site.
            let mxcsr_nan_trap = if std::env::var("GLLM_NAN_TRAP").is_ok() {
                0x1F80 & !0x01 // Unmask IE (Invalid Exception) — bit 0
            } else {
                0x1F80 // Default: all exceptions masked
            };
            std::arch::asm!("ldmxcsr [{}]", in(reg) &mxcsr_nan_trap, options(nostack));
            let result = match &mega.executable {
                CompiledExecutable::Cpu { entry_fn, .. } => {
                    (entry_fn)(
                        input_ids.as_ptr(),
                        ctx.weight_blob_ptr,
                        ctx.kv_cache_ptr,
                        positions.as_ptr(),
                        std::ptr::null(),
                        1,
                        prompt_len,
                        ctx.scratch_buffer_ptr,
                        output_tokens.as_mut_ptr(),
                        temperature.to_bits() as usize,
                        top_k,
                        top_p.to_bits() as usize,
                        max_new_tokens,
                        self.eos_token_id as usize,
                        ctx.hook_ctx_ptr as *const u8,
                        ctx.telemetry_ptr,
                        session_position, // session_position (0=new, >0=resume)
                        fused_hidden.map_or(std::ptr::null(), |fh| fh.as_ptr() as *const u8), // fused_hidden_ptr
                        num_mm_tokens,      // num_mm_tokens
                        ctx.callback_table_ptr as *const u8,
                        page_table_ptr,     // page_table_ptr: NULL = contiguous KV, u32[] = paged KV
                        ctx.batch_ctx_ptr,
                        ctx.kv_page_header_ptr,
                    )
                },
                CompiledExecutable::Gpu { launcher, .. } => {
                    let args = MegaKernelArgs {
                        input_ids_ptr: input_ids.as_ptr(),
                        weight_blob_ptr: ctx.weight_blob_ptr,
                        kv_cache_ptr: ctx.kv_cache_ptr,
                        positions_ptr: positions.as_ptr(),
                        aux_ptr: std::ptr::null(),
                        batch_size: 1,
                        prompt_len,
                        scratchpad_ptr: ctx.scratch_buffer_ptr,
                        output_tokens_ptr: output_tokens.as_mut_ptr(),
                        temperature_u32: temperature.to_bits() as usize,
                        top_k,
                        top_p_u32: top_p.to_bits() as usize,
                        max_new_tokens,
                        eos_token_id: self.eos_token_id as usize,
                        hook_ctx_ptr: ctx.hook_ctx_ptr as *const u8,
                        telemetry_ptr: ctx.telemetry_ptr,
                        session_position,
                        fused_hidden_ptr: fused_hidden.map_or(std::ptr::null(), |fh| fh.as_ptr() as *const u8),
                        num_mm_tokens,
                        callback_table_ptr: ctx.callback_table_ptr as *const u8,
                        page_table_ptr,
                        batch_ctx_ptr: ctx.batch_ctx_ptr,
                        kv_page_header_ptr: ctx.kv_page_header_ptr,
                        scratchpad_bytes: scratchpad.len(),
                        output_tokens_bytes: output_tokens.len() * 4,
                    };
                    launcher(&args)?;
                    // GPU arm: generated_count 占位 = max_new_tokens (D2H copy 后由 3C 校准)
                    max_new_tokens
                }
            };
            // Read MXCSR after JIT call — check if JIT modified FP exception state
            let mut mxcsr_after: u32 = 0;
            std::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr_after, options(nostack));
            // Restore MXCSR: reset all FP exception flags and set default masks
            // Default MXCSR = 0x1F80 (all exceptions masked, round-to-nearest, no flush-to-zero)
            std::arch::asm!("ldmxcsr [{}]", in(reg) &0x1F80u32, options(nostack));
            let _ = (mxcsr_saved, mxcsr_after); // suppress unused warnings
            result
        };

        log::debug!(
            "[mega] prompt_len={} max_new_tokens={} generated_count={} eos={} output_first={}",
            prompt_len,
            max_new_tokens,
            generated_count,
            self.eos_token_id,
            output_tokens.first().copied().unwrap_or(0), // [LEGAL-PSC10+25] debug log only — NOT a generation sentinel
        );
        // BCE-20260623-004: Removed output_tokens[0] != 0 heuristic.
        // Token ID 0 is a valid token; using it as a sentinel discards legitimate output.
        // Trust generated_count from the JIT kernel — if it reports 0, no tokens were generated.
        // If generated_count is wrong, that's a JIT bug to fix in the kernel, not here.
        let actual_count = generated_count;
        // Build output: main tokens followed by MTP candidate tokens (if enabled).
        // MTP candidates layout: output_tokens[max_new_tokens + step * mtp_depth + k]
        let mut result = Vec::with_capacity(actual_count * (1 + mtp_depth));
        result.extend_from_slice(&output_tokens[..actual_count]);
        if mtp_depth > 0 && actual_count > 0 {
            for step in 0..actual_count {
                let mtp_base = max_new_tokens + step * mtp_depth;
                for k in 0..mtp_depth {
                    if mtp_base + k < output_tokens.len() {
                        result.push(output_tokens[mtp_base + k]);
                    }
                }
            }
        }
        Ok(result)
    }

    /// Like `generate_single_sequence`, but with position-agnostic range support.
    ///
    /// REQ-KV-OPT-010: When system prompt pages are marked position-agnostic,
    /// the RoPE cache is set to identity (cos=1, sin=0) for positions `[agnostic_start, agnostic_end)`.
    /// This enables CacheSlide cross-request KV reuse.
    pub fn generate_single_sequence_with_position_agnostic(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        hook_ctx_ptr: *const u8,
        callback_table_ptr: *const u8,
        page_table: Option<&[u32]>,
        pool_base: *const u8,
        session_position: usize,
        fused_hidden: Option<&[f32]>,
        num_mm_tokens: usize,
        agnostic_start: usize,
        agnostic_end: usize,
    ) -> Result<Vec<u32>, MegaKernelError> {
        self.generate_single_sequence_inner(
            prompt_tokens,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            hook_ctx_ptr,
            callback_table_ptr,
            page_table,
            pool_base,
            session_position,
            fused_hidden,
            num_mm_tokens,
            Some((agnostic_start, agnostic_end)),
        )
    }

    /// SPEC/20 REQ-BCI-003/008: Batch inference via single mega-kernel CALL.
    ///
    /// One CALL: prefill (M=total_prefill_tokens) → per-seq argmax →
    /// decode step loop (M=num_active per step) → all sequences complete.
    /// Sampling params read from batch_ctx.sampling_params_ptr per-seq.
    ///
    /// Returns total decode steps completed across all sequences.
    pub fn generate_batch(
        &self,
        batch_ctx: &super::batch_context::BatchContext,
        input_ids_flat: &[u32],
        positions_flat: &[u32],
        total_prefill_tokens: usize,
        max_decode_steps: usize,
        pool_base: *const u8,
    ) -> Result<usize, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        // Scratchpad must hold both prefill + decode (max_decode_steps × num_seqs tokens)
        let num_seqs = batch_ctx.num_seqs;
        let max_decode_tokens = max_decode_steps.max(1) * num_seqs;
        let max_total = total_prefill_tokens + max_decode_tokens;
        let mut scratchpad = MmapGuardedBuffer::new(
            mega.runtime_scratchpad_bytes(max_total)
                .map_err(MegaKernelError::Execution)?,
        )?;

        // Fill RoPE cos/sin table for all positions [0..max_total).
        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = max_total * rc.head_dim;
            if rc.cache_offset + rope_elems * std::mem::size_of::<f32>() <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                let positions: Vec<u32> = (0..max_total as u32).collect();
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice,
                    &positions,
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling,
                );
            }
            if let Some(ref sec) = rc.secondary_cache {
                let sec_elems = max_total * sec.head_dim;
                if sec.cache_offset + sec_elems * std::mem::size_of::<f32>() <= scratchpad.len() {
                    let sec_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            scratchpad[sec.cache_offset..].as_mut_ptr() as *mut f32,
                            sec_elems,
                        )
                    };
                    let positions: Vec<u32> = (0..max_total as u32).collect();
                    gllm_kernels::compiler::fill_cos_sin_table_partial(
                        sec_slice,
                        &positions,
                        sec.head_dim,
                        sec.theta,
                        sec.partial,
                        sec.rope_scaling,
                    );
                }
            }
        }

        // Output tokens buffer sized for all sequences (prompt + decode)
        let mut output_tokens = vec![0u32; max_total];

        // Allocate KV cache buffer
        let kv_cache_bytes = mega.kv_cache_bytes(self.num_layers);
        let kv_cache = MmapGuardedBuffer::new(kv_cache_bytes)?;
        if std::env::var_os("GLLM_DEBUG_RESOURCE").is_some() {
            eprintln!(
                "[GUARD-ALLOC] batch scratch={:p} len={} kv={:p} len={}",
                scratchpad.as_mut_ptr(),
                scratchpad.len(),
                kv_cache.as_mut_ptr(),
                kv_cache.len(),
            );
        }
        let effective_pool_base = if !kv_cache.is_empty() {
            kv_cache.as_mut_ptr() as *const u8
        } else {
            pool_base
        };

        let generated_count = unsafe {
            // R1: Build KernelContext
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.kv_cache_ptr = effective_pool_base as *mut u8;
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();
            ctx.batch_ctx_ptr = batch_ctx.as_ptr();

            let result = match &mega.executable {
                CompiledExecutable::Cpu { entry_fn, .. } => {
                    (entry_fn)(
                        input_ids_flat.as_ptr(),
                        ctx.weight_blob_ptr,
                        ctx.kv_cache_ptr,
                        positions_flat.as_ptr(),
                        std::ptr::null(),               // aux
                        1,                              // batch_size (forward pass dimension)
                        total_prefill_tokens,           // prompt_len (forward dimension)
                        ctx.scratch_buffer_ptr,         // scratchpad
                        output_tokens.as_mut_ptr(),     // output
                        0,                              // temperature (batch mode: read from sampling_params_ptr per-seq)
                        0,                              // top_k (batch mode: read from sampling_params_ptr per-seq)
                        0,                              // top_p (batch mode: read from sampling_params_ptr per-seq)
                        max_decode_steps,               // max_new_tokens — non-zero triggers decode step loop
                        0,                              // eos_token_id (batch mode: read from sampling_params_ptr per-seq)
                        std::ptr::null(),               // hook_ctx (from batch_ctx)
                        std::ptr::null_mut(),           // telemetry
                        0,                              // session_position (from batch_ctx)
                        std::ptr::null(),               // fused_hidden (from batch_ctx)
                        0,                              // num_mm_tokens
                        std::ptr::null(),               // callback_table (from batch_ctx)
                        std::ptr::null(),               // page_table (from batch_ctx)
                        ctx.batch_ctx_ptr,              // batch_ctx_ptr — triggers JIT batch path
                        ctx.kv_page_header_ptr,         // kv_page_header_ptr
                    )
                },
                CompiledExecutable::Gpu { launcher, .. } => {
                    let args = MegaKernelArgs {
                        input_ids_ptr: input_ids_flat.as_ptr(),
                        weight_blob_ptr: ctx.weight_blob_ptr,
                        kv_cache_ptr: ctx.kv_cache_ptr,
                        positions_ptr: positions_flat.as_ptr(),
                        aux_ptr: std::ptr::null(),
                        batch_size: 1,
                        prompt_len: total_prefill_tokens,
                        scratchpad_ptr: ctx.scratch_buffer_ptr,
                        output_tokens_ptr: output_tokens.as_mut_ptr(),
                        temperature_u32: 0,
                        top_k: 0,
                        top_p_u32: 0,
                        max_new_tokens: max_decode_steps,
                        eos_token_id: 0,
                        hook_ctx_ptr: std::ptr::null(),
                        telemetry_ptr: std::ptr::null_mut(),
                        session_position: 0,
                        fused_hidden_ptr: std::ptr::null(),
                        num_mm_tokens: 0,
                        callback_table_ptr: std::ptr::null(),
                        page_table_ptr: std::ptr::null(),
                        batch_ctx_ptr: ctx.batch_ctx_ptr,
                        kv_page_header_ptr: ctx.kv_page_header_ptr,
                        scratchpad_bytes: scratchpad.len(),
                        output_tokens_bytes: output_tokens.len() * 4,
                    };
                    launcher(&args)?;
                    // GPU arm: generated_count 占位 = max_decode_steps (D2H copy 后由 3C 校准)
                    max_decode_steps
                }
            };
            result
        };

        log::debug!(
            "[mega] batch: total_prefill_tokens={} max_decode_steps={} generated_count={}",
            total_prefill_tokens,
            max_decode_steps,
            generated_count,
        );

        Ok(generated_count)
    }
}
