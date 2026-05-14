//! Mega-Kernel 执行器 (SPEC §9.1)
//!
//! ARCH-RUST-IS-CODEGEN 铁律: 推理时 Rust 只做一次 CALL。
//! 整个 decoder 模型（embedding → N 层 → lm_head → sampling → generate loop）
//! 编译为单一 JIT 机器码，推理时通过 MegaKernelFn 单次 CALL 完成。
//! Encoder 模型（embedding/rerank/classify）通过 CompiledLayerFn 单次 CALL 完成。
//!
//! 无 fallback。编译失败 = 致命错误。

use gllm_kernels::types::DType;

// ============================================================================
// MegaKernelExecutor
// ============================================================================

/// Mega-Kernel 编译错误
#[derive(Debug, thiserror::Error)]
pub enum MegaKernelError {
    #[error("compilation failed: {0}")]
    Compilation(String),
    #[error("execution failed: {0}")]
    Execution(String),
}

/// True mega-kernel 编译产物。
///
/// 持有完整的 mega-kernel 机器码（embedding → layer loop → lm_head → sampling → generate loop）
/// + 全模型权重布局 + 缓冲布局。推理时通过单次 CALL 执行。
struct MegaKernelCompiled {
    /// graph.weight_layout() — 权重 blob 中每个 tensor 的偏移
    weight_layout: gllm_kernels::compiler::graph::WeightLayout,
    /// (canonical_name, byte_offset) 对 — 用于诊断查询
    named_offsets: Vec<(String, usize)>,
    /// 运行时缓冲布局（activation ping/pong, logits, sampling workspace）
    buffer_layout: gllm_kernels::compiler::MegaKernelBufferLayout,
    /// Logits 区域在 scratchpad 中的偏移（alloc + RoPE cache 之后）
    logits_scratch_offset: usize,
    /// 预打包的连续权重 blob
    weight_blob: Vec<u8>,
    /// mmap'd 完整 mega-kernel 机器码（generate loop + embedded forward code，单一连续函数）
    exec_code: gllm_kernels::compiler::CompiledLayer,
    /// MegaKernelFn 函数指针（指向 exec_code 的入口）
    entry_fn: gllm_kernels::compiler::MegaKernelFn,
    /// RoPE cos/sin 表需求（caller 必须在每次调用前填充 scratchpad）
    rope_cache: Option<gllm_kernels::compiler::codegen::RopeCacheRequirement>,
    /// scratchpad 固定部分大小（intermediate tensors + RoPE cache），不含运行时 logits
    scratchpad_base_bytes: usize,
    /// GPU mega-kernel PTX/HIP 代码（可选，仅 GPU 路径使用）
    gpu_code: Option<Vec<u8>>,
    /// vocab_size — logits 每行元素数
    vocab_size: usize,
    /// hidden_dim — SG scratchpad 需要
    hidden: usize,
    /// 每元素字节数 (compute dtype)
    elem_bytes: usize,
}

impl MegaKernelCompiled {
    /// 计算运行时 scratchpad 大小：固定部分 + logits(max_total 行) + sampling + SG
    fn runtime_scratchpad_bytes(&self, max_total: usize) -> usize {
        let vocab_bytes = self.vocab_size * self.elem_bytes;
        let logits_bytes = max_total * vocab_bytes;
        let sampling_bytes = vocab_bytes * 4;
        let sg_end = if self.buffer_layout.sg_data_bytes > 0 {
            let sg_start = (self.logits_scratch_offset + logits_bytes + sampling_bytes + 63) & !63;
            sg_start + self.hidden * self.elem_bytes * 2
        } else {
            0
        };
        let total = (self.scratchpad_base_bytes + logits_bytes)
            .max(sg_end)
            .max(self.buffer_layout.total_scratchpad_bytes)
            .max(64);

        total
    }
}

/// Forward-only 编译产物 (encoder 路径: embedding / rerank / classify)。
///
/// 通过 `InferenceCompiler::compile_graph` 编译任意 CompilerGraph，
/// 推理时通过 `CompiledLayerFn` (10 参数通用 ABI) 单次 CALL 完成。
struct ForwardCompiled {
    /// JIT 编译产物（通用 forward-pass 机器码）
    exec_code: gllm_kernels::compiler::CompiledLayer,
    /// CompiledLayerFn 函数指针（10 参数通用 ABI）
    entry_fn: gllm_kernels::compiler::CompiledLayerFn,
    /// 预打包的连续权重 blob（按 graph.weight_layout() 排列）
    weight_blob: Vec<u8>,
    /// graph.weight_layout() 记录的权重布局
    weight_layout: gllm_kernels::compiler::graph::WeightLayout,
    /// scratchpad 大小
    total_scratchpad_bytes: usize,
    /// Output buffer 所需字节数 (基于 graph outputs shape + graph.max_seq_len)
    output_bytes: usize,
    /// GPU forward-only PTX/HIP 代码（可选）
    gpu_code: Option<Vec<u8>>,
}

/// Mega-Kernel 执行器 (§9.1)
///
/// 唯一推理路径: 编译 → 单次 CALL。
/// 编译在模型加载时完成，推理时零 Rust 开销。
pub struct MegaKernelExecutor {
    /// Decoder mega-kernel 编译产物 (generate loop 路径)
    mega_compiled: Option<MegaKernelCompiled>,
    /// Forward-only 编译产物 (encoder 路径: embedding / rerank / classify)
    forward_compiled: Option<ForwardCompiled>,
    /// 模型配置
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    dtype: DType,
    /// EOS token ID — 从 ModelConfig 读取，传给 JIT 停止条件
    eos_token_id: u32,
    /// §19 KV-OPT-009: Mega-Kernel Variant 矩阵 (按 PrecisionTier 编译独立 Variant)
    variant_registry: crate::jit::variant_registry::VariantRegistry,
}

impl MegaKernelExecutor {
    /// 从 auto-derived CompilerGraph 编译 decoder mega-kernel。
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
        business_config: gllm_kernels::compiler::MegaKernelBusinessConfig,
        hetero_config: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig>,
        gpu_sm_version: Option<u32>,
    ) -> Result<Self, MegaKernelError> {
        // Derive all geometry from graph — CompilerGraph is the SSOT.
        let geometry = gllm_kernels::compiler::graph_geometry::GraphDerivedGeometry::from_graph(&graph)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;


        // Build slim config with only non-derivable fields.
        let config = gllm_kernels::compiler::mega_kernel_abi::MegaKernelConfig {
            max_seq_len,
            num_eos_tokens: 1,
            business_config,
            hetero: hetero_config.clone(),
        };

        let hetero_layout = hetero_config.as_ref().map(|hc| {
            gllm_kernels::compiler::mega_kernel_abi::HeteroWeightLayout::from_geometry_and_config(&geometry, hc)
        });

        // Pre-resolve weight layout before moving graph into compiler.
        // Clone graph for GPU compilation (CPU compiler takes ownership).
        let graph_for_gpu = if gpu_sm_version.is_some() {
            Some(graph.clone())
        } else {
            None
        };
        let weight_layout = graph.weight_layout();
        let named_offsets: Vec<(String, usize)> = weight_layout.offsets.iter()
            .filter_map(|&(tid, offset)| {
                graph.tensors.get(tid.0 as usize).map(|t| (t.name.clone(), offset))
            })
            .collect();
        let total_weight_bytes = weight_layout.total_bytes;

        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let output = compiler.compile_mega_kernel_from_graph(graph, &config, hetero_layout)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        let exec_code = output.layer_code;
        let entry_fn = unsafe { exec_code.entry_point_as_mega_kernel() };
        if std::env::var("GLLM_DUMP_JIT_CODE").is_ok() {
            let dump_path = "/tmp/jit_code_mega_auto.bin";
            let _ = std::fs::write(dump_path, exec_code.code_bytes());
        }

        // Pack weights using pre-resolved named offsets — unified with encoder path.
        let weight_blob = pack_weights_from_graph(
            &named_offsets,
            total_weight_bytes,
            weight_ptrs,
            weight_sizes,
            raw_floats,
            name_map,
        );

        // Compile GPU PTX/HIP code if sm_version provided and GPU JIT backend available.
        #[cfg(any(feature = "cuda", feature = "hip"))]
        let gpu_code = match (graph_for_gpu, gpu_sm_version) {
            (Some(g), Some(sm)) => {
                match compiler.compile_mega_kernel_to_gpu(g, &config, sm) {
                    Ok(gpu_output) => {
                        log::info!(
                            "[mega] GPU PTX compiled: {} bytes, {} layers",
                            gpu_output.gpu_code.len(),
                            gpu_output.num_layers,
                        );
                        Some(gpu_output.gpu_code)
                    }
                    Err(e) => {
                        log::warn!("[mega] GPU compilation failed (GPU path unavailable): {e}");
                        None
                    }
                }
            }
            _ => None,
        };
        #[cfg(not(any(feature = "cuda", feature = "hip")))]
        let gpu_code = {
            let _ = (graph_for_gpu, gpu_sm_version);
            None
        };

        let mega_compiled = MegaKernelCompiled {
            weight_layout,
            named_offsets,
            buffer_layout: output.buffer_layout,
            logits_scratch_offset: output.logits_scratch_offset,
            weight_blob,
            exec_code,
            entry_fn,
            rope_cache: output.rope_cache,
            scratchpad_base_bytes: output.logits_scratch_offset,
            vocab_size: output.vocab_size,
            hidden: output.hidden,
            elem_bytes: geometry.compute_dtype.size_bytes(),
            gpu_code,
        };

        Ok(Self {
            mega_compiled: Some(mega_compiled),
            forward_compiled: None,
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden,
            vocab_size: geometry.vocab_size,
            dtype: geometry.compute_dtype,
            eos_token_id,
            variant_registry: {
                let mut registry = crate::jit::variant_registry::VariantRegistry::new();
                // Register default FP16 variant (the compiled mega-kernel itself)
                let default_key = crate::jit::variant_registry::VariantRegistry::derive_key(
                    "default", false, false, None, false, 64, None, None,
                );
                let default_variant = crate::jit::variant_registry::CompiledVariant {
                    code: Vec::new(), // The actual code lives in mega_compiled
                    instruction_footprint_bytes: 8192, // estimated
                    mechanisms: vec![crate::jit::variant_registry::MechanismId::Dense],
                    section: crate::jit::variant_registry::CodeSection::Hot,
                    key: default_key.clone(),
                };
                let _ = registry.register(default_variant);
                registry
            },
        })
    }

    /// 从 auto-derived CompilerGraph 编译 forward-only encoder mega-kernel。
    ///
    /// Graph 由 `auto_graph::build_compiler_graph()` 从 tensor names + shapes 生成。
    pub fn compile_forward_from_graph(
        graph: gllm_kernels::compiler::graph::CompilerGraph,
        geometry: &crate::model_config::ModelGeometry,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        raw_floats: &std::collections::HashMap<String, crate::loader::RawFloatTensor>,
        name_map: &crate::loader::name_map::TensorNameMap,
        gpu_sm_version: Option<u32>,
    ) -> Result<Self, MegaKernelError> {
        let weight_layout = graph.weight_layout();

        // Pre-resolve named offsets for packing.
        let named_offsets: Vec<(String, usize)> = weight_layout.offsets.iter()
            .filter_map(|&(tid, offset)| {
                graph.tensors.get(tid.0 as usize).map(|t| (t.name.clone(), offset))
            })
            .collect();
        let total_weight_bytes = weight_layout.total_bytes;

        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let exec_code = compiler.compile_graph(&graph)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        let entry_fn = unsafe { exec_code.entry_point() };
        let total_scratchpad_bytes = exec_code.scratchpad_bytes;

        let max_seq = graph.max_seq_len;
        let output_bytes: usize = graph.outputs.iter()
            .map(|&tid| {
                let t = graph.tensor(tid).expect("output_bytes: graph output tensor must exist");
                let numel = graph.tensor_numel_for_alloc(tid, max_seq).unwrap_or(0);
                numel * t.dtype.size_bytes()
            })
            .sum();


        if std::env::var("GLLM_DUMP_JIT_CODE").is_ok() {
            let dump_path = "/tmp/jit_code_forward_auto.bin";
            let _ = std::fs::write(dump_path, exec_code.code_bytes());
        }

        let weight_blob = pack_weights_from_graph(
            &named_offsets,
            total_weight_bytes,
            weight_ptrs,
            weight_sizes,
            raw_floats,
            name_map,
        );

        // Forward-only GPU compilation.
        #[cfg(any(feature = "cuda", feature = "hip"))]
        let gpu_code = match gpu_sm_version {
            Some(sm) => {
                match compiler.compile_graph_to_gpu(&graph, sm) {
                    Ok(gpu_output) => {
                        log::info!(
                            "[mega] GPU forward PTX compiled: {} bytes, {} scratchpad",
                            gpu_output.gpu_code.len(),
                            gpu_output.total_scratchpad_bytes,
                        );
                        Some(gpu_output.gpu_code)
                    }
                    Err(e) => {
                        log::warn!("[mega] GPU forward compilation failed: {e}");
                        None
                    }
                }
            }
            None => None,
        };
        #[cfg(not(any(feature = "cuda", feature = "hip")))]
        let gpu_code = {
            let _ = gpu_sm_version;
            None
        };

        let forward_compiled = ForwardCompiled {
            exec_code,
            entry_fn,
            weight_blob,
            weight_layout,
            total_scratchpad_bytes,
            output_bytes,
            gpu_code,
        };

        Ok(Self {
            mega_compiled: None,
            forward_compiled: Some(forward_compiled),
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden_size,
            vocab_size: geometry.vocab_size,
            dtype: geometry.dtype,
            eos_token_id: 0,
            variant_registry: crate::jit::variant_registry::VariantRegistry::new(),
        })
    }

    /// 执行 forward-only 推理 (embedding / rerank / classify)。
    ///
    /// ARCH-RUST-IS-CODEGEN: 一次 CALL 完成。
    /// CompiledLayerFn ABI (10 参数):
    ///   input_ptr → token IDs, weight_ptr → weight blob,
    ///   output → 输出缓冲, scratchpad → 临时空间。
    pub fn execute_forward(
        &self,
        input_ids: &[u32],
        output_elems: usize,
    ) -> Result<Vec<f32>, MegaKernelError> {
        let fw = self.forward_compiled.as_ref()
            .ok_or_else(|| MegaKernelError::Execution(
                "not a forward-compiled model".into()
            ))?;

        let seq_len = input_ids.len();
        let mut scratchpad = vec![0u8; fw.total_scratchpad_bytes];
        let jit_output_elems = fw.output_bytes / 4;
        let actual_output_elems = output_elems.max(jit_output_elems);
        let mut output = vec![0.0f32; actual_output_elems];

        unsafe {
            (fw.entry_fn)(
                input_ids.as_ptr() as *const u8,
                fw.weight_blob.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                seq_len,
                output.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
                std::ptr::null_mut(),
            );
        }
        let nan_count = output.iter().filter(|&&x| x.is_nan()).count();
        // Check scratchpad: sample first hidden values at the beginning (layer output)
        if scratchpad.len() >= 32 {
            let sp_f32 = unsafe { std::slice::from_raw_parts(scratchpad.as_ptr() as *const f32, scratchpad.len() / 4) };
        }
        output.truncate(output_elems);
        Ok(output)
    }

    /// Returns total scratchpad bytes needed for execution.
    pub fn total_scratchpad_bytes(&self) -> usize {
        {
            if let Some(ref mega) = self.mega_compiled {
                return mega.scratchpad_base_bytes;
            }
            if let Some(ref fw) = self.forward_compiled {
                return fw.total_scratchpad_bytes;
            }
        }
        0
    }

    /// 单序列 mega-kernel 生成。
    ///
    /// ARCH-RUST-IS-CODEGEN: 一次 CALL 完成。
    /// JIT mega-kernel 内部执行完整的 generate loop:
    ///   LoopBegin → embed → N 层 → lm_head → Argmax → StoreToken → CheckStopCondition → LoopEnd
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
        session_position: usize,
        fused_hidden: Option<&[f32]>,
        num_mm_tokens: usize,
    ) -> Result<Vec<u32>, MegaKernelError> {
        let mega = self.mega_compiled.as_ref()
            .ok_or_else(|| MegaKernelError::Execution(
                "not a decoder mega-kernel model".into()
            ))?;

        let prompt_len = prompt_tokens.len();
        let max_total = prompt_len + max_new_tokens;

        let mut input_ids = vec![0u32; max_total];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..max_total as u32).collect();
        let mut output_tokens = vec![0u32; max_new_tokens];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(max_total)];

        // Pre-fill RoPE cos/sin table for all positions [0..max_total).
        if let Some(ref rc) = mega.rope_cache {
            // Primary cache
            let rope_elems = max_total * rc.head_dim;
            let rope_bytes = rope_elems * 4;
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
                    rc.rope_scaling.clone(),
                );
            }
            // Secondary cache (for heterogeneous models with 2 head_dim values)
            if let Some(ref sec) = rc.secondary_cache {
                let sec_elems = max_total * sec.head_dim;
                let sec_bytes = sec_elems * 4;
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
                        sec.rope_scaling.clone(),
                    );
                }
            }
        }
        let page_table_ptr = page_table.map_or(std::ptr::null(), |pt| pt.as_ptr());

        let generated_count = unsafe {
            (mega.entry_fn)(
                input_ids.as_ptr(),
                mega.weight_blob.as_ptr(),
                std::ptr::null_mut(),
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                scratchpad.as_mut_ptr(),
                output_tokens.as_mut_ptr(),
                temperature.to_bits() as usize,
                top_k,
                top_p.to_bits() as usize,
                max_new_tokens,
                self.eos_token_id as usize,
                0,  // output_mode_selector: Generate
                hook_ctx_ptr,  // hook_ctx_ptr: SG shared memory (NULL = disabled)
                std::ptr::null_mut(),
                session_position,  // session_position (0=new, >0=resume)
                fused_hidden.map_or(std::ptr::null(), |fh| fh.as_ptr() as *const u8),  // fused_hidden_ptr
                num_mm_tokens,  // num_mm_tokens
                callback_table_ptr, // callback_table_ptr: C-style fn_ptr array (NULL = no callbacks)
                page_table_ptr,  // page_table_ptr: NULL = contiguous KV, u32[] = paged KV
            )
        };

        // Diagnostic: check scratchpad beginning (after embed gather)
        {
            let sp_f32 = unsafe { std::slice::from_raw_parts(scratchpad.as_ptr() as *const f32, 64.min(scratchpad.len() / 4)) };

        }
        // Diagnostic: check logits for last few rows
        {
            let logits_off = mega.logits_scratch_offset;
            let vocab = self.vocab_size;
            let row_bytes = vocab * 4;
            let total_rows = (generated_count as usize) + prompt_len;
            for row_idx in [prompt_len - 1, prompt_len, prompt_len + 1] {
                if row_idx < total_rows {
                    let row_off = logits_off + row_idx * row_bytes;
                    if row_off + row_bytes <= scratchpad.len() {
                        let row_data = unsafe {
                            std::slice::from_raw_parts(scratchpad[row_off..].as_ptr() as *const f32, vocab)
                        };
                        let (max_idx, max_val) = row_data.iter().enumerate()
                            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                                if v > acc.1 { (i, v) } else { acc }
                            });
                        let nonzero = row_data.iter().filter(|&&v| v != 0.0).count();
                    }
                }
            }
        }

        log::debug!(
            "[mega] prompt_len={} max_new_tokens={} generated_count={} eos={} output_first={}",
            prompt_len, max_new_tokens, generated_count, self.eos_token_id,
            output_tokens.first().copied().unwrap_or(0),
        );

        let actual_count = if generated_count == 0 && max_new_tokens > 0 && output_tokens[0] != 0 {
            1
        } else {
            generated_count
        };

        Ok(output_tokens[..actual_count].to_vec())
    }

    /// Diagnostic: run prefill only (max_new_tokens=0) and return logits from scratchpad.
    ///
    /// Returns logits for the last prompt token: shape [vocab_size].
    pub fn diagnostic_prefill_logits(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self.mega_compiled.as_ref()
            .ok_or_else(|| MegaKernelError::Execution(
                "not a decoder mega-kernel model".into()
            ))?;

        let prompt_len = prompt_tokens.len();
        let mut input_ids = vec![0u32; prompt_len + 1];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..(prompt_len + 1) as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(prompt_len + 1)];

        // Pre-fill RoPE cache
        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = (prompt_len + 1) * rc.head_dim;
            let rope_bytes = rope_elems * 4;
            if rc.cache_offset + rope_bytes <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice,
                    &positions,
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling.clone(),
                );
            }
        }

        // Run with max_new_tokens=1 to get prefill logits
        let _generated = unsafe {
            (mega.entry_fn)(
                input_ids.as_ptr(),
                mega.weight_blob.as_ptr(),
                std::ptr::null_mut(),
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                scratchpad.as_mut_ptr(),
                output_tokens.as_mut_ptr(),
                0,  // temperature=0 (greedy)
                1,  // top_k=1
                0,  // top_p=0
                1,  // max_new_tokens=1
                self.eos_token_id as usize,
                0,  // output_mode_selector: Generate
                std::ptr::null(),  // hook_ctx_ptr
                std::ptr::null_mut(),  // telemetry
                0,  // session_position
                std::ptr::null(),  // fused_hidden_ptr
                0,  // num_mm_tokens
                std::ptr::null(),  // callback_table_ptr
                std::ptr::null(),  // page_table_ptr
            )
        };

        // Read logits for last prompt token from scratchpad
        let logits_off = mega.logits_scratch_offset;
        let vocab = self.vocab_size;
        let row_bytes = vocab * 4;
        // Last prompt token's logits are at row (prompt_len - 1)
        let last_row_off = logits_off + (prompt_len - 1) * row_bytes;

        let mut logits = vec![0.0f32; vocab];
        if last_row_off + row_bytes <= scratchpad.len() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratchpad[last_row_off..].as_ptr() as *const f32,
                    logits.as_mut_ptr(),
                    vocab,
                );
            }
        }

        Ok(logits)
    }
    /// Diagnostic: run prefill and return the full scratchpad for intermediate inspection.
    pub fn diagnostic_prefill_scratchpad(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<DiagnosticScratchpad, MegaKernelError> {
        let mega = self.mega_compiled.as_ref()
            .ok_or_else(|| MegaKernelError::Execution(
                "not a decoder mega-kernel model".into()
            ))?;

        let prompt_len = prompt_tokens.len();
        let mut input_ids = vec![0u32; prompt_len + 1];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..(prompt_len + 1) as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(prompt_len + 1)];

        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = (prompt_len + 1) * rc.head_dim;
            let rope_bytes = rope_elems * 4;
            if rc.cache_offset + rope_bytes <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice,
                    &positions,
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling.clone(),
                );
            }
        }

        let _generated = unsafe {
            (mega.entry_fn)(
                input_ids.as_ptr(),
                mega.weight_blob.as_ptr(),
                std::ptr::null_mut(),
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                scratchpad.as_mut_ptr(),
                output_tokens.as_mut_ptr(),
                0, 1, 0, 1,
                self.eos_token_id as usize,
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        Ok(DiagnosticScratchpad {
            data: scratchpad,
            logits_offset: mega.logits_scratch_offset,
            vocab_size: self.vocab_size,
            prompt_len,
            hidden_size: self.hidden_size,
        })
    }

    /// Execute mega-kernel in EncodeToLayer mode (output_mode_selector = 3).
    ///
    /// Runs a single forward pass through all layers, then extracts the hidden
    /// state from the activation buffer. The mega-kernel skips the generate loop
    /// and argmax, jumping directly to the encode path.
    ///
    /// Returns the hidden state as a Vec<f32> of shape [seq_len * hidden_size].
    pub fn execute_encode(
        &self,
        prompt_tokens: &[u32],
        output_elems: usize,
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self.mega_compiled.as_ref()
            .ok_or_else(|| MegaKernelError::Execution(
                "not a decoder mega-kernel model".into()
            ))?;

        let prompt_len = prompt_tokens.len();
        let mut input_ids = vec![0u32; prompt_len];
        input_ids.copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..prompt_len as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(prompt_len)];

        // Pre-fill RoPE cache
        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = prompt_len * rc.head_dim;
            let rope_bytes = rope_elems * 4;
            if rc.cache_offset + rope_bytes <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice,
                    &positions,
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling.clone(),
                );
            }
        }

        let _generated_count = unsafe {
            (mega.entry_fn)(
                input_ids.as_ptr(),
                mega.weight_blob.as_ptr(),
                std::ptr::null_mut(),       // kv_cache: null
                positions.as_ptr(),
                std::ptr::null(),           // seq_lens: null
                1,                          // batch_size
                prompt_len,
                scratchpad.as_mut_ptr(),
                output_tokens.as_mut_ptr(),
                0,                          // temperature (unused in encode mode)
                0,                          // top_k (unused)
                0,                          // top_p bits (unused)
                1,                          // max_new_tokens = 1 (minimal)
                self.eos_token_id as usize,
                3,                          // output_mode_selector: EncodeToLayer
                std::ptr::null(),           // hook_ctx_ptr: null
                std::ptr::null_mut(),       // telemetry: null
                0,                          // session_position: new
                std::ptr::null(),           // fused_hidden_ptr: no MM
                0,                          // num_mm_tokens: 0
                std::ptr::null(),           // callback_table_ptr: null
                std::ptr::null(),           // page_table_ptr: null (contiguous KV)
            )
        };

        // The activation buffer is at the beginning of the scratchpad.
        // BufferAllocation places intermediate tensors sequentially from offset 0.
        // For EncodeToLayer, the final hidden state is the last activation buffer content.
        let mut output = vec![0.0f32; output_elems];
        let copy_bytes = output_elems * 4;
        if copy_bytes <= scratchpad.len() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratchpad.as_ptr() as *const f32,
                    output.as_mut_ptr(),
                    output_elems,
                );
            }
        }

        Ok(output)
    }

    /// Execute mega-kernel in ClassifyBinary mode for decoder-based reranker.
    ///
    /// Runs forward pass through all layers (including lm_head), then extracts
    /// the logits for yes/no tokens from the scratchpad logits region.
    /// Returns [score_for_yes_token, score_for_no_token].
    pub fn execute_rerank(
        &self,
        prompt_tokens: &[u32],
        yes_token_id: u32,
        no_token_id: u32,
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self.mega_compiled.as_ref()
            .ok_or_else(|| MegaKernelError::Execution(
                "not a decoder mega-kernel model".into()
            ))?;

        let prompt_len = prompt_tokens.len();
        let mut input_ids = vec![0u32; prompt_len];
        input_ids.copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..prompt_len as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(prompt_len)];

        // Pre-fill RoPE cache
        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = prompt_len * rc.head_dim;
            let rope_bytes = rope_elems * 4;
            if rc.cache_offset + rope_bytes <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice,
                    &positions,
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling.clone(),
                );
            }
        }

        let _generated_count = unsafe {
            (mega.entry_fn)(
                input_ids.as_ptr(),
                mega.weight_blob.as_ptr(),
                std::ptr::null_mut(),
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                scratchpad.as_mut_ptr(),
                output_tokens.as_mut_ptr(),
                0,
                0,
                0,
                1,                          // max_new_tokens = 1 (one iteration for forward pass)
                self.eos_token_id as usize,
                1,                          // output_mode_selector: ClassifyBinary
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        // Extract logits for yes/no tokens from the last token position
        let logits_off = mega.logits_scratch_offset;
        let vocab = self.vocab_size;
        let row_bytes = vocab * 4;
        // Last token position = prompt_len - 1
        let logits_row_off = logits_off + (prompt_len - 1) * row_bytes;

        let yes_logit = if logits_row_off + (yes_token_id as usize + 1) * 4 <= scratchpad.len() {
            unsafe {
                let ptr = scratchpad[logits_row_off..].as_ptr() as *const f32;
                *ptr.add(yes_token_id as usize)
            }
        } else {
            0.0f32
        };

        let no_logit = if logits_row_off + (no_token_id as usize + 1) * 4 <= scratchpad.len() {
            unsafe {
                let ptr = scratchpad[logits_row_off..].as_ptr() as *const f32;
                *ptr.add(no_token_id as usize)
            }
        } else {
            0.0f32
        };
        // Softmax: score = exp(yes) / (exp(yes) + exp(no))
        let max_logit = yes_logit.max(no_logit);
        let exp_yes = (yes_logit - max_logit).exp();
        let exp_no = (no_logit - max_logit).exp();
        let score = exp_yes / (exp_yes + exp_no);

        Ok(vec![score])
    }

    /// Returns true if this executor has a decoder mega-kernel (compile_from_template/geometry path).
    pub fn has_mega_compiled(&self) -> bool {
        self.mega_compiled.is_some()
    }

    /// Returns the expected output element count for embedding inference.
    ///
    /// - Decoder path (mega_compiled): seq_len × hidden_size (raw hidden states)
    /// - Encoder path (forward_compiled): hidden_size (after MeanPool in JIT graph)
    pub fn output_elems_for_embed(&self, seq_len: usize, hidden_size: usize) -> usize {
        if self.mega_compiled.is_some() {
            seq_len * hidden_size
        } else if let Some(ref fw) = self.forward_compiled {
            // Use actual JIT-compiled output size
            fw.output_bytes / 4
        } else {
            hidden_size
        }
    }

    /// Returns true if this executor has a forward-only compiled layer (encoder path).
    pub fn has_forward_compiled(&self) -> bool {
        self.forward_compiled.is_some()
    }

    /// Returns the byte offset of a named weight tensor in the weight blob.
    pub fn weight_offset(&self, name: &str) -> Option<usize> {
        let mega = self.mega_compiled.as_ref()?;
        mega.named_offsets.iter()
            .find(|(n, _)| n == name)
            .map(|(_, off)| *off)
    }

    /// Returns all named weight offsets for diagnostic inspection.
    pub fn weight_offsets(&self) -> Option<&[(String, usize)]> {
        self.mega_compiled.as_ref().map(|m| m.named_offsets.as_slice())
    }

    /// Read F32 weight data at a given blob offset + row offset.
    pub fn read_weight_row(&self, tensor_name: &str, row: usize, cols: usize) -> Option<Vec<f32>> {
        let mega = self.mega_compiled.as_ref()?;
        let offset = mega.named_offsets.iter()
            .find(|(n, _)| n == tensor_name)
            .map(|(_, off)| *off)?;
        let row_offset = offset + row * cols * 4;
        if row_offset + cols * 4 > mega.weight_blob.len() {
            return None;
        }
        let data = unsafe {
            std::slice::from_raw_parts(
                mega.weight_blob[row_offset..].as_ptr() as *const f32,
                cols,
            )
        };
        Some(data.to_vec())
    }

    /// Returns the weight blob for GPU upload.
    pub fn weight_blob(&self) -> Option<&[u8]> {
        if let Some(ref mega) = self.mega_compiled {
            Some(&mega.weight_blob)
        } else if let Some(ref fw) = self.forward_compiled {
            Some(&fw.weight_blob)
        } else {
            None
        }
    }

    /// Returns the GPU PTX/HIP code if available.
    pub fn gpu_code(&self) -> Option<&[u8]> {
        if let Some(ref mega) = self.mega_compiled {
            mega.gpu_code.as_deref()
        } else if let Some(ref fw) = self.forward_compiled {
            fw.gpu_code.as_deref()
        } else {
            None
        }
    }

    /// Returns total scratchpad bytes needed.
    pub fn scratchpad_bytes(&self) -> usize {
        if let Some(ref mega) = self.mega_compiled {
            mega.scratchpad_base_bytes
        } else if let Some(ref fw) = self.forward_compiled {
            fw.total_scratchpad_bytes
        } else {
            0
        }
    }

    /// Store GPU mega-kernel PTX/HIP code for decoder path.
    pub fn set_decoder_gpu_code(&mut self, code: Vec<u8>) {
        if let Some(ref mut mega) = self.mega_compiled {
            mega.gpu_code = Some(code);
        }
    }

    /// Store GPU forward-only PTX/HIP code for encoder path.
    pub fn set_forward_gpu_code(&mut self, code: Vec<u8>) {
        if let Some(ref mut fw) = self.forward_compiled {
            fw.gpu_code = Some(code);
        }
    }

    /// Diagnostic: run forward-only compilation of the same graph, execute once.
    /// Bypasses mega-kernel generate loop to isolate forward pass correctness.
    pub fn diagnostic_forward_only(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self.mega_compiled.as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a decoder mega-kernel model".into()))?;

        // We reuse the mega-kernel weight_blob — same weights, same layout.
        // But compile a fresh forward-only graph via InferenceCompiler::compile_graph.
        // This requires the original CompilerGraph, which we do not store.
        // Instead, we run the mega-kernel with output_mode_selector=3 (encode mode)
        // which runs the forward pass and returns gen_counter (skips argmax/generate).

        let prompt_len = prompt_tokens.len();
        let mut input_ids = vec![0u32; prompt_len];
        input_ids.copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..prompt_len as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(prompt_len)];

        // Pre-fill RoPE cache
        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = prompt_len * rc.head_dim;
            let rope_bytes = rope_elems * 4;
            if rc.cache_offset + rope_bytes <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice, &positions, rc.head_dim, rc.theta, rc.partial, rc.rope_scaling.clone(),
                );
            }
        }

        // Run with output_mode_selector=3 (encode) — forward pass only, no generate loop logic
        let _result = unsafe {
            (mega.entry_fn)(
                input_ids.as_ptr(),
                mega.weight_blob.as_ptr(),
                std::ptr::null_mut(),
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                scratchpad.as_mut_ptr(),
                output_tokens.as_mut_ptr(),
                0, 1, 0,
                1,                  // max_new_tokens=1
                self.eos_token_id as usize,
                3,                  // output_mode_selector=3 (encode!)
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        // Read logits from scratchpad (same offset as generate mode)
        let logits_off = mega.logits_scratch_offset;
        let vocab = self.vocab_size;
        let row_bytes = vocab * 4;
        let last_row_off = logits_off + (prompt_len - 1) * row_bytes;

        let mut logits = vec![0.0f32; vocab];
        if last_row_off + row_bytes <= scratchpad.len() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratchpad[last_row_off..].as_ptr() as *const f32,
                    logits.as_mut_ptr(),
                    vocab,
                );
            }
        }

        Ok(logits)
    }
}

// ============================================================================
// Weight Blob Packing (graph.weight_layout() driven)
// ============================================================================

/// 通用权重打包：按预解析的 (name, offset) 对将权重排列到连续 blob。
///
/// 完全由图结构驱动，所有名字都是 canonical name。
/// weight_ptrs/weight_sizes 以 canonical name 为 key。
/// raw_floats 以外部名为 key，通过 name_map 做反向查找。
fn pack_weights_from_graph(
    named_offsets: &[(String, usize)],
    total_bytes: usize,
    weight_ptrs: &std::collections::HashMap<String, *const u8>,
    weight_sizes: &std::collections::HashMap<String, usize>,
    raw_floats: &std::collections::HashMap<String, crate::loader::RawFloatTensor>,
    name_map: &crate::loader::name_map::TensorNameMap,
) -> Vec<u8> {
    let mut blob = vec![0u8; total_bytes];
    let mut packed_count = 0usize;
    let mut missing_count = 0usize;
    if !raw_floats.is_empty() {
        for (cn, offset) in named_offsets {
            if cn.contains("qkv_proj") || cn.contains("gate_proj") {
                let ext = name_map.resolve_external_to_string(cn);
                if let Some(raw) = raw_floats.get(&ext) {
                    let esz = match raw.dtype {
                        ::safetensors::Dtype::BF16 | ::safetensors::Dtype::F16 => 2,
                        _ => 4,
                    };
                    let numel = raw.data.len() / esz;
                    eprintln!("[pack] {} -> {} dtype={:?} numel={} f32_bytes={} offset={}", cn, ext, raw.dtype, numel, numel * 4, offset);
                } else {
                    eprintln!("[pack] {} -> {} NOT IN raw_floats", cn, ext);
                }
            }
        }
    }

    for (canonical_name, offset) in named_offsets {
        // BF16/F16 raw float tensors need widening to F32 for GEMM computation.
        // Resolve canonical → external name for raw_floats lookup.
        let ext_name = name_map.resolve_external_to_string(canonical_name);
        if let Some(raw) = raw_floats.get(&ext_name) {
            let elem_size = match raw.dtype {
                ::safetensors::Dtype::BF16 | ::safetensors::Dtype::F16 => 2,
                ::safetensors::Dtype::F32 => 4,
                _ => 4,
            };
            let numel = raw.data.len() / elem_size;
            let f32_bytes = numel * 4;
            let copy_size = f32_bytes.min(blob.len().saturating_sub(*offset));
            if copy_size == 0 || *offset >= blob.len() {
                continue;
            }
            let dst_f32s = unsafe {
                std::slice::from_raw_parts_mut(
                    blob[*offset..].as_mut_ptr() as *mut f32,
                    copy_size / 4,
                )
            };
            match raw.dtype {
                ::safetensors::Dtype::BF16 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            raw.data.as_ptr() as *const half::bf16,
                            numel,
                        )
                    };
                    for (i, &v) in src.iter().enumerate() {
                        if i >= dst_f32s.len() { break; }
                        dst_f32s[i] = v.to_f32();
                    }
                }
                ::safetensors::Dtype::F16 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            raw.data.as_ptr() as *const half::f16,
                            numel,
                        )
                    };
                    for (i, &v) in src.iter().enumerate() {
                        if i >= dst_f32s.len() { break; }
                        dst_f32s[i] = v.to_f32();
                    }
                }
                _ => {
                    let copy_size = raw.data.len().min(blob.len().saturating_sub(*offset));
                    blob[*offset..*offset + copy_size].copy_from_slice(&raw.data[..copy_size]);
                }
            }
            packed_count += 1;
            continue;
        }

        // Standard weight: direct lookup by canonical name.
        let ptr = match weight_ptrs.get(canonical_name) {
            Some(&p) if !p.is_null() => p,
            _ => {
                missing_count += 1;
                if missing_count <= 5 {
                }
                continue;
            }
        };
        let size = *weight_sizes.get(canonical_name).unwrap_or(&0);
        if size == 0 {
            continue;
        }

        let copy_size = size.min(blob.len().saturating_sub(*offset));
        if copy_size == 0 || *offset >= blob.len() {
            continue;
        }
        let src = unsafe { std::slice::from_raw_parts(ptr, copy_size) };
        blob[*offset..*offset + copy_size].copy_from_slice(src);
        packed_count += 1;
    }
    // Sample weight blob to verify packing
    let _ = (packed_count, missing_count);
    blob
}

// ============================================================================

/// Structured observation extracted from Mega-Kernel epilogue telemetry buffer.
#[derive(Debug, Clone, Copy)]
pub struct MegaKernelObservation {
    pub layer_idx: usize,
    pub entropy: f32,
    pub residual_delta: f32,
    pub cosine_similarity: f32,
    pub dead_neuron_count: u32,
    pub is_attention_sink: bool,
    pub per_channel_scale: f32,
    pub row_l1_norm: f32,
    pub row_max: f32,
}

impl MegaKernelObservation {
    pub fn from_buffer(layer_idx: usize, buffer: &[u8]) -> Self {
        use gllm_kernels::compiler::graph::telemetry_offsets;

        let read_f32 = |offset: usize| -> f32 {
            if offset + 4 <= buffer.len() {
                f32::from_le_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                ])
            } else {
                0.0
            }
        };
        let read_u32 = |offset: usize| -> u32 {
            if offset + 4 <= buffer.len() {
                u32::from_le_bytes([
                    buffer[offset],
                    buffer[offset + 1],
                    buffer[offset + 2],
                    buffer[offset + 3],
                ])
            } else {
                0
            }
        };

        Self {
            layer_idx,
            entropy: read_f32(telemetry_offsets::SOFTMAX_SHARPNESS_OFFSET),
            residual_delta: read_f32(telemetry_offsets::RESIDUAL_DELTA_OFFSET),
            cosine_similarity: read_f32(telemetry_offsets::COSINE_SIMILARITY_OFFSET),
            dead_neuron_count: read_u32(telemetry_offsets::SILU_DEAD_NEURON_MASK_OFFSET),
            is_attention_sink: read_u32(telemetry_offsets::IS_ATTENTION_SINK_OFFSET) != 0,
            per_channel_scale: read_f32(telemetry_offsets::CHANNEL_SCALE_PTR_OFFSET),
            row_l1_norm: read_f32(telemetry_offsets::GEMM_ROW_NORM_L1_OFFSET),
            row_max: read_f32(telemetry_offsets::GEMM_ROW_MAX_OFFSET),
        }
    }

    pub fn dead_neuron_ratio(&self, hidden_size: usize) -> f32 {
        if hidden_size == 0 { return 0.0; }
        self.dead_neuron_count as f32 / hidden_size as f32
    }

    pub fn is_bypass_candidate(&self, delta_threshold: f32, cosine_threshold: f32) -> bool {
        self.residual_delta < delta_threshold && self.cosine_similarity > cosine_threshold
    }
}

// ============================================================================
// Diagnostic types for intermediate activation inspection
// ============================================================================

/// Diagnostic scratchpad data returned by `diagnostic_prefill_scratchpad`.
pub struct DiagnosticScratchpad {
    pub data: Vec<u8>,
    pub logits_offset: usize,
    pub vocab_size: usize,
    pub prompt_len: usize,
    pub hidden_size: usize,
}

impl DiagnosticScratchpad {
    /// Read f32 values from scratchpad at given byte offset and count.
    pub fn read_f32_at(&self, byte_offset: usize, count: usize) -> Vec<f32> {
        let end = byte_offset + count * 4;
        if end > self.data.len() {
            return vec![];
        }
        let mut out = vec![0.0f32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data[byte_offset..].as_ptr() as *const f32,
                out.as_mut_ptr(),
                count,
            );
        }
        out
    }

    /// Read embedding output from scratchpad (at offset 0).
    pub fn embedding(&self) -> Vec<f32> {
        let count = self.prompt_len * self.hidden_size;
        self.read_f32_at(0, count)
    }

    /// Read logits for the last prompt token.
    pub fn last_token_logits(&self) -> Vec<f32> {
        let row_bytes = self.vocab_size * 4;
        let off = self.logits_offset + (self.prompt_len - 1) * row_bytes;
        self.read_f32_at(off, self.vocab_size)
    }
}
