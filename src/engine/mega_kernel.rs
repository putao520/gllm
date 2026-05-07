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
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
struct MegaKernelCompiled {
    /// 全模型权重布局（embed → layer_0 → ... → layer_N → lm_head）
    weight_layout: gllm_kernels::compiler::MegaKernelWeightLayout,
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
    /// 实际需要的 scratchpad 大小（buffer_layout + intermediate tensors + RoPE cache）
    total_scratchpad_bytes: usize,
    /// Heterogeneous weight layout (for models like Gemma-4 E2B).
    hetero_layout: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroWeightLayout>,
}

/// Forward-only 编译产物 (encoder 路径: embedding / rerank / classify)。
///
/// 通过 `InferenceCompiler::compile_graph` 编译任意 CompilerGraph，
/// 推理时通过 `CompiledLayerFn` (10 参数通用 ABI) 单次 CALL 完成。
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
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
}

/// Mega-Kernel 执行器 (§9.1)
///
/// 唯一推理路径: 编译 → 单次 CALL。
/// 编译在模型加载时完成，推理时零 Rust 开销。
pub struct MegaKernelExecutor {
    /// Decoder mega-kernel 编译产物 (generate loop 路径)
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    mega_compiled: Option<MegaKernelCompiled>,
    /// Forward-only 编译产物 (encoder 路径: embedding / rerank / classify)
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    forward_compiled: Option<ForwardCompiled>,
    /// 模型配置
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    dtype: DType,
    /// EOS token ID — 从 ModelConfig 读取，传给 JIT 停止条件
    eos_token_id: u32,
}

impl MegaKernelExecutor {
    /// 从 ModelGeometry 编译 true mega-kernel。
    ///
    /// 单一函数管线: 整个 decoder 模型 (embed → N 层 → lm_head → argmax → generate loop)
    /// 编译为单一 JIT 机器码函数，通过一次 `compile_mega_kernel` 调用完成。
    ///
    /// 编译失败直接返回错误，不 fallback。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn compile_from_geometry(
        geometry: &crate::model_config::ModelGeometry,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        eos_token_id: u32,
        business_config: gllm_kernels::compiler::MegaKernelBusinessConfig,
        hetero_config: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig>,
    ) -> Result<Self, MegaKernelError> {
        eprintln!(
            "[mega] compiling: layers={} hidden={} heads={} kv_heads={} head_dim={} vocab={} eps={} rope_theta={} rope_partial={} hetero={}",
            geometry.num_layers, geometry.hidden_size, geometry.num_heads, geometry.num_kv_heads,
            geometry.head_dim, geometry.vocab_size, geometry.norm_eps, geometry.rope_theta,
            geometry.rope_partial_ratio, hetero_config.is_some(),
        );
        let embed_scale = business_config.embedding_scale;
        let rope_scaling = geometry.rope_scaling.as_ref().and_then(|cfg| {
            use crate::model_config::RopeScalingType;
            match cfg.scaling_type.as_ref()? {
                RopeScalingType::Yarn => Some(gllm_kernels::compiler::graph::RopeScaling::Yarn {
                    factor: cfg.factor.unwrap_or(1.0),
                    beta_fast: cfg.beta_fast.unwrap_or(32.0),
                    beta_slow: cfg.beta_slow.unwrap_or(1.0),
                    original_max_position: cfg.original_max_position_embeddings.unwrap_or(4096),
                }),
                RopeScalingType::Linear => Some(gllm_kernels::compiler::graph::RopeScaling::Linear {
                    factor: cfg.factor.unwrap_or(1.0),
                }),
                _ => None,
            }
        });
        let hetero_config_for_packing = hetero_config.clone();
        let config = gllm_kernels::compiler::ModelMegaConfig {
            num_layers: geometry.num_layers,
            hidden: geometry.hidden_size,
            num_heads: geometry.num_heads,
            num_kv_heads: geometry.num_kv_heads,
            head_dim: geometry.head_dim,
            intermediate: geometry.intermediate_size,
            vocab_size: geometry.vocab_size,
            rms_eps: geometry.norm_eps,
            rope_theta: geometry.rope_theta,
            rope_partial: geometry.rope_partial_ratio,
            dtype: DType::F32,
            max_seq_len: geometry.max_seq_len,
            num_eos_tokens: 1,
            rope_scaling,
            business_config,
            hetero: hetero_config,
        };
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let output = compiler.compile_mega_kernel(&config)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;
        compiler.print_resource_report();

        let exec_code = output.layer_code;
        let entry_fn = unsafe { exec_code.entry_point_as_mega_kernel() };

        let weight_blob = pack_mega_kernel_weights(
            &output.weight_layout,
            geometry.num_layers,
            geometry.hidden_size,
            weight_ptrs,
            weight_sizes,
            embed_scale,
            output.hetero_layout.as_ref(),
            hetero_config_for_packing.as_ref(),
        );

        let mega_compiled = MegaKernelCompiled {
            weight_layout: output.weight_layout,
            buffer_layout: output.buffer_layout,
            logits_scratch_offset: output.logits_scratch_offset,
            weight_blob,
            exec_code,
            entry_fn,
            rope_cache: output.rope_cache,
            total_scratchpad_bytes: output.total_scratchpad_bytes,
            hetero_layout: output.hetero_layout,
        };

        Ok(Self {
            mega_compiled: Some(mega_compiled),
            forward_compiled: None,
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden_size,
            vocab_size: geometry.vocab_size,
            dtype: geometry.dtype,
            eos_token_id,
        })
    }
    ///
    /// 使用 `ArchTemplate::to_compiler_graph()` 直接构建 CompilerGraph，
    /// 绕过 `graph_builders.rs` 中的手写 `decoder_model()` / `decoder_model_hetero()`。
    /// 新增模型只需写 YAML 模板 + 标量算子注册，零 Rust 代码修改。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn compile_from_template(
        template: &crate::arch::ArchTemplate,
        resolved_config: &crate::arch::ResolvedConfig,
        geometry: &crate::model_config::ModelGeometry,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        eos_token_id: u32,
        business_config: gllm_kernels::compiler::MegaKernelBusinessConfig,
        hetero_config: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig>,
    ) -> Result<Self, MegaKernelError> {
        eprintln!(
            "[mega] compiling from YAML template: layers={} hidden={} heads={} hetero={}",
            geometry.num_layers, geometry.hidden_size, geometry.num_heads,
            hetero_config.is_some(),
        );

        // Build CompilerGraph directly from YAML template
        let graph = template.to_compiler_graph(resolved_config, &business_config)
            .map_err(|e| MegaKernelError::Compilation(format!("template→CompilerGraph: {e}")))?;

        let rope_scaling = geometry.rope_scaling.as_ref().and_then(|cfg| {
            use crate::model_config::RopeScalingType;
            match cfg.scaling_type.as_ref()? {
                RopeScalingType::Yarn => Some(gllm_kernels::compiler::graph::RopeScaling::Yarn {
                    factor: cfg.factor.unwrap_or(1.0),
                    beta_fast: cfg.beta_fast.unwrap_or(32.0),
                    beta_slow: cfg.beta_slow.unwrap_or(1.0),
                    original_max_position: cfg.original_max_position_embeddings.unwrap_or(4096),
                }),
                RopeScalingType::Linear => Some(gllm_kernels::compiler::graph::RopeScaling::Linear {
                    factor: cfg.factor.unwrap_or(1.0),
                }),
                _ => None,
            }
        });

        let config = gllm_kernels::compiler::ModelMegaConfig {
            num_layers: geometry.num_layers,
            hidden: geometry.hidden_size,
            num_heads: geometry.num_heads,
            num_kv_heads: geometry.num_kv_heads,
            head_dim: geometry.head_dim,
            intermediate: geometry.intermediate_size,
            vocab_size: geometry.vocab_size,
            rms_eps: geometry.norm_eps,
            rope_theta: geometry.rope_theta,
            rope_partial: geometry.rope_partial_ratio,
            dtype: DType::F32,
            max_seq_len: geometry.max_seq_len,
            num_eos_tokens: 1,
            rope_scaling,
            business_config,
            hetero: hetero_config.clone(),
        };

        // Compute hetero layout if applicable
        let hetero_layout = hetero_config.as_ref().map(|hc| {
            gllm_kernels::compiler::mega_kernel_abi::HeteroWeightLayout::from_config(&config, hc)
        });

        let embed_scale = config.business_config.embedding_scale;
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let output = compiler.compile_mega_kernel_from_graph(graph, &config, hetero_layout)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        let exec_code = output.layer_code;
        let entry_fn = unsafe { exec_code.entry_point_as_mega_kernel() };

        let weight_blob = pack_mega_kernel_weights(
            &output.weight_layout,
            geometry.num_layers,
            geometry.hidden_size,
            weight_ptrs,
            weight_sizes,
            embed_scale,
            output.hetero_layout.as_ref(),
            hetero_config.as_ref(),
        );

        let mega_compiled = MegaKernelCompiled {
            weight_layout: output.weight_layout,
            buffer_layout: output.buffer_layout,
            logits_scratch_offset: output.logits_scratch_offset,
            weight_blob,
            exec_code,
            entry_fn,
            rope_cache: output.rope_cache,
            total_scratchpad_bytes: output.total_scratchpad_bytes,
            hetero_layout: output.hetero_layout,
        };

        Ok(Self {
            mega_compiled: Some(mega_compiled),
            forward_compiled: None,
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden_size,
            vocab_size: geometry.vocab_size,
            dtype: geometry.dtype,
            eos_token_id,
        })
    }

    /// 从 YAML 模板编译 forward-only encoder mega-kernel。
    ///
    /// 使用 `InferenceCompiler::compile_graph`（通用编译器）编译任意 CompilerGraph。
    /// Encoder 模型不需要 generate loop / argmax / sampling — 仅单次 forward pass。
    ///
    /// 编译失败直接返回错误，不 fallback。
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    pub fn compile_forward(
        template: &crate::arch::ArchTemplate,
        resolved_config: &crate::arch::ResolvedConfig,
        geometry: &crate::model_config::ModelGeometry,
        weight_ptrs: &std::collections::HashMap<String, *const u8>,
        weight_sizes: &std::collections::HashMap<String, usize>,
        business_config: &gllm_kernels::compiler::MegaKernelBusinessConfig,
    ) -> Result<Self, MegaKernelError> {
        let graph = template.to_compiler_graph(resolved_config, business_config)
            .map_err(|e| MegaKernelError::Compilation(format!("template→CompilerGraph: {e}")))?;

        let weight_layout = graph.weight_layout();

        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let exec_code = compiler.compile_graph(&graph)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        let entry_fn = unsafe { exec_code.entry_point() };
        let total_scratchpad_bytes = exec_code.scratchpad_bytes;

        let weight_blob = pack_weights_from_graph(
            &graph,
            &weight_layout,
            weight_ptrs,
            weight_sizes,
        );

        let forward_compiled = ForwardCompiled {
            exec_code,
            entry_fn,
            weight_blob,
            weight_layout,
            total_scratchpad_bytes,
        };

        Ok(Self {
            mega_compiled: None,
            forward_compiled: Some(forward_compiled),
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden_size,
            vocab_size: geometry.vocab_size,
            dtype: geometry.dtype,
            eos_token_id: 0,
        })
    }

    /// 执行 forward-only 推理 (embedding / rerank / classify)。
    ///
    /// ARCH-RUST-IS-CODEGEN: 一次 CALL 完成。
    /// CompiledLayerFn ABI (10 参数):
    ///   input_ptr → token IDs, weight_ptr → weight blob,
    ///   output → 输出缓冲, scratchpad → 临时空间。
    #[cfg(target_arch = "x86_64")]
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
        let mut output = vec![0.0f32; output_elems];

        unsafe {
            (fw.entry_fn)(
                input_ids.as_ptr() as *const u8,
                fw.weight_blob.as_ptr(),
                std::ptr::null_mut(),       // kv_cache: null
                std::ptr::null(),           // positions: null
                std::ptr::null(),           // seq_lens: null
                0,                          // batch_size: unused
                seq_len,                    // seq_len
                output.as_mut_ptr() as *mut u8,
                scratchpad.as_mut_ptr(),
                std::ptr::null_mut(),       // telemetry: null
            );
        }

        Ok(output)
    }

    /// Returns total scratchpad bytes needed for execution.
    pub fn total_scratchpad_bytes(&self) -> usize {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        {
            if let Some(ref mega) = self.mega_compiled {
                return mega.total_scratchpad_bytes;
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
    #[cfg(target_arch = "x86_64")]
    pub fn generate_single_sequence(
        &self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        hook_ctx_ptr: *const u8,
        callback_table_ptr: *const u8,
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
        let mut scratchpad = vec![0u8; mega.total_scratchpad_bytes];

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
                gllm_kernels::compiler::fill_cos_sin_table(
                    rope_slice,
                    &positions[..max_total],
                    rc.head_dim,
                    rc.theta,
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
                    gllm_kernels::compiler::fill_cos_sin_table(
                        sec_slice,
                        &positions[..max_total],
                        sec.head_dim,
                        sec.theta,
                        sec.rope_scaling.clone(),
                    );
                }
            }
        }

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
                0,  // session_position: new session
                std::ptr::null(),  // fused_hidden_ptr: no multimodal
                0,  // num_mm_tokens: no multimodal tokens
                callback_table_ptr, // callback_table_ptr: C-style fn_ptr array (NULL = no callbacks)
            )
        };

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
                        eprintln!(
                            "[DIAG] row{}: argmax={} max={:.4} nonzero={}/{} first8={:?}",
                            row_idx, max_idx, max_val, nonzero, vocab,
                            &row_data[..8.min(row_data.len())]
                        );
                    }
                }
            }
            eprintln!("[DIAG] output_tokens={:?}", &output_tokens[..generated_count.min(10) as usize]);
        }

        log::debug!(
            "[mega] prompt_len={} max_new_tokens={} generated_count={} eos={} output_first={}",
            prompt_len, max_new_tokens, generated_count, self.eos_token_id,
            output_tokens.first().copied().unwrap_or(0),
        );
        eprintln!(
            "[mega] prompt_len={} generated_count={} output_tokens={:?}",
            prompt_len, generated_count, &output_tokens[..generated_count.min(10) as usize]
        );

        let actual_count = if generated_count == 0 && max_new_tokens > 0 && output_tokens[0] != 0 {
            1
        } else {
            generated_count
        };

        Ok(output_tokens[..actual_count].to_vec())
    }

    /// Execute mega-kernel in EncodeToLayer mode (output_mode_selector = 3).
    ///
    /// Runs a single forward pass through all layers, then extracts the hidden
    /// state from the activation buffer. The mega-kernel skips the generate loop
    /// and argmax, jumping directly to the encode path.
    ///
    /// Returns the hidden state as a Vec<f32> of shape [seq_len * hidden_size].
    #[cfg(target_arch = "x86_64")]
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
        let mut scratchpad = vec![0u8; mega.total_scratchpad_bytes];

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
                gllm_kernels::compiler::fill_cos_sin_table(
                    rope_slice,
                    &positions,
                    rc.head_dim,
                    rc.theta,
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

    /// Returns true if this executor has a decoder mega-kernel (compile_from_template/geometry path).
    pub fn has_mega_compiled(&self) -> bool {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        { self.mega_compiled.is_some() }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda")))]
        { false }
    }

    /// Returns the expected output element count for embedding inference.
    ///
    /// - Decoder path (mega_compiled): seq_len × hidden_size (raw hidden states)
    /// - Encoder path (forward_compiled): hidden_size (after MeanPool in JIT graph)
    pub fn output_elems_for_embed(&self, seq_len: usize, hidden_size: usize) -> usize {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        {
            if self.mega_compiled.is_some() {
                seq_len * hidden_size
            } else {
                // Encoder: graph includes MeanPool → output is [hidden_size]
                hidden_size
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda")))]
        { seq_len * hidden_size }
    }

    /// Returns true if this executor has a forward-only compiled layer (encoder path).
    pub fn has_forward_compiled(&self) -> bool {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
        { self.forward_compiled.is_some() }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda")))]
        { false }
    }
}

// ============================================================================
// Weight Blob Packing
// ============================================================================

/// 将所有模型权重打包到单一连续 blob。
///
/// 按照 MegaKernelWeightLayout 定义的顺序:
/// embed_weight → layer_0_weights → layer_1_weights → ... → lm_head_weight
///
/// ARCH-COMPUTE-F32: 所有权重在 upload_weights 阶段已被转为 F32。
/// weight_ptrs 指向的数据全部是 F32，直接 memcpy。

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn copy_weight(
    blob: &mut [u8],
    offset: usize,
    ptr: *const u8,
    src_bytes: usize,
    slot_size: usize,
) -> bool {
    if ptr.is_null() || src_bytes == 0 {
        return true;
    }
    let copy_size = src_bytes.min(slot_size).min(blob.len().saturating_sub(offset));
    if copy_size == 0 || offset >= blob.len() {
        return false;
    }
    let src = unsafe { std::slice::from_raw_parts(ptr, copy_size) };
    blob[offset..offset + copy_size].copy_from_slice(src);
    true
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn pack_mega_kernel_weights(
    layout: &gllm_kernels::compiler::MegaKernelWeightLayout,
    num_layers: usize,
    hidden_size: usize,
    weight_ptrs: &std::collections::HashMap<String, *const u8>,
    weight_sizes: &std::collections::HashMap<String, usize>,
    embedding_scale: Option<f32>,
    hetero_layout: Option<&gllm_kernels::compiler::mega_kernel_abi::HeteroWeightLayout>,
    hetero_config: Option<&gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig>,
    // full_attention_layer_indices: original indices of full attention layers
) -> Vec<u8> {
    let mut blob = vec![0u8; layout.total_bytes];

    // Embedding weight (with optional scaling for Gemma-style models)
    let embed_key = weight_ptrs.keys().find(|k| k.ends_with("embed_tokens.weight"))
        .map(|k| k.as_str());
    if let Some(key) = embed_key {
        if let Some(&ptr) = weight_ptrs.get(key) {
            let size = *weight_sizes.get(key).unwrap_or(&0);
            if let Some(scale) = embedding_scale {
                // Scale embedding weights in-place: each f32 element *= scale.
                // Gemma models: embed *= sqrt(hidden_size).
                let copy_size = size.min(layout.embed_bytes).min(blob.len().saturating_sub(layout.embed_offset));
                let num_elems = copy_size / 4;
                if num_elems > 0 {
                    let src = unsafe { std::slice::from_raw_parts(ptr as *const f32, num_elems) };
                    for (i, &val) in src.iter().enumerate() {
                        let off = layout.embed_offset + i * 4;
                        if off + 4 <= blob.len() {
                            let scaled = val * scale;
                            blob[off..off + 4].copy_from_slice(&scaled.to_le_bytes());
                        }
                    }
                    log::info!("[mega] embedding scaled by {:.2} ({} elements)", scale, num_elems);
                }
            } else {
                copy_weight(&mut blob, layout.embed_offset, ptr, size, layout.embed_bytes);
            }
        }
    }

    // Per-layer weights (already in canonical [in, out] layout from loader normalization)
    if let (Some(hl), Some(hc)) = (hetero_layout, hetero_config) {
        // ── Heterogeneous packing: 4-type layout [sliding/full × small/large] × 7 segments ──
        let full_set: std::collections::HashSet<usize> = hc.full_layer_indices.iter().copied().collect();
        let mut original_layers: Vec<usize> = (0..num_layers).collect();
        // Reorder into segment layout: sliding_per_segment sliding, then 1 full, repeat
        let mut reordered = Vec::with_capacity(num_layers);
        let mut sliding_buf: Vec<usize> = Vec::with_capacity(hc.sliding_per_segment);
        for &idx in &original_layers {
            if full_set.contains(&idx) {
                // Flush sliding buffer + add full layer
                reordered.extend(sliding_buf.drain(..));
                reordered.push(idx);
            } else {
                sliding_buf.push(idx);
                if sliding_buf.len() == hc.sliding_per_segment {
                    reordered.extend(sliding_buf.drain(..));
                }
            }
        }
        reordered.extend(sliding_buf.drain(..));

        // Compute cumulative segment base offsets (variable per segment type).
        let mut seg_base_offsets = Vec::with_capacity(hc.num_segments);
        let mut offset = hl.layer_0_offset;
        for seg in 0..hc.num_segments {
            seg_base_offsets.push(offset);
            if seg < hl.large_ffn_start_segment {
                offset += hl.small_segment_stride;
            } else {
                offset += hl.large_segment_stride;
            }
        }

        for (seg_idx, &orig_idx) in reordered.iter().enumerate() {
            let is_full = full_set.contains(&orig_idx);
            let segment_idx = seg_idx / (hc.sliding_per_segment + 1);
            let pos_in_segment = seg_idx % (hc.sliding_per_segment + 1);
            let is_small_seg = segment_idx < hl.large_ffn_start_segment;

            let (sliding_stride, full_stride, pl_sliding, pl_full) = if is_small_seg {
                (hl.sliding_small_stride, hl.full_small_stride, &hl.sliding_small_per_layer, &hl.full_small_per_layer)
            } else {
                (hl.sliding_large_stride, hl.full_large_stride, &hl.sliding_large_per_layer, &hl.full_large_per_layer)
            };

            let (pl, layer_base) = if is_full {
                (pl_full, seg_base_offsets[segment_idx] + hc.sliding_per_segment * sliding_stride)
            } else {
                (pl_sliding, seg_base_offsets[segment_idx] + pos_in_segment * sliding_stride)
            };

            pack_single_layer(
                &mut blob, layer_base, orig_idx, hidden_size, pl, weight_ptrs, weight_sizes,
            );
        }
    } else {
        // ── Homogeneous packing: uniform stride ──
        let per_layer = &layout.per_layer;
        for layer_idx in 0..num_layers {
            let layer_base = layout.layer_base_offset(layer_idx);
            pack_single_layer(
                &mut blob, layer_base, layer_idx, hidden_size, per_layer, weight_ptrs, weight_sizes,
            );
        }
    }

    // Final layer norm weight
    let final_norm_key = weight_ptrs.keys().find(|k| k.ends_with("model.norm.weight"))
        .map(|k| k.as_str());
    if let Some(key) = final_norm_key {
        if let Some(&ptr) = weight_ptrs.get(key) {
            let size = *weight_sizes.get(key).unwrap_or(&0);
            copy_weight(&mut blob, layout.final_norm_offset, ptr, size, layout.final_norm_bytes);
        }
    }

    // lm_head weight
    let lm_key = weight_ptrs.keys().find(|k| k.ends_with("lm_head.weight"))
        .map(|k| k.as_str());
    if let Some(key) = lm_key {
        // Separate lm_head weight (non-tied models)
        if let Some(&ptr) = weight_ptrs.get(key) {
            let size = *weight_sizes.get(key).unwrap_or(&0);
            copy_weight(&mut blob, layout.lm_head_offset, ptr, size, layout.lm_head_bytes);
        }
    } else {
        // Tied embeddings: no separate lm_head.weight — transpose embed_tokens
        // from [vocab, hidden] (Gather layout) to [hidden, vocab] (GEMM canonical).
        let embed_key = weight_ptrs.keys().find(|k| k.ends_with("embed_tokens.weight"))
            .map(|k| k.as_str());
        if let Some(key) = embed_key {
            if let Some(&ptr) = weight_ptrs.get(key) {
                let size = *weight_sizes.get(key).unwrap_or(&0);
                let embed_elems = size / 4;
                let vocab = if hidden_size > 0 { embed_elems / hidden_size } else { 0 };
                if vocab > 0 && embed_elems == vocab * hidden_size {
                    let src = unsafe { std::slice::from_raw_parts(ptr as *const f32, embed_elems) };
                    for v in 0..vocab {
                        for h in 0..hidden_size {
                            let dst_off = layout.lm_head_offset + (h * vocab + v) * 4;
                            if dst_off + 4 <= blob.len() {
                                let bytes = src[v * hidden_size + h].to_le_bytes();
                                blob[dst_off..dst_off + 4].copy_from_slice(&bytes);
                            }
                        }
                    }
                    log::info!(
                        "[mega] tied embeddings: transposed embed_tokens [{}x{}] → lm_head [{}x{}]",
                        vocab, hidden_size, hidden_size, vocab,
                    );
                }
            }
        }
    }

    blob
}

/// Pack a single layer's weights into the blob at the given base offset.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn pack_single_layer(
    blob: &mut [u8],
    layer_base: usize,
    layer_idx: usize,
    hidden_size: usize,
    per_layer: &gllm_kernels::compiler::mega_kernel_abi::PerLayerWeightLayout,
    weight_ptrs: &std::collections::HashMap<String, *const u8>,
    weight_sizes: &std::collections::HashMap<String, usize>,
) {
    let q_row_bytes = per_layer.w_q_bytes / hidden_size;
    let k_row_bytes = per_layer.w_k_bytes / hidden_size;
    let v_row_bytes = per_layer.w_v_bytes / hidden_size;
    let gate_row_bytes = per_layer.w_gate_bytes / hidden_size;
    let up_row_bytes = per_layer.w_up_bytes / hidden_size;

    // Simple weights: norm, o_proj, ffn_norm, down_proj
    let simple: [(&str, usize, usize); 4] = [
        ("model.layers.{L}.input_layernorm.weight", per_layer.attn_norm_offset, per_layer.attn_norm_bytes),
        ("model.layers.{L}.self_attn.o_proj.weight", per_layer.w_o_offset, per_layer.w_o_bytes),
        ("model.layers.{L}.post_attention_layernorm.weight", per_layer.ffn_norm_offset, per_layer.ffn_norm_bytes),
        ("model.layers.{L}.mlp.down_proj.weight", per_layer.w_down_offset, per_layer.w_down_bytes),
    ];
    for (tmpl, off, slot) in &simple {
        let name = tmpl.replace("{L}", &layer_idx.to_string());
        if let Some(&ptr) = weight_ptrs.get(&name) {
            let size = *weight_sizes.get(&name).unwrap_or(slot);
            copy_weight(blob, layer_base + off, ptr, size, *slot);
        }
    }

    // QKV: separate or fused
    let q_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
    let qkv_name = format!("model.layers.{}.self_attn.qkv_proj.weight", layer_idx);
    if weight_ptrs.contains_key(&q_name) {
        for (name, off, rb) in [
            (&q_name, per_layer.w_q_offset, q_row_bytes),
            (&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx), per_layer.w_k_offset, k_row_bytes),
            (&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx), per_layer.w_v_offset, v_row_bytes),
        ] {
            if let Some(&p) = weight_ptrs.get(name) {
                let s = *weight_sizes.get(name).unwrap_or(&0);
                copy_weight(blob, layer_base + off, p, s, rb * hidden_size);
            }
        }
    } else if let Some(&ptr) = weight_ptrs.get(&qkv_name) {
        let qkv_row_stride = q_row_bytes + k_row_bytes + v_row_bytes;
        for r in 0..hidden_size {
            let src_row = unsafe { ptr.add(r * qkv_row_stride) };
            let q_dst = layer_base + per_layer.w_q_offset + r * q_row_bytes;
            let k_dst = layer_base + per_layer.w_k_offset + r * k_row_bytes;
            let v_dst = layer_base + per_layer.w_v_offset + r * v_row_bytes;
            if q_dst + q_row_bytes <= blob.len() {
                let src = unsafe { std::slice::from_raw_parts(src_row, q_row_bytes) };
                blob[q_dst..q_dst + q_row_bytes].copy_from_slice(src);
            }
            if k_dst + k_row_bytes <= blob.len() {
                let src = unsafe { std::slice::from_raw_parts(src_row.add(q_row_bytes), k_row_bytes) };
                blob[k_dst..k_dst + k_row_bytes].copy_from_slice(src);
            }
            if v_dst + v_row_bytes <= blob.len() {
                let src = unsafe { std::slice::from_raw_parts(src_row.add(q_row_bytes + k_row_bytes), v_row_bytes) };
                blob[v_dst..v_dst + v_row_bytes].copy_from_slice(src);
            }
        }
    }

    // Head norm weights
    for (tmpl, off, slot) in [
        ("model.layers.{L}.self_attn.q_norm.weight", per_layer.w_q_norm_offset, per_layer.w_q_norm_bytes),
        ("model.layers.{L}.self_attn.k_norm.weight", per_layer.w_k_norm_offset, per_layer.w_k_norm_bytes),
    ] {
        let name = tmpl.replace("{L}", &layer_idx.to_string());
        if let Some(&ptr) = weight_ptrs.get(&name) {
            let size = *weight_sizes.get(&name).unwrap_or(&slot);
            copy_weight(blob, layer_base + off, ptr, size, slot);
        }
    }

    // FFN gate/up: separate or fused
    let gate_name = format!("model.layers.{}.mlp.gate_proj.weight", layer_idx);
    let gate_up_name = format!("model.layers.{}.mlp.gate_up_proj.weight", layer_idx);
    if weight_ptrs.contains_key(&gate_name) {
        for (name, off, rb) in [
            (&gate_name, per_layer.w_gate_offset, gate_row_bytes),
            (&format!("model.layers.{}.mlp.up_proj.weight", layer_idx), per_layer.w_up_offset, up_row_bytes),
        ] {
            if let Some(&p) = weight_ptrs.get(name) {
                let s = *weight_sizes.get(name).unwrap_or(&0);
                copy_weight(blob, layer_base + off, p, s, rb * hidden_size);
            }
        }
    } else if let Some(&ptr) = weight_ptrs.get(&gate_up_name) {
        let row_stride = gate_row_bytes + up_row_bytes;
        for r in 0..hidden_size {
            let src_row = unsafe { ptr.add(r * row_stride) };
            let g_dst = layer_base + per_layer.w_gate_offset + r * gate_row_bytes;
            let u_dst = layer_base + per_layer.w_up_offset + r * up_row_bytes;
            if g_dst + gate_row_bytes <= blob.len() {
                let src = unsafe { std::slice::from_raw_parts(src_row, gate_row_bytes) };
                blob[g_dst..g_dst + gate_row_bytes].copy_from_slice(src);
            }
            if u_dst + up_row_bytes <= blob.len() {
                let src = unsafe { std::slice::from_raw_parts(src_row.add(gate_row_bytes), up_row_bytes) };
                blob[u_dst..u_dst + up_row_bytes].copy_from_slice(src);
            }
        }
    }
}
// ============================================================================
// Generic Weight Packing (graph.weight_layout() driven)
// ============================================================================

/// 通用权重打包：按 CompilerGraph 的 weight_layout() 将权重排列到连续 blob。
///
/// 遍历 weight_layout 中的每个 (TensorId, offset) 对，
/// 从 weight_ptrs 中查找对应的实际权重并 memcpy 到 blob 中。
///
/// 无硬编码权重名 — 完全由图结构驱动。
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
fn pack_weights_from_graph(
    graph: &gllm_kernels::compiler::CompilerGraph,
    layout: &gllm_kernels::compiler::graph::WeightLayout,
    weight_ptrs: &std::collections::HashMap<String, *const u8>,
    weight_sizes: &std::collections::HashMap<String, usize>,
) -> Vec<u8> {
    let mut blob = vec![0u8; layout.total_bytes];
    let mut packed_count = 0usize;
    let mut missing_count = 0usize;

    for &(tid, offset) in &layout.offsets {
        let tensor = match graph.tensors.get(tid.0 as usize) {
            Some(t) => t,
            None => continue,
        };
        let name = &tensor.name;

        // Resolve weight name via alias system (weight_names.rs).
        // YAML templates define canonical names (e.g., "roberta.encoder.layer.0.attention.self.query.weight");
        // safetensors files may store them under a different prefix (e.g., "encoder.layer.0.attention.self.query.weight").
        // `all_encoder_weight_aliases` generates all known prefix variants for matching.
        let resolved_name = match weight_ptrs.get(name) {
            Some(_) => name.clone(),
            None => {
                let aliases = crate::weight_names::all_encoder_weight_aliases(name);
                let found = aliases.iter().find(|a| weight_ptrs.contains_key(a.as_str()));
                match found {
                    Some(n) => n.clone(),
                    None => {
                        missing_count += 1;
                        if missing_count <= 5 {
                            eprintln!("[pack_weights] MISSING: {:?} name={:?}", tid, name);
                        }
                        continue;
                    }
                }
            }
        };

        let ptr = match weight_ptrs.get(&resolved_name) {
            Some(&p) if !p.is_null() => p,
            _ => continue,
        };
        let size = *weight_sizes.get(&resolved_name).unwrap_or(&0);
        if size == 0 {
            continue;
        }

        let copy_size = size.min(blob.len().saturating_sub(offset));
        if copy_size == 0 || offset >= blob.len() {
            continue;
        }
        let src = unsafe { std::slice::from_raw_parts(ptr, copy_size) };
        blob[offset..offset + copy_size].copy_from_slice(src);
        packed_count += 1;
    }
    if missing_count > 0 || std::env::var("GLLM_DEBUG_BUFFER_ALLOC").is_ok() {
        eprintln!("[pack_weights] packed={} missing={}/{} total_bytes={}",
            packed_count, missing_count, layout.offsets.len(), layout.total_bytes);
    }

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
