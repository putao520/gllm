//! Mega-Kernel 执行器 (SPEC §9.1)
//!
//! ARCH-RUST-IS-CODEGEN 铁律: 推理时 Rust 只做一次 CALL。
//! 整个 decoder 模型（embedding → N 层 → lm_head → sampling → generate loop）
//! 编译为单一 JIT 机器码，推理时通过 MegaKernelFn 单次 CALL 完成。
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
}

/// Mega-Kernel 执行器 (§9.1)
///
/// 唯一推理路径: 编译 → 单次 CALL。
/// 编译在模型加载时完成，推理时零 Rust 开销。
pub struct MegaKernelExecutor {
    /// True mega-kernel 编译产物 — 单次 CALL 路径
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", feature = "cuda"))]
    mega_compiled: MegaKernelCompiled,
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
    ) -> Result<Self, MegaKernelError> {
        log::info!(
            "[mega] compiling: layers={} hidden={} heads={} kv_heads={} head_dim={} vocab={} eps={} rope_theta={} rope_partial={}",
            geometry.num_layers, geometry.hidden_size, geometry.num_heads, geometry.num_kv_heads,
            geometry.head_dim, geometry.vocab_size, geometry.norm_eps, geometry.rope_theta,
            geometry.rope_partial_ratio,
        );
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
            max_seq_len: 128, // Buffer allocation upper bound; runtime seq_len may be smaller
            num_eos_tokens: 1,
            rope_scaling: None,
            business_config: gllm_kernels::compiler::MegaKernelBusinessConfig::default(),
        };
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let output = compiler.compile_mega_kernel(&config)
            .map_err(|e| MegaKernelError::Compilation(e.to_string()))?;

        let exec_code = output.layer_code;
        let entry_fn = unsafe { exec_code.entry_point_as_mega_kernel() };

        let weight_blob = pack_mega_kernel_weights(
            &output.weight_layout,
            geometry.num_layers,
            weight_ptrs,
            weight_sizes,
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
        };

        Ok(Self {
            mega_compiled,
            num_layers: geometry.num_layers,
            hidden_size: geometry.hidden_size,
            vocab_size: geometry.vocab_size,
            dtype: geometry.dtype,
            eos_token_id,
        })
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
    ) -> Result<Vec<u32>, MegaKernelError> {
        let mega = &self.mega_compiled;
        let prompt_len = prompt_tokens.len();
        let max_total = prompt_len + max_new_tokens;

        let mut input_ids = vec![0u32; max_total];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..max_total as u32).collect();
        let mut output_tokens = vec![0u32; max_new_tokens];
        let mut scratchpad = vec![0u8; mega.total_scratchpad_bytes];

        // Pre-fill RoPE cos/sin table for all positions [0..max_total).
        if let Some(ref rc) = mega.rope_cache {
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
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        log::debug!(
            "[mega] prompt_len={} max_new_tokens={} generated_count={} eos={}",
            prompt_len, max_new_tokens, generated_count, self.eos_token_id,
        );

        // JIT may return 0 when gen_counter hasn't been incremented yet
        // (EOS on first iteration). In that case check if output_tokens[0]
        // was written (non-zero) to detect the off-by-one.
        let actual_count = if generated_count == 0 && max_new_tokens > 0 && output_tokens[0] != 0 {
            1
        } else {
            generated_count
        };

        Ok(output_tokens[..actual_count].to_vec())
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
fn pack_mega_kernel_weights(
    layout: &gllm_kernels::compiler::MegaKernelWeightLayout,
    num_layers: usize,
    weight_ptrs: &std::collections::HashMap<String, *const u8>,
    weight_sizes: &std::collections::HashMap<String, usize>,
) -> Vec<u8> {
    let mut blob = vec![0u8; layout.total_bytes];

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

    // Embedding weight
    let embed_key = weight_ptrs.keys().find(|k| k.ends_with("embed_tokens.weight"))
        .map(|k| k.as_str());
    if let Some(key) = embed_key {
        if let Some(&ptr) = weight_ptrs.get(key) {
            let size = *weight_sizes.get(key).unwrap_or(&0);
            copy_weight(&mut blob, layout.embed_offset, ptr, size, layout.embed_bytes);
        }
    }

    // Per-layer weights (already in canonical [in, out] layout from loader normalization)
    let per_layer = &layout.per_layer;
    let weight_names: [(&str, usize, usize); 9] = [
        ("model.layers.{L}.input_layernorm.weight", per_layer.attn_norm_offset, per_layer.attn_norm_bytes),
        ("model.layers.{L}.self_attn.q_proj.weight", per_layer.w_q_offset, per_layer.w_q_bytes),
        ("model.layers.{L}.self_attn.k_proj.weight", per_layer.w_k_offset, per_layer.w_k_bytes),
        ("model.layers.{L}.self_attn.v_proj.weight", per_layer.w_v_offset, per_layer.w_v_bytes),
        ("model.layers.{L}.self_attn.o_proj.weight", per_layer.w_o_offset, per_layer.w_o_bytes),
        ("model.layers.{L}.post_attention_layernorm.weight", per_layer.ffn_norm_offset, per_layer.ffn_norm_bytes),
        ("model.layers.{L}.mlp.gate_proj.weight", per_layer.w_gate_offset, per_layer.w_gate_bytes),
        ("model.layers.{L}.mlp.up_proj.weight", per_layer.w_up_offset, per_layer.w_up_bytes),
        ("model.layers.{L}.mlp.down_proj.weight", per_layer.w_down_offset, per_layer.w_down_bytes),
    ];

    for layer_idx in 0..num_layers {
        let layer_base = layout.layer_base_offset(layer_idx);
        for (name_template, rel_offset, slot_size) in &weight_names {
            let name = name_template.replace("{L}", &layer_idx.to_string());
            if let Some(&ptr) = weight_ptrs.get(&name) {
                let size = *weight_sizes.get(&name).unwrap_or(slot_size);
                copy_weight(&mut blob, layer_base + rel_offset, ptr, size, *slot_size);
            }
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

    // lm_head weight (tied models: executor already transposes embed_tokens to [hidden, vocab])
    let lm_key = weight_ptrs.keys().find(|k| k.ends_with("lm_head.weight"))
        .map(|k| k.as_str());
    if let Some(key) = lm_key {
        if let Some(&ptr) = weight_ptrs.get(key) {
            let size = *weight_sizes.get(key).unwrap_or(&0);
            copy_weight(&mut blob, layout.lm_head_offset, ptr, size, layout.lm_head_bytes);
        }
    }

    blob
}

// ============================================================================
// MegaKernelObservation — Type-Safe Telemetry Interface (SPEC §9.5)
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
