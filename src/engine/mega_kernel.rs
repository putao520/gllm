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
    /// Heterogeneous weight layout (for models like Gemma-4 E2B).
    hetero_layout: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroWeightLayout>,
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
        business_config: gllm_kernels::compiler::MegaKernelBusinessConfig,
        hetero_config: Option<gllm_kernels::compiler::mega_kernel_abi::HeteroLayerConfig>,
    ) -> Result<Self, MegaKernelError> {
        log::info!(
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
            max_seq_len: 128, // Buffer allocation upper bound; runtime seq_len may be smaller
            num_eos_tokens: 1,
            rope_scaling,
            business_config,
            hetero: hetero_config,
        };
        let mut compiler = gllm_kernels::compiler::InferenceCompiler::new();
        let output = compiler.compile_mega_kernel(&config)
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
            if let Some((sec_head_dim, sec_offset)) = rc.secondary_cache {
                let sec_elems = max_total * sec_head_dim;
                let sec_bytes = sec_elems * 4;
                if sec_offset + sec_bytes <= scratchpad.len() {
                    let sec_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            scratchpad[sec_offset..].as_mut_ptr() as *mut f32,
                            sec_elems,
                        )
                    };
                    gllm_kernels::compiler::fill_cos_sin_table(
                        sec_slice,
                        &positions[..max_total],
                        sec_head_dim,
                        rc.theta,
                        rc.rope_scaling.clone(),
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
                0,
                std::ptr::null(),
                std::ptr::null_mut(),
            )
        };

        log::debug!(
            "[mega] prompt_len={} max_new_tokens={} generated_count={} eos={} output_first={}",
            prompt_len, max_new_tokens, generated_count, self.eos_token_id,
            output_tokens.first().copied().unwrap_or(0),
        );
        eprintln!(
            "[mega] prompt_len={} generated_count={} output_tokens={:?}",
            prompt_len, generated_count, &output_tokens[..generated_count.min(10) as usize]
        );

        // DIAG: comprehensive post-JIT diagnostics
        {
            let logits_off = mega.logits_scratch_offset;
            // 1. Logits: dump first 10 + find max
            let logits_count = self.vocab_size.min(scratchpad.len().saturating_sub(logits_off) / 4);
            let logits_ptr = unsafe { scratchpad.as_ptr().add(logits_off) as *const f32 };
            // The last row of logits: row = decode_seq_len - 1 = (prompt_len + generated_count - 1)
            let last_row_idx = prompt_len + generated_count as usize;
            let row_byte_off = (last_row_idx.saturating_sub(1)) * self.vocab_size * 4;
            let last_row_ptr = if row_byte_off + self.vocab_size * 4 <= logits_count * 4 {
                unsafe { logits_ptr.add(row_byte_off / 4) }
            } else {
                logits_ptr
            };
            let logits_first10: Vec<f32> = unsafe {
                std::slice::from_raw_parts(last_row_ptr, 10.min(self.vocab_size))
            }.to_vec();
            let mut max_logit = f32::NEG_INFINITY;
            let mut max_idx = 0usize;
            for i in 0..self.vocab_size.min(logits_count) {
                let v = unsafe { *last_row_ptr.add(i) };
                if v > max_logit { max_logit = v; max_idx = i; }
            }
            log::debug!(
                "[DIAG] logits_off={} logits_count={} last_row_idx={} row_byte_off={}",
                logits_off, logits_count, last_row_idx, row_byte_off,
            );
            log::debug!(
                "[DIAG] logits_first10={:?} max_logit={} at idx={}",
                logits_first10, max_logit, max_idx,
            );

            // 2. Check if all logits are the same
            let first_logit = unsafe { *last_row_ptr };
            let mut all_same = true;
            for i in 1..self.vocab_size.min(logits_count).min(1000) {
                let v = unsafe { *last_row_ptr.add(i) };
                if v != first_logit { all_same = false; break; }
            }
            log::debug!("[DIAG] all_logits_same={} first_logit={}", all_same, first_logit);
            eprintln!(
                "[DIAG] logits_first10={:?} max_logit={:.4} at idx={} all_same={}",
                logits_first10, max_logit, max_idx, all_same,
            );

            // 3. Weight blob verification
            let blob = &mega.weight_blob;
            // Embedding: first 8 f32 values
            let embed_f32: Vec<f32> = blob.iter().take(32).map(|&b| b as f32).collect();
            let embed_vals: Vec<f32> = unsafe {
                std::slice::from_raw_parts(blob.as_ptr() as *const f32, 8.min(blob.len() / 4))
            }.to_vec();
            log::debug!("[DIAG] weight_blob_len={} embed_first8={:?}", blob.len(), embed_vals);

            // First layer attn_norm (at layer_blob_base_offset)
            let layer_base_off = mega.hetero_layout.as_ref()
                .map(|hl| hl.layer_0_offset)
                .unwrap_or(mega.weight_layout.layer_0_offset);
            let norm_vals: Vec<f32> = if layer_base_off + 32 <= blob.len() {
                unsafe {
                    std::slice::from_raw_parts(blob.as_ptr().add(layer_base_off) as *const f32, 8)
                }.to_vec()
            } else { vec![] };
            log::debug!("[DIAG] layer_base_off={} first_layer_norm_first8={:?}", layer_base_off, norm_vals);

            // lm_head (transposed embed): check first few values
            let lm_off = mega.weight_layout.lm_head_offset;
            let lm_vals: Vec<f32> = if lm_off + 32 <= blob.len() {
                unsafe {
                    std::slice::from_raw_parts(blob.as_ptr().add(lm_off) as *const f32, 8)
                }.to_vec()
            } else { vec![] };
            log::debug!("[DIAG] lm_head_off={} lm_head_first8={:?}", lm_off, lm_vals);

            // 4. CRITICAL: Read final_normed at BufferAllocation offset 0
            // But argmax_token_id also maps to offset 0, overwriting first 512 bytes.
            // Read from offset 512 to get the actual final_normed data.
            let sp = scratchpad.as_ptr();
            let hidden = self.hidden_size;
            let final_normed_off = 0usize;
            let after_argmax = 512usize; // Skip argmax_token_id region

            // Read final_normed data after the argmax region
            let fn_vals: Vec<f32> = if after_argmax + 64 <= scratchpad.len() {
                unsafe {
                    std::slice::from_raw_parts(sp.add(after_argmax) as *const f32, 16)
                }.to_vec()
            } else { vec![] };
            log::debug!("[DIAG] final_normed@{}+{} first16={:?}", final_normed_off, after_argmax, fn_vals);

            // Check if final_normed is all zeros or has diverse values
            let fn_ptr = unsafe { sp.add(after_argmax) as *const f32 };
            let mut nonzero_count = 0usize;
            let mut sum_sq = 0.0f64;
            let check_count = (hidden.min(1536) - after_argmax / 4).max(0);
            for i in 0..check_count {
                let v = unsafe { *fn_ptr.add(i) };
                if v != 0.0 { nonzero_count += 1; }
                sum_sq += (v as f64) * (v as f64);
            }
            let rms = if check_count > 0 { (sum_sq / check_count as f64).sqrt() } else { 0.0 };
            log::debug!("[DIAG] final_normed@{} nonzero={}/{} rms={:.6}", after_argmax, nonzero_count, check_count, rms);

            // Also check offset 0 directly (argmax result)
            let argmax_vals: Vec<f32> = if 16 * 4 <= scratchpad.len() {
                unsafe {
                    std::slice::from_raw_parts(sp as *const f32, 16)
                }.to_vec()
            } else { vec![] };
            log::debug!("[DIAG] offset0_argmax_result first16={:?}", argmax_vals);

            // Check what the GEMM actually read: the scratchpad offset 0 after argmax
            // The GEMM input (final_normed) was at offset 0 BEFORE argmax overwrote it.
            // Since the diagnostic runs AFTER JIT, we can't see the original final_normed.
            // But we can check the BufferAllocation logits area (where the ACTUAL logits were written
            // by the JIT, which is Output { offset: 0 } = scratchpad + logits_scratch_offset).
            // Note: logits tensor in BufferAllocation is at offset 786432, but the mega-kernel
            // override redirects it to Output { offset: 0 }.

            // 5. Logits for ALL rows from Rust's logits_off (where Rust reads)
            if logits_off + self.vocab_size * 4 * 16 <= scratchpad.len() {
                let mut row_firsts = Vec::new();
                for row in 0..16usize {
                    let row_off = logits_off + row * self.vocab_size * 4;
                    let val = unsafe { *(sp.add(row_off) as *const f32) };
                    row_firsts.push(val);
                }
                log::debug!("[DIAG] logits_row_first_element={:?}", row_firsts);
            }

            // 6. Manual GEMV from final_normed@512 to see if the layer output was diverse
            if after_argmax + hidden * 4 <= scratchpad.len() && lm_off + hidden * 4 <= blob.len() {
                let act_ptr = unsafe { sp.add(after_argmax) as *const f32 };
                let act_first16: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(act_ptr, 16)
                }.to_vec();

                let w_ptr = unsafe { blob.as_ptr().add(lm_off) as *const f32 };

                let mut dot_full = 0.0f64;
                for i in 0..check_count {
                    let a = unsafe { *act_ptr.add(i) } as f64;
                    let w = unsafe { *w_ptr.add(after_argmax / 4 + i) } as f64;
                    dot_full += a * w;
                }
                log::debug!(
                    "[DIAG] manual_gemv_from_final_normed@{}: act_first16={:?} dot(partial={})={:.8}",
                    after_argmax, act_first16, check_count, dot_full,
                );
            }

            // 7. Check if the layer loop produced any intermediate results
            // Read from the layer_sliding_small_attn_normed offset (786432)
            let attn_norm_off = 786432usize;
            if attn_norm_off + 32 <= scratchpad.len() {
                let vals: Vec<f32> = unsafe {
                    std::slice::from_raw_parts(sp.add(attn_norm_off) as *const f32, 8)
                }.to_vec();
                log::debug!("[DIAG] attn_normed@786432 first8={:?}", vals);
            }

            // 8. Check if there's actual data further into the scratchpad (layer intermediates)
            for off in &[0usize, 786432, 2359296, 5505024, 8650752, 21233664] {
                if *off + 32 <= scratchpad.len() {
                    let vals: Vec<f32> = unsafe {
                        std::slice::from_raw_parts(sp.add(*off) as *const f32, 4)
                    }.to_vec();
                    let non_zero = vals.iter().filter(|&&v| v != 0.0).count();
                    log::debug!("[DIAG] sp@{} first4={:?} nonzero={}", off, vals, non_zero);
                }
            }
        }

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
