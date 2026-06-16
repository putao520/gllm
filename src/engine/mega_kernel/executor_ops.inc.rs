impl MegaKernelExecutor {

    /// Diagnostic: run prefill only (max_new_tokens=0) and return logits from scratchpad.
    ///
    /// Returns logits for the last prompt token: shape [vocab_size].
    pub fn diagnostic_prefill_logits(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        let prompt_len = prompt_tokens.len();
        let mut input_ids = vec![0u32; prompt_len + 1];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..(prompt_len + 1) as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(prompt_len + 1)];

        // Pre-fill RoPE cache
        // PERF: RoPE cos/sin table 用 F32 精度(三角函数标准精度),非统一精度假设
        // (具体 compute dtype 由 JIT codegen 按 op inputs 推导,RoPE table 是独立预处理)
        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = (prompt_len + 1) * rc.head_dim;
            let rope_bytes = rope_elems * 4; // F32 elem_bytes = 4
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
                    rc.rope_scaling,
                );
            }
        }

        // Run with max_new_tokens=1 to get prefill logits
        let _generated = unsafe {
            // R1: Build KernelContext
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();

            (mega.entry_fn)(
                input_ids.as_ptr(),
                ctx.weight_blob_ptr,
                std::ptr::null_mut(), // kv_cache_ptr: null — graph has no persistent KV
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                ctx.scratch_buffer_ptr,
                output_tokens.as_mut_ptr(),
                0, // temperature=0 (greedy)
                1, // top_k=1
                0, // top_p=0
                1, // max_new_tokens=1
                self.eos_token_id as usize,
                std::ptr::null(),     // hook_ctx_ptr
                std::ptr::null_mut(), // telemetry
                0,                    // session_position
                std::ptr::null(),     // fused_hidden_ptr
                0,                    // num_mm_tokens
                std::ptr::null(),     // callback_table_ptr
                std::ptr::null(),     // page_table_ptr
                std::ptr::null(),     // batch_ctx_ptr: NULL = single-seq legacy mode
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
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        let prompt_len = prompt_tokens.len();
        let mut input_ids = vec![0u32; prompt_len + 1];
        input_ids[..prompt_len].copy_from_slice(prompt_tokens);

        let positions: Vec<u32> = (0..(prompt_len + 1) as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(prompt_len + 1)];

        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = (prompt_len + 1) * rc.head_dim;
            let rope_bytes = rope_elems * 4; // F32 elem_bytes = 4 (RoPE 数学精度)
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
                    rc.rope_scaling,
                );
            }
        }

        let _generated = unsafe {
            // R1: Build KernelContext
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();

            (mega.entry_fn)(
                input_ids.as_ptr(),
                ctx.weight_blob_ptr,
                std::ptr::null_mut(), // kv_cache_ptr: null — graph has no persistent KV
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                ctx.scratch_buffer_ptr,
                output_tokens.as_mut_ptr(),
                0,
                1,
                0,
                1,
                self.eos_token_id as usize,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),     // batch_ctx_ptr: NULL = single-seq legacy mode
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

    /// Execute mega-kernel in EncodeToLayer mode (forward pass only, no generate loop).
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
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

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
                    rc.rope_scaling,
                );
            }
        }

        let _generated_count = unsafe {
            // R1: Build KernelContext
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();

            (mega.entry_fn)(
                input_ids.as_ptr(),
                ctx.weight_blob_ptr,
                std::ptr::null_mut(), // kv_cache_ptr: null — encode graph has no persistent KV
                positions.as_ptr(),
                std::ptr::null(), // seq_lens: null
                1,                // batch_size
                prompt_len,
                ctx.scratch_buffer_ptr,
                output_tokens.as_mut_ptr(),
                0, // temperature (unused in encode mode)
                0, // top_k (unused)
                0, // top_p bits (unused)
                1, // max_new_tokens = 1 (minimal)
                self.eos_token_id as usize,
                std::ptr::null(),     // hook_ctx_ptr: null
                std::ptr::null_mut(), // telemetry: null
                0,                    // session_position: new
                std::ptr::null(),     // fused_hidden_ptr: no MM
                0,                    // num_mm_tokens: 0
                std::ptr::null(),     // callback_table_ptr: null
                std::ptr::null(),     // page_table_ptr: null (contiguous KV)
                std::ptr::null(),     // batch_ctx_ptr: NULL = single-seq legacy mode
            )
        };

        // The graph output tensor (MeanPool/classifier result) is redirected to the Output region
        // at scratchpad + logits_scratch_offset (same mechanism as logits-producer output for 含 Argmax 的图).
        let mut output = vec![0.0f32; output_elems];
        let offset = mega.logits_scratch_offset;
        let copy_bytes = output_elems * 4;
        if offset + copy_bytes <= scratchpad.len() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratchpad[offset..].as_ptr() as *const f32,
                    output.as_mut_ptr(),
                    output_elems,
                );
            }
        }
        Ok(output)
    }

    /// Execute mega-kernel in ClassifyBinary mode for decoder-based reranker.
    ///
    /// Runs forward pass through all layers (including logits-producer), then extracts
    /// the logits for yes/no tokens from the scratchpad logits region.
    /// Returns [score_for_yes_token, score_for_no_token].
    pub fn execute_rerank(
        &self,
        prompt_tokens: &[u32],
        yes_token_id: u32,
        no_token_id: u32,
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

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
                    rc.rope_scaling,
                );
            }
        }

        let _generated_count = unsafe {
            // R1: Build KernelContext
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();

            (mega.entry_fn)(
                input_ids.as_ptr(),
                ctx.weight_blob_ptr,
                std::ptr::null_mut(), // kv_cache_ptr: null — rerank graph has no generate loop
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                ctx.scratch_buffer_ptr,
                output_tokens.as_mut_ptr(),
                0,
                0,
                0,
                1, // max_new_tokens = 1 (one iteration for forward pass)
                self.eos_token_id as usize,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),     // batch_ctx_ptr: NULL = single-seq legacy mode
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

    /// HR score_tokens: dot-product between last-token hidden state and target token embeddings.
    ///
    /// Runs a single forward pass, then reads logits from
    /// the scratchpad logits region for each target_token_id.
    /// Returns Vec<f32> with one score per target token.
    pub fn execute_score_tokens(
        &self,
        tokens: &[u32],
        target_token_ids: &[u32],
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        let seq_len = tokens.len();
        let mut input_ids = vec![0u32; seq_len];
        input_ids.copy_from_slice(tokens);

        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(seq_len)];

        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = seq_len * rc.head_dim;
            if rc.cache_offset + rope_elems * 4 <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice, &positions, rc.head_dim, rc.theta, rc.partial,
                    rc.rope_scaling,
                );
            }
        }

        let _generated_count = unsafe {
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();

            (mega.entry_fn)(
                input_ids.as_ptr(),
                ctx.weight_blob_ptr,
                std::ptr::null_mut(), // kv_cache_ptr: null — graph has no generate loop
                positions.as_ptr(),
                std::ptr::null(),
                1,
                seq_len,
                ctx.scratch_buffer_ptr,
                output_tokens.as_mut_ptr(),
                0, 0, 0,
                1,
                self.eos_token_id as usize,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        let logits_off = mega.logits_scratch_offset;
        let vocab = self.vocab_size;
        let row_bytes = vocab * 4;
        let logits_row_off = logits_off + (seq_len - 1) * row_bytes;

        let mut scores = Vec::with_capacity(target_token_ids.len());
        for &tid in target_token_ids {
            let offset = logits_row_off + (tid as usize) * 4;
            let score = if offset + 4 <= scratchpad.len() {
                unsafe {
                    let ptr = scratchpad[offset..].as_ptr() as *const f32;
                    *ptr
                }
            } else {
                0.0f32
            };
            scores.push(score);
        }

        Ok(scores)
    }

    /// HR encode_at_layer / Intent encode_intent: forward pass truncated at anchor_layer.
    ///
    /// Runs a single forward pass, stopping at anchor_layer.
    /// Returns the hidden state from the anchor layer as flat f32.
    pub fn execute_encode_at_layer(
        &self,
        tokens: &[u32],
        anchor_layer: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        let seq_len = tokens.len();
        let mut input_ids = vec![0u32; seq_len];
        input_ids.copy_from_slice(tokens);

        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let mut output_tokens = vec![0u32; 1];
        let mut scratchpad = vec![0u8; mega.runtime_scratchpad_bytes(seq_len)];

        if let Some(ref rc) = mega.rope_cache {
            let rope_elems = seq_len * rc.head_dim;
            if rc.cache_offset + rope_elems * 4 <= scratchpad.len() {
                let rope_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        scratchpad[rc.cache_offset..].as_mut_ptr() as *mut f32,
                        rope_elems,
                    )
                };
                gllm_kernels::compiler::fill_cos_sin_table_partial(
                    rope_slice, &positions, rc.head_dim, rc.theta, rc.partial,
                    rc.rope_scaling,
                );
            }
        }

        let _generated_count = unsafe {
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();

            (mega.entry_fn)(
                input_ids.as_ptr(),
                ctx.weight_blob_ptr,
                std::ptr::null_mut(), // kv_cache_ptr: null — graph has no generate loop
                positions.as_ptr(),
                std::ptr::null(),
                1,
                seq_len,
                ctx.scratch_buffer_ptr,
                output_tokens.as_mut_ptr(),
                0, 0, 0,
                1,
                self.eos_token_id as usize,
                std::ptr::null(),
                std::ptr::null_mut(),
                anchor_layer, // session_position repurposed as anchor_layer for EncodeToLayer
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
            )
        };

        let output_elems = seq_len * hidden_size;
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

    /// Returns true if this executor has a mega-kernel compiled (compile_from_template/geometry path).
    pub fn has_mega_compiled(&self) -> bool {
        self.mega_compiled.is_some()
    }

    /// REQ-COMP11: Register decompress callbacks in the callback table.
    ///
    /// When `decompress_inject.enabled` is true, callers should invoke this method
    /// before `generate_single_sequence_inner` to register Lz4, BitPackRle, and
    /// Nvcomp decompress callbacks in the provided callback table. The JIT will
    /// invoke these callbacks when it encounters a compressed KV page.
    ///
    /// The `page_headers` slice must contain one `KvPageHeader` per page in the
    /// page table, and remain valid for the duration of the mega-kernel call.
    pub fn setup_decompress_callbacks(
        &self,
        table: &mut super::mega_kernel_callback::MegaKernelCallbackTable,
        page_headers: &[crate::kv_cache::KvPageHeader],
    ) {
        if !self.decompress_inject.enabled {
            return;
        }
        let decompress_ctx = super::mega_kernel_callback::KvDecompressCtx {
            page_headers: page_headers.as_ptr() as *const u8,
            num_pages: page_headers.len() as u32,
            page_size_bytes: self.decompress_inject.page_size_bytes as u32,
            header_stride: std::mem::size_of::<crate::kv_cache::KvPageHeader>() as u32,
        };
        // The KvDecompressCtx must live at least as long as the callback table is in use.
        // We box it and leak a raw pointer — the caller is responsible for cleanup after
        // the mega-kernel invocation completes.
        let ctx_box = Box::new(decompress_ctx);
        let ctx_ptr = Box::into_raw(ctx_box) as *const u8;

        unsafe {
            use super::mega_kernel_callback::slot;
            table.register(
                slot::KV_DECOMPRESS_LZ4,
                super::mega_kernel_callback::kv_decompress_lz4_callback as *const u8,
                ctx_ptr,
            );
            table.register(
                slot::KV_DECOMPRESS_BITPACKRLE,
                super::mega_kernel_callback::kv_decompress_bitpackrle_callback as *const u8,
                ctx_ptr,
            );
            table.register(
                slot::KV_DECOMPRESS_NVCOMP,
                super::mega_kernel_callback::kv_decompress_nvcomp_callback as *const u8,
                ctx_ptr,
            );
        }
    }

    /// Returns the expected output element count for embedding inference.
    ///
    /// SPEC/39: unified path — all models use mega_compiled.
    pub fn output_elems_for_embed(&self, _seq_len: usize, hidden_size: usize) -> usize {
        // Always return hidden_size: the MeanPool output is a single pooled vector
        // of shape [hidden], regardless of seq_len. The buffer allocator reuses
        // the activation slot for the MeanPool result.
        hidden_size
    }

    /// Returns the MTP depth (0 = MTP disabled, >0 = candidate tokens per decode step).
    pub fn mtp_depth(&self) -> usize {
        self.mega_compiled.as_ref().map(|m| m.mtp_depth).unwrap_or(0)
    }

    /// Returns the byte offset of a named weight tensor in the weight blob.
    pub fn weight_offset(&self, name: &str) -> Option<usize> {
        let mega = self.mega_compiled.as_ref()?;
        mega.named_offsets
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, off)| *off)
    }

    /// Returns all named weight offsets for diagnostic inspection.
    pub fn weight_offsets(&self) -> Option<&[(String, usize)]> {
        self.mega_compiled
            .as_ref()
            .map(|m| m.named_offsets.as_slice())
    }

    /// Read F32 weight data at a given blob offset + row offset.
    pub fn read_weight_row(&self, tensor_name: &str, row: usize, cols: usize) -> Option<Vec<f32>> {
        let mega = self.mega_compiled.as_ref()?;
        let offset = mega
            .named_offsets
            .iter()
            .find(|(n, _)| n == tensor_name)
            .map(|(_, off)| *off)?;
        let row_offset = offset + row * cols * 4;
        if row_offset + cols * 4 > mega.weight_blob.len() {
            return None;
        }
        let data = unsafe {
            std::slice::from_raw_parts(mega.weight_blob[row_offset..].as_ptr() as *const f32, cols)
        };
        Some(data.to_vec())
    }

    /// Returns the weight blob for GPU upload.
    pub fn weight_blob(&self) -> Option<&[u8]> {
        self.mega_compiled.as_ref().map(|m| m.weight_blob.as_slice())
    }

    /// Returns the GPU PTX/HIP code if available.
    pub fn gpu_code(&self) -> Option<&[u8]> {
        self.mega_compiled.as_ref().and_then(|m| m.gpu_code.as_deref())
    }

    /// Returns total scratchpad bytes needed.
    pub fn scratchpad_bytes(&self) -> usize {
        self.mega_compiled
            .as_ref()
            .map(|m| m.scratchpad_base_bytes)
            .unwrap_or(0)
    }

    /// Layer 6: 将 JIT source map 写入文本文件（供 DAP 调试器使用）。
    /// 仅当 debug_jit=true 时有内容。
    pub fn dump_source_map(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        if let Some(ref mega) = self.mega_compiled {
            if let Some(ref sm) = mega.source_map {
                let mut f = std::fs::File::create(path)?;
                writeln!(f, "=== JIT Source Map ({} entries) ===", sm.entries.len())?;
                write!(f, "{}", sm.to_text())?;
            }
        }
        Ok(())
    }

    /// §19 KV-OPT-009: 查询当前 batch 属性对应的最优 Variant。
    ///
    /// 在 build_batch() 阶段调用，根据 KV tier / MoE / Guardrail 等属性
    /// 选择已编译的 variant。当前只有 default FP16 variant，框架先接通。
    /// NOP out an expert's code region in the JIT mega-kernel (Hot JMP Patching §14.4).
    ///
    /// Used by executor when MoE dispatch skips an evicted expert.
    /// The code region must have been registered during JIT compilation.
    /// Returns the saved original bytes for later restore via `restore_expert_code()`.
    pub fn nop_expert_code(
        &self,
        expert_idx: usize,
        layer_idx: usize,
        code_offset: usize,
        code_len: usize,
    ) -> Result<Option<Vec<u8>>, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("no mega-kernel compiled".into()))?;

        if code_len == 0 {
            return Ok(None);
        }

        // Save original bytes before NOP-out
        let saved = mega
            .exec_code
            .save_code_region(code_offset, code_len)
            .map_err(|e| MegaKernelError::Execution(format!("save_code_region failed: {e}")))?;

        // NOP out the region
        mega.exec_code
            .nop_code_region(code_offset, code_len)
            .map_err(|e| MegaKernelError::Execution(format!("nop_code_region failed: {e}")))?;

        log::info!(
            "[mega] §14.4 NOP'd expert {} at layer {} code [{}, {})",
            expert_idx,
            layer_idx,
            code_offset,
            code_offset + code_len,
        );

        Ok(Some(saved))
    }

    /// Restore an expert's code region from previously saved bytes (OSR Bailout §15.4).
    pub fn restore_expert_code(
        &self,
        expert_idx: usize,
        layer_idx: usize,
        code_offset: usize,
        saved_bytes: &[u8],
    ) -> Result<(), MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("no mega-kernel compiled".into()))?;

        if saved_bytes.is_empty() {
            return Ok(());
        }

        mega.exec_code
            .write_code_region(code_offset, saved_bytes)
            .map_err(|e| MegaKernelError::Execution(format!("write_code_region failed: {e}")))?;

        log::info!(
            "[mega] §15.4 restored expert {} at layer {} code [{}, {})",
            expert_idx,
            layer_idx,
            code_offset,
            code_offset + saved_bytes.len(),
        );

        Ok(())
    }

    /// Get the compiled layer's code size (for bounds checking).
    pub fn mega_code_size(&self) -> usize {
        self.mega_compiled
            .as_ref()
            .map(|m| m.exec_code.code_size())
            .unwrap_or(0)
    }

    pub fn select_variant_for_batch(
        &self,
        kv_tier: Option<&str>,
        moe_config: Option<crate::jit::variant_registry::MoeConfigBrief>,
        guardrail_active: bool,
        rag_active: bool,
        batch_golden_size: Option<usize>,
    ) -> Option<&crate::jit::variant_registry::CompiledVariant> {
        let key = crate::jit::variant_registry::VariantRegistry::derive_key(
            "default",
            moe_config,
            guardrail_active,
            None,
            rag_active,
            64,
            None,
            kv_tier.map(|s| s.to_string()),
            batch_golden_size,
        );
        self.variant_registry.find_closest(&key)
    }

    /// §19 KV-OPT-009: 返回 variant_registry 引用（供 executor 查询 tier 分布）
    pub fn variant_registry(&self) -> &crate::jit::variant_registry::VariantRegistry {
        &self.variant_registry
    }

    /// Store GPU mega-kernel PTX/HIP code for mega-kernel path.
    pub fn set_decoder_gpu_code(&mut self, code: Vec<u8>) {
        if let Some(ref mut mega) = self.mega_compiled {
            mega.gpu_code = Some(code);
        }
    }

    /// Diagnostic: run forward-only compilation of the same graph, execute once.
    /// Bypasses mega-kernel generate loop to isolate forward pass correctness.
    pub fn diagnostic_forward_only(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, MegaKernelError> {
        let mega = self
            .mega_compiled
            .as_ref()
            .ok_or_else(|| MegaKernelError::Execution("not a generate-loop mega-kernel".into()))?;

        // We reuse the mega-kernel weight_blob — same weights, same layout.
        // But compile a fresh forward-only graph via InferenceCompiler::compile_graph.
        // This requires the original CompilerGraph, which we do not store.
        // Instead, we run the mega-kernel in encode mode (forward pass only)
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
                    rope_slice,
                    &positions,
                    rc.head_dim,
                    rc.theta,
                    rc.partial,
                    rc.rope_scaling,
                );
            }
        }

        // Run forward pass only, no generate loop logic
        let _result = unsafe {
            // R1: Build KernelContext
            let mut ctx = KernelContext::zeroed();
            ctx.weight_blob_ptr = mega.weight_blob.as_ptr();
            ctx.scratch_buffer_ptr = scratchpad.as_mut_ptr();

            (mega.entry_fn)(
                input_ids.as_ptr(),
                ctx.weight_blob_ptr,
                std::ptr::null_mut(), // kv_cache_ptr: null — forward-only graph has no persistent KV
                positions.as_ptr(),
                std::ptr::null(),
                1,
                prompt_len,
                ctx.scratch_buffer_ptr,
                output_tokens.as_mut_ptr(),
                0,
                1,
                0,
                1, // max_new_tokens=1
                self.eos_token_id as usize,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),     // batch_ctx_ptr: NULL = single-seq legacy mode
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
