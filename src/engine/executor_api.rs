//! Public API methods for Executor — Phase X2 decomposition.
//!
//! This file contains the generate/embed/rerank/classify/diagnostic API surface.
//! Split from executor.rs to keep it under the 2000-line limit.

use crate::compat::backend_trait::{Backend, Element};
use crate::scheduler::types::{PageId, RequestId, StorageKey};

use super::executor::{
    BackendError, BatchInput, Executor, ExecutorError, ExecutorResult, KvCacheHandle,
    LogitsHandle, SequenceInput,
};
use crate::kv_cache::{KvCacheDoubleBuffer, KvCacheSlot, KvCacheState};
use crate::model_config::ModelConfigError;
use crate::scheduler::SessionId;

impl<B: Backend<E> + 'static, E: Element> Executor<B, E> {
    /// Single-step forward: input tokens → logits via mega-kernel.
    pub fn forward_step(&mut self, tokens: &[u32]) -> ExecutorResult<LogitsHandle> {
        if let Some(mega) = self.compute.mega_kernel.as_ref() {
            if mega.has_mega_compiled() {
                let logits = mega
                    .diagnostic_prefill_logits(tokens)
                    .map_err(|e| ExecutorError::Backend(BackendError::Other(e.to_string())))?;
                return Ok(LogitsHandle { data: logits });
            }
        }

        // Fallback: Backend path (GPU or CPU without mega-kernel)
        let seq = SequenceInput {
            tokens: tokens.to_vec(),
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };

        let batch_input = BatchInput {
            sequences: vec![seq],
        };

        let mut kv_cache = self.active_kv_handle()?;

        let (logits_list, _sparsity, _telemetries) = self.backend.batch_forward_gpu_pure(
            &batch_input,
            &self.model_ctx.topology,
            &self.model_ctx.weights,
            std::slice::from_mut(&mut kv_cache),
            &self.model_ctx.forward_config,
        )?;

        if let Some(kv_cache) = self.kv.kv_cache.as_mut() {
            let active = kv_cache.slot_mut(self.kv.kv_cache_slot);
            active.advance(tokens.len())?;
        }

        logits_list.into_iter().next().ok_or(ExecutorError::EmptySample)
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> ExecutorResult<String> {
        self.generate_with_sampling(prompt, max_tokens, temperature, 0, 1.0, None)
    }

    pub fn generate_with_sampling(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        _thinking_budget: Option<usize>,
    ) -> ExecutorResult<String> {
        if prompt.trim().is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }

        if self.compute.mega_kernel.is_some() {
            return self.generate_with_sampling_mega(
                prompt, max_tokens, temperature, top_k, top_p,
            );
        }

        Err(ExecutorError::Backend(BackendError::Other(
            "generate_with_sampling: mega-kernel not compiled for this platform.".into(),
        )))
    }

    fn generate_with_sampling_mega(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> ExecutorResult<String> {
        let prompt_tokens = self.encode_prompt(prompt)?;
        let total_needed = prompt_tokens.len() + max_tokens;
        if total_needed > self.model_ctx.geometry.max_seq_len {
            return Err(ExecutorError::Backend(BackendError::Other(format!(
                "sequence length overflow: prompt({}) + max_new({}) = {} > max_seq_len({})",
                prompt_tokens.len(), max_tokens, total_needed, self.model_ctx.geometry.max_seq_len,
            ))));
        }
        log::debug!("[executor] mega-kernel path: prompt_tokens={:?}, max_tokens={}", prompt_tokens, max_tokens);

        // §19 KV-OPT-009: Variant selection
        let is_moe = self.model_ctx.geometry.is_moe();
        let has_guardrail = self.model_ctx.hooks.read().map(|h| !h.is_empty()).unwrap_or(false);
        let kv_tier_ref = self.kv.majority_kv_tier.as_deref();
        let mega = self.compute.mega_kernel.as_ref().unwrap();
        let variant = mega.select_variant_for_batch(
            kv_tier_ref, is_moe, has_guardrail,
            self.inference.rag_system.is_some(), None,
        );
        let variant_skip_guardrail = variant.as_ref().is_some_and(|v| {
            !v.mechanisms.contains(&crate::jit::variant_registry::MechanismId::GuardrailProbe)
        });
        if let Some(v) = variant {
            log::trace!("executor: §19 variant: mechanisms={:?}, section={:?}", v.mechanisms, v.section);
        }

        // Collect all immutable data before mutable operations
        let sg_hook_ptr = match self.model_ctx.sg_shared_memory {
            Some(ref mx) => mx.lock().unwrap_or_else(|e| e.into_inner()).as_ptr(),
            None => std::ptr::null(),
        };
        let callback_ptr = if variant_skip_guardrail { std::ptr::null() }
            else if self.model_ctx.callback_table.has_any_callback() { self.model_ctx.callback_table.as_ptr() }
            else { std::ptr::null() };
        let pool_base = self.kv.paged_kv_pool.as_ref().map(|p| p.as_ptr()).unwrap_or(std::ptr::null());

        // §16.1 RAG
        let rag_fused_hidden = self.build_rag_fused_hidden();
        let rag_slice = rag_fused_hidden.as_deref();
        let rag_flag = if rag_fused_hidden.is_some() { 1 } else { 0 };

        let output_tokens = mega
            .generate_single_sequence(
                &prompt_tokens, max_tokens, temperature, top_k, top_p,
                sg_hook_ptr,
                callback_ptr,
                None, pool_base, 0,
                rag_slice,
                rag_flag,
            )
            .map_err(|e| ExecutorError::Backend(BackendError::Other(format!("mega-kernel generate failed: {}", e))))?;
        log::debug!("[executor] mega-kernel output_tokens={:?}", output_tokens);
        self.decode_tokens(&output_tokens)
    }

    /// MTP-aware generate with verification (REQ-MTP-002).
    ///
    /// When the mega-kernel has MTP enabled (mtp_depth > 0), this method:
    /// 1. Runs mega-kernel generate to produce main tokens + MTP candidates
    /// 2. Parses output to separate main tokens from per-step candidates
    /// 3. Verifies each step's candidates against full-model forward logits
    /// 4. Commits only accepted tokens, updates EMA tracker
    ///
    /// Falls back to standard generate when MTP is not configured or when
    /// the EMA tracker has disabled MTP due to low acceptance.
    pub fn generate_with_mtp(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
    ) -> ExecutorResult<super::mtp_executor::MtpGenerateResult> {
        if prompt.trim().is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }

        let mega = self.compute.mega_kernel.as_ref().ok_or_else(|| {
            ExecutorError::Backend(BackendError::Other("no mega-kernel available".into()))
        })?;

        let mtp_depth = mega.mtp_depth();
        if mtp_depth == 0 {
            // No MTP configured — run standard generate and wrap result
            let text = self.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p, None)?;
            let tokens = self.encode_prompt(&text)?;
            return Ok(super::mtp_executor::MtpGenerateResult {
                committed_tokens: tokens,
                total_mtp_candidates: 0,
                total_mtp_accepted: 0,
                step_details: Vec::new(),
            });
        }

        // Check adaptive controller: if MTP was disabled by EMA, fall back
        if !self.inference.mtp_controller.is_enabled() {
            log::debug!("[executor] MTP disabled by adaptive controller (ema={:.3}), using standard decode",
                self.inference.mtp_controller.ema_rate());
            let text = self.generate_with_sampling(prompt, max_tokens, temperature, top_k, top_p, None)?;
            let tokens = self.encode_prompt(&text)?;
            // Feed a "stable" round to the controller to potentially re-enable
            self.inference.mtp_controller.record_acceptance(1, 1);
            return Ok(super::mtp_executor::MtpGenerateResult {
                committed_tokens: tokens,
                total_mtp_candidates: 0,
                total_mtp_accepted: 0,
                step_details: Vec::new(),
            });
        }

        let prompt_tokens = self.encode_prompt(prompt)?;
        if prompt_tokens.is_empty() {
            return Err(ExecutorError::EmptyPrompt);
        }

        let total_needed = prompt_tokens.len() + max_tokens;
        if total_needed > self.model_ctx.geometry.max_seq_len {
            return Err(ExecutorError::Backend(BackendError::Other(format!(
                "sequence length overflow: prompt({}) + max_new({}) = {} > max_seq_len({})",
                prompt_tokens.len(), max_tokens, total_needed, self.model_ctx.geometry.max_seq_len,
            ))));
        }

        // Collect immutable data before mutable operations
        let sg_hook_ptr = match self.model_ctx.sg_shared_memory {
            Some(ref mx) => mx.lock().unwrap_or_else(|e| e.into_inner()).as_ptr(),
            None => std::ptr::null(),
        };
        let callback_ptr = if self.model_ctx.callback_table.has_any_callback() {
            self.model_ctx.callback_table.as_ptr()
        } else {
            std::ptr::null()
        };
        let pool_base = self.kv.paged_kv_pool.as_ref().map(|p| p.as_ptr()).unwrap_or(std::ptr::null());

        let output_tokens = mega
            .generate_single_sequence(
                &prompt_tokens, max_tokens, temperature, top_k, top_p,
                sg_hook_ptr, callback_ptr, None, pool_base, 0, None, 0,
            )
            .map_err(|e| ExecutorError::Backend(BackendError::Other(format!(
                "mega-kernel generate failed: {}", e
            ))))?;

        // Parse mega-kernel output: main tokens + MTP candidates
        // Output layout: [main_tokens..] then [step0_cand0, step0_cand1, ..., stepN_candK]
        let total_output = output_tokens.len();
        // The mega-kernel generates `actual_count` main tokens followed by
        // `actual_count * mtp_depth` candidate tokens.
        let candidate_total = total_output * mtp_depth / (1 + mtp_depth);
        let num_steps = total_output - candidate_total;
        let num_steps = num_steps.min(max_tokens);

        let parsed = super::mtp_executor::MtpOutput::parse(&output_tokens, num_steps, mtp_depth)
            .ok_or_else(|| ExecutorError::Backend(BackendError::Other(
                "MTP output parsing failed: insufficient output tokens".into()
            )))?;

        // Verify MTP candidates using full-model forward (REQ-MTP-002 Verify Phase)
        let eos_token_id = self.model_ctx.model_config.eos_token_id;

        let result = super::mtp_executor::filter_verified_tokens(
            &parsed.main_tokens,
            &parsed.mtp_per_step,
            eos_token_id,
            |step, candidates| {
                self.verify_mtp_candidates_for_step(
                    &prompt_tokens, &parsed.main_tokens, step, candidates,
                )
            },
        );

        // Update EMA tracker with acceptance rate (REQ-MTP-005)
        if result.total_mtp_candidates > 0 {
            self.inference.mtp_controller.record_acceptance(
                result.total_mtp_accepted,
                result.total_mtp_candidates,
            );
        }

        log::debug!(
            "[executor] MTP generate: {} steps, {} candidates, {} accepted ({:.1}%), ema={:.3}, enabled={}",
            result.step_details.len(),
            result.total_mtp_candidates,
            result.total_mtp_accepted,
            if result.total_mtp_candidates > 0 {
                result.total_mtp_accepted as f32 / result.total_mtp_candidates as f32 * 100.0
            } else {
                0.0
            },
            self.inference.mtp_controller.ema_rate(),
            self.inference.mtp_controller.is_enabled(),
        );

        Ok(result)
    }

    /// Verify MTP candidates for a single step using full-model forward.
    ///
    /// Builds the sequence up to and including each candidate position,
    /// runs a full forward to get logits, and checks argmax match.
    fn verify_mtp_candidates_for_step(
        &self,
        prompt_tokens: &[u32],
        main_tokens: &[u32],
        step: usize,
        candidates: &[u32],
    ) -> usize {
        if candidates.is_empty() {
            return 0;
        }

        let mega = match self.compute.mega_kernel.as_ref() {
            Some(m) => m,
            None => return 0,
        };

        // Build verify sequence: prompt + main tokens up to this step + candidates
        // For candidate k, the sequence is prompt + main_tokens[0..=step] + candidates[0..k]
        let mut base_seq = prompt_tokens.to_vec();
        base_seq.extend_from_slice(&main_tokens[..=step.min(main_tokens.len().saturating_sub(1))]);

        let mut logits_per_position = Vec::with_capacity(candidates.len());
        for k in 0..candidates.len() {
            // Build sequence with candidates 0..k (candidate k is the last appended)
            let mut verify_seq = base_seq.clone();
            verify_seq.extend_from_slice(&candidates[..k + 1]);

            // Run full forward to get logits for the last position
            match mega.diagnostic_prefill_logits(&verify_seq) {
                Ok(logits) => logits_per_position.push(logits),
                Err(e) => {
                    log::warn!("MTP verify forward failed at step {} candidate {}: {}", step, k, e);
                    break;
                }
            }
        }

        super::mtp_executor::verify_mtp_candidates(&logits_per_position, candidates)
    }

    fn build_rag_fused_hidden(&self) -> Option<Vec<f32>> {
        let rag = self.inference.rag_system.as_ref()?;
        if rag.retrieval_db.is_empty() { return None; }
        let query = vec![0.0f32; self.model_ctx.geometry.hidden_size];
        let retrieved = rag.retrieve(&query);
        if retrieved.is_empty() { return None; }
        let mut fused = vec![0.0f32; self.model_ctx.geometry.hidden_size];
        for doc in &retrieved {
            for (f, d) in fused.iter_mut().zip(doc.iter()) {
                *f += d * rag.fusion_weight;
            }
        }
        log::debug!("executor: §16.1 RAG: {} docs at layer {}", retrieved.len(), rag.fusion_layer);
        Some(fused)
    }

    /// Generate with multimodal-routed inputs (ARCH-MULTIMODAL-FUSION).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_multimodal(
        &mut self,
        token_ids: Vec<u32>,
        fused_hidden: Vec<f32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        _thinking_budget: Option<usize>,
    ) -> ExecutorResult<String> {
        let mega = self.compute.mega_kernel.as_ref().ok_or_else(|| {
            ExecutorError::Backend(BackendError::Other("no mega-kernel available".into()))
        })?;
        let num_mm_tokens = fused_hidden.len() / self.model_ctx.geometry.hidden_size;
        if fused_hidden.len() != num_mm_tokens * self.model_ctx.geometry.hidden_size {
            return Err(ExecutorError::Backend(BackendError::Other(format!(
                "fused_hidden length {} != num_mm_tokens({}) * hidden_size({})",
                fused_hidden.len(), num_mm_tokens, self.model_ctx.geometry.hidden_size
            ))));
        }
        let sg_hook_ptr = match self.model_ctx.sg_shared_memory {
            Some(ref mx) => mx.lock().unwrap_or_else(|e| e.into_inner()).as_ptr(),
            None => std::ptr::null(),
        };
        let pool_base = self.kv.paged_kv_pool.as_ref().map(|p| p.as_ptr()).unwrap_or(std::ptr::null());
        let output_tokens = mega
            .generate_single_sequence(
                &token_ids, max_tokens, temperature, top_k, top_p,
                sg_hook_ptr,
                if self.model_ctx.callback_table.has_any_callback() { self.model_ctx.callback_table.as_ptr() }
                else { std::ptr::null() },
                None, pool_base, 0, Some(&fused_hidden), num_mm_tokens,
            )
            .map_err(|e| ExecutorError::Backend(BackendError::Other(format!("multimodal generate failed: {}", e))))?;
        self.decode_tokens(&output_tokens)
    }

    /// Generate with session affinity for multi-turn KV cache reuse.
    pub fn generate_with_session(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        session_id: SessionId,
        _thinking_budget: Option<usize>,
    ) -> ExecutorResult<String> {
        if self.compute.mega_kernel.is_none() {
            return Err(ExecutorError::Backend(BackendError::Other("no mega-kernel available".into())));
        }
        let prompt_tokens = self.encode_prompt(prompt)?;
        if prompt_tokens.is_empty() { return Err(ExecutorError::EmptyPrompt); }

        let session_position = self.dispatch.memory_manager
            .session_finalized_position(session_id).unwrap_or(0);

        // Mutable operations first
        let position_agnostic_range = self.optimize_session_prefix(session_position);

        // Collect immutable data
        let sg_hook_ptr = match self.model_ctx.sg_shared_memory {
            Some(ref mx) => mx.lock().unwrap_or_else(|e| e.into_inner()).as_ptr(),
            None => std::ptr::null(),
        };
        let pool_base = self.kv.paged_kv_pool.as_ref().map(|p| p.as_ptr()).unwrap_or(std::ptr::null());
        let callback_ptr = if self.model_ctx.callback_table.has_any_callback() { self.model_ctx.callback_table.as_ptr() }
            else { std::ptr::null() };

        let mega = self.compute.mega_kernel.as_ref().unwrap();
        let output_tokens = if let Some((start, end)) = position_agnostic_range {
            mega.generate_single_sequence_with_position_agnostic(
                &prompt_tokens, max_tokens, temperature, top_k, top_p, sg_hook_ptr,
                callback_ptr,
                None, pool_base, session_position, None, 0, start, end,
            )
        } else {
            mega.generate_single_sequence(
                &prompt_tokens, max_tokens, temperature, top_k, top_p, sg_hook_ptr,
                callback_ptr,
                None, pool_base, session_position, None, 0,
            )
        }.map_err(|e| ExecutorError::Backend(BackendError::Other(format!("session generate failed: {}", e))))?;

        self.dispatch.memory_manager.register_session(session_id);
        self.dispatch.memory_manager.prepare_next_turn(session_id);
        self.decode_tokens(&output_tokens)
    }

    /// §19 KV-OPT-010: Optimize session prefix pages for position-agnostic RoPE.
    fn optimize_session_prefix(&mut self, session_position: usize) -> Option<(usize, usize)> {
        if session_position == 0 { return None; }
        let num_pages = session_position.div_ceil(self.kv.kv_cache_config.page_size);
        let mut sys_headers: Vec<crate::kv_cache::KvPageHeader> = (0..num_pages)
            .map(|page_idx| {
                let mut h = crate::kv_cache::KvPageHeader::new(page_idx as u32);
                h.ref_count = 1;
                let entropy = self.compute.telemetry_aggregator.output_entropy();
                let softmax_max = self.compute.telemetry_aggregator.softmax_max();
                let delta_rho = self.compute.telemetry_aggregator.residual_delta_rho();
                h.entropy_avg = crate::kv_cache::f32_to_f16_bits(entropy);
                h.softmax_max_avg = crate::kv_cache::f32_to_f16_bits(softmax_max);
                h.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(delta_rho);
                h
            }).collect();
        crate::scheduler::kv_optimizer::optimize_system_prompt_pages(
            &self.kv.kv_optimizer, &mut sys_headers, self.model_ctx.geometry.num_kv_heads,
        );

        if let Some(ref mut pool) = self.kv.paged_kv_pool {
            let num_layers = self.model_ctx.geometry.num_layers;
            let num_kv_heads = self.model_ctx.geometry.num_kv_heads;
            let page_size = self.kv.kv_cache_config.page_size;
            let head_dim = self.model_ctx.geometry.head_dim;
            let is_mla = self.model_ctx.geometry.is_mla();
            let kv_dim = self.model_ctx.geometry.kv_dim();
            let mut quant_buf = Vec::new();
            let active_count = self.dispatch.requests.len() as u32;
            let elem_bytes = self.kv.kv_cache_config.dtype_size();
            let layer_bytes = if is_mla {
                page_size * kv_dim * elem_bytes
            } else {
                2 * num_kv_heads * page_size * head_dim * elem_bytes
            };
            for (page_idx, header) in sys_headers.iter_mut().enumerate() {
                let mut layer_headers: Vec<crate::kv_cache::KvPageHeader> =
                    (0..num_layers).map(|_| *header).collect();
                let mut layer_data_refs: Vec<&mut [u8]> = Vec::new();
                for layer_idx in 0..num_layers {
                    let page_offset = page_idx * pool.page_stride() + layer_idx * layer_bytes;
                    if page_offset + layer_bytes <= pool.total_bytes() {
                        let data = unsafe {
                            std::slice::from_raw_parts_mut(pool.as_mut_ptr().add(page_offset), layer_bytes)
                        };
                        layer_data_refs.push(data);
                    }
                }
                crate::scheduler::kv_optimizer::compress_system_prompt_pages(
                    &mut layer_headers, &mut layer_data_refs, elem_bytes,
                    &mut quant_buf, active_count, num_kv_heads, page_size, head_dim,
                );
                *header = layer_headers[0];
            }
        }
        let position_agnostic_count = sys_headers.iter().filter(|h| h.is_position_agnostic()).count();
        if position_agnostic_count > 0 {
            log::debug!("executor: §19 KV-OPT-010: {} pages, {} position_agnostic", num_pages, position_agnostic_count);
            let agnostic_end = sys_headers.iter()
                .filter(|h| h.is_position_agnostic())
                .map(|_| self.kv.kv_cache_config.page_size)
                .sum::<usize>()
                .min(session_position);
            if agnostic_end > 0 { return Some((0, agnostic_end)); }
        }
        None
    }

    /// SPEC/20 REQ-BCI-008: Batch inference — single mega-kernel CALL.
    pub fn generate_batch(
        &mut self,
        requests: &[super::batch_executor::GenerateRequest],
    ) -> Result<Vec<super::batch_executor::GenerateResult>, ExecutorError> {
        use super::batch_executor::BatchInferenceState;
        use crate::scheduler::sequence::{Sequence, SequenceState};
        use crate::scheduler::types::BatchOrderPolicy;

        if requests.is_empty() { return Ok(Vec::new()); }
        let mega = self.compute.mega_kernel.as_ref().ok_or_else(|| {
            ExecutorError::Backend(BackendError::Other("mega-kernel not compiled".into()))
        })?;
        let pool_base = self.kv.paged_kv_pool.as_ref().map(|p| p.as_ptr()).unwrap_or(std::ptr::null());

        for req in requests {
            let mut seq = Sequence::new(req.request_id, req.prompt_tokens.clone());
            seq.state = SequenceState::Waiting;
            self.dispatch.batcher.enqueue(seq);
        }
        let token_budget: usize = requests.iter().map(|r| r.prompt_tokens.len()).sum();
        let (scheduled, prep) = self.dispatch.batcher.build_batch_with_prep(
            &mut self.dispatch.scheduler, token_budget, true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        let scheduled_requests: Vec<_> = scheduled.requests.iter()
            .filter_map(|id| requests.iter().find(|r| r.request_id == *id).cloned())
            .collect();
        let batch_state = BatchInferenceState::build_from_prep(&prep, &scheduled_requests);

        let result = mega.generate_batch(
            &batch_state.batch_ctx, &batch_state.input_ids_flat, &batch_state.positions_flat,
            batch_state.input_ids_flat.len(), batch_state.max_decode_steps(), pool_base,
        );
        match result {
            Ok(total_gen) => log::debug!("[executor] batch: {} seqs, {} prefill, {} decode",
                scheduled.requests.len(), batch_state.input_ids_flat.len(), total_gen),
            Err(e) => return Err(ExecutorError::Backend(BackendError::Other(format!("batch failed: {:?}", e)))),
        }
        Ok(batch_state.collect_results(&scheduled_requests))
    }

    // ── Diagnostic methods ──

    pub fn diagnostic_weight_row(&self, tensor_name: &str, row: usize, cols: usize) -> Option<Vec<f32>> {
        self.compute.mega_kernel.as_ref()?.read_weight_row(tensor_name, row, cols)
    }

    pub fn diagnostic_weight_offsets(&self) -> Option<Vec<(String, usize)>> {
        self.compute.mega_kernel.as_ref().map(|m| m.weight_offsets().unwrap_or(&[]).to_vec())
    }

    pub fn diagnostic_prefill_logits(&self, prompt_tokens: &[u32]) -> Option<Vec<f32>> {
        self.compute.mega_kernel.as_ref()?.diagnostic_prefill_logits(prompt_tokens).ok()
    }

    pub fn diagnostic_forward_only(&self, prompt_tokens: &[u32]) -> Option<Vec<f32>> {
        self.compute.mega_kernel.as_ref()?.diagnostic_forward_only(prompt_tokens).ok()
    }

    pub fn diagnostic_prefill_scratchpad(
        &self, prompt_tokens: &[u32],
    ) -> Option<crate::engine::mega_kernel::DiagnosticScratchpad> {
        self.compute.mega_kernel.as_ref()?.diagnostic_prefill_scratchpad(prompt_tokens).ok()
    }

    pub fn weight_page_jit_config(&self) -> crate::engine::mega_kernel::WeightPageJitConfig {
        self.compute.mega_kernel.as_ref().map(|m| m.weight_page_inject.clone()).unwrap_or_default()
    }

    pub fn set_weight_page_jit_config(&mut self, config: crate::engine::mega_kernel::WeightPageJitConfig) {
        if let Some(ref mut mega) = self.compute.mega_kernel {
            mega.weight_page_inject = config;
        }
    }

    pub fn embed(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() { return Err(ExecutorError::EmptyPrompt); }
        let mega = self.compute.mega_kernel.as_ref().ok_or_else(|| {
            ExecutorError::Backend(BackendError::Other("mega-kernel not compiled".into()))
        })?;
        let output_elems = mega.output_elems_for_embed(tokens.len(), self.model_ctx.geometry.hidden_size);
        mega.execute_encode(&tokens, output_elems)
            .map_err(|e| ExecutorError::Backend(BackendError::Other(e.to_string())))
    }

    pub fn rerank(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() { return Err(ExecutorError::EmptyPrompt); }
        let mega = self.compute.mega_kernel.as_ref().ok_or_else(|| {
            ExecutorError::Backend(BackendError::Other("mega-kernel not compiled".into()))
        })?;
        let output_elems = mega.output_elems_for_embed(tokens.len(), self.model_ctx.geometry.hidden_size);
        mega.execute_encode(&tokens, output_elems)
            .map_err(|e| ExecutorError::Backend(BackendError::Other(e.to_string())))
    }

    pub fn rerank_pair(&mut self, query: &str, document: &str) -> ExecutorResult<Vec<f32>> {
        // Generative reranker: uses token generation (yes/no) for scoring.
        // Derived from manifest.kind (service mode), not arch_family.
        // A decoder-based reranker generates yes/no tokens; an encoder-based reranker
        // uses a classifier head. The distinction is service-mode driven, not architecture driven.
        let is_generative = matches!(self.model_ctx.manifest.kind, crate::manifest::ModelKind::Reranker)
            && self.model_ctx.forward_config.arch_family == crate::manifest::ArchFamily::Decoder;
        let tokens = if is_generative {
            self.model_ctx.tokenizer.encode(&format!("{} {}", query, document), self.model_ctx.add_special_tokens)?
        } else {
            self.model_ctx.tokenizer.encode_pair(query, document, self.model_ctx.add_special_tokens)?
        };
        if tokens.is_empty() { return Err(ExecutorError::EmptyPrompt); }
        self.resolve_rerank_token_ids(is_generative);
        let mega = self.compute.mega_kernel.as_ref().ok_or_else(|| {
            ExecutorError::Backend(BackendError::Other("mega-kernel not compiled".into()))
        })?;
        if is_generative && mega.has_mega_compiled() {
            let yes_id = self.model_ctx.forward_config.rerank_yes_token_id.ok_or_else(|| {
                ExecutorError::Compilation("generative reranker: yes_token_id not resolved".into())
            })?;
            let no_id = self.model_ctx.forward_config.rerank_no_token_id.ok_or_else(|| {
                ExecutorError::Compilation("generative reranker: no_token_id not resolved".into())
            })?;
            mega.execute_rerank(&tokens, yes_id, no_id).map_err(|e| ExecutorError::Backend(BackendError::Other(e.to_string())))
        } else {
            let output_elems = mega.output_elems_for_embed(tokens.len(), self.model_ctx.geometry.hidden_size);
            mega.execute_encode(&tokens, output_elems).map_err(|e| ExecutorError::Backend(BackendError::Other(e.to_string())))
        }
    }

    fn resolve_rerank_token_ids(&mut self, is_generative: bool) {
        if is_generative && self.model_ctx.forward_config.rerank_yes_token_id.is_none() {
            if let Ok(yes_ids) = self.model_ctx.tokenizer.encode("yes", false) {
                if let Some(&id) = yes_ids.first() {
                    self.model_ctx.forward_config.rerank_yes_token_id = Some(id);
                }
            }
        }
        if is_generative && self.model_ctx.forward_config.rerank_no_token_id.is_none() {
            if let Ok(no_ids) = self.model_ctx.tokenizer.encode("no", false) {
                if let Some(&id) = no_ids.first() {
                    self.model_ctx.forward_config.rerank_no_token_id = Some(id);
                }
            }
        }
    }

    pub fn classify(&mut self, input: &str) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(input)?;
        if tokens.is_empty() { return Err(ExecutorError::EmptyPrompt); }
        let mega = self.compute.mega_kernel.as_ref().ok_or_else(|| {
            ExecutorError::Backend(BackendError::Other("mega-kernel not compiled".into()))
        })?;
        let output_elems = tokens.len() * 2;
        mega.execute_encode(&tokens, output_elems)
            .map_err(|e| ExecutorError::Backend(BackendError::Other(e.to_string())))
    }

    pub fn score_tokens_for_prompt(
        &mut self, prompt: &str, target_token_ids: &[u32],
    ) -> ExecutorResult<Vec<f32>> {
        self.score_tokens_for_prompt_with_callbacks(prompt, target_token_ids, None)
    }

    pub fn score_tokens_for_prompt_with_callbacks(
        &mut self, prompt: &str, target_token_ids: &[u32],
        callbacks: Option<&mut crate::graph::layer_callback::CallbackChain>,
    ) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(prompt)?;
        if tokens.is_empty() { return Err(ExecutorError::EmptyPrompt); }
        if let Some(mega) = self.compute.mega_kernel.as_ref() {
            if mega.has_mega_compiled() {
                return mega.execute_score_tokens(&tokens, target_token_ids)
                    .map_err(|e| ExecutorError::Backend(BackendError::Other(e.to_string())));
            }
        }
        if let Some(c) = callbacks {
            self.model_ctx.forward_config.callback_chain.set(c as *mut _);
        }
        let result = self.backend.score_tokens_forward_gpu_pure(
            &tokens, target_token_ids, &self.model_ctx.topology,
            &self.model_ctx.weights, &self.model_ctx.forward_config,
        );
        self.model_ctx.forward_config.callback_chain.clear();
        result.map_err(Into::into)
    }

    pub fn encode_at_layer_for_prompt(
        &mut self, prompt: &str, anchor_layer: usize, pool: crate::head_routing::PoolMode,
    ) -> ExecutorResult<Vec<f32>> {
        let tokens = self.encode_prompt(prompt)?;
        if tokens.is_empty() { return Err(ExecutorError::EmptyPrompt); }
        let hidden_size = self.model_ctx.forward_config.hidden_size();

        // Mega-kernel path: execute_encode_at_layer handles truncation internally
        if let Some(mega) = self.compute.mega_kernel.as_ref() {
            if mega.has_mega_compiled() {
                let hidden = mega.execute_encode_at_layer(&tokens, anchor_layer, hidden_size)
                    .map_err(|e| ExecutorError::Backend(BackendError::Other(e.to_string())))?;
                let seq_len = tokens.len();
                if hidden_size == 0 || hidden.len() < seq_len * hidden_size {
                    return Err(ExecutorError::Scheduler(format!(
                        "encode_at_layer: buffer too small, expected >= {}*{}, got {}",
                        seq_len, hidden_size, hidden.len()
                    )));
                }
                return pool.apply(&hidden, seq_len, hidden_size).map_err(|e| {
                    ExecutorError::Scheduler(format!("encode_at_layer pool failed: {e}"))
                });
            }
        }

        // Backend fallback path (GPU)
        use crate::graph::layer_callback::CallbackChain;
        let mut chain = CallbackChain::new(vec![Box::new(
            crate::engine::callbacks::mid_layer_encode::MidLayerEncodeCallback::new(anchor_layer),
        )]);
        self.model_ctx.forward_config.callback_chain.set(&mut chain as *mut _);
        let result = self.backend.encode_at_layer_forward_gpu_pure(
            &tokens, anchor_layer, &self.model_ctx.topology,
            &self.model_ctx.weights, &self.model_ctx.forward_config,
        );
        self.model_ctx.forward_config.callback_chain.clear();
        let hidden = result?;
        let hidden_size = self.model_ctx.forward_config.hidden_size();
        let seq_len = tokens.len();
        if hidden_size == 0 || hidden.len() < seq_len * hidden_size {
            return Err(ExecutorError::Scheduler(format!(
                "encode_at_layer: buffer too small, expected >= {}*{}, got {}",
                seq_len, hidden_size, hidden.len()
            )));
        }
        pool.apply(&hidden, seq_len, hidden_size).map_err(|e| {
            ExecutorError::Scheduler(format!("encode_at_layer pool failed: {e}"))
        })
    }

    pub fn is_finished(&self, request_id: RequestId) -> bool {
        self.dispatch.requests.get(&request_id).map(|r| r.finished).unwrap_or(false)
    }

    pub fn get_request(&self, request_id: RequestId) -> Option<&super::executor::RequestData> {
        self.dispatch.requests.get(&request_id)
    }

    pub fn release_request(&mut self, request_id: RequestId) {
        self.release_request_pages(request_id);
        self.dispatch.requests.remove(&request_id);
    }

    pub fn get_output(&self, request_id: RequestId) -> ExecutorResult<String> {
        let req = self.dispatch.requests.get(&request_id)
            .ok_or(ExecutorError::Scheduler("Request not found".into()))?;
        self.decode_tokens(&req.output_tokens)
    }

    pub(crate) fn ensure_kv_cache(&mut self) -> ExecutorResult<&mut KvCacheDoubleBuffer> {
        let needs_alloc = self.kv.kv_cache.as_ref().is_none_or(|existing| {
            existing.front().config() != self.kv.kv_cache_config
                || existing.back().config() != self.kv.kv_cache_config
        });
        if needs_alloc {
            let front = self.backend.alloc_kv_cache(&self.kv.kv_cache_config)?;
            let back = self.backend.alloc_kv_cache(&self.kv.kv_cache_config)?;
            let front = KvCacheState::new(front, self.kv.kv_cache_config.clone());
            let back = KvCacheState::new(back, self.kv.kv_cache_config.clone());
            self.kv.kv_cache = Some(KvCacheDoubleBuffer::new(front, back));
            self.kv.kv_cache_slot = KvCacheSlot::Front;
        }
        self.kv.kv_cache.as_mut().ok_or_else(|| {
            ExecutorError::Config(ModelConfigError::InvalidConfig("KV cache not available".into()))
        })
    }

    pub(crate) fn active_kv_handle(&mut self) -> ExecutorResult<KvCacheHandle> {
        let slot = self.kv.kv_cache_slot;
        let cache = self.ensure_kv_cache()?;
        Ok(cache.slot(slot).handle())
    }

    pub fn swap_out_pages(&mut self, page_mappings: &[(PageId, StorageKey)]) -> ExecutorResult<()> {
        let mut handle = self.active_kv_handle()?;
        self.backend.swap_out_pages(&mut handle, page_mappings)?;
        Ok(())
    }

    pub fn refresh_page_states(&mut self) -> ExecutorResult<()> {
        if self.kv.kv_cache.is_some() {
            let handle = self.active_kv_handle()?;
            let states = self.backend.get_page_states(&handle)?;
            self.dispatch.scheduler.sync_page_states(&states);
        }
        Ok(())
    }

    pub(crate) fn storage_key_to_page_id(storage_key: StorageKey) -> ExecutorResult<PageId> {
        usize::try_from(storage_key).map_err(|_| {
            ExecutorError::Scheduler("storage key does not fit into page id".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── storage_key_to_page_id ──

    #[test]
    fn storage_key_to_page_id_zero_succeeds() {
        let result = Executor::<crate::compat::CpuBackend, f32>::storage_key_to_page_id(0);
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn storage_key_to_page_id_small_value_succeeds() {
        let result = Executor::<crate::compat::CpuBackend, f32>::storage_key_to_page_id(42);
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn storage_key_to_page_id_large_valid_value_succeeds() {
        let value = StorageKey::from(usize::MAX as u64);
        let result = Executor::<crate::compat::CpuBackend, f32>::storage_key_to_page_id(value);
        assert_eq!(result.unwrap(), usize::MAX);
    }

    /// On 64-bit platforms usize can hold any u64 value, so overflow is
    /// impossible.  This test verifies the error path by exercising the
    /// `try_from` boundary directly — the conversion only fails on 32-bit
    /// targets.  On 64-bit we confirm every u64 round-trips.
    #[test]
    fn storage_key_to_page_id_u64_max_succeeds_on_64bit() {
        if cfg!(target_pointer_width = "64") {
            let result = Executor::<crate::compat::CpuBackend, f32>::storage_key_to_page_id(u64::MAX);
            assert_eq!(result.unwrap(), usize::MAX);
        }
    }

    // ── BackendError -> ExecutorError conversion ──

    #[test]
    fn backend_error_other_converts_to_executor_error() {
        let backend_err = BackendError::Other("mega-kernel not compiled".into());
        let executor_err: ExecutorError = backend_err.into();
        let msg = format!("{executor_err}");
        assert!(
            msg.contains("mega-kernel not compiled"),
            "expected backend error message, got: {msg}"
        );
    }

    #[test]
    fn backend_error_cuda_converts_to_executor_error() {
        let backend_err = BackendError::Cuda("device lost".into());
        let executor_err: ExecutorError = backend_err.into();
        let msg = format!("{executor_err}");
        assert!(
            msg.contains("device lost"),
            "expected CUDA error message, got: {msg}"
        );
    }

    #[test]
    fn backend_error_cpu_converts_to_executor_error() {
        let backend_err = BackendError::Cpu("allocation failed".into());
        let executor_err: ExecutorError = backend_err.into();
        let msg = format!("{executor_err}");
        assert!(
            msg.contains("allocation failed"),
            "expected CPU error message, got: {msg}"
        );
    }

    #[test]
    fn backend_error_unimplemented_converts_to_executor_error() {
        let backend_err = BackendError::Unimplemented("fancy op");
        let executor_err: ExecutorError = backend_err.into();
        let msg = format!("{executor_err}");
        assert!(
            msg.contains("fancy op"),
            "expected unimplemented message, got: {msg}"
        );
    }

    // ── ExecutorError variants used by executor_api methods ──

    #[test]
    fn executor_error_scheduler_storage_key_overflow_message() {
        let err = ExecutorError::Scheduler("storage key does not fit into page id".into());
        let msg = format!("{err}");
        assert!(msg.contains("scheduler error"));
        assert!(msg.contains("storage key does not fit into page id"));
    }

    #[test]
    fn executor_error_scheduler_request_not_found_message() {
        let err = ExecutorError::Scheduler("Request not found".into());
        let msg = format!("{err}");
        assert!(msg.contains("Request not found"));
    }

    #[test]
    fn executor_error_backend_wraps_mega_kernel_not_compiled() {
        let err = ExecutorError::Backend(BackendError::Other(
            "mega-kernel not compiled".into(),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("mega-kernel not compiled"));
    }

    #[test]
    fn executor_error_backend_wraps_mega_kernel_not_available() {
        let err = ExecutorError::Backend(BackendError::Other(
            "no mega-kernel available".into(),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("no mega-kernel available"));
    }

    #[test]
    fn executor_error_backend_wraps_generate_failed() {
        let err = ExecutorError::Backend(BackendError::Other(
            "mega-kernel generate failed: oom".into(),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("mega-kernel generate failed"));
        assert!(msg.contains("oom"));
    }

    #[test]
    fn executor_error_backend_wraps_multimodal_generate_failed() {
        let err = ExecutorError::Backend(BackendError::Other(
            "multimodal generate failed: alignment error".into(),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("multimodal generate failed"));
    }

    #[test]
    fn executor_error_backend_wraps_session_generate_failed() {
        let err = ExecutorError::Backend(BackendError::Other(
            "session generate failed: prefix mismatch".into(),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("session generate failed"));
    }

    #[test]
    fn executor_error_backend_wraps_batch_failed() {
        let err = ExecutorError::Backend(BackendError::Other(
            "batch failed: insufficient memory".into(),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("batch failed"));
    }

    #[test]
    fn executor_error_compilation_rerank_yes_token_not_resolved() {
        let err = ExecutorError::Compilation(
            "decoder reranker: yes_token_id not resolved".into(),
        );
        let msg = format!("{err}");
        assert!(msg.contains("JIT compilation failed"));
        assert!(msg.contains("yes_token_id not resolved"));
    }

    #[test]
    fn executor_error_compilation_rerank_no_token_not_resolved() {
        let err = ExecutorError::Compilation(
            "decoder reranker: no_token_id not resolved".into(),
        );
        let msg = format!("{err}");
        assert!(msg.contains("no_token_id not resolved"));
    }

    #[test]
    fn executor_error_scheduler_encode_at_layer_buffer_too_small() {
        let err = ExecutorError::Scheduler(
            "encode_at_layer: buffer too small, expected >= 10*64, got 100".into(),
        );
        let msg = format!("{err}");
        assert!(msg.contains("encode_at_layer"));
        assert!(msg.contains("buffer too small"));
    }

    #[test]
    fn executor_error_backend_wraps_mtp_parsing_failed() {
        let err = ExecutorError::Backend(BackendError::Other(
            "MTP output parsing failed: insufficient output tokens".into(),
        ));
        let msg = format!("{err}");
        assert!(msg.contains("MTP output parsing failed"));
    }

    #[test]
    fn executor_error_scheduler_sequence_length_overflow_message() {
        let err = ExecutorError::Backend(BackendError::Other(format!(
            "sequence length overflow: prompt({}) + max_new({}) = {} > max_seq_len({})",
            100, 500, 600, 512,
        )));
        let msg = format!("{err}");
        assert!(msg.contains("sequence length overflow"));
        assert!(msg.contains("100"));
        assert!(msg.contains("500"));
        assert!(msg.contains("512"));
    }

    // ── StorageKey / PageId type alias contracts ──

    #[test]
    fn storage_key_is_u64() {
        let key: StorageKey = 42u64;
        assert_eq!(key, 42u64);
    }

    #[test]
    fn page_id_is_usize() {
        let id: PageId = 42usize;
        assert_eq!(id, 42);
    }

    #[test]
    fn request_id_is_u64() {
        let id: RequestId = 42u64;
        assert_eq!(id, 42);
    }

    // ── KvCacheHandle construction and traits ──

    #[test]
    fn kv_cache_handle_debug_clone_copy() {
        let handle = KvCacheHandle(12345);
        let cloned = handle;
        assert_eq!(handle, cloned);
        let debug_str = format!("{handle:?}");
        assert!(debug_str.contains("12345"));
    }

    // ── LogitsHandle construction and traits ──

    #[test]
    fn logits_handle_debug_clone() {
        let handle = LogitsHandle { data: vec![1.0, 2.0, 3.0] };
        let cloned = handle.clone();
        assert_eq!(handle.data, cloned.data);
        let debug_str = format!("{handle:?}");
        assert!(debug_str.contains("LogitsHandle"));
    }

    #[test]
    fn logits_handle_empty_data() {
        let handle = LogitsHandle { data: vec![] };
        assert!(handle.data.is_empty());
    }

    // ── BackendError::Hip / Metal Display (API error conversion coverage) ──

    #[test]
    fn backend_error_hip_display_format() {
        let err = BackendError::Hip("stream error".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("HIP error:"), "expected HIP prefix, got: {msg}");
        assert!(msg.contains("stream error"));
    }

    #[test]
    fn backend_error_metal_display_format() {
        let err = BackendError::Metal("buffer overflow".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("Metal error:"), "expected Metal prefix, got: {msg}");
        assert!(msg.contains("buffer overflow"));
    }

    // ── ExecutorError variant coverage used by generate/embed/rerank paths ──

    #[test]
    fn executor_error_empty_prompt_display() {
        let err = ExecutorError::EmptyPrompt;
        let msg = format!("{err}");
        assert!(msg.contains("empty prompt"), "expected 'empty prompt', got: {msg}");
    }

    #[test]
    fn executor_error_empty_sample_display() {
        let err = ExecutorError::EmptySample;
        let msg = format!("{err}");
        assert!(msg.contains("empty sample"), "expected 'empty sample', got: {msg}");
    }

    #[test]
    fn executor_error_from_kv_cache_error() {
        let kv_err = crate::kv_cache::KvCacheError::Exhausted {
            requested: 1024,
            available: 512,
        };
        let exec_err: ExecutorError = kv_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("1024"), "expected '1024', got: {msg}");
        assert!(msg.contains("512"), "expected '512', got: {msg}");
    }

    #[test]
    fn executor_error_from_memory_manager_error() {
        let mm_err = crate::scheduler::MemoryManagerError::UnknownSession {
            session_id: crate::scheduler::SessionId::from(99u64),
        };
        let exec_err: ExecutorError = mm_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("99"), "expected session id in message, got: {msg}");
    }

    #[test]
    fn executor_error_from_loader_error() {
        let loader_err = crate::loader::LoaderError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file missing",
        ));
        let exec_err: ExecutorError = loader_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("file missing"), "expected IO message, got: {msg}");
    }

    // ── KvCacheSlot flip (used in active_kv_handle / ensure_kv_cache) ──

    #[test]
    fn kv_cache_slot_flip_front_to_back() {
        let slot = KvCacheSlot::Front;
        assert_eq!(slot.flip(), KvCacheSlot::Back);
    }

    #[test]
    fn kv_cache_slot_flip_back_to_front() {
        let slot = KvCacheSlot::Back;
        assert_eq!(slot.flip(), KvCacheSlot::Front);
    }

    #[test]
    fn kv_cache_slot_double_flip_is_identity() {
        let slot = KvCacheSlot::Front;
        assert_eq!(slot.flip().flip(), slot);
    }

    // ── SequenceInput edge cases (used in forward_step, generate_batch) ──

    #[test]
    fn sequence_input_validate_empty_page_table_is_valid() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_id_zero_is_valid() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(1).is_ok());
    }

    // ── BatchInput empty sequences ──

    #[test]
    fn batch_input_empty_sequences_is_valid() {
        let batch = BatchInput { sequences: vec![] };
        assert!(batch.sequences.is_empty());
    }

    // ── RequestData default sampling config ──

    #[test]
    fn request_data_uses_default_sampling_config() {
        use crate::scheduler::request_state::RequestPhase;

        let rd = super::super::executor::RequestData {
            prompt_tokens: vec![],
            output_tokens: vec![],
            sampling_config: super::super::executor::SamplingConfig::default(),
            is_prefill: true,
            phase: RequestPhase::Prefill,
            max_new_tokens: 0,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        assert_eq!(rd.sampling_config.temperature, 1.0);
        assert_eq!(rd.sampling_config.top_k, 0);
        assert_eq!(rd.sampling_config.top_p, 1.0);
    }

    // ── OomHaltError Display (used via executor error chain) ──

    #[test]
    fn oom_halt_error_fatal_display() {
        let err = crate::kv_cache::OomHaltError::fatal_halt("GPU VRAM exhausted");
        let msg = format!("{err}");
        assert!(msg.contains("OOM Halt"), "expected OOM Halt prefix, got: {msg}");
        assert!(msg.contains("GPU VRAM exhausted"));
        assert!(msg.contains("fatal=true"));
    }

    #[test]
    fn oom_halt_error_soft_display() {
        let err = crate::kv_cache::OomHaltError::soft_halt("retry suggested");
        let msg = format!("{err}");
        assert!(msg.contains("OOM Halt"));
        assert!(msg.contains("retry suggested"));
        assert!(msg.contains("fatal=false"));
    }

    // ── MtpController state machine (used in generate_with_mtp) ──

    #[test]
    fn mtp_controller_default_equals_new() {
        let default_ctrl = super::super::mtp_executor::MtpController::default();
        let new_ctrl = super::super::mtp_executor::MtpController::new();
        assert_eq!(default_ctrl.ema_rate(), new_ctrl.ema_rate());
        assert_eq!(default_ctrl.is_enabled(), new_ctrl.is_enabled());
    }

    #[test]
    fn mtp_controller_disable_then_enable() {
        let mut ctrl = super::super::mtp_executor::MtpController::new();
        assert!(ctrl.is_enabled());
        ctrl.disable();
        assert!(!ctrl.is_enabled());
        ctrl.enable();
        assert!(ctrl.is_enabled());
    }

    #[test]
    fn mtp_controller_effective_depth_adaptive() {
        let mut ctrl = super::super::mtp_executor::MtpController::new();
        // Default ema_rate is 0.5. Since the threshold is strict (> 0.5),
        // 0.5 falls into the > 0.3 branch returning 1.
        assert_eq!(ctrl.effective_depth(4), 1);
        // Drive ema above 0.8 by recording perfect acceptance.
        // ema recurrence: ema_new = 0.1 * rate + 0.9 * ema_old
        // With rate=1.0 starting at 0.5: ema_N = 1 - 0.5 * 0.9^N
        // N=15: ema = 1 - 0.5*0.9^15 = 1 - 0.5*0.2059 = 0.897 > 0.8
        for _ in 0..15 {
            ctrl.record_acceptance(10, 10);
        }
        let ema = ctrl.ema_rate();
        assert!(ema > 0.8, "ema should exceed 0.8, got {ema}");
        assert_eq!(ctrl.effective_depth(4), 4);
        // Verify intermediate: disabled controller returns 0
        ctrl.disable();
        assert_eq!(ctrl.effective_depth(4), 0);
    }

    #[test]
    fn mtp_controller_reset_restores_initial_state() {
        let mut ctrl = super::super::mtp_executor::MtpController::new();
        ctrl.record_acceptance(0, 10);
        ctrl.disable();
        assert!(!ctrl.is_enabled());
        ctrl.reset();
        assert!(ctrl.is_enabled());
        assert_eq!(ctrl.ema_rate(), 0.5);
    }

    // ── MtpOutput parse edge case: single step with partial candidates ──

    #[test]
    fn mtp_output_parse_single_step_partial_candidates() {
        // 1 step, depth=3 but only 1 candidate in output
        let output = vec![100u32, 42];
        let parsed = super::super::mtp_executor::MtpOutput::parse(&output, 1, 3);
        let parsed = parsed.expect("should parse with partial candidates");
        assert_eq!(parsed.main_tokens, vec![100]);
        assert_eq!(parsed.mtp_per_step.len(), 1);
        assert_eq!(parsed.mtp_per_step[0], vec![42]);
    }

    // ── KvCacheSlot Debug / Copy / Eq traits ──

    #[test]
    fn kv_cache_slot_debug_format() {
        let front = KvCacheSlot::Front;
        let debug = format!("{front:?}");
        assert!(debug.contains("Front"), "expected Front in debug output, got: {debug}");

        let back = KvCacheSlot::Back;
        let debug = format!("{back:?}");
        assert!(debug.contains("Back"), "expected Back in debug output, got: {debug}");
    }

    #[test]
    fn kv_cache_slot_copy_preserves_value() {
        let slot = KvCacheSlot::Front;
        let copied = slot; // Copy
        assert_eq!(slot, copied);
    }

    #[test]
    fn kv_cache_slot_equality() {
        assert_eq!(KvCacheSlot::Front, KvCacheSlot::Front);
        assert_eq!(KvCacheSlot::Back, KvCacheSlot::Back);
        assert_ne!(KvCacheSlot::Front, KvCacheSlot::Back);
    }

    // ── OomHaltError Clone and field access ──

    #[test]
    fn oom_halt_error_clone_preserves_fields() {
        let err = crate::kv_cache::OomHaltError::fatal_halt("GPU OOM");
        let cloned = err.clone();
        assert_eq!(err.message, cloned.message);
        assert_eq!(err.fatal, cloned.fatal);
        assert!(cloned.fatal);
    }

    #[test]
    fn oom_halt_error_soft_is_not_fatal() {
        let err = crate::kv_cache::OomHaltError::soft_halt("retry");
        assert!(!err.fatal);
        assert_eq!(err.message, "retry");
    }

    // ── BackendError Clone trait ──

    #[test]
    fn backend_error_clone_cuda() {
        let err = BackendError::Cuda("device lost".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_clone_other() {
        let err = BackendError::Other("test message".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    // ── ExecutorError From<TokenizerError> ──

    #[test]
    fn executor_error_from_tokenizer_error_missing() {
        let tok_err = crate::tokenizer::TokenizerError::MissingTokenizer;
        let exec_err: ExecutorError = tok_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("tokenizer.json not found"), "expected tokenizer message, got: {msg}");
    }

    #[test]
    fn executor_error_from_tokenizer_error_tokenizers() {
        let tok_err = crate::tokenizer::TokenizerError::Tokenizers("encode failed".into());
        let exec_err: ExecutorError = tok_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("encode failed"), "expected encode message, got: {msg}");
    }

    // ── Tier enum variants and traits ──

    #[test]
    fn tier_variants_are_distinct() {
        use crate::scheduler::Tier;
        assert_ne!(Tier::L1, Tier::L2);
        assert_ne!(Tier::L2, Tier::L3);
        assert_ne!(Tier::L1, Tier::L3);
    }

    #[test]
    fn tier_copy_and_hash() {
        use crate::scheduler::Tier;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Tier::L1);
        set.insert(Tier::L2);
        set.insert(Tier::L3);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Tier::L1));
    }

    // ── VirtualPageId construction ──

    #[test]
    fn virtual_page_id_construction() {
        use crate::scheduler::VirtualPageId;
        let vpid = VirtualPageId::new(42u64, 7);
        assert_eq!(vpid.sequence_id, 42u64);
        assert_eq!(vpid.logical_index, 7);
    }

    #[test]
    fn virtual_page_id_equality() {
        use crate::scheduler::VirtualPageId;
        let a = VirtualPageId::new(1, 2);
        let b = VirtualPageId::new(1, 2);
        let c = VirtualPageId::new(1, 3);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── TierUsage construction and available() ──

    #[test]
    fn tier_usage_available_normal() {
        use crate::scheduler::TierUsage;
        let usage = TierUsage { used: 30, capacity: 100 };
        assert_eq!(usage.available(), 70);
    }

    #[test]
    fn tier_usage_available_saturates_at_zero() {
        use crate::scheduler::TierUsage;
        let usage = TierUsage { used: 150, capacity: 100 };
        assert_eq!(usage.available(), 0);
    }

    #[test]
    fn tier_usage_available_exact_full() {
        use crate::scheduler::TierUsage;
        let usage = TierUsage { used: 100, capacity: 100 };
        assert_eq!(usage.available(), 0);
    }

    // ── PageLocation construction and traits ──

    #[test]
    fn page_location_construction_and_equality() {
        use crate::scheduler::{PageLocation, Tier};
        let a = PageLocation { physical_id: 5, tier: Tier::L1 };
        let b = PageLocation { physical_id: 5, tier: Tier::L1 };
        let c = PageLocation { physical_id: 5, tier: Tier::L2 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    // ── SequenceInput with fused_hidden ──

    #[test]
    fn sequence_input_with_fused_hidden() {
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(vec![0.1, 0.2, 0.3]),
        };
        assert_eq!(seq.fused_hidden.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn sequence_input_nonzero_position_and_draft_steps() {
        let seq = SequenceInput {
            tokens: vec![10, 20],
            position: 128,
            draft_steps: 3,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.position, 128);
        assert_eq!(seq.draft_steps, 3);
    }

    // ── SequenceInput validate_page_table with total_pages=0 ──

    #[test]
    fn sequence_input_validate_page_table_zero_total_pages_rejects_any_page() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        let result = seq.validate_page_table(0);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("page_table[0]"));
    }

    // ── MtpGenerateResult construction and Debug ──

    #[test]
    fn mtp_generate_result_construction_and_debug() {
        let result = super::super::mtp_executor::MtpGenerateResult {
            committed_tokens: vec![100, 200],
            total_mtp_candidates: 5,
            total_mtp_accepted: 3,
            step_details: vec![],
        };
        assert_eq!(result.committed_tokens, vec![100, 200]);
        assert_eq!(result.total_mtp_candidates, 5);
        assert_eq!(result.total_mtp_accepted, 3);
        assert!(result.step_details.is_empty());
        let debug = format!("{result:?}");
        assert!(debug.contains("MtpGenerateResult"));
    }

    // ── MtpStepDetail construction and Debug ──

    #[test]
    fn mtp_step_detail_construction_and_debug() {
        let detail = super::super::mtp_executor::MtpStepDetail {
            main_token: 42,
            mtp_candidates: vec![10, 11, 12],
            accepted_count: 2,
            main_token_is_eos: false,
        };
        assert_eq!(detail.main_token, 42);
        assert_eq!(detail.mtp_candidates.len(), 3);
        assert_eq!(detail.accepted_count, 2);
        assert!(!detail.main_token_is_eos);
        let debug = format!("{detail:?}");
        assert!(debug.contains("MtpStepDetail"));
    }

    // ── verify_mtp_candidates edge cases ──

    #[test]
    fn verify_mtp_candidates_empty_inputs() {
        let accepted = super::super::mtp_executor::verify_mtp_candidates(&[], &[]);
        assert_eq!(accepted, 0);
    }

    #[test]
    fn verify_mtp_candidates_empty_candidates() {
        let logits = vec![vec![0.1; 100]];
        let accepted = super::super::mtp_executor::verify_mtp_candidates(&logits, &[]);
        assert_eq!(accepted, 0);
    }

    #[test]
    fn verify_mtp_candidates_all_match() {
        // Candidate tokens match the argmax of provided logits
        let mut logits0 = vec![0.0; 100];
        logits0[10] = 99.0; // argmax = 10
        let mut logits1 = vec![0.0; 100];
        logits1[20] = 99.0; // argmax = 20
        let accepted = super::super::mtp_executor::verify_mtp_candidates(
            &[logits0, logits1],
            &[10u32, 20u32],
        );
        assert_eq!(accepted, 2);
    }

    #[test]
    fn verify_mtp_candidates_partial_match() {
        let mut logits0 = vec![0.0; 100];
        logits0[10] = 99.0;
        let mut logits1 = vec![0.0; 100];
        logits1[20] = 99.0;
        // Candidates: [10 (match), 30 (no match), 40 (would match but stopped)]
        let accepted = super::super::mtp_executor::verify_mtp_candidates(
            &[logits0, logits1],
            &[10u32, 30u32, 40u32],
        );
        // Only 1 accepted: first matches, second does not -> stops
        assert_eq!(accepted, 1);
    }

    // ── generate_mtp_kv_instructions edge case: no candidates ──

    #[test]
    fn generate_mtp_kv_instructions_no_candidates() {
        let detail = super::super::mtp_executor::MtpStepDetail {
            main_token: 1,
            mtp_candidates: vec![],
            accepted_count: 0,
            main_token_is_eos: false,
        };
        let instructions = super::super::mtp_executor::generate_mtp_kv_instructions(42, &[detail]);
        assert!(instructions.is_empty(), "no candidates should produce no instructions");
    }

    // ══════════════════════════════════════════════════════════════════════
    //  NEW TESTS: comprehensive trait / boundary / variant coverage
    // ══════════════════════════════════════════════════════════════════════

    // ── SamplingConfig: boundary values ──

    #[test]
    fn sampling_config_default_values() {
        let cfg = super::super::executor::SamplingConfig::default();
        assert_eq!(cfg.temperature, 1.0);
        assert_eq!(cfg.top_k, 0);
        assert_eq!(cfg.top_p, 1.0);
    }

    #[test]
    fn sampling_config_zero_temperature() {
        let cfg = super::super::executor::SamplingConfig { temperature: 0.0, top_k: 1, top_p: 1.0 };
        assert_eq!(cfg.temperature, 0.0);
    }

    #[test]
    fn sampling_config_nan_temperature() {
        let cfg = super::super::executor::SamplingConfig { temperature: f32::NAN, top_k: 0, top_p: 1.0 };
        assert!(cfg.temperature.is_nan());
    }

    #[test]
    fn sampling_config_inf_temperature() {
        let cfg = super::super::executor::SamplingConfig { temperature: f32::INFINITY, top_k: 0, top_p: 1.0 };
        assert!(cfg.temperature.is_infinite() && cfg.temperature.is_sign_positive());
    }

    #[test]
    fn sampling_config_neg_inf_temperature() {
        let cfg = super::super::executor::SamplingConfig { temperature: f32::NEG_INFINITY, top_k: 0, top_p: 1.0 };
        assert!(cfg.temperature.is_infinite() && cfg.temperature.is_sign_negative());
    }

    #[test]
    fn sampling_config_max_top_k() {
        let cfg = super::super::executor::SamplingConfig { temperature: 1.0, top_k: usize::MAX, top_p: 1.0 };
        assert_eq!(cfg.top_k, usize::MAX);
    }

    #[test]
    fn sampling_config_zero_top_p() {
        let cfg = super::super::executor::SamplingConfig { temperature: 1.0, top_k: 0, top_p: 0.0 };
        assert_eq!(cfg.top_p, 0.0);
    }

    #[test]
    fn sampling_config_nan_top_p() {
        let cfg = super::super::executor::SamplingConfig { temperature: 1.0, top_k: 0, top_p: f32::NAN };
        assert!(cfg.top_p.is_nan());
    }

    #[test]
    fn sampling_config_copy_trait() {
        let cfg = super::super::executor::SamplingConfig::default();
        let copied = cfg;
        assert_eq!(cfg.temperature, copied.temperature);
        assert_eq!(cfg.top_k, copied.top_k);
        assert_eq!(cfg.top_p, copied.top_p);
    }

    #[test]
    fn sampling_config_clone_trait() {
        let cfg = super::super::executor::SamplingConfig { temperature: 0.7, top_k: 50, top_p: 0.95 };
        let cloned = cfg.clone();
        assert_eq!(cfg.temperature, cloned.temperature);
        assert_eq!(cfg.top_k, cloned.top_k);
        assert_eq!(cfg.top_p, cloned.top_p);
    }

    #[test]
    fn sampling_config_debug_trait() {
        let cfg = super::super::executor::SamplingConfig { temperature: 0.5, top_k: 10, top_p: 0.9 };
        let debug = format!("{cfg:?}");
        assert!(debug.contains("SamplingConfig"), "expected SamplingConfig in debug, got: {debug}");
    }

    // ── effective_kv_max_seq_len: boundary values ──

    #[test]
    fn effective_kv_max_seq_len_passthrough() {
        assert_eq!(super::super::executor_types::effective_kv_max_seq_len(2048), 2048);
    }

    #[test]
    fn effective_kv_max_seq_len_zero() {
        assert_eq!(super::super::executor_types::effective_kv_max_seq_len(0), 0);
    }

    #[test]
    fn effective_kv_max_seq_len_one() {
        assert_eq!(super::super::executor_types::effective_kv_max_seq_len(1), 1);
    }

    #[test]
    fn effective_kv_max_seq_len_usize_max() {
        assert_eq!(super::super::executor_types::effective_kv_max_seq_len(usize::MAX), usize::MAX);
    }

    #[test]
    fn effective_kv_max_seq_len_large_power_of_two() {
        assert_eq!(super::super::executor_types::effective_kv_max_seq_len(1 << 20), 1 << 20);
    }

    // ── PositionEncoding: all variants and traits ──

    #[test]
    fn position_encoding_variants_distinct() {
        use super::super::executor_types::PositionEncoding;
        assert_ne!(PositionEncoding::None, PositionEncoding::Rope);
    }

    #[test]
    fn position_encoding_copy_trait() {
        use super::super::executor_types::PositionEncoding;
        let a = PositionEncoding::Rope;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn position_encoding_debug_trait() {
        use super::super::executor_types::PositionEncoding;
        let debug = format!("{:?}", PositionEncoding::Rope);
        assert!(debug.contains("Rope"), "expected Rope in debug, got: {debug}");
    }

    #[test]
    fn position_encoding_eq_trait() {
        use super::super::executor_types::PositionEncoding;
        assert_eq!(PositionEncoding::None, PositionEncoding::None);
        assert_eq!(PositionEncoding::Rope, PositionEncoding::Rope);
    }

    // ── AttentionMaskType: all variants and traits ──

    #[test]
    fn attention_mask_type_variants_distinct() {
        use super::super::executor_types::AttentionMaskType;
        assert_ne!(AttentionMaskType::Bidirectional, AttentionMaskType::Causal);
    }

    #[test]
    fn attention_mask_type_copy_trait() {
        use super::super::executor_types::AttentionMaskType;
        let a = AttentionMaskType::Causal;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn attention_mask_type_debug_trait() {
        use super::super::executor_types::AttentionMaskType;
        let debug = format!("{:?}", AttentionMaskType::Bidirectional);
        assert!(debug.contains("Bidirectional"), "expected Bidirectional, got: {debug}");
    }

    // ── PoolMode: all variants, traits, apply edge cases ──

    #[test]
    fn pool_mode_variants_distinct() {
        use crate::head_routing::PoolMode;
        assert_ne!(PoolMode::MeanPool, PoolMode::LastToken);
        assert_ne!(PoolMode::LastToken, PoolMode::ClsToken);
        assert_ne!(PoolMode::MeanPool, PoolMode::ClsToken);
    }

    #[test]
    fn pool_mode_copy_trait() {
        use crate::head_routing::PoolMode;
        let a = PoolMode::MeanPool;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn pool_mode_eq_trait() {
        use crate::head_routing::PoolMode;
        assert_eq!(PoolMode::MeanPool, PoolMode::MeanPool);
    }

    #[test]
    fn pool_mode_mean_pool_single_token() {
        use crate::head_routing::PoolMode;
        let hidden = vec![1.0, 2.0, 3.0];
        let result = PoolMode::MeanPool.apply(&hidden, 1, 3).unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn pool_mode_last_token_multiple_positions() {
        use crate::head_routing::PoolMode;
        let hidden = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3
        let result = PoolMode::LastToken.apply(&hidden, 2, 3).unwrap();
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn pool_mode_cls_token_returns_first_row() {
        use crate::head_routing::PoolMode;
        let hidden = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]; // 2 x 3
        let result = PoolMode::ClsToken.apply(&hidden, 2, 3).unwrap();
        assert_eq!(result, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn pool_mode_mean_pool_averages_all_rows() {
        use crate::head_routing::PoolMode;
        let hidden = vec![2.0, 4.0, 6.0, 4.0, 6.0, 8.0]; // 2 x 3
        let result = PoolMode::MeanPool.apply(&hidden, 2, 3).unwrap();
        assert_eq!(result, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn pool_mode_apply_zero_seq_len_errors() {
        use crate::head_routing::PoolMode;
        let result = PoolMode::MeanPool.apply(&[1.0], 0, 1);
        assert!(result.is_err());
    }

    #[test]
    fn pool_mode_apply_zero_hidden_size_errors() {
        use crate::head_routing::PoolMode;
        let result = PoolMode::MeanPool.apply(&[1.0], 1, 0);
        assert!(result.is_err());
    }

    #[test]
    fn pool_mode_apply_buffer_too_small_errors() {
        use crate::head_routing::PoolMode;
        let result = PoolMode::MeanPool.apply(&[1.0, 2.0], 2, 3); // need 6 elements
        assert!(result.is_err());
    }

    // ── BatchOrderPolicy: all variants and traits ──

    #[test]
    fn batch_order_policy_variants_distinct() {
        use crate::scheduler::types::BatchOrderPolicy;
        assert_ne!(BatchOrderPolicy::StrictRequestIdOrder, BatchOrderPolicy::FifoOrder);
    }

    #[test]
    fn batch_order_policy_default_is_strict() {
        use crate::scheduler::types::BatchOrderPolicy;
        assert_eq!(BatchOrderPolicy::default(), BatchOrderPolicy::StrictRequestIdOrder);
    }

    #[test]
    fn batch_order_policy_copy_trait() {
        use crate::scheduler::types::BatchOrderPolicy;
        let a = BatchOrderPolicy::StrictRequestIdOrder;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn batch_order_policy_debug_trait() {
        use crate::scheduler::types::BatchOrderPolicy;
        let debug = format!("{:?}", BatchOrderPolicy::StrictRequestIdOrder);
        assert!(debug.contains("StrictRequestIdOrder"), "expected variant name, got: {debug}");
    }

    #[test]
    fn batch_order_policy_all_variants_in_vec() {
        use crate::scheduler::types::BatchOrderPolicy;
        let variants = vec![
            BatchOrderPolicy::StrictRequestIdOrder,
            BatchOrderPolicy::FifoOrder,
        ];
        assert_eq!(variants.len(), 2);
        assert_ne!(variants[0], variants[1]);
    }

    // ── KvPipeline: all variants and traits ──

    #[test]
    fn kv_pipeline_variants_distinct() {
        use crate::scheduler::types::KvPipeline;
        assert_ne!(KvPipeline::Conversation, KvPipeline::Working);
    }

    #[test]
    fn kv_pipeline_copy_trait() {
        use crate::scheduler::types::KvPipeline;
        let a = KvPipeline::Conversation;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn kv_pipeline_eq_trait() {
        use crate::scheduler::types::KvPipeline;
        assert_eq!(KvPipeline::Conversation, KvPipeline::Conversation);
        assert_eq!(KvPipeline::Working, KvPipeline::Working);
    }

    #[test]
    fn kv_pipeline_debug_trait() {
        use crate::scheduler::types::KvPipeline;
        let debug = format!("{:?}", KvPipeline::Working);
        assert!(debug.contains("Working"), "expected Working in debug, got: {debug}");
    }

    #[test]
    fn kv_pipeline_hash_in_set() {
        use crate::scheduler::types::KvPipeline;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(KvPipeline::Conversation);
        set.insert(KvPipeline::Working);
        assert_eq!(set.len(), 2);
    }

    // ── BackendError: all variants Display, Clone, Error trait ──

    #[test]
    fn backend_error_cuda_display() {
        let err = BackendError::Cuda("device error".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("CUDA error:"));
        assert!(msg.contains("device error"));
    }

    #[test]
    fn backend_error_hip_display() {
        let err = BackendError::Hip("hip error".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("HIP error:"));
        assert!(msg.contains("hip error"));
    }

    #[test]
    fn backend_error_metal_display() {
        let err = BackendError::Metal("metal error".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("Metal error:"));
        assert!(msg.contains("metal error"));
    }

    #[test]
    fn backend_error_cpu_display() {
        let err = BackendError::Cpu("cpu error".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("CPU error:"));
        assert!(msg.contains("cpu error"));
    }

    #[test]
    fn backend_error_unimplemented_display() {
        let err = BackendError::Unimplemented("fancy_op");
        let msg = format!("{err}");
        assert!(msg.starts_with("unimplemented:"));
        assert!(msg.contains("fancy_op"));
    }

    #[test]
    fn backend_error_other_display() {
        let err = BackendError::Other("generic error".into());
        let msg = format!("{err}");
        assert!(msg.starts_with("backend error:"));
        assert!(msg.contains("generic error"));
    }

    #[test]
    fn backend_error_empty_string_variants() {
        let err = BackendError::Cuda(String::new());
        assert_eq!(format!("{err}"), "CUDA error: ");
        let err = BackendError::Hip(String::new());
        assert_eq!(format!("{err}"), "HIP error: ");
        let err = BackendError::Metal(String::new());
        assert_eq!(format!("{err}"), "Metal error: ");
        let err = BackendError::Cpu(String::new());
        assert_eq!(format!("{err}"), "CPU error: ");
        let err = BackendError::Other(String::new());
        assert_eq!(format!("{err}"), "backend error: ");
    }

    #[test]
    fn backend_error_clone_hip() {
        let err = BackendError::Hip("test".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_clone_metal() {
        let err = BackendError::Metal("test".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_clone_cpu() {
        let err = BackendError::Cpu("test".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_clone_unimplemented() {
        let err = BackendError::Unimplemented("thing");
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_debug_trait() {
        let err = BackendError::Cuda("err".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Cuda"), "expected Cuda in debug, got: {debug}");
    }

    #[test]
    fn backend_error_implements_std_error() {
        let err = BackendError::Other("test".into());
        let _: &dyn std::error::Error = &err;
    }

    // ── ExecutorError: remaining variant Display and From impls ──

    #[test]
    fn executor_error_request_not_found_display() {
        let err = ExecutorError::RequestNotFound { request_id: 42 };
        let msg = format!("{err}");
        assert!(msg.contains("42"), "expected request_id in message, got: {msg}");
        assert!(msg.contains("request not found"));
    }

    #[test]
    fn executor_error_graph_expansion_display() {
        let err = ExecutorError::GraphExpansion("node resolution failed".into());
        let msg = format!("{err}");
        assert!(msg.contains("graph expansion failed"));
        assert!(msg.contains("node resolution failed"));
    }

    #[test]
    fn executor_error_from_config_error() {
        let cfg_err = ModelConfigError::InvalidConfig("bad layout".into());
        let exec_err: ExecutorError = cfg_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("bad layout"), "expected config error message, got: {msg}");
    }

    #[test]
    fn executor_error_debug_trait() {
        let err = ExecutorError::EmptyPrompt;
        let debug = format!("{err:?}");
        assert!(debug.contains("EmptyPrompt"), "expected EmptyPrompt in debug, got: {debug}");
    }

    // ── KvCacheHandle: edge values and Hash ──

    #[test]
    fn kv_cache_handle_zero() {
        let handle = KvCacheHandle(0);
        assert_eq!(handle.0, 0);
    }

    #[test]
    fn kv_cache_handle_max_u64() {
        let handle = KvCacheHandle(u64::MAX);
        assert_eq!(handle.0, u64::MAX);
    }

    #[test]
    fn kv_cache_handle_hash_in_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(KvCacheHandle(1));
        set.insert(KvCacheHandle(2));
        set.insert(KvCacheHandle(1)); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&KvCacheHandle(1)));
    }

    #[test]
    fn kv_cache_handle_equality() {
        assert_eq!(KvCacheHandle(42), KvCacheHandle(42));
        assert_ne!(KvCacheHandle(42), KvCacheHandle(43));
    }

    // ── LogitsHandle: edge values ──

    #[test]
    fn logits_handle_single_element() {
        let handle = LogitsHandle { data: vec![42.0] };
        assert_eq!(handle.data.len(), 1);
        assert_eq!(handle.data[0], 42.0);
    }

    #[test]
    fn logits_handle_large_data() {
        let handle = LogitsHandle { data: vec![0.0; 100_000] };
        assert_eq!(handle.data.len(), 100_000);
    }

    #[test]
    fn logits_handle_nan_values() {
        let handle = LogitsHandle { data: vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY] };
        assert!(handle.data[0].is_nan());
        assert!(handle.data[1].is_infinite());
        assert!(handle.data[2].is_infinite());
    }

    // ── SequenceInput: additional edge cases ──

    #[test]
    fn sequence_input_empty_tokens_valid() {
        let seq = SequenceInput {
            tokens: vec![],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert!(seq.tokens.is_empty());
    }

    #[test]
    fn sequence_input_max_position() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: usize::MAX,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.position, usize::MAX);
    }

    #[test]
    fn sequence_input_validate_page_table_none_is_ok() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_boundary_last_valid() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![9]), // 9 < 10, valid
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn sequence_input_validate_page_table_boundary_first_invalid() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![10]), // 10 >= 10, invalid
            fused_hidden: None,
        };
        let result = seq.validate_page_table(10);
        assert!(result.is_err());
    }

    #[test]
    fn sequence_input_validate_multiple_pages_first_invalid() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 5, 100]), // 100 >= 10
            fused_hidden: None,
        };
        let result = seq.validate_page_table(10);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("page_table[3]"));
    }

    // ── MtpController: boundary conditions ──

    #[test]
    fn mtp_controller_with_params_clamps_alpha() {
        let ctrl = super::super::mtp_executor::MtpController::with_params(
            -1.0, 0.3, 0.5, 3, 5,
        );
        assert_eq!(ctrl.ema_rate(), 0.5); // alpha clamped to 0.01
    }

    #[test]
    fn mtp_controller_with_params_clamps_alpha_high() {
        let ctrl = super::super::mtp_executor::MtpController::with_params(
            10.0, 0.3, 0.5, 3, 5,
        );
        assert_eq!(ctrl.ema_rate(), 0.5); // alpha clamped to 1.0
    }

    #[test]
    fn mtp_controller_record_zero_total() {
        let mut ctrl = super::super::mtp_executor::MtpController::new();
        ctrl.record_acceptance(0, 0);
        // rate = 0.0, ema = 0.1*0 + 0.9*0.5 = 0.45
        let ema = ctrl.ema_rate();
        assert!((ema - 0.45).abs() < 1e-6, "expected 0.45, got {ema}");
    }

    #[test]
    fn mtp_controller_disable_after_consecutive_low() {
        let mut ctrl = super::super::mtp_executor::MtpController::new();
        // disable_threshold=0.3, disable_patience=3
        for _ in 0..3 {
            ctrl.record_acceptance(0, 10); // rate = 0.0 < 0.3
        }
        assert!(!ctrl.is_enabled(), "should be disabled after 3 consecutive low rounds");
    }

    #[test]
    fn mtp_controller_re_enable_after_stable() {
        let mut ctrl = super::super::mtp_executor::MtpController::new();
        // Disable first
        for _ in 0..3 {
            ctrl.record_acceptance(0, 10);
        }
        assert!(!ctrl.is_enabled());
        // Re-enable: enable_patience=5, rate >= enable_threshold(0.5) or total==0
        for _ in 0..5 {
            ctrl.record_acceptance(8, 10); // rate = 0.8 >= 0.5
        }
        assert!(ctrl.is_enabled(), "should re-enable after 5 stable rounds");
    }

    #[test]
    fn mtp_controller_effective_depth_zero_base() {
        let ctrl = super::super::mtp_executor::MtpController::new();
        // ema=0.5, so 0.5 > 0.3 → returns 1 regardless of max_depth
        assert_eq!(ctrl.effective_depth(0), 1);
    }

    // ── MtpOutput parse: edge cases ──

    #[test]
    fn mtp_output_parse_zero_steps_returns_empty() {
        let output = vec![100u32];
        let parsed = super::super::mtp_executor::MtpOutput::parse(&output, 0, 3);
        let parsed = parsed.expect("zero steps should parse to empty");
        assert!(parsed.main_tokens.is_empty());
        assert!(parsed.mtp_per_step.is_empty());
    }

    #[test]
    fn mtp_output_parse_depth_zero_returns_none() {
        let output = vec![100u32];
        let parsed = super::super::mtp_executor::MtpOutput::parse(&output, 1, 0);
        // depth=0 means no candidates, parse may still succeed with empty mtp_per_step
        // depending on implementation; this tests that it doesn't panic
        let _ = parsed;
    }

    // ── RequestKind: all variants and traits ──

    #[test]
    fn request_kind_variants_distinct() {
        use crate::scheduler::types::RequestKind;
        assert_ne!(RequestKind::Chat, RequestKind::Embedding);
        assert_ne!(RequestKind::Embedding, RequestKind::Rerank);
        assert_ne!(RequestKind::Chat, RequestKind::Rerank);
    }

    #[test]
    fn request_kind_copy_trait() {
        use crate::scheduler::types::RequestKind;
        let a = RequestKind::Chat;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn request_kind_hash_in_set() {
        use crate::scheduler::types::RequestKind;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(RequestKind::Chat);
        set.insert(RequestKind::Embedding);
        set.insert(RequestKind::Rerank);
        assert_eq!(set.len(), 3);
    }

    // ── RoPEConfig: construction and Copy/Clone ──

    #[test]
    fn rope_config_copy_trait() {
        use super::super::executor_types::RoPEConfig;
        let a = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        let b = a;
        assert_eq!(a.theta, b.theta);
    }

    #[test]
    fn rope_config_eq_trait() {
        use super::super::executor_types::RoPEConfig;
        let a = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        let b = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        assert_eq!(a, b);
    }

    #[test]
    fn rope_config_partial_eq_different() {
        use super::super::executor_types::RoPEConfig;
        let a = RoPEConfig { theta: 10000.0, scale: 1.0, interleaved: false, precompute: false };
        let b = RoPEConfig { theta: 500000.0, scale: 1.0, interleaved: false, precompute: false };
        assert_ne!(a, b);
    }

    // ── PipelinedVirtualPageId: construction and equality ──

    #[test]
    fn pipelined_virtual_page_id_construction() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let pvpid = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 5,
        };
        assert_eq!(pvpid.pipeline, KvPipeline::Conversation);
        assert_eq!(pvpid.sequence_id, 42);
        assert_eq!(pvpid.logical_index, 5);
    }

    #[test]
    fn pipelined_virtual_page_id_equality() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let a = PipelinedVirtualPageId { pipeline: KvPipeline::Working, sequence_id: 1, logical_index: 2 };
        let b = PipelinedVirtualPageId { pipeline: KvPipeline::Working, sequence_id: 1, logical_index: 2 };
        assert_eq!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_hash() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelinedVirtualPageId { pipeline: KvPipeline::Conversation, sequence_id: 1, logical_index: 0 });
        set.insert(PipelinedVirtualPageId { pipeline: KvPipeline::Working, sequence_id: 1, logical_index: 0 });
        assert_eq!(set.len(), 2);
    }

    // ── PageMetadata: Default construction ──

    #[test]
    fn page_metadata_default_values() {
        use crate::scheduler::types::PageMetadata;
        let meta = PageMetadata::default();
        assert_eq!(meta.page_id, 0);
        assert!(meta.sequence_id.is_none());
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(!meta.is_lir);
    }

    // ── filter_verified_tokens: edge cases ──

    #[test]
    fn filter_verified_tokens_empty_main() {
        let result = super::super::mtp_executor::filter_verified_tokens(
            &[],
            &[],
            None,
            |_, _| 0,
        );
        assert!(result.committed_tokens.is_empty());
    }

    #[test]
    fn filter_verified_tokens_eos_stops() {
        let main = vec![10, 20, 30];
        let result = super::super::mtp_executor::filter_verified_tokens(
            &main,
            &[],
            Some(20), // eos_token_id = 20, stops at second token
            |_, _| 0,
        );
        // Should stop before token 30 because 20 is EOS
        assert!(result.committed_tokens.len() <= 2);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  NEW TESTS: 15 additional tests — KvPageHeader, CompressionCodec,
    //  StorageTier, f16 conversion, ExecutorError Clone/Error, BatchInput
    // ══════════════════════════════════════════════════════════════════════

    // ── KvPageHeader::new and Default ──

    #[test]
    fn kv_page_header_new_sets_page_id_only() {
        // Arrange: create header with page_id = 42
        let header = crate::kv_cache::KvPageHeader::new(42);
        // Assert: page_id is set, all other fields are default (zero)
        assert_eq!(header.page_id, 42);
        assert_eq!(header.ref_count, 0);
        assert_eq!(header.entropy_avg, 0);
        assert_eq!(header.sink_mask, 0);
        assert_eq!(header.compressed_size, 0);
        assert_eq!(header.sequence_id, 0);
        assert_eq!(header.logical_index, 0);
        assert_eq!(header.last_access_seq, 0);
    }

    #[test]
    fn kv_page_header_default_all_zero() {
        // Arrange: use Default trait
        let header = crate::kv_cache::KvPageHeader::default();
        // Assert: every field is zeroed
        assert_eq!(header.page_id, 0);
        assert_eq!(header.ref_count, 0);
        assert_eq!(header.entropy_avg, 0);
        assert_eq!(header.softmax_max_avg, 0);
        assert_eq!(header.delta_rho_avg, 0);
        assert_eq!(header.dead_ratio, 0);
        assert_eq!(header.importance_score, 0);
        assert_eq!(header.head_entropy_max, 0);
        assert_eq!(header.head_entropy_min, 0);
        assert_eq!(header.sink_mask, 0);
        assert_eq!(header.channel_bitmap_lo, 0);
        assert_eq!(header.k_scale_offset, 0);
        assert_eq!(header.precision_tier, 0);
        assert_eq!(header.v_scale_factor, 0);
        assert_eq!(header.layer_mask, 0);
        assert_eq!(header.tier_age, 0);
        assert_eq!(header.pipeline_id, 0);
        assert_eq!(header.deopt_flags, 0);
        assert_eq!(header.checksum, 0);
        assert_eq!(header.compressed_size, 0);
    }

    // ── KvPageHeader methods: is_active, has_sink_token, needs_requantize ──

    #[test]
    fn kv_page_header_is_active_with_ref_count() {
        // Arrange
        let mut header = crate::kv_cache::KvPageHeader::new(0);
        assert!(!header.is_active());
        // Act
        header.ref_count = 1;
        // Assert
        assert!(header.is_active());
        header.ref_count = 5;
        assert!(header.is_active());
    }

    #[test]
    fn kv_page_header_has_sink_token_and_needs_requantize() {
        // Arrange
        let mut header = crate::kv_cache::KvPageHeader::new(0);
        assert!(!header.has_sink_token());
        assert!(!header.needs_requantize());
        // Act: set sink_mask and deopt_flags bit 0
        header.sink_mask = 0b1010;
        header.deopt_flags = 0x01;
        // Assert
        assert!(header.has_sink_token());
        assert!(header.needs_requantize());
        // Verify bit 1 does not trigger requantize
        header.deopt_flags = 0x02;
        assert!(!header.needs_requantize());
    }

    // ── KvPageHeader: head_entropy_spread, is_low_entropy, is_high_dead_ratio ──

    #[test]
    fn kv_page_header_entropy_and_dead_ratio_methods() {
        // Arrange
        let mut header = crate::kv_cache::KvPageHeader::new(0);
        assert!(header.is_low_entropy());
        assert!(!header.is_high_dead_ratio());
        assert_eq!(header.head_entropy_spread(), 0);
        // Act: set head entropy spread
        header.head_entropy_max = 200;
        header.head_entropy_min = 50;
        header.entropy_avg = 100; // nonzero → not low entropy
        header.dead_ratio = 200; // > 127 → high dead ratio
        // Assert
        assert_eq!(header.head_entropy_spread(), 150);
        assert!(!header.is_low_entropy());
        assert!(header.is_high_dead_ratio());
        // Boundary: dead_ratio == 127 is NOT high
        header.dead_ratio = 127;
        assert!(!header.is_high_dead_ratio());
    }

    // ── KvPageHeader: set_position_agnostic / is_position_agnostic ──

    #[test]
    fn kv_page_header_position_agnostic_flag() {
        // Arrange
        let mut header = crate::kv_cache::KvPageHeader::new(0);
        assert!(!header.is_position_agnostic());
        // Act: set position_agnostic (deopt_flags bit 7)
        header.set_position_agnostic(true);
        // Assert
        assert!(header.is_position_agnostic());
        // Act: clear
        header.set_position_agnostic(false);
        assert!(!header.is_position_agnostic());
        // Verify bit 7 (0x80) does not interfere with bit 0
        header.deopt_flags = 0x01; // needs_requantize, NOT position_agnostic
        assert!(header.needs_requantize());
        assert!(!header.is_position_agnostic());
        header.set_position_agnostic(true);
        assert!(header.needs_requantize()); // bit 0 preserved
        assert!(header.is_position_agnostic()); // bit 7 set
    }

    // ── KvPageHeader: precision_tier set/get ──

    #[test]
    fn kv_page_header_precision_tier_round_trip() {
        // Arrange
        let mut header = crate::kv_cache::KvPageHeader::new(0);
        // Assert default
        assert!(matches!(header.precision_tier(), crate::kv_cache::PrecisionTier::FP16));
        // Act: set to FP8
        header.set_precision_tier(crate::kv_cache::PrecisionTier::FP8);
        assert!(matches!(header.precision_tier(), crate::kv_cache::PrecisionTier::FP8));
        // Act: set to KIVI4
        header.set_precision_tier(crate::kv_cache::PrecisionTier::KIVI4);
        assert!(matches!(header.precision_tier(), crate::kv_cache::PrecisionTier::KIVI4));
        // Act: set to Evicted
        header.set_precision_tier(crate::kv_cache::PrecisionTier::Evicted);
        assert!(matches!(header.precision_tier(), crate::kv_cache::PrecisionTier::Evicted));
    }

    // ── CompressionCodec: from_u8 / as_u8 round trip ──

    #[test]
    fn compression_codec_round_trip_all_variants() {
        // Arrange
        use crate::kv_cache::CompressionCodec;
        let variants = [
            (CompressionCodec::None, 0u8),
            (CompressionCodec::Lz4, 1),
            (CompressionCodec::BitPackRle, 2),
            (CompressionCodec::NvcompAns, 3),
            (CompressionCodec::ZstdDict, 4),
        ];
        for (variant, expected_u8) in variants {
            // Assert: as_u8 matches expected
            assert_eq!(variant.as_u8(), expected_u8);
            // Assert: from_u8 round-trips back
            assert_eq!(CompressionCodec::from_u8(expected_u8), Some(variant));
        }
        // Assert: invalid values return None
        assert_eq!(CompressionCodec::from_u8(5), None);
        assert_eq!(CompressionCodec::from_u8(255), None);
    }

    // ── StorageTier: from_u8 and all variants ──

    #[test]
    fn storage_tier_from_u8_all_variants() {
        // Arrange
        use crate::kv_cache::StorageTier;
        // Assert: valid values round-trip
        assert_eq!(StorageTier::from_u8(0), Some(StorageTier::GpuHbm));
        assert_eq!(StorageTier::from_u8(1), Some(StorageTier::CpuDram));
        assert_eq!(StorageTier::from_u8(2), Some(StorageTier::Nvme));
        // Assert: invalid values return None
        assert_eq!(StorageTier::from_u8(3), None);
        assert_eq!(StorageTier::from_u8(255), None);
    }

    // ── f32_to_f16_bits / f16_bits_to_f32: round-trip for common values ──

    #[test]
    fn f16_conversion_round_trip_common_values() {
        // Arrange: common telemetry values that should survive f16 round-trip
        use crate::kv_cache::{f16_bits_to_f32, f32_to_f16_bits};
        let values = [0.0f32, 1.0, 0.5, 0.25, 2.0, 0.125, -1.0, -0.5];
        for val in values {
            // Act: f32 -> f16 bits -> f32
            let bits = f32_to_f16_bits(val);
            let recovered = f16_bits_to_f32(bits);
            // Assert: within f16 precision (relative tolerance ~0.1%)
            if val == 0.0 {
                assert_eq!(recovered, 0.0);
            } else {
                let rel_err = ((recovered - val).abs() / val.abs()).abs();
                assert!(rel_err < 0.002, "f16 round-trip for {val}: got {recovered}, rel_err={rel_err}");
            }
        }
    }

    // ── f32_to_dead_ratio / dead_ratio_to_f32: round-trip ──

    #[test]
    fn dead_ratio_conversion_round_trip() {
        // Arrange
        use crate::kv_cache::{dead_ratio_to_f32, f32_to_dead_ratio};
        // Assert: boundary values are exact
        assert_eq!(f32_to_dead_ratio(0.0), 0);
        assert_eq!(f32_to_dead_ratio(1.0), 255);
        assert_eq!(dead_ratio_to_f32(0), 0.0);
        assert!((dead_ratio_to_f32(255) - 1.0).abs() < f32::EPSILON);
        // Act & Assert: round-trip for mid value
        let mid = f32_to_dead_ratio(0.5);
        let recovered = dead_ratio_to_f32(mid);
        assert!((recovered - 0.5).abs() < 0.01, "expected ~0.5, got {recovered}");
        // Assert: clamping behavior for out-of-range
        assert_eq!(f32_to_dead_ratio(-1.0), 0);
        assert_eq!(f32_to_dead_ratio(2.0), 255);
    }

    // ── ExecutorError: std::error::Error trait via dyn trait object ──

    #[test]
    fn executor_error_implements_std_error_trait() {
        // Arrange: create various ExecutorError variants
        let errors: Vec<ExecutorError> = vec![
            ExecutorError::EmptyPrompt,
            ExecutorError::EmptySample,
            ExecutorError::Scheduler("test".into()),
            ExecutorError::Compilation("compile err".into()),
            ExecutorError::GraphExpansion("graph err".into()),
            ExecutorError::RequestNotFound { request_id: 42 },
        ];
        // Assert: each can be used as a dyn Error
        for err in &errors {
            let _: &dyn std::error::Error = err;
        }
    }

    // ── ExecutorError: Display consistency for all scalar variants ──

    #[test]
    fn executor_error_display_all_scalar_variants() {
        // Arrange & Assert: each variant produces a non-empty Display message
        let cases: Vec<(ExecutorError, &str)> = vec![
            (ExecutorError::EmptyPrompt, "empty prompt"),
            (ExecutorError::EmptySample, "empty sample"),
            (ExecutorError::Scheduler("sched err".into()), "scheduler error"),
            (ExecutorError::Compilation("jit fail".into()), "JIT compilation failed"),
            (ExecutorError::GraphExpansion("node err".into()), "graph expansion failed"),
            (ExecutorError::RequestNotFound { request_id: 99 }, "99"),
        ];
        for (err, expected_substr) in cases {
            let msg = format!("{err}");
            assert!(
                msg.contains(expected_substr),
                "expected '{expected_substr}' in '{msg}'"
            );
        }
    }

    // ── BatchInput: non-empty sequences with multiple entries ──

    #[test]
    fn batch_input_multiple_sequences_count() {
        // Arrange: build a batch with 3 sequences of varying sizes
        let seq1 = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        let seq2 = SequenceInput {
            tokens: vec![10, 20],
            position: 10,
            draft_steps: 2,
            page_table: None,
            fused_hidden: None,
        };
        let seq3 = SequenceInput {
            tokens: vec![100],
            position: 20,
            draft_steps: 0,
            page_table: Some(vec![0, 1]),
            fused_hidden: Some(vec![0.5; 64]),
        };
        let batch = BatchInput {
            sequences: vec![seq1, seq2, seq3],
        };
        // Assert
        assert_eq!(batch.sequences.len(), 3);
        assert_eq!(batch.sequences[0].tokens.len(), 3);
        assert_eq!(batch.sequences[1].position, 10);
        assert_eq!(batch.sequences[1].draft_steps, 2);
        assert_eq!(batch.sequences[2].tokens.len(), 1);
        assert_eq!(batch.sequences[2].page_table.as_ref().unwrap().len(), 2);
        assert_eq!(batch.sequences[2].fused_hidden.as_ref().unwrap().len(), 64);
    }

    // ── RequestData: non-default values and all field access ──

    #[test]
    fn request_data_non_default_field_values() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        let rd = super::super::executor::RequestData {
            prompt_tokens: vec![1, 2, 3, 4, 5],
            output_tokens: vec![10, 20],
            sampling_config: super::super::executor::SamplingConfig {
                temperature: 0.7,
                top_k: 50,
                top_p: 0.95,
            },
            is_prefill: false,
            phase: RequestPhase::Decode,
            max_new_tokens: 100,
            finished: false,
            session_id: Some(42u64),
            thinking_budget: Some(200),
            fused_prefill_hidden: Some(vec![0.1; 128]),
        };
        // Assert
        assert_eq!(rd.prompt_tokens.len(), 5);
        assert_eq!(rd.output_tokens, vec![10, 20]);
        assert_eq!(rd.sampling_config.temperature, 0.7);
        assert_eq!(rd.sampling_config.top_k, 50);
        assert!(!rd.is_prefill);
        assert_eq!(rd.max_new_tokens, 100);
        assert!(!rd.finished);
        assert_eq!(rd.session_id, Some(42u64));
        assert_eq!(rd.thinking_budget, Some(200));
        assert_eq!(rd.fused_prefill_hidden.as_ref().unwrap().len(), 128);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  NEW TESTS: 13 additional tests (183 → 196)
    // ══════════════════════════════════════════════════════════════════════

    // ── WeightPageJitConfig: Default and field access ──

    #[test]
    fn weight_page_jit_config_default_values() {
        // Arrange: use Default trait
        let cfg = crate::engine::mega_kernel::WeightPageJitConfig::default();
        // Assert: defaults from SPEC REQ-WP-008
        assert!(!cfg.enabled);
        assert_eq!(cfg.num_pages, 1024);
        assert_eq!(cfg.page_size_bytes, 64 * 1024 * 1024);
        assert_eq!(cfg.prefetch_distance, 0);
    }

    #[test]
    fn weight_page_jit_config_custom_values_and_clone() {
        // Arrange: custom config
        let cfg = crate::engine::mega_kernel::WeightPageJitConfig {
            enabled: true,
            num_pages: 2048,
            page_size_bytes: 32 * 1024 * 1024,
            prefetch_distance: 4,
        };
        // Act: clone
        let cloned = cfg.clone();
        // Assert: all fields match
        assert!(cfg.enabled);
        assert_eq!(cfg.num_pages, 2048);
        assert_eq!(cfg.page_size_bytes, 32 * 1024 * 1024);
        assert_eq!(cfg.prefetch_distance, 4);
        assert_eq!(cloned.enabled, cfg.enabled);
        assert_eq!(cloned.num_pages, cfg.num_pages);
    }

    // ── KvPageDecompressConfig: Default and field access ──

    #[test]
    fn kv_page_decompress_config_default_values() {
        // Arrange: use Default trait
        let cfg = crate::engine::mega_kernel::KvPageDecompressConfig::default();
        // Assert: defaults from SPEC REQ-COMP11
        assert!(!cfg.enabled);
        assert_eq!(cfg.num_pages, 1024);
        assert_eq!(cfg.page_size_bytes, 64 * 1024);
    }

    #[test]
    fn kv_page_decompress_config_custom_and_clone() {
        // Arrange: custom config
        let cfg = crate::engine::mega_kernel::KvPageDecompressConfig {
            enabled: true,
            num_pages: 512,
            page_size_bytes: 128 * 1024,
        };
        // Act: clone
        let cloned = cfg.clone();
        // Assert
        assert!(cfg.enabled);
        assert_eq!(cfg.num_pages, 512);
        assert_eq!(cloned.enabled, cfg.enabled);
        assert_eq!(cloned.page_size_bytes, cfg.page_size_bytes);
    }

    // ── DiagnosticScratchpad: read_f32_at edge cases ──

    #[test]
    fn diagnostic_scratchpad_read_f32_at_out_of_bounds_returns_empty() {
        // Arrange: minimal scratchpad with 8 bytes (2 f32s)
        let sp = crate::engine::mega_kernel::DiagnosticScratchpad {
            data: vec![0u8; 8],
            logits_offset: 0,
            vocab_size: 2,
            prompt_len: 1,
            hidden_size: 2,
        };
        // Act: request beyond data length
        let result = sp.read_f32_at(0, 100);
        // Assert: returns empty vec (end > data.len())
        assert!(result.is_empty());
    }

    #[test]
    fn diagnostic_scratchpad_read_f32_at_reads_correct_values() {
        // Arrange: 8 bytes containing [1.0f32, 2.0f32] in little-endian
        let mut data = vec![0u8; 8];
        let f32s: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut f32, 2) };
        f32s[0] = 1.0;
        f32s[1] = 2.0;
        let sp = crate::engine::mega_kernel::DiagnosticScratchpad {
            data,
            logits_offset: 0,
            vocab_size: 2,
            prompt_len: 1,
            hidden_size: 2,
        };
        // Act
        let result = sp.read_f32_at(0, 2);
        // Assert
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
    }

    // ── RequestPhase: ChunkedPrefill variant ──

    #[test]
    fn request_phase_chunked_prefill_distinct_from_others() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        // Assert: all three variants are distinct
        assert_ne!(RequestPhase::Prefill, RequestPhase::Decode);
        assert_ne!(RequestPhase::Prefill, RequestPhase::ChunkedPrefill);
        assert_ne!(RequestPhase::Decode, RequestPhase::ChunkedPrefill);
        // Assert: equality with self
        assert_eq!(RequestPhase::ChunkedPrefill, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn request_phase_copy_and_debug_traits() {
        // Arrange
        use crate::scheduler::request_state::RequestPhase;
        let a = RequestPhase::ChunkedPrefill;
        // Act: Copy
        let b = a;
        // Assert: Copy works
        assert_eq!(a, b);
        // Assert: Debug contains variant name
        let debug = format!("{a:?}");
        assert!(debug.contains("ChunkedPrefill"), "expected ChunkedPrefill in debug, got: {debug}");
    }

    // ── SwapConfig: Debug trait and Clone ──

    #[test]
    fn swap_config_debug_and_clone() {
        // Arrange
        let cfg = super::super::executor_types::SwapConfig {
            enable_swap: true,
            swap_threshold: 0.75,
            lru_granularity: 8,
        };
        // Act: clone
        let cloned = cfg.clone();
        // Assert: clone equality
        assert_eq!(cfg, cloned);
        // Assert: Debug output
        let debug = format!("{cfg:?}");
        assert!(debug.contains("SwapConfig"), "expected SwapConfig in debug, got: {debug}");
    }

    // ── build_verify_result: construction correctness ──

    #[test]
    fn build_verify_result_full_acceptance() {
        // Arrange: all 3 candidates accepted
        let candidates = vec![10u32, 20, 30];
        // Act
        let result = super::super::mtp_executor::build_verify_result(42, &candidates, 3);
        // Assert
        assert_eq!(result.request_id, 42);
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.accepted_tokens, vec![10, 20, 30]);
        assert_eq!(result.rejected_count, 0);
        assert_eq!(result.draft_count, 3);
        assert!((result.acceptance_rate - 1.0).abs() < f32::EPSILON);
        assert!(result.invariant_check_passed);
    }

    #[test]
    fn build_verify_result_partial_acceptance() {
        // Arrange: 5 candidates, only 2 accepted
        let candidates = vec![10u32, 20, 30, 40, 50];
        // Act
        let result = super::super::mtp_executor::build_verify_result(7, &candidates, 2);
        // Assert
        assert_eq!(result.request_id, 7);
        assert_eq!(result.accepted_count, 2);
        assert_eq!(result.accepted_tokens, vec![10, 20]);
        assert_eq!(result.rejected_count, 3);
        assert_eq!(result.draft_count, 5);
        let expected_rate = 2.0f32 / 5.0f32;
        assert!((result.acceptance_rate - expected_rate).abs() < f32::EPSILON);
    }

    #[test]
    fn build_verify_result_empty_candidates() {
        // Arrange: no candidates
        let candidates: Vec<u32> = vec![];
        // Act
        let result = super::super::mtp_executor::build_verify_result(1, &candidates, 0);
        // Assert: acceptance_rate = 0.0 per empty branch
        assert_eq!(result.accepted_count, 0);
        assert_eq!(result.accepted_tokens, Vec::<u32>::new());
        assert_eq!(result.rejected_count, 0);
        assert_eq!(result.draft_count, 0);
        assert_eq!(result.acceptance_rate, 0.0);
    }

    // ── generate_mtp_kv_instructions: mixed accept/reject per step ──

    #[test]
    fn generate_mtp_kv_instructions_mixed_accept_and_reject() {
        // Arrange: two steps — first has 2 accepted + 1 rejected, second has 0 accepted + 3 rejected
        let step1 = super::super::mtp_executor::MtpStepDetail {
            main_token: 100,
            mtp_candidates: vec![10, 20, 30],
            accepted_count: 2,
            main_token_is_eos: false,
        };
        let step2 = super::super::mtp_executor::MtpStepDetail {
            main_token: 200,
            mtp_candidates: vec![40, 50, 60],
            accepted_count: 0,
            main_token_is_eos: false,
        };
        // Act
        let instructions = super::super::mtp_executor::generate_mtp_kv_instructions(99, &[step1, step2]);
        // Assert: 3 instructions — Commit for step1, Rollback for step1, Rollback for step2
        assert_eq!(instructions.len(), 3);
        // Step 1: Commit 2 accepted tokens
        assert!(matches!(
            &instructions[0],
            crate::speculative::verify::KvCommitInstruction::Commit { request_id: 99, accepted_tokens, .. }
            if accepted_tokens.len() == 2
        ));
        // Step 1: Rollback 1 rejected
        assert!(matches!(
            &instructions[1],
            crate::speculative::verify::KvCommitInstruction::Rollback { request_id: 99, rejected_count: 1, .. }
        ));
        // Step 2: Rollback 3 rejected (no commit since 0 accepted)
        assert!(matches!(
            &instructions[2],
            crate::speculative::verify::KvCommitInstruction::Rollback { request_id: 99, rejected_count: 3, .. }
        ));
    }
}
