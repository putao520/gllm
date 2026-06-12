//! Step-loop methods for Executor — decomposition from executor.rs.
//!
//! This file contains the per-step decode loop and all methods called within it:
//! batch scheduling, sequence building, MoE dispatch, post-forward processing,
//! KV optimization, epilogue decisions, speculative decode, observability, and
//! finalization. Split from executor.rs to keep it under the 2000-line limit.

use std::collections::HashMap;

use crate::compat::backend_trait::{Backend, Element};
use crate::scheduler::types::RequestId;

use super::executor::{
    BackendError, BatchInput, Executor, ExecutorError, ExecutorResult,
    LogitsHandle, SequenceInput,
};
use crate::scheduler::batcher::{BatchAction, BatchResult};
use crate::scheduler::vllm2024::{AdaptiveChunkPolicy, Scheduler2024Config};
use crate::scheduler::{
    PrefillPlan, ScheduledBatch, SequenceTelemetry, Tier, VirtualPageId,
};

// ---------------------------------------------------------------------------
// Helper functions (used exclusively within step-loop methods)
// ---------------------------------------------------------------------------

/// Extract the top-k token IDs from a logits vector, sorted by descending probability.
///
/// Used by speculative decode to seed the SpecTree with adapter top-k candidates.
fn extract_top_k_token_ids(logits: &[f32], k: usize) -> Vec<u32> {
    if logits.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(k).map(|(id, _)| id).collect()
}

/// Return the token ID with the highest logit (greedy argmax).
///
/// Used by speculative decode verification to determine the target model's prediction.
fn argmax_token(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Shannon entropy of a logits distribution (in nats).
fn shannon_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() {
        return 0.0;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    let log_sum = exp_sum.ln();
    let mut entropy = 0.0f32;
    for &x in logits {
        let log_p = (x - max) - log_sum;
        let p = log_p.exp();
        if p > 0.0 {
            entropy -= p * log_p;
        }
    }
    entropy
}

// ---------------------------------------------------------------------------
// BuildSequencesOutput — intermediate struct for build_sequences()
// ---------------------------------------------------------------------------

struct BuildSequencesOutput {
    sequences: Vec<SequenceInput>,
    request_indices: Vec<RequestId>,
    batch_results: Vec<BatchResult>,
}

// ---------------------------------------------------------------------------
// Step-loop impl block
// ---------------------------------------------------------------------------

impl<B: Backend<E> + 'static, E: Element> Executor<B, E> {
    pub fn step(&mut self) -> ExecutorResult<()> {
        self.drain_swap_completions();
        self.check_memory_pressure()?;

        // 0. Observability: Capture System State
        let pressure_result = self
            .backend
            .get_memory_pressure()
            .map_err(|e| format!("{e}"));
        self.observability.capture_state(pressure_result, &self.dispatch);
        let system_state = self.observability.observer.last_state;

        let decision = self.dispatch.policy.decide(&system_state);

        let (batch, spec_enabled) = self.schedule_batch(decision)?;
        if batch.requests.is_empty() {
            return Ok(());
        }

        let mut batch_results = Vec::new();

        let (_, l1_ratio, concurrent) = self.plan_prefill_sessions(&batch.requests)?;

        let dispatch_plan = self.prepare_sub_batch_dispatch(&batch.requests);

        let BuildSequencesOutput { sequences, request_indices: req_indices, batch_results: fail_results } =
            self.build_sequences(batch)?;
        let request_indices = req_indices;
        batch_results.extend(fail_results);

        if sequences.is_empty() {
            return Ok(());
        }

        let total_pages = self.dispatch.scheduler.total_pages();
        for seq in &sequences {
            if let Err(e) = seq.validate_page_table(total_pages) {
                log::error!("executor: page table validation failed: {e}");
                return Err(ExecutorError::Backend(BackendError::Other(e)));
            }
        }

        let batch_input = BatchInput { sequences };

        let moe_dispatch_plan = self.moe_dispatch_and_prefetch();

        self.update_histogram_and_compact(&batch_input, l1_ratio, concurrent, &dispatch_plan);

        // 4. Run Backend Forward
        let (logits_list, batch_sparsity, batch_telemetry) =
            self.run_batch_forward(&batch_input)?;

        self.advance_prefetch_pipeline(&moe_dispatch_plan);

        {
            let num_layers = self.kv.kv_cache_config.num_layers();
            let kv_dim = self.kv.kv_cache_config.num_heads() * self.model_ctx.geometry.head_dim;
            self.compute.record_turboquant_scales(num_layers, kv_dim);
        }

        let total_tokens = self.validate_and_count_tokens(&logits_list, &batch_telemetry, &request_indices, &batch_input)?;

        self.update_step_observability(&logits_list, batch_sparsity);

        self.push_telemetry_to_director(&batch_telemetry);

        let page_entropies = self.execute_post_forward(
            &logits_list,
            &batch_telemetry,
            &request_indices,
            &batch_input,
            spec_enabled,
            total_tokens,
            &mut batch_results,
        )?;

        self.step_finalize(&batch_results, &page_entropies);

        Ok(())
    }

    /// §19 KV-OPT-003: Execute KV cache optimization after each decode step.
    ///
    /// Reads page headers from the optimizer's tier decisions (computed earlier
    /// in `step()` via `optimize_with_cross_layer_reuse`), applies requantize
    /// for pages whose precision tier changed, and logs results.
    ///
    /// Only processes pages for **decode** requests — prefill pages must remain
    /// at FP16 until their first decode step.
    pub fn optimize_kv_cache(&mut self) {
        use crate::scheduler::kv_optimizer;

        // Collect page IDs for decode (non-prefill) requests that have page tables.
        let mut decode_page_ids: Vec<u32> = Vec::new();
        for (&req_id, req) in &self.dispatch.requests {
            if req.is_prefill {
                continue;
            }
            if let Some(page_table) = self.dispatch.scheduler.get_page_table(req_id) {
                decode_page_ids.extend(page_table);
            }
        }
        if decode_page_ids.is_empty() {
            return;
        }
        // Deduplicate (pages may be shared across requests via KvPrefixIndex).
        decode_page_ids.sort_unstable();
        decode_page_ids.dedup();

        let num_kv_heads = self.model_ctx.geometry.num_kv_heads;
        let num_layers = self.model_ctx.geometry.num_layers;
        let page_size = self.kv.kv_cache_config.page_size;
        let head_dim = self.model_ctx.geometry.head_dim;

        let mut total_requantized = 0usize;
        let mut quant_buffer = Vec::new();

        // For each decode page, build a per-layer header from the telemetry
        // aggregator and apply the optimizer's tier decision + requantize.
        for &page_id in &decode_page_ids {
            for layer_idx in 0..num_layers {
                let mut header = crate::kv_cache::KvPageHeader::new(page_id);
                header.ref_count = 1;
                // Populate telemetry fields from the aggregator (per-layer decay).
                let depth_ratio = layer_idx as f32 / num_layers.max(1) as f32;
                let entropy_decay = 1.0 - 0.2 * depth_ratio;
                let head_spread_decay = 1.0 - 0.3 * depth_ratio;
                let entropy = self.compute.telemetry_aggregator.output_entropy() * entropy_decay;
                let softmax_max = if entropy < 1.0 { 0.85 } else { 0.3 };
                let delta_rho =
                    self.compute.telemetry_aggregator.residual_delta_rho() * (1.0 - 0.1 * depth_ratio);
                header.entropy_avg = crate::kv_cache::f32_to_f16_bits(entropy);
                header.softmax_max_avg = crate::kv_cache::f32_to_f16_bits(softmax_max);
                header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(delta_rho);
                header.head_entropy_max = (entropy * 25.5 * head_spread_decay).min(255.0) as u8;
                header.head_entropy_min = (entropy * 10.0 * head_spread_decay).min(255.0) as u8;

                // Compute importance + decide tier.
                let importance = self.kv.kv_optimizer.write_importance(&mut header);
                let target_tier = self.kv.kv_optimizer.decide_tier(&header, layer_idx);
                let current_tier = header.precision_tier();
                let _ = importance;

                // Requantize if tier changed and we have paged KV pool data.
                if target_tier != current_tier {
                    if let Some(ref mut pool) = self.kv.paged_kv_pool {
                        let elem_bytes = self.kv.kv_cache_config.dtype_size();
                        let is_mla = self.model_ctx.geometry.is_mla();
                        let kv_dim = self.model_ctx.geometry.kv_dim();
                        let layer_bytes = if is_mla {
                            page_size * kv_dim * elem_bytes
                        } else {
                            2 * num_kv_heads * page_size * head_dim * elem_bytes
                        };
                        let page_offset =
                            page_id as usize * pool.page_stride() + layer_idx * layer_bytes;
                        if page_offset + layer_bytes <= pool.total_bytes() {
                            let kv_data = unsafe {
                                std::slice::from_raw_parts_mut(
                                    pool.as_mut_ptr().add(page_offset),
                                    layer_bytes,
                                )
                            };
                            let saved = kv_optimizer::requantize_page(
                                kv_data,
                                elem_bytes,
                                current_tier,
                                target_tier,
                                &mut quant_buffer,
                                num_kv_heads,
                                page_size,
                                head_dim,
                            );
                            if saved > 0 {
                                total_requantized += 1;
                            }
                        }
                    }
                }
            }
        }

        if total_requantized > 0 {
            log::debug!(
                "executor: §19 KV-OPT-003 optimize_kv_cache: requantized {} page-layers across {} decode pages",
                total_requantized,
                decode_page_ids.len(),
            );
        }
    }

    fn speculative_decode(
        &mut self,
        spec_enabled: bool,
        batch_results: &[BatchResult],
        request_indices: &[RequestId],
        logits_list: &[LogitsHandle],
    ) -> usize {
        if !spec_enabled || !self.inference.spec_decoding.is_active() {
            return 0;
        }

        let continuing_decode_reqs: Vec<RequestId> = batch_results
            .iter()
            .filter(|r| r.action == BatchAction::Continue)
            .map(|r| r.request_id)
            .collect();

        if continuing_decode_reqs.is_empty() {
            return 0;
        }

        let first_req_idx = request_indices
            .iter()
            .position(|&rid| rid == continuing_decode_reqs[0]);
        let Some(logits_idx) = first_req_idx else {
            return 0;
        };

        let top_k_tokens = extract_top_k_token_ids(&logits_list[logits_idx].data, 3);
        let all_tokens: Vec<u32> = {
            let req = &self.dispatch.requests[&continuing_decode_reqs[0]];
            let mut tokens = req.prompt_tokens.clone();
            tokens.extend_from_slice(&req.output_tokens);
            tokens
        };

        let tree = self.inference.spec_decoding.draft_phase(&top_k_tokens, &all_tokens);
        let spine_tokens = tree.spine_token_ids();
        if spine_tokens.len() <= 1 {
            return 0;
        }

        let mut verify_sequences = Vec::with_capacity(continuing_decode_reqs.len());
        let mut verify_req_indices = Vec::with_capacity(continuing_decode_reqs.len());
        for &req_id in &continuing_decode_reqs {
            let req = &self.dispatch.requests[&req_id];
            if req.finished {
                continue;
            }
            let position = req.prompt_tokens.len() + req.output_tokens.len();
            verify_sequences.push(SequenceInput {
                tokens: spine_tokens.clone(),
                position,
                draft_steps: spine_tokens.len(),
                page_table: self.dispatch.scheduler.get_page_table(req_id),
                fused_hidden: None,
            });
            verify_req_indices.push(req_id);
        }

        if verify_sequences.is_empty() {
            return 0;
        }

        let verify_input = BatchInput { sequences: verify_sequences };
        let mut spec_extra_tokens = 0usize;

        match self.run_batch_forward(&verify_input) {
            Ok((verify_logits, _verify_sparsity, _verify_telemetry)) => {
                let mut seq_results = Vec::with_capacity(verify_req_indices.len());
                for (vi, &req_id) in verify_req_indices.iter().enumerate() {
                    let target_tokens: Vec<u32> = if vi < verify_logits.len() {
                        vec![argmax_token(&verify_logits[vi].data)]
                    } else {
                        Vec::new()
                    };

                    let seq_result = crate::speculative::verify::SequenceVerifyResult::verify_spine(
                        req_id, &spine_tokens, &target_tokens,
                    );

                    if seq_result.accepted_count > 0 {
                        if let Some(req) = self.dispatch.requests.get_mut(&req_id) {
                            let eos = self.model_ctx.model_config.eos_token_id;
                            for &tok in &seq_result.accepted_tokens {
                                if req.finished { break; }
                                req.output_tokens.push(tok);
                                spec_extra_tokens += 1;
                                if eos.is_some_and(|id| id == tok)
                                    || req.output_tokens.len() >= req.max_new_tokens
                                {
                                    req.finished = true;
                                }
                            }
                        }
                    }
                    seq_results.push(seq_result);
                }

                let verify_result = crate::speculative::verify::VerifyResult::from_sequence_results(seq_results);
                let kv_instructions =
                    crate::speculative::verify::generate_kv_commit_instructions(&verify_result);
                for instr in &kv_instructions {
                    match instr {
                        crate::speculative::verify::KvCommitInstruction::Commit {
                            request_id, accepted_tokens, ..
                        } => {
                            log::debug!(
                                "executor: §17.4 spec KV commit req={} accepted={} tokens",
                                request_id, accepted_tokens.len(),
                            );
                        }
                        crate::speculative::verify::KvCommitInstruction::Rollback {
                            request_id, rejected_count, ..
                        } => {
                            log::debug!(
                                "executor: §17.4 spec KV rollback req={} rejected={} tokens",
                                request_id, rejected_count,
                            );
                        }
                    }
                }
                self.inference.spec_decoding.verify_phase(&verify_result);
                log::debug!(
                    "executor: §17.1 spec decode complete — drafted={}, accepted={}, rate={:.2}",
                    verify_result.total_draft_tokens,
                    verify_result.total_accepted_tokens,
                    verify_result.avg_acceptance_rate,
                );
            }
            Err(e) => {
                log::warn!("executor: §17.1 spec verify forward failed: {}", e);
            }
        }

        spec_extra_tokens
    }

    fn execute_post_forward(
        &mut self,
        logits_list: &[LogitsHandle],
        batch_telemetry: &[SequenceTelemetry],
        request_indices: &[RequestId],
        batch_input: &BatchInput,
        spec_enabled: bool,
        total_tokens: usize,
        batch_results: &mut Vec<BatchResult>,
    ) -> ExecutorResult<HashMap<usize, f32>> {
        let page_size = self.dispatch.scheduler.page_size().max(1);
        let page_entropies = HashMap::new();

        let epilogue_summary = self.compute.epilogue_subsystem.ingest_and_decide(batch_telemetry);
        if epilogue_summary.compact_required {
            log::debug!(
                "executor: Epilogue compact required (waste={:.2}%)",
                epilogue_summary.waste_ratio * 100.0
            );
        }

        self.run_kv_optimizer(batch_telemetry);
        self.apply_epilogue_decisions(&epilogue_summary, request_indices);

        batch_results.extend(self.process_results(
            logits_list,
            batch_telemetry,
            request_indices,
            page_size,
            batch_input,
        )?);

        let spec_extra_tokens = self.speculative_decode(
            spec_enabled,
            batch_results,
            request_indices,
            logits_list,
        );

        {
            let slot = self.kv.kv_cache_slot;
            if let Some(kv_cache) = self.kv.kv_cache.as_mut() {
                let active = kv_cache.slot_mut(slot);
                active.advance(total_tokens + spec_extra_tokens)?;
            }
        }

        Ok(page_entropies)
    }

    fn advance_prefetch_pipeline(&mut self, moe_dispatch_plan: &Option<crate::moe::dispatch::MoeDispatchPlan>) {
        if let Some(ref mut pipeline) = self.inference.prefetch_pipeline {
            let centroid_tokens: Vec<usize> = moe_dispatch_plan
                .as_ref()
                .map(|p| p.gpu_experts.iter().map(|e| e.expert_idx).collect())
                .unwrap_or_default();
            let kv_block_size = self.kv.kv_cache_config.num_heads() * self.model_ctx.geometry.head_dim * 2;
            pipeline.advance_layer(&centroid_tokens, kv_block_size);
        }
    }

    fn validate_and_count_tokens(
        &mut self,
        logits_list: &[LogitsHandle],
        batch_telemetry: &[SequenceTelemetry],
        request_indices: &[RequestId],
        batch_input: &BatchInput,
    ) -> ExecutorResult<usize> {
        if logits_list.len() != request_indices.len()
            || batch_telemetry.len() != request_indices.len()
        {
            return Err(ExecutorError::Backend(BackendError::Other(format!(
                "Backend returned {} logits and {} telemetries for {} requests",
                logits_list.len(),
                batch_telemetry.len(),
                request_indices.len()
            ))));
        }
        let mut total_tokens = 0;
        for seq in &batch_input.sequences {
            total_tokens += seq.tokens.len();
        }
        Ok(total_tokens)
    }

    fn moe_dispatch_and_prefetch(
        &mut self,
    ) -> Option<crate::moe::dispatch::MoeDispatchPlan> {
        let moe_dispatch_plan = if let (Some(ref dispatcher), Some(ref thermal)) =
            (&self.inference.moe_dispatcher, &self.inference.moe_thermal)
        {
            let heat_levels: Vec<crate::moe::thermal::ExpertHeatLevel> =
                (0..dispatcher.config().num_experts)
                    .map(|i| {
                        thermal.state(i).map(|s| s.heat_level)
                            .unwrap_or(crate::moe::thermal::ExpertHeatLevel::Warm)
                    })
                    .collect();
            let route_config = crate::moe::routing::ExpertRouteConfig::new(
                dispatcher.config().num_experts,
                dispatcher.config().top_k,
            );
            let route_table = crate::moe::routing::ExpertRouteTable {
                config: route_config,
                token_routes: Vec::new(),
                expert_token_counts: vec![0; dispatcher.config().num_experts],
                overflow_count: 0,
            };
            let plan = dispatcher.dispatch(&route_table, &heat_levels);
            if !plan.skipped_experts.is_empty() {
                log::info!(
                    "executor: §15.4 MoE dispatch skipped {} evicted experts: {:?}",
                    plan.skipped_experts.len(),
                    &plan.skipped_experts[..plan.skipped_experts.len().min(8)],
                );
                if let Some(ref mut fh) = self.inference.moe_fault_handler {
                    fh.record_step();
                    let mem_pressure = self.backend.get_memory_pressure().unwrap_or(0.0);
                    for &expert_idx in &plan.skipped_experts {
                        let fault = crate::moe::fault_handler::ExpertFault {
                            expert_idx,
                            layer_idx: 0,
                            request_id: 0,
                            fault_time: std::time::Instant::now(),
                        };
                        fh.handle_fault(fault, mem_pressure, crate::moe::prefetch::ExpertWeightLocation::CpuRam);
                    }
                }
            } else if let Some(ref mut fh) = self.inference.moe_fault_handler {
                fh.record_step();
            }
            log::trace!(
                "executor: §15.3 MoE dispatch: gpu={}, cpu={}, skipped={}, gpu_us={:.0}, cpu_us={:.0}",
                plan.gpu_experts.len(), plan.cpu_experts.len(),
                plan.skipped_experts.len(), plan.gpu_total_us, plan.cpu_total_us,
            );
            Some(plan)
        } else {
            None
        };

        // Sync skipped experts to HotPatchManager + NOP write
        if let (Some(ref plan), Some(ref mut patch_mgr)) =
            (&moe_dispatch_plan, self.inference.hot_patch_manager.as_mut())
        {
            for &expert_idx in &plan.skipped_experts {
                for layer in 0..self.model_ctx.geometry.num_layers {
                    if patch_mgr.is_expert_patched(expert_idx, layer) {
                        continue;
                    }
                    if let Some(&(offset, len)) = self.inference.expert_code_regions.get(&(expert_idx, layer)) {
                        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                        if let Some(ref mega) = self.compute.mega_kernel {
                            match mega.nop_expert_code(expert_idx, layer, offset, len) {
                                Ok(Some(saved)) => {
                                    self.inference.expert_saved_bytes.insert((expert_idx, layer), saved);
                                }
                                Ok(None) => {}
                                Err(e) => {
                                    log::warn!("executor: §14.4 NOP expert {} layer {} failed: {}", expert_idx, layer, e);
                                }
                            }
                        }
                    }
                    log::debug!(
                        "executor: §15.4 expert {} skipped at dispatch (evicted) layer {}",
                        expert_idx, layer,
                    );
                }
            }
        }

        // §21 WP-005: MoE expert weight prefetch
        if let (Some(ref mut prefetcher), Some(ref thermal)) =
            (&mut self.inference.moe_prefetcher, &self.inference.moe_thermal)
        {
            let num_experts = prefetcher.num_experts();
            let heat_levels: Vec<crate::moe::thermal::ExpertHeatLevel> =
                (0..num_experts)
                    .map(|i| {
                        thermal.state(i).map(|s| s.heat_level)
                            .unwrap_or(crate::moe::thermal::ExpertHeatLevel::Warm)
                    })
                    .collect();
            for expert_idx in 0..num_experts {
                let location = crate::moe::prefetch::ExpertWeightLocation::from_heat_level(
                    heat_levels.get(expert_idx).copied()
                        .unwrap_or(crate::moe::thermal::ExpertHeatLevel::Warm),
                );
                prefetcher.update_location(expert_idx, location);
            }
            let routed_experts: Vec<usize> = moe_dispatch_plan
                .as_ref()
                .map(|p| p.gpu_experts.iter().map(|e| e.expert_idx).collect())
                .unwrap_or_default();
            if !routed_experts.is_empty() {
                let requests = prefetcher.schedule_prefetch(&routed_experts, &heat_levels);
                if !requests.is_empty() {
                    log::debug!(
                        "executor: §21 WP-005 prefetch: {} requests for routed experts {:?}",
                        requests.len(),
                        &routed_experts[..routed_experts.len().min(8)],
                    );
                    for req in &requests {
                        if req.destination == crate::moe::prefetch::ExpertWeightLocation::GpuVram
                            && req.source != crate::moe::prefetch::ExpertWeightLocation::GpuL2
                        {
                            let src_tier = match req.source {
                                crate::moe::prefetch::ExpertWeightLocation::CpuRam => {
                                    Some(crate::scheduler::memory_manager::Tier::L2)
                                }
                                crate::moe::prefetch::ExpertWeightLocation::RemoteNode => {
                                    Some(crate::scheduler::memory_manager::Tier::L3)
                                }
                                _ => None,
                            };
                            if let Some(src) = src_tier {
                                match self.dispatch.scheduler.memory_manager.migrate_page(
                                    src,
                                    crate::scheduler::memory_manager::Tier::L1,
                                    req.expert_idx,
                                ) {
                                    Ok(new_id) => {
                                        log::debug!(
                                            "executor: §21 WP-006 migrated expert {} from {:?} to L1 (new_id={})",
                                            req.expert_idx, src, new_id,
                                        );
                                        if let Some(pages) = self.model_ctx.weight_page_table.get_mut(&req.expert_idx) {
                                            if !pages.is_empty() {
                                                pages[0] = new_id;
                                            }
                                        }
                                        self.dispatch.scheduler.hgal.update_page_state(
                                            new_id, None,
                                            crate::scheduler::types::PageState::Active,
                                        );
                                    }
                                    Err(e) => {
                                        log::warn!(
                                            "executor: §21 WP-006 migrate expert {} failed: {}",
                                            req.expert_idx, e,
                                        );
                                        if let Some(ref mut fh) = self.inference.moe_fault_handler {
                                            let fault = crate::moe::fault_handler::ExpertFault {
                                                expert_idx: req.expert_idx,
                                                layer_idx: 0,
                                                request_id: 0,
                                                fault_time: std::time::Instant::now(),
                                            };
                                            fh.handle_fault(fault, 0.5, req.source);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        moe_dispatch_plan
    }

    fn build_sequences(
        &mut self,
        batch: ScheduledBatch,
    ) -> ExecutorResult<BuildSequencesOutput> {
        let mut sequences = Vec::with_capacity(batch.requests.len());
        let mut request_indices = Vec::with_capacity(batch.requests.len());
        let mut batch_results = Vec::new();

        for (idx, req_id) in batch.requests.into_iter().enumerate() {
            let current_draft_steps = batch.draft_steps.get(idx).copied().unwrap_or(0);
            self.ensure_pages_resident(req_id)?;

            let (tokens, position, fused_hidden) = {
                let Some(req) = self.dispatch.requests.get_mut(&req_id) else {
                    continue;
                };
                if req.finished {
                    continue;
                }

                let tokens = if req.is_prefill {
                    let all_tokens = req.prompt_tokens.clone();
                    if self.dispatch.chunked_prefill_scheduler.should_chunk(all_tokens.len()) {
                        let chunk_size = self.dispatch.chunked_prefill_scheduler
                            .next_chunk_size(all_tokens.len(), self.model_ctx.geometry.max_seq_len);
                        let end = chunk_size.min(all_tokens.len());
                        log::debug!(
                            "executor: §10 chunked prefill: req={} total={} chunk=[0,{})",
                            req_id, all_tokens.len(), end,
                        );
                        req.prompt_tokens = all_tokens[end..].to_vec();
                        req.is_prefill = !req.prompt_tokens.is_empty();
                        req.phase = crate::scheduler::request_state::RequestPhase::ChunkedPrefill;
                        all_tokens[..end].to_vec()
                    } else {
                        all_tokens
                    }
                } else {
                    req.output_tokens.last().map(|t| vec![*t]).unwrap_or_default()
                };

                let position = if req.is_prefill {
                    0
                } else {
                    req.prompt_tokens.len() + req.output_tokens.len().saturating_sub(1)
                };

                let fused = if req.is_prefill {
                    req.fused_prefill_hidden.take()
                } else {
                    None
                };
                (tokens, position, fused)
            };

            if tokens.is_empty() {
                if let Some(req) = self.dispatch.requests.get_mut(&req_id) {
                    req.finished = true;
                }
                batch_results.push(BatchResult::fail(req_id));
                continue;
            }

            if self.inference.gate_skip_flags.get(&req_id).copied().unwrap_or(false) {
                self.inference.gate_skip_flags.remove(&req_id);
                continue;
            }

            sequences.push(SequenceInput {
                tokens,
                position,
                draft_steps: current_draft_steps,
                page_table: self.dispatch.scheduler.get_page_table(req_id),
                fused_hidden,
            });
            request_indices.push(req_id);
        }

        Ok(BuildSequencesOutput { sequences, request_indices, batch_results })
    }

    fn update_histogram_and_compact(
        &mut self,
        batch_input: &BatchInput,
        l1_ratio: f32,
        concurrent: usize,
        dispatch_plan: &Option<crate::jit::sub_batch::DispatchPlan>,
    ) {
        for seq in &batch_input.sequences {
            let seq_len = seq.tokens.len();
            self.compute.collapse_seq_len(seq_len);
            self.compute.record_seq_histogram(seq_len);
        }
        self.compute.evolve_golden_buckets();

        self.dispatch.chunked_prefill_scheduler.update_l1_ratio(l1_ratio);
        self.dispatch.chunked_prefill_scheduler.update_concurrency(concurrent);

        if let Some(ref plan) = dispatch_plan {
            use crate::scheduler::chunked_prefill::SlotType;

            let mut decode_ready = Vec::new();
            let mut prefill_queue = Vec::new();
            for sb in &plan.sub_batches {
                for &rid in &sb.request_ids {
                    if let Some(r) = self.dispatch.requests.get(&rid) {
                        if r.is_prefill {
                            prefill_queue.push((rid, r.prompt_tokens.len(), 0));
                        } else {
                            decode_ready.push((rid, r.output_tokens.len()));
                        }
                    }
                }
            }
            let mem_pressure = self.backend.get_memory_pressure().unwrap_or(1.0);
            let mut compact_manifest = self.dispatch.chunked_prefill_scheduler.compose_batch(
                &decode_ready, &prefill_queue, mem_pressure,
            );
            for (i, slot) in compact_manifest.slots.iter_mut().enumerate() {
                let raw_seq_len = slot.token_end - slot.token_start;
                let (_, golden_size) = self.compute.golden_buckets.collapse(raw_seq_len);
                slot.token_end = slot.token_start + golden_size.seq_len;
                slot.compact_target = i as i32;
            }
            compact_manifest.total_tokens = compact_manifest.slots.iter().map(|s| s.token_end - s.token_start).sum();
            compact_manifest.decode_tokens = compact_manifest.slots.iter()
                .filter(|s| matches!(s.slot_type, SlotType::Decode))
                .map(|s| s.token_end - s.token_start)
                .sum();
            compact_manifest.prefill_tokens = compact_manifest.total_tokens - compact_manifest.decode_tokens;

            let cp_config = self.dispatch.chunked_prefill_scheduler.config();
            if compact_manifest.should_compact(cp_config) {
                let compact_config = crate::scheduler::compact::CompactConfig::default();
                let compact_decision = crate::scheduler::compact::evaluate_compact(
                    &compact_manifest,
                    crate::scheduler::compact::OpKind::Gemm,
                    &compact_config,
                );
                if compact_decision.should_compact {
                    log::info!(
                        "executor: §10.6.3 Compact decision TRIGGERED — waste={:.1}%, active={}/{}, reason={:?}",
                        compact_decision.waste_ratio * 100.0,
                        compact_decision.active_count,
                        compact_decision.total_count,
                        compact_decision.reason,
                    );
                }
            }

            if plan.needs_ragged_compaction {
                let batch_size = batch_input.sequences.len();
                let active_flags = vec![true; batch_size];
                let mask = crate::jit::ragged::RequestActiveMask::new(active_flags);
                if self.compute.ragged_compaction.should_compact(&mask) {
                    log::debug!(
                        "executor: §12.2 RaggedCompaction active — waste={:.1}%, batch_size={}",
                        mask.waste_ratio() * 100.0, batch_size,
                    );
                }
            }
        }
    }

    fn schedule_batch(
        &mut self,
        decision: crate::scheduler::jit_types::SchedulerDecision,
    ) -> ExecutorResult<(ScheduledBatch, bool)> {
        if !self.dispatch.batcher.has_pending_work() {
            return Ok((ScheduledBatch {
                requests: Vec::new(),
                seq_offsets: Vec::new(),
                draft_steps: Vec::new(),
            }, false));
        }
        let interleaved = self.dispatch.batcher.build_interleaved_batch(
            &mut self.dispatch.scheduler,
            decision.max_batch_size,
            decision.admit_new_prefill,
            crate::scheduler::types::BatchOrderPolicy::StrictRequestIdOrder,
        );
        let batch = interleaved.inner.clone();
        if interleaved.is_interleaved() {
            log::debug!(
                "executor: interleaved batch — {} decode + {} prefill tokens",
                interleaved.decode_tokens(),
                interleaved.prefill_tokens(),
            );
        }
        let decode_count = interleaved.decode_slots.len();
        let spec_advice = self.inference.spec_decoding.should_speculate(decode_count);
        let spec_enabled = matches!(
            spec_advice,
            crate::jit::epilogue::SpecScheduleAdvice::EnableSpec
        );
        match spec_advice {
            crate::jit::epilogue::SpecScheduleAdvice::EnableSpec => {
                log::debug!(
                    "executor: §17.9 speculative decoding ENABLED (acceptance_rate={:.2})",
                    self.inference.spec_decoding.avg_acceptance_rate()
                );
            }
            crate::jit::epilogue::SpecScheduleAdvice::Fallback => {
                log::debug!("executor: §17.9 speculative decoding FALLBACK (low acceptance streak)");
            }
            crate::jit::epilogue::SpecScheduleAdvice::StandardDecode => {}
        }
        Ok((batch, spec_enabled))
    }

    fn prepare_sub_batch_dispatch(
        &mut self,
        request_ids: &[RequestId],
    ) -> Option<crate::jit::sub_batch::DispatchPlan> {
        let mut shape_map = std::collections::HashMap::new();
        let has_moe_ops = self.model_ctx.forward_config.moe_config.is_some();
        for req_id in request_ids {
            let seq_len = self
                .dispatch.requests
                .get(req_id)
                .map(|r| if r.is_prefill { r.prompt_tokens.len() } else { 1 })
                .unwrap_or(1);
            let golden_seq = self.compute.collapse_seq_len(seq_len);
            if golden_seq != seq_len {
                log::trace!("executor: §12.4 Golden Bucket: seq_len {} → {}", seq_len, golden_seq);
            }
            let shape = self.compute.classify_request_shape(has_moe_ops, 0.0);
            shape_map.insert(*req_id, shape);
        }

        if shape_map.is_empty() {
            return None;
        }

        use crate::scheduler::chunked_prefill::SlotType;
        let mut decode_ready = Vec::new();
        let mut prefill_queue = Vec::new();
        for &rid in request_ids {
            if let Some(r) = self.dispatch.requests.get(&rid) {
                if r.is_prefill {
                    prefill_queue.push((rid, r.prompt_tokens.len(), 0));
                } else {
                    decode_ready.push((rid, r.output_tokens.len()));
                }
            }
        }
        let mem_pressure = self.backend.get_memory_pressure().unwrap_or(1.0);
        let mut manifest = self.dispatch.chunked_prefill_scheduler.compose_batch(
            &decode_ready, &prefill_queue, mem_pressure,
        );
        for (i, slot) in manifest.slots.iter_mut().enumerate() {
            let raw_seq_len = slot.token_end - slot.token_start;
            let (_, golden_size) = self.compute.golden_buckets.collapse(raw_seq_len);
            slot.token_end = slot.token_start + golden_size.seq_len;
            slot.compact_target = i as i32;
        }
        manifest.total_tokens = manifest.slots.iter().map(|s| s.token_end - s.token_start).sum();
        manifest.decode_tokens = manifest.slots.iter()
            .filter(|s| matches!(s.slot_type, SlotType::Decode))
            .map(|s| s.token_end - s.token_start)
            .sum();
        manifest.prefill_tokens = manifest.total_tokens - manifest.decode_tokens;

        let cp_config = self.dispatch.chunked_prefill_scheduler.config();
        manifest.compact_required = manifest.should_compact(cp_config);

        let plan = self.compute.sub_batch_dispatcher.dispatch(&manifest, &shape_map);
        if plan.sub_batches.len() > 1 {
            log::info!(
                "executor: §12.1 Sub-Batch dispatched {} sub-batches ({} orphans, reason={:?})",
                plan.sub_batches.len(), plan.orphan_count, plan.reason,
            );
        }
        Some(plan)
    }

    fn plan_prefill_sessions(
        &mut self,
        request_ids: &[RequestId],
    ) -> ExecutorResult<(usize, f32, usize)> {
        let page_size = self.dispatch.scheduler.page_size().max(1);
        let adaptive = AdaptiveChunkPolicy::new(&Scheduler2024Config::default().chunked);
        let l1_usage = self.dispatch.memory_manager.tier_usage(Tier::L1);
        let l1_ratio = if l1_usage.capacity > 0 {
            (l1_usage.capacity.saturating_sub(l1_usage.used)) as f32 / l1_usage.capacity as f32
        } else {
            1.0
        };
        let concurrent = request_ids.len();
        for &req_id in request_ids {
            let (is_prefill, prompt_len, session_id) = {
                let Some(req) = self.dispatch.requests.get(&req_id) else {
                    continue;
                };
                (req.is_prefill, req.prompt_tokens.len(), req.session_id)
            };
            if !is_prefill {
                continue;
            }
            let chunk_size = adaptive.compute(l1_ratio, concurrent, prompt_len);
            let plan = self.dispatch.memory_manager.plan_prefill(prompt_len, chunk_size, page_size);
            if let PrefillPlan::Pipelined { l1_pages, .. } = plan {
                if l1_pages > 0 {
                    self.reclaim_memory(l1_pages)?;
                }
            }
            if let Some(sid) = session_id {
                let finalized = match self.dispatch.memory_manager.session_finalized_position(sid) {
                    Some(pos) => pos,
                    None => {
                        return Err(ExecutorError::Scheduler(format!(
                            "session_finalized_position returned None for session {sid}"
                        )));
                    }
                };
                let prefix_tokens = prompt_len.min(finalized);
                if prefix_tokens > 0 {
                    if let Err(e) =
                        self.dispatch.memory_manager.claim_session_prefix(sid, req_id, prefix_tokens)
                    {
                        log::warn!("executor: claim_session_prefix failed for session {sid}: {e}");
                    }
                }
            }
        }
        Ok((page_size, l1_ratio, concurrent))
    }

    fn process_results(
        &mut self,
        logits_list: &[LogitsHandle],
        batch_telemetry: &[SequenceTelemetry],
        request_indices: &[RequestId],
        page_size: usize,
        batch_input: &BatchInput,
    ) -> ExecutorResult<Vec<BatchResult>> {
        let mut results = Vec::with_capacity(logits_list.len());
        for (i, logits) in logits_list.iter().enumerate() {
            let req_id = request_indices[i];
            let req_telemetry = batch_telemetry[i];

            if self
                .model_ctx.profile_accumulator
                .record_and_check(0, req_telemetry.transform_ratio)
            {
                log::info!("executor: ProfileAccumulator triggered Re-Fusion due to high stability.");
            }

            let logical_index = batch_input.sequences[i].position / page_size;
            let vpid = VirtualPageId::new(req_id, logical_index);
            if let Ok((tier, _physical_id)) = self.dispatch.memory_manager.resolve(vpid) {
                if tier == crate::scheduler::Tier::L1 {
                    // page_entropies handled separately via return
                }
            }

            let sampling_config = self
                .dispatch.requests
                .get(&req_id)
                .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?
                .sampling_config;
            let next_token = self.sample_from_logits(logits, &sampling_config)?;

            if std::env::var("GLLM_DEBUG_SAMPLING").is_ok() {
                let l = &logits.data;
                let mut top: [(u32, f32); 3] = [(0, f32::NEG_INFINITY); 3];
                for (i, &v) in l.iter().enumerate() {
                    if v > top[0].1 {
                        top[2] = top[1];
                        top[1] = top[0];
                        top[0] = (i as u32, v);
                    } else if v > top[1].1 {
                        top[2] = top[1];
                        top[1] = (i as u32, v);
                    } else if v > top[2].1 {
                        top[2] = (i as u32, v);
                    }
                }
                let mean: f32 = l.iter().sum::<f32>() / l.len() as f32;
                let var: f32 = l.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / l.len() as f32;
                eprintln!(
                    "[SAMPLE req={req_id} logits.len={} next={next_token} \
                           top3=[{},{}],[{},{}],[{},{}] mean={:.4} var={:.4}]",
                    l.len(), top[0].0, top[0].1, top[1].0, top[1].1,
                    top[2].0, top[2].1, mean, var
                );
            }

            let hooks_guard = self.model_ctx.hooks.read();
            let hooks_decision = if let Ok(hooks) = &hooks_guard {
                let mut decision = crate::generation::HookDecision::Continue;
                let generated_tokens = self
                    .dispatch.requests
                    .get(&req_id)
                    .map(|req| req.output_tokens.clone())
                    .unwrap_or_default();
                for hook in hooks.iter() {
                    match hook.post_step(&logits.data, &generated_tokens) {
                        crate::generation::HookDecision::Continue => continue,
                        crate::generation::HookDecision::Veto(reason) => {
                            log::debug!("executor: hook vetoed token {} for request {}: {}", next_token, req_id, reason);
                            decision = crate::generation::HookDecision::Veto(reason);
                            break;
                        }
                        crate::generation::HookDecision::Terminate => {
                            log::debug!("executor: hook terminated generation for request {}", req_id);
                            decision = crate::generation::HookDecision::Terminate;
                            break;
                        }
                    }
                }
                decision
            } else {
                crate::generation::HookDecision::Continue
            };
            drop(hooks_guard);

            match hooks_decision {
                crate::generation::HookDecision::Continue => {}
                crate::generation::HookDecision::Veto(_) => {
                    let req = self
                        .dispatch.requests
                        .get_mut(&req_id)
                        .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
                    req.finished = true;
                    results.push(BatchResult::complete(req_id, None, req_telemetry));
                    continue;
                }
                crate::generation::HookDecision::Terminate => {
                    let req = self
                        .dispatch.requests
                        .get_mut(&req_id)
                        .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
                    req.finished = true;
                    results.push(BatchResult::complete(req_id, None, req_telemetry));
                    continue;
                }
            }

            let req = self
                .dispatch.requests
                .get_mut(&req_id)
                .ok_or(ExecutorError::RequestNotFound { request_id: req_id })?;
            req.output_tokens.push(next_token);
            req.is_prefill = false;
            req.phase = crate::scheduler::request_state::RequestPhase::Decode;

            let eos_token = self.model_ctx.model_config.eos_token_id;
            let mut request_finished = false;
            if eos_token.is_some_and(|id| id == next_token)
                || req.output_tokens.len() >= req.max_new_tokens
            {
                req.finished = true;
                request_finished = true;
            }

            if request_finished {
                results.push(BatchResult::complete(req_id, Some(next_token), req_telemetry));
            } else {
                results.push(BatchResult::continue_with_token(req_id, next_token, req_telemetry));
            }
        }
        Ok(results)
    }

    fn update_step_observability(
        &mut self,
        logits_list: &[LogitsHandle],
        batch_sparsity: f32,
    ) {
        let batch_entropy = {
            let mut total = 0.0f32;
            for logits in logits_list {
                total += shannon_entropy(&logits.data);
            }
            if !logits_list.is_empty() {
                total / logits_list.len() as f32
            } else {
                0.0
            }
        };
        self.observability.observer.update_logits_entropy(batch_entropy);
        self.observability.observer.update_attention_sparsity(batch_sparsity);

        let l2_usage = self.dispatch.memory_manager.tier_usage(crate::scheduler::memory_manager::Tier::L2);
        let l2_ratio = if l2_usage.capacity > 0 {
            l2_usage.used as f32 / l2_usage.capacity as f32
        } else {
            0.0
        };
        self.observability.observer.update_swap_io_rate(l2_ratio);

        if let Some(ref fault_handler) = self.inference.moe_fault_handler {
            let stats = fault_handler.stats();
            let working_set = self.inference.moe_thermal.as_ref().map_or(0, |t| t.working_set_size());
            self.observability.observer.update_moe_fault_metrics(
                stats.fault_rate as f32,
                stats.avg_recovery_us as f32,
                working_set,
            );
        }

        if self.model_ctx.weight_pages_registered {
            let total = self.model_ctx.weight_page_table.len();
            let l1_usage = self.dispatch.scheduler.memory_manager.tier_usage(
                crate::scheduler::memory_manager::Tier::L1,
            );
            let l2_usage = self.dispatch.scheduler.memory_manager.tier_usage(
                crate::scheduler::memory_manager::Tier::L2,
            );
            let (eviction_count, recovery_count) = if let Some(ref fh) = self.inference.moe_fault_handler {
                let s = fh.stats();
                let recovered = s.total_faults.saturating_sub(s.in_flight_restorations as u64);
                (s.total_faults as usize, recovered as usize)
            } else {
                (0, 0)
            };
            self.observability.observer.update_weight_metrics(
                total,
                l1_usage.used.min(total),
                l2_usage.used.min(total),
                0,
                eviction_count,
                recovery_count,
            );
        }
    }

    fn push_telemetry_to_director(
        &mut self,
        batch_telemetry: &[SequenceTelemetry],
    ) {
        if let Some(ref director) = self.compute.jit_director {
            let shared = director.shared();
            shared.advance_step();
            if let Some(ref thermal) = self.inference.moe_thermal {
                for expert_idx in 0..self.model_ctx.geometry.num_experts {
                    if let Some(state) = thermal.state(expert_idx) {
                        if state.hit_count > 0 {
                            shared.record_expert_hit(expert_idx);
                        }
                    }
                }
            }
            for tel in batch_telemetry {
                let mut header = crate::kv_cache::KvPageHeader::new(0);
                header.ref_count = 1;
                header.entropy_avg = crate::kv_cache::f32_to_f16_bits(tel.output_entropy);
                header.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(tel.transform_ratio);
                self.compute.telemetry_aggregator.ingest_from_page_header(&header);
            }
            for event in shared.drain_events() {
                match &event {
                    crate::jit::director::ConsensusEvent::ExpertFrozen {
                        expert_idx,
                        zero_hit_steps,
                    } => {
                        log::info!("executor: JIT Director detected frozen expert {} (zero hits for {} steps)", expert_idx, zero_hit_steps);
                        if let Some(ref mut thermal) = self.inference.moe_thermal {
                            if thermal.evict_expert(*expert_idx) {
                                log::info!("executor: Expert {} evicted via thermal manager", expert_idx);
                            }
                        }
                        self.apply_hot_patches_for_frozen_expert(*expert_idx);
                    }
                    crate::jit::director::ConsensusEvent::AttentionSilent {
                        avg_entropy,
                        duration_steps,
                    } => {
                        log::warn!("executor: JIT Director detected attention silence (entropy={:.4}, duration={})", avg_entropy, duration_steps);
                    }
                    crate::jit::director::ConsensusEvent::LayerRedundant {
                        avg_delta_rho,
                        duration_steps,
                    } => {
                        log::info!("executor: JIT Director detected redundant layer (delta_rho={:.6}, duration={})", avg_delta_rho, duration_steps);
                    }
                }
            }
        }
    }

    fn apply_hot_patches_for_frozen_expert(&mut self, _expert_idx: usize) {
        if let (Some(ref mut patch_mgr), Some(ref thermal)) =
            (&mut self.inference.hot_patch_manager, &self.inference.moe_thermal)
        {
            let active_requests = self.dispatch.requests.len();
            let instructions = patch_mgr.generate_expert_patch_instructions(
                thermal,
                self.model_ctx.geometry.num_layers,
                active_requests,
            );
            for instr in &instructions {
                let result = patch_mgr.apply_patch(instr);
                if result.success {
                    log::info!(
                        "executor: §14.4 Hot JMP Patch applied: {:?} → {:?}",
                        instr.target, instr.operation,
                    );
                    if let crate::moe::PatchTarget::ExpertCode {
                        expert_idx,
                        layer_idx,
                    } = &instr.target
                    {
                        if let Some(&(offset, len)) =
                            self.inference.expert_code_regions.get(&(*expert_idx, *layer_idx))
                        {
                            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                            if let Some(ref mega) = self.compute.mega_kernel {
                                match mega.nop_expert_code(*expert_idx, *layer_idx, offset, len) {
                                    Ok(Some(saved)) => {
                                        self.inference.expert_saved_bytes
                                            .insert((*expert_idx, *layer_idx), saved);
                                    }
                                    Ok(None) => {}
                                    Err(e) => {
                                        log::warn!("executor: §14.4 NOP write failed: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn run_kv_optimizer(
        &mut self,
        batch_telemetry: &[SequenceTelemetry],
    ) {
        use crate::kv_cache::KvPageHeader;
        use crate::scheduler::kv_optimizer;
        let num_kv_heads = self.model_ctx.geometry.num_kv_heads;
        let num_layers = self.model_ctx.geometry.num_layers;
        let mut all_layer_headers: Vec<Vec<KvPageHeader>> = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let depth_ratio = layer_idx as f32 / num_layers.max(1) as f32;
            let entropy_decay = 1.0 - 0.2 * depth_ratio;
            let head_spread_decay = 1.0 - 0.3 * depth_ratio;
            let layer_headers: Vec<KvPageHeader> = batch_telemetry
                .iter()
                .map(|tel| {
                    let mut h = KvPageHeader::new(0);
                    h.ref_count = 1;
                    h.entropy_avg =
                        crate::kv_cache::f32_to_f16_bits(tel.output_entropy * entropy_decay);
                    h.softmax_max_avg =
                        crate::kv_cache::f32_to_f16_bits(if tel.output_entropy < 1.0 {
                            0.85
                        } else {
                            0.3
                        });
                    h.delta_rho_avg = crate::kv_cache::f32_to_f16_bits(
                        tel.transform_ratio * (1.0 - 0.1 * depth_ratio),
                    );
                    h.dead_ratio = crate::kv_cache::f32_to_dead_ratio(tel.dead_density);
                    h.head_entropy_max =
                        (tel.per_head_entropy * 25.5 * head_spread_decay).min(255.0) as u8;
                    h.head_entropy_min =
                        (tel.per_head_entropy * 10.0 * head_spread_decay).min(255.0) as u8;
                    h
                })
                .collect();
            all_layer_headers.push(layer_headers);
        }
        kv_optimizer::optimize_with_cross_layer_reuse(
            &self.kv.kv_optimizer,
            &mut all_layer_headers,
            num_kv_heads,
        );
        let total_tier_changes: usize = all_layer_headers
            .iter()
            .flat_map(|v| v.iter())
            .filter(|h| h.needs_requantize())
            .count();
        let total_pages = all_layer_headers.iter().map(|v| v.len()).sum::<usize>();
        if total_tier_changes > 0 {
            log::debug!(
                "executor: §19 KV optimizer (cross-layer K={}): {}/{} page-tier changes across {} layers",
                self.kv.kv_optimizer.chunk_cross_layer_k,
                total_tier_changes, total_pages, num_layers,
            );
        }
        let mid_layer = num_layers / 2;
        if mid_layer < all_layer_headers.len() {
            for h in &all_layer_headers[mid_layer] {
                self.compute.telemetry_aggregator.ingest_from_page_header(h);
            }
        }
        if !all_layer_headers.is_empty() && !all_layer_headers[0].is_empty() {
            let tier_strings: Vec<String> = all_layer_headers[0]
                .iter()
                .map(|h| format!("{:?}", h.precision_tier()))
                .collect();
            let majority_tier =
                crate::jit::variant_registry::VariantRegistry::majority_kv_tier(&tier_strings);
            self.kv.majority_kv_tier = majority_tier.clone();
            log::trace!(
                "executor: §19 KV optimizer majority_tier={:?} (from {} headers)",
                majority_tier,
                all_layer_headers[0].len(),
            );
        }
    }

    fn apply_epilogue_decisions(
        &mut self,
        epilogue_summary: &crate::jit::epilogue_subsystem::EpilogueBatchSummary,
        request_indices: &[RequestId],
    ) {
        use crate::jit::epilogue::GateSkipDecision;
        let mut skip_layer_count = 0usize;
        let mut bypass_layer_count = 0usize;
        self.inference.gate_skip_flags.clear();
        for (i, decision) in epilogue_summary.per_request.iter().enumerate() {
            if let Some(&req_id) = request_indices.get(i) {
                match decision.gate_skip {
                    GateSkipDecision::Skip => {
                        skip_layer_count += 1;
                        self.inference.gate_skip_flags.insert(req_id, true);
                    }
                    GateSkipDecision::MaskedCompute => {}
                    GateSkipDecision::FullCompute => {}
                }
            }
            if decision.bypass_decision == crate::jit::epilogue::ResidualBypassDecision::Bypass { bypass_layer_count += 1 }
        }
        if skip_layer_count > 0 {
            log::debug!(
                "executor: §13.1 Gate-First Skip active for {}/{} requests (flagged for next step)",
                skip_layer_count, epilogue_summary.per_request.len(),
            );
        }
        if bypass_layer_count > 0 {
            log::debug!(
                "executor: §13.3 Residual Bypass active for {}/{} requests",
                bypass_layer_count,
                epilogue_summary.per_request.len(),
            );
        }
        for decision in &epilogue_summary.per_request {
            match &decision.prefetch_advice {
                crate::jit::prefetch::PrefetchAdvice::Forward(distance) => {
                    log::trace!("executor: §13.2 Centroid Prefetch forward {} tokens", distance);
                }
                crate::jit::prefetch::PrefetchAdvice::Backward(distance) => {
                    log::trace!("executor: §13.2 Centroid Prefetch backward {} tokens", distance);
                }
                crate::jit::prefetch::PrefetchAdvice::Sink(count) => {
                    log::trace!("executor: §13.2 Sink Prefetch {} tokens", count);
                }
                crate::jit::prefetch::PrefetchAdvice::None => {}
            }
        }
        for (i, decision) in epilogue_summary.per_request.iter().enumerate() {
            if let Some(&req_id) = request_indices.get(i) {
                match decision.spec_advice {
                    crate::jit::epilogue::SpecScheduleAdvice::EnableSpec => {
                        if let Some(seq) = self.dispatch.batcher.get_running_mut(req_id) {
                            seq.draft_budget = 8;
                        }
                    }
                    crate::jit::epilogue::SpecScheduleAdvice::Fallback => {
                        if let Some(seq) = self.dispatch.batcher.get_running_mut(req_id) {
                            seq.draft_budget = 0;
                        }
                    }
                    crate::jit::epilogue::SpecScheduleAdvice::StandardDecode => {}
                }
            }
        }
    }

    fn step_finalize(
        &mut self,
        batch_results: &[BatchResult],
        page_entropies: &HashMap<usize, f32>,
    ) {
        for result in batch_results {
            if !matches!(result.action, BatchAction::Complete | BatchAction::Fail) {
                continue;
            }
            let request_id = result.request_id;
            if let Some(req) = self.dispatch.requests.get(&request_id) {
                if let Some(sid) = req.session_id {
                    let total_processed = req.prompt_tokens.len() + req.output_tokens.len();
                    self.dispatch.memory_manager
                        .finalize_session_tokens(sid, total_processed);
                }
            }
            self.release_request_pages(request_id);
        }

        self.dispatch.batcher
            .update_batch(&mut self.dispatch.scheduler, batch_results);

        let evicted =
            self.dispatch.memory_manager
                .entropy_evict(page_entropies, 0.1, crate::scheduler::Tier::L1);
        if evicted > 0 {
            log::debug!("entropy_evict: freed {evicted} low-entropy KV pages synchronously after batch step");
        }

        self.optimize_kv_cache();
    }
}

// ---------------------------------------------------------------------------
// Tests for helper functions
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor_types::{AttentionMaskType, KvCacheHandle};

    #[test]
    fn shannon_entropy_uniform_distribution() {
        let logits = [1.0, 1.0, 1.0, 1.0];
        let h = shannon_entropy(&logits);
        assert!(h > 1.3 && h < 1.4, "entropy should be ~ln(4) ≈ 1.386, got {h}");
    }

    #[test]
    fn shannon_entropy_peaked_distribution() {
        let logits = [100.0, 0.0, 0.0, 0.0];
        let h = shannon_entropy(&logits);
        assert!(h < 0.01, "peaked distribution entropy should be ~0, got {h}");
    }

    #[test]
    fn shannon_entropy_empty() {
        let h = shannon_entropy(&[]);
        assert_eq!(h, 0.0);
    }

    #[test]
    fn extract_top_k_basic() {
        let logits = [0.1, 0.5, 0.3, 0.9, 0.2];
        let top3 = extract_top_k_token_ids(&logits, 3);
        assert_eq!(top3, vec![3, 1, 2]);
    }

    #[test]
    fn extract_top_k_edge_cases() {
        assert!(extract_top_k_token_ids(&[], 5).is_empty());
        assert!(extract_top_k_token_ids(&[1.0, 2.0], 0).is_empty());
    }

    #[test]
    fn extract_top_k_fewer_than_k() {
        let logits = [1.0];
        let top5 = extract_top_k_token_ids(&logits, 5);
        assert_eq!(top5, vec![0]);
    }

    #[test]
    fn argmax_token_basic() {
        let logits = [0.1, 0.2, 0.5, 0.9, 0.3];
        assert_eq!(argmax_token(&logits), 3);
    }

    #[test]
    fn argmax_token_single() {
        assert_eq!(argmax_token(&[42.0]), 0);
    }

    // =======================================================================
    // Additional tests for extract_top_k_token_ids
    // =======================================================================

    #[test]
    fn extract_top_k_tie_values() {
        // When logits have identical values, all tied indices should be returned.
        let logits = [5.0, 5.0, 5.0];
        let top2 = extract_top_k_token_ids(&logits, 2);
        assert_eq!(top2.len(), 2);
        // Both returned IDs must be valid indices 0, 1, or 2.
        for &id in &top2 {
            assert!(id < 3, "tie-breaking produced invalid index {id}");
        }
    }

    #[test]
    fn extract_top_k_all_negative() {
        let logits = [-5.0, -1.0, -3.0, -10.0];
        let top2 = extract_top_k_token_ids(&logits, 2);
        assert_eq!(top2[0], 1, "highest logit at index 1");
        assert_eq!(top2[1], 2, "second highest at index 2");
    }

    #[test]
    fn extract_top_k_nan_handling() {
        // NaN values should not crash. With partial_cmp -> Equal, NaN sorts as
        // equal to everything. The result length must be correct; no panic.
        let logits = [1.0, f32::NAN, 3.0];
        let top2 = extract_top_k_token_ids(&logits, 2);
        assert_eq!(top2.len(), 2);
        // Verify no panic and all indices are valid.
        for &id in &top2 {
            assert!(id < 3, "invalid index {id}");
        }
    }

    #[test]
    fn extract_top_k_single_element() {
        let logits = [7.0];
        let top1 = extract_top_k_token_ids(&logits, 1);
        assert_eq!(top1, vec![0]);
    }

    #[test]
    fn extract_top_k_k_equals_length() {
        let logits = [0.3, 0.1, 0.5];
        let top3 = extract_top_k_token_ids(&logits, 3);
        assert_eq!(top3.len(), 3);
        // First element should be index 2 (highest value 0.5).
        assert_eq!(top3[0], 2);
    }

    #[test]
    fn extract_top_k_preserves_order() {
        // Descending order: index 4 (0.9) > index 1 (0.5) > index 2 (0.3) > ...
        let logits = [0.1, 0.5, 0.3, 0.0, 0.9];
        let top3 = extract_top_k_token_ids(&logits, 3);
        assert_eq!(top3, vec![4, 1, 2]);
    }

    #[test]
    fn extract_top_k_with_infinity() {
        let logits = [f32::INFINITY, 0.0, f32::NEG_INFINITY];
        let top2 = extract_top_k_token_ids(&logits, 2);
        assert_eq!(top2[0], 0, "positive infinity should be first");
    }

    // =======================================================================
    // Additional tests for argmax_token
    // =======================================================================

    #[test]
    fn argmax_token_empty_returns_zero() {
        // Empty slice: unwrap_or(0) kicks in.
        assert_eq!(argmax_token(&[]), 0);
    }

    #[test]
    fn argmax_token_tie_returns_last() {
        // Tied maximum values: max_by with partial_cmp(Equal) returns last occurrence.
        let logits = [1.0, 5.0, 5.0, 2.0];
        let result = argmax_token(&logits);
        assert_eq!(result, 2, "last occurrence of max wins with partial_cmp Equal");
    }

    #[test]
    fn argmax_token_all_equal() {
        // All equal: max_by returns the last index.
        let logits = [3.0, 3.0, 3.0];
        assert_eq!(argmax_token(&logits), 2, "last index when all equal");
    }

    #[test]
    fn argmax_token_last_is_max() {
        let logits = [0.1, 0.2, 0.3, 0.4, 99.0];
        assert_eq!(argmax_token(&logits), 4);
    }

    #[test]
    fn argmax_token_with_negative() {
        let logits = [-10.0, -5.0, -20.0];
        assert_eq!(argmax_token(&logits), 1, "least negative wins");
    }

    #[test]
    fn argmax_token_with_nan() {
        // NaN partial_cmp with anything is None -> Equal via unwrap_or.
        // max_by returns the last element when all compare as Equal.
        let logits = [f32::NAN, 1.0];
        let result = argmax_token(&logits);
        assert_eq!(result, 1, "last element wins when NaN compares as Equal");
    }

    #[test]
    fn argmax_token_large_slice() {
        let mut logits = vec![0.0; 10000];
        logits[7777] = 1.0;
        assert_eq!(argmax_token(&logits), 7777);
    }

    // =======================================================================
    // Additional tests for shannon_entropy
    // =======================================================================

    #[test]
    fn shannon_entropy_single_element() {
        // Single element: p=1.0, entropy = -1.0 * ln(1.0) = 0.
        let h = shannon_entropy(&[42.0]);
        assert!(
            h.abs() < 1e-6,
            "single element distribution should have zero entropy, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_two_equal() {
        // [0, 0] -> uniform over 2 -> entropy = ln(2) ~ 0.693.
        let h = shannon_entropy(&[0.0, 0.0]);
        assert!(
            (h - 2.0f32.ln()).abs() < 1e-4,
            "two equal logits should give ln(2), got {h}"
        );
    }

    #[test]
    fn shannon_entropy_non_negativity() {
        let logits = [-100.0, -50.0, 0.0, 50.0, 100.0];
        let h = shannon_entropy(&logits);
        assert!(h >= 0.0, "entropy must be non-negative, got {h}");
    }

    #[test]
    fn shannon_entropy_monotonic_with_uniformity() {
        // More uniform => higher entropy.
        let peaked = shannon_entropy(&[100.0, 0.0, 0.0, 0.0]);
        let uniform = shannon_entropy(&[0.0, 0.0, 0.0, 0.0]);
        assert!(
            peaked < uniform,
            "peaked entropy ({peaked}) should be < uniform ({uniform})"
        );
    }

    #[test]
    fn shannon_entropy_with_nan() {
        // NaN logits should not panic. Entropy should still be finite or at least
        // not crash. The softmax exp(NaN) produces NaN, ln(NaN) is NaN.
        // Verify it does not panic.
        let _ = shannon_entropy(&[f32::NAN, 1.0]);
    }

    #[test]
    fn shannon_entropy_with_infinity() {
        // Very large positive logit makes p->1 for that entry, others -> 0.
        let h = shannon_entropy(&[f32::INFINITY, 0.0]);
        // Infinity - Infinity in exp is problematic; verify no panic.
        let _ = h;
    }

    #[test]
    fn shannon_entropy_large_uniform() {
        // 8 equal logits => entropy = ln(8) ~ 2.079.
        let logits = [1.0; 8];
        let h = shannon_entropy(&logits);
        assert!(
            (h - 8.0f32.ln()).abs() < 1e-3,
            "8 uniform logits should give ln(8), got {h}"
        );
    }

    #[test]
    fn shannon_entropy_scales_with_size() {
        // Entropy of uniform distribution over N = ln(N).
        let h4 = shannon_entropy(&[1.0; 4]);
        let h8 = shannon_entropy(&[1.0; 8]);
        assert!(
            h8 > h4,
            "entropy of uniform over 8 ({h8}) > uniform over 4 ({h4})"
        );
    }

    #[test]
    fn shannon_entropy_extreme_values() {
        // Very large and very small logits mixed.
        let logits = [1e30, 1e-30, 1e30, 1e-30];
        let h = shannon_entropy(&logits);
        assert!(
            h.is_finite(),
            "extreme mixed values should produce finite entropy, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_all_zeros() {
        // All-zero logits: uniform distribution over N elements.
        let h = shannon_entropy(&[0.0, 0.0, 0.0]);
        assert!(
            (h - 3.0f32.ln()).abs() < 1e-4,
            "all-zero logits should give ln(3), got {h}"
        );
    }

    // =======================================================================
    // Tests for SequenceInput::validate_page_table
    // =======================================================================

    #[test]
    fn validate_page_table_no_page_table() {
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(100).is_ok());
    }

    #[test]
    fn validate_page_table_valid_entries() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 5, 9]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn validate_page_table_boundary_valid() {
        // Page ID = total_pages - 1 is valid (zero-indexed).
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![9]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn validate_page_table_out_of_bounds() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 10]),
            fused_hidden: None,
        };
        let err = seq.validate_page_table(10).unwrap_err();
        assert!(
            err.contains("page_table[1] = 10 >= total_pages 10"),
            "unexpected error message: {err}"
        );
    }

    #[test]
    fn validate_page_table_zero_total_pages() {
        // Any non-empty page table with total_pages=0 must fail.
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(0).is_err());
    }

    #[test]
    fn validate_page_table_empty_page_table() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![]),
            fused_hidden: None,
        };
        // Empty page table has no entries to validate.
        assert!(seq.validate_page_table(10).is_ok());
    }

    #[test]
    fn validate_page_table_multiple_violations_reports_first() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![15, 20, 30]),
            fused_hidden: None,
        };
        let err = seq.validate_page_table(10).unwrap_err();
        assert!(
            err.contains("page_table[0] = 15"),
            "should report first violation, got: {err}"
        );
    }

    #[test]
    fn validate_page_table_single_page_exact() {
        // Only page 0 exists, page_table references it.
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(1).is_ok());
    }

    // =======================================================================
    // Tests for BuildSequencesOutput struct (private, but accessible in test)
    // =======================================================================

    #[test]
    fn build_sequences_output_fields_accessible() {
        let output = BuildSequencesOutput {
            sequences: vec![
                SequenceInput {
                    tokens: vec![1, 2, 3],
                    position: 0,
                    draft_steps: 0,
                    page_table: None,
                    fused_hidden: None,
                },
            ],
            request_indices: vec![42],
            batch_results: vec![],
        };
        assert_eq!(output.sequences.len(), 1);
        assert_eq!(output.request_indices, vec![42]);
        assert!(output.batch_results.is_empty());
    }

    #[test]
    fn build_sequences_output_empty() {
        let output = BuildSequencesOutput {
            sequences: vec![],
            request_indices: vec![],
            batch_results: vec![],
        };
        assert!(output.sequences.is_empty());
        assert!(output.request_indices.is_empty());
        assert!(output.batch_results.is_empty());
    }

    // =======================================================================
    // Tests for SequenceInput construction edge cases
    // =======================================================================

    #[test]
    fn sequence_input_with_fused_hidden() {
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 5,
            draft_steps: 2,
            page_table: Some(vec![0, 1]),
            fused_hidden: Some(vec![0.1; 96]),
        };
        assert_eq!(seq.tokens.len(), 3);
        assert_eq!(seq.position, 5);
        assert_eq!(seq.draft_steps, 2);
        assert_eq!(seq.page_table.as_ref().unwrap().len(), 2);
        assert_eq!(seq.fused_hidden.as_ref().unwrap().len(), 96);
    }

    #[test]
    fn sequence_input_large_position() {
        let seq = SequenceInput {
            tokens: vec![100],
            position: usize::MAX / 2,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.position, usize::MAX / 2);
    }

    #[test]
    fn sequence_input_empty_tokens() {
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
    fn sequence_input_validate_page_table_large_page_id() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![u32::MAX]),
            fused_hidden: None,
        };
        let err = seq.validate_page_table(u32::MAX as usize).unwrap_err();
        assert!(
            err.contains(&format!("page_table[0] = {}", u32::MAX)),
            "unexpected error: {err}"
        );
    }

    // =======================================================================
    // Tests for BatchInput
    // =======================================================================

    #[test]
    fn batch_input_empty_sequences() {
        let batch = BatchInput { sequences: vec![] };
        assert!(batch.sequences.is_empty());
    }

    #[test]
    fn batch_input_single_sequence() {
        let seq = SequenceInput {
            tokens: vec![1, 2, 3],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        let batch = BatchInput { sequences: vec![seq] };
        assert_eq!(batch.sequences.len(), 1);
        assert_eq!(batch.sequences[0].tokens, vec![1, 2, 3]);
    }

    #[test]
    fn batch_input_multiple_sequences_varied_page_tables() {
        let seq1 = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        let seq2 = SequenceInput {
            tokens: vec![2, 3],
            position: 10,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2]),
            fused_hidden: None,
        };
        let seq3 = SequenceInput {
            tokens: vec![4],
            position: 100,
            draft_steps: 3,
            page_table: Some(vec![]),
            fused_hidden: Some(vec![0.5; 64]),
        };
        let batch = BatchInput {
            sequences: vec![seq1, seq2, seq3],
        };
        assert_eq!(batch.sequences.len(), 3);
        assert!(batch.sequences[0].page_table.is_none());
        assert_eq!(batch.sequences[1].page_table.as_ref().unwrap().len(), 3);
        assert!(batch.sequences[2].page_table.as_ref().unwrap().is_empty());
        assert_eq!(batch.sequences[2].fused_hidden.as_ref().unwrap().len(), 64);
    }

    #[test]
    fn batch_input_clone_preserves_data() {
        let seq = SequenceInput {
            tokens: vec![7, 8],
            position: 5,
            draft_steps: 1,
            page_table: Some(vec![3]),
            fused_hidden: Some(vec![1.0; 10]),
        };
        let original = BatchInput { sequences: vec![seq] };
        let cloned = original.clone();
        assert_eq!(original.sequences.len(), cloned.sequences.len());
        assert_eq!(
            original.sequences[0].tokens,
            cloned.sequences[0].tokens
        );
        assert_eq!(
            original.sequences[0].page_table,
            cloned.sequences[0].page_table
        );
    }

    // =======================================================================
    // Tests for SequenceInput edge cases
    // =======================================================================

    #[test]
    fn sequence_input_position_zero() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.position, 0);
    }

    #[test]
    fn sequence_input_position_max() {
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
    fn sequence_input_draft_steps_zero() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.draft_steps, 0);
    }

    #[test]
    fn sequence_input_draft_steps_large() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: usize::MAX,
            page_table: None,
            fused_hidden: None,
        };
        assert_eq!(seq.draft_steps, usize::MAX);
    }

    #[test]
    fn sequence_input_large_fused_hidden() {
        let hidden = vec![0.123; 4096];
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(hidden.clone()),
        };
        assert_eq!(seq.fused_hidden.as_ref().unwrap().len(), 4096);
        assert_eq!(seq.fused_hidden.as_ref().unwrap()[0], 0.123);
    }

    #[test]
    fn sequence_input_fused_hidden_with_special_floats() {
        let hidden = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0];
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: Some(hidden),
        };
        let fh = seq.fused_hidden.as_ref().unwrap();
        assert!(fh[0].is_nan());
        assert!(fh[1].is_infinite() && fh[1].is_sign_positive());
        assert!(fh[2].is_infinite() && fh[2].is_sign_negative());
    }

    #[test]
    fn sequence_input_large_page_table() {
        let pt: Vec<u32> = (0..10000).collect();
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(pt),
            fused_hidden: None,
        };
        assert_eq!(seq.page_table.as_ref().unwrap().len(), 10000);
        assert!(seq.validate_page_table(10000).is_ok());
    }

    #[test]
    fn sequence_input_page_table_all_zeros() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 0, 0]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(1).is_ok());
    }

    #[test]
    fn sequence_input_debug_format() {
        let seq = SequenceInput {
            tokens: vec![1, 2],
            position: 5,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        let debug = format!("{seq:?}");
        assert!(debug.contains("SequenceInput"));
        assert!(debug.contains("tokens"));
        assert!(debug.contains("position"));
    }

    // =======================================================================
    // Tests for BatchResult factory methods
    // =======================================================================

    #[test]
    fn batch_result_fail_action_is_fail() {
        let result = BatchResult::fail(42);
        assert_eq!(result.request_id, 42);
        assert_eq!(result.action, BatchAction::Fail);
        assert!(result.generated_token.is_none());
    }

    #[test]
    fn batch_result_complete_with_token() {
        let telemetry = SequenceTelemetry::default();
        let result = BatchResult::complete(7, Some(99), telemetry);
        assert_eq!(result.request_id, 7);
        assert_eq!(result.action, BatchAction::Complete);
        assert_eq!(result.generated_token, Some(99));
    }

    #[test]
    fn batch_result_complete_without_token() {
        let telemetry = SequenceTelemetry::default();
        let result = BatchResult::complete(7, None, telemetry);
        assert_eq!(result.request_id, 7);
        assert_eq!(result.action, BatchAction::Complete);
        assert!(result.generated_token.is_none());
    }

    #[test]
    fn batch_result_continue_with_token() {
        let telemetry = SequenceTelemetry::default();
        let result = BatchResult::continue_with_token(3, 50, telemetry);
        assert_eq!(result.request_id, 3);
        assert_eq!(result.action, BatchAction::Continue);
        assert_eq!(result.generated_token, Some(50));
    }

    #[test]
    fn batch_result_pause() {
        let result = BatchResult::pause(15);
        assert_eq!(result.request_id, 15);
        assert_eq!(result.action, BatchAction::Pause);
        assert!(result.generated_token.is_none());
    }

    #[test]
    fn batch_result_request_id_zero() {
        let result = BatchResult::fail(0);
        assert_eq!(result.request_id, 0);
        assert_eq!(result.action, BatchAction::Fail);
    }

    #[test]
    fn batch_result_request_id_max() {
        let result = BatchResult::fail(RequestId::MAX);
        assert_eq!(result.request_id, RequestId::MAX);
    }

    // =======================================================================
    // Tests for BatchAction derive traits
    // =======================================================================

    #[test]
    fn batch_action_equality() {
        assert_eq!(BatchAction::Continue, BatchAction::Continue);
        assert_eq!(BatchAction::Complete, BatchAction::Complete);
        assert_eq!(BatchAction::Pause, BatchAction::Pause);
        assert_eq!(BatchAction::Fail, BatchAction::Fail);
    }

    #[test]
    fn batch_action_inequality() {
        assert_ne!(BatchAction::Continue, BatchAction::Complete);
        assert_ne!(BatchAction::Complete, BatchAction::Fail);
        assert_ne!(BatchAction::Pause, BatchAction::Fail);
        assert_ne!(BatchAction::Continue, BatchAction::Pause);
    }

    #[test]
    fn batch_action_copy_trait() {
        let a = BatchAction::Continue;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn batch_action_debug_format() {
        let debug = format!("{:?}", BatchAction::Continue);
        assert_eq!(debug, "Continue");
    }

    // =======================================================================
    // More extract_top_k_token_ids edge cases
    // =======================================================================

    #[test]
    fn extract_top_k_large_k_with_small_logits() {
        let logits = [1.0, 2.0];
        let result = extract_top_k_token_ids(&logits, 1000);
        assert_eq!(result.len(), 2, "k > len should return all elements");
        assert_eq!(result[0], 1, "index 1 has the highest value");
        assert_eq!(result[1], 0);
    }

    #[test]
    fn extract_top_k_descending_input() {
        let logits = [10.0, 5.0, 1.0];
        let result = extract_top_k_token_ids(&logits, 3);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn extract_top_k_ascending_input() {
        let logits = [1.0, 5.0, 10.0];
        let result = extract_top_k_token_ids(&logits, 3);
        assert_eq!(result, vec![2, 1, 0]);
    }

    #[test]
    fn extract_top_k_with_both_infinities() {
        let logits = [f32::NEG_INFINITY, 0.0, f32::INFINITY];
        let result = extract_top_k_token_ids(&logits, 2);
        assert_eq!(result[0], 2, "positive infinity should be first");
        assert_eq!(result[1], 1, "0.0 should be second");
    }

    #[test]
    fn extract_top_k_single_element_k_zero() {
        let logits = [5.0];
        let result = extract_top_k_token_ids(&logits, 0);
        assert!(result.is_empty());
    }

    // =======================================================================
    // More argmax_token edge cases
    // =======================================================================

    #[test]
    fn argmax_token_first_is_max() {
        let logits = [99.0, 0.1, 0.2, 0.3];
        assert_eq!(argmax_token(&logits), 0);
    }

    #[test]
    fn argmax_token_middle_is_max() {
        let logits = [0.1, 99.0, 0.2, 0.3];
        assert_eq!(argmax_token(&logits), 1);
    }

    #[test]
    fn argmax_token_with_neg_infinity() {
        let logits = [f32::NEG_INFINITY, 5.0, f32::NEG_INFINITY];
        assert_eq!(argmax_token(&logits), 1);
    }

    #[test]
    fn argmax_token_with_positive_infinity() {
        let logits = [1.0, f32::INFINITY, 3.0];
        assert_eq!(argmax_token(&logits), 1);
    }

    #[test]
    fn argmax_token_all_zeros() {
        let logits = [0.0, 0.0, 0.0, 0.0];
        let result = argmax_token(&logits);
        assert_eq!(result, 3, "all equal: last index wins");
    }

    #[test]
    fn argmax_token_two_elements() {
        let logits = [3.0, 7.0];
        assert_eq!(argmax_token(&logits), 1);
    }

    // =======================================================================
    // More shannon_entropy edge cases
    // =======================================================================

    #[test]
    fn shannon_entropy_three_uniform() {
        let h = shannon_entropy(&[0.0, 0.0, 0.0]);
        let expected = 3.0f32.ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "expected ~{expected}, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_binary_peaked() {
        // One dominant, one suppressed
        let h = shannon_entropy(&[10.0, -10.0]);
        assert!(
            h < 0.1,
            "binary peaked distribution should have near-zero entropy, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_negative_logits() {
        let h = shannon_entropy(&[-1.0, -1.0, -1.0]);
        assert!(
            h > 0.9,
            "uniform negative logits should have positive entropy, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_mixed_signs() {
        let h = shannon_entropy(&[-5.0, 0.0, 5.0]);
        assert!(h >= 0.0, "entropy should be non-negative, got {h}");
        assert!(h.is_finite(), "entropy should be finite");
    }

    #[test]
    fn shannon_entropy_doubling_uniform_doubles_entropy() {
        let h2 = shannon_entropy(&[0.0, 0.0]);
        let h4 = shannon_entropy(&[0.0, 0.0, 0.0, 0.0]);
        let ratio = h4 / h2;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "entropy ratio for 4 vs 2 uniform should be ~2.0, got {ratio}"
        );
    }

    // =======================================================================
    // BuildSequencesOutput edge cases
    // =======================================================================

    #[test]
    fn build_sequences_output_many_sequences() {
        let sequences: Vec<SequenceInput> = (0..100)
            .map(|i| SequenceInput {
                tokens: vec![i],
                position: i as usize,
                draft_steps: 0,
                page_table: None,
                fused_hidden: None,
            })
            .collect();
        let request_indices: Vec<RequestId> = (0..100).collect();
        let output = BuildSequencesOutput {
            sequences,
            request_indices,
            batch_results: vec![],
        };
        assert_eq!(output.sequences.len(), 100);
        assert_eq!(output.request_indices.len(), 100);
    }

    #[test]
    fn build_sequences_output_with_batch_results() {
        let results = vec![
            BatchResult::fail(1),
            BatchResult::fail(2),
        ];
        let output = BuildSequencesOutput {
            sequences: vec![],
            request_indices: vec![],
            batch_results: results,
        };
        assert_eq!(output.batch_results.len(), 2);
        assert_eq!(output.batch_results[0].request_id, 1);
        assert_eq!(output.batch_results[1].request_id, 2);
    }

    // =======================================================================
    // validate_page_table additional edge cases
    // =======================================================================

    #[test]
    fn validate_page_table_max_u32_single_page() {
        // page_table contains u32::MAX, total_pages = u32::MAX + 1 (overflow to 0 for u32)
        // total_pages is usize, so u32::MAX as usize + 1 is valid
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![u32::MAX]),
            fused_hidden: None,
        };
        // total_pages must be > u32::MAX for this to be valid
        let total_pages = u32::MAX as usize + 1;
        assert!(seq.validate_page_table(total_pages).is_ok());
    }

    #[test]
    fn validate_page_table_page_zero_with_one_total() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(1).is_ok());
    }

    #[test]
    fn validate_page_table_page_one_with_one_total_fails() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![1]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(1).is_err());
    }

    #[test]
    fn validate_page_table_duplicate_page_ids_valid() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![3, 3, 3]),
            fused_hidden: None,
        };
        // Duplicates are valid — same page referenced multiple times
        assert!(seq.validate_page_table(5).is_ok());
    }

    #[test]
    fn validate_page_table_last_valid_page_id() {
        let total = 100;
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![total as u32 - 1]),
            fused_hidden: None,
        };
        assert!(seq.validate_page_table(total).is_ok());
    }

    #[test]
    fn validate_page_table_error_message_format() {
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![5, 6, 100]),
            fused_hidden: None,
        };
        let err = seq.validate_page_table(50).unwrap_err();
        assert!(
            err.contains("page_table[2] = 100 >= total_pages 50"),
            "unexpected error: {err}"
        );
    }

    // =======================================================================
    // LogitsHandle edge cases
    // =======================================================================

    #[test]
    fn logits_handle_empty_data() {
        let handle = LogitsHandle { data: vec![] };
        assert!(handle.data.is_empty());
    }

    #[test]
    fn logits_handle_with_nan() {
        let handle = LogitsHandle {
            data: vec![f32::NAN, 1.0, 2.0],
        };
        assert!(handle.data[0].is_nan());
        assert_eq!(handle.data[1], 1.0);
    }

    #[test]
    fn logits_handle_with_infinity() {
        let handle = LogitsHandle {
            data: vec![f32::INFINITY, f32::NEG_INFINITY],
        };
        assert!(handle.data[0].is_infinite() && handle.data[0].is_sign_positive());
        assert!(handle.data[1].is_infinite() && handle.data[1].is_sign_negative());
    }

    #[test]
    fn logits_handle_large_data() {
        let data = vec![0.5; 100000];
        let handle = LogitsHandle { data };
        assert_eq!(handle.data.len(), 100000);
    }

    #[test]
    fn logits_handle_clone() {
        let handle = LogitsHandle {
            data: vec![1.0, 2.0, 3.0],
        };
        let cloned = handle.clone();
        assert_eq!(handle.data, cloned.data);
    }

    // =======================================================================
    // SequenceTelemetry default and construction
    // =======================================================================

    #[test]
    fn sequence_telemetry_default_values() {
        let tel = SequenceTelemetry::default();
        assert_eq!(tel.l2_delta, 0.0);
        assert!(!tel.has_outlier);
        assert_eq!(tel.dead_density, 0.0);
        assert_eq!(tel.per_head_entropy, 0.0);
        assert_eq!(tel.transform_ratio, 0.0);
        assert_eq!(tel.output_entropy, 0.0);
    }

    #[test]
    fn sequence_telemetry_new_equals_default() {
        assert_eq!(SequenceTelemetry::new(), SequenceTelemetry::default());
    }

    #[test]
    fn sequence_telemetry_custom_values() {
        let tel = SequenceTelemetry {
            l2_delta: 0.5,
            has_outlier: true,
            dead_density: 0.1,
            per_head_entropy: 2.3,
            transform_ratio: 0.95,
            output_entropy: 1.5,
        };
        assert_eq!(tel.l2_delta, 0.5);
        assert!(tel.has_outlier);
        assert_eq!(tel.dead_density, 0.1);
        assert_eq!(tel.per_head_entropy, 2.3);
        assert_eq!(tel.transform_ratio, 0.95);
        assert_eq!(tel.output_entropy, 1.5);
    }

    #[test]
    fn sequence_telemetry_equality() {
        let a = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: false,
            dead_density: 0.0,
            per_head_entropy: 0.5,
            transform_ratio: 0.8,
            output_entropy: 2.0,
        };
        let b = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: false,
            dead_density: 0.0,
            per_head_entropy: 0.5,
            transform_ratio: 0.8,
            output_entropy: 2.0,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn sequence_telemetry_inequality() {
        let a = SequenceTelemetry::default();
        let b = SequenceTelemetry {
            has_outlier: true,
            ..Default::default()
        };
        assert_ne!(a, b);
    }

    #[test]
    fn sequence_telemetry_copy_trait() {
        let a = SequenceTelemetry {
            l2_delta: 3.14,
            ..Default::default()
        };
        let b = a;
        assert_eq!(a.l2_delta, b.l2_delta);
    }

    #[test]
    fn sequence_telemetry_debug_format() {
        let tel = SequenceTelemetry::default();
        let debug = format!("{tel:?}");
        assert!(debug.contains("SequenceTelemetry"));
    }

    // =======================================================================
    // Tests for BackendError Display, Clone, Debug
    // =======================================================================

    #[test]
    fn backend_error_cuda_display() {
        let err = BackendError::Cuda("device lost".into());
        let msg = format!("{err}");
        assert!(msg.contains("CUDA error: device lost"), "unexpected display: {msg}");
    }

    #[test]
    fn backend_error_hip_display() {
        let err = BackendError::Hip("kernel fail".into());
        let msg = format!("{err}");
        assert!(msg.contains("HIP error: kernel fail"), "unexpected display: {msg}");
    }

    #[test]
    fn backend_error_metal_display() {
        let err = BackendError::Metal("shader crash".into());
        let msg = format!("{err}");
        assert!(msg.contains("Metal error: shader crash"), "unexpected display: {msg}");
    }

    #[test]
    fn backend_error_cpu_display() {
        let err = BackendError::Cpu("oom".into());
        let msg = format!("{err}");
        assert!(msg.contains("CPU error: oom"), "unexpected display: {msg}");
    }

    #[test]
    fn backend_error_unimplemented_display() {
        let err = BackendError::Unimplemented("fp8_gemm");
        let msg = format!("{err}");
        assert!(msg.contains("unimplemented: fp8_gemm"), "unexpected display: {msg}");
    }

    #[test]
    fn backend_error_other_display() {
        let err = BackendError::Other("misc".into());
        let msg = format!("{err}");
        assert!(msg.contains("backend error: misc"), "unexpected display: {msg}");
    }

    #[test]
    fn backend_error_clone_preserves_message() {
        let err = BackendError::Cuda("original".into());
        let cloned = err.clone();
        assert_eq!(format!("{err}"), format!("{cloned}"));
    }

    #[test]
    fn backend_error_debug_format() {
        let err = BackendError::Cuda("test".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Cuda"));
        assert!(debug.contains("test"));
    }

    // =======================================================================
    // Tests for KvCacheHandle
    // =======================================================================

    #[test]
    fn kv_cache_handle_default_value() {
        let h = KvCacheHandle(0);
        assert_eq!(h.0, 0);
    }

    #[test]
    fn kv_cache_handle_equality() {
        let a = KvCacheHandle(42);
        let b = KvCacheHandle(42);
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_handle_inequality() {
        let a = KvCacheHandle(1);
        let b = KvCacheHandle(2);
        assert_ne!(a, b);
    }

    #[test]
    fn kv_cache_handle_copy_trait() {
        let a = KvCacheHandle(99);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn kv_cache_handle_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(KvCacheHandle(10));
        assert!(set.contains(&KvCacheHandle(10)));
        assert!(!set.contains(&KvCacheHandle(11)));
    }

    #[test]
    fn kv_cache_handle_debug_format() {
        let h = KvCacheHandle(12345);
        let debug = format!("{h:?}");
        assert!(debug.contains("12345"));
    }

    // =======================================================================
    // Tests for AttentionMaskType
    // =======================================================================

    #[test]
    fn attention_mask_type_equality() {
        assert_eq!(AttentionMaskType::Bidirectional, AttentionMaskType::Bidirectional);
        assert_eq!(AttentionMaskType::Causal, AttentionMaskType::Causal);
    }

    #[test]
    fn attention_mask_type_inequality() {
        assert_ne!(AttentionMaskType::Bidirectional, AttentionMaskType::Causal);
    }

    #[test]
    fn attention_mask_type_copy_trait() {
        let a = AttentionMaskType::Causal;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn attention_mask_type_debug_format() {
        let debug_bi = format!("{:?}", AttentionMaskType::Bidirectional);
        assert_eq!(debug_bi, "Bidirectional");
        let debug_ca = format!("{:?}", AttentionMaskType::Causal);
        assert_eq!(debug_ca, "Causal");
    }

    #[test]
    fn attention_mask_type_hash_in_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(AttentionMaskType::Causal);
        assert!(set.contains(&AttentionMaskType::Causal));
        assert!(!set.contains(&AttentionMaskType::Bidirectional));
    }

    // =======================================================================
    // Tests for ExecutorError variants
    // =======================================================================

    #[test]
    fn executor_error_scheduler_display() {
        let err = ExecutorError::Scheduler("queue full".into());
        let msg = format!("{err}");
        assert!(msg.contains("scheduler error: queue full"), "unexpected: {msg}");
    }

    #[test]
    fn executor_error_empty_prompt_display() {
        let err = ExecutorError::EmptyPrompt;
        let msg = format!("{err}");
        assert!(msg.contains("empty prompt tokens"), "unexpected: {msg}");
    }

    #[test]
    fn executor_error_empty_sample_display() {
        let err = ExecutorError::EmptySample;
        let msg = format!("{err}");
        assert!(msg.contains("backend returned empty sample"), "unexpected: {msg}");
    }

    #[test]
    fn executor_error_request_not_found_display() {
        let err = ExecutorError::RequestNotFound { request_id: 42 };
        let msg = format!("{err}");
        assert!(msg.contains("request not found: 42"), "unexpected: {msg}");
    }

    #[test]
    fn executor_error_compilation_display() {
        let err = ExecutorError::Compilation("codegen failed".into());
        let msg = format!("{err}");
        assert!(msg.contains("JIT compilation failed: codegen failed"), "unexpected: {msg}");
    }

    #[test]
    fn executor_error_graph_expansion_display() {
        let err = ExecutorError::GraphExpansion("bad graph".into());
        let msg = format!("{err}");
        assert!(msg.contains("graph expansion failed: bad graph"), "unexpected: {msg}");
    }

    #[test]
    fn executor_error_from_backend_error() {
        let backend_err = BackendError::Other("test".into());
        let exec_err: ExecutorError = backend_err.into();
        let msg = format!("{exec_err}");
        assert!(msg.contains("backend error: test"), "unexpected: {msg}");
    }

    #[test]
    fn executor_error_debug_format() {
        let err = ExecutorError::Scheduler("msg".into());
        let debug = format!("{err:?}");
        assert!(debug.contains("Scheduler"));
    }

    // =======================================================================
    // Tests for VirtualPageId
    // =======================================================================

    #[test]
    fn virtual_page_id_new() {
        let vpid = VirtualPageId::new(42, 7);
        assert_eq!(vpid.sequence_id, 42);
        assert_eq!(vpid.logical_index, 7);
    }

    #[test]
    fn virtual_page_id_equality() {
        let a = VirtualPageId::new(1, 2);
        let b = VirtualPageId::new(1, 2);
        assert_eq!(a, b);
    }

    #[test]
    fn virtual_page_id_inequality_different_seq() {
        let a = VirtualPageId::new(1, 2);
        let b = VirtualPageId::new(99, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn virtual_page_id_inequality_different_logical() {
        let a = VirtualPageId::new(1, 2);
        let b = VirtualPageId::new(1, 3);
        assert_ne!(a, b);
    }

    #[test]
    fn virtual_page_id_copy_trait() {
        let a = VirtualPageId::new(5, 10);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn virtual_page_id_zero_values() {
        let vpid = VirtualPageId::new(0, 0);
        assert_eq!(vpid.sequence_id, 0);
        assert_eq!(vpid.logical_index, 0);
    }

    #[test]
    fn virtual_page_id_max_values() {
        let vpid = VirtualPageId::new(RequestId::MAX, usize::MAX);
        assert_eq!(vpid.sequence_id, RequestId::MAX);
        assert_eq!(vpid.logical_index, usize::MAX);
    }

    #[test]
    fn virtual_page_id_hash_in_map() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let vpid = VirtualPageId::new(10, 20);
        map.insert(vpid, "test");
        assert_eq!(map.get(&VirtualPageId::new(10, 20)), Some(&"test"));
        assert_eq!(map.get(&VirtualPageId::new(10, 21)), None);
    }

    #[test]
    fn virtual_page_id_debug_format() {
        let vpid = VirtualPageId::new(42, 7);
        let debug = format!("{vpid:?}");
        assert!(debug.contains("VirtualPageId"));
    }

    // =======================================================================
    // Tests for Tier enum
    // =======================================================================

    #[test]
    fn tier_equality() {
        assert_eq!(Tier::L1, Tier::L1);
        assert_eq!(Tier::L2, Tier::L2);
        assert_eq!(Tier::L3, Tier::L3);
    }

    #[test]
    fn tier_inequality() {
        assert_ne!(Tier::L1, Tier::L2);
        assert_ne!(Tier::L2, Tier::L3);
        assert_ne!(Tier::L1, Tier::L3);
    }

    #[test]
    fn tier_copy_trait() {
        let a = Tier::L1;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn tier_debug_format() {
        assert_eq!(format!("{:?}", Tier::L1), "L1");
        assert_eq!(format!("{:?}", Tier::L2), "L2");
        assert_eq!(format!("{:?}", Tier::L3), "L3");
    }

    #[test]
    fn tier_hash_in_set() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Tier::L1);
        set.insert(Tier::L2);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Tier::L1));
        assert!(set.contains(&Tier::L2));
        assert!(!set.contains(&Tier::L3));
    }

    // =======================================================================
    // Tests for ScheduledBatch construction
    // =======================================================================

    #[test]
    fn scheduled_batch_empty() {
        let batch = ScheduledBatch {
            requests: vec![],
            seq_offsets: vec![],
            draft_steps: vec![],
        };
        assert!(batch.requests.is_empty());
        assert!(batch.seq_offsets.is_empty());
        assert!(batch.draft_steps.is_empty());
    }

    #[test]
    fn scheduled_batch_with_requests() {
        let batch = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 10, 20],
            draft_steps: vec![0, 0, 4],
        };
        assert_eq!(batch.requests.len(), 3);
        assert_eq!(batch.seq_offsets, vec![0, 10, 20]);
        assert_eq!(batch.draft_steps[2], 4);
    }

    #[test]
    fn scheduled_batch_clone() {
        let batch = ScheduledBatch {
            requests: vec![42],
            seq_offsets: vec![0],
            draft_steps: vec![2],
        };
        let cloned = batch.clone();
        assert_eq!(batch.requests, cloned.requests);
        assert_eq!(batch.seq_offsets, cloned.seq_offsets);
        assert_eq!(batch.draft_steps, cloned.draft_steps);
    }

    // =======================================================================
    // Tests for TierUsage
    // =======================================================================

    #[test]
    fn tier_usage_available_with_capacity() {
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 30,
            capacity: 100,
        };
        assert_eq!(usage.available(), 70);
    }

    #[test]
    fn tier_usage_available_zero_capacity() {
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 0,
            capacity: 0,
        };
        assert_eq!(usage.available(), 0);
    }

    #[test]
    fn tier_usage_available_over_capacity() {
        // used > capacity should not panic due to saturating_sub
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 150,
            capacity: 100,
        };
        assert_eq!(usage.available(), 0);
    }

    #[test]
    fn tier_usage_available_fully_used() {
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 100,
            capacity: 100,
        };
        assert_eq!(usage.available(), 0);
    }

    #[test]
    fn tier_usage_available_none_used() {
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 0,
            capacity: 50,
        };
        assert_eq!(usage.available(), 50);
    }

    // =======================================================================
    // Tests for RequestId type alias
    // =======================================================================

    #[test]
    fn request_id_is_u64() {
        let rid: RequestId = 42u64;
        assert_eq!(rid, 42u64);
    }

    #[test]
    fn request_id_max() {
        let rid: RequestId = RequestId::MAX;
        assert_eq!(rid, u64::MAX);
    }

    #[test]
    fn request_id_zero() {
        let rid: RequestId = 0;
        assert_eq!(rid, 0);
    }

    // =======================================================================
    // InterleavedSlot construction and Debug
    // =======================================================================

    #[test]
    fn interleaved_slot_construction() {
        let slot = crate::scheduler::batcher::InterleavedSlot {
            request_id: 42,
            batch_index: 3,
            token_count: 128,
            draft_steps: 2,
        };
        assert_eq!(slot.request_id, 42);
        assert_eq!(slot.batch_index, 3);
        assert_eq!(slot.token_count, 128);
        assert_eq!(slot.draft_steps, 2);
    }

    #[test]
    fn interleaved_slot_debug_format() {
        let slot = crate::scheduler::batcher::InterleavedSlot {
            request_id: 1,
            batch_index: 0,
            token_count: 1,
            draft_steps: 0,
        };
        let debug = format!("{slot:?}");
        assert!(debug.contains("InterleavedSlot"), "debug should contain type name: {debug}");
    }

    #[test]
    fn interleaved_slot_clone() {
        let slot = crate::scheduler::batcher::InterleavedSlot {
            request_id: 99,
            batch_index: 5,
            token_count: 64,
            draft_steps: 4,
        };
        let cloned = slot.clone();
        assert_eq!(slot.request_id, cloned.request_id);
        assert_eq!(slot.batch_index, cloned.batch_index);
        assert_eq!(slot.token_count, cloned.token_count);
        assert_eq!(slot.draft_steps, cloned.draft_steps);
    }

    #[test]
    fn interleaved_slot_zero_token_count() {
        let slot = crate::scheduler::batcher::InterleavedSlot {
            request_id: 0,
            batch_index: 0,
            token_count: 0,
            draft_steps: 0,
        };
        assert_eq!(slot.token_count, 0);
    }

    // =======================================================================
    // InterleavedBatch construction and methods
    // =======================================================================

    #[test]
    fn interleaved_batch_empty() {
        let ib = crate::scheduler::batcher::InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![],
                seq_offsets: vec![],
                draft_steps: vec![],
            },
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        assert_eq!(ib.decode_tokens(), 0);
        assert_eq!(ib.prefill_tokens(), 0);
        assert_eq!(ib.total_tokens(), 0);
        assert!(!ib.is_interleaved());
        assert!(ib.request_ids().is_empty());
    }

    #[test]
    fn interleaved_batch_decode_only() {
        let ib = crate::scheduler::batcher::InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![1, 2],
                seq_offsets: vec![0, 1],
                draft_steps: vec![0, 0],
            },
            decode_slots: vec![
                crate::scheduler::batcher::InterleavedSlot {
                    request_id: 1,
                    batch_index: 0,
                    token_count: 1,
                    draft_steps: 0,
                },
                crate::scheduler::batcher::InterleavedSlot {
                    request_id: 2,
                    batch_index: 1,
                    token_count: 1,
                    draft_steps: 0,
                },
            ],
            prefill_slots: vec![],
        };
        assert_eq!(ib.decode_tokens(), 2);
        assert_eq!(ib.prefill_tokens(), 0);
        assert_eq!(ib.total_tokens(), 2);
        assert!(!ib.is_interleaved());
        assert_eq!(ib.request_ids(), &[1, 2]);
    }

    #[test]
    fn interleaved_batch_prefill_only() {
        let ib = crate::scheduler::batcher::InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![3],
                seq_offsets: vec![0],
                draft_steps: vec![0],
            },
            decode_slots: vec![],
            prefill_slots: vec![
                crate::scheduler::batcher::InterleavedSlot {
                    request_id: 3,
                    batch_index: 0,
                    token_count: 512,
                    draft_steps: 0,
                },
            ],
        };
        assert_eq!(ib.decode_tokens(), 0);
        assert_eq!(ib.prefill_tokens(), 512);
        assert_eq!(ib.total_tokens(), 512);
        assert!(!ib.is_interleaved());
    }

    #[test]
    fn interleaved_batch_truly_interleaved() {
        let ib = crate::scheduler::batcher::InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![1, 2],
                seq_offsets: vec![0, 1],
                draft_steps: vec![0, 0],
            },
            decode_slots: vec![
                crate::scheduler::batcher::InterleavedSlot {
                    request_id: 1,
                    batch_index: 0,
                    token_count: 1,
                    draft_steps: 0,
                },
            ],
            prefill_slots: vec![
                crate::scheduler::batcher::InterleavedSlot {
                    request_id: 2,
                    batch_index: 1,
                    token_count: 256,
                    draft_steps: 0,
                },
            ],
        };
        assert_eq!(ib.decode_tokens(), 1);
        assert_eq!(ib.prefill_tokens(), 256);
        assert_eq!(ib.total_tokens(), 257);
        assert!(ib.is_interleaved());
    }

    #[test]
    fn interleaved_batch_debug_format() {
        let ib = crate::scheduler::batcher::InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![1],
                seq_offsets: vec![0],
                draft_steps: vec![0],
            },
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        let debug = format!("{ib:?}");
        assert!(debug.contains("InterleavedBatch"), "debug should contain type name: {debug}");
    }

    #[test]
    fn interleaved_batch_clone() {
        let ib = crate::scheduler::batcher::InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![10, 20],
                seq_offsets: vec![0, 5],
                draft_steps: vec![1, 0],
            },
            decode_slots: vec![
                crate::scheduler::batcher::InterleavedSlot {
                    request_id: 10,
                    batch_index: 0,
                    token_count: 1,
                    draft_steps: 1,
                },
            ],
            prefill_slots: vec![],
        };
        let cloned = ib.clone();
        assert_eq!(ib.decode_tokens(), cloned.decode_tokens());
        assert_eq!(ib.request_ids(), cloned.request_ids());
    }

    // =======================================================================
    // BatchPrepData construction and methods
    // =======================================================================

    #[test]
    fn batch_prep_data_new_zero_seqs() {
        let prep = crate::scheduler::batcher::BatchPrepData::new(0);
        assert!(prep.prompt_lens.is_empty());
        assert!(prep.kv_lens.is_empty());
        assert!(prep.active_flags.is_empty());
        assert_eq!(prep.max_decode_steps, 0);
        assert_eq!(prep.total_prefill_tokens, 0);
    }

    #[test]
    fn batch_prep_data_new_single_seq() {
        let prep = crate::scheduler::batcher::BatchPrepData::new(1);
        assert_eq!(prep.prompt_lens.len(), 1);
        assert_eq!(prep.prompt_lens[0], 0);
        assert_eq!(prep.active_flags.len(), 1);
        assert_eq!(prep.active_flags[0], 1, "default active flag should be 1");
        assert_eq!(prep.sampling_params_packed.len(), 4);
    }

    #[test]
    fn batch_prep_data_new_multiple_seqs() {
        let prep = crate::scheduler::batcher::BatchPrepData::new(8);
        assert_eq!(prep.prompt_lens.len(), 8);
        assert_eq!(prep.sampling_params_packed.len(), 32);
        assert!(prep.active_flags.iter().all(|&f| f == 1));
    }

    #[test]
    fn batch_prep_data_set_sampling_params() {
        let mut prep = crate::scheduler::batcher::BatchPrepData::new(2);
        prep.set_sampling_params(0, 0.7, 50, 0.9, 2);
        let base = 0;
        assert_eq!(prep.sampling_params_packed[base], 0.7f32.to_bits());
        assert_eq!(prep.sampling_params_packed[base + 1], 50);
        assert_eq!(prep.sampling_params_packed[base + 2], 0.9f32.to_bits());
        assert_eq!(prep.sampling_params_packed[base + 3], 2);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_second_seq() {
        let mut prep = crate::scheduler::batcher::BatchPrepData::new(3);
        prep.set_sampling_params(2, 1.5, 100, 0.95, 501);
        let base = 2 * 4;
        assert_eq!(prep.sampling_params_packed[base], 1.5f32.to_bits());
        assert_eq!(prep.sampling_params_packed[base + 1], 100);
        assert_eq!(prep.sampling_params_packed[base + 2], 0.95f32.to_bits());
        assert_eq!(prep.sampling_params_packed[base + 3], 501);
    }

    #[test]
    fn batch_prep_data_debug_format() {
        let prep = crate::scheduler::batcher::BatchPrepData::new(1);
        let debug = format!("{prep:?}");
        assert!(debug.contains("BatchPrepData"), "debug should contain type name: {debug}");
    }

    #[test]
    fn batch_prep_data_clone() {
        let prep = crate::scheduler::batcher::BatchPrepData::new(2);
        let cloned = prep.clone();
        assert_eq!(prep.prompt_lens, cloned.prompt_lens);
        assert_eq!(prep.active_flags, cloned.active_flags);
        assert_eq!(prep.sampling_params_packed, cloned.sampling_params_packed);
    }

    // =======================================================================
    // PrefillPlan enum variants and equality
    // =======================================================================

    #[test]
    fn prefill_plan_fully_resident_equality() {
        let a = PrefillPlan::FullyResident { pages: 10 };
        let b = PrefillPlan::FullyResident { pages: 10 };
        assert_eq!(a, b);
    }

    #[test]
    fn prefill_plan_fully_resident_inequality() {
        let a = PrefillPlan::FullyResident { pages: 10 };
        let b = PrefillPlan::FullyResident { pages: 20 };
        assert_ne!(a, b);
    }

    #[test]
    fn prefill_plan_pipelined_equality() {
        let a = PrefillPlan::Pipelined {
            l1_pages: 4,
            l2_prefetch: 8,
            chunk_schedule: vec![2, 4, 2],
        };
        let b = PrefillPlan::Pipelined {
            l1_pages: 4,
            l2_prefetch: 8,
            chunk_schedule: vec![2, 4, 2],
        };
        assert_eq!(a, b);
    }

    #[test]
    fn prefill_plan_different_variants_inequality() {
        let a = PrefillPlan::FullyResident { pages: 5 };
        let b = PrefillPlan::Pipelined {
            l1_pages: 5,
            l2_prefetch: 0,
            chunk_schedule: vec![],
        };
        assert_ne!(a, b);
    }

    #[test]
    fn prefill_plan_debug_format() {
        let plan = PrefillPlan::FullyResident { pages: 42 };
        let debug = format!("{plan:?}");
        assert!(debug.contains("FullyResident"), "unexpected debug: {debug}");
    }

    #[test]
    fn prefill_plan_pipelined_empty_schedule() {
        let plan = PrefillPlan::Pipelined {
            l1_pages: 0,
            l2_prefetch: 0,
            chunk_schedule: vec![],
        };
        let cloned = plan.clone();
        assert_eq!(plan, cloned);
    }

    // =======================================================================
    // PositionEncoding enum
    // =======================================================================

    #[test]
    fn position_encoding_equality() {
        assert_eq!(
            crate::engine::executor_types::PositionEncoding::None,
            crate::engine::executor_types::PositionEncoding::None
        );
        assert_eq!(
            crate::engine::executor_types::PositionEncoding::Rope,
            crate::engine::executor_types::PositionEncoding::Rope
        );
    }

    #[test]
    fn position_encoding_inequality() {
        assert_ne!(
            crate::engine::executor_types::PositionEncoding::None,
            crate::engine::executor_types::PositionEncoding::Rope
        );
    }

    #[test]
    fn position_encoding_debug_format() {
        let none_debug = format!("{:?}", crate::engine::executor_types::PositionEncoding::None);
        assert_eq!(none_debug, "None");
        let rope_debug = format!("{:?}", crate::engine::executor_types::PositionEncoding::Rope);
        assert_eq!(rope_debug, "Rope");
    }

    // =======================================================================
    // SamplingConfig default and construction
    // =======================================================================

    #[test]
    fn sampling_config_default_values() {
        let config = crate::engine::executor_types::SamplingConfig::default();
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.top_k, 0);
        assert_eq!(config.top_p, 1.0);
    }

    #[test]
    fn sampling_config_custom_values() {
        let config = crate::engine::executor_types::SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        };
        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_k, 50);
        assert_eq!(config.top_p, 0.9);
    }

    #[test]
    fn sampling_config_copy_trait() {
        let a = crate::engine::executor_types::SamplingConfig {
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
        };
        let b = a;
        assert_eq!(a.temperature, b.temperature);
        assert_eq!(a.top_k, b.top_k);
        assert_eq!(a.top_p, b.top_p);
    }

    #[test]
    fn sampling_config_debug_format() {
        let config = crate::engine::executor_types::SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("SamplingConfig"), "debug should contain type name: {debug}");
    }

    #[test]
    fn sampling_config_zero_temperature() {
        let config = crate::engine::executor_types::SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
        };
        assert_eq!(config.temperature, 0.0);
    }

    // =======================================================================
    // RoPEConfig construction and equality
    // =======================================================================

    #[test]
    fn rope_config_equality() {
        let a = crate::engine::executor_types::RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let b = crate::engine::executor_types::RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn rope_config_inequality() {
        let a = crate::engine::executor_types::RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let b = crate::engine::executor_types::RoPEConfig {
            theta: 500000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn rope_config_copy_trait() {
        let a = crate::engine::executor_types::RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: true,
            precompute: true,
        };
        let b = a;
        assert_eq!(a.theta, b.theta);
        assert_eq!(a.scale, b.scale);
        assert_eq!(a.interleaved, b.interleaved);
        assert_eq!(a.precompute, b.precompute);
    }

    #[test]
    fn rope_config_debug_format() {
        let config = crate::engine::executor_types::RoPEConfig {
            theta: 10000.0,
            scale: 1.0,
            interleaved: false,
            precompute: false,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("RoPEConfig"), "debug should contain type name: {debug}");
    }

    // =======================================================================
    // AttentionHeadConfig construction
    // =======================================================================

    #[test]
    fn attention_head_config_construction() {
        let config = crate::engine::executor_types::AttentionHeadConfig {
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
        };
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.num_kv_heads, 8);
        assert_eq!(config.head_dim, 128);
    }

    #[test]
    fn attention_head_config_debug_format() {
        let config = crate::engine::executor_types::AttentionHeadConfig {
            num_heads: 16,
            num_kv_heads: 4,
            head_dim: 64,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("AttentionHeadConfig"), "debug should contain type name: {debug}");
    }

    #[test]
    fn attention_head_config_copy_trait() {
        let a = crate::engine::executor_types::AttentionHeadConfig {
            num_heads: 8,
            num_kv_heads: 2,
            head_dim: 32,
        };
        let b = a;
        assert_eq!(a.num_heads, b.num_heads);
        assert_eq!(a.num_kv_heads, b.num_kv_heads);
        assert_eq!(a.head_dim, b.head_dim);
    }

    // =======================================================================
    // PagedKvConfig construction
    // =======================================================================

    #[test]
    fn paged_kv_config_no_page_table() {
        let config = crate::engine::executor_types::PagedKvConfig {
            page_table: None,
            page_size: 16,
        };
        assert!(config.page_table.is_none());
        assert_eq!(config.page_size, 16);
    }

    #[test]
    fn paged_kv_config_with_page_table() {
        let config = crate::engine::executor_types::PagedKvConfig {
            page_table: Some(vec![0, 1, 2, 3]),
            page_size: 32,
        };
        assert_eq!(config.page_table.as_ref().unwrap().len(), 4);
        assert_eq!(config.page_size, 32);
    }

    #[test]
    fn paged_kv_config_debug_format() {
        let config = crate::engine::executor_types::PagedKvConfig {
            page_table: None,
            page_size: 64,
        };
        let debug = format!("{config:?}");
        assert!(debug.contains("PagedKvConfig"), "debug should contain type name: {debug}");
    }

    #[test]
    fn paged_kv_config_clone() {
        let config = crate::engine::executor_types::PagedKvConfig {
            page_table: Some(vec![10, 20, 30]),
            page_size: 16,
        };
        let cloned = config.clone();
        assert_eq!(config.page_size, cloned.page_size);
        assert_eq!(config.page_table, cloned.page_table);
    }

    // =======================================================================
    // BatchResult Copy and Clone
    // =======================================================================

    #[test]
    fn batch_result_copy_trait() {
        let result = BatchResult::continue_with_token(5, 100, SequenceTelemetry::default());
        let copied = result;
        assert_eq!(result.request_id, copied.request_id);
        assert_eq!(result.action, copied.action);
        assert_eq!(result.generated_token, copied.generated_token);
    }

    #[test]
    fn batch_result_debug_format() {
        let result = BatchResult::fail(42);
        let debug = format!("{result:?}");
        assert!(debug.contains("BatchResult"), "debug should contain type name: {debug}");
        assert!(debug.contains("42"));
    }

    #[test]
    fn batch_result_clone() {
        let result = BatchResult::complete(1, Some(99), SequenceTelemetry::default());
        let cloned = result.clone();
        assert_eq!(result.request_id, cloned.request_id);
        assert_eq!(result.action, cloned.action);
        assert_eq!(result.generated_token, cloned.generated_token);
    }

    #[test]
    fn batch_result_all_action_types() {
        let cont = BatchResult::continue_with_token(1, 10, SequenceTelemetry::default());
        assert_eq!(cont.action, BatchAction::Continue);
        let complete = BatchResult::complete(2, Some(10), SequenceTelemetry::default());
        assert_eq!(complete.action, BatchAction::Complete);
        let pause = BatchResult::pause(3);
        assert_eq!(pause.action, BatchAction::Pause);
        let fail = BatchResult::fail(4);
        assert_eq!(fail.action, BatchAction::Fail);
    }

    // =======================================================================
    // ScheduledBatch Debug
    // =======================================================================

    #[test]
    fn scheduled_batch_debug_format() {
        let batch = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 10, 20],
            draft_steps: vec![0, 0, 4],
        };
        let debug = format!("{batch:?}");
        assert!(debug.contains("ScheduledBatch"), "debug should contain type name: {debug}");
    }

    // =======================================================================
    // SessionId type alias
    // =======================================================================

    #[test]
    fn session_id_is_u64() {
        let sid: crate::scheduler::memory_manager::SessionId = 123u64;
        assert_eq!(sid, 123u64);
    }

    #[test]
    fn session_id_zero() {
        let sid: crate::scheduler::memory_manager::SessionId = 0;
        assert_eq!(sid, 0);
    }

    #[test]
    fn session_id_max() {
        let sid: crate::scheduler::memory_manager::SessionId = u64::MAX;
        assert_eq!(sid, u64::MAX);
    }

    // =======================================================================
    // TierUsage additional edge cases
    // =======================================================================

    #[test]
    fn tier_usage_clone() {
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 50,
            capacity: 100,
        };
        let cloned = usage.clone();
        assert_eq!(usage.used, cloned.used);
        assert_eq!(usage.capacity, cloned.capacity);
    }

    #[test]
    fn tier_usage_equality() {
        let a = crate::scheduler::memory_manager::TierUsage { used: 10, capacity: 20 };
        let b = crate::scheduler::memory_manager::TierUsage { used: 10, capacity: 20 };
        assert_eq!(a, b);
    }

    #[test]
    fn tier_usage_inequality() {
        let a = crate::scheduler::memory_manager::TierUsage { used: 10, capacity: 20 };
        let b = crate::scheduler::memory_manager::TierUsage { used: 15, capacity: 20 };
        assert_ne!(a, b);
    }

    #[test]
    fn tier_usage_copy_trait() {
        let a = crate::scheduler::memory_manager::TierUsage { used: 42, capacity: 100 };
        let b = a;
        assert_eq!(a.used, b.used);
        assert_eq!(a.capacity, b.capacity);
    }

    #[test]
    fn tier_usage_debug_format() {
        let usage = crate::scheduler::memory_manager::TierUsage { used: 42, capacity: 100 };
        let debug = format!("{usage:?}");
        assert!(debug.contains("TierUsage"), "debug should contain type name: {debug}");
    }

    // =======================================================================
    // Shannon entropy additional edge cases
    // =======================================================================

    #[test]
    fn shannon_entropy_many_uniform_1024() {
        let logits = vec![1.0f32; 1024];
        let h = shannon_entropy(&logits);
        let expected = 1024.0f32.ln();
        assert!(
            (h - expected).abs() < 0.1,
            "expected ~{expected}, got {h}"
        );
    }

    #[test]
    fn shannon_entropy_two_very_peaked() {
        let h = shannon_entropy(&[1000.0, -1000.0]);
        assert!(h < 0.001, "very peaked binary should have near-zero entropy, got {h}");
    }

    #[test]
    fn shannon_entropy_gradient_like() {
        // Linearly spaced logits: entropy should be moderate
        let logits: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let h = shannon_entropy(&logits);
        assert!(h > 0.0, "gradient logits should have positive entropy");
        assert!(h < 10.0f32.ln() + 0.1, "entropy should be less than uniform");
    }

    // =======================================================================
    // extract_top_k_token_ids additional edge cases
    // =======================================================================

    #[test]
    fn extract_top_k_all_same_value() {
        let logits = [3.0, 3.0, 3.0, 3.0, 3.0];
        let top3 = extract_top_k_token_ids(&logits, 3);
        assert_eq!(top3.len(), 3);
        for &id in &top3 {
            assert!(id < 5);
        }
    }

    #[test]
    fn extract_top_k_very_large_k_returns_all() {
        let logits = [1.0, 2.0, 3.0];
        let result = extract_top_k_token_ids(&logits, 1000000);
        assert_eq!(result.len(), 3);
    }

    // =======================================================================
    // argmax_token additional edge cases
    // =======================================================================

    #[test]
    fn argmax_token_alternating_pattern() {
        let logits = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        // Last max (index 5) wins due to max_by + partial_cmp(Equal)
        assert_eq!(argmax_token(&logits), 5);
    }

    #[test]
    fn argmax_token_very_small_positive() {
        let logits = [f32::MIN_POSITIVE, 0.0, 0.0];
        assert_eq!(argmax_token(&logits), 0);
    }

    // =======================================================================
    // VirtualPageId additional tests
    // =======================================================================

    #[test]
    fn virtual_page_id_clone() {
        let vpid = VirtualPageId::new(42, 7);
        let cloned = vpid.clone();
        assert_eq!(vpid, cloned);
    }

    #[test]
    fn virtual_page_id_in_hashset() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(VirtualPageId::new(1, 0));
        set.insert(VirtualPageId::new(1, 1));
        set.insert(VirtualPageId::new(2, 0));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&VirtualPageId::new(1, 0)));
        assert!(set.contains(&VirtualPageId::new(1, 1)));
        assert!(set.contains(&VirtualPageId::new(2, 0)));
    }

    // =======================================================================
    // BackendError additional tests
    // =======================================================================

    #[test]
    fn backend_error_all_variants_display() {
        let variants: Vec<BackendError> = vec![
            BackendError::Cuda("c".into()),
            BackendError::Hip("h".into()),
            BackendError::Metal("m".into()),
            BackendError::Cpu("p".into()),
            BackendError::Unimplemented("u"),
            BackendError::Other("o".into()),
        ];
        for err in &variants {
            let msg = format!("{err}");
            assert!(!msg.is_empty(), "Display should produce non-empty string for {err:?}");
        }
    }

    #[test]
    fn backend_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(BackendError::Other("test".into()));
        let msg = format!("{err}");
        assert!(msg.contains("backend error: test"));
    }

    // =======================================================================
    // ExecutorError additional tests
    // =======================================================================

    #[test]
    fn executor_error_backend_display() {
        let err = ExecutorError::Backend(BackendError::Cuda("timeout".into()));
        let msg = format!("{err}");
        assert!(msg.contains("CUDA error: timeout"), "unexpected: {msg}");
    }

    #[test]
    fn executor_error_is_std_error() {
        let err: Box<dyn std::error::Error> = Box::new(ExecutorError::EmptyPrompt);
        let msg = format!("{err}");
        assert!(msg.contains("empty prompt"));
    }

    #[test]
    fn executor_error_from_backend_error_conversion() {
        let backend_err = BackendError::Hip("bad kernel".into());
        let exec_err: ExecutorError = backend_err.into();
        match exec_err {
            ExecutorError::Backend(_) => {}
            other => panic!("expected Backend variant, got {other:?}"),
        }
    }

    // =======================================================================
    // PageState enum derive tests
    // =======================================================================

    #[test]
    fn page_state_equality() {
        use crate::scheduler::types::PageState;
        assert_eq!(PageState::Free, PageState::Free);
        assert_eq!(PageState::Active, PageState::Active);
        assert_eq!(PageState::Standby, PageState::Standby);
        assert_eq!(PageState::SwappedOut, PageState::SwappedOut);
        assert_eq!(PageState::Warm, PageState::Warm);
        assert_eq!(PageState::Protected, PageState::Protected);
        assert_eq!(PageState::Swapped, PageState::Swapped);
    }

    #[test]
    fn page_state_inequality() {
        use crate::scheduler::types::PageState;
        assert_ne!(PageState::Free, PageState::Active);
        assert_ne!(PageState::Active, PageState::Standby);
        assert_ne!(PageState::SwappedOut, PageState::Swapped);
        assert_ne!(PageState::Warm, PageState::Protected);
    }

    #[test]
    fn page_state_copy_trait() {
        use crate::scheduler::types::PageState;
        let a = PageState::Active;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn page_state_hash_in_set() {
        use crate::scheduler::types::PageState;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PageState::Active);
        set.insert(PageState::Free);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&PageState::Active));
        assert!(set.contains(&PageState::Free));
        assert!(!set.contains(&PageState::Standby));
    }

    #[test]
    fn page_state_debug_format() {
        use crate::scheduler::types::PageState;
        assert_eq!(format!("{:?}", PageState::Free), "Free");
        assert_eq!(format!("{:?}", PageState::Active), "Active");
        assert_eq!(format!("{:?}", PageState::SwappedOut), "SwappedOut");
        assert_eq!(format!("{:?}", PageState::Protected), "Protected");
    }

    // =======================================================================
    // GroupState enum derive tests
    // =======================================================================

    #[test]
    fn group_state_equality() {
        use crate::scheduler::types::GroupState;
        assert_eq!(GroupState::Running, GroupState::Running);
        assert_eq!(GroupState::Swapped, GroupState::Swapped);
        assert_eq!(GroupState::Paused, GroupState::Paused);
    }

    #[test]
    fn group_state_inequality() {
        use crate::scheduler::types::GroupState;
        assert_ne!(GroupState::Running, GroupState::Swapped);
        assert_ne!(GroupState::Swapped, GroupState::Paused);
        assert_ne!(GroupState::Running, GroupState::Paused);
    }

    #[test]
    fn group_state_copy_trait() {
        use crate::scheduler::types::GroupState;
        let a = GroupState::Running;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn group_state_debug_format() {
        use crate::scheduler::types::GroupState;
        assert_eq!(format!("{:?}", GroupState::Running), "Running");
        assert_eq!(format!("{:?}", GroupState::Swapped), "Swapped");
        assert_eq!(format!("{:?}", GroupState::Paused), "Paused");
    }

    // =======================================================================
    // RequestKind enum derive tests
    // =======================================================================

    #[test]
    fn request_kind_equality() {
        use crate::scheduler::types::RequestKind;
        assert_eq!(RequestKind::Chat, RequestKind::Chat);
        assert_eq!(RequestKind::Embedding, RequestKind::Embedding);
        assert_eq!(RequestKind::Rerank, RequestKind::Rerank);
    }

    #[test]
    fn request_kind_inequality() {
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
    fn request_kind_debug_format() {
        use crate::scheduler::types::RequestKind;
        assert_eq!(format!("{:?}", RequestKind::Chat), "Chat");
        assert_eq!(format!("{:?}", RequestKind::Embedding), "Embedding");
        assert_eq!(format!("{:?}", RequestKind::Rerank), "Rerank");
    }

    // =======================================================================
    // BatchOrderPolicy enum derive tests
    // =======================================================================

    #[test]
    fn batch_order_policy_default_is_strict() {
        use crate::scheduler::types::BatchOrderPolicy;
        assert_eq!(BatchOrderPolicy::default(), BatchOrderPolicy::StrictRequestIdOrder);
    }

    #[test]
    fn batch_order_policy_equality() {
        use crate::scheduler::types::BatchOrderPolicy;
        assert_eq!(BatchOrderPolicy::StrictRequestIdOrder, BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(BatchOrderPolicy::FifoOrder, BatchOrderPolicy::FifoOrder);
    }

    #[test]
    fn batch_order_policy_inequality() {
        use crate::scheduler::types::BatchOrderPolicy;
        assert_ne!(BatchOrderPolicy::StrictRequestIdOrder, BatchOrderPolicy::FifoOrder);
    }

    #[test]
    fn batch_order_policy_copy_trait() {
        use crate::scheduler::types::BatchOrderPolicy;
        let a = BatchOrderPolicy::StrictRequestIdOrder;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn batch_order_policy_debug_format() {
        use crate::scheduler::types::BatchOrderPolicy;
        let debug = format!("{:?}", BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(debug, "StrictRequestIdOrder");
    }

    // =======================================================================
    // KvPipeline enum derive tests
    // =======================================================================

    #[test]
    fn kv_pipeline_equality() {
        use crate::scheduler::types::KvPipeline;
        assert_eq!(KvPipeline::Conversation, KvPipeline::Conversation);
        assert_eq!(KvPipeline::Working, KvPipeline::Working);
    }

    #[test]
    fn kv_pipeline_inequality() {
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
    fn kv_pipeline_hash_in_map() {
        use crate::scheduler::types::KvPipeline;
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(KvPipeline::Conversation, 1);
        map.insert(KvPipeline::Working, 2);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&KvPipeline::Conversation), Some(&1));
        assert_eq!(map.get(&KvPipeline::Working), Some(&2));
    }

    #[test]
    fn kv_pipeline_debug_format() {
        use crate::scheduler::types::KvPipeline;
        assert_eq!(format!("{:?}", KvPipeline::Conversation), "Conversation");
        assert_eq!(format!("{:?}", KvPipeline::Working), "Working");
    }

    // =======================================================================
    // PipelinedVirtualPageId construction and equality
    // =======================================================================

    #[test]
    fn pipelined_virtual_page_id_construction() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 7,
        };
        assert_eq!(pvp.sequence_id, 42);
        assert_eq!(pvp.logical_index, 7);
        assert_eq!(pvp.pipeline, KvPipeline::Conversation);
    }

    #[test]
    fn pipelined_virtual_page_id_equality() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 1,
            logical_index: 2,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 1,
            logical_index: 2,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_inequality_pipeline() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        };
        let b = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 1,
            logical_index: 0,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_copy_trait() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let a = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 10,
            logical_index: 5,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn pipelined_virtual_page_id_hash_in_set() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 1,
            logical_index: 0,
        });
        set.insert(PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 1,
            logical_index: 0,
        });
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn pipelined_virtual_page_id_debug_format() {
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 3,
        };
        let debug = format!("{pvp:?}");
        assert!(debug.contains("PipelinedVirtualPageId"));
    }

    // =======================================================================
    // PageLocation construction
    // =======================================================================

    #[test]
    fn page_location_construction() {
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let loc = PageLocation {
            physical_id: 42,
            tier: Tier::L1,
        };
        assert_eq!(loc.physical_id, 42);
        assert_eq!(loc.tier, Tier::L1);
    }

    #[test]
    fn page_location_equality() {
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let a = PageLocation { physical_id: 5, tier: Tier::L2 };
        let b = PageLocation { physical_id: 5, tier: Tier::L2 };
        assert_eq!(a, b);
    }

    #[test]
    fn page_location_inequality() {
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let a = PageLocation { physical_id: 5, tier: Tier::L1 };
        let b = PageLocation { physical_id: 5, tier: Tier::L2 };
        assert_ne!(a, b);
    }

    #[test]
    fn page_location_copy_trait() {
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let a = PageLocation { physical_id: 99, tier: Tier::L3 };
        let b = a;
        assert_eq!(a.physical_id, b.physical_id);
        assert_eq!(a.tier, b.tier);
    }

    #[test]
    fn page_location_debug_format() {
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let loc = PageLocation { physical_id: 7, tier: Tier::L1 };
        let debug = format!("{loc:?}");
        assert!(debug.contains("PageLocation"));
    }

    // =======================================================================
    // CompactScatterMeta construction and traits
    // =======================================================================

    #[test]
    fn compact_scatter_meta_construction() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let meta = CompactScatterMeta {
            original_slot: 3,
            compacted_slot: 1,
            active: 1,
        };
        assert_eq!(meta.original_slot, 3);
        assert_eq!(meta.compacted_slot, 1);
        assert_eq!(meta.active, 1);
    }

    #[test]
    fn compact_scatter_meta_equality() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let a = CompactScatterMeta { original_slot: 0, compacted_slot: 0, active: 1 };
        let b = CompactScatterMeta { original_slot: 0, compacted_slot: 0, active: 1 };
        assert_eq!(a, b);
    }

    #[test]
    fn compact_scatter_meta_inequality() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let a = CompactScatterMeta { original_slot: 0, compacted_slot: 0, active: 1 };
        let b = CompactScatterMeta { original_slot: 0, compacted_slot: 0, active: 0 };
        assert_ne!(a, b);
    }

    #[test]
    fn compact_scatter_meta_copy_trait() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let a = CompactScatterMeta { original_slot: 5, compacted_slot: 2, active: 1 };
        let b = a;
        assert_eq!(a.original_slot, b.original_slot);
        assert_eq!(a.compacted_slot, b.compacted_slot);
        assert_eq!(a.active, b.active);
    }

    #[test]
    fn compact_scatter_meta_debug_format() {
        use crate::scheduler::request_state::CompactScatterMeta;
        let meta = CompactScatterMeta { original_slot: 1, compacted_slot: 0, active: 1 };
        let debug = format!("{meta:?}");
        assert!(debug.contains("CompactScatterMeta"));
    }

    // =======================================================================
    // RequestTelemetry default and construction
    // =======================================================================

    #[test]
    fn request_telemetry_default_values() {
        use crate::scheduler::request_state::RequestTelemetry;
        let tel = RequestTelemetry::default();
        assert_eq!(tel.entropy, 0.0);
        assert_eq!(tel.centroid, 0.0);
        assert_eq!(tel.residual_delta, 1.0);
        assert_eq!(tel.residual_cosine, 1.0);
        assert_eq!(tel.range_group, 0);
    }

    #[test]
    fn request_telemetry_custom_values() {
        use crate::scheduler::request_state::RequestTelemetry;
        let tel = RequestTelemetry {
            entropy: 2.5,
            centroid: 0.7,
            residual_delta: 0.01,
            residual_cosine: 0.99,
            range_group: 3,
        };
        assert_eq!(tel.entropy, 2.5);
        assert_eq!(tel.centroid, 0.7);
        assert!(tel.residual_delta < 0.02);
        assert!(tel.residual_cosine > 0.98);
        assert_eq!(tel.range_group, 3);
    }

    #[test]
    fn request_telemetry_copy_trait() {
        use crate::scheduler::request_state::RequestTelemetry;
        let a = RequestTelemetry {
            entropy: 1.5,
            centroid: 0.5,
            residual_delta: 0.3,
            residual_cosine: 0.95,
            range_group: 7,
        };
        let b = a;
        assert_eq!(a.entropy, b.entropy);
        assert_eq!(a.range_group, b.range_group);
    }

    #[test]
    fn request_telemetry_equality() {
        use crate::scheduler::request_state::RequestTelemetry;
        let a = RequestTelemetry {
            entropy: 1.0,
            centroid: 0.0,
            residual_delta: 1.0,
            residual_cosine: 1.0,
            range_group: 0,
        };
        let b = RequestTelemetry::default();
        assert_ne!(a, b);
    }

    // =======================================================================
    // RequestPhase enum derive tests
    // =======================================================================

    #[test]
    fn request_phase_equality() {
        use crate::scheduler::request_state::RequestPhase;
        assert_eq!(RequestPhase::Prefill, RequestPhase::Prefill);
        assert_eq!(RequestPhase::Decode, RequestPhase::Decode);
        assert_eq!(RequestPhase::ChunkedPrefill, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn request_phase_inequality() {
        use crate::scheduler::request_state::RequestPhase;
        assert_ne!(RequestPhase::Prefill, RequestPhase::Decode);
        assert_ne!(RequestPhase::Decode, RequestPhase::ChunkedPrefill);
        assert_ne!(RequestPhase::Prefill, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn request_phase_copy_trait() {
        use crate::scheduler::request_state::RequestPhase;
        let a = RequestPhase::Decode;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn request_phase_hash_in_set() {
        use crate::scheduler::request_state::RequestPhase;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(RequestPhase::Prefill);
        set.insert(RequestPhase::Decode);
        set.insert(RequestPhase::ChunkedPrefill);
        assert_eq!(set.len(), 3);
        assert!(set.contains(&RequestPhase::Decode));
    }

    #[test]
    fn request_phase_debug_format() {
        use crate::scheduler::request_state::RequestPhase;
        assert_eq!(format!("{:?}", RequestPhase::Prefill), "Prefill");
        assert_eq!(format!("{:?}", RequestPhase::Decode), "Decode");
        assert_eq!(format!("{:?}", RequestPhase::ChunkedPrefill), "ChunkedPrefill");
    }

    // =======================================================================
    // PagePayloadKind enum derive tests
    // =======================================================================

    #[test]
    fn page_payload_kind_equality() {
        use crate::scheduler::types::PagePayloadKind;
        assert_eq!(PagePayloadKind::KvContext, PagePayloadKind::KvContext);
        assert_eq!(PagePayloadKind::ExpertWeight, PagePayloadKind::ExpertWeight);
    }

    #[test]
    fn page_payload_kind_inequality() {
        use crate::scheduler::types::PagePayloadKind;
        assert_ne!(PagePayloadKind::KvContext, PagePayloadKind::ExpertWeight);
    }

    #[test]
    fn page_payload_kind_copy_trait() {
        use crate::scheduler::types::PagePayloadKind;
        let a = PagePayloadKind::KvContext;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn page_payload_kind_debug_format() {
        use crate::scheduler::types::PagePayloadKind;
        let debug = format!("{:?}", PagePayloadKind::KvContext);
        assert_eq!(debug, "KvContext");
    }

    // =======================================================================
    // MemoryResidency enum derive tests
    // =======================================================================

    #[test]
    fn memory_residency_equality() {
        use crate::scheduler::types::MemoryResidency;
        assert_eq!(MemoryResidency::DeviceLocal, MemoryResidency::DeviceLocal);
        assert_eq!(MemoryResidency::HostLocal, MemoryResidency::HostLocal);
        assert_eq!(MemoryResidency::DiskSwap, MemoryResidency::DiskSwap);
    }

    #[test]
    fn memory_residency_inequality() {
        use crate::scheduler::types::MemoryResidency;
        assert_ne!(MemoryResidency::DeviceLocal, MemoryResidency::HostLocal);
        assert_ne!(MemoryResidency::HostLocal, MemoryResidency::DiskSwap);
    }

    #[test]
    fn memory_residency_copy_trait() {
        use crate::scheduler::types::MemoryResidency;
        let a = MemoryResidency::DeviceLocal;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn memory_residency_hash_in_set() {
        use crate::scheduler::types::MemoryResidency;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(MemoryResidency::DeviceLocal);
        set.insert(MemoryResidency::DiskSwap);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn memory_residency_debug_format() {
        use crate::scheduler::types::MemoryResidency;
        assert_eq!(format!("{:?}", MemoryResidency::DeviceLocal), "DeviceLocal");
        assert_eq!(format!("{:?}", MemoryResidency::DiskSwap), "DiskSwap");
    }

    // =======================================================================
    // WeightTier enum derive tests
    // =======================================================================

    #[test]
    fn weight_tier_equality() {
        use crate::scheduler::types::WeightTier;
        assert_eq!(WeightTier::Hot, WeightTier::Hot);
        assert_eq!(WeightTier::Warm, WeightTier::Warm);
        assert_eq!(WeightTier::Cold, WeightTier::Cold);
    }

    #[test]
    fn weight_tier_inequality() {
        use crate::scheduler::types::WeightTier;
        assert_ne!(WeightTier::Hot, WeightTier::Warm);
        assert_ne!(WeightTier::Warm, WeightTier::Cold);
        assert_ne!(WeightTier::Hot, WeightTier::Cold);
    }

    #[test]
    fn weight_tier_copy_trait() {
        use crate::scheduler::types::WeightTier;
        let a = WeightTier::Hot;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn weight_tier_hash_in_set() {
        use crate::scheduler::types::WeightTier;
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WeightTier::Hot);
        set.insert(WeightTier::Cold);
        assert_eq!(set.len(), 2);
        assert!(set.contains(&WeightTier::Hot));
        assert!(!set.contains(&WeightTier::Warm));
    }

    #[test]
    fn weight_tier_debug_format() {
        use crate::scheduler::types::WeightTier;
        assert_eq!(format!("{:?}", WeightTier::Hot), "Hot");
        assert_eq!(format!("{:?}", WeightTier::Warm), "Warm");
        assert_eq!(format!("{:?}", WeightTier::Cold), "Cold");
    }

    // =======================================================================
    // SessionKvCache construction
    // =======================================================================

    #[test]
    fn session_kv_cache_construction() {
        use crate::scheduler::memory_manager::{SessionId, SessionKvCache, VirtualPageId};
        let session = SessionKvCache {
            session_id: 42 as SessionId,
            pages: vec![
                VirtualPageId::new(1, 0),
                VirtualPageId::new(1, 1),
            ],
            finalized_position: 128,
        };
        assert_eq!(session.session_id, 42);
        assert_eq!(session.pages.len(), 2);
        assert_eq!(session.finalized_position, 128);
    }

    #[test]
    fn session_kv_cache_empty() {
        use crate::scheduler::memory_manager::SessionKvCache;
        let session = SessionKvCache {
            session_id: 0,
            pages: vec![],
            finalized_position: 0,
        };
        assert!(session.pages.is_empty());
        assert_eq!(session.finalized_position, 0);
    }

    #[test]
    fn session_kv_cache_debug_format() {
        use crate::scheduler::memory_manager::SessionKvCache;
        let session = SessionKvCache {
            session_id: 1,
            pages: vec![],
            finalized_position: 0,
        };
        let debug = format!("{session:?}");
        assert!(debug.contains("SessionKvCache"));
    }

    // =======================================================================
    // Additional Shannon entropy property tests
    // =======================================================================

    #[test]
    fn shannon_entropy_very_peaked_single_dominant() {
        // One entry vastly larger than all others combined.
        let logits = [10000.0, -10000.0, -10000.0, -10000.0];
        let h = shannon_entropy(&logits);
        assert!(h < 0.01, "extremely peaked should have near-zero entropy, got {h}");
    }

    #[test]
    fn shannon_entropy_uniform_invariant_to_shift() {
        // Shifting all logits by a constant should not change entropy.
        let base = [1.0, 2.0, 3.0, 4.0];
        let shifted = [101.0, 102.0, 103.0, 104.0];
        let h_base = shannon_entropy(&base);
        let h_shifted = shannon_entropy(&shifted);
        assert!(
            (h_base - h_shifted).abs() < 1e-3,
            "entropy should be shift-invariant: base={h_base}, shifted={h_shifted}"
        );
    }

    #[test]
    fn shannon_entropy_uniform_input_maximal() {
        // Uniform distribution should have maximal entropy for given length.
        let vals = [1.0, 1.0, 1.0, 1.0];
        let h = shannon_entropy(&vals);
        let expected = (vals.len() as f32).ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "uniform entropy should be ln(4)={expected}, got {h}"
        );
    }

    // =======================================================================
    // Additional argmax_token property tests
    // =======================================================================

    #[test]
    fn argmax_token_returns_valid_index() {
        let logits = [0.5, -0.3, 1.7, 0.1];
        let result = argmax_token(&logits);
        assert!(result < logits.len() as u32, "result must be a valid index");
        // The returned index should point to a value >= all others.
        for (i, &v) in logits.iter().enumerate() {
            assert!(
                logits[result as usize] >= v - 1e-6,
                "result value should be >= all others"
            );
            let _ = i;
        }
    }

    // =======================================================================
    // Additional extract_top_k property tests
    // =======================================================================

    #[test]
    fn extract_top_k_returns_unique_indices() {
        let logits = [5.0, 3.0, 1.0, 4.0, 2.0];
        let top3 = extract_top_k_token_ids(&logits, 3);
        assert_eq!(top3.len(), 3);
        // All indices should be unique.
        let mut seen = std::collections::HashSet::new();
        for &id in &top3 {
            assert!(seen.insert(id), "duplicate index {id} in top-k result");
        }
    }

    #[test]
    fn extract_top_k_order_descending() {
        let logits = [10.0, 30.0, 20.0, 50.0, 40.0];
        let top3 = extract_top_k_token_ids(&logits, 3);
        // Values at returned indices should be in descending order.
        assert_eq!(top3.len(), 3);
        let v0 = logits[top3[0] as usize];
        let v1 = logits[top3[1] as usize];
        let v2 = logits[top3[2] as usize];
        assert!(v0 >= v1, "first pick should have >= value than second");
        assert!(v1 >= v2, "second pick should have >= value than third");
    }

    // =======================================================================
    // BatchPrepData additional edge cases
    // =======================================================================

    #[test]
    fn batch_prep_data_new_large_count() {
        let prep = crate::scheduler::batcher::BatchPrepData::new(1024);
        assert_eq!(prep.prompt_lens.len(), 1024);
        assert_eq!(prep.active_flags.len(), 1024);
        assert_eq!(prep.sampling_params_packed.len(), 1024 * 4);
        assert!(prep.active_flags.iter().all(|&f| f == 1));
    }

    #[test]
    fn batch_prep_data_default_fields_zero() {
        let prep = crate::scheduler::batcher::BatchPrepData::new(4);
        assert_eq!(prep.max_decode_steps, 0);
        assert_eq!(prep.total_prefill_tokens, 0);
        assert!(prep.prompt_lens.iter().all(|&v| v == 0));
        assert!(prep.kv_lens.iter().all(|&v| v == 0));
        assert!(prep.session_positions.iter().all(|&v| v == 0));
        assert!(prep.rope_pos_offsets.iter().all(|&v| v == 0));
        assert!(prep.max_new_tokens.iter().all(|&v| v == 0));
        assert!(prep.gen_counts.iter().all(|&v| v == 0));
        assert!(prep.last_sampled_tokens.iter().all(|&v| v == 0));
    }

    // =======================================================================
    // SequenceTelemetry serialization roundtrip
    // =======================================================================

    #[test]
    fn sequence_telemetry_serde_roundtrip() {
        let tel = SequenceTelemetry {
            l2_delta: 0.42,
            has_outlier: true,
            dead_density: 0.15,
            per_head_entropy: 3.7,
            transform_ratio: 0.88,
            output_entropy: 2.1,
        };
        let json = serde_json::to_string(&tel).expect("serialize");
        let back: SequenceTelemetry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(tel, back);
    }

    #[test]
    fn sequence_telemetry_serde_default_roundtrip() {
        let tel = SequenceTelemetry::default();
        let json = serde_json::to_string(&tel).expect("serialize");
        let back: SequenceTelemetry = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(tel, back);
    }

    // =======================================================================
    // New tests: ExecutorError additional Display variants
    // =======================================================================

    #[test]
    fn executor_error_scheduler_long_message_display() {
        // Arrange: construct a Scheduler error with a long descriptive message.
        let long_msg = "request queue overflow: max_batch_size=512 exceeded with 1024 pending requests";
        let err = ExecutorError::Scheduler(long_msg.into());
        // Act: format via Display trait.
        let msg = format!("{err}");
        // Assert: the full message is preserved in the output.
        assert!(
            msg.contains("max_batch_size=512"),
            "Scheduler Display should preserve the message, got: {msg}"
        );
        assert!(
            msg.starts_with("scheduler error:"),
            "Scheduler Display should start with prefix, got: {msg}"
        );
    }

    #[test]
    fn executor_error_compilation_long_message_display() {
        // Arrange: construct a Compilation error with detailed failure info.
        let detail = "failed to lower OpKind::FusedRmsNorm: register pressure exceeded L1 budget";
        let err = ExecutorError::Compilation(detail.into());
        // Act: format via Display.
        let msg = format!("{err}");
        // Assert: the detail string appears in the formatted output.
        assert!(
            msg.contains("register pressure"),
            "Compilation Display should preserve detail, got: {msg}"
        );
        assert!(
            msg.starts_with("JIT compilation failed:"),
            "Compilation Display should start with prefix, got: {msg}"
        );
    }

    #[test]
    fn executor_error_graph_expansion_with_path_display() {
        // Arrange: construct a GraphExpansion error referencing a specific graph node.
        let detail = "node 'attention_layer_23' has incompatible shapes: [1,32,128] vs [1,16,128]";
        let err = ExecutorError::GraphExpansion(detail.into());
        // Act: format via Display.
        let msg = format!("{err}");
        // Assert: the graph node name is in the output.
        assert!(
            msg.contains("attention_layer_23"),
            "GraphExpansion Display should preserve node name, got: {msg}"
        );
    }

    // =======================================================================
    // New tests: LogitsHandle with boundary float values
    // =======================================================================

    #[test]
    fn logits_handle_with_subnormal_floats() {
        // Arrange: create logits with subnormal (denormalized) f32 values.
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        let handle = LogitsHandle {
            data: vec![subnormal, -subnormal, 0.0],
        };
        // Act: verify the subnormal values are preserved.
        assert!(handle.data[0] > 0.0, "subnormal should be positive");
        assert!(handle.data[1] < 0.0, "negated subnormal should be negative");
        // Assert: subnormal arithmetic does not panic in extract_top_k.
        let top = extract_top_k_token_ids(&handle.data, 3);
        assert_eq!(top.len(), 3, "all three entries returned");
    }

    // =======================================================================
    // New tests: extract_top_k_token_ids with alternating high/low pattern
    // =======================================================================

    #[test]
    fn extract_top_k_alternating_high_low_pattern() {
        // Arrange: even indices high, odd indices low.
        let logits = [10.0, 0.0, 9.0, 0.0, 8.0, 0.0, 7.0, 0.0];
        // Act: extract top 4.
        let top4 = extract_top_k_token_ids(&logits, 4);
        // Assert: all 4 results should be even indices (the high values).
        assert_eq!(top4.len(), 4);
        for &id in &top4 {
            assert_eq!(id % 2, 0, "expected even index, got {id}");
        }
    }

    // =======================================================================
    // New tests: argmax_token with all f32::MIN values
    // =======================================================================

    #[test]
    fn argmax_token_all_min_float() {
        // Arrange: all entries are f32 minimum (most negative finite value).
        let logits = [f32::MIN; 5];
        // Act: argmax on all-equal minimal values.
        let result = argmax_token(&logits);
        // Assert: last index wins for tied values.
        assert_eq!(
            result, 4,
            "all f32::MIN: last index should win, got {result}"
        );
    }

    // =======================================================================
    // New tests: shannon_entropy with single very negative value among positives
    // =======================================================================

    #[test]
    fn shannon_entropy_single_dominant_negative() {
        // Arrange: one entry so negative it has negligible probability.
        let logits = [-100000.0, 0.0, 0.0, 0.0];
        // Act: compute entropy.
        let h = shannon_entropy(&logits);
        // Assert: nearly uniform over the three non-negative entries.
        let expected = 3.0f32.ln();
        assert!(
            (h - expected).abs() < 0.1,
            "3 equal + 1 negligible should give ~ln(3)={expected}, got {h}"
        );
    }

    // =======================================================================
    // New tests: BatchInput with varied draft_steps values
    // =======================================================================

    #[test]
    fn batch_input_varied_draft_steps() {
        // Arrange: sequences with draft_steps = 0, 1, and max.
        let seq1 = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        };
        let seq2 = SequenceInput {
            tokens: vec![2],
            position: 5,
            draft_steps: 3,
            page_table: Some(vec![0, 1]),
            fused_hidden: None,
        };
        let seq3 = SequenceInput {
            tokens: vec![3],
            position: 10,
            draft_steps: usize::MAX,
            page_table: None,
            fused_hidden: None,
        };
        let batch = BatchInput {
            sequences: vec![seq1, seq2, seq3],
        };
        // Act & Assert: verify draft_steps are preserved per sequence.
        assert_eq!(batch.sequences[0].draft_steps, 0);
        assert_eq!(batch.sequences[1].draft_steps, 3);
        assert_eq!(batch.sequences[2].draft_steps, usize::MAX);
    }

    // =======================================================================
    // New tests: BuildSequencesOutput with mixed results and sequences
    // =======================================================================

    #[test]
    fn build_sequences_output_mixed_sequences_and_failures() {
        // Arrange: 2 valid sequences and 3 failure results.
        let sequences: Vec<SequenceInput> = (0..2)
            .map(|i| SequenceInput {
                tokens: vec![i],
                position: i as usize * 10,
                draft_steps: 0,
                page_table: None,
                fused_hidden: None,
            })
            .collect();
        let request_indices: Vec<RequestId> = vec![100, 101];
        let batch_results: Vec<BatchResult> = (200..205)
            .map(BatchResult::fail)
            .collect();
        // Act: construct BuildSequencesOutput.
        let output = BuildSequencesOutput {
            sequences,
            request_indices,
            batch_results,
        };
        // Assert: both lists are independent and correct.
        assert_eq!(output.sequences.len(), 2);
        assert_eq!(output.request_indices.len(), 2);
        assert_eq!(output.batch_results.len(), 5);
        assert_eq!(output.batch_results[2].request_id, 202);
    }

    // =======================================================================
    // New tests: ScheduledBatch with draft_steps mismatch handling
    // =======================================================================

    #[test]
    fn scheduled_batch_requests_without_offsets() {
        // Arrange: requests with empty offsets/draft_steps (valid edge case
        // where only request IDs matter for routing).
        let batch = ScheduledBatch {
            requests: vec![10, 20, 30],
            seq_offsets: vec![],
            draft_steps: vec![],
        };
        // Act & Assert: requests vector is populated but offsets empty.
        assert_eq!(batch.requests.len(), 3);
        assert!(batch.seq_offsets.is_empty());
        assert!(batch.draft_steps.is_empty());
    }

    // =======================================================================
    // New tests: BatchResult telemetry field access
    // =======================================================================

    #[test]
    fn batch_result_telemetry_preserved_in_continue() {
        // Arrange: construct a SequenceTelemetry with non-default values.
        let telemetry = SequenceTelemetry {
            l2_delta: 1.23,
            has_outlier: true,
            dead_density: 0.5,
            per_head_entropy: 4.0,
            transform_ratio: 0.77,
            output_entropy: 3.14,
        };
        // Act: create a continue BatchResult.
        let result = BatchResult::continue_with_token(42, 7, telemetry);
        // Assert: telemetry data is preserved.
        assert_eq!(result.action, BatchAction::Continue);
        let tel = result.telemetry;
        assert_eq!(tel.l2_delta, 1.23);
        assert!(tel.has_outlier);
        assert_eq!(tel.output_entropy, 3.14);
    }

    // =======================================================================
    // New tests: KvCacheHandle with MAX value
    // =======================================================================

    #[test]
    fn kv_cache_handle_max_value_equality() {
        // Arrange: two handles with u64::MAX.
        let a = KvCacheHandle(u64::MAX);
        let b = KvCacheHandle(u64::MAX);
        // Act & Assert: they compare equal.
        assert_eq!(a, b);
        let c = KvCacheHandle(0);
        assert_ne!(a, c);
    }

    // =======================================================================
    // New tests: TierUsage available at exact boundary
    // =======================================================================

    #[test]
    fn tier_usage_available_one_remaining() {
        // Arrange: capacity 1000, used 999.
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 999,
            capacity: 1000,
        };
        // Act: compute available.
        let avail = usage.available();
        // Assert: exactly 1 remaining.
        assert_eq!(avail, 1);
    }

    // =======================================================================
    // New tests: extract_top_k_token_ids with repeated maximum value
    // =======================================================================

    #[test]
    fn extract_top_k_two_maxima_identifies_both() {
        // Arrange: two entries share the maximum, one is lower.
        let logits = [5.0, 10.0, 10.0, 3.0];
        // Act: request top 2.
        let top2 = extract_top_k_token_ids(&logits, 2);
        // Assert: both returned indices should point to value 10.0.
        assert_eq!(top2.len(), 2);
        for &id in &top2 {
            assert_eq!(logits[id as usize], 10.0, "expected max value at index {id}");
        }
    }

    // =======================================================================
    // New tests: shannon_entropy invariant under negation
    // =======================================================================

    #[test]
    fn shannon_entropy_invariant_under_negation() {
        // Arrange: two distributions that are negations of each other.
        let positive = [1.0, 2.0, 3.0];
        let negated = [-1.0, -2.0, -3.0];
        // Act: compute entropy for both.
        let h_pos = shannon_entropy(&positive);
        let h_neg = shannon_entropy(&negated);
        // Assert: negation is equivalent to a constant shift, so entropy is the same.
        assert!(
            (h_pos - h_neg).abs() < 1e-4,
            "entropy should be invariant under negation: pos={h_pos}, neg={h_neg}"
        );
    }

    // =======================================================================
    // New tests: LogitsHandle cloning preserves length and values
    // =======================================================================

    #[test]
    fn logits_handle_clone_independent_mutation() {
        // Arrange: create a LogitsHandle and clone it.
        let original = LogitsHandle {
            data: vec![1.0, 2.0, 3.0],
        };
        let mut cloned = original.clone();
        // Act: mutate the clone.
        cloned.data[0] = 99.0;
        // Assert: original is unaffected.
        assert_eq!(original.data[0], 1.0, "original should not be mutated");
        assert_eq!(cloned.data[0], 99.0, "clone should reflect the mutation");
    }

    // =======================================================================
    // New tests: BatchAction Debug format and exhaustive variants
    // =======================================================================

    // @trace TEST-EXEC-STEP-001 [level:unit]
    #[test]
    fn batch_action_debug_format_all_variants() {
        // Arrange: exercise Debug format for all four BatchAction variants.
        // Act & Assert: each debug string matches the variant name.
        assert_eq!(format!("{:?}", BatchAction::Continue), "Continue");
        assert_eq!(format!("{:?}", BatchAction::Complete), "Complete");
        assert_eq!(format!("{:?}", BatchAction::Pause), "Pause");
        assert_eq!(format!("{:?}", BatchAction::Fail), "Fail");
    }

    // =======================================================================
    // New tests: BatchResult::pause constructor
    // =======================================================================

    // @trace TEST-EXEC-STEP-002 [level:unit]
    #[test]
    fn batch_result_pause_constructor() {
        // Arrange: request ID 777 should be paused.
        // Act: construct via BatchResult::pause.
        let result = BatchResult::pause(777);
        // Assert: action is Pause, request_id matches, no generated token.
        assert_eq!(result.action, BatchAction::Pause);
        assert_eq!(result.request_id, 777);
        assert!(result.generated_token.is_none());
    }

    // =======================================================================
    // New tests: BatchResult::fail independent instances
    // =======================================================================

    // @trace TEST-EXEC-STEP-003 [level:unit]
    #[test]
    fn batch_result_fail_independent_ids() {
        // Arrange: two fail results with distinct request IDs.
        // Act: construct both.
        let a = BatchResult::fail(10);
        let b = BatchResult::fail(20);
        // Assert: each preserves its own request_id and Fail action.
        assert_eq!(a.action, BatchAction::Fail);
        assert_eq!(b.action, BatchAction::Fail);
        assert_eq!(a.request_id, 10);
        assert_eq!(b.request_id, 20);
        assert!(a.generated_token.is_none());
        assert!(b.generated_token.is_none());
    }

    // =======================================================================
    // New tests: BatchResult::complete with generated_token None
    // =======================================================================

    // @trace TEST-EXEC-STEP-004 [level:unit]
    #[test]
    fn batch_result_complete_no_generated_token() {
        // Arrange: complete result where no token was generated (edge case).
        // Act: construct with generated_token = None.
        let result = BatchResult::complete(55, None, SequenceTelemetry::default());
        // Assert: action is Complete, generated_token is None.
        assert_eq!(result.action, BatchAction::Complete);
        assert_eq!(result.request_id, 55);
        assert!(result.generated_token.is_none());
    }

    // =======================================================================
    // New tests: extract_top_k_token_ids with k equals length
    // =======================================================================

    // @trace TEST-EXEC-STEP-005 [level:unit]
    #[test]
    fn extract_top_k_k_equals_input_length() {
        // Arrange: logits has exactly 4 elements, request k=4.
        let logits = [3.0, 1.0, 4.0, 2.0];
        // Act: request all elements sorted descending.
        let top = extract_top_k_token_ids(&logits, 4);
        // Assert: all 4 indices returned in descending value order.
        assert_eq!(top.len(), 4);
        assert_eq!(top[0], 2, "highest value 4.0 at index 2");
        assert_eq!(top[1], 0, "second value 3.0 at index 0");
        assert_eq!(top[2], 3, "third value 2.0 at index 3");
        assert_eq!(top[3], 1, "lowest value 1.0 at index 1");
    }

    // =======================================================================
    // New tests: argmax_token with alternating positive infinity
    // =======================================================================

    // @trace TEST-EXEC-STEP-006 [level:unit]
    #[test]
    fn argmax_token_with_mixed_infinity() {
        // Arrange: one positive infinity, one negative infinity, one finite.
        let logits = [f32::NEG_INFINITY, f32::INFINITY, 0.0];
        // Act: argmax should find the positive infinity.
        let result = argmax_token(&logits);
        // Assert: index 1 has positive infinity.
        assert_eq!(result, 1, "positive infinity should win at index 1");
    }

    // =======================================================================
    // New tests: shannon_entropy with two extremely peaked values
    // =======================================================================

    // @trace TEST-EXEC-STEP-007 [level:unit]
    #[test]
    fn shannon_entropy_two_peaked_near_zero() {
        // Arrange: one extremely large value among small ones.
        let logits = [1e10, 0.0, 0.0, 0.0, 0.0];
        // Act: compute entropy.
        let h = shannon_entropy(&logits);
        // Assert: nearly zero entropy since one entry dominates.
        assert!(
            h < 1e-3,
            "extremely peaked distribution should have near-zero entropy, got {h}"
        );
    }

    // =======================================================================
    // New tests: LogitsHandle with empty data
    // =======================================================================

    // @trace TEST-EXEC-STEP-008 [level:unit]
    #[test]
    fn logits_handle_empty_data_and_topk() {
        // Arrange: LogitsHandle with zero-length data vector.
        let handle = LogitsHandle { data: vec![] };
        // Act & Assert: data is empty, extract_top_k returns empty.
        assert!(handle.data.is_empty());
        let top = extract_top_k_token_ids(&handle.data, 5);
        assert!(top.is_empty(), "empty logits should produce empty top-k");
    }

    // =======================================================================
    // New tests: BuildSequencesOutput with large batch_results
    // =======================================================================

    // @trace TEST-EXEC-STEP-009 [level:unit]
    #[test]
    fn build_sequences_output_large_batch_results() {
        // Arrange: many fail results and one valid sequence.
        let sequences: Vec<SequenceInput> = vec![SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: None,
            fused_hidden: None,
        }];
        let batch_results: Vec<BatchResult> = (1000..2000)
            .map(BatchResult::fail)
            .collect();
        // Act: construct BuildSequencesOutput.
        let output = BuildSequencesOutput {
            sequences,
            request_indices: vec![42],
            batch_results,
        };
        // Assert: 1 sequence, 1 request index, 1000 fail results.
        assert_eq!(output.sequences.len(), 1);
        assert_eq!(output.request_indices.len(), 1);
        assert_eq!(output.batch_results.len(), 1000);
        // Verify first and last fail results have correct request IDs.
        assert_eq!(output.batch_results[0].request_id, 1000);
        assert_eq!(output.batch_results[999].request_id, 1999);
    }

    // =======================================================================
    // New tests: ScheduledBatch with mismatched lengths is constructible
    // =======================================================================

    // @trace TEST-EXEC-STEP-010 [level:unit]
    #[test]
    fn scheduled_batch_mismatched_lengths() {
        // Arrange: more requests than seq_offsets and draft_steps.
        // Act: construct with mismatched vector lengths (valid struct, unusual state).
        let batch = ScheduledBatch {
            requests: vec![1, 2, 3, 4, 5],
            seq_offsets: vec![0],
            draft_steps: vec![0],
        };
        // Assert: struct is usable despite mismatched lengths.
        assert_eq!(batch.requests.len(), 5);
        assert_eq!(batch.seq_offsets.len(), 1);
        assert_eq!(batch.draft_steps.len(), 1);
    }

    // =======================================================================
    // New tests: KvCacheHandle equality and ordering with sequential values
    // =======================================================================

    // @trace TEST-EXEC-STEP-011 [level:unit]
    #[test]
    fn kv_cache_handle_sequential_inequality() {
        // Arrange: three handles with consecutive values.
        let a = KvCacheHandle(0);
        let b = KvCacheHandle(1);
        let c = KvCacheHandle(2);
        // Act & Assert: all pairwise unequal.
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        // Verify self-equality.
        assert_eq!(a, a);
        assert_eq!(b, b);
    }

    // =======================================================================
    // New tests: TierUsage with u64::MAX capacity
    // =======================================================================

    // @trace TEST-EXEC-STEP-012 [level:unit]
    #[test]
    fn tier_usage_available_max_capacity() {
        // Arrange: capacity at usize::MAX, used 0.
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 0,
            capacity: usize::MAX,
        };
        // Act: compute available.
        let avail = usage.available();
        // Assert: available equals capacity.
        assert_eq!(avail, usize::MAX);
    }

    // =======================================================================
    // New tests: SequenceTelemetry custom values roundtrip through equality
    // =======================================================================

    // @trace TEST-EXEC-STEP-013 [level:unit]
    #[test]
    fn sequence_telemetry_equality_custom_values() {
        // Arrange: two identical custom telemetry instances.
        let tel = SequenceTelemetry {
            l2_delta: 99.99,
            has_outlier: false,
            dead_density: 0.33,
            per_head_entropy: 5.5,
            transform_ratio: 0.11,
            output_entropy: 7.77,
        };
        let tel2 = SequenceTelemetry {
            l2_delta: 99.99,
            has_outlier: false,
            dead_density: 0.33,
            per_head_entropy: 5.5,
            transform_ratio: 0.11,
            output_entropy: 7.77,
        };
        // Act & Assert: they compare equal.
        assert_eq!(tel, tel2);
    }

    // =======================================================================
    // New tests: SequenceTelemetry inequality when fields differ
    // =======================================================================

    // @trace TEST-EXEC-STEP-014 [level:unit]
    #[test]
    fn sequence_telemetry_inequality_single_field_differs() {
        // Arrange: two telemetry instances differing only in has_outlier.
        let a = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: true,
            dead_density: 0.5,
            per_head_entropy: 2.0,
            transform_ratio: 0.5,
            output_entropy: 3.0,
        };
        let b = SequenceTelemetry {
            l2_delta: 1.0,
            has_outlier: false,
            dead_density: 0.5,
            per_head_entropy: 2.0,
            transform_ratio: 0.5,
            output_entropy: 3.0,
        };
        // Act & Assert: they are not equal.
        assert_ne!(a, b, "telemetry should differ when has_outlier differs");
    }

    // =======================================================================
    // New tests: extract_top_k_token_ids with all identical negative values
    // =======================================================================

    // @trace TEST-EXEC-STEP-015 [level:unit]
    #[test]
    fn extract_top_k_all_identical_negatives() {
        // Arrange: all values are the same negative number.
        let logits = [-7.5, -7.5, -7.5, -7.5];
        // Act: request top 3.
        let top3 = extract_top_k_token_ids(&logits, 3);
        // Assert: 3 results, all valid indices, all values match.
        assert_eq!(top3.len(), 3);
        for &id in &top3 {
            assert!((id as usize) < logits.len(), "index {id} out of bounds");
            assert_eq!(logits[id as usize], -7.5);
        }
    }

    // =======================================================================
    // Wave-12x34: additional unit tests for executor_step.rs
    // =======================================================================

    // @trace TEST-EXEC-STEP-016 [level:unit]
    #[test]
    fn effective_kv_max_seq_len_passthrough_identity() {
        // Arrange: various geometry max_seq_len values.
        let cases = [0, 1, 512, 8192, 131072, usize::MAX];
        for &val in &cases {
            // Act: call the passthrough function.
            let result = crate::engine::executor_types::effective_kv_max_seq_len(val);
            // Assert: returns the input unchanged (identity function).
            assert_eq!(result, val, "passthrough failed for {val}");
        }
    }

    // @trace TEST-EXEC-STEP-017 [level:unit]
    #[test]
    fn adaptive_chunk_policy_compute_low_l1_returns_min() {
        // Arrange: policy with min=64, max=2048; L1 ratio below 0.25 threshold.
        let policy = AdaptiveChunkPolicy {
            min_chunk: 64,
            max_chunk: 2048,
        };
        // Act: compute chunk size with very low L1 ratio.
        let chunk = policy.compute(0.1, 1, 10000);
        // Assert: should clamp to min_chunk when L1 is scarce.
        assert_eq!(chunk, 64, "low L1 should produce min_chunk, got {chunk}");
    }

    // @trace TEST-EXEC-STEP-018 [level:unit]
    #[test]
    fn adaptive_chunk_policy_compute_high_l1_returns_max_capped_by_budget() {
        // Arrange: policy with min=64, max=2048; L1 ratio above 0.75.
        let policy = AdaptiveChunkPolicy {
            min_chunk: 64,
            max_chunk: 2048,
        };
        // Act: compute chunk with high L1 ratio but small remaining budget.
        let chunk = policy.compute(0.9, 1, 100);
        // Assert: should cap to remaining_budget since it's less than max_chunk.
        assert_eq!(chunk, 100, "should be capped by remaining_budget, got {chunk}");
    }

    // @trace TEST-EXEC-STEP-019 [level:unit]
    #[test]
    fn adaptive_chunk_policy_compute_zero_remaining_budget_returns_one() {
        // Arrange: policy with min=64, max=2048; remaining_budget = 0.
        let policy = AdaptiveChunkPolicy {
            min_chunk: 64,
            max_chunk: 2048,
        };
        // Act: compute with zero remaining budget.
        // The formula: .clamp(min, max).min(remaining_budget.max(1))
        // With remaining_budget=0, .min(1) caps the result to 1.
        let chunk = policy.compute(0.5, 1, 0);
        // Assert: min(0.max(1)) = min(1), so result is 1.
        assert_eq!(chunk, 1, "zero budget produces .min(1), got {chunk}");
    }

    // @trace TEST-EXEC-STEP-020 [level:unit]
    #[test]
    fn adaptive_chunk_policy_compute_concurrent_penalty_reduces_chunk() {
        // Arrange: policy with min=10, max=100; moderate L1 ratio.
        let policy = AdaptiveChunkPolicy {
            min_chunk: 10,
            max_chunk: 100,
        };
        // Act: compute with 1 concurrent req vs 10 concurrent reqs.
        let chunk_single = policy.compute(0.5, 1, 10000);
        let chunk_many = policy.compute(0.5, 10, 10000);
        // Assert: more concurrent requests should reduce (or equal) the chunk size.
        assert!(
            chunk_many <= chunk_single,
            "concurrency penalty should reduce chunk: single={chunk_single}, many={chunk_many}"
        );
    }

    // @trace TEST-EXEC-STEP-021 [level:unit]
    #[test]
    fn shannon_entropy_invariant_under_additive_shift() {
        // Arrange: two distributions related by a constant additive shift.
        // Softmax is shift-invariant: softmax(x + c) = softmax(x).
        let base = [1.0, 2.0, 3.0, 4.0];
        let shifted = [101.0, 102.0, 103.0, 104.0];
        // Act: compute entropy for both.
        let h_base = shannon_entropy(&base);
        let h_shifted = shannon_entropy(&shifted);
        // Assert: additive shift preserves the softmax distribution, hence entropy.
        assert!(
            (h_base - h_shifted).abs() < 1e-4,
            "additive shift should not change entropy: base={h_base}, shifted={h_shifted}"
        );
    }

    // @trace TEST-EXEC-STEP-022 [level:unit]
    #[test]
    fn extract_top_k_preserves_descending_value_order() {
        // Arrange: logits with clear value separation.
        let logits = [0.0, 100.0, 25.0, 75.0, 50.0];
        // Act: extract top 5 (all elements).
        let top5 = extract_top_k_token_ids(&logits, 5);
        // Assert: each successive index points to a non-increasing value.
        assert_eq!(top5.len(), 5);
        for window in top5.windows(2) {
            let v_prev = logits[window[0] as usize];
            let v_next = logits[window[1] as usize];
            assert!(
                v_prev >= v_next,
                "descending order violated: {v_prev} at idx {} < {v_next} at idx {}",
                window[0], window[1]
            );
        }
    }

    // @trace TEST-EXEC-STEP-023 [level:unit]
    #[test]
    fn argmax_token_large_negative_beats_small_negative() {
        // Arrange: mix of negative values where -0.001 is the maximum.
        let logits = [-1000.0, -500.0, -0.001, -999.0];
        // Act: argmax should find the least negative value.
        let result = argmax_token(&logits);
        // Assert: index 2 has value -0.001 which is the largest.
        assert_eq!(result, 2, "least negative should win, got {result}");
    }

    // @trace TEST-EXEC-STEP-024 [level:unit]
    #[test]
    fn executor_error_request_not_found_preserves_id() {
        // Arrange: construct RequestNotFound with a specific ID.
        let rid: RequestId = 98765;
        let err = ExecutorError::RequestNotFound { request_id: rid };
        // Act: format via Display.
        let msg = format!("{err}");
        // Assert: the request_id appears in the formatted message.
        assert!(
            msg.contains("98765"),
            "RequestNotFound Display should contain the ID, got: {msg}"
        );
        assert!(
            msg.starts_with("request not found:"),
            "RequestNotFound should start with prefix, got: {msg}"
        );
    }

    // @trace TEST-EXEC-STEP-025 [level:unit]
    #[test]
    fn backend_error_unimplemented_static_str() {
        // Arrange: BackendError::Unimplemented with a static string.
        let err = BackendError::Unimplemented("fused_attention_sm100");
        // Act: format via Display and Debug.
        let display = format!("{err}");
        let debug = format!("{err:?}");
        // Assert: Display shows the operation name.
        assert!(
            display.contains("fused_attention_sm100"),
            "Display should contain the feature name, got: {display}"
        );
        // Assert: Debug shows the variant name.
        assert!(
            debug.contains("Unimplemented"),
            "Debug should contain variant name, got: {debug}"
        );
    }

    // @trace TEST-EXEC-STEP-026 [level:unit]
    #[test]
    fn batch_input_sequences_independent_after_clone() {
        // Arrange: BatchInput with one sequence containing a token vector.
        let seq = SequenceInput {
            tokens: vec![10, 20, 30],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0]),
            fused_hidden: None,
        };
        let original = BatchInput { sequences: vec![seq] };
        // Act: clone and modify the clone's token data.
        let mut cloned = original.clone();
        cloned.sequences[0].tokens[0] = 99;
        // Assert: original is unaffected by the clone mutation.
        assert_eq!(original.sequences[0].tokens[0], 10, "original should be unchanged");
        assert_eq!(cloned.sequences[0].tokens[0], 99, "clone should reflect mutation");
    }

    // @trace TEST-EXEC-STEP-027 [level:unit]
    #[test]
    fn validate_page_table_reports_correct_index_in_error() {
        // Arrange: page_table with valid entries followed by an out-of-bounds entry
        // at a specific index.
        let seq = SequenceInput {
            tokens: vec![1],
            position: 0,
            draft_steps: 0,
            page_table: Some(vec![0, 1, 2, 3, 999]),
            fused_hidden: None,
        };
        // Act: validate with total_pages=100 (999 >= 100 is invalid).
        let err = seq.validate_page_table(100).unwrap_err();
        // Assert: error message reports the exact index (4) and value (999).
        assert!(
            err.contains("page_table[4] = 999"),
            "error should pinpoint index 4, got: {err}"
        );
    }

    // @trace TEST-EXEC-STEP-028 [level:unit]
    #[test]
    fn shannon_entropy_binary_uniform_equals_ln2() {
        // Arrange: two equal logits (uniform binary distribution).
        let logits = [5.0, 5.0];
        // Act: compute entropy.
        let h = shannon_entropy(&logits);
        // Assert: should equal ln(2) for a fair coin.
        let expected = 2.0f32.ln();
        assert!(
            (h - expected).abs() < 1e-4,
            "binary uniform entropy should be ln(2)={expected}, got {h}"
        );
    }

    // @trace TEST-EXEC-STEP-029 [level:unit]
    #[test]
    fn kv_cache_handle_hash_distinguishes_values() {
        // Arrange: three handles with distinct values.
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(KvCacheHandle(0), "zero");
        map.insert(KvCacheHandle(1), "one");
        map.insert(KvCacheHandle(u64::MAX), "max");
        // Act: look up each value.
        // Assert: all three entries coexist with correct values.
        assert_eq!(map.get(&KvCacheHandle(0)), Some(&"zero"));
        assert_eq!(map.get(&KvCacheHandle(1)), Some(&"one"));
        assert_eq!(map.get(&KvCacheHandle(u64::MAX)), Some(&"max"));
        assert_eq!(map.get(&KvCacheHandle(2)), None);
    }

    // @trace TEST-EXEC-STEP-030 [level:unit]
    #[test]
    fn extract_top_k_with_mixed_nan_and_valid_returns_valid_length() {
        // Arrange: logits with NaN scattered among valid values.
        let logits = [1.0, f32::NAN, 3.0, f32::NAN, 5.0];
        // Act: request top 3.
        let top3 = extract_top_k_token_ids(&logits, 3);
        // Assert: returns exactly 3 valid indices without panic.
        assert_eq!(top3.len(), 3, "should return k results even with NaN");
        for &id in &top3 {
            assert!(
                (id as usize) < logits.len(),
                "index {id} must be within bounds"
            );
        }
    }
}
