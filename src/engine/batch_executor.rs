//! Batch executor (SPEC/20 REQ-BCI-008)
//!
//! M 维度统一架构的批量推理执行器。
//! batch_size=1 是 batch_size=N 的特例。

use crate::engine::batch_context::{
    BatchContext, BATCH_CTX_HEADER_SIZE, SEQ_META_STRIDE,
    SEQ_ACTIVE_FLAG, SEQ_SEQ_POSITION, SEQ_LAST_SAMPLED_TOKEN,
};
use crate::scheduler::RequestId;
use crate::scheduler::batcher::BatchPrepData;

/// Sampling parameters packed as 4 × u32 per sequence (SPEC/20 §4.3)
const SAMPLING_STRIDE_U32: usize = 4;

/// Generate request for batch inference (SPEC/20 REQ-BCI-008)
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub request_id: RequestId,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub session_id: Option<crate::scheduler::SessionId>,
    pub eos_token_id: u32,
    pub hook_ctx_ptr: *const u8,
    pub callback_table_ptr: *const u8,
}

unsafe impl Send for GenerateRequest {}

/// Generate result for batch inference (SPEC/20 REQ-BCI-008)
#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub request_id: RequestId,
    pub output_tokens: Vec<u32>,
    pub finished: bool,
    pub error: Option<String>,
}

/// Batch inference state (SPEC/20 REQ-BCI-008)
pub struct BatchInferenceState {
    /// M 维度最大值 (用于 buffer 分配)
    pub max_m: usize,
    /// 当前活动序列数
    pub num_active: usize,
    /// BatchContext flat memory
    pub batch_ctx: BatchContext,
    /// input_ids flat buffer
    pub input_ids_flat: Vec<u32>,
    /// output_tokens flat buffer
    pub output_tokens_flat: Vec<u32>,
    /// positions flat buffer
    pub positions_flat: Vec<u32>,
    /// page_table flat buffer
    pub page_table_flat: Vec<u32>,
    /// sampling params packed buffer [temp_bits, top_k, top_p_bits, eos] × N
    pub sampling_params: Vec<u32>,
    /// Per-seq prompt lengths (for result extraction)
    pub prompt_lens: Vec<usize>,
    /// Per-seq max_new_tokens (for result extraction)
    pub max_new_tokens_per_seq: Vec<usize>,
}

impl BatchInferenceState {
    /// Maximum decode steps across all sequences (= max(max_new_tokens)).
    pub fn max_decode_steps(&self) -> usize {
        if self.max_new_tokens_per_seq.is_empty() {
            0
        } else {
            self.max_new_tokens_per_seq.iter().copied().max().unwrap()
        }
    }

    /// Build BatchInferenceState from a list of GenerateRequests.
    ///
    /// Constructs flat memory buffers and BatchContext per SPEC/20 §1.3.
    pub fn build_from_requests(requests: &[GenerateRequest]) -> Self {
        let num_seqs = requests.len();
        let mut prompt_lens = Vec::with_capacity(num_seqs);
        let mut max_new_tokens_per_seq = Vec::with_capacity(num_seqs);
        let mut total_prefill_tokens = 0usize;

        // 1. Compute dimensions
        for req in requests {
            prompt_lens.push(req.prompt_tokens.len());
            max_new_tokens_per_seq.push(req.max_new_tokens);
            total_prefill_tokens += req.prompt_tokens.len();
        }

        let max_decode_steps = if max_new_tokens_per_seq.is_empty() {
            0
        } else {
            max_new_tokens_per_seq.iter().copied().max().unwrap()
        };

        // 2. Allocate flat buffers
        let mut input_ids_flat = Vec::with_capacity(total_prefill_tokens);
        let mut positions_flat = Vec::with_capacity(total_prefill_tokens);
        let max_output_tokens: usize = requests
            .iter()
            .zip(&prompt_lens)
            .map(|(req, pl)| pl + req.max_new_tokens)
            .sum();
        let mut output_tokens_flat = vec![0u32; max_output_tokens];
        let mut sampling_params = vec![0u32; num_seqs * SAMPLING_STRIDE_U32];

        // 3. Fill flat buffers
        let mut token_offset = 0usize;
        for (i, req) in requests.iter().enumerate() {
            // input_ids: concatenate all prompts
            input_ids_flat.extend_from_slice(&req.prompt_tokens);

            // positions: [rope_pos_offset .. rope_pos_offset + prompt_len]
            // For new sequences, rope_pos_offset = 0
            for pos in 0..req.prompt_tokens.len() {
                positions_flat.push((token_offset + pos) as u32);
            }

            // sampling params: packed [temp_bits, top_k, top_p_bits, eos]
            let s_off = i * SAMPLING_STRIDE_U32;
            sampling_params[s_off] = req.temperature.to_bits();
            sampling_params[s_off + 1] = req.top_k as u32;
            sampling_params[s_off + 2] = req.top_p.to_bits();
            sampling_params[s_off + 3] = req.eos_token_id;

            token_offset += req.prompt_tokens.len();
        }

        // 4. Build BatchContext
        let mut batch_ctx = BatchContext::new(num_seqs);
        batch_ctx.set_num_seqs(num_seqs as u32);
        batch_ctx.set_max_decode_steps(max_decode_steps as u32);
        batch_ctx.set_total_prefill_tokens(total_prefill_tokens as u32);
        // Pointers will be set just before calling mega-kernel (after buffers are pinned)
        batch_ctx.set_input_ids_flat_ptr(input_ids_flat.as_ptr());
        batch_ctx.set_output_tokens_flat_ptr(output_tokens_flat.as_mut_ptr());
        batch_ctx.set_positions_ptr(positions_flat.as_ptr());
        batch_ctx.set_sampling_params_ptr(sampling_params.as_ptr());

        // Shared hook/callback pointers from first request (or null)
        let hook_ptr = requests.first().map_or(std::ptr::null(), |r| r.hook_ctx_ptr);
        let cb_ptr = requests.first().map_or(std::ptr::null(), |r| r.callback_table_ptr);
        batch_ctx.set_hook_ctx_ptr(hook_ptr);
        batch_ctx.set_callback_table_ptr(cb_ptr);

        // Per-seq metadata (SPEC/20 §1.2 layout, matches JIT mega_kernel_emit.rs)
        let mut output_offset_acc = 0u32;
        for (i, req) in requests.iter().enumerate() {
            batch_ctx.set_seq_prompt_len(i, req.prompt_tokens.len() as u32);
            batch_ctx.set_seq_kv_len(i, 0); // new sequences have no KV yet
            batch_ctx.set_seq_rope_pos_offset(i, 0); // new sequence starts at position 0
            batch_ctx.set_seq_max_new_tokens(i, req.max_new_tokens as u32);
            batch_ctx.set_seq_session_position(i, req.session_id.map_or(0, |_| 0)); // session resume handled elsewhere
            batch_ctx.set_seq_page_table_offset(i, 0); // contiguous KV (no paging)
            batch_ctx.set_seq_page_table_len(i, 0); // new sequences start with 0 KV pages
            batch_ctx.set_seq_fused_hidden_offset(i, 0); // no multimodal
            batch_ctx.set_seq_num_mm_tokens(i, 0); // no multimodal
            batch_ctx.set_seq_active_flag(i, 1); // 1 = active
            batch_ctx.set_seq_position(i, 0); // current decode position
            batch_ctx.set_seq_gen_count(i, 0); // no generated tokens yet
            batch_ctx.set_seq_last_sampled_token(i, 0); // no sampled token yet
            batch_ctx.set_seq_output_offset(i, output_offset_acc);
            output_offset_acc += (req.prompt_tokens.len() + req.max_new_tokens) as u32;
        }

        // BCI6: Build per-token seq_mapping — seq_mapping[token_idx] → seq_id
        // Replaces CSR indptr (batch_ids, seq_start_pos, seq_lens) tuple.
        // Each sequence contributes prompt_len copies of its seq_idx.
        {
            let mut seq_mapping = Vec::with_capacity(total_prefill_tokens);
            for (i, req) in requests.iter().enumerate() {
                let pl = req.prompt_tokens.len();
                for _ in 0..pl {
                    seq_mapping.push(i as u32);
                }
            }
            batch_ctx.set_seq_mapping_ptr(seq_mapping.as_ptr());
            batch_ctx.seq_mapping = seq_mapping;
        }

        Self {
            max_m: total_prefill_tokens.max(num_seqs),
            num_active: num_seqs,
            batch_ctx,
            input_ids_flat,
            output_tokens_flat,
            positions_flat,
            page_table_flat: Vec::new(), // no paging for now
            sampling_params,
            prompt_lens,
            max_new_tokens_per_seq,
        }
    }

    /// Build BatchInferenceState from BatchPrepData + GenerateRequests (SPEC/20 REQ-BCI-007/008).
    ///
    /// BatchPrepData comes from ContinuousBatcher::build_batch_with_prep() and provides
    /// scheduler-derived per-seq metadata (prompt_lens, kv_lens, page_table info).
    /// GenerateRequests provide sampling params, session info, and hook/callback pointers
    /// that are not available in the scheduler layer.
    pub fn build_from_prep(
        prep: &BatchPrepData,
        requests: &[GenerateRequest],
    ) -> Self {
        let num_seqs = prep.prompt_lens.len();
        let total_prefill_tokens: usize = prep.prompt_lens.iter().map(|&l| l as usize).sum();
        let max_decode_steps = prep.max_decode_steps as usize;

        let mut input_ids_flat = Vec::with_capacity(total_prefill_tokens);
        let mut positions_flat = Vec::with_capacity(total_prefill_tokens);
        let max_output_tokens: usize = prep
            .prompt_lens
            .iter()
            .zip(&prep.max_new_tokens)
            .map(|(&pl, &mn)| pl as usize + mn as usize)
            .sum();
        let mut output_tokens_flat = vec![0u32; max_output_tokens];
        let mut sampling_params = prep.sampling_params_packed.clone();

        // Fill input_ids and positions from requests (batcher returns them in schedule order)
        let mut token_offset = 0usize;
        for (i, req) in requests.iter().enumerate() {
            input_ids_flat.extend_from_slice(&req.prompt_tokens);
            for pos in 0..req.prompt_tokens.len() {
                positions_flat.push((token_offset + pos) as u32);
            }
            // Inject sampling params from request (batcher doesn't have these)
            let s_off = i * SAMPLING_STRIDE_U32;
            sampling_params[s_off] = req.temperature.to_bits();
            sampling_params[s_off + 1] = req.top_k as u32;
            sampling_params[s_off + 2] = req.top_p.to_bits();
            sampling_params[s_off + 3] = req.eos_token_id;
            token_offset += req.prompt_tokens.len();
        }

        // Build BatchContext from BatchPrepData fields
        let mut batch_ctx = BatchContext::new(num_seqs);
        batch_ctx.set_num_seqs(num_seqs as u32);
        batch_ctx.set_max_decode_steps(max_decode_steps as u32);
        batch_ctx.set_total_prefill_tokens(total_prefill_tokens as u32);
        batch_ctx.set_input_ids_flat_ptr(input_ids_flat.as_ptr());
        batch_ctx.set_output_tokens_flat_ptr(output_tokens_flat.as_mut_ptr());
        batch_ctx.set_positions_ptr(positions_flat.as_ptr());
        batch_ctx.set_sampling_params_ptr(sampling_params.as_ptr());

        let hook_ptr = requests.first().map_or(std::ptr::null(), |r| r.hook_ctx_ptr);
        let cb_ptr = requests.first().map_or(std::ptr::null(), |r| r.callback_table_ptr);
        batch_ctx.set_hook_ctx_ptr(hook_ptr);
        batch_ctx.set_callback_table_ptr(cb_ptr);

        // Per-seq metadata from BatchPrepData
        for i in 0..num_seqs {
            batch_ctx.set_seq_prompt_len(i, prep.prompt_lens[i]);
            batch_ctx.set_seq_kv_len(i, prep.kv_lens[i]);
            batch_ctx.set_seq_rope_pos_offset(i, prep.rope_pos_offsets[i]);
            batch_ctx.set_seq_max_new_tokens(i, prep.max_new_tokens[i]);
            batch_ctx.set_seq_session_position(i, prep.session_positions[i]);
            batch_ctx.set_seq_page_table_offset(i, prep.page_table_offsets[i]);
            batch_ctx.set_seq_page_table_len(i, prep.page_table_lens[i]);
            batch_ctx.set_seq_fused_hidden_offset(i, prep.fused_hidden_offsets[i]);
            batch_ctx.set_seq_num_mm_tokens(i, prep.num_mm_tokens[i]);
            batch_ctx.set_seq_active_flag(i, prep.active_flags[i]);
            batch_ctx.set_seq_position(i, prep.seq_positions[i]);
            batch_ctx.set_seq_gen_count(i, prep.gen_counts[i]);
            batch_ctx.set_seq_last_sampled_token(i, prep.last_sampled_tokens[i]);
        }

        // Set output_offsets (cumulative prompt_len + max_new_tokens)
        let mut output_offset_acc = 0u32;
        for i in 0..num_seqs {
            batch_ctx.set_seq_output_offset(i, output_offset_acc);
            output_offset_acc += prep.prompt_lens[i] + prep.max_new_tokens[i];
        }

        // BCI6: per-token seq_mapping
        {
            let mut seq_mapping = Vec::with_capacity(total_prefill_tokens);
            for i in 0..num_seqs {
                for _ in 0..prep.prompt_lens[i] {
                    seq_mapping.push(i as u32);
                }
            }
            batch_ctx.set_seq_mapping_ptr(seq_mapping.as_ptr());
            batch_ctx.seq_mapping = seq_mapping;
        }

        Self {
            max_m: total_prefill_tokens.max(num_seqs),
            num_active: num_seqs,
            batch_ctx,
            input_ids_flat,
            output_tokens_flat,
            positions_flat,
            page_table_flat: Vec::new(),
            sampling_params,
            prompt_lens: prep.prompt_lens.iter().map(|&l| l as usize).collect(),
            max_new_tokens_per_seq: prep.max_new_tokens.iter().map(|&m| m as usize).collect(),
        }
    }

    /// Extract per-sequence results from output_tokens_flat after batch inference.
    pub fn collect_results(&self, requests: &[GenerateRequest]) -> Vec<GenerateResult> {
        let mut results = Vec::with_capacity(requests.len());
        let mut out_off = 0usize;

        for (i, req) in requests.iter().enumerate() {
            let prompt_len = self.prompt_lens[i];
            let max_new = self.max_new_tokens_per_seq[i];
            let seq_total = prompt_len + max_new;

            // Read generated tokens from after prompt position
            let gen_start = out_off + prompt_len;
            let gen_end = (out_off + seq_total).min(self.output_tokens_flat.len());

            let mut output = Vec::new();
            if gen_start < gen_end {
                for &tok in &self.output_tokens_flat[gen_start..gen_end] {
                    if tok == req.eos_token_id {
                        break;
                    }
                    output.push(tok);
                }
            }

            results.push(GenerateResult {
                request_id: req.request_id,
                output_tokens: output,
                finished: true,
                error: None,
            });

            out_off += seq_total;
        }

        results
    }
}

/// Batch executor — M 维度统一架构的批量推理执行器 (SPEC/20 §5)
///
/// 封装 BatchInferenceState 的构建、单指针 CALL (MegaKernelFn(ctx) ABI)、
/// 以及结果提取全流程。batch_size=1 是 batch_size=N 的特例。
///
/// # Single-pointer ABI
/// `prefill_batch` 和 `decode_step` 的 `mega_fn` 闭包接收 `&BatchContext` 单一指针，
/// JIT 通过 BatchContext 中的指针读取所有输入缓冲（input_ids_flat, positions,
/// output_tokens_flat, sampling_params 等），实现 ARCH-RUST-IS-CODEGEN 铁律。
pub struct BatchExecutor<'a> {
    /// Original requests reference for result extraction.
    requests: &'a [GenerateRequest],
    /// Batch inference state (flat buffers + BatchContext).
    state: BatchInferenceState,
}

impl<'a> BatchExecutor<'a> {
    /// Create a new `BatchExecutor` from a slice of `GenerateRequest`s.
    ///
    /// Builds `BatchInferenceState` via `build_from_requests`, which constructs
    /// all flat buffers (input_ids_flat, output_tokens_flat, positions_flat,
    /// sampling_params) and `BatchContext` with per-seq metadata.
    pub fn new(requests: &'a [GenerateRequest]) -> Self {
        let state = BatchInferenceState::build_from_requests(requests);
        Self { requests, state }
    }

    /// Prefill batch: M=sum(prompt_lens), one CALL via single-pointer ABI (REQ-BCI-008).
    ///
    /// Calls `mega_fn` with single `&BatchContext` pointer. The JIT reads all
    /// input buffers (input_ids_flat, positions, sampling_params, seq_mapping)
    /// from BatchContext pointers. This is the unified single-pointer ABI path.
    ///
    /// After prefill completes, the JIT sets `current_pos = prompt_len` and
    /// `phase = Decode` for each sequence in per-seq metadata.
    pub fn prefill_batch<F, E>(&mut self, mega_fn: F) -> Result<(), E>
    where
        F: FnOnce(&BatchContext) -> Result<usize, E>,
    {
        let _total = mega_fn(&self.state.batch_ctx)?;
        Ok(())
    }

    /// Decode one step: M=num_active, one CALL via single-pointer ABI (REQ-BCI-008).
    ///
    /// Collects active (non-terminated) sequences from `batch_ctx` per-seq metadata.
    /// Updates `phase=Decode` and `current_pos` in `BatchContext`, then calls `mega_fn`
    /// with single `&BatchContext` pointer.
    ///
    /// Returns the number of active sequences that decoded (0 = all terminated).
    pub fn decode_step<F, E>(&mut self, mega_fn: F) -> Result<usize, E>
    where
        F: FnOnce(&BatchContext) -> Result<usize, E>,
    {
        let num_seqs = self.state.batch_ctx.num_seqs;

        // Collect active (active_flag=1) sequences from batch_ctx per-seq metadata
        let mut active_indices = Vec::with_capacity(num_seqs);
        for i in 0..num_seqs {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            let active = u32::from_le_bytes(
                self.state.batch_ctx.data[off..off + 4].try_into().unwrap(),
            );
            if active == 1 {
                active_indices.push(i);
            }
        }

        let num_active = active_indices.len();
        if num_active == 0 {
            return Ok(0);
        }

        // Single CALL with BatchContext pointer — JIT reads everything from ctx
        let _tokens_decoded = mega_fn(&self.state.batch_ctx)?;

        // Update active sequence count
        self.state.num_active = num_active;
        Ok(num_active)
    }

    /// Collect per-sequence results after batch inference completes.
    ///
    /// Delegates to `BatchInferenceState::collect_results` with the original
    /// requests slice. Extracts generated tokens from `output_tokens_flat`
    /// per per-seq `prompt_lens` and `max_new_tokens_per_seq`.
    pub fn collect_results(&self) -> Vec<GenerateResult> {
        self.state.collect_results(self.requests)
    }

    /// Returns a reference to the `BatchContext` (for reading per-seq metadata, etc.).
    pub fn batch_ctx(&self) -> &BatchContext {
        &self.state.batch_ctx
    }

    /// Returns a mutable reference to the `BatchContext` (for updating per-seq state).
    pub fn batch_ctx_mut(&mut self) -> &mut BatchContext {
        &mut self.state.batch_ctx
    }
}

/// Execute prefill batch: M=sum(prompt_lens), one CALL for all tokens (BCI3).
///
/// Collects N sequences' prompts via `build_from_requests`, builds flat buffers
/// with M=sum(prompt_lens), and processes all tokens in a single CALL.
/// Uses VmInstr SeqIdLookup to resolve per-seq metadata from seq_id.
///
/// The `execute` closure receives `(batch_ctx, input_ids, positions, total_prefill_tokens)`
/// and should return `Ok(total_generated)` on success.
pub fn execute_prefill<F, E>(
    requests: &[GenerateRequest],
    execute: F,
) -> Result<Vec<GenerateResult>, E>
where
    F: FnOnce(&BatchContext, &[u32], &[u32], usize) -> Result<usize, E>,
{
    let state = BatchInferenceState::build_from_requests(requests);
    let total_prefill = state.input_ids_flat.len();
    execute(
        &state.batch_ctx,
        &state.input_ids_flat,
        &state.positions_flat,
        total_prefill,
    )?;
    Ok(state.collect_results(requests))
}

/// Alias for `execute_prefill` (BCI3).
pub use execute_prefill as prefill_batch;

/// Execute decode step: M=num_active, one token per active sequence (BCI4).
///
/// Collects active (active_flag=1) sequences from the current batch state,
/// builds decode input from the last sampled token for each active sequence,
/// and processes all in a single CALL with M=num_active.
///
/// The `execute` closure receives `(batch_ctx, decode_input, decode_positions, num_active)`
/// and should return `Ok(tokens_decoded)` on success.
pub fn batch_decode<F, E>(
    state: &mut BatchInferenceState,
    execute: F,
) -> Result<usize, E>
where
    F: FnOnce(&BatchContext, &[u32], &[u32], usize) -> Result<usize, E>,
{
    let num_seqs = state.batch_ctx.num_seqs;

    // 1. Collect active (active_flag=1) sequences from batch_ctx metadata
    let mut active_indices = Vec::with_capacity(num_seqs);
    for i in 0..num_seqs {
        let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        let active = u32::from_le_bytes(
            state.batch_ctx.data[off..off + 4].try_into().unwrap(),
        );
        if active == 1 {
            active_indices.push(i);
        }
    }

    let num_active = active_indices.len();
    if num_active == 0 {
        return Ok(0);
    }

    // 2. Build decode input: 1 token per active seq from output_tokens_flat.
    //    Also build decode positions: seq_position per active seq.
    let mut decode_input = Vec::with_capacity(num_active);
    let mut decode_positions = Vec::with_capacity(num_active);

    for &seq_idx in &active_indices {
        // Read seq_position from batch_ctx per-seq metadata
        let pos_off = BATCH_CTX_HEADER_SIZE + seq_idx * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        let seq_pos = u32::from_le_bytes(
            state.batch_ctx.data[pos_off..pos_off + 4].try_into().unwrap(),
        );
        decode_positions.push(seq_pos);

        // Read last_sampled_token for this seq
        let tok_off = BATCH_CTX_HEADER_SIZE + seq_idx * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        let last_token = u32::from_le_bytes(
            state.batch_ctx.data[tok_off..tok_off + 4].try_into().unwrap(),
        );
        decode_input.push(if last_token != 0 { last_token } else { 0 });
    }

    // 3. Single CALL with M = num_active
    let tokens_decoded = execute(
        &state.batch_ctx,
        &decode_input,
        &decode_positions,
        num_active,
    )?;

    // 4. Update state
    state.num_active = num_active;

    Ok(tokens_decoded)
}

/// BCI7: Execute full batch lifecycle via single CALL.
///
/// Builds `BatchInferenceState` from requests, then issues one CALL that
/// handles prefill + decode step loop entirely inside JIT (batch_lifecycle).
/// batch_size=1 is a natural special case: `BatchContext` with one sequence.
///
/// The `execute` closure receives `(batch_ctx, input_ids, positions, total_prefill_tokens)`
/// and should return `Ok(total_generated)` on success. The JIT internally manages
/// the prefill→decode transition and per-seq argmax sampling.
///
/// Returns per-sequence `GenerateResult` vector extracted from flat output buffer.
pub fn execute_batch<F, E>(
    requests: &[GenerateRequest],
    execute: F,
) -> Result<Vec<GenerateResult>, E>
where
    F: FnOnce(&BatchContext, &[u32], &[u32], usize) -> Result<usize, E>,
{
    let state = BatchInferenceState::build_from_requests(requests);
    let total_prefill = state.input_ids_flat.len();
    let max_decode = state.max_decode_steps();
    let _ = max_decode; // JIT uses batch_ctx.max_decode_steps internally

    // Single CALL: prefill + decode step loop entirely inside JIT
    execute(
        &state.batch_ctx,
        &state.input_ids_flat,
        &state.positions_flat,
        total_prefill,
    )?;

    Ok(state.collect_results(requests))
}

/// Alias for `execute_batch` (BCI7).
pub use execute_batch as batch_call;

/// Alias for `execute_batch` (BCI7).
pub use execute_batch as batch_lifecycle;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::batch_context::{
        SEQ_OUTPUT_OFFSET, SEQ_KV_LEN, SEQ_GEN_COUNT, SEQ_SEQ_POSITION,
        SEQ_LAST_SAMPLED_TOKEN, SEQ_ACTIVE_FLAG, SEQ_MAX_NEW_TOKENS,
        SEQ_ROPE_POS_OFFSET,
    };

    #[test]
    fn test_generate_request_send() {
        fn assert_send<T: Send>() {}
        assert_send::<GenerateRequest>();
    }

    #[test]
    fn test_batch_context_builder() {
        let ctx = BatchContext::new(2);
        assert_eq!(ctx.num_seqs, 2);
        // Header (96) + seq_meta_base_ptr + per-seq data
        assert_eq!(ctx.byte_size(), 96 + 2 * 64);
    }

    #[test]
    fn test_build_from_requests() {
        let requests = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![1, 2, 3],
                max_new_tokens: 5,
                temperature: 0.8,
                top_k: 50,
                top_p: 0.95,
                session_id: None,
                eos_token_id: 2,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            GenerateRequest {
                request_id: 2u64,
                prompt_tokens: vec![4, 5],
                max_new_tokens: 3,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 2,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
        ];

        let state = BatchInferenceState::build_from_requests(&requests);

        // Verify dimensions
        assert_eq!(state.num_active, 2);
        assert_eq!(state.prompt_lens, vec![3, 2]);
        assert_eq!(state.max_new_tokens_per_seq, vec![5, 3]);

        // Verify flat input_ids
        assert_eq!(state.input_ids_flat, vec![1, 2, 3, 4, 5]);

        // Verify sampling params
        assert_eq!(state.sampling_params.len(), 8); // 2 seqs × 4
        assert_eq!(state.sampling_params[0], 0.8f32.to_bits()); // seq 0 temperature
        assert_eq!(state.sampling_params[1], 50); // seq 0 top_k
        assert_eq!(state.sampling_params[4], 1.0f32.to_bits()); // seq 1 temperature

        // Verify BatchContext header
        let ctx = &state.batch_ctx;
        assert_eq!(ctx.num_seqs, 2);

        // Verify per-seq metadata
        // seq 0: prompt_len=3, max_new_tokens=5
        // seq 1: prompt_len=2, max_new_tokens=3
    }

    #[test]
    fn test_collect_results() {
        let requests = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![1, 2, 3],
                max_new_tokens: 5,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 99,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
        ];

        let mut state = BatchInferenceState::build_from_requests(&requests);

        // Simulate mega-kernel output: [prompt: 1,2,3, gen: 10, 20, 30, 99(eos), 0]
        // EOS token 99 terminates scanning; token ID 0 is a valid token, not a sentinel.
        state.output_tokens_flat = vec![1, 2, 3, 10, 20, 30, 99, 0];

        let results = state.collect_results(&requests);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
        assert!(results[0].finished);
    }

    /// BCE-20260623-004: Token ID 0 must NOT be treated as EOS sentinel.
    /// The tok==0 sentinel was removed from production code; this test ensures
    /// token ID 0 is correctly collected as a valid generated token.
    #[test]
    fn collect_results_zero_token_not_sentinel() {
        let requests = vec![
            GenerateRequest {
                request_id: 2u64,
                prompt_tokens: vec![5, 6],
                max_new_tokens: 4,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 99,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
        ];

        let mut state = BatchInferenceState::build_from_requests(&requests);

        // Simulate mega-kernel output: [prompt: 5,6, gen: 7, 0(valid!), 8, 99(eos)]
        // Token 0 is a legitimate generated token and must appear in output.
        state.output_tokens_flat = vec![5, 6, 7, 0, 8, 99];

        let results = state.collect_results(&requests);
        assert_eq!(results.len(), 1);
        // 0 must be collected as a valid token, not treated as EOS
        assert_eq!(results[0].output_tokens, vec![7, 0, 8]);
        assert!(results[0].finished);
    }

    // ── Additional data structure tests ──

    #[test]
    fn generate_request_debug_clone() {
        let req = GenerateRequest {
            request_id: 42u64,
            prompt_tokens: vec![1, 2],
            max_new_tokens: 10,
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            session_id: Some(99),
            eos_token_id: 2,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let cloned = req.clone();
        assert_eq!(cloned.request_id, 42);
        assert_eq!(cloned.prompt_tokens, vec![1, 2]);
        assert_eq!(cloned.session_id, Some(99));
        let debug_str = format!("{:?}", req);
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn generate_result_debug_clone() {
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![10, 20, 30],
            finished: true,
            error: None,
        };
        let cloned = result.clone();
        assert_eq!(cloned.output_tokens, vec![10, 20, 30]);
        assert!(cloned.finished);
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("finished"));
    }

    #[test]
    fn generate_result_with_error() {
        let result = GenerateResult {
            request_id: 5u64,
            output_tokens: vec![],
            finished: false,
            error: Some("timeout".to_string()),
        };
        assert!(!result.finished);
        assert_eq!(result.error.as_deref(), Some("timeout"));
    }

    #[test]
    fn max_decode_steps_empty() {
        let state = BatchInferenceState::build_from_requests(&[]);
        assert_eq!(state.max_decode_steps(), 0);
    }

    #[test]
    fn max_decode_steps_single_seq() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 7,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.max_decode_steps(), 7);
    }

    #[test]
    fn max_decode_steps_multi_seq_picks_max() {
        let reqs = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![1],
                max_new_tokens: 3,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            GenerateRequest {
                request_id: 2u64,
                prompt_tokens: vec![2],
                max_new_tokens: 10,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.max_decode_steps(), 10);
    }

    #[test]
    fn collect_results_eos_terminates_early() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1, 2],
            max_new_tokens: 10,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 99,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // prompt_len=2, max_new=10 → output has 12 slots
        // Fill: [1, 2, 10, 20, 99, 0, 0, 0, 0, 0, 0, 0]
        state.output_tokens_flat = vec![1, 2, 10, 20, 99, 0, 0, 0, 0, 0, 0, 0];
        let results = state.collect_results(std::slice::from_ref(&req));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].output_tokens, vec![10, 20]);
    }

    /// BCE-20260623-004: Token ID 0 is NOT a sentinel. Zero-terminated scanning
    /// was removed. This test now verifies that when EOS=999 and generated tokens
    /// contain zeros, those zeros are collected as valid tokens (not treated as EOS).
    #[test]
    fn collect_results_zero_token_terminates() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 5,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 999,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // prompt_len=1, max_new=5 → 6 slots
        // Token 0 is valid; EOS=999 terminates scanning.
        state.output_tokens_flat = vec![1, 10, 20, 0, 999, 0];
        let results = state.collect_results(std::slice::from_ref(&req));
        // 0 is collected as a valid token; 999 (EOS) stops scanning
        assert_eq!(results[0].output_tokens, vec![10, 20, 0]);
    }

    #[test]
    fn build_from_requests_positions() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![10, 20, 30],
            max_new_tokens: 2,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.input_ids_flat, vec![10, 20, 30]);
        assert_eq!(state.positions_flat.len(), 3);
    }

    #[test]
    fn batch_executor_new_and_ctx() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 5,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let reqs = [req];
        let exec = BatchExecutor::new(&reqs);
        let ctx = exec.batch_ctx();
        assert_eq!(ctx.num_seqs, 1);
    }

    #[test]
    fn batch_state_page_table_initially_empty() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert!(state.page_table_flat.is_empty());
    }

    #[test]
    fn sampling_params_packed_correctly() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 3,
            temperature: 0.5,
            top_k: 100,
            top_p: 0.8,
            session_id: None,
            eos_token_id: 2,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params.len(), 4);
        assert_eq!(state.sampling_params[0], 0.5f32.to_bits());
        assert_eq!(state.sampling_params[1], 100);
        assert_eq!(state.sampling_params[2], 0.8f32.to_bits());
        assert_eq!(state.sampling_params[3], 2);
    }

    #[test]
    fn multi_seq_sampling_params_interleaved() {
        let reqs = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![1],
                max_new_tokens: 1,
                temperature: 0.1,
                top_k: 10,
                top_p: 0.5,
                session_id: None,
                eos_token_id: 1,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            GenerateRequest {
                request_id: 2u64,
                prompt_tokens: vec![2],
                max_new_tokens: 1,
                temperature: 0.9,
                top_k: 20,
                top_p: 0.7,
                session_id: None,
                eos_token_id: 3,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.sampling_params.len(), 8);
        assert_eq!(state.sampling_params[0], 0.1f32.to_bits());
        assert_eq!(state.sampling_params[4], 0.9f32.to_bits());
        assert_eq!(state.sampling_params[5], 20);
        assert_eq!(state.sampling_params[7], 3);
    }

    // ── Additional coverage tests ──

    #[test]
    fn max_m_equals_total_prefill_tokens_for_multi_seq() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 2),
            make_req(2, vec![4, 5], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // total_prefill_tokens = 3 + 2 = 5, num_seqs = 2 → max_m = 5
        assert_eq!(state.max_m, 5);
    }

    #[test]
    fn max_m_equals_num_seqs_when_prompts_short() {
        let reqs = vec![
            make_req(1, vec![1], 10),
            make_req(2, vec![2], 10),
            make_req(3, vec![3], 10),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // total_prefill_tokens = 3, num_seqs = 3 → max_m = 3
        assert_eq!(state.max_m, 3);
    }

    #[test]
    fn output_tokens_flat_allocation_size() {
        let reqs = vec![
            make_req(1, vec![1, 2], 3), // prompt_len=2 + max_new=3 = 5
            make_req(2, vec![3], 4),    // prompt_len=1 + max_new=4 = 5
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.output_tokens_flat.len(), 10); // 5 + 5
    }

    #[test]
    fn seq_mapping_correctness() {
        let reqs = vec![
            make_req(1, vec![10, 20], 1), // seq 0 → 2 tokens
            make_req(2, vec![30, 40, 50], 1), // seq 1 → 3 tokens
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let mapping = &state.batch_ctx.seq_mapping;
        assert_eq!(mapping.len(), 5);
        assert_eq!(mapping, &vec![0u32, 0, 1, 1, 1]);
    }

    #[test]
    fn positions_flat_single_seq() {
        let req = make_req(1, vec![100, 200, 300], 2);
        let state = BatchInferenceState::build_from_requests(&[req]);
        // Single seq: positions = [0, 1, 2] (token_offset starts at 0)
        assert_eq!(state.positions_flat, vec![0, 1, 2]);
    }

    #[test]
    fn positions_flat_multi_seq_offset() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 1), // 3 tokens, offset 0..2
            make_req(2, vec![4, 5], 1),     // 2 tokens, offset 3..4
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: token_offset=0 → [0,1,2]; seq 1: token_offset=3 → [3,4]
        assert_eq!(state.positions_flat, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn collect_results_multi_seq_extraction() {
        let reqs = vec![
            make_req_with_eos(1, vec![1, 2], 3, 99),
            make_req_with_eos(2, vec![3], 2, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: 2+3=5 slots → [1,2, 10,20,30]
        // seq 1: 1+2=3 slots → [3, 40,50]
        state.output_tokens_flat = vec![1, 2, 10, 20, 30, 3, 40, 50];
        let results = state.collect_results(&reqs);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
        assert_eq!(results[1].output_tokens, vec![40, 50]);
    }

    /// BCE-20260623-004: All-zero generation output with EOS=0 means first token
    /// hits eos_token_id, yielding empty output (0 is NOT a sentinel, it matches EOS).
    #[test]
    fn collect_results_all_zeros_yields_empty() {
        let req = make_req_with_eos(1, vec![1], 5, 0);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // output_tokens_flat is all zeros from build; EOS=0 means first token is EOS
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    #[test]
    fn build_from_requests_empty() {
        let state = BatchInferenceState::build_from_requests(&[]);
        assert_eq!(state.num_active, 0);
        assert_eq!(state.max_m, 0);
        assert!(state.input_ids_flat.is_empty());
        assert!(state.prompt_lens.is_empty());
        assert!(state.max_new_tokens_per_seq.is_empty());
        assert_eq!(state.sampling_params.len(), 0);
    }

    #[test]
    fn batch_executor_prefill_batch_calls_fn() {
        let req = make_req(1, vec![1, 2], 3);
        let reqs = [req];
        let mut exec = BatchExecutor::new(&reqs);
        let result = exec.prefill_batch(|ctx| -> Result<usize, std::convert::Infallible> {
            assert_eq!(ctx.num_seqs, 1);
            Ok(2)
        });
        assert!(result.is_ok());
    }

    #[test]
    fn batch_executor_decode_step_no_active() {
        let req = make_req(1, vec![1], 1);
        let reqs = [req];
        let mut exec = BatchExecutor::new(&reqs);
        // Set all sequences to inactive
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        exec.batch_ctx_mut().data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        let result: Result<usize, std::convert::Infallible> = exec.decode_step(|_| Ok(0));
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn batch_executor_batch_ctx_mut_allows_mutation() {
        let req = make_req(1, vec![1], 1);
        let reqs = [req];
        let mut exec = BatchExecutor::new(&reqs);
        let ctx = exec.batch_ctx_mut();
        let old = ctx.num_seqs;
        assert_eq!(old, 1);
    }

    #[test]
    fn execute_prefill_with_mock_fn() {
        let reqs = vec![make_req(1, vec![1, 2], 3)];
        let results = execute_prefill(&reqs, |_ctx, inputs, positions, total| {
            assert_eq!(inputs, &[1, 2]);
            assert_eq!(total, 2);
            assert_eq!(positions.len(), 2);
            Ok::<usize, std::convert::Infallible>(2)
        }).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn batch_decode_no_active_returns_zero() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // Mark inactive
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        state.batch_ctx.data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        let result = batch_decode(&mut state, |_, _, _, _| Ok::<usize, std::convert::Infallible>(0));
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn execute_batch_with_mock_fn() {
        let reqs = vec![make_req(1, vec![1], 2)];
        let results = execute_batch(&reqs, |_ctx, inputs, _positions, total| {
            assert_eq!(inputs, &[1]);
            assert_eq!(total, 1);
            Ok::<usize, std::convert::Infallible>(1)
        }).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, 1u64);
    }

    #[test]
    fn batch_executor_collect_results_delegates() {
        let req = make_req_with_eos(1, vec![1], 3, 99);
        let reqs = [req.clone()];
        let mut exec = BatchExecutor::new(&reqs);
        // Simulate output
        exec.state.output_tokens_flat = vec![1, 10, 20, 30];
        let results = exec.collect_results();
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn prompt_lens_matches_requests() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 1),
            make_req(2, vec![4], 1),
            make_req(3, vec![5, 6], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.prompt_lens, vec![3, 1, 2]);
    }

    #[test]
    fn max_new_tokens_per_seq_matches_requests() {
        let reqs = vec![
            make_req(1, vec![1], 5),
            make_req(2, vec![2], 10),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.max_new_tokens_per_seq, vec![5, 10]);
    }

    #[test]
    fn generate_request_fields_default_values() {
        let req = GenerateRequest {
            request_id: 0u64,
            prompt_tokens: vec![],
            max_new_tokens: 0,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert!(req.prompt_tokens.is_empty());
        assert!(req.session_id.is_none());
        assert!(req.hook_ctx_ptr.is_null());
    }

    #[test]
    fn collect_results_request_id_matches() {
        let reqs = vec![
            make_req(42, vec![1], 2),
            make_req(99, vec![2], 2),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![1, 10, 20, 2, 30, 40];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].request_id, 42u64);
        assert_eq!(results[1].request_id, 99u64);
    }

    // ── Constant correctness ──

    #[test]
    fn sampling_stride_u32_is_four() {
        // SPEC/20 §4.3: 4 × u32 per sequence (temp_bits, top_k, top_p_bits, eos)
        assert_eq!(SAMPLING_STRIDE_U32, 4);
    }

    // ── GenerateRequest field variations ──

    #[test]
    fn generate_request_with_session_id_some() {
        let req = GenerateRequest {
            request_id: 7u64,
            prompt_tokens: vec![1, 2],
            max_new_tokens: 3,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.9,
            session_id: Some(42),
            eos_token_id: 2,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.session_id, Some(42));
    }

    #[test]
    fn generate_request_zero_max_new_tokens() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 0,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.max_new_tokens, 0);
    }

    #[test]
    fn generate_request_single_token_prompt() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![42],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.prompt_tokens.len(), 1);
        assert_eq!(req.prompt_tokens[0], 42);
    }

    // ── GenerateResult edge cases ──

    #[test]
    fn generate_result_empty_output_finished() {
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![],
            finished: true,
            error: None,
        };
        assert!(result.finished);
        assert!(result.output_tokens.is_empty());
        assert!(result.error.is_none());
    }

    #[test]
    fn generate_result_with_output_and_error() {
        // Error can coexist with partial output tokens
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![10, 20],
            finished: false,
            error: Some("max tokens exceeded".to_string()),
        };
        assert!(!result.finished);
        assert_eq!(result.output_tokens, vec![10, 20]);
        assert_eq!(result.error.as_deref(), Some("max tokens exceeded"));
    }

    #[test]
    fn generate_result_error_some_none_distinction() {
        let with_error = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![],
            finished: false,
            error: Some("OOM".to_string()),
        };
        let without_error = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![],
            finished: true,
            error: None,
        };
        assert!(with_error.error.is_some());
        assert!(without_error.error.is_none());
    }

    // ── BatchInferenceState field access ──

    #[test]
    fn batch_state_num_active_matches_request_count() {
        let reqs = vec![
            make_req(1, vec![1], 1),
            make_req(2, vec![2], 1),
            make_req(3, vec![3], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.num_active, 3);
    }

    #[test]
    fn batch_state_max_m_single_prompt_dominates() {
        // One long prompt should dominate max_m over num_seqs
        let reqs = vec![
            make_req(1, vec![1; 100], 1),
            make_req(2, vec![2], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // total_prefill = 101, num_seqs = 2 → max_m = 101
        assert_eq!(state.max_m, 101);
    }

    #[test]
    fn batch_state_input_ids_concatenated_in_order() {
        let reqs = vec![
            make_req(1, vec![10, 20], 1),
            make_req(2, vec![30], 1),
            make_req(3, vec![40, 50, 60], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.input_ids_flat, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn batch_state_positions_flat_offsets_accumulate() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 1), // offset 0 → [0,1,2]
            make_req(2, vec![4], 1),         // offset 3 → [3]
            make_req(3, vec![5, 6], 1),      // offset 4 → [4,5]
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.positions_flat, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn batch_state_output_tokens_initially_all_zeros() {
        let reqs = vec![make_req(1, vec![1, 2], 3)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // prompt_len=2 + max_new=3 = 5 slots
        assert_eq!(state.output_tokens_flat.len(), 5);
        assert!(state.output_tokens_flat.iter().all(|&t| t == 0));
    }

    // ── collect_results edge cases ──

    #[test]
    fn collect_results_output_shorter_than_expected() {
        // output_tokens_flat is shorter than prompt_len + max_new_tokens
        let req = make_req_with_eos(1, vec![1, 2], 5, 999);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // Expected 7 slots but only 3 provided
        state.output_tokens_flat = vec![1, 2, 10];
        let results = state.collect_results(&[req]);
        // gen_start = 2 (prompt_len), gen_end = min(7, 3) = 3
        assert_eq!(results[0].output_tokens, vec![10]);
    }

    #[test]
    fn collect_results_first_token_is_eos() {
        let req = make_req_with_eos(1, vec![1, 2], 5, 99);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // First generated token is EOS
        state.output_tokens_flat = vec![1, 2, 99, 10, 20, 30, 40];
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    /// BCE-20260623-004: First generated token is 0 with EOS=0 → correctly stops
    /// because 0 matches eos_token_id, not because 0 is a sentinel.
    #[test]
    fn collect_results_first_token_is_zero() {
        let req = make_req_with_eos(1, vec![1], 5, 0);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // First generated token is 0 which equals eos_token_id=0
        state.output_tokens_flat = vec![1, 0, 10, 20, 30, 40];
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    // ── BatchExecutor lifecycle ──

    #[test]
    fn batch_executor_decode_step_with_active_seqs() {
        let reqs = vec![make_req(1, vec![1, 2], 3)];
        let mut exec = BatchExecutor::new(&reqs);
        let result = exec.decode_step(|ctx| -> Result<usize, std::convert::Infallible> {
            // Verify the context has num_seqs=1
            assert_eq!(ctx.num_seqs, 1);
            Ok(1)
        });
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn batch_executor_prefill_error_propagates() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let mut exec = BatchExecutor::new(&reqs);
        let result: Result<(), &str> = exec.prefill_batch(|_| Err("prefill failed"));
        assert_eq!(result.unwrap_err(), "prefill failed");
    }

    #[test]
    fn batch_executor_decode_step_error_propagates() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let mut exec = BatchExecutor::new(&reqs);
        let result: Result<usize, &str> = exec.decode_step(|_| Err("decode error"));
        assert_eq!(result.unwrap_err(), "decode error");
    }

    // ── execute_prefill / execute_batch error propagation ──

    #[test]
    fn execute_prefill_error_propagates() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let result: Result<Vec<GenerateResult>, &str> = execute_prefill(&reqs, |_, _, _, _| Err("exec fail"));
        assert_eq!(result.unwrap_err(), "exec fail");
    }

    #[test]
    fn execute_batch_error_propagates() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let result: Result<Vec<GenerateResult>, &str> = execute_batch(&reqs, |_, _, _, _| Err("batch fail"));
        assert_eq!(result.unwrap_err(), "batch fail");
    }

    // ── Alias verification ──

    #[test]
    fn prefill_batch_alias_works() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let results = prefill_batch(&reqs, |_, _, _, _| Ok::<usize, std::convert::Infallible>(1)).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn batch_call_alias_works() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let results = batch_call(&reqs, |_, _, _, _| Ok::<usize, std::convert::Infallible>(1)).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn batch_lifecycle_alias_works() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let results = batch_lifecycle(&reqs, |_, _, _, _| Ok::<usize, std::convert::Infallible>(1)).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].request_id, 1u64);
    }

    // ── GenerateRequest temperature edge cases ──

    #[test]
    fn generate_request_temperature_zero() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.temperature, 0.0);
        assert_eq!(req.temperature.to_bits(), 0u32);
    }

    #[test]
    fn generate_request_temperature_negative() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: -1.5,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert!(req.temperature.is_sign_negative());
    }

    #[test]
    fn generate_request_temperature_infinity() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: f32::INFINITY,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert!(req.temperature.is_infinite());
        assert!(req.temperature.is_sign_positive());
    }

    #[test]
    fn generate_request_temperature_nan_preserved_in_sampling() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: f32::NAN,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert!(req.temperature.is_nan());
        // Verify NaN bits are preserved through build_from_requests
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[0], f32::NAN.to_bits());
    }

    // ── Per-seq metadata verification ──

    #[test]
    fn build_from_requests_output_offset_accumulates() {
        let reqs = vec![
            make_req(1, vec![1, 2], 3),   // prompt=2, max_new=3 → 5 slots, offset 0
            make_req(2, vec![3, 4, 5], 2), // prompt=3, max_new=2 → 5 slots, offset 5
            make_req(3, vec![6], 1),       // prompt=1, max_new=1 → 2 slots, offset 10
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        // Verify output offsets via reading flat memory
        let base0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
        let base1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
        let base2 = BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;

        let off0 = u32::from_le_bytes(ctx.data[base0..base0 + 4].try_into().unwrap());
        let off1 = u32::from_le_bytes(ctx.data[base1..base1 + 4].try_into().unwrap());
        let off2 = u32::from_le_bytes(ctx.data[base2..base2 + 4].try_into().unwrap());

        assert_eq!(off0, 0);
        assert_eq!(off1, 5);
        assert_eq!(off2, 10);
    }

    #[test]
    fn build_from_requests_active_flags_all_set() {
        let reqs = vec![
            make_req(1, vec![1], 1),
            make_req(2, vec![2], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        for i in 0..2 {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            let flag = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(flag, 1, "seq {} should be active", i);
        }
    }

    #[test]
    fn build_from_requests_kv_lens_all_zero_for_new_seqs() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 5),
            make_req(2, vec![4], 10),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        for i in 0..2 {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_KV_LEN;
            let kv_len = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(kv_len, 0, "new seq {} should have kv_len=0", i);
        }
    }

    #[test]
    fn build_from_requests_gen_count_and_position_zero() {
        let reqs = vec![make_req(1, vec![1, 2], 3)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        let gen_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_GEN_COUNT;
        let pos_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        let tok_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;

        let gen_count = u32::from_le_bytes(ctx.data[gen_off..gen_off + 4].try_into().unwrap());
        let seq_pos = u32::from_le_bytes(ctx.data[pos_off..pos_off + 4].try_into().unwrap());
        let last_tok = u32::from_le_bytes(ctx.data[tok_off..tok_off + 4].try_into().unwrap());

        assert_eq!(gen_count, 0);
        assert_eq!(seq_pos, 0);
        assert_eq!(last_tok, 0);
    }

    // ── collect_results edge cases ──

    /// BCE-20260623-004: Token 0 is not a sentinel. For a sequence with all-zero
    /// generation output and EOS=99, zeros are valid tokens, not terminators.
    /// To test "one seq has no real generation", use EOS=0 so 0 matches eos_token_id.
    #[test]
    fn collect_results_one_seq_empty_generation_other_has_tokens() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 3, 99), // seq 0: prompt=1, max_new=3, EOS=99
            make_req_with_eos(2, vec![2], 3, 0),  // seq 1: prompt=1, max_new=3, EOS=0
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: [prompt=1, gen=10, 20, 30] → 4 slots
        // seq 1: [prompt=2, gen=0, 0, 0]    → 4 slots (0 = EOS for seq 1)
        state.output_tokens_flat = vec![1, 10, 20, 30, 2, 0, 0, 0];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
        assert!(results[1].output_tokens.is_empty());
    }

    #[test]
    fn collect_results_empty_prompt_with_generation() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![],
            max_new_tokens: 3,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 999,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // prompt_len=0, max_new=3 → output starts immediately
        state.output_tokens_flat = vec![10, 20, 30];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn collect_results_max_new_zero_yields_empty() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 0,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 999,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req.clone()]);
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    // ── batch_decode with active sequences ──

    #[test]
    fn batch_decode_active_seq_reads_last_sampled_token() {
        let reqs = vec![make_req_with_eos(1, vec![1, 2], 3, 99)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);

        // Set last_sampled_token for seq 0 to a non-zero value
        let tok_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        state.batch_ctx.data[tok_off..tok_off + 4].copy_from_slice(&42u32.to_le_bytes());

        // Set seq_position to non-zero
        let pos_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        state.batch_ctx.data[pos_off..pos_off + 4].copy_from_slice(&5u32.to_le_bytes());

        let result = batch_decode(&mut state, |_, decode_input, decode_positions, num_active| {
            assert_eq!(num_active, 1);
            assert_eq!(decode_input.len(), 1);
            assert_eq!(decode_input[0], 42);
            assert_eq!(decode_positions[0], 5);
            Ok::<usize, std::convert::Infallible>(1)
        });
        assert_eq!(result.unwrap(), 1);
        assert_eq!(state.num_active, 1);
    }

    // ── BatchExecutor num_active tracking ──

    #[test]
    fn batch_executor_decode_step_updates_num_active() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        assert_eq!(exec.state.num_active, 2);

        let result = exec.decode_step(|_| Ok::<usize, std::convert::Infallible>(2));
        assert_eq!(result.unwrap(), 2);
        assert_eq!(exec.state.num_active, 2);
    }

    // ── Free function empty requests ──

    #[test]
    fn execute_prefill_empty_requests() {
        let results = execute_prefill(&[], |_ctx, inputs, _positions, total| {
            assert!(inputs.is_empty());
            assert_eq!(total, 0);
            Ok::<usize, std::convert::Infallible>(0)
        }).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn execute_batch_empty_requests() {
        let results = execute_batch(&[], |_ctx, inputs, _positions, total| {
            assert!(inputs.is_empty());
            assert_eq!(total, 0);
            Ok::<usize, std::convert::Infallible>(0)
        }).unwrap();
        assert!(results.is_empty());
    }

    // ── Sampling params edge cases ──

    #[test]
    fn sampling_params_top_k_zero_allowed() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[1], 0); // top_k = 0
    }

    #[test]
    fn sampling_params_top_p_boundary_values() {
        let req_zero = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req_zero]);
        assert_eq!(state.sampling_params[2], 0.0f32.to_bits());

        let req_one = GenerateRequest {
            request_id: 2u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req_one]);
        assert_eq!(state.sampling_params[2], 1.0f32.to_bits());
    }

    // ── New tests: BatchPrepData + build_from_prep integration ──

    #[test]
    fn build_from_prep_single_seq_metadata_fields() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 3;
        prep.kv_lens[0] = 5;
        prep.rope_pos_offsets[0] = 2;
        prep.max_new_tokens[0] = 4;
        prep.active_flags[0] = 1;
        prep.seq_positions[0] = 7;
        prep.gen_counts[0] = 1;
        prep.last_sampled_tokens[0] = 42;
        prep.max_decode_steps = 4;
        prep.total_prefill_tokens = 3;

        let reqs = vec![make_req(1, vec![10, 20, 30], 4)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);

        assert_eq!(state.num_active, 1);
        assert_eq!(state.prompt_lens, vec![3]);
        assert_eq!(state.max_new_tokens_per_seq, vec![4]);
        assert_eq!(state.input_ids_flat, vec![10, 20, 30]);
        assert_eq!(state.max_m, 3); // max(3, 1) = 3
    }

    #[test]
    fn build_from_prep_multi_seq_sampling_params_from_requests() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![2, 1];
        prep.kv_lens = vec![0, 0];
        prep.rope_pos_offsets = vec![0, 0];
        prep.max_new_tokens = vec![3, 2];
        prep.active_flags = vec![1, 1];
        prep.seq_positions = vec![0, 0];
        prep.gen_counts = vec![0, 0];
        prep.last_sampled_tokens = vec![0, 0];
        prep.page_table_offsets = vec![0, 0];
        prep.page_table_lens = vec![0, 0];
        prep.fused_hidden_offsets = vec![0, 0];
        prep.num_mm_tokens = vec![0, 0];
        prep.session_positions = vec![0, 0];
        prep.sampling_params_packed = vec![0u32; 8];
        prep.max_decode_steps = 3;
        prep.total_prefill_tokens = 3;

        let reqs = vec![
            make_req_with_eos(1, vec![1, 2], 3, 99),
            make_req_with_eos(2, vec![3], 2, 77),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);

        // sampling_params should be overwritten from requests
        assert_eq!(state.sampling_params[3], 99); // seq 0 eos
        assert_eq!(state.sampling_params[7], 77); // seq 1 eos
    }

    #[test]
    fn build_from_prep_output_offsets_accumulate() {
        let mut prep = BatchPrepData::new(3);
        prep.prompt_lens = vec![2, 3, 1];
        prep.kv_lens = vec![0, 0, 0];
        prep.rope_pos_offsets = vec![0, 0, 0];
        prep.max_new_tokens = vec![2, 1, 3];
        prep.active_flags = vec![1, 1, 1];
        prep.seq_positions = vec![0, 0, 0];
        prep.gen_counts = vec![0, 0, 0];
        prep.last_sampled_tokens = vec![0, 0, 0];
        prep.page_table_offsets = vec![0, 0, 0];
        prep.page_table_lens = vec![0, 0, 0];
        prep.fused_hidden_offsets = vec![0, 0, 0];
        prep.num_mm_tokens = vec![0, 0, 0];
        prep.session_positions = vec![0, 0, 0];
        prep.sampling_params_packed = vec![0u32; 12];
        prep.max_decode_steps = 3;
        prep.total_prefill_tokens = 6;

        let reqs = vec![
            make_req(1, vec![1, 2], 2),
            make_req(2, vec![3, 4, 5], 1),
            make_req(3, vec![6], 3),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;

        let base0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
        let base1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
        let base2 = BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;

        let off0 = u32::from_le_bytes(ctx.data[base0..base0 + 4].try_into().unwrap());
        let off1 = u32::from_le_bytes(ctx.data[base1..base1 + 4].try_into().unwrap());
        let off2 = u32::from_le_bytes(ctx.data[base2..base2 + 4].try_into().unwrap());

        // seq 0: prompt=2 + max_new=2 = 4 → offset 0
        // seq 1: prompt=3 + max_new=1 = 4 → offset 4
        // seq 2: prompt=1 + max_new=3 = 4 → offset 8
        assert_eq!(off0, 0);
        assert_eq!(off1, 4);
        assert_eq!(off2, 8);
    }

    #[test]
    fn build_from_prep_seq_mapping_per_token() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![3, 2];
        prep.kv_lens = vec![0, 0];
        prep.rope_pos_offsets = vec![0, 0];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.seq_positions = vec![0, 0];
        prep.gen_counts = vec![0, 0];
        prep.last_sampled_tokens = vec![0, 0];
        prep.page_table_offsets = vec![0, 0];
        prep.page_table_lens = vec![0, 0];
        prep.fused_hidden_offsets = vec![0, 0];
        prep.num_mm_tokens = vec![0, 0];
        prep.session_positions = vec![0, 0];
        prep.sampling_params_packed = vec![0u32; 8];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 5;

        let reqs = vec![
            make_req(1, vec![1, 2, 3], 1),
            make_req(2, vec![4, 5], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);

        let mapping = &state.batch_ctx.seq_mapping;
        assert_eq!(mapping, &vec![0u32, 0, 0, 1, 1]);
    }

    // ── Large batch collect_results ──

    #[test]
    fn collect_results_five_seqs_distinct_outputs() {
        let reqs: Vec<GenerateRequest> = (1..=5)
            .map(|i| make_req_with_eos(i, vec![i as u32], 2, 99))
            .collect();
        let mut state = BatchInferenceState::build_from_requests(&reqs);

        // Each seq: prompt=1 token + max_new=2 = 3 slots
        // seq 1: [1, 10, 20]
        // seq 2: [2, 30, 40]
        // seq 3: [3, 50, 60]
        // seq 4: [4, 70, 80]
        // seq 5: [5, 90, 100]
        state.output_tokens_flat = vec![1, 10, 20, 2, 30, 40, 3, 50, 60, 4, 70, 80, 5, 90, 100];
        let results = state.collect_results(&reqs);

        assert_eq!(results.len(), 5);
        assert_eq!(results[0].output_tokens, vec![10, 20]);
        assert_eq!(results[1].output_tokens, vec![30, 40]);
        assert_eq!(results[2].output_tokens, vec![50, 60]);
        assert_eq!(results[3].output_tokens, vec![70, 80]);
        assert_eq!(results[4].output_tokens, vec![90, 100]);

        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.request_id, (i + 1) as u64);
        }
    }

    // ── BatchExecutor with non-null hook/callback pointers ──

    #[test]
    fn batch_executor_null_pointers_when_no_requests() {
        let state = BatchInferenceState::build_from_requests(&[]);
        assert!(state.input_ids_flat.is_empty());
        assert!(state.positions_flat.is_empty());
        assert!(state.sampling_params.is_empty());
    }

    // ── eos_token_id = 0 behavior ──

    /// When eos_token_id=0, token 0 correctly stops generation via
    /// `tok == req.eos_token_id` (not via a sentinel).
    #[test]
    fn collect_results_eos_zero_stops_on_first_zero() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1, 2],
            max_new_tokens: 5,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // eos=0 → first gen token is 0 which matches eos_token_id → empty output
        state.output_tokens_flat = vec![1, 2, 0, 10, 20, 30, 40];
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    // ── max_new_tokens all equal across batch ──

    #[test]
    fn max_decode_steps_all_equal() {
        let reqs: Vec<GenerateRequest> = (1..=4)
            .map(|i| make_req(i, vec![i as u32], 5))
            .collect();
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.max_decode_steps(), 5);
    }

    // ── BatchContext header fields after build_from_requests ──

    #[test]
    fn batch_ctx_header_total_prefill_tokens_matches_input() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 2),
            make_req(2, vec![4, 5], 3),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        // total_prefill_tokens in header = sum of all prompt lens = 3 + 2 = 5
        let tpf_off = 8; // TOTAL_PREFILL_TOKENS offset
        let tpf = u32::from_le_bytes(ctx.data[tpf_off..tpf_off + 4].try_into().unwrap());
        assert_eq!(tpf, 5);
    }

    #[test]
    fn batch_ctx_header_max_decode_steps_matches_max_new() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 7),
            make_req(3, vec![3], 2),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        let mds_off = 4; // MAX_DECODE_STEPS offset
        let mds = u32::from_le_bytes(ctx.data[mds_off..mds_off + 4].try_into().unwrap());
        assert_eq!(mds, 7);
    }

    // ── Per-seq prompt_len metadata ──

    #[test]
    fn per_seq_prompt_len_matches_actual_prompt() {
        let reqs = vec![
            make_req(1, vec![10, 20], 1),
            make_req(2, vec![30, 40, 50, 60], 1),
            make_req(3, vec![70], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        let expected_lens = [2u32, 4, 1];
        for (i, &expected) in expected_lens.iter().enumerate() {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + 0; // SEQ_PROMPT_LEN = 0
            let pl = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(pl, expected, "seq {} prompt_len mismatch", i);
        }
    }

    // ── Per-seq max_new_tokens metadata ──

    #[test]
    fn per_seq_max_new_tokens_matches_request() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 8),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        let off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_MAX_NEW_TOKENS;
        let off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_MAX_NEW_TOKENS;

        let mnt0 = u32::from_le_bytes(ctx.data[off0..off0 + 4].try_into().unwrap());
        let mnt1 = u32::from_le_bytes(ctx.data[off1..off1 + 4].try_into().unwrap());

        assert_eq!(mnt0, 3);
        assert_eq!(mnt1, 8);
    }

    // ── build_from_prep propagates active flags from prep ──

    #[test]
    fn build_from_prep_respects_inactive_flags() {
        let mut prep = BatchPrepData::new(3);
        prep.prompt_lens = vec![1, 1, 1];
        prep.kv_lens = vec![0, 5, 0];
        prep.max_new_tokens = vec![2, 1, 3];
        prep.active_flags = vec![1, 0, 1]; // seq 1 is inactive
        prep.max_decode_steps = 3;
        prep.total_prefill_tokens = 3;

        let reqs = vec![
            make_req(1, vec![1], 2),
            make_req(2, vec![2], 1),
            make_req(3, vec![3], 3),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;

        // Check active flags in flat memory
        for i in 0..3 {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            let flag = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(flag, prep.active_flags[i], "seq {} active flag mismatch", i);
        }
    }

    // ── build_from_prep propagates kv_lens from prep ──

    #[test]
    fn build_from_prep_propagates_kv_lens() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.kv_lens = vec![7, 3];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;

        let reqs = vec![
            make_req(1, vec![1], 1),
            make_req(2, vec![2], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;

        let off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_KV_LEN;
        let off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_KV_LEN;

        let kv0 = u32::from_le_bytes(ctx.data[off0..off0 + 4].try_into().unwrap());
        let kv1 = u32::from_le_bytes(ctx.data[off1..off1 + 4].try_into().unwrap());

        assert_eq!(kv0, 7);
        assert_eq!(kv1, 3);
    }

    // ── build_from_prep propagates gen_counts and last_sampled_tokens ──

    #[test]
    fn build_from_prep_propagates_gen_counts_and_last_token() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.kv_lens = vec![3, 2];
        prep.max_new_tokens = vec![5, 3];
        prep.active_flags = vec![1, 1];
        prep.gen_counts = vec![2, 1];
        prep.last_sampled_tokens = vec![42, 77];
        prep.seq_positions = vec![3, 2];
        prep.max_decode_steps = 5;
        prep.total_prefill_tokens = 2;

        let reqs = vec![
            make_req(1, vec![1], 5),
            make_req(2, vec![2], 3),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;

        let gc_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_GEN_COUNT;
        let gc_off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_GEN_COUNT;
        let tok_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        let tok_off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;

        let gc0 = u32::from_le_bytes(ctx.data[gc_off0..gc_off0 + 4].try_into().unwrap());
        let gc1 = u32::from_le_bytes(ctx.data[gc_off1..gc_off1 + 4].try_into().unwrap());
        let tok0 = u32::from_le_bytes(ctx.data[tok_off0..tok_off0 + 4].try_into().unwrap());
        let tok1 = u32::from_le_bytes(ctx.data[tok_off1..tok_off1 + 4].try_into().unwrap());

        assert_eq!(gc0, 2);
        assert_eq!(gc1, 1);
        assert_eq!(tok0, 42);
        assert_eq!(tok1, 77);
    }

    // ── build_from_prep output_tokens_flat allocation size ──

    #[test]
    fn build_from_prep_output_allocation_size() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![3, 2];
        prep.max_new_tokens = vec![4, 1];
        prep.kv_lens = vec![0, 0];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 4;
        prep.total_prefill_tokens = 5;

        let reqs = vec![
            make_req(1, vec![1, 2, 3], 4),
            make_req(2, vec![4, 5], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);

        // seq 0: prompt=3 + max_new=4 = 7; seq 1: prompt=2 + max_new=1 = 3 → total=10
        assert_eq!(state.output_tokens_flat.len(), 10);
        assert!(state.output_tokens_flat.iter().all(|&t| t == 0));
    }

    // ── BatchExecutor collect_results preserves request order ──

    #[test]
    fn batch_executor_collect_results_preserves_order() {
        let reqs = vec![
            make_req(10, vec![1], 2),
            make_req(20, vec![2], 2),
            make_req(30, vec![3], 2),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        // Simulate output: each seq has prompt=1 + max_new=2 = 3 slots
        exec.state.output_tokens_flat = vec![1, 100, 101, 2, 200, 201, 3, 300, 301];
        let results = exec.collect_results();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].request_id, 10u64);
        assert_eq!(results[0].output_tokens, vec![100, 101]);
        assert_eq!(results[1].request_id, 20u64);
        assert_eq!(results[1].output_tokens, vec![200, 201]);
        assert_eq!(results[2].request_id, 30u64);
        assert_eq!(results[2].output_tokens, vec![300, 301]);
    }

    // ── batch_decode updates num_active correctly ──

    #[test]
    fn batch_decode_updates_num_active() {
        let reqs = vec![make_req(1, vec![1], 3), make_req(2, vec![2], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.num_active, 2);

        let result = batch_decode(&mut state, |_, _, _, num_active| {
            assert_eq!(num_active, 2);
            Ok::<usize, std::convert::Infallible>(2)
        });
        assert_eq!(result.unwrap(), 2);
        assert_eq!(state.num_active, 2);
    }

    // ── GenerateRequest with non-null hook_ctx_ptr preserves pointer ──

    #[test]
    fn generate_request_non_null_hook_ctx_ptr() {
        let dummy_data: u8 = 42;
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: &dummy_data as *const u8,
            callback_table_ptr: std::ptr::null(),
        };
        assert!(!req.hook_ctx_ptr.is_null());
        assert!(req.callback_table_ptr.is_null());
    }

    // ── New tests: build_from_prep with empty requests ──

    #[test]
    fn build_from_prep_single_seq_positions_accumulate() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 4;
        prep.max_new_tokens[0] = 2;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 2;
        prep.total_prefill_tokens = 4;

        let reqs = vec![make_req(1, vec![10, 20, 30, 40], 2)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        // Single seq: token_offset=0 → [0, 1, 2, 3]
        assert_eq!(state.positions_flat, vec![0, 1, 2, 3]);
        assert_eq!(state.input_ids_flat, vec![10, 20, 30, 40]);
    }

    #[test]
    fn build_from_prep_multi_seq_positions_accumulate() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![2, 3];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 5;

        let reqs = vec![
            make_req(1, vec![1, 2], 1),
            make_req(2, vec![3, 4, 5], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        // seq 0: token_offset=0 → [0, 1]; seq 1: token_offset=2 → [2, 3, 4]
        assert_eq!(state.positions_flat, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn build_from_prep_output_tokens_all_zeros() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![2, 3];
        prep.max_new_tokens = vec![3, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 3;
        prep.total_prefill_tokens = 5;

        let reqs = vec![
            make_req(1, vec![1, 2], 3),
            make_req(2, vec![3, 4, 5], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        // seq 0: 2+3=5 slots; seq 1: 3+1=4 slots → total=9
        assert_eq!(state.output_tokens_flat.len(), 9);
        assert!(state.output_tokens_flat.iter().all(|&t| t == 0));
    }

    #[test]
    fn build_from_prep_sampling_params_temperature_from_request() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 1;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 1;
        prep.sampling_params_packed = vec![0u32; 4];

        let mut req = make_req(1, vec![1], 1);
        req.temperature = 0.42;
        let state = BatchInferenceState::build_from_prep(&prep, &[req]);
        assert_eq!(state.sampling_params[0], 0.42f32.to_bits());
    }

    // ── Sampling params NaN/Inf preservation through build_from_requests ──

    #[test]
    fn sampling_params_nan_temperature_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.temperature = f32::NAN;
        let state = BatchInferenceState::build_from_requests(&[req]);
        let bits = state.sampling_params[0];
        assert!(f32::from_bits(bits).is_nan());
    }

    #[test]
    fn sampling_params_inf_temperature_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.temperature = f32::INFINITY;
        let state = BatchInferenceState::build_from_requests(&[req]);
        let bits = state.sampling_params[0];
        assert!(f32::from_bits(bits).is_infinite() && f32::from_bits(bits).is_sign_positive());
    }

    #[test]
    fn sampling_params_neg_inf_temperature_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.temperature = f32::NEG_INFINITY;
        let state = BatchInferenceState::build_from_requests(&[req]);
        let bits = state.sampling_params[0];
        assert!(f32::from_bits(bits).is_infinite() && f32::from_bits(bits).is_sign_negative());
    }

    #[test]
    fn sampling_params_nan_top_p_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.top_p = f32::NAN;
        let state = BatchInferenceState::build_from_requests(&[req]);
        let bits = state.sampling_params[2];
        assert!(f32::from_bits(bits).is_nan());
    }

    // ── Multi-seq result collection with mixed EOS patterns ──

    #[test]
    fn collect_results_mixed_eos_patterns() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 5, 10), // eos=10
            make_req_with_eos(2, vec![2], 5, 20), // eos=20
            make_req_with_eos(3, vec![3], 5, 99), // eos=99 (not present)
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // Each seq: prompt=1 + max_new=5 = 6 slots
        // seq 1: [1, 5, 10, 0, 0, 0] → output: [5]
        // seq 2: [2, 7, 8, 20, 0, 0] → output: [7, 8]
        // seq 3: [3, 11, 12, 13, 14, 15] → output: [11, 12, 13, 14, 15]
        state.output_tokens_flat = vec![
            1, 5, 10, 0, 0, 0,
            2, 7, 8, 20, 0, 0,
            3, 11, 12, 13, 14, 15,
        ];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![5]);
        assert_eq!(results[1].output_tokens, vec![7, 8]);
        assert_eq!(results[2].output_tokens, vec![11, 12, 13, 14, 15]);
    }

    #[test]
    fn collect_results_all_seqs_hit_eos_immediately() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 3, 10),
            make_req_with_eos(2, vec![2], 3, 20),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: [1, 10, 0, 0]; seq 1: [2, 20, 0, 0]
        state.output_tokens_flat = vec![1, 10, 0, 0, 2, 20, 0, 0];
        let results = state.collect_results(&reqs);
        assert!(results[0].output_tokens.is_empty());
        assert!(results[1].output_tokens.is_empty());
    }

    #[test]
    fn collect_results_output_filled_exactly_to_boundary() {
        let req = make_req_with_eos(1, vec![1, 2], 3, 99);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // prompt=2, max_new=3 → 5 slots total
        // Fill exactly: [1, 2, 10, 20, 30] — all gen slots used, no EOS, no zero
        state.output_tokens_flat = vec![1, 2, 10, 20, 30];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
    }

    // ── batch_decode with mixed active/inactive sequences ──

    #[test]
    fn batch_decode_partial_inactive_seqs() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);

        // Mark seq 1 as inactive
        let off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        state.batch_ctx.data[off1..off1 + 4].copy_from_slice(&0u32.to_le_bytes());

        // Set last_sampled_token and position for active seqs
        let tok_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        let pos_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        state.batch_ctx.data[tok_off0..tok_off0 + 4].copy_from_slice(&100u32.to_le_bytes());
        state.batch_ctx.data[pos_off0..pos_off0 + 4].copy_from_slice(&7u32.to_le_bytes());

        let tok_off2 = BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        let pos_off2 = BATCH_CTX_HEADER_SIZE + 2 * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        state.batch_ctx.data[tok_off2..tok_off2 + 4].copy_from_slice(&200u32.to_le_bytes());
        state.batch_ctx.data[pos_off2..pos_off2 + 4].copy_from_slice(&3u32.to_le_bytes());

        let result = batch_decode(&mut state, |_, decode_input, decode_positions, num_active| {
            assert_eq!(num_active, 2);
            assert_eq!(decode_input.len(), 2);
            assert_eq!(decode_input[0], 100);
            assert_eq!(decode_input[1], 200);
            assert_eq!(decode_positions[0], 7);
            assert_eq!(decode_positions[1], 3);
            Ok::<usize, std::convert::Infallible>(2)
        });
        assert_eq!(result.unwrap(), 2);
        assert_eq!(state.num_active, 2);
    }

    // ── BatchExecutor decode_step with mixed active/inactive ──

    #[test]
    fn batch_executor_decode_step_mixed_active() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
        ];
        let mut exec = BatchExecutor::new(&reqs);

        // Mark seq 0 inactive
        let off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        exec.batch_ctx_mut().data[off0..off0 + 4].copy_from_slice(&0u32.to_le_bytes());

        let result = exec.decode_step(|ctx| -> Result<usize, std::convert::Infallible> {
            // Only seq 1 is active → num_active should be 1
            assert_eq!(ctx.num_seqs, 2);
            Ok(1)
        });
        assert_eq!(result.unwrap(), 1);
    }

    // ── Per-seq metadata: rope_pos_offset is zero for new sequences ──

    #[test]
    fn build_from_requests_rope_pos_offset_zero() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 5),
            make_req(2, vec![4, 5], 3),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        for i in 0..2 {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ROPE_POS_OFFSET;
            let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(val, 0, "new seq {} should have rope_pos_offset=0", i);
        }
    }

    // ── Per-seq metadata: session_position ──

    #[test]
    fn build_from_requests_session_position_zero() {
        let reqs = vec![
            make_req(1, vec![1], 1),
            make_req(2, vec![2], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        for i in 0..2 {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + 16; // SEQ_SESSION_POSITION
            let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(val, 0, "new seq {} should have session_position=0", i);
        }
    }

    // ── Per-seq metadata: fused_hidden_offset and num_mm_tokens ──

    #[test]
    fn build_from_requests_no_multimodal_fields_zero() {
        let reqs = vec![make_req(1, vec![1, 2], 3)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        let fh_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 28; // SEQ_FUSED_HIDDEN_OFFSET
        let mm_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 32; // SEQ_NUM_MM_TOKENS

        let fh = u32::from_le_bytes(ctx.data[fh_off..fh_off + 4].try_into().unwrap());
        let mm = u32::from_le_bytes(ctx.data[mm_off..mm_off + 4].try_into().unwrap());
        assert_eq!(fh, 0);
        assert_eq!(mm, 0);
    }

    // ── Per-seq metadata: page_table_offset and page_table_len ──

    #[test]
    fn build_from_requests_no_paging_fields_zero() {
        let reqs = vec![make_req(1, vec![1, 2, 3], 2)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        let pt_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 20; // SEQ_PAGE_TABLE_OFFSET
        let pt_len = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 24; // SEQ_PAGE_TABLE_LEN

        let off = u32::from_le_bytes(ctx.data[pt_off..pt_off + 4].try_into().unwrap());
        let len = u32::from_le_bytes(ctx.data[pt_len..pt_len + 4].try_into().unwrap());
        assert_eq!(off, 0);
        assert_eq!(len, 0);
    }

    // ── BatchContext header pointer fields ──

    #[test]
    fn batch_ctx_header_pointers_set_after_build() {
        let reqs = vec![make_req(1, vec![1, 2], 3)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        // Check input_ids_flat_ptr is non-null (it points to the Vec's buffer)
        let ptr_bytes = &ctx.data[16..24];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_ne!(ptr_val, 0, "input_ids_flat_ptr should be non-null");

        // Check output_tokens_flat_ptr is non-null
        let ptr_bytes = &ctx.data[24..32];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_ne!(ptr_val, 0, "output_tokens_flat_ptr should be non-null");

        // Check positions_ptr is non-null
        let ptr_bytes = &ctx.data[32..40];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_ne!(ptr_val, 0, "positions_ptr should be non-null");

        // Check sampling_params_ptr is non-null
        let ptr_bytes = &ctx.data[56..64];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_ne!(ptr_val, 0, "sampling_params_ptr should be non-null");
    }

    #[test]
    fn batch_ctx_hook_callback_pointers_null_for_null_requests() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        // hook_ctx_ptr at offset 64
        let hook_bytes = &ctx.data[64..72];
        let hook_val = u64::from_le_bytes(hook_bytes.try_into().unwrap());
        assert_eq!(hook_val, 0, "hook_ctx_ptr should be null");

        // callback_table_ptr at offset 72
        let cb_bytes = &ctx.data[72..80];
        let cb_val = u64::from_le_bytes(cb_bytes.try_into().unwrap());
        assert_eq!(cb_val, 0, "callback_table_ptr should be null");
    }

    // ── Large batch: 10 sequences ──

    #[test]
    fn build_from_requests_ten_seqs_dimensions() {
        let reqs: Vec<GenerateRequest> = (1..=10)
            .map(|i| make_req(i, vec![i as u32; i as usize], 5))
            .collect();
        let state = BatchInferenceState::build_from_requests(&reqs);

        assert_eq!(state.num_active, 10);
        // total_prefill = 1+2+3+4+5+6+7+8+9+10 = 55
        let expected_total: usize = (1..=10).sum();
        assert_eq!(state.input_ids_flat.len(), expected_total);
        assert_eq!(state.max_m, expected_total); // 55 > 10
        assert_eq!(state.sampling_params.len(), 10 * 4);
    }

    #[test]
    fn collect_results_ten_seqs_extraction() {
        let reqs: Vec<GenerateRequest> = (1..=10)
            .map(|i| make_req_with_eos(i, vec![i as u32], 2, 99))
            .collect();
        let mut state = BatchInferenceState::build_from_requests(&reqs);

        // Each seq: prompt=1 + max_new=2 = 3 slots
        // Fill with prompt_token, gen+10, gen+20 for each
        let mut flat = Vec::new();
        for i in 1u32..=10 {
            flat.push(i);      // prompt
            flat.push(i + 10); // gen 1
            flat.push(i + 20); // gen 2
        }
        state.output_tokens_flat = flat;

        let results = state.collect_results(&reqs);
        assert_eq!(results.len(), 10);
        for (i, r) in results.iter().enumerate() {
            let seq_i = (i + 1) as u32;
            assert_eq!(r.request_id, seq_i as u64);
            assert_eq!(r.output_tokens, vec![seq_i + 10, seq_i + 20]);
        }
    }

    // ── Positions flat buffer: token_offset accumulation across many sequences ──

    #[test]
    fn positions_flat_accumulate_across_four_seqs() {
        let reqs = vec![
            make_req(1, vec![1], 1),         // 1 token, offset 0
            make_req(2, vec![2, 3], 1),      // 2 tokens, offset 1
            make_req(3, vec![4, 5, 6], 1),   // 3 tokens, offset 3
            make_req(4, vec![7, 8, 9, 10], 1), // 4 tokens, offset 6
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.positions_flat, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    // ── Empty prompt with max_new > 0: edge case ──

    #[test]
    fn empty_prompt_output_allocation_is_max_new() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![],
            max_new_tokens: 7,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.output_tokens_flat.len(), 7);
        assert_eq!(state.prompt_lens, vec![0]);
    }

    #[test]
    fn empty_prompt_max_decode_steps_correct() {
        let reqs = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![],
                max_new_tokens: 10,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            GenerateRequest {
                request_id: 2u64,
                prompt_tokens: vec![1],
                max_new_tokens: 3,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.max_decode_steps(), 10);
    }

    // ── Sampling params: large top_k value ──

    #[test]
    fn sampling_params_large_top_k_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.top_k = u32::MAX as usize;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[1], u32::MAX);
    }

    // ── Sampling params: eos_token_id edge values ──

    #[test]
    fn sampling_params_eos_token_id_max_u32() {
        let mut req = make_req(1, vec![1], 1);
        req.eos_token_id = u32::MAX;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[3], u32::MAX);
    }

    // ── BatchExecutor with non-null callback_table_ptr ──

    #[test]
    fn generate_request_non_null_callback_table_ptr() {
        let dummy_data: u8 = 99;
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: &dummy_data as *const u8,
        };
        assert!(req.callback_table_ptr.is_null() == false);
    }

    // ── build_from_prep: sampling params from request override prep ──

    #[test]
    fn build_from_prep_sampling_top_k_from_request() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 1;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 1;
        prep.sampling_params_packed = vec![0u32; 4]; // all zeros

        let mut req = make_req(1, vec![1], 1);
        req.top_k = 42;
        let state = BatchInferenceState::build_from_prep(&prep, &[req]);
        assert_eq!(state.sampling_params[1], 42);
    }

    #[test]
    fn build_from_prep_sampling_top_p_from_request() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 1;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 1;
        prep.sampling_params_packed = vec![0u32; 4];

        let mut req = make_req(1, vec![1], 1);
        req.top_p = 0.77;
        let state = BatchInferenceState::build_from_prep(&prep, &[req]);
        assert_eq!(state.sampling_params[2], 0.77f32.to_bits());
    }

    // ── BatchPrepData default active_flags ──

    #[test]
    fn batch_prep_data_default_all_active() {
        let prep = BatchPrepData::new(5);
        assert_eq!(prep.active_flags, vec![1u32; 5]);
        assert_eq!(prep.prompt_lens, vec![0u32; 5]);
        assert_eq!(prep.kv_lens, vec![0u32; 5]);
        assert_eq!(prep.gen_counts, vec![0u32; 5]);
        assert_eq!(prep.sampling_params_packed.len(), 20); // 5 * 4
    }

    // ── BatchContext num_seqs field ──

    #[test]
    fn batch_context_num_seqs_in_header() {
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;

        // num_seqs at offset 0
        let ns = u32::from_le_bytes(ctx.data[0..4].try_into().unwrap());
        assert_eq!(ns, 2);
    }

    // ── Build from requests: max_new_tokens=0 for one seq still allocates output slot ──

    #[test]
    fn output_allocation_with_zero_max_new_tokens() {
        let reqs = vec![
            make_req(1, vec![1, 2], 0), // prompt=2, max_new=0 → 2 slots
            make_req(2, vec![3], 3),     // prompt=1, max_new=3 → 4 slots
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.output_tokens_flat.len(), 6); // 2 + 4
    }

    // ── collect_results: seq with max_new=0 and another with tokens ──

    #[test]
    fn collect_results_mixed_zero_and_nonzero_max_new() {
        let reqs = vec![
            make_req_with_eos(1, vec![1, 2], 0, 99), // prompt=2, max_new=0
            make_req_with_eos(2, vec![3], 3, 99),     // prompt=1, max_new=3
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: 2 slots [1, 2]; seq 1: 4 slots [3, 10, 20, 30]
        state.output_tokens_flat = vec![1, 2, 3, 10, 20, 30];
        let results = state.collect_results(&reqs);
        assert!(results[0].output_tokens.is_empty()); // max_new=0 → nothing to extract
        assert_eq!(results[1].output_tokens, vec![10, 20, 30]);
    }

    // ── BatchInferenceState max_m with empty prompts and multiple seqs ──

    #[test]
    fn max_m_with_all_empty_prompts() {
        let reqs = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![],
                max_new_tokens: 1,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            GenerateRequest {
                request_id: 2u64,
                prompt_tokens: vec![],
                max_new_tokens: 1,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // total_prefill=0, num_seqs=2 → max_m = max(0, 2) = 2
        assert_eq!(state.max_m, 2);
    }

    // ── batch_decode with zero last_sampled_token produces zero in input ──

    #[test]
    fn batch_decode_zero_last_sampled_token_in_decode_input() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // last_sampled_token defaults to 0

        let result = batch_decode(&mut state, |_, decode_input, _, _| {
            assert_eq!(decode_input[0], 0);
            Ok::<usize, std::convert::Infallible>(1)
        });
        assert_eq!(result.unwrap(), 1);
    }

    // ── GenerateRequest with session_id set propagates into state ──

    #[test]
    fn generate_request_session_id_some_does_not_crash() {
        let mut req = make_req(1, vec![1, 2], 3);
        req.session_id = Some(42);
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.num_active, 1);
    }

    // ── BatchExecutor: prefill_batch returns Ok on success ──

    #[test]
    fn batch_executor_prefill_returns_ok_unit() {
        let reqs = vec![make_req(1, vec![1, 2, 3], 2)];
        let mut exec = BatchExecutor::new(&reqs);
        let result: Result<(), std::convert::Infallible> = exec.prefill_batch(|_| Ok(3));
        assert!(result.is_ok());
    }

    // ── Seq mapping with empty prompt in one sequence ──

    #[test]
    fn seq_mapping_with_empty_prompt_seq() {
        let reqs = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![1, 2],
                max_new_tokens: 1,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            GenerateRequest {
                request_id: 2u64,
                prompt_tokens: vec![],  // empty
                max_new_tokens: 1,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            GenerateRequest {
                request_id: 3u64,
                prompt_tokens: vec![3],
                max_new_tokens: 1,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: 2 tokens → [0, 0]; seq 1: 0 tokens → []; seq 2: 1 token → [2]
        assert_eq!(state.batch_ctx.seq_mapping, vec![0u32, 0, 2]);
    }

    // ── input_ids_flat preserves exact token values ──

    #[test]
    fn input_ids_flat_preserves_large_token_ids() {
        let reqs = vec![
            make_req(1, vec![u32::MAX, 0, u32::MAX / 2], 1),
            make_req(2, vec![1], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.input_ids_flat, vec![u32::MAX, 0, u32::MAX / 2, 1]);
    }

    // ── output_tokens_flat slice bounds in collect_results ──

    #[test]
    fn collect_results_output_flat_exactly_matches_expected() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 2, 99),
            make_req_with_eos(2, vec![2, 3], 1, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: prompt=1 + max_new=2 = 3 slots
        // seq 1: prompt=2 + max_new=1 = 3 slots
        // Total = 6
        state.output_tokens_flat = vec![1, 10, 20, 2, 3, 30];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20]);
        assert_eq!(results[1].output_tokens, vec![30]);
    }

    // ── build_from_prep with page_table_offsets ──

    #[test]
    fn build_from_prep_propagates_page_table_fields() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.page_table_offsets = vec![10, 20];
        prep.page_table_lens = vec![3, 5];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;

        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;

        let pt_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 20;
        let pt_len0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 24;
        let pt_off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + 20;
        let pt_len1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + 24;

        let off0 = u32::from_le_bytes(ctx.data[pt_off0..pt_off0 + 4].try_into().unwrap());
        let len0 = u32::from_le_bytes(ctx.data[pt_len0..pt_len0 + 4].try_into().unwrap());
        let off1 = u32::from_le_bytes(ctx.data[pt_off1..pt_off1 + 4].try_into().unwrap());
        let len1 = u32::from_le_bytes(ctx.data[pt_len1..pt_len1 + 4].try_into().unwrap());

        assert_eq!(off0, 10);
        assert_eq!(len0, 3);
        assert_eq!(off1, 20);
        assert_eq!(len1, 5);
    }

    // ── build_from_prep with multimodal fields ──

    #[test]
    fn build_from_prep_propagates_multimodal_fields() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 1;
        prep.active_flags[0] = 1;
        prep.fused_hidden_offsets[0] = 100;
        prep.num_mm_tokens[0] = 5;
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 1;

        let reqs = vec![make_req(1, vec![1], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;

        let fh_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 28; // SEQ_FUSED_HIDDEN_OFFSET
        let mm_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 32; // SEQ_NUM_MM_TOKENS

        let fh = u32::from_le_bytes(ctx.data[fh_off..fh_off + 4].try_into().unwrap());
        let mm = u32::from_le_bytes(ctx.data[mm_off..mm_off + 4].try_into().unwrap());
        assert_eq!(fh, 100);
        assert_eq!(mm, 5);
    }

    // ── 50 additional tests (target: 188+) ──

    #[test]
    fn generate_request_clone_independent_after_mutation() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1, 2, 3],
            max_new_tokens: 5,
            temperature: 0.7,
            top_k: 10,
            top_p: 0.9,
            session_id: None,
            eos_token_id: 2,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let mut cloned = req.clone();
        cloned.prompt_tokens.push(99);
        assert_eq!(req.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(cloned.prompt_tokens, vec![1, 2, 3, 99]);
    }

    #[test]
    fn generate_result_clone_independent_after_mutation() {
        let result = GenerateResult {
            request_id: 42u64,
            output_tokens: vec![10, 20, 30],
            finished: true,
            error: None,
        };
        let mut cloned = result.clone();
        cloned.output_tokens.push(40);
        assert_eq!(result.output_tokens, vec![10, 20, 30]);
        assert_eq!(cloned.output_tokens, vec![10, 20, 30, 40]);
    }

    #[test]
    fn batch_prep_data_new_zero_seqs() {
        let prep = BatchPrepData::new(0);
        assert!(prep.prompt_lens.is_empty());
        assert!(prep.kv_lens.is_empty());
        assert!(prep.sampling_params_packed.is_empty());
        assert_eq!(prep.max_decode_steps, 0);
        assert_eq!(prep.total_prefill_tokens, 0);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_single_seq() {
        let mut prep = BatchPrepData::new(1);
        prep.set_sampling_params(0, 0.5, 42, 0.9, 3);
        assert_eq!(prep.sampling_params_packed[0], 0.5f32.to_bits());
        assert_eq!(prep.sampling_params_packed[1], 42);
        assert_eq!(prep.sampling_params_packed[2], 0.9f32.to_bits());
        assert_eq!(prep.sampling_params_packed[3], 3);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_out_of_bounds_no_panic() {
        let mut prep = BatchPrepData::new(1);
        prep.set_sampling_params(5, 1.0, 0, 1.0, 0);
        assert_eq!(prep.sampling_params_packed[0], 0);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_multi_seq() {
        let mut prep = BatchPrepData::new(3);
        prep.set_sampling_params(0, 0.1, 10, 0.5, 1);
        prep.set_sampling_params(2, 0.9, 50, 0.95, 5);
        assert_eq!(prep.sampling_params_packed[0], 0.1f32.to_bits());
        assert_eq!(prep.sampling_params_packed[1], 10);
        assert_eq!(prep.sampling_params_packed[3], 1);
        assert_eq!(prep.sampling_params_packed[8], 0.9f32.to_bits());
        assert_eq!(prep.sampling_params_packed[9], 50);
        assert_eq!(prep.sampling_params_packed[11], 5);
        assert_eq!(prep.sampling_params_packed[4], 0);
    }

    #[test]
    fn batch_prep_data_default_rope_pos_offsets_zero() {
        let prep = BatchPrepData::new(4);
        assert_eq!(prep.rope_pos_offsets, vec![0u32; 4]);
    }

    #[test]
    fn batch_prep_data_default_session_positions_zero() {
        let prep = BatchPrepData::new(3);
        assert_eq!(prep.session_positions, vec![0u32; 3]);
    }

    #[test]
    fn batch_prep_data_default_page_table_fields_zero() {
        let prep = BatchPrepData::new(2);
        assert_eq!(prep.page_table_offsets, vec![0u32; 2]);
        assert_eq!(prep.page_table_lens, vec![0u32; 2]);
    }

    #[test]
    fn batch_prep_data_default_multimodal_fields_zero() {
        let prep = BatchPrepData::new(2);
        assert_eq!(prep.fused_hidden_offsets, vec![0u32; 2]);
        assert_eq!(prep.num_mm_tokens, vec![0u32; 2]);
    }

    #[test]
    fn batch_prep_data_default_last_sampled_tokens_zero() {
        let prep = BatchPrepData::new(5);
        assert_eq!(prep.last_sampled_tokens, vec![0u32; 5]);
    }

    #[test]
    fn batch_prep_data_clone_independent() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens[0] = 10;
        prep.kv_lens[0] = 5;
        let cloned = prep.clone();
        prep.prompt_lens[0] = 99;
        assert_eq!(cloned.prompt_lens[0], 10);
        assert_eq!(cloned.kv_lens[0], 5);
    }

    #[test]
    fn build_from_prep_input_ids_concatenated_from_requests() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![2, 3];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 5;
        let reqs = vec![
            make_req(1, vec![10, 20], 1),
            make_req(2, vec![30, 40, 50], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.input_ids_flat, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn build_from_prep_max_m_computation() {
        let mut prep = BatchPrepData::new(3);
        prep.prompt_lens = vec![2, 3, 1];
        prep.max_new_tokens = vec![1, 1, 1];
        prep.active_flags = vec![1, 1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 6;
        let reqs = vec![
            make_req(1, vec![1, 2], 1),
            make_req(2, vec![3, 4, 5], 1),
            make_req(3, vec![6], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.max_m, 6);
    }

    #[test]
    fn build_from_prep_sampling_eos_overwrites_prep_garbage() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;
        prep.sampling_params_packed = vec![0xFF_u32; 8];
        let reqs = vec![
            make_req_with_eos(1, vec![1], 1, 42),
            make_req_with_eos(2, vec![2], 1, 77),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.sampling_params[3], 42);
        assert_eq!(state.sampling_params[7], 77);
    }

    #[test]
    fn seq_mapping_all_single_token_prompts_five_seqs() {
        let reqs: Vec<GenerateRequest> = (0..5)
            .map(|i| make_req(i, vec![i as u32], 1))
            .collect();
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.batch_ctx.seq_mapping, vec![0u32, 1, 2, 3, 4]);
    }

    #[test]
    fn collect_results_output_truncated_multi_seq() {
        let reqs = vec![
            make_req_with_eos(1, vec![1, 2], 3, 99),
            make_req_with_eos(2, vec![3], 2, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![1, 2, 10, 20];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20]);
        assert!(results[1].output_tokens.is_empty());
    }

    #[test]
    fn batch_executor_prefill_called_twice_same_state() {
        let reqs = vec![make_req(1, vec![1, 2], 3)];
        let mut exec = BatchExecutor::new(&reqs);
        let r1: Result<(), std::convert::Infallible> = exec.prefill_batch(|_| Ok(2));
        assert!(r1.is_ok());
        let r2: Result<(), std::convert::Infallible> = exec.prefill_batch(|_| Ok(2));
        assert!(r2.is_ok());
    }

    #[test]
    fn batch_executor_decode_step_returns_active_count() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        let off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        exec.batch_ctx_mut().data[off1..off1 + 4].copy_from_slice(&0u32.to_le_bytes());
        let result = exec.decode_step(|_| Ok::<usize, std::convert::Infallible>(2));
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn batch_decode_returns_execute_closure_result() {
        let reqs = vec![make_req(1, vec![1], 5)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        let result = batch_decode(&mut state, |_, _, _, _| Ok::<usize, std::convert::Infallible>(42));
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn build_from_requests_page_table_flat_always_empty() {
        let reqs = vec![make_req(1, vec![1, 2, 3], 5), make_req(2, vec![4, 5], 10)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert!(state.page_table_flat.is_empty());
    }

    #[test]
    fn build_from_prep_page_table_flat_always_empty() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 1;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 1;
        let reqs = vec![make_req(1, vec![1], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert!(state.page_table_flat.is_empty());
    }

    #[test]
    fn execute_prefill_receives_exact_input_ids() {
        let reqs = vec![make_req(1, vec![100, 200], 1), make_req(2, vec![300], 1)];
        let results = execute_prefill(&reqs, |_ctx, inputs, _positions, total| {
            assert_eq!(inputs, &[100, 200, 300]);
            assert_eq!(total, 3);
            Ok::<usize, std::convert::Infallible>(3)
        }).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn execute_batch_receives_correct_total_prefill() {
        let reqs = vec![make_req(1, vec![1, 2, 3, 4, 5], 2)];
        let _ = execute_batch(&reqs, |_ctx, _inputs, _positions, total| {
            assert_eq!(total, 5);
            Ok::<usize, std::convert::Infallible>(5)
        }).unwrap();
    }

    #[test]
    fn collect_results_finished_always_true_error_always_none() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 3, 99),
            make_req_with_eos(2, vec![2], 3, 99),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let results = state.collect_results(&reqs);
        for r in &results {
            assert!(r.finished);
            assert!(r.error.is_none());
        }
    }

    #[test]
    fn build_from_requests_num_active_twenty_seqs() {
        let reqs: Vec<GenerateRequest> = (1..=20)
            .map(|i| make_req(i, vec![i as u32], 1))
            .collect();
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.num_active, 20);
    }

    #[test]
    fn build_from_prep_num_active_equals_prep_len() {
        let mut prep = BatchPrepData::new(7);
        prep.active_flags = vec![1, 1, 1, 1, 1, 1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 7;
        let reqs: Vec<GenerateRequest> = (1..=7)
            .map(|i| make_req(i, vec![i as u32], 1))
            .collect();
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.num_active, 7);
    }

    #[test]
    fn positions_flat_sequential_for_equal_length_prompts() {
        let reqs = vec![
            make_req(1, vec![10, 11], 1),
            make_req(2, vec![20, 21], 1),
            make_req(3, vec![30, 31], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.positions_flat, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn max_decode_steps_some_zero_max_new_tokens() {
        let reqs = vec![
            make_req(1, vec![1], 0),
            make_req(2, vec![2], 10),
            make_req(3, vec![3], 0),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.max_decode_steps(), 10);
    }

    #[test]
    fn batch_executor_batch_ctx_num_seqs_three() {
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1), make_req(3, vec![3], 1)];
        let exec = BatchExecutor::new(&reqs);
        assert_eq!(exec.batch_ctx().num_seqs, 3);
    }

    #[test]
    fn batch_executor_decode_step_all_inactive_zero() {
        let reqs = vec![make_req(1, vec![1], 3), make_req(2, vec![2], 3)];
        let mut exec = BatchExecutor::new(&reqs);
        for i in 0..2 {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            exec.batch_ctx_mut().data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        }
        let result: Result<usize, std::convert::Infallible> = exec.decode_step(|_| Ok(0));
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn output_tokens_flat_zero_initialized_large_batch() {
        let reqs: Vec<GenerateRequest> = (1..=8)
            .map(|i| make_req(i, vec![i as u32; 5], 3))
            .collect();
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.output_tokens_flat.len(), 64);
        assert!(state.output_tokens_flat.iter().all(|&t| t == 0));
    }

    #[test]
    fn batch_decode_positions_match_seq_positions() {
        let reqs = vec![make_req(1, vec![1], 3), make_req(2, vec![2], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        for i in 0..2 {
            let pos_off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
            let val = (10 + i * 5) as u32;
            state.batch_ctx.data[pos_off..pos_off + 4].copy_from_slice(&val.to_le_bytes());
        }
        let _ = batch_decode(&mut state, |_, _, decode_positions, _| {
            assert_eq!(decode_positions[0], 10);
            assert_eq!(decode_positions[1], 15);
            Ok::<usize, std::convert::Infallible>(2)
        });
    }

    #[test]
    fn sampling_params_exact_bit_representation() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.5,
            top_k: 100,
            top_p: 0.75,
            session_id: None,
            eos_token_id: 50,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params.len(), 4);
        assert_eq!(f32::from_bits(state.sampling_params[0]), 1.5);
        assert_eq!(state.sampling_params[1], 100);
        assert_eq!(f32::from_bits(state.sampling_params[2]), 0.75);
        assert_eq!(state.sampling_params[3], 50);
    }

    #[test]
    fn collect_results_exact_fill_no_early_stop() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 4, 99),
            make_req_with_eos(2, vec![2], 3, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![1, 11, 12, 13, 14, 2, 21, 22, 23];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![11, 12, 13, 14]);
        assert_eq!(results[1].output_tokens, vec![21, 22, 23]);
    }

    #[test]
    fn generate_request_id_zero() {
        let req = GenerateRequest {
            request_id: 0u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.request_id, 0);
    }

    #[test]
    fn generate_request_id_max_u64() {
        let req = GenerateRequest {
            request_id: u64::MAX,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.request_id, u64::MAX);
    }

    #[test]
    fn generate_result_preserves_request_id_in_collect() {
        let reqs = vec![make_req(100, vec![1], 2), make_req(200, vec![2], 2)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![1, 10, 20, 2, 30, 40];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].request_id, 100);
        assert_eq!(results[1].request_id, 200);
    }

    #[test]
    fn build_from_prep_hook_ptr_from_first_request() {
        let dummy: u8 = 42;
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;
        let mut req0 = make_req(1, vec![1], 1);
        req0.hook_ctx_ptr = &dummy as *const u8;
        let req1 = make_req(2, vec![2], 1);
        let state = BatchInferenceState::build_from_prep(&prep, &[req0, req1]);
        let hook_bytes = &state.batch_ctx.data[64..72];
        let hook_val = u64::from_le_bytes(hook_bytes.try_into().unwrap());
        assert_ne!(hook_val, 0);
    }

    #[test]
    fn build_from_requests_hook_ptr_null_empty_requests() {
        let state = BatchInferenceState::build_from_requests(&[]);
        let hook_bytes = &state.batch_ctx.data[64..72];
        let hook_val = u64::from_le_bytes(hook_bytes.try_into().unwrap());
        assert_eq!(hook_val, 0);
    }

    #[test]
    fn build_from_prep_session_positions_propagated() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.session_positions = vec![10, 20];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;
        let sp_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 16;
        let sp_off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + 16;
        let sp0 = u32::from_le_bytes(ctx.data[sp_off0..sp_off0 + 4].try_into().unwrap());
        let sp1 = u32::from_le_bytes(ctx.data[sp_off1..sp_off1 + 4].try_into().unwrap());
        assert_eq!(sp0, 10);
        assert_eq!(sp1, 20);
    }

    #[test]
    fn batch_decode_single_active_among_three() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        for &i in &[0usize, 2] {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            state.batch_ctx.data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        }
        let tok_off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        state.batch_ctx.data[tok_off..tok_off + 4].copy_from_slice(&55u32.to_le_bytes());
        let result = batch_decode(&mut state, |_, decode_input, _, num_active| {
            assert_eq!(num_active, 1);
            assert_eq!(decode_input.len(), 1);
            assert_eq!(decode_input[0], 55);
            Ok::<usize, std::convert::Infallible>(1)
        });
        assert_eq!(result.unwrap(), 1);
        assert_eq!(state.num_active, 1);
    }

    #[test]
    fn batch_executor_batch_ctx_mut_set_seq_kv_len() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut exec = BatchExecutor::new(&reqs);
        exec.batch_ctx_mut().set_seq_kv_len(0, 42);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_KV_LEN;
        let val = u32::from_le_bytes(exec.batch_ctx().data[off..off + 4].try_into().unwrap());
        assert_eq!(val, 42);
    }

    #[test]
    fn sampling_stride_allows_three_seqs() {
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1), make_req(3, vec![3], 1)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.sampling_params.len(), 3 * SAMPLING_STRIDE_U32);
    }

    #[test]
    fn collect_results_large_token_ids() {
        let req = make_req_with_eos(1, vec![u32::MAX], 3, 99);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![u32::MAX, u32::MAX - 1, u32::MAX - 2, u32::MAX - 3];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![u32::MAX - 1, u32::MAX - 2, u32::MAX - 3]);
    }

    #[test]
    fn batch_prep_data_max_decode_steps_default_zero() {
        let prep = BatchPrepData::new(3);
        assert_eq!(prep.max_decode_steps, 0);
    }

    #[test]
    fn batch_prep_data_total_prefill_tokens_default_zero() {
        let prep = BatchPrepData::new(3);
        assert_eq!(prep.total_prefill_tokens, 0);
    }

    #[test]
    fn build_from_prep_rope_pos_offsets_propagated() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.rope_pos_offsets = vec![100, 200];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;
        let rp_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ROPE_POS_OFFSET;
        let rp_off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_ROPE_POS_OFFSET;
        let rp0 = u32::from_le_bytes(ctx.data[rp_off0..rp_off0 + 4].try_into().unwrap());
        let rp1 = u32::from_le_bytes(ctx.data[rp_off1..rp_off1 + 4].try_into().unwrap());
        assert_eq!(rp0, 100);
        assert_eq!(rp1, 200);
    }

    #[test]
    fn batch_ctx_total_prefill_tokens_single_seq() {
        let reqs = vec![make_req(1, vec![1, 2, 3, 4, 5], 2)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let tpf = u32::from_le_bytes(state.batch_ctx.data[8..12].try_into().unwrap());
        assert_eq!(tpf, 5);
    }

    #[test]
    fn batch_decode_error_propagates() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        let result: Result<usize, &str> = batch_decode(&mut state, |_, _, _, _| Err("decode boom"));
        assert_eq!(result.unwrap_err(), "decode boom");
    }

    #[test]
    fn build_from_requests_many_single_token_seqs() {
        let reqs: Vec<GenerateRequest> = (1..=50)
            .map(|i| make_req(i, vec![i as u32], 2))
            .collect();
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.num_active, 50);
        assert_eq!(state.input_ids_flat.len(), 50);
        assert_eq!(state.prompt_lens, vec![1usize; 50]);
        assert_eq!(state.max_m, 50);
    }

    #[test]
    fn collect_results_varied_prompt_and_max_new() {
        let reqs = vec![
            make_req_with_eos(1, vec![1, 2, 3], 2, 99),
            make_req_with_eos(2, vec![4], 4, 99),
            make_req_with_eos(3, vec![5, 6], 1, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![
            1, 2, 3, 10, 20,
            4, 30, 40, 50, 60,
            5, 6, 70,
        ];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20]);
        assert_eq!(results[1].output_tokens, vec![30, 40, 50, 60]);
        assert_eq!(results[2].output_tokens, vec![70]);
    }

    #[test]
    fn build_from_prep_uses_prep_prompt_lens_not_request() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![2, 2];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 4;
        let reqs = vec![make_req(1, vec![10, 20], 1), make_req(2, vec![30, 40], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.prompt_lens, vec![2, 2]);
    }

    #[test]
    fn sampling_params_subnormal_temperature_preserved() {
        let subnormal = f32::from_bits(1u32);
        let mut req = make_req(1, vec![1], 1);
        req.temperature = subnormal;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[0], 1u32);
        assert_eq!(f32::from_bits(state.sampling_params[0]), subnormal);
    }

    // ── 50 additional tests ──

    #[test]
    fn batch_context_byte_size_one_seq() {
        let ctx = BatchContext::new(1);
        assert_eq!(ctx.byte_size(), 96 + 1 * 64);
    }

    #[test]
    fn batch_context_byte_size_four_seqs() {
        let ctx = BatchContext::new(4);
        assert_eq!(ctx.byte_size(), 96 + 4 * 64);
    }

    #[test]
    fn batch_context_new_initializes_max_batch_size() {
        let ctx = BatchContext::new(3);
        assert_eq!(ctx.max_batch_size, 3);
        assert!(!ctx.has_v2_extension);
    }

    #[test]
    fn batch_context_new_seq_mapping_initially_empty() {
        let ctx = BatchContext::new(5);
        assert!(ctx.seq_mapping.is_empty());
    }

    #[test]
    fn generate_request_eos_token_id_one() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 1,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.eos_token_id, 1);
    }

    #[test]
    fn collect_results_eos_one_stops_generation() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![10],
            max_new_tokens: 5,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 1,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // [prompt=10, gen=5, 1, 8, 9, 10] → EOS=1 stops at token 1
        state.output_tokens_flat = vec![10, 5, 1, 8, 9, 10];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![5]);
    }

    #[test]
    fn generate_request_top_k_one() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[1], 1);
    }

    #[test]
    fn generate_request_top_p_subnormal_preserved() {
        let subnormal_top_p = f32::from_bits(1u32);
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: subnormal_top_p,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[2], 1u32);
        assert_eq!(f32::from_bits(state.sampling_params[2]), subnormal_top_p);
    }

    #[test]
    fn generate_result_error_empty_string() {
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![],
            finished: false,
            error: Some(String::new()),
        };
        assert!(result.error.as_deref() == Some(""));
    }

    #[test]
    fn generate_result_error_multi_line() {
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![],
            finished: false,
            error: Some("line1\nline2\nline3".to_string()),
        };
        assert!(result.error.as_ref().unwrap().contains('\n'));
        assert_eq!(result.error.as_ref().unwrap().lines().count(), 3);
    }

    #[test]
    fn collect_results_single_token_generation() {
        let req = make_req_with_eos(1, vec![1], 1, 99);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![1, 42];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![42]);
    }

    #[test]
    fn collect_results_eos_is_first_gen_token_produces_empty() {
        let req = make_req_with_eos(1, vec![1, 2, 3], 4, 50);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![1, 2, 3, 50, 10, 20, 30];
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    #[test]
    fn batch_inference_state_num_active_field_public() {
        let reqs = vec![make_req(1, vec![1], 2), make_req(2, vec![2], 2)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.num_active, 2);
        state.num_active = 1;
        assert_eq!(state.num_active, 1);
    }

    #[test]
    fn batch_executor_decode_step_single_active_among_three() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        // Deactivate seqs 0 and 2
        for &i in &[0usize, 2] {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            exec.batch_ctx_mut().data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        }
        let result = exec.decode_step(|ctx| {
            assert_eq!(ctx.num_seqs, 3);
            Ok::<usize, std::convert::Infallible>(1)
        });
        assert_eq!(result.unwrap(), 1);
        assert_eq!(exec.state.num_active, 1);
    }

    #[test]
    fn batch_decode_active_seq_position_read_correctly() {
        let reqs = vec![make_req(1, vec![1, 2, 3], 5)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        let pos_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        state.batch_ctx.data[pos_off..pos_off + 4].copy_from_slice(&123u32.to_le_bytes());
        let _ = batch_decode(&mut state, |_, _, decode_positions, _| {
            assert_eq!(decode_positions[0], 123);
            Ok::<usize, std::convert::Infallible>(1)
        });
    }

    #[test]
    fn batch_executor_ctx_mut_set_active_flag() {
        let reqs = vec![make_req(1, vec![1], 2)];
        let mut exec = BatchExecutor::new(&reqs);
        exec.batch_ctx_mut().set_seq_active_flag(0, 0);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        let flag = u32::from_le_bytes(exec.batch_ctx().data[off..off + 4].try_into().unwrap());
        assert_eq!(flag, 0);
    }

    #[test]
    fn batch_executor_ctx_mut_set_gen_count() {
        let reqs = vec![make_req(1, vec![1], 5)];
        let mut exec = BatchExecutor::new(&reqs);
        exec.batch_ctx_mut().set_seq_gen_count(0, 3);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_GEN_COUNT;
        let gc = u32::from_le_bytes(exec.batch_ctx().data[off..off + 4].try_into().unwrap());
        assert_eq!(gc, 3);
    }

    #[test]
    fn batch_executor_ctx_mut_set_last_sampled_token() {
        let reqs = vec![make_req(1, vec![1], 5)];
        let mut exec = BatchExecutor::new(&reqs);
        exec.batch_ctx_mut().set_seq_last_sampled_token(0, 777);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        let tok = u32::from_le_bytes(exec.batch_ctx().data[off..off + 4].try_into().unwrap());
        assert_eq!(tok, 777);
    }

    #[test]
    fn batch_executor_ctx_mut_set_rope_pos_offset() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut exec = BatchExecutor::new(&reqs);
        exec.batch_ctx_mut().set_seq_rope_pos_offset(0, 256);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ROPE_POS_OFFSET;
        let val = u32::from_le_bytes(exec.batch_ctx().data[off..off + 4].try_into().unwrap());
        assert_eq!(val, 256);
    }

    #[test]
    fn build_from_prep_seq_positions_propagated() {
        let mut prep = BatchPrepData::new(3);
        prep.prompt_lens = vec![1, 1, 1];
        prep.max_new_tokens = vec![1, 1, 1];
        prep.active_flags = vec![1, 1, 1];
        prep.seq_positions = vec![10, 20, 30];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 3;
        let reqs = vec![
            make_req(1, vec![1], 1),
            make_req(2, vec![2], 1),
            make_req(3, vec![3], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;
        for i in 0..3 {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
            let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(val, (10 + i * 10) as u32, "seq {} position mismatch", i);
        }
    }

    #[test]
    fn build_from_prep_all_inactive_flags_propagated() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![0, 0]; // both inactive
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;
        for i in 0..2 {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            let flag = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(flag, 0, "seq {} should be inactive", i);
        }
    }

    #[test]
    fn build_from_requests_max_new_tokens_large_value() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 10000,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.max_decode_steps(), 10000);
        assert_eq!(state.output_tokens_flat.len(), 10001); // 1 prompt + 10000 gen
    }

    #[test]
    fn collect_results_two_seqs_one_truncated_output() {
        let reqs = vec![
            make_req_with_eos(1, vec![1, 2], 3, 99),
            make_req_with_eos(2, vec![3], 2, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // Only provide enough for seq 0: 5 slots. Seq 1 has 0 slots available.
        state.output_tokens_flat = vec![1, 2, 10, 20, 30];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
        assert!(results[1].output_tokens.is_empty());
    }

    #[test]
    fn batch_executor_decode_step_two_active_one_inactive() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        // Deactivate seq 1
        let off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        exec.batch_ctx_mut().data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        let result = exec.decode_step(|_| Ok::<usize, std::convert::Infallible>(2));
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn batch_decode_with_zero_active_returns_early() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        state.batch_ctx.data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        let result = batch_decode(&mut state, |_, _, _, _| {
            panic!("should not be called");
            #[allow(unreachable_code)]
            Ok::<usize, std::convert::Infallible>(0)
        });
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn generate_request_session_id_none_default() {
        let req = make_req(1, vec![1], 1);
        assert!(req.session_id.is_none());
    }

    #[test]
    fn generate_request_session_id_is_copy_semantic() {
        let id: crate::scheduler::SessionId = 42;
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: Some(id),
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.session_id, Some(42));
        // SessionId is still accessible after move into req
        assert_eq!(id, 42);
    }

    #[test]
    fn collect_results_prompt_tokens_not_in_output() {
        let req = make_req_with_eos(1, vec![100, 200], 3, 99);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![100, 200, 10, 20, 30];
        let results = state.collect_results(&[req]);
        // Prompt tokens 100, 200 should NOT appear in output
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
        assert!(!results[0].output_tokens.contains(&100));
        assert!(!results[0].output_tokens.contains(&200));
    }

    #[test]
    fn batch_context_as_ptr_non_null() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert!(!state.batch_ctx.as_ptr().is_null());
    }

    #[test]
    fn batch_context_data_all_zeros_initially() {
        let ctx = BatchContext::new(2);
        // The header should be all zeros except seq_meta_base_ptr at offset 88
        // Bytes 0..88 and 96.. should be zero
        for i in 0..88 {
            if i >= 88 && i < 96 {
                continue; // skip seq_meta_base_ptr field
            }
            assert_eq!(ctx.data[i], 0, "byte {} should be zero", i);
        }
    }

    #[test]
    fn build_from_prep_callback_ptr_from_first_request() {
        let dummy: u8 = 55;
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 1;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 1;
        let mut req = make_req(1, vec![1], 1);
        req.callback_table_ptr = &dummy as *const u8;
        let state = BatchInferenceState::build_from_prep(&prep, &[req]);
        let cb_bytes = &state.batch_ctx.data[72..80];
        let cb_val = u64::from_le_bytes(cb_bytes.try_into().unwrap());
        assert_ne!(cb_val, 0);
    }

    #[test]
    fn batch_decode_two_active_different_positions_and_tokens() {
        let reqs = vec![make_req(1, vec![1], 5), make_req(2, vec![2], 5)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);

        // Set different positions and tokens for each seq
        let pos_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        let pos_off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        let tok_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        let tok_off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;

        state.batch_ctx.data[pos_off0..pos_off0 + 4].copy_from_slice(&100u32.to_le_bytes());
        state.batch_ctx.data[pos_off1..pos_off1 + 4].copy_from_slice(&200u32.to_le_bytes());
        state.batch_ctx.data[tok_off0..tok_off0 + 4].copy_from_slice(&11u32.to_le_bytes());
        state.batch_ctx.data[tok_off1..tok_off1 + 4].copy_from_slice(&22u32.to_le_bytes());

        let _ = batch_decode(&mut state, |_, decode_input, decode_positions, num_active| {
            assert_eq!(num_active, 2);
            assert_eq!(decode_input, vec![11, 22]);
            assert_eq!(decode_positions, vec![100, 200]);
            Ok::<usize, std::convert::Infallible>(2)
        });
    }

    #[test]
    fn execute_batch_returns_correct_request_ids() {
        let reqs = vec![
            make_req(10, vec![1], 2),
            make_req(20, vec![2], 2),
        ];
        let results = execute_batch(&reqs, |_, _, _, _| Ok::<usize, std::convert::Infallible>(2)).unwrap();
        assert_eq!(results[0].request_id, 10u64);
        assert_eq!(results[1].request_id, 20u64);
    }

    #[test]
    fn execute_prefill_returns_correct_request_count() {
        let reqs = vec![
            make_req(1, vec![1], 1),
            make_req(2, vec![2], 1),
            make_req(3, vec![3], 1),
        ];
        let results = execute_prefill(&reqs, |_, _, _, _| Ok::<usize, std::convert::Infallible>(3)).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn batch_context_seq_meta_base_ptr_set() {
        let ctx = BatchContext::new(4);
        let ptr_bytes = &ctx.data[88..96];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        // seq_meta_base_ptr should point to offset 96 (= BATCH_CTX_HEADER_SIZE)
        let expected = ctx.data.as_ptr() as u64 + 96;
        assert_eq!(ptr_val, expected);
    }

    #[test]
    fn sampling_params_negative_temperature_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.temperature = -0.5;
        let state = BatchInferenceState::build_from_requests(&[req]);
        let bits = state.sampling_params[0];
        let recovered = f32::from_bits(bits);
        assert!(recovered.is_sign_negative());
        assert_eq!(recovered, -0.5);
    }

    #[test]
    fn sampling_params_negative_top_p_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.top_p = -0.3;
        let state = BatchInferenceState::build_from_requests(&[req]);
        let bits = state.sampling_params[2];
        let recovered = f32::from_bits(bits);
        assert!(recovered.is_sign_negative());
    }

    #[test]
    fn batch_prep_data_set_sampling_params_first_of_three() {
        let mut prep = BatchPrepData::new(3);
        prep.set_sampling_params(0, 0.1, 5, 0.8, 3);
        // Verify only first seq written
        assert_eq!(prep.sampling_params_packed[0], 0.1f32.to_bits());
        assert_eq!(prep.sampling_params_packed[1], 5);
        assert_eq!(prep.sampling_params_packed[2], 0.8f32.to_bits());
        assert_eq!(prep.sampling_params_packed[3], 3);
        // Second seq should still be zero
        assert_eq!(prep.sampling_params_packed[4], 0);
        assert_eq!(prep.sampling_params_packed[5], 0);
        assert_eq!(prep.sampling_params_packed[6], 0);
        assert_eq!(prep.sampling_params_packed[7], 0);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_overwrite() {
        let mut prep = BatchPrepData::new(1);
        prep.set_sampling_params(0, 0.5, 10, 0.9, 2);
        assert_eq!(prep.sampling_params_packed[0], 0.5f32.to_bits());
        // Overwrite with new values
        prep.set_sampling_params(0, 1.5, 20, 0.5, 7);
        assert_eq!(prep.sampling_params_packed[0], 1.5f32.to_bits());
        assert_eq!(prep.sampling_params_packed[1], 20);
        assert_eq!(prep.sampling_params_packed[2], 0.5f32.to_bits());
        assert_eq!(prep.sampling_params_packed[3], 7);
    }

    #[test]
    fn collect_results_three_seqs_varied_eos_positions() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 4, 10), // eos=10
            make_req_with_eos(2, vec![2], 4, 20), // eos=20
            make_req_with_eos(3, vec![3], 4, 99), // eos=99, not in output
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: [1, 5, 10, x, x] → stops at 10 → [5]
        // seq 1: [2, 7, 8, 20, x] → stops at 20 → [7, 8]
        // seq 2: [3, 11, 12, 13, 14] → no eos → [11, 12, 13, 14]
        state.output_tokens_flat = vec![
            1, 5, 10, 0, 0,
            2, 7, 8, 20, 0,
            3, 11, 12, 13, 14,
        ];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![5]);
        assert_eq!(results[1].output_tokens, vec![7, 8]);
        assert_eq!(results[2].output_tokens, vec![11, 12, 13, 14]);
    }

    #[test]
    fn batch_inference_state_output_tokens_flat_mutable() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        assert!(state.output_tokens_flat.iter().all(|&t| t == 0));
        state.output_tokens_flat[1] = 42;
        assert_eq!(state.output_tokens_flat[1], 42);
    }

    #[test]
    fn batch_executor_new_preserves_requests_len() {
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1)];
        let exec = BatchExecutor::new(&reqs);
        let results = exec.collect_results();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn build_from_prep_max_decode_steps_from_prep() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 10;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 10;
        prep.total_prefill_tokens = 1;
        let reqs = vec![make_req(1, vec![1], 10)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.max_decode_steps(), 10);
    }

    #[test]
    fn build_from_requests_prompt_lens_empty_request() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.prompt_lens, vec![0]);
    }

    #[test]
    fn batch_executor_prefill_then_collect() {
        let reqs = vec![make_req_with_eos(1, vec![1, 2], 3, 99)];
        let mut exec = BatchExecutor::new(&reqs);
        let _: Result<(), std::convert::Infallible> = exec.prefill_batch(|_| Ok(2));
        exec.state.output_tokens_flat = vec![1, 2, 10, 20, 30];
        let results = exec.collect_results();
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn batch_executor_decode_then_collect() {
        let reqs = vec![make_req_with_eos(1, vec![1], 3, 99)];
        let mut exec = BatchExecutor::new(&reqs);
        let _: Result<usize, std::convert::Infallible> = exec.decode_step(|_| Ok(1));
        exec.state.output_tokens_flat = vec![1, 10, 20, 30];
        let results = exec.collect_results();
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn batch_prep_data_default_max_new_tokens_zero() {
        let prep = BatchPrepData::new(3);
        assert_eq!(prep.max_new_tokens, vec![0u32; 3]);
    }

    #[test]
    fn batch_prep_data_default_seq_positions_zero() {
        let prep = BatchPrepData::new(4);
        assert_eq!(prep.seq_positions, vec![0u32; 4]);
    }

    #[test]
    fn batch_prep_data_sampling_params_len_is_four_times_seqs() {
        for n in &[0usize, 1, 3, 10] {
            let prep = BatchPrepData::new(*n);
            assert_eq!(prep.sampling_params_packed.len(), n * 4, "for {} seqs", n);
        }
    }

    #[test]
    fn positions_flat_single_token_three_seqs() {
        let reqs = vec![
            make_req(1, vec![10], 1),
            make_req(2, vec![20], 1),
            make_req(3, vec![30], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // Each seq has 1 token, offsets accumulate: 0, 1, 2
        assert_eq!(state.positions_flat, vec![0, 1, 2]);
    }

    #[test]
    fn seq_mapping_single_seq_long_prompt() {
        let reqs = vec![make_req(1, vec![1, 2, 3, 4, 5, 6, 7, 8], 1)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.batch_ctx.seq_mapping, vec![0u32; 8]);
    }

    #[test]
    fn collect_results_output_offset_with_uneven_prompts() {
        let reqs = vec![
            make_req_with_eos(1, vec![1, 2, 3, 4, 5], 2, 99), // 7 slots, offset 0
            make_req_with_eos(2, vec![6], 3, 99),               // 4 slots, offset 7
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![
            1, 2, 3, 4, 5, 10, 20,
            6, 30, 40, 50,
        ];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20]);
        assert_eq!(results[1].output_tokens, vec![30, 40, 50]);
    }

    #[test]
    fn batch_executor_decode_step_inactive_first_active_second() {
        let reqs = vec![make_req(1, vec![1], 3), make_req(2, vec![2], 3)];
        let mut exec = BatchExecutor::new(&reqs);
        // Inactivate seq 0, keep seq 1 active
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        exec.batch_ctx_mut().data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        // Set last sampled token for seq 1
        let tok_off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        exec.batch_ctx_mut().data[tok_off..tok_off + 4].copy_from_slice(&88u32.to_le_bytes());
        let result = exec.decode_step(|_| Ok::<usize, std::convert::Infallible>(1));
        assert_eq!(result.unwrap(), 1);
        assert_eq!(exec.state.num_active, 1);
    }

    #[test]
    fn generate_request_with_both_non_null_pointers() {
        let hook_data: u8 = 10;
        let cb_data: u8 = 20;
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: &hook_data as *const u8,
            callback_table_ptr: &cb_data as *const u8,
        };
        assert!(!req.hook_ctx_ptr.is_null());
        assert!(!req.callback_table_ptr.is_null());
    }

    #[test]
    fn build_from_requests_positions_start_at_zero_for_each_new_seq() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 1), // offset 0 → [0, 1, 2]
            make_req(2, vec![4, 5], 1),     // offset 3 → [3, 4]
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // Positions are absolute across all tokens, not per-seq relative
        assert_eq!(state.positions_flat[0], 0);
        assert_eq!(state.positions_flat[3], 3); // seq 1 starts at offset 3
    }

    // ── 55 additional tests ──

    #[test]
    fn generate_request_prompt_tokens_not_aliased_after_build() {
        let mut req = make_req(1, vec![1, 2, 3], 2);
        let state = BatchInferenceState::build_from_requests(&[req.clone()]);
        assert_eq!(state.input_ids_flat, vec![1, 2, 3]);
        req.prompt_tokens.push(99);
        assert_eq!(state.input_ids_flat, vec![1, 2, 3]);
    }

    #[test]
    fn generate_result_finished_true_with_empty_output() {
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![],
            finished: true,
            error: None,
        };
        assert!(result.finished);
        assert!(result.output_tokens.is_empty());
    }

    #[test]
    fn build_from_requests_sampling_params_stride_four_seqs() {
        let reqs: Vec<GenerateRequest> = (1..=4)
            .map(|i| {
                let mut r = make_req(i, vec![i as u32], 1);
                r.temperature = i as f32 * 0.1;
                r.top_k = (i * 10) as usize;
                r.eos_token_id = (i * 100) as u32;
                r
            })
            .collect();
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.sampling_params.len(), 16);
        for i in 0..4 {
            let base = i * 4;
            assert_eq!(
                state.sampling_params[base],
                ((i + 1) as f32 * 0.1).to_bits(),
                "seq {} temperature bits mismatch",
                i
            );
            assert_eq!(state.sampling_params[base + 1], ((i + 1) * 10) as u32);
            assert_eq!(state.sampling_params[base + 3], (i + 1) as u32 * 100);
        }
    }

    #[test]
    fn batch_executor_decode_step_updates_num_active_to_one() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        for &i in &[0usize, 2] {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            exec.batch_ctx_mut().data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        }
        let result = exec.decode_step(|_| Ok::<usize, std::convert::Infallible>(1));
        assert_eq!(result.unwrap(), 1);
        assert_eq!(exec.state.num_active, 1);
    }

    #[test]
    fn collect_results_eos_mid_generation_stops_correctly() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 6, 42),
            make_req_with_eos(2, vec![2], 4, 42),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![
            1, 10, 20, 42, 30, 40, 50,
            2, 5, 6, 42, 7,
        ];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20]);
        assert_eq!(results[1].output_tokens, vec![5, 6]);
    }

    #[test]
    fn build_from_prep_page_table_flat_still_empty() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.page_table_offsets = vec![5, 10];
        prep.page_table_lens = vec![3, 7];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert!(state.page_table_flat.is_empty());
    }

    #[test]
    fn batch_prep_data_fields_len_matches_num_seqs() {
        let n = 6;
        let prep = BatchPrepData::new(n);
        assert_eq!(prep.prompt_lens.len(), n);
        assert_eq!(prep.kv_lens.len(), n);
        assert_eq!(prep.session_positions.len(), n);
        assert_eq!(prep.rope_pos_offsets.len(), n);
        assert_eq!(prep.max_new_tokens.len(), n);
        assert_eq!(prep.page_table_offsets.len(), n);
        assert_eq!(prep.page_table_lens.len(), n);
        assert_eq!(prep.fused_hidden_offsets.len(), n);
        assert_eq!(prep.num_mm_tokens.len(), n);
        assert_eq!(prep.active_flags.len(), n);
        assert_eq!(prep.seq_positions.len(), n);
        assert_eq!(prep.gen_counts.len(), n);
        assert_eq!(prep.last_sampled_tokens.len(), n);
    }

    #[test]
    fn batch_context_with_v2_extension_has_extension() {
        let ctx = BatchContext::with_v2_extension(4, 2);
        assert!(ctx.has_v2_extension);
        assert_eq!(ctx.max_batch_size, 4);
        assert_eq!(ctx.num_seqs, 2);
    }

    #[test]
    fn batch_context_v2_extension_byte_size_larger() {
        let ctx = BatchContext::with_v2_extension(2, 1);
        let normal_ctx = BatchContext::new(1);
        assert!(ctx.byte_size() > normal_ctx.byte_size());
    }

    #[test]
    fn build_from_requests_input_ids_preserves_order() {
        let reqs = vec![
            make_req(1, vec![100, 200, 300], 1),
            make_req(2, vec![400], 1),
            make_req(3, vec![500, 600], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.input_ids_flat, vec![100, 200, 300, 400, 500, 600]);
    }

    #[test]
    fn execute_batch_closure_receives_non_null_ctx_pointer() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let _ = execute_batch(&reqs, |ctx, _, _, _| {
            assert!(!std::ptr::from_ref(ctx).is_null());
            Ok::<usize, std::convert::Infallible>(1)
        }).unwrap();
    }

    #[test]
    fn collect_results_single_seq_all_eos_tokens() {
        let req = make_req_with_eos(1, vec![1], 4, 10);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![1, 10, 10, 10, 10];
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    #[test]
    fn batch_decode_five_active_seqs_input_order() {
        let reqs: Vec<GenerateRequest> = (1..=5)
            .map(|i| make_req(i, vec![i as u32], 3))
            .collect();
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        for i in 0..5 {
            let tok_off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
            state.batch_ctx.data[tok_off..tok_off + 4].copy_from_slice(&((i + 10) as u32).to_le_bytes());
        }
        let _ = batch_decode(&mut state, |_, decode_input, _, num_active| {
            assert_eq!(num_active, 5);
            assert_eq!(decode_input, vec![10u32, 11, 12, 13, 14]);
            Ok::<usize, std::convert::Infallible>(5)
        });
    }

    #[test]
    fn batch_executor_prefill_receives_correct_ctx() {
        let reqs = vec![
            make_req(1, vec![1, 2], 3),
            make_req(2, vec![3], 2),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        let result = exec.prefill_batch(|ctx| {
            assert_eq!(ctx.num_seqs, 2);
            Ok::<usize, std::convert::Infallible>(3)
        });
        assert!(result.is_ok());
    }

    #[test]
    fn build_from_prep_max_m_with_zero_prefill_tokens() {
        let mut prep = BatchPrepData::new(3);
        prep.prompt_lens = vec![0, 0, 0];
        prep.max_new_tokens = vec![1, 1, 1];
        prep.active_flags = vec![1, 1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 0;
        let reqs = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![],
                max_new_tokens: 1,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            make_req(2, vec![], 1),
            make_req(3, vec![], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.max_m, 3); // max(0, 3) = 3
    }

    #[test]
    fn batch_context_set_and_read_num_seqs() {
        let mut ctx = BatchContext::new(2);
        ctx.set_num_seqs(5);
        let ns = u32::from_le_bytes(ctx.data[0..4].try_into().unwrap());
        assert_eq!(ns, 5);
    }

    #[test]
    fn batch_context_set_and_read_max_decode_steps() {
        let mut ctx = BatchContext::new(1);
        ctx.set_max_decode_steps(42);
        let mds = u32::from_le_bytes(ctx.data[4..8].try_into().unwrap());
        assert_eq!(mds, 42);
    }

    #[test]
    fn batch_context_set_and_read_total_prefill_tokens() {
        let mut ctx = BatchContext::new(1);
        ctx.set_total_prefill_tokens(128);
        let tpf = u32::from_le_bytes(ctx.data[8..12].try_into().unwrap());
        assert_eq!(tpf, 128);
    }

    #[test]
    fn batch_context_seq_prompt_len_set_and_read() {
        let mut ctx = BatchContext::new(2);
        ctx.set_seq_prompt_len(0, 10);
        ctx.set_seq_prompt_len(1, 20);
        let off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 0;
        let off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + 0;
        let pl0 = u32::from_le_bytes(ctx.data[off0..off0 + 4].try_into().unwrap());
        let pl1 = u32::from_le_bytes(ctx.data[off1..off1 + 4].try_into().unwrap());
        assert_eq!(pl0, 10);
        assert_eq!(pl1, 20);
    }

    #[test]
    fn batch_context_set_seq_max_new_tokens() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_max_new_tokens(0, 77);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_MAX_NEW_TOKENS;
        let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
        assert_eq!(val, 77);
    }

    #[test]
    fn batch_context_set_seq_session_position() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_session_position(0, 999);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 16;
        let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
        assert_eq!(val, 999);
    }

    #[test]
    fn batch_context_set_seq_output_offset() {
        let mut ctx = BatchContext::new(2);
        ctx.set_seq_output_offset(0, 0);
        ctx.set_seq_output_offset(1, 50);
        let off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
        let off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
        let val0 = u32::from_le_bytes(ctx.data[off0..off0 + 4].try_into().unwrap());
        let val1 = u32::from_le_bytes(ctx.data[off1..off1 + 4].try_into().unwrap());
        assert_eq!(val0, 0);
        assert_eq!(val1, 50);
    }

    #[test]
    fn batch_context_set_seq_fused_hidden_offset() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_fused_hidden_offset(0, 2048);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 28;
        let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
        assert_eq!(val, 2048);
    }

    #[test]
    fn batch_context_set_seq_num_mm_tokens() {
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_num_mm_tokens(0, 16);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 32;
        let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
        assert_eq!(val, 16);
    }

    #[test]
    fn batch_context_per_seq_padding_is_zero() {
        let ctx = BatchContext::new(1);
        let pad_start = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 56;
        let pad_end = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 64;
        for i in pad_start..pad_end {
            assert_eq!(ctx.data[i], 0, "padding byte {} should be zero", i);
        }
    }

    #[test]
    fn build_from_requests_positions_continue_across_seqs() {
        let reqs = vec![
            make_req(1, vec![1, 2], 1),
            make_req(2, vec![3, 4, 5], 1),
            make_req(3, vec![6], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.positions_flat.len(), 6);
        assert_eq!(state.positions_flat[0], 0);
        assert_eq!(state.positions_flat[2], 2);
        assert_eq!(state.positions_flat[3], 3);
        assert_eq!(state.positions_flat[5], 5);
    }

    #[test]
    fn collect_results_with_eos_equal_to_generated_token() {
        let req = make_req_with_eos(1, vec![1], 5, 20);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![1, 10, 15, 20, 25, 30];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![10, 15]);
    }

    #[test]
    fn batch_executor_state_accessible_after_build() {
        let reqs = vec![make_req(1, vec![1, 2, 3], 4)];
        let exec = BatchExecutor::new(&reqs);
        assert_eq!(exec.state.prompt_lens, vec![3]);
        assert_eq!(exec.state.max_new_tokens_per_seq, vec![4]);
        assert_eq!(exec.state.input_ids_flat, vec![1, 2, 3]);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_second_of_three() {
        let mut prep = BatchPrepData::new(3);
        prep.set_sampling_params(1, 0.3, 30, 0.7, 7);
        // Verify second seq only written
        assert_eq!(prep.sampling_params_packed[0], 0);
        assert_eq!(prep.sampling_params_packed[4], 0.3f32.to_bits());
        assert_eq!(prep.sampling_params_packed[5], 30);
        assert_eq!(prep.sampling_params_packed[6], 0.7f32.to_bits());
        assert_eq!(prep.sampling_params_packed[7], 7);
        assert_eq!(prep.sampling_params_packed[8], 0);
    }

    #[test]
    fn build_from_prep_output_tokens_flat_size_with_zero_max_new() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![3, 2];
        prep.max_new_tokens = vec![0, 0];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 0;
        prep.total_prefill_tokens = 5;
        let reqs = vec![make_req(1, vec![1, 2, 3], 0), make_req(2, vec![4, 5], 0)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.output_tokens_flat.len(), 5); // 3 + 2
    }

    #[test]
    fn generate_request_temperature_max_f32_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.temperature = f32::MAX;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[0], f32::MAX.to_bits());
    }

    #[test]
    fn generate_request_temperature_min_positive_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.temperature = f32::MIN_POSITIVE;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[0], f32::MIN_POSITIVE.to_bits());
    }

    #[test]
    fn generate_request_top_p_one_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.top_p = 1.0;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[2], 1.0f32.to_bits());
    }

    #[test]
    fn generate_request_top_p_zero_preserved() {
        let mut req = make_req(1, vec![1], 1);
        req.top_p = 0.0;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[2], 0.0f32.to_bits());
    }

    #[test]
    fn collect_results_all_eos_tokens_no_valid_output() {
        let req = make_req_with_eos(1, vec![1, 2], 3, 5);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![1, 2, 5, 5, 5];
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    #[test]
    fn execute_prefill_single_req_passes_total_one() {
        let reqs = vec![make_req(1, vec![42], 1)];
        let _ = execute_prefill(&reqs, |_ctx, inputs, _positions, total| {
            assert_eq!(inputs, &[42]);
            assert_eq!(total, 1);
            Ok::<usize, std::convert::Infallible>(1)
        }).unwrap();
    }

    #[test]
    fn batch_decode_incorrect_active_count_propagates_error() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        let result: Result<usize, &str> = batch_decode(&mut state, |_, _, _, _| Err("JIT error"));
        assert_eq!(result.unwrap_err(), "JIT error");
    }

    #[test]
    fn batch_executor_new_with_two_requests_state_dimensions() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3, 4, 5], 10),
            make_req(2, vec![6], 3),
        ];
        let exec = BatchExecutor::new(&reqs);
        assert_eq!(exec.state.max_m, 6);
        assert_eq!(exec.state.num_active, 2);
        assert_eq!(exec.state.input_ids_flat.len(), 6);
        assert_eq!(exec.state.sampling_params.len(), 8);
    }

    #[test]
    fn build_from_requests_max_new_tokens_per_seq_matches_order() {
        let reqs = vec![
            make_req(1, vec![1], 7),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 15),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.max_new_tokens_per_seq, vec![7, 3, 15]);
        assert_eq!(state.max_decode_steps(), 15);
    }

    #[test]
    fn collect_results_output_tokens_within_range() {
        let req = make_req_with_eos(1, vec![1], 100, 999);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![1u32; 101];
        // First gen token is 1 (not EOS=999, not 0) — but all tokens are 1
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens.len(), 100);
        assert!(results[0].output_tokens.iter().all(|&t| t == 1));
    }

    #[test]
    fn batch_context_header_size_is_96() {
        assert_eq!(BATCH_CTX_HEADER_SIZE, 96);
    }

    #[test]
    fn batch_context_seq_meta_stride_is_64() {
        assert_eq!(SEQ_META_STRIDE, 64);
    }

    #[test]
    fn sampling_stride_constant_four() {
        assert_eq!(SAMPLING_STRIDE_U32, 4);
        assert_eq!(SAMPLING_STRIDE_U32 * 4, 16); // bytes per seq
    }

    #[test]
    fn batch_context_clone_produces_independent_data() {
        let mut ctx = BatchContext::new(2);
        ctx.set_num_seqs(5);
        let cloned = ctx.clone();
        let ns_orig = u32::from_le_bytes(ctx.data[0..4].try_into().unwrap());
        let ns_clone = u32::from_le_bytes(cloned.data[0..4].try_into().unwrap());
        assert_eq!(ns_orig, ns_clone);
        assert_eq!(ns_orig, 5);
    }

    #[test]
    fn build_from_requests_seq_mapping_len_equals_total_prefill() {
        let reqs = vec![
            make_req(1, vec![1, 2], 1),
            make_req(2, vec![3, 4, 5], 1),
            make_req(3, vec![6], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.batch_ctx.seq_mapping.len(), 6);
        assert_eq!(state.batch_ctx.seq_mapping, vec![0u32, 0, 1, 1, 1, 2]);
    }

    #[test]
    fn build_from_requests_positions_len_equals_total_prefill() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 1),
            make_req(2, vec![4, 5], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.positions_flat.len(), 5);
    }

    #[test]
    fn execute_batch_alias_same_as_execute_batch() {
        let reqs = vec![make_req(1, vec![1, 2], 2)];
        let r1 = execute_batch(&reqs, |_, _, _, _| Ok::<usize, std::convert::Infallible>(2)).unwrap();
        let r2 = batch_lifecycle(&reqs, |_, _, _, _| Ok::<usize, std::convert::Infallible>(2)).unwrap();
        assert_eq!(r1.len(), r2.len());
        assert_eq!(r1[0].request_id, r2[0].request_id);
    }

    #[test]
    fn batch_executor_decode_step_with_three_active() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        let result = exec.decode_step(|ctx| {
            assert_eq!(ctx.num_seqs, 3);
            Ok::<usize, std::convert::Infallible>(3)
        });
        assert_eq!(result.unwrap(), 3);
        assert_eq!(exec.state.num_active, 3);
    }

    #[test]
    fn batch_executor_state_output_tokens_flat_mutable_after_build() {
        let reqs = vec![make_req(1, vec![1], 2)];
        let mut exec = BatchExecutor::new(&reqs);
        exec.state.output_tokens_flat[0] = 99;
        assert_eq!(exec.state.output_tokens_flat[0], 99);
    }

    #[test]
    fn generate_request_large_prompt_tokens_preserved() {
        let large_prompt: Vec<u32> = (0..1000).map(|i| i as u32 * 2).collect();
        let req = make_req(1, large_prompt.clone(), 10);
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.input_ids_flat, large_prompt);
        assert_eq!(state.prompt_lens, vec![1000]);
    }

    #[test]
    fn collect_results_output_tokens_not_containing_prompt() {
        let reqs = vec![
            make_req_with_eos(1, vec![50, 60], 3, 99),
            make_req_with_eos(2, vec![70], 2, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![50, 60, 10, 20, 30, 70, 40, 50];
        let results = state.collect_results(&reqs);
        assert!(!results[0].output_tokens.contains(&50));
        assert!(!results[0].output_tokens.contains(&60));
        assert!(!results[1].output_tokens.contains(&70));
    }

    #[test]
    fn batch_prep_data_clone_preserves_all_fields() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens[0] = 5;
        prep.kv_lens[1] = 10;
        prep.max_decode_steps = 7;
        prep.total_prefill_tokens = 8;
        let cloned = prep.clone();
        assert_eq!(cloned.prompt_lens[0], 5);
        assert_eq!(cloned.kv_lens[1], 10);
        assert_eq!(cloned.max_decode_steps, 7);
        assert_eq!(cloned.total_prefill_tokens, 8);
    }

    #[test]
    fn build_from_prep_positions_flat_accumulate_three_seqs() {
        let mut prep = BatchPrepData::new(3);
        prep.prompt_lens = vec![2, 3, 1];
        prep.max_new_tokens = vec![1, 1, 1];
        prep.active_flags = vec![1, 1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 6;
        let reqs = vec![
            make_req(1, vec![1, 2], 1),
            make_req(2, vec![3, 4, 5], 1),
            make_req(3, vec![6], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.positions_flat, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn batch_context_data_len_matches_byte_size() {
        let ctx = BatchContext::new(3);
        assert_eq!(ctx.data.len(), ctx.byte_size());
    }

    #[test]
    fn batch_context_seq_meta_stride_times_seqs_plus_header() {
        let n = 5;
        let ctx = BatchContext::new(n);
        assert_eq!(ctx.byte_size(), BATCH_CTX_HEADER_SIZE + n * SEQ_META_STRIDE);
    }

    #[test]
    fn generate_request_debug_format_contains_field_names() {
        let req = GenerateRequest {
            request_id: 99u64,
            prompt_tokens: vec![1],
            max_new_tokens: 5,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            session_id: None,
            eos_token_id: 2,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let debug = format!("{:?}", req);
        assert!(debug.contains("request_id"));
        assert!(debug.contains("prompt_tokens"));
        assert!(debug.contains("temperature"));
    }

    #[test]
    fn generate_result_debug_format_contains_field_names() {
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![10, 20],
            finished: true,
            error: None,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("request_id"));
        assert!(debug.contains("output_tokens"));
        assert!(debug.contains("finished"));
    }

    #[test]
    fn batch_inference_state_fields_public_accessible() {
        let reqs = vec![make_req(1, vec![1, 2], 3)];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let _ = state.max_m;
        let _ = state.num_active;
        let _ = &state.batch_ctx;
        let _ = &state.input_ids_flat;
        let _ = &state.output_tokens_flat;
        let _ = &state.positions_flat;
        let _ = &state.page_table_flat;
        let _ = &state.sampling_params;
        let _ = &state.prompt_lens;
        let _ = &state.max_new_tokens_per_seq;
    }

    #[test]
    fn build_from_requests_output_offset_calculation_three_seqs() {
        let reqs = vec![
            make_req(1, vec![1], 2),       // 1+2=3 slots, offset 0
            make_req(2, vec![1, 2], 3),    // 2+3=5 slots, offset 3
            make_req(3, vec![1, 2, 3], 1), // 3+1=4 slots, offset 8
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        let ctx = &state.batch_ctx;
        for (i, expected) in [0u32, 3, 8].iter().enumerate() {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
            let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(val, *expected, "seq {} output_offset mismatch", i);
        }
    }

    #[test]
    fn build_from_prep_positions_from_requests_not_prep() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 3;
        prep.max_new_tokens[0] = 2;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 2;
        prep.total_prefill_tokens = 3;
        let reqs = vec![make_req(1, vec![10, 20, 30], 2)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        // positions come from token_offset accumulation, not from prep
        assert_eq!(state.positions_flat, vec![0, 1, 2]);
    }

    #[test]
    fn execute_prefill_results_have_correct_count() {
        let reqs: Vec<GenerateRequest> = (1..=7)
            .map(|i| make_req(i, vec![i as u32], 1))
            .collect();
        let results = execute_prefill(&reqs, |_, _, _, _| Ok::<usize, std::convert::Infallible>(7)).unwrap();
        assert_eq!(results.len(), 7);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.request_id, (i + 1) as u64);
        }
    }

    #[test]
    fn batch_executor_ctx_readonly_access() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let exec = BatchExecutor::new(&reqs);
        let ctx = exec.batch_ctx();
        assert_eq!(ctx.num_seqs, 1);
    }

    #[test]
    fn batch_context_set_page_table_flat_ptr() {
        let mut ctx = BatchContext::new(1);
        let dummy: u32 = 42;
        ctx.set_page_table_flat_ptr(&dummy as *const u32);
        let ptr_bytes = &ctx.data[40..48];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_ne!(ptr_val, 0);
    }

    #[test]
    fn batch_context_set_kv_pool_base() {
        let mut ctx = BatchContext::new(1);
        let dummy: u8 = 0;
        ctx.set_kv_pool_base(&dummy as *const u8);
        let ptr_bytes = &ctx.data[48..56];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_ne!(ptr_val, 0);
    }

    #[test]
    fn batch_context_set_sampling_params_ptr() {
        let mut ctx = BatchContext::new(1);
        let dummy: u32 = 0;
        ctx.set_sampling_params_ptr(&dummy as *const u32);
        let ptr_bytes = &ctx.data[56..64];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_ne!(ptr_val, 0);
    }

    #[test]
    fn batch_executor_prefill_batch_return_type_is_result_unit() {
        let reqs = vec![make_req(1, vec![1], 1)];
        let mut exec = BatchExecutor::new(&reqs);
        let result: Result<(), std::convert::Infallible> = exec.prefill_batch(|_| Ok(1));
        assert!(result.is_ok());
        // Verify it's () not some other type
        let inner = result.unwrap();
        assert_eq!(format!("{:?}", inner), "()");
    }

    // ── 40 additional coverage tests ──

    #[test]
    fn batch_context_v2_extension_zero_seqs() {
        let ctx = BatchContext::with_v2_extension(2, 0);
        assert_eq!(ctx.num_seqs, 0);
        assert!(ctx.has_v2_extension);
        assert_eq!(ctx.max_batch_size, 2);
    }

    #[test]
    fn batch_context_with_v2_extension_num_seqs_independent() {
        let ctx = BatchContext::with_v2_extension(8, 3);
        assert_eq!(ctx.max_batch_size, 8);
        assert_eq!(ctx.num_seqs, 3);
        assert!(ctx.has_v2_extension);
    }

    #[test]
    fn build_from_requests_positions_are_monotonically_increasing() {
        let reqs = vec![
            make_req(1, vec![10, 20, 30], 1),
            make_req(2, vec![40], 1),
            make_req(3, vec![50, 60], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        for window in state.positions_flat.windows(2) {
            assert!(
                window[0] < window[1],
                "positions must be monotonically increasing: {} >= {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn collect_results_token_id_u32_max() {
        let req = make_req_with_eos(1, vec![1], 3, 99);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![1, u32::MAX, u32::MAX - 1, u32::MAX - 2];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![u32::MAX, u32::MAX - 1, u32::MAX - 2]);
    }

    #[test]
    fn batch_decode_input_and_positions_count_equal() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        let _ = batch_decode(&mut state, |_, decode_input, decode_positions, num_active| {
            assert_eq!(decode_input.len(), num_active);
            assert_eq!(decode_positions.len(), num_active);
            Ok::<usize, std::convert::Infallible>(num_active)
        });
    }

    #[test]
    fn batch_executor_decode_step_three_consecutive() {
        let reqs = vec![make_req(1, vec![1], 5)];
        let mut exec = BatchExecutor::new(&reqs);
        for step in 0..3 {
            let result = exec.decode_step(|_| Ok::<usize, std::convert::Infallible>(1));
            assert_eq!(result.unwrap(), 1, "step {} should return 1 active", step);
        }
        assert_eq!(exec.state.num_active, 1);
    }

    #[test]
    fn build_from_prep_sampling_temp_overwrites_prep() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 1;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 1;
        prep.sampling_params_packed = vec![0xFF_u32; 4];
        let mut req = make_req(1, vec![1], 1);
        req.temperature = 0.42;
        let state = BatchInferenceState::build_from_prep(&prep, &[req]);
        assert_eq!(state.sampling_params[0], 0.42f32.to_bits());
    }

    #[test]
    fn generate_request_eos_zero_in_collect() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![10],
            max_new_tokens: 4,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // eos=0, first gen token is 5 → zero-sentinel not hit yet
        state.output_tokens_flat = vec![10, 5, 6, 0, 9];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![5, 6]);
    }

    #[test]
    fn collect_results_two_seqs_identical_prompts() {
        let reqs = vec![
            make_req_with_eos(1, vec![5, 5, 5], 2, 99),
            make_req_with_eos(2, vec![5, 5, 5], 2, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        state.output_tokens_flat = vec![5, 5, 5, 10, 20, 5, 5, 5, 30, 40];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20]);
        assert_eq!(results[1].output_tokens, vec![30, 40]);
    }

    #[test]
    fn batch_context_v2_extension_data_has_extra_space() {
        let normal = BatchContext::new(2);
        let extended = BatchContext::with_v2_extension(4, 2);
        assert!(extended.data.len() > normal.data.len());
    }

    #[test]
    fn build_from_requests_max_m_single_empty_prompt_dominates() {
        let reqs = vec![
            GenerateRequest {
                request_id: 1u64,
                prompt_tokens: vec![],
                max_new_tokens: 1,
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                session_id: None,
                eos_token_id: 0,
                hook_ctx_ptr: std::ptr::null(),
                callback_table_ptr: std::ptr::null(),
            },
            make_req(2, vec![1, 2, 3, 4, 5], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.max_m, 5); // max(total_prefill=5, num_seqs=2) = 5
    }

    #[test]
    fn batch_prep_data_active_flags_mutable() {
        let mut prep = BatchPrepData::new(3);
        assert_eq!(prep.active_flags, vec![1, 1, 1]);
        prep.active_flags[1] = 0;
        assert_eq!(prep.active_flags, vec![1, 0, 1]);
    }

    #[test]
    fn batch_executor_state_max_m_readonly_after_build() {
        let reqs = vec![make_req(1, vec![1, 2, 3], 5)];
        let exec = BatchExecutor::new(&reqs);
        let max_m_before = exec.state.max_m;
        assert_eq!(max_m_before, 3);
    }

    #[test]
    fn collect_results_output_all_same_nonzero_token() {
        let req = make_req_with_eos(1, vec![1], 5, 99);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        state.output_tokens_flat = vec![1, 42, 42, 42, 42, 42];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![42, 42, 42, 42, 42]);
    }

    #[test]
    fn generate_result_output_tokens_large_vec() {
        let large_output: Vec<u32> = (0..1000).collect();
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: large_output.clone(),
            finished: true,
            error: None,
        };
        assert_eq!(result.output_tokens.len(), 1000);
        assert_eq!(result.output_tokens[0], 0);
        assert_eq!(result.output_tokens[999], 999);
    }

    #[test]
    fn batch_context_set_multiple_seq_fields_independent() {
        let mut ctx = BatchContext::new(2);
        ctx.set_seq_prompt_len(0, 10);
        ctx.set_seq_kv_len(0, 5);
        ctx.set_seq_max_new_tokens(0, 3);
        ctx.set_seq_prompt_len(1, 20);
        ctx.set_seq_kv_len(1, 15);
        ctx.set_seq_max_new_tokens(1, 7);

        let pl0_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE;
        let kv0_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_KV_LEN;
        let mnt0_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_MAX_NEW_TOKENS;
        let pl1_off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE;
        let kv1_off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_KV_LEN;
        let mnt1_off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_MAX_NEW_TOKENS;

        assert_eq!(u32::from_le_bytes(ctx.data[pl0_off..pl0_off + 4].try_into().unwrap()), 10);
        assert_eq!(u32::from_le_bytes(ctx.data[kv0_off..kv0_off + 4].try_into().unwrap()), 5);
        assert_eq!(u32::from_le_bytes(ctx.data[mnt0_off..mnt0_off + 4].try_into().unwrap()), 3);
        assert_eq!(u32::from_le_bytes(ctx.data[pl1_off..pl1_off + 4].try_into().unwrap()), 20);
        assert_eq!(u32::from_le_bytes(ctx.data[kv1_off..kv1_off + 4].try_into().unwrap()), 15);
        assert_eq!(u32::from_le_bytes(ctx.data[mnt1_off..mnt1_off + 4].try_into().unwrap()), 7);
    }

    #[test]
    fn build_from_prep_output_offsets_from_prep_data() {
        let mut prep = BatchPrepData::new(3);
        prep.prompt_lens = vec![4, 1, 3];
        prep.max_new_tokens = vec![2, 5, 1];
        prep.active_flags = vec![1, 1, 1];
        prep.max_decode_steps = 5;
        prep.total_prefill_tokens = 8;
        let reqs = vec![
            make_req(1, vec![1, 2, 3, 4], 2),
            make_req(2, vec![5], 5),
            make_req(3, vec![6, 7, 8], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;
        // seq 0: 4+2=6 → offset 0
        // seq 1: 1+5=6 → offset 6
        // seq 2: 3+1=4 → offset 12
        let expected_offsets = [0u32, 6, 12];
        for (i, &expected) in expected_offsets.iter().enumerate() {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_OUTPUT_OFFSET;
            let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
            assert_eq!(val, expected, "seq {} output_offset mismatch", i);
        }
    }

    #[test]
    fn batch_executor_decode_step_error_does_not_modify_state() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut exec = BatchExecutor::new(&reqs);
        let num_active_before = exec.state.num_active;
        let result: Result<usize, &str> = exec.decode_step(|_| Err("fail"));
        assert!(result.is_err());
        assert_eq!(exec.state.num_active, num_active_before);
    }

    #[test]
    fn build_from_requests_sampling_params_eos_differs_per_seq() {
        let reqs = vec![
            make_req_with_eos(1, vec![1], 1, 10),
            make_req_with_eos(2, vec![2], 1, 20),
            make_req_with_eos(3, vec![3], 1, 30),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.sampling_params[3], 10);
        assert_eq!(state.sampling_params[7], 20);
        assert_eq!(state.sampling_params[11], 30);
    }

    #[test]
    fn collect_results_many_small_seqs_rapid_extraction() {
        let reqs: Vec<GenerateRequest> = (1..=30)
            .map(|i| make_req_with_eos(i, vec![i as u32], 1, 99))
            .collect();
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        let mut flat = Vec::with_capacity(60);
        for i in 1u32..=30 {
            flat.push(i);       // prompt
            flat.push(i + 100); // gen
        }
        state.output_tokens_flat = flat;
        let results = state.collect_results(&reqs);
        assert_eq!(results.len(), 30);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.output_tokens, vec![(i + 1) as u32 + 100]);
        }
    }

    #[test]
    fn generate_request_prompt_tokens_len_zero() {
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![],
            max_new_tokens: 5,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        assert_eq!(req.prompt_tokens.len(), 0);
    }

    #[test]
    fn batch_decode_single_seq_position_zero() {
        let reqs = vec![make_req(1, vec![1], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq_position defaults to 0
        let _ = batch_decode(&mut state, |_, _, decode_positions, _| {
            assert_eq!(decode_positions[0], 0);
            Ok::<usize, std::convert::Infallible>(1)
        });
    }

    #[test]
    fn batch_prep_data_max_new_tokens_mutable() {
        let mut prep = BatchPrepData::new(2);
        assert_eq!(prep.max_new_tokens, vec![0, 0]);
        prep.max_new_tokens[0] = 5;
        prep.max_new_tokens[1] = 10;
        assert_eq!(prep.max_new_tokens, vec![5, 10]);
    }

    #[test]
    fn batch_context_set_seq_page_table_offset_and_len() {
        let mut ctx = BatchContext::new(2);
        ctx.set_seq_page_table_offset(0, 100);
        ctx.set_seq_page_table_len(0, 7);
        ctx.set_seq_page_table_offset(1, 200);
        ctx.set_seq_page_table_len(1, 13);
        let pt_off0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 20;
        let pt_len0 = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + 24;
        let pt_off1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + 20;
        let pt_len1 = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + 24;
        assert_eq!(u32::from_le_bytes(ctx.data[pt_off0..pt_off0 + 4].try_into().unwrap()), 100);
        assert_eq!(u32::from_le_bytes(ctx.data[pt_len0..pt_len0 + 4].try_into().unwrap()), 7);
        assert_eq!(u32::from_le_bytes(ctx.data[pt_off1..pt_off1 + 4].try_into().unwrap()), 200);
        assert_eq!(u32::from_le_bytes(ctx.data[pt_len1..pt_len1 + 4].try_into().unwrap()), 13);
    }

    #[test]
    fn build_from_prep_sampling_top_p_overwrites_prep() {
        let mut prep = BatchPrepData::new(1);
        prep.prompt_lens[0] = 1;
        prep.max_new_tokens[0] = 1;
        prep.active_flags[0] = 1;
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 1;
        prep.sampling_params_packed = vec![0u32; 4];
        let mut req = make_req(1, vec![1], 1);
        req.top_p = 0.33;
        let state = BatchInferenceState::build_from_prep(&prep, &[req]);
        assert_eq!(state.sampling_params[2], 0.33f32.to_bits());
    }

    #[test]
    fn batch_executor_prefill_and_decode_lifecycle() {
        let reqs = vec![make_req_with_eos(1, vec![1, 2], 3, 99)];
        let mut exec = BatchExecutor::new(&reqs);
        // Prefill
        let r: Result<(), std::convert::Infallible> = exec.prefill_batch(|_| Ok(2));
        assert!(r.is_ok());
        // Decode step
        let r: Result<usize, std::convert::Infallible> = exec.decode_step(|_| Ok(1));
        assert_eq!(r.unwrap(), 1);
        // Collect
        exec.state.output_tokens_flat = vec![1, 2, 10, 20, 30];
        let results = exec.collect_results();
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn generate_request_temperature_smallest_normal() {
        let mut req = make_req(1, vec![1], 1);
        req.temperature = f32::MIN_POSITIVE;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[0], f32::MIN_POSITIVE.to_bits());
        let recovered = f32::from_bits(state.sampling_params[0]);
        assert!(recovered.is_normal());
        assert!(recovered > 0.0);
    }

    #[test]
    fn collect_results_eos_after_all_valid_tokens() {
        let req = make_req_with_eos(1, vec![1], 4, 50);
        let mut state = BatchInferenceState::build_from_requests(&[req.clone()]);
        // gen tokens: 10, 20, 30, 50 — EOS at last position
        state.output_tokens_flat = vec![1, 10, 20, 30, 50];
        let results = state.collect_results(&[req]);
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn build_from_requests_max_m_equals_max_of_prefill_and_count() {
        let reqs = vec![
            make_req(1, vec![1, 2], 1), // 2 tokens
            make_req(2, vec![3], 1),     // 1 token
            make_req(3, vec![4], 1),     // 1 token
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // total_prefill=4, num_seqs=3 → max_m = 4
        assert_eq!(state.max_m, 4);
    }

    #[test]
    fn batch_decode_num_active_updated_to_correct_count() {
        let reqs = vec![make_req(1, vec![1], 3), make_req(2, vec![2], 3)];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // Deactivate seq 0
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        state.batch_ctx.data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        let result = batch_decode(&mut state, |_, _, _, num_active| {
            assert_eq!(num_active, 1);
            Ok::<usize, std::convert::Infallible>(1)
        });
        assert_eq!(result.unwrap(), 1);
        assert_eq!(state.num_active, 1);
    }

    #[test]
    fn batch_context_set_num_seqs_zero() {
        let mut ctx = BatchContext::new(3);
        ctx.set_num_seqs(0);
        let ns = u32::from_le_bytes(ctx.data[0..4].try_into().unwrap());
        assert_eq!(ns, 0);
    }

    #[test]
    fn execute_batch_single_seq_dimensions() {
        let reqs = vec![make_req(1, vec![10, 20, 30], 2)];
        let _ = execute_batch(&reqs, |ctx, inputs, positions, total| {
            assert_eq!(ctx.num_seqs, 1);
            assert_eq!(inputs.len(), 3);
            assert_eq!(positions.len(), 3);
            assert_eq!(total, 3);
            Ok::<usize, std::convert::Infallible>(3)
        }).unwrap();
    }

    #[test]
    fn build_from_requests_positions_len_equals_input_ids_len() {
        let reqs = vec![
            make_req(1, vec![1, 2, 3], 1),
            make_req(2, vec![4], 1),
            make_req(3, vec![5, 6], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.positions_flat.len(), state.input_ids_flat.len());
    }

    #[test]
    fn collect_results_gen_start_eq_gen_end_empty() {
        // prompt_len equals total output → no gen slots
        let req = make_req_with_eos(1, vec![1, 2, 3], 0, 99);
        let state = BatchInferenceState::build_from_requests(&[req.clone()]);
        let results = state.collect_results(&[req]);
        assert!(results[0].output_tokens.is_empty());
    }

    #[test]
    fn batch_executor_new_empty_requests_num_active_zero() {
        let exec = BatchExecutor::new(&[]);
        assert_eq!(exec.state.num_active, 0);
        assert_eq!(exec.batch_ctx().num_seqs, 0);
    }

    #[test]
    fn batch_context_set_input_ids_flat_ptr_null() {
        let mut ctx = BatchContext::new(1);
        ctx.set_input_ids_flat_ptr(std::ptr::null());
        let ptr_bytes = &ctx.data[16..24];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_eq!(ptr_val, 0);
    }

    #[test]
    fn batch_context_set_output_tokens_flat_ptr_null() {
        let mut ctx = BatchContext::new(1);
        ctx.set_output_tokens_flat_ptr(std::ptr::null_mut());
        let ptr_bytes = &ctx.data[24..32];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_eq!(ptr_val, 0);
    }

    #[test]
    fn batch_context_set_positions_ptr_null() {
        let mut ctx = BatchContext::new(1);
        ctx.set_positions_ptr(std::ptr::null());
        let ptr_bytes = &ctx.data[32..40];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_eq!(ptr_val, 0);
    }

    #[test]
    fn batch_executor_decode_step_inactive_middle_seq() {
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
            make_req(4, vec![4], 3),
        ];
        let mut exec = BatchExecutor::new(&reqs);
        // Deactivate only seq 1
        let off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
        exec.batch_ctx_mut().data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        let result = exec.decode_step(|_| Ok::<usize, std::convert::Infallible>(3));
        assert_eq!(result.unwrap(), 3);
        assert_eq!(exec.state.num_active, 3);
    }

    #[test]
    fn build_from_prep_sampling_eos_from_request_not_prep() {
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;
        prep.sampling_params_packed = vec![0u32; 8];
        let reqs = vec![
            make_req_with_eos(1, vec![1], 1, 42),
            make_req_with_eos(2, vec![2], 1, 88),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        assert_eq!(state.sampling_params[3], 42);
        assert_eq!(state.sampling_params[7], 88);
    }

    #[test]
    fn batch_context_set_hook_ctx_ptr_explicit() {
        let mut ctx = BatchContext::new(1);
        ctx.set_hook_ctx_ptr(std::ptr::null());
        let ptr_bytes = &ctx.data[64..72];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_eq!(ptr_val, 0);
    }

    #[test]
    fn batch_context_set_callback_table_ptr_explicit() {
        let mut ctx = BatchContext::new(1);
        ctx.set_callback_table_ptr(std::ptr::null());
        let ptr_bytes = &ctx.data[72..80];
        let ptr_val = u64::from_le_bytes(ptr_bytes.try_into().unwrap());
        assert_eq!(ptr_val, 0);
    }

    #[test]
    fn generate_request_top_k_large_value_preserved_in_state() {
        let mut req = make_req(1, vec![1], 1);
        req.top_k = 999999;
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[1], 999999u32);
    }

    #[test]
    fn build_from_requests_positions_match_input_indices() {
        let reqs = vec![
            make_req(1, vec![10, 20], 1),
            make_req(2, vec![30, 40, 50], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        // Positions should be 0..total_prefill_tokens-1
        for (i, &pos) in state.positions_flat.iter().enumerate() {
            assert_eq!(pos, i as u32, "position at index {} should be {}", i, i);
        }
    }

    #[test]
    fn batch_prep_data_prompt_lens_mutable() {
        let mut prep = BatchPrepData::new(2);
        assert_eq!(prep.prompt_lens, vec![0, 0]);
        prep.prompt_lens[0] = 10;
        prep.prompt_lens[1] = 20;
        assert_eq!(prep.prompt_lens, vec![10, 20]);
    }


    #[test]
    fn generate_result_output_tokens_can_be_modified_after_clone() {
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![10, 20],
            finished: true,
            error: None,
        };
        let mut cloned = result.clone();
        cloned.output_tokens.clear();
        assert!(cloned.output_tokens.is_empty());
        assert_eq!(result.output_tokens, vec![10, 20]);
    }

    // ── 15 additional edge-case tests ──

    // @trace TEST-BEXEC-001 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn collect_results_empty_prompt_first_seq_second_has_tokens() {
        // Seq 0 has empty prompt, seq 1 has tokens. Verify output extraction
        // boundaries are correct when prompt_len=0 for the first sequence.
        let reqs = vec![
            make_req_with_eos(1, vec![], 3, 99),
            make_req_with_eos(2, vec![5, 6], 2, 99),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: 0+3=3 slots → [10, 20, 30]
        // seq 1: 2+2=4 slots → [5, 6, 40, 50]
        state.output_tokens_flat = vec![10, 20, 30, 5, 6, 40, 50];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10, 20, 30]);
        assert_eq!(results[1].output_tokens, vec![40, 50]);
    }

    // @trace TEST-BEXEC-002 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn build_from_requests_all_max_new_tokens_zero() {
        // All sequences have max_new_tokens=0: output_tokens_flat should only have
        // prompt slots and collect_results should produce empty outputs.
        let reqs = vec![
            make_req(1, vec![10, 20], 0),
            make_req(2, vec![30], 0),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.output_tokens_flat.len(), 3); // 2 + 1
        assert_eq!(state.max_decode_steps(), 0);
        let results = state.collect_results(&reqs);
        assert!(results[0].output_tokens.is_empty());
        assert!(results[1].output_tokens.is_empty());
    }

    // @trace TEST-BEXEC-003 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn generate_result_error_with_unicode_string() {
        // Verify error field correctly preserves unicode content.
        let result = GenerateResult {
            request_id: 1u64,
            output_tokens: vec![],
            finished: false,
            error: Some("错误: 模型加载失败 🚫".to_string()),
        };
        assert_eq!(result.error.as_deref(), Some("错误: 模型加载失败 🚫"));
        assert!(result.error.as_ref().unwrap().len() > 10);
    }

    // @trace TEST-BEXEC-004 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn build_from_prep_with_large_kv_lens_propagated() {
        // Verify build_from_prep propagates large kv_lens values correctly.
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![1, 1];
        prep.kv_lens = vec![4096, 8192];
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 2;
        let reqs = vec![make_req(1, vec![1], 1), make_req(2, vec![2], 1)];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        let ctx = &state.batch_ctx;
        let kv0_off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_KV_LEN;
        let kv1_off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_KV_LEN;
        assert_eq!(
            u32::from_le_bytes(ctx.data[kv0_off..kv0_off + 4].try_into().unwrap()),
            4096
        );
        assert_eq!(
            u32::from_le_bytes(ctx.data[kv1_off..kv1_off + 4].try_into().unwrap()),
            8192
        );
    }

    // @trace TEST-BEXEC-005 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn batch_executor_empty_requests_prefill_and_collect() {
        // BatchExecutor with empty request slice should produce zero results.
        let mut exec = BatchExecutor::new(&[]);
        let result: Result<(), std::convert::Infallible> = exec.prefill_batch(|_| Ok(0));
        assert!(result.is_ok());
        let results = exec.collect_results();
        assert!(results.is_empty());
    }

    // @trace TEST-BEXEC-006 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn collect_results_eos_in_middle_stops_correctly_multi_seq() {
        // EOS appears at different positions in each sequence's generation.
        let reqs = vec![
            make_req_with_eos(1, vec![1], 5, 15),
            make_req_with_eos(2, vec![2], 5, 25),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        // seq 0: [1, 10, 15, x, x, x] → stops at 15 → [10]
        // seq 1: [2, 20, 21, 22, 25, x] → stops at 25 → [20, 21, 22]
        state.output_tokens_flat = vec![1, 10, 15, 0, 0, 0, 2, 20, 21, 22, 25, 0];
        let results = state.collect_results(&reqs);
        assert_eq!(results[0].output_tokens, vec![10]);
        assert_eq!(results[1].output_tokens, vec![20, 21, 22]);
    }

    // @trace TEST-BEXEC-007 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn batch_context_num_seqs_zero_byte_size_header_only() {
        // BatchContext with 0 seqs should have byte_size equal to header only.
        let ctx = BatchContext::new(0);
        assert_eq!(ctx.num_seqs, 0);
        assert_eq!(ctx.byte_size(), BATCH_CTX_HEADER_SIZE);
    }

    // @trace TEST-BEXEC-008 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn sampling_params_all_zeros_temperature_and_top_p() {
        // Verify that temperature=0.0 and top_p=0.0 are stored as bit-exact zeros.
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 1,
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.sampling_params[0], 0u32); // 0.0 → bits = 0
        assert_eq!(state.sampling_params[2], 0u32); // 0.0 → bits = 0
    }

    // @trace TEST-BEXEC-009 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn build_from_prep_input_ids_from_requests_independent_of_prep_prompt_lens() {
        // build_from_prep concatenates input_ids from requests regardless of prep.prompt_lens.
        let mut prep = BatchPrepData::new(2);
        prep.prompt_lens = vec![5, 5]; // prep claims 5+5, but actual request prompts are different
        prep.max_new_tokens = vec![1, 1];
        prep.active_flags = vec![1, 1];
        prep.max_decode_steps = 1;
        prep.total_prefill_tokens = 10;
        let reqs = vec![
            make_req(1, vec![100, 200], 1),
            make_req(2, vec![300], 1),
        ];
        let state = BatchInferenceState::build_from_prep(&prep, &reqs);
        // input_ids comes from requests, not prep
        assert_eq!(state.input_ids_flat, vec![100, 200, 300]);
    }

    // @trace TEST-BEXEC-010 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn generate_request_max_new_tokens_large_value_preserved() {
        // Verify a large but non-overflowing max_new_tokens is preserved in state.
        let req = GenerateRequest {
            request_id: 1u64,
            prompt_tokens: vec![1],
            max_new_tokens: 100_000,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: 0,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        };
        let state = BatchInferenceState::build_from_requests(&[req]);
        assert_eq!(state.max_new_tokens_per_seq, vec![100_000]);
        assert_eq!(state.max_decode_steps(), 100_000);
        assert_eq!(state.output_tokens_flat.len(), 100_001); // 1 prompt + 100000 gen
    }

    // @trace TEST-BEXEC-011 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn batch_decode_preserves_active_count_with_two_inactive_one_active() {
        // Three sequences: seqs 0,2 inactive, seq 1 active.
        // num_active should be set to 1 after batch_decode.
        let reqs = vec![
            make_req(1, vec![1], 3),
            make_req(2, vec![2], 3),
            make_req(3, vec![3], 3),
        ];
        let mut state = BatchInferenceState::build_from_requests(&reqs);
        for &i in &[0usize, 2] {
            let off = BATCH_CTX_HEADER_SIZE + i * SEQ_META_STRIDE + SEQ_ACTIVE_FLAG;
            state.batch_ctx.data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        }
        let tok_off = BATCH_CTX_HEADER_SIZE + 1 * SEQ_META_STRIDE + SEQ_LAST_SAMPLED_TOKEN;
        state.batch_ctx.data[tok_off..tok_off + 4].copy_from_slice(&99u32.to_le_bytes());
        let result = batch_decode(&mut state, |_, decode_input, _, num_active| {
            assert_eq!(num_active, 1);
            assert_eq!(decode_input[0], 99);
            Ok::<usize, std::convert::Infallible>(1)
        });
        assert_eq!(result.unwrap(), 1);
        assert_eq!(state.num_active, 1);
    }

    // @trace TEST-BEXEC-012 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn generate_result_debug_clone_preserves_error_content() {
        // Clone a GenerateResult with error and verify the error string is independent.
        let result = GenerateResult {
            request_id: 42u64,
            output_tokens: vec![10],
            finished: false,
            error: Some("catastrophic failure".to_string()),
        };
        let cloned = result.clone();
        assert_eq!(cloned.error.as_deref(), Some("catastrophic failure"));
        assert_eq!(cloned.output_tokens, vec![10]);
        assert!(!cloned.finished);
        let debug = format!("{:?}", result);
        assert!(debug.contains("catastrophic"));
    }

    // @trace TEST-BEXEC-013 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn build_from_requests_prompt_lens_accumulate_correctly_with_mixed_lengths() {
        // Verify prompt_lens array reflects each request's actual prompt length.
        let reqs = vec![
            make_req(1, vec![1], 1),
            make_req(2, vec![2, 3, 4, 5, 6, 7, 8], 1),
            make_req(3, vec![], 1),
            make_req(4, vec![9], 1),
        ];
        let state = BatchInferenceState::build_from_requests(&reqs);
        assert_eq!(state.prompt_lens, vec![1, 7, 0, 1]);
        assert_eq!(state.input_ids_flat.len(), 9);
    }

    // @trace TEST-BEXEC-014 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn batch_context_set_and_get_seq_position_round_trip() {
        // Write a seq_position via the setter, read it back from raw bytes.
        let mut ctx = BatchContext::new(1);
        ctx.set_seq_position(0, 65535);
        let off = BATCH_CTX_HEADER_SIZE + 0 * SEQ_META_STRIDE + SEQ_SEQ_POSITION;
        let val = u32::from_le_bytes(ctx.data[off..off + 4].try_into().unwrap());
        assert_eq!(val, 65535);
    }

    // @trace TEST-BEXEC-015 [req:REQ-BCI-008] [level:unit]
    #[test]
    fn execute_batch_error_type_preserves_message() {
        // Verify execute_batch propagates the exact error message from the closure.
        let reqs = vec![make_req(1, vec![1], 1)];
        let result: Result<Vec<GenerateResult>, String> = execute_batch(&reqs, |_, _, _, _| {
            Err("kernel panic: stack overflow at layer 42".to_string())
        });
        let err = result.expect_err("should be error");
        assert_eq!(err, "kernel panic: stack overflow at layer 42");
    }

    // ── Helpers ──

    fn make_req(id: u64, tokens: Vec<u32>, max_new: usize) -> GenerateRequest {
        make_req_with_eos(id, tokens, max_new, 2)
    }

    fn make_req_with_eos(id: u64, tokens: Vec<u32>, max_new: usize, eos: u32) -> GenerateRequest {
        GenerateRequest {
            request_id: id,
            prompt_tokens: tokens,
            max_new_tokens: max_new,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            session_id: None,
            eos_token_id: eos,
            hook_ctx_ptr: std::ptr::null(),
            callback_table_ptr: std::ptr::null(),
        }
    }
}
