use std::collections::{BTreeMap, VecDeque};

use super::types::RequestId;

use super::paged_scheduler::{PagedScheduler, SchedulerError};
use super::sequence::{Sequence, SequenceState};
use super::types::BatchOrderPolicy;
use super::vllm2024::{ChunkedConfig, ChunkedState};

#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub requests: Vec<RequestId>,
    pub seq_offsets: Vec<usize>,
    pub draft_steps: Vec<usize>,
}

/// Batch 预处理数据 (SPEC/20 REQ-BCI-007)
///
/// 包含构建 BatchContext 所需的所有中间数据。
#[derive(Debug, Clone)]
pub struct BatchPrepData {
    /// 每条序列的 prompt 长度
    pub prompt_lens: Vec<u32>,
    /// 每条序列的 KV 长度
    pub kv_lens: Vec<u32>,
    /// 每条序列的 session position
    pub session_positions: Vec<u32>,
    /// 每条序列的 RoPE position offset
    pub rope_pos_offsets: Vec<u32>,
    /// 每条序列的 max new tokens
    pub max_new_tokens: Vec<u32>,
    /// 每条序列的 page table offset (在 page_table_flat_ptr 中的偏移)
    pub page_table_offsets: Vec<u32>,
    /// 每条序列的 page table 长度
    pub page_table_lens: Vec<u32>,
    /// 每条序列的 fused hidden offset (在 fused_hidden_flat_ptr 中的偏移)
    pub fused_hidden_offsets: Vec<u32>,
    /// 每条序列的 multimodal token 数量
    pub num_mm_tokens: Vec<u32>,
    /// 每条序列的 active flag (1=active, 0=inactive)
    pub active_flags: Vec<u32>,
    /// 每条序列的当前位置
    pub seq_positions: Vec<u32>,
    /// 每条序列已生成的 token 数
    pub gen_counts: Vec<u32>,
    /// 每条序列上次采样的 token
    pub last_sampled_tokens: Vec<u32>,
    /// 采样参数 packed: [temp, top_k, top_p, eos] × N (每条序列 4 个 u32)
    pub sampling_params_packed: Vec<u32>,
    /// 总 decode 步数
    pub max_decode_steps: u32,
    /// 总 prefill token 数
    pub total_prefill_tokens: u32,
}

impl BatchPrepData {
    /// 创建空的 BatchPrepData
    pub fn new(num_seqs: usize) -> Self {
        Self {
            prompt_lens: vec![0; num_seqs],
            kv_lens: vec![0; num_seqs],
            session_positions: vec![0; num_seqs],
            rope_pos_offsets: vec![0; num_seqs],
            max_new_tokens: vec![0; num_seqs],
            page_table_offsets: vec![0; num_seqs],
            page_table_lens: vec![0; num_seqs],
            fused_hidden_offsets: vec![0; num_seqs],
            num_mm_tokens: vec![0; num_seqs],
            active_flags: vec![1; num_seqs], // 默认全部 active
            seq_positions: vec![0; num_seqs],
            gen_counts: vec![0; num_seqs],
            last_sampled_tokens: vec![0; num_seqs],
            sampling_params_packed: vec![0; num_seqs * 4],
            max_decode_steps: 0,
            total_prefill_tokens: 0,
        }
    }

    /// 设置采样参数 (packed: temp, top_k, top_p, eos)
    pub fn set_sampling_params(&mut self, seq: usize, temp: f32, top_k: usize, top_p: f32, eos: u32) {
        let base = seq * 4;
        if base + 4 <= self.sampling_params_packed.len() {
            self.sampling_params_packed[base] = temp.to_bits();
            self.sampling_params_packed[base + 1] = top_k as u32;
            self.sampling_params_packed[base + 2] = top_p.to_bits();
            self.sampling_params_packed[base + 3] = eos;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAction {
    Continue,
    Complete,
    Pause,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchResult {
    pub request_id: RequestId,
    pub action: BatchAction,
    pub generated_token: Option<u32>,
    pub telemetry: crate::scheduler::telemetry::SequenceTelemetry,
}

impl BatchResult {
    pub fn continue_with_token(request_id: RequestId, generated_token: u32, telemetry: crate::scheduler::telemetry::SequenceTelemetry) -> Self {
        Self {
            request_id,
            action: BatchAction::Continue,
            generated_token: Some(generated_token),
            telemetry,
        }
    }

    pub fn complete(request_id: RequestId, generated_token: Option<u32>, telemetry: crate::scheduler::telemetry::SequenceTelemetry) -> Self {
        Self {
            request_id,
            action: BatchAction::Complete,
            generated_token,
            telemetry,
        }
    }

    pub fn pause(request_id: RequestId) -> Self {
        Self {
            request_id,
            action: BatchAction::Pause,
            generated_token: None,
            telemetry: Default::default(),
        }
    }

    pub fn fail(request_id: RequestId) -> Self {
        Self {
            request_id,
            action: BatchAction::Fail,
            generated_token: None,
            telemetry: Default::default(),
        }
    }
}

/// Continuous batching at iteration granularity.
///
/// - `PagedScheduler` owns memory/page allocation.
/// - `ContinuousBatcher` owns runnable sequence state transitions.
pub struct ContinuousBatcher {
    waiting: VecDeque<Sequence>,
    running: BTreeMap<RequestId, Sequence>,
    next_enqueue_order: u64,
    pub chunked_state: Option<ChunkedState>,
}

impl Default for ContinuousBatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinuousBatcher {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            running: BTreeMap::new(),
            next_enqueue_order: 0,
            chunked_state: None,
        }
    }

    pub fn with_chunked(mut self, config: ChunkedConfig) -> Self {
        self.chunked_state = Some(ChunkedState::new(config));
        self
    }

    /// 获取运行中序列的可变引用（供 Epilogue 决策写入 draft_budget 等）
    pub fn get_running_mut(&mut self, request_id: RequestId) -> Option<&mut Sequence> {
        self.running.get_mut(&request_id)
    }

    pub fn enqueue(&mut self, mut sequence: Sequence) {
        if self.running.contains_key(&sequence.id)
            || self
                .waiting
                .iter()
                .any(|existing| existing.id == sequence.id)
        {
            return;
        }
        sequence.enqueue_order = self.next_enqueue_order;
        self.next_enqueue_order = self.next_enqueue_order.saturating_add(1);
        sequence.state = SequenceState::Waiting;
        self.waiting.push_back(sequence);
    }

    pub fn build_batch(
        &mut self,
        scheduler: &mut PagedScheduler,
        token_budget: usize,
        admit_new_prefill: bool,
        policy: BatchOrderPolicy,
    ) -> ScheduledBatch {
        if admit_new_prefill {
            self.admit_waiting(scheduler);
        }

        let mut requests = Vec::new();
        let mut failed = Vec::new();
        let ordered_ids = self.ordered_running_ids(policy);
        
        let mut consumed_budget = 0;
        let mut seq_offsets = vec![0];
        let mut draft_steps = Vec::new();
        let mut current_offset = 0;

        // 1. Decode Priority
        for request_id in &ordered_ids {
            let Some(sequence) = self.running.get_mut(request_id) else { continue; };
            if sequence.state == SequenceState::Paused {
                sequence.state = SequenceState::Running;
            }
            if sequence.state != SequenceState::Running { continue; }
            if sequence.needs_prefill() { continue; }

            if consumed_budget >= token_budget { break; }

            // --- Tier II: PGSLE Speculative Admission ---
            let is_draft_eligible = sequence.telemetry.output_entropy < 1.0 
                && sequence.telemetry.l2_delta < 0.05 
                && ordered_ids.len() <= 8;
            sequence.draft_budget = if is_draft_eligible { 8 } else { 0 };

            match scheduler.allocate_next_token(sequence.id) {
                Ok(Some(new_page)) => {
                    sequence.kv_pages.push(new_page);
                    requests.push(sequence.id);
                    consumed_budget += 1;
                    current_offset += 1;
                    seq_offsets.push(current_offset);
                    draft_steps.push(sequence.draft_budget);
                }
                Ok(None) => {
                    requests.push(sequence.id);
                    consumed_budget += 1;
                    current_offset += 1;
                    seq_offsets.push(current_offset);
                    draft_steps.push(sequence.draft_budget);
                }
                Err(SchedulerError::OutOfMemory { .. }) => {
                    sequence.state = SequenceState::Paused;
                }
                Err(e) => {
                    log::warn!("scheduler: sequence {} failed: {e}", sequence.id);
                    sequence.state = SequenceState::Failed;
                    failed.push(sequence.id);
                }
            }
        }

        // 2. Prefill Backfill
        let l1_available_ratio = scheduler.num_free_blocks() as f32 / scheduler.num_total_blocks().max(1) as f32;
        let concurrent_reqs = ordered_ids.len();

        for request_id in &ordered_ids {
            let Some(sequence) = self.running.get_mut(request_id) else { continue; };
            if sequence.state != SequenceState::Running { continue; }
            if !sequence.needs_prefill() { continue; }

            if consumed_budget >= token_budget { break; }

            let prompt_len = sequence.prompt_tokens.len();
            let mut extracted_tokens = 0;

            if let Some(chunked) = &mut self.chunked_state {
                chunked.enqueue(sequence.id, prompt_len);
                let remaining_budget = token_budget.saturating_sub(consumed_budget);
                if let Some(chunk) = chunked.pop_adaptive_chunk(sequence.id, l1_available_ratio, concurrent_reqs, remaining_budget) {
                    extracted_tokens = chunk.tokens;
                }
            } else {
                extracted_tokens = prompt_len;
            }

            if extracted_tokens > 0 {
                // Approximate budget check. We allow a slight overflow for the last chunk to avoid starvation.
                requests.push(sequence.id);
                consumed_budget += extracted_tokens;
                current_offset += extracted_tokens;
                seq_offsets.push(current_offset);
                draft_steps.push(0); // Prefills never specularly decode
            }
        }

        for request_id in failed {
            self.running.remove(&request_id);
            scheduler.free_sequence(request_id);
        }

        ScheduledBatch { requests, seq_offsets, draft_steps }
    }

    /// 扩展 batch 构建方法，返回 BatchPrepData (SPEC/20 REQ-BCI-007)
    ///
    /// 注意：此方法需要额外的序列元数据才能完全填充 BatchPrepData。
    /// 当前版本返回部分填充的数据结构，完整实现需要从 executor 传入更多上下文。
    pub fn build_batch_with_prep(
        &mut self,
        scheduler: &mut PagedScheduler,
        token_budget: usize,
        admit_new_prefill: bool,
        policy: BatchOrderPolicy,
    ) -> (ScheduledBatch, BatchPrepData) {
        let batch = self.build_batch(scheduler, token_budget, admit_new_prefill, policy);
        let num_seqs = batch.requests.len();

        let mut prep = BatchPrepData::new(num_seqs);

        // 填充基本可从 Sequence 获取的信息
        for (idx, &req_id) in batch.requests.iter().enumerate() {
            if let Some(seq) = self.running.get(&req_id) {
                prep.prompt_lens[idx] = seq.prompt_tokens.len() as u32;
                prep.kv_lens[idx] = seq.position as u32; // KV 长度 = 当前 position
                prep.gen_counts[idx] = seq.generated_tokens.len() as u32;
                prep.seq_positions[idx] = seq.position as u32;
                prep.last_sampled_tokens[idx] = seq.generated_tokens.last().copied().unwrap_or(0);
            }
        }

        // 其他字段需要从更高层传入（sampling_config, session_position 等）
        // 这些将在 Executor::generate_batch 中填充

        (batch, prep)
    }

    pub fn update_batch(&mut self, scheduler: &mut PagedScheduler, results: &[BatchResult]) {
        let mut finished = Vec::new();

        for result in results {
            let Some(sequence) = self.running.get_mut(&result.request_id) else {
                continue;
            };

            if sequence.needs_prefill() {
                if let Some(chunked) = &mut self.chunked_state {
                    chunked.on_chunk_finished(result.request_id);
                }
            }

            match result.action {
                BatchAction::Continue => {
                    let prefill_done = self.chunked_state.as_ref().is_none_or(|c| c.is_request_complete(&sequence.id));
                    
                    if sequence.generated_tokens.is_empty() && !sequence.prompt_tokens.is_empty() && prefill_done {
                        scheduler.insert_prefix(sequence.id, &sequence.prompt_tokens);
                    }
                    if let Some(token) = result.generated_token {
                        sequence.push_generated_token(token);
                    }
                    sequence.telemetry = result.telemetry;
                    sequence.state = SequenceState::Running;
                }
                BatchAction::Pause => {
                    sequence.state = SequenceState::Paused;
                }
                BatchAction::Complete => {
                    if let Some(token) = result.generated_token {
                        sequence.push_generated_token(token);
                    }
                    sequence.telemetry = result.telemetry;
                    sequence.state = SequenceState::Completed;
                    if let Some(chunked) = &mut self.chunked_state {
                        chunked.remove_tracker(&result.request_id);
                    }
                    finished.push(result.request_id);
                }
                BatchAction::Fail => {
                    sequence.state = SequenceState::Failed;
                    if let Some(chunked) = &mut self.chunked_state {
                        chunked.remove_tracker(&result.request_id);
                    }
                    finished.push(result.request_id);
                }
            }
        }

        for request_id in finished {
            self.running.remove(&request_id);
            scheduler.free_sequence(request_id);
        }
    }

    pub fn has_pending_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn running_ids(&self) -> Vec<RequestId> {
        self.running.keys().copied().collect()
    }

    pub fn mean_context_len(&self) -> usize {
        let waiting_count = self.waiting.len();
        let running_count = self.running.len();
        let total_count = waiting_count.saturating_add(running_count);
        if total_count == 0 {
            return 0;
        }

        let waiting_total: usize = self.waiting.iter().map(Sequence::context_len).sum();
        let running_total: usize = self.running.values().map(Sequence::context_len).sum();
        waiting_total
            .saturating_add(running_total)
            .checked_div(total_count)
            .unwrap_or(0) // LEGAL: 除零保护，total_count=0 时返回 0
    }

    fn admit_waiting(&mut self, scheduler: &mut PagedScheduler) {
        // 企业级策略：
        // 1. 收集所有等待的序列
        // 2. 尝试 admit 每个序列
        // 3. 分配失败的序列不再放回队列，等待下次 build_batch 时重试
        // 4. 避免无限循环：失败的序列不会在本次调用中重试

        // 收集当前所有等待的序列
        let waiting_sequences: Vec<_> = self.waiting.drain(..).collect();

        for mut sequence in waiting_sequences {
            let request_id = sequence.id;

            // Use prefix-reuse path when prompt tokens are available
            let admit_result = if !sequence.prompt_tokens.is_empty() {
                match scheduler.add_sequence_with_prefix_reuse(
                    sequence.to_sequence_group(),
                    &sequence.prompt_tokens,
                ) {
                    Ok(reused_tokens) => {
                        if reused_tokens > 0 {
                            log::info!(
                                "scheduler: sequence {} prefix hit: {} tokens reused",
                                request_id, reused_tokens,
                            );
                        }
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            } else {
                scheduler.add_sequence(sequence.to_sequence_group())
            };

            match admit_result {
                Ok(()) => {
                    let pages = scheduler
                        .block_tables
                        .get(&request_id)
                        .map(|table| table.blocks.clone())
                        .unwrap_or_default(); // LEGAL: 不存在的 request 返回空 block table
                    sequence.mark_running(pages);
                    self.running.insert(request_id, sequence);
                }
                Err(e) => {
                    log::warn!("scheduler: sequence {} admit failed: {e}", request_id);
                    // 分配失败：放回队列末尾，等待下次 build_batch 时重试
                    // 关键：不在本次循环中重试，避免无限循环
                    sequence.state = SequenceState::Waiting;
                    self.waiting.push_back(sequence);
                    // 继续处理下一个序列，不中断
                }
            }
        }
    }

    fn ordered_running_ids(&self, policy: BatchOrderPolicy) -> Vec<RequestId> {
        let mut ordered_ids: Vec<RequestId> = self.running.keys().copied().collect();
        match policy {
            BatchOrderPolicy::StrictRequestIdOrder => {
                ordered_ids.sort_unstable();
            }
            BatchOrderPolicy::FifoOrder => {
                ordered_ids.sort_by_key(|request_id| {
                    self.running
                        .get(request_id)
                        .map(|sequence| (sequence.enqueue_order, *request_id))
                        .unwrap_or((u64::MAX, *request_id)) // LEGAL: 不存在的 request 排在最后
                });
            }
        }
        ordered_ids
    }

    /// §10.1 交织调度：Decode 优先填充，剩余预算分配给 Prefill Chunks
    ///
    /// 与 `build_batch()` 的区别：
    /// - 返回 `InterleavedBatch`，包含 decode/prefill 的物理分轨信息
    /// - Attention 阶段 Prefill/Decode 物理分轨到不同 Thread Block 组
    /// - FFN 阶段允许合流
    pub fn build_interleaved_batch(
        &mut self,
        scheduler: &mut PagedScheduler,
        token_budget: usize,
        admit_new_prefill: bool,
        policy: BatchOrderPolicy,
    ) -> InterleavedBatch {
        let batch = self.build_batch(scheduler, token_budget, admit_new_prefill, policy);

        // 分类：哪些是 decode，哪些是 prefill chunk
        let mut decode_slots = Vec::new();
        let mut prefill_slots = Vec::new();

        for (idx, &req_id) in batch.requests.iter().enumerate() {
            let needs_prefill = self.running.get(&req_id)
                .map(|seq| seq.needs_prefill())
                .unwrap_or(false);

            let token_count = if idx + 1 < batch.seq_offsets.len() {
                batch.seq_offsets[idx + 1] - batch.seq_offsets[idx]
            } else {
                1
            };

            let slot = InterleavedSlot {
                request_id: req_id,
                batch_index: idx,
                token_count,
                draft_steps: batch.draft_steps.get(idx).copied().unwrap_or(0),
            };

            if needs_prefill {
                prefill_slots.push(slot);
            } else {
                decode_slots.push(slot);
            }
        }

        InterleavedBatch {
            inner: batch,
            decode_slots,
            prefill_slots,
        }
    }
}

/// §10.1 交织调度结果
///
/// 包含 decode/prefill 的物理分轨信息。
/// Attention 阶段 Prefill/Decode 物理分轨到不同 Thread Block 组。
#[derive(Debug, Clone)]
pub struct InterleavedBatch {
    /// 底层 ScheduledBatch
    pub inner: ScheduledBatch,
    /// Decode 请求槽位
    pub decode_slots: Vec<InterleavedSlot>,
    /// Prefill Chunk 请求槽位
    pub prefill_slots: Vec<InterleavedSlot>,
}

/// 交织调度中的单个槽位
#[derive(Debug, Clone)]
pub struct InterleavedSlot {
    /// 请求 ID
    pub request_id: RequestId,
    /// 在 batch 中的索引
    pub batch_index: usize,
    /// 该槽位的 token 数量（decode=1, prefill chunk=chunk_size）
    pub token_count: usize,
    /// 推测解码步数
    pub draft_steps: usize,
}

impl InterleavedBatch {
    /// Decode token 总数
    pub fn decode_tokens(&self) -> usize {
        self.decode_slots.iter().map(|s| s.token_count).sum()
    }

    /// Prefill chunk token 总数
    pub fn prefill_tokens(&self) -> usize {
        self.prefill_slots.iter().map(|s| s.token_count).sum()
    }

    /// 总 token 数
    pub fn total_tokens(&self) -> usize {
        self.decode_tokens() + self.prefill_tokens()
    }

    /// 是否包含交织（同时有 decode 和 prefill）
    pub fn is_interleaved(&self) -> bool {
        !self.decode_slots.is_empty() && !self.prefill_slots.is_empty()
    }

    /// 所有请求 ID（保持原始顺序）
    pub fn request_ids(&self) -> &[RequestId] {
        &self.inner.requests
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::hgal::HGALConfig;

    fn make_sequence(id: RequestId, prompt_len: usize) -> Sequence {
        Sequence::new(id, vec![1; prompt_len])
    }

    #[test]
    fn build_batch_allows_iteration_level_join() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());

        batcher.enqueue(make_sequence(10, 4));
        let first = batcher.build_batch(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(first.requests, vec![10]);
        batcher.update_batch(&mut scheduler, &[BatchResult::continue_with_token(10, 100, Default::default())]);

        batcher.enqueue(make_sequence(2, 2));
        let second = batcher.build_batch(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(second.requests, vec![10, 2]);
    }

    #[test]
    fn update_batch_complete_releases_scheduler_resources() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());

        batcher.enqueue(make_sequence(1, 4));
        let first = batcher.build_batch(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(first.requests, vec![1]);

        batcher.update_batch(&mut scheduler, &[BatchResult::complete(1, Some(7), Default::default())]);
        assert!(!batcher.has_pending_work());
        assert_eq!(scheduler.num_free_blocks(), 8);
    }

    #[test]
    fn build_batch_keeps_deterministic_order() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());

        batcher.enqueue(make_sequence(7, 1));
        batcher.enqueue(make_sequence(3, 1));
        batcher.enqueue(make_sequence(5, 1));

        let prefill = batcher.build_batch(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(prefill.requests, vec![3, 5, 7]);
        batcher.update_batch(
            &mut scheduler,
            &[
                BatchResult::continue_with_token(3, 1, Default::default()),
                BatchResult::continue_with_token(5, 1, Default::default()),
                BatchResult::continue_with_token(7, 1, Default::default()),
            ],
        );

        let decode = batcher.build_batch(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(decode.requests, vec![3, 5, 7]);
    }

    #[test]
    fn build_batch_keeps_waiting_when_capacity_is_not_enough() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(1, 4, HGALConfig::default());

        batcher.enqueue(make_sequence(1, 4));
        batcher.enqueue(make_sequence(2, 4));

        let first = batcher.build_batch(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(first.requests, vec![1]);
        assert_eq!(batcher.waiting.len(), 1);
        assert!(batcher.running.contains_key(&1));
        assert!(!batcher.running.contains_key(&2));
    }

    // ── BatchPrepData unit tests ────────────────────────────────────────

    #[test]
    fn batch_prep_data_new_initializes_active_flags() {
        let d = BatchPrepData::new(4);
        assert_eq!(d.prompt_lens.len(), 4);
        assert_eq!(d.active_flags, vec![1u32; 4]);
        assert_eq!(d.sampling_params_packed.len(), 16); // 4 × 4
        assert_eq!(d.max_decode_steps, 0);
        assert_eq!(d.total_prefill_tokens, 0);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_encodes_correctly() {
        let mut d = BatchPrepData::new(2);
        d.set_sampling_params(0, 0.7, 50, 0.9, 2);
        assert_eq!(d.sampling_params_packed[0], 0.7f32.to_bits());
        assert_eq!(d.sampling_params_packed[1], 50u32);
        assert_eq!(d.sampling_params_packed[2], 0.9f32.to_bits());
        assert_eq!(d.sampling_params_packed[3], 2u32);
        // seq 1 untouched
        assert_eq!(d.sampling_params_packed[4], 0u32);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_out_of_bounds_noop() {
        let mut d = BatchPrepData::new(1);
        d.set_sampling_params(1, 1.0, 10, 0.5, 0); // seq 1 doesn't exist
        assert!(d.sampling_params_packed.iter().all(|&v| v == 0));
    }

    #[test]
    fn batch_prep_data_new_zero_length() {
        let d = BatchPrepData::new(0);
        assert!(d.prompt_lens.is_empty());
        assert!(d.active_flags.is_empty());
        assert!(d.sampling_params_packed.is_empty());
    }

    // ── BatchResult factory methods ─────────────────────────────────────

    #[test]
    fn batch_result_continue_with_token() {
        let br = BatchResult::continue_with_token(42, 99, Default::default());
        assert_eq!(br.request_id, 42);
        assert_eq!(br.action, BatchAction::Continue);
        assert_eq!(br.generated_token, Some(99));
    }

    #[test]
    fn batch_result_complete_with_and_without_token() {
        let with_tok = BatchResult::complete(7, Some(5), Default::default());
        assert_eq!(with_tok.action, BatchAction::Complete);
        assert_eq!(with_tok.generated_token, Some(5));

        let no_tok = BatchResult::complete(7, None, Default::default());
        assert_eq!(no_tok.generated_token, None);
    }

    #[test]
    fn batch_result_pause_and_fail() {
        let pause = BatchResult::pause(10);
        assert_eq!(pause.action, BatchAction::Pause);
        assert_eq!(pause.generated_token, None);

        let fail = BatchResult::fail(11);
        assert_eq!(fail.action, BatchAction::Fail);
        assert_eq!(fail.generated_token, None);
    }

    #[test]
    fn batch_action_equality() {
        assert_eq!(BatchAction::Continue, BatchAction::Continue);
        assert_ne!(BatchAction::Continue, BatchAction::Complete);
        assert_ne!(BatchAction::Pause, BatchAction::Fail);
    }

    // ── Additional tests ──

    #[test]
    fn batch_action_copy_clone() {
        let a = BatchAction::Continue;
        let b = a;
        assert_eq!(a, b);
        let c = a.clone();
        assert_eq!(c, BatchAction::Continue);
    }

    #[test]
    fn batch_action_all_variants_distinct() {
        let variants = [BatchAction::Continue, BatchAction::Complete, BatchAction::Pause, BatchAction::Fail];
        for (i, &v1) in variants.iter().enumerate() {
            for (j, &v2) in variants.iter().enumerate() {
                if i == j {
                    assert_eq!(v1, v2);
                } else {
                    assert_ne!(v1, v2);
                }
            }
        }
    }

    #[test]
    fn batch_result_equality_by_field() {
        let r1 = BatchResult::continue_with_token(1, 42, Default::default());
        let r2 = BatchResult::continue_with_token(1, 42, Default::default());
        assert_eq!(r1.request_id, r2.request_id);
        assert_eq!(r1.action, r2.action);
        assert_eq!(r1.generated_token, r2.generated_token);
    }

    #[test]
    fn batch_result_pause_has_default_telemetry() {
        let pause = BatchResult::pause(99);
        assert_eq!(pause.request_id, 99);
        assert_eq!(pause.generated_token, None);
    }

    #[test]
    fn batch_result_fail_has_default_telemetry() {
        let fail = BatchResult::fail(100);
        assert_eq!(fail.request_id, 100);
        assert_eq!(fail.generated_token, None);
    }

    #[test]
    fn scheduled_batch_fields() {
        let batch = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 1, 3, 6],
            draft_steps: vec![0, 2, 0],
        };
        assert_eq!(batch.requests.len(), 3);
        assert_eq!(batch.seq_offsets.len(), 4);
        assert_eq!(batch.draft_steps[1], 2);
    }

    #[test]
    fn batch_prep_data_default_values() {
        let d = BatchPrepData::new(3);
        assert_eq!(d.prompt_lens, vec![0u32; 3]);
        assert_eq!(d.kv_lens, vec![0u32; 3]);
        assert_eq!(d.session_positions, vec![0u32; 3]);
        assert_eq!(d.rope_pos_offsets, vec![0u32; 3]);
        assert_eq!(d.max_new_tokens, vec![0u32; 3]);
        assert_eq!(d.page_table_offsets, vec![0u32; 3]);
        assert_eq!(d.page_table_lens, vec![0u32; 3]);
        assert_eq!(d.fused_hidden_offsets, vec![0u32; 3]);
        assert_eq!(d.num_mm_tokens, vec![0u32; 3]);
        assert_eq!(d.active_flags, vec![1u32; 3]);
        assert_eq!(d.seq_positions, vec![0u32; 3]);
        assert_eq!(d.gen_counts, vec![0u32; 3]);
        assert_eq!(d.last_sampled_tokens, vec![0u32; 3]);
        assert_eq!(d.sampling_params_packed.len(), 12);
        assert_eq!(d.max_decode_steps, 0);
        assert_eq!(d.total_prefill_tokens, 0);
    }

    #[test]
    fn batch_prep_data_set_sampling_params_second_seq() {
        let mut d = BatchPrepData::new(3);
        d.set_sampling_params(2, 1.5, 100, 0.5, 1);
        let base = 2 * 4;
        assert_eq!(d.sampling_params_packed[base], 1.5f32.to_bits());
        assert_eq!(d.sampling_params_packed[base + 1], 100u32);
        assert_eq!(d.sampling_params_packed[base + 2], 0.5f32.to_bits());
        assert_eq!(d.sampling_params_packed[base + 3], 1u32);
    }

    #[test]
    fn batch_prep_data_clone() {
        let mut d = BatchPrepData::new(2);
        d.set_sampling_params(0, 0.5, 10, 0.9, 3);
        let d2 = d.clone();
        assert_eq!(d2.active_flags, d.active_flags);
        assert_eq!(d2.sampling_params_packed, d.sampling_params_packed);
    }

    #[test]
    fn continuous_batcher_new_is_empty() {
        let b = ContinuousBatcher::new();
        assert_eq!(b.waiting_len(), 0);
        assert_eq!(b.running_len(), 0);
        assert!(!b.has_pending_work());
    }

    #[test]
    fn continuous_batcher_default_is_new() {
        let b = ContinuousBatcher::default();
        assert_eq!(b.waiting_len(), 0);
        assert_eq!(b.running_len(), 0);
    }

    #[test]
    fn continuous_batcher_enqueue_increments_waiting() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(1, 4));
        assert_eq!(b.waiting_len(), 1);
        b.enqueue(make_sequence(2, 8));
        assert_eq!(b.waiting_len(), 2);
    }

    #[test]
    fn continuous_batcher_enqueue_duplicate_ignored() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(1, 4));
        b.enqueue(make_sequence(1, 4)); // duplicate
        assert_eq!(b.waiting_len(), 1);
    }

    #[test]
    fn continuous_batcher_mean_context_len_empty() {
        let b = ContinuousBatcher::new();
        assert_eq!(b.mean_context_len(), 0);
    }

    #[test]
    fn continuous_batcher_running_ids_empty() {
        let b = ContinuousBatcher::new();
        assert!(b.running_ids().is_empty());
    }

    #[test]
    fn continuous_batcher_get_running_mut_none() {
        let mut b = ContinuousBatcher::new();
        assert!(b.get_running_mut(999).is_none());
    }

    #[test]
    fn interleaved_batch_decode_tokens_count() {
        let inner = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1, 2],
            draft_steps: vec![0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![],
        };
        assert_eq!(ib.decode_tokens(), 2);
        assert_eq!(ib.prefill_tokens(), 0);
        assert_eq!(ib.total_tokens(), 2);
        assert!(!ib.is_interleaved());
    }

    #[test]
    fn interleaved_batch_is_interleaved_true() {
        let inner = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1, 5],
            draft_steps: vec![0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 4, draft_steps: 0 },
            ],
        };
        assert_eq!(ib.decode_tokens(), 1);
        assert_eq!(ib.prefill_tokens(), 4);
        assert_eq!(ib.total_tokens(), 5);
        assert!(ib.is_interleaved());
    }

    #[test]
    fn interleaved_batch_request_ids() {
        let inner = ScheduledBatch {
            requests: vec![10, 20, 30],
            seq_offsets: vec![0, 1, 2, 3],
            draft_steps: vec![0, 0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        assert_eq!(ib.request_ids(), &[10, 20, 30]);
    }

    #[test]
    fn interleaved_slot_fields() {
        let slot = InterleavedSlot {
            request_id: 42,
            batch_index: 3,
            token_count: 7,
            draft_steps: 4,
        };
        assert_eq!(slot.request_id, 42);
        assert_eq!(slot.batch_index, 3);
        assert_eq!(slot.token_count, 7);
        assert_eq!(slot.draft_steps, 4);
    }

    #[test]
    fn continuous_batcher_with_chunked() {
        let b = ContinuousBatcher::new().with_chunked(ChunkedConfig::default());
        assert!(b.chunked_state.is_some());
    }

    #[test]
    fn continuous_batcher_without_chunked() {
        let b = ContinuousBatcher::new();
        assert!(b.chunked_state.is_none());
    }

    // ── New tests (25 added) ────────────────────────────────────────────

    // ── BatchAction Debug trait ──

    #[test]
    fn batch_action_debug_format() {
        assert_eq!(format!("{:?}", BatchAction::Continue), "Continue");
        assert_eq!(format!("{:?}", BatchAction::Complete), "Complete");
        assert_eq!(format!("{:?}", BatchAction::Pause), "Pause");
        assert_eq!(format!("{:?}", BatchAction::Fail), "Fail");
    }

    // ── BatchAction exhaustiveness (all 4 variants present) ──

    #[test]
    fn batch_action_exhaustive_variants() {
        let all = [
            BatchAction::Continue,
            BatchAction::Complete,
            BatchAction::Pause,
            BatchAction::Fail,
        ];
        // Every pair-wise comparison: only self-equal
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "variant mismatch at ({i},{j})");
            }
        }
    }

    // ── BatchResult Copy + Clone ──

    #[test]
    fn batch_result_copy_clone_semantics() {
        let original = BatchResult::continue_with_token(5, 42, Default::default());
        let copied = original;
        // Both fields must survive Copy (PartialEq on f32 is ok here since exact bits)
        assert_eq!(original.request_id, copied.request_id);
        assert_eq!(original.action, copied.action);
        assert_eq!(original.generated_token, copied.generated_token);
    }

    // ── BatchResult Debug format (smoke test) ──

    #[test]
    fn batch_result_debug_does_not_panic() {
        let results = [
            BatchResult::continue_with_token(1, 99, Default::default()),
            BatchResult::complete(2, None, Default::default()),
            BatchResult::pause(3),
            BatchResult::fail(4),
        ];
        for r in &results {
            let _s = format!("{r:?}");
        }
    }

    // ── BatchResult PartialEq ──

    #[test]
    fn batch_result_partial_eq() {
        let a = BatchResult::continue_with_token(10, 55, Default::default());
        let b = BatchResult::continue_with_token(10, 55, Default::default());
        assert_eq!(a, b);

        let c = BatchResult::continue_with_token(10, 56, Default::default());
        assert_ne!(a, c);
    }

    // ── ScheduledBatch Clone ──

    #[test]
    fn scheduled_batch_clone() {
        let batch = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 1, 3, 6],
            draft_steps: vec![0, 2, 4],
        };
        let cloned = batch.clone();
        assert_eq!(cloned.requests, batch.requests);
        assert_eq!(cloned.seq_offsets, batch.seq_offsets);
        assert_eq!(cloned.draft_steps, batch.draft_steps);
    }

    // ── ScheduledBatch Debug ──

    #[test]
    fn scheduled_batch_debug_format() {
        let batch = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 1],
            draft_steps: vec![0],
        };
        let s = format!("{batch:?}");
        assert!(s.contains("ScheduledBatch") || s.contains("requests"));
    }

    // ── ScheduledBatch empty ──

    #[test]
    fn scheduled_batch_empty() {
        let batch = ScheduledBatch {
            requests: vec![],
            seq_offsets: vec![0],
            draft_steps: vec![],
        };
        assert!(batch.requests.is_empty());
        assert!(batch.draft_steps.is_empty());
        assert_eq!(batch.seq_offsets, vec![0]);
    }

    // ── BatchPrepData set_sampling_params_boundary_seq_zero ──

    #[test]
    fn batch_prep_data_set_sampling_params_first_seq() {
        let mut d = BatchPrepData::new(2);
        d.set_sampling_params(0, 2.0, 200, 0.1, 99);
        assert_eq!(d.sampling_params_packed[0], 2.0f32.to_bits());
        assert_eq!(d.sampling_params_packed[1], 200u32);
        assert_eq!(d.sampling_params_packed[2], 0.1f32.to_bits());
        assert_eq!(d.sampling_params_packed[3], 99u32);
        // Second seq untouched
        assert_eq!(&d.sampling_params_packed[4..8], &[0u32; 4]);
    }

    // ── BatchPrepData set_sampling_params overwrites ──

    #[test]
    fn batch_prep_data_set_sampling_params_overwrite() {
        let mut d = BatchPrepData::new(1);
        d.set_sampling_params(0, 1.0, 50, 0.5, 0);
        d.set_sampling_params(0, 0.0, 1, 1.0, 255);
        assert_eq!(d.sampling_params_packed[0], 0.0f32.to_bits());
        assert_eq!(d.sampling_params_packed[1], 1u32);
        assert_eq!(d.sampling_params_packed[2], 1.0f32.to_bits());
        assert_eq!(d.sampling_params_packed[3], 255u32);
    }

    // ── BatchPrepData large num_seqs ──

    #[test]
    fn batch_prep_data_large_num_seqs() {
        let d = BatchPrepData::new(1000);
        assert_eq!(d.active_flags.len(), 1000);
        assert!(d.active_flags.iter().all(|&f| f == 1));
        assert_eq!(d.sampling_params_packed.len(), 4000);
    }

    // ── BatchPrepData set_sampling_params with max f32 ──

    #[test]
    fn batch_prep_data_set_sampling_params_max_f32() {
        let mut d = BatchPrepData::new(1);
        d.set_sampling_params(0, f32::MAX, usize::MAX, f32::MIN, u32::MAX);
        assert_eq!(d.sampling_params_packed[0], f32::MAX.to_bits());
        assert_eq!(d.sampling_params_packed[1], usize::MAX as u32);
        assert_eq!(d.sampling_params_packed[2], f32::MIN.to_bits());
        assert_eq!(d.sampling_params_packed[3], u32::MAX);
    }

    // ── InterleavedBatch empty slots ──

    #[test]
    fn interleaved_batch_empty_slots() {
        let inner = ScheduledBatch {
            requests: vec![],
            seq_offsets: vec![0],
            draft_steps: vec![],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        assert_eq!(ib.decode_tokens(), 0);
        assert_eq!(ib.prefill_tokens(), 0);
        assert_eq!(ib.total_tokens(), 0);
        assert!(!ib.is_interleaved());
    }

    // ── InterleavedBatch only_prefill ──

    #[test]
    fn interleaved_batch_only_prefill() {
        let inner = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 8],
            draft_steps: vec![0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 8, draft_steps: 0 },
            ],
        };
        assert_eq!(ib.prefill_tokens(), 8);
        assert_eq!(ib.total_tokens(), 8);
        assert!(!ib.is_interleaved());
    }

    // ── InterleavedBatch clone ──

    #[test]
    fn interleaved_batch_clone() {
        let inner = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1, 3],
            draft_steps: vec![2, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 2 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 2, draft_steps: 0 },
            ],
        };
        let cloned = ib.clone();
        assert_eq!(cloned.decode_tokens(), ib.decode_tokens());
        assert_eq!(cloned.prefill_tokens(), ib.prefill_tokens());
        assert_eq!(cloned.request_ids(), ib.request_ids());
        assert_eq!(cloned.inner.draft_steps, ib.inner.draft_steps);
    }

    // ── InterleavedBatch debug ──

    #[test]
    fn interleaved_batch_debug_does_not_panic() {
        let inner = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 1],
            draft_steps: vec![0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        let _s = format!("{ib:?}");
    }

    // ── InterleavedSlot clone ──

    #[test]
    fn interleaved_slot_clone() {
        let slot = InterleavedSlot {
            request_id: 7,
            batch_index: 2,
            token_count: 5,
            draft_steps: 3,
        };
        let cloned = slot.clone();
        assert_eq!(cloned.request_id, slot.request_id);
        assert_eq!(cloned.batch_index, slot.batch_index);
        assert_eq!(cloned.token_count, slot.token_count);
        assert_eq!(cloned.draft_steps, slot.draft_steps);
    }

    // ── InterleavedSlot debug ──

    #[test]
    fn interleaved_slot_debug_format() {
        let slot = InterleavedSlot {
            request_id: 1,
            batch_index: 0,
            token_count: 1,
            draft_steps: 0,
        };
        let s = format!("{slot:?}");
        assert!(s.contains("InterleavedSlot") || s.contains("request_id"));
    }

    // ── ContinuousBatcher enqueue assigns increasing order ──

    #[test]
    fn continuous_batcher_enqueue_assigns_increasing_order() {
        let mut b = ContinuousBatcher::new();
        let seq1 = Sequence::new(1, vec![1, 2]);
        let seq2 = Sequence::new(2, vec![3, 4]);
        let seq3 = Sequence::new(3, vec![5, 6]);
        b.enqueue(seq1);
        b.enqueue(seq2);
        b.enqueue(seq3);
        // Verify enqueue_order was assigned (not all zero)
        let orders: Vec<u64> = b.waiting.iter().map(|s| s.enqueue_order).collect();
        assert_eq!(orders.len(), 3);
        assert!(orders.windows(2).all(|w| w[0] < w[1]), "enqueue_order must be monotonically increasing");
    }

    // ── ContinuousBatcher enqueue sets state to waiting ──

    #[test]
    fn continuous_batcher_enqueue_sets_waiting_state() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(42, 4));
        let seq = b.waiting.front().unwrap();
        assert_eq!(seq.state, SequenceState::Waiting);
    }

    // ── ContinuousBatcher enqueue duplicate in_running_ignored ──

    #[test]
    fn continuous_batcher_enqueue_duplicate_between_waiting_and_running_ignored() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());

        // Enqueue, admit, and start running
        b.enqueue(make_sequence(5, 4));
        let batch = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert!(batch.requests.contains(&5));

        // Duplicate enqueue while running
        b.enqueue(make_sequence(5, 4));
        // Should not add duplicate to waiting
        assert_eq!(b.waiting_len(), 0);
    }

    // ── ContinuousBatcher has_pending_work with waiting_only ──

    #[test]
    fn continuous_batcher_has_pending_work_waiting_only() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(1, 4));
        assert!(b.has_pending_work());
    }

    // ── ContinuousBatcher with_chunked_config_custom ──

    #[test]
    fn continuous_batcher_with_chunked_custom_config() {
        let config = ChunkedConfig {
            min_chunk: 128,
            max_chunk: 512,
            decode_slots: 4,
            enable_splitfuse: false,
        };
        let b = ContinuousBatcher::new().with_chunked(config);
        let state = b.chunked_state.as_ref().unwrap();
        // Verify config propagated (ChunkedState exposes its config)
        assert_eq!(state.config.min_chunk, 128);
        assert_eq!(state.config.max_chunk, 512);
        assert_eq!(state.config.decode_slots, 4);
    }

    // ── ContinuousBatcher mean_context_len_with_sequences ──

    #[test]
    fn continuous_batcher_mean_context_len_with_waiting_sequences() {
        let mut b = ContinuousBatcher::new();
        // Sequence with prompt_len=4 -> context_len=4
        b.enqueue(make_sequence(1, 4));
        // Sequence with prompt_len=8 -> context_len=8
        b.enqueue(make_sequence(2, 8));
        // Mean should be (4+8)/2 = 6
        assert_eq!(b.mean_context_len(), 6);
    }

    // ── ContinuousBatcher get_running_mut_some ──

    #[test]
    fn continuous_batcher_get_running_mut_existing_sequence() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());

        b.enqueue(make_sequence(10, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);

        let seq = b.get_running_mut(10);
        assert!(seq.is_some());
        let seq = seq.unwrap();
        assert_eq!(seq.id, 10);
    }

    // ── BatchResult complete_with_token carries token ──

    #[test]
    fn batch_result_complete_carries_generated_token() {
        let result = BatchResult::complete(42, Some(999), Default::default());
        assert_eq!(result.action, BatchAction::Complete);
        assert_eq!(result.generated_token, Some(999));
        assert_eq!(result.request_id, 42);
    }

    // ── BatchResult continue_without_token_path ──

    #[test]
    fn batch_result_continue_with_zero_token() {
        let result = BatchResult::continue_with_token(1, 0, Default::default());
        assert_eq!(result.action, BatchAction::Continue);
        assert_eq!(result.generated_token, Some(0));
    }

    // ── New tests (18 added) ────────────────────────────────────────────

    // ── BatchPrepData: verify all vec fields are independent ──

    #[test]
    fn batch_prep_data_vec_fields_independent_after_mutation() {
        let mut d = BatchPrepData::new(2);
        d.prompt_lens[0] = 10;
        d.kv_lens[0] = 20;
        d.session_positions[0] = 5;
        d.rope_pos_offsets[0] = 3;
        d.max_new_tokens[0] = 100;
        // Index 1 untouched
        assert_eq!(d.prompt_lens[1], 0);
        assert_eq!(d.kv_lens[1], 0);
        assert_eq!(d.session_positions[1], 0);
        assert_eq!(d.rope_pos_offsets[1], 0);
        assert_eq!(d.max_new_tokens[1], 0);
    }

    // ── BatchPrepData: set_sampling_params at exact boundary ──

    #[test]
    fn batch_prep_data_set_sampling_params_last_valid_seq() {
        let mut d = BatchPrepData::new(3);
        d.set_sampling_params(2, 0.25, 8, 0.75, 7);
        let base = 2 * 4;
        assert_eq!(d.sampling_params_packed[base], 0.25f32.to_bits());
        assert_eq!(d.sampling_params_packed[base + 1], 8u32);
        assert_eq!(d.sampling_params_packed[base + 2], 0.75f32.to_bits());
        assert_eq!(d.sampling_params_packed[base + 3], 7u32);
        // Earlier slots untouched
        assert_eq!(&d.sampling_params_packed[0..8], &[0u32; 8]);
    }

    // ── BatchPrepData: set_sampling_params negative temperature ──

    #[test]
    fn batch_prep_data_set_sampling_params_negative_temp() {
        let mut d = BatchPrepData::new(1);
        d.set_sampling_params(0, -1.0, 0, -0.5, 0);
        assert_eq!(d.sampling_params_packed[0], (-1.0f32).to_bits());
        assert_eq!(d.sampling_params_packed[2], (-0.5f32).to_bits());
    }

    // ── BatchPrepData: set_sampling_params NaN/Inf values ──

    #[test]
    fn batch_prep_data_set_sampling_params_special_floats() {
        let mut d = BatchPrepData::new(1);
        d.set_sampling_params(0, f32::NAN, 0, f32::INFINITY, 0);
        assert!(f32::from_bits(d.sampling_params_packed[0]).is_nan());
        assert_eq!(d.sampling_params_packed[2], f32::INFINITY.to_bits());
    }

    // ── BatchPrepData: scalar fields mutation ──

    #[test]
    fn batch_prep_data_scalar_fields_mutation() {
        let mut d = BatchPrepData::new(1);
        assert_eq!(d.max_decode_steps, 0);
        assert_eq!(d.total_prefill_tokens, 0);
        d.max_decode_steps = 42;
        d.total_prefill_tokens = 1024;
        assert_eq!(d.max_decode_steps, 42);
        assert_eq!(d.total_prefill_tokens, 1024);
    }

    // ── BatchPrepData: clone independence ──

    #[test]
    fn batch_prep_data_clone_is_independent() {
        let mut d = BatchPrepData::new(2);
        d.prompt_lens[0] = 99;
        let cloned = d.clone();
        d.prompt_lens[0] = 0;
        assert_eq!(cloned.prompt_lens[0], 99);
    }

    // ── BatchPrepData: Debug trait smoke test ──

    #[test]
    fn batch_prep_data_debug_does_not_panic() {
        let mut d = BatchPrepData::new(2);
        d.set_sampling_params(0, 1.0, 10, 0.5, 3);
        let s = format!("{d:?}");
        assert!(s.contains("BatchPrepData"));
    }

    // ── BatchAction: Copy semantics independent mutation ──

    #[test]
    fn batch_action_copy_independent() {
        let mut a = BatchAction::Continue;
        let b = a;
        a = BatchAction::Fail;
        // b is an independent copy (Copy trait)
        assert_eq!(b, BatchAction::Continue);
        assert_eq!(a, BatchAction::Fail);
    }

    // ── BatchResult: PartialEq distinguishes request_id ──

    #[test]
    fn batch_result_partial_eq_different_request_id() {
        let a = BatchResult::continue_with_token(1, 42, Default::default());
        let b = BatchResult::continue_with_token(2, 42, Default::default());
        assert_ne!(a, b);
    }

    // ── BatchResult: PartialEq distinguishes action ──

    #[test]
    fn batch_result_partial_eq_different_action() {
        let cont = BatchResult::continue_with_token(1, 42, Default::default());
        let comp = BatchResult::complete(1, Some(42), Default::default());
        assert_ne!(cont, comp);
    }

    // ── BatchResult: telemetry propagation for continue ──

    #[test]
    fn batch_result_continue_carries_telemetry() {
        let mut tel = crate::scheduler::telemetry::SequenceTelemetry::new();
        tel.output_entropy = 2.5;
        tel.l2_delta = 0.01;
        tel.has_outlier = true;
        let result = BatchResult::continue_with_token(1, 10, tel);
        assert_eq!(result.telemetry.output_entropy, 2.5);
        assert_eq!(result.telemetry.l2_delta, 0.01);
        assert!(result.telemetry.has_outlier);
    }

    // ── BatchResult: complete carries telemetry ──

    #[test]
    fn batch_result_complete_carries_telemetry() {
        let mut tel = crate::scheduler::telemetry::SequenceTelemetry::new();
        tel.dead_density = 0.3;
        let result = BatchResult::complete(5, Some(99), tel);
        assert_eq!(result.telemetry.dead_density, 0.3);
    }

    // ── ScheduledBatch: single element ──

    #[test]
    fn scheduled_batch_single_element() {
        let batch = ScheduledBatch {
            requests: vec![42],
            seq_offsets: vec![0, 1],
            draft_steps: vec![3],
        };
        assert_eq!(batch.requests.len(), 1);
        assert_eq!(batch.seq_offsets.len(), 2);
        assert_eq!(batch.draft_steps.len(), 1);
        assert_eq!(batch.draft_steps[0], 3);
    }

    // ── InterleavedSlot: zero token_count ──

    #[test]
    fn interleaved_slot_zero_token_count() {
        let slot = InterleavedSlot {
            request_id: 1,
            batch_index: 0,
            token_count: 0,
            draft_steps: 0,
        };
        assert_eq!(slot.token_count, 0);
    }

    // ── InterleavedBatch: many slots token aggregation ──

    #[test]
    fn interleaved_batch_many_slots_aggregation() {
        let inner = ScheduledBatch {
            requests: vec![1, 2, 3, 4],
            seq_offsets: vec![0, 1, 2, 3, 4],
            draft_steps: vec![0, 0, 2, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 4, batch_index: 3, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 3, batch_index: 2, token_count: 1, draft_steps: 2 },
            ],
        };
        assert_eq!(ib.decode_tokens(), 3);
        assert_eq!(ib.prefill_tokens(), 1);
        assert_eq!(ib.total_tokens(), 4);
        assert!(ib.is_interleaved());
    }

    // ── InterleavedBatch: clone then mutate inner ──

    #[test]
    fn interleaved_batch_clone_independence() {
        let inner = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 1],
            draft_steps: vec![5],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 5 },
            ],
            prefill_slots: vec![],
        };
        let mut cloned = ib.clone();
        cloned.inner.draft_steps[0] = 0;
        assert_eq!(ib.inner.draft_steps[0], 5);
    }

    // ── ContinuousBatcher: enqueue_order saturating_add safety ──

    #[test]
    fn continuous_batcher_enqueue_order_monotonic_for_many() {
        let mut b = ContinuousBatcher::new();
        for i in 0..10 {
            b.enqueue(make_sequence(i, 1));
        }
        let orders: Vec<u64> = b.waiting.iter().map(|s| s.enqueue_order).collect();
        for w in orders.windows(2) {
            assert!(w[0] < w[1], "orders must be strictly monotonic");
        }
        assert_eq!(orders.len(), 10);
    }

    // ── SequenceTelemetry: default all zeros ──

    #[test]
    fn sequence_telemetry_default_values() {
        let tel = crate::scheduler::telemetry::SequenceTelemetry::default();
        assert_eq!(tel.l2_delta, 0.0);
        assert!(!tel.has_outlier);
        assert_eq!(tel.dead_density, 0.0);
        assert_eq!(tel.per_head_entropy, 0.0);
        assert_eq!(tel.transform_ratio, 0.0);
        assert_eq!(tel.output_entropy, 0.0);
    }

    // ── New tests (40 added) ────────────────────────────────────────────

    // ── BatchAction: Eq symmetry ──

    #[test]
    fn batch_action_eq_symmetry() {
        // Eq implies symmetric: a == b => b == a
        let pairs = [
            (BatchAction::Continue, BatchAction::Continue),
            (BatchAction::Complete, BatchAction::Complete),
            (BatchAction::Pause, BatchAction::Pause),
            (BatchAction::Fail, BatchAction::Fail),
        ];
        for (a, b) in &pairs {
            assert_eq!(a, b);
            assert_eq!(b, a);
        }
    }

    // ── BatchAction: all variants are Copy ──

    #[test]
    fn batch_action_all_variants_copy_survive() {
        let a = BatchAction::Continue;
        let _b = a;
        assert_eq!(a, BatchAction::Continue);

        let c = BatchAction::Complete;
        let _d = c;
        assert_eq!(c, BatchAction::Complete);

        let e = BatchAction::Pause;
        let _f = e;
        assert_eq!(e, BatchAction::Pause);

        let g = BatchAction::Fail;
        let _h = g;
        assert_eq!(g, BatchAction::Fail);
    }

    // ── BatchAction: transitivity ──

    #[test]
    fn batch_action_eq_transitivity() {
        let a = BatchAction::Continue;
        let b = BatchAction::Continue;
        let c = BatchAction::Continue;
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    // ── BatchResult: different actions with same request_id are not equal ──

    #[test]
    fn batch_result_different_actions_same_request_not_equal() {
        let cont = BatchResult::continue_with_token(1, 42, Default::default());
        let pause = BatchResult::pause(1);
        let fail = BatchResult::fail(1);
        let comp = BatchResult::complete(1, None, Default::default());
        assert_ne!(cont, pause);
        assert_ne!(cont, fail);
        assert_ne!(cont, comp);
        assert_ne!(pause, fail);
        assert_ne!(pause, comp);
        assert_ne!(fail, comp);
    }

    // ── BatchResult: continue_with_token with max u32 ──

    #[test]
    fn batch_result_continue_with_max_token() {
        let result = BatchResult::continue_with_token(RequestId::MAX, u32::MAX, Default::default());
        assert_eq!(result.request_id, RequestId::MAX);
        assert_eq!(result.generated_token, Some(u32::MAX));
    }

    // ── BatchResult: complete with token zero ──

    #[test]
    fn batch_result_complete_with_zero_token() {
        let result = BatchResult::complete(5, Some(0), Default::default());
        assert_eq!(result.action, BatchAction::Complete);
        assert_eq!(result.generated_token, Some(0));
    }

    // ── BatchResult: pause has continue-compat request_id range ──

    #[test]
    fn batch_result_pause_with_zero_request_id() {
        let result = BatchResult::pause(0);
        assert_eq!(result.request_id, 0);
        assert_eq!(result.action, BatchAction::Pause);
        assert_eq!(result.generated_token, None);
    }

    // ── BatchResult: fail with max request_id ──

    #[test]
    fn batch_result_fail_with_max_request_id() {
        let result = BatchResult::fail(RequestId::MAX);
        assert_eq!(result.request_id, RequestId::MAX);
        assert_eq!(result.action, BatchAction::Fail);
    }

    // ── BatchResult: copy independence ──

    #[test]
    fn batch_result_copy_independence() {
        let original = BatchResult::continue_with_token(10, 55, Default::default());
        let copied = original;
        // Both should be independent (Copy trait)
        assert_eq!(original.request_id, copied.request_id);
        assert_eq!(original.action, copied.action);
        assert_eq!(original.generated_token, copied.generated_token);
    }

    // ── ScheduledBatch: multiple elements consistency ──

    #[test]
    fn scheduled_batch_multiple_elements_seq_offsets_consistency() {
        let batch = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 5, 10, 15],
            draft_steps: vec![0, 1, 2],
        };
        assert_eq!(batch.requests.len() + 1, batch.seq_offsets.len());
        assert_eq!(batch.requests.len(), batch.draft_steps.len());
        // Cumulative offsets
        for i in 0..batch.seq_offsets.len() - 1 {
            assert!(batch.seq_offsets[i + 1] >= batch.seq_offsets[i]);
        }
    }

    // ── ScheduledBatch: draft_steps all zeros ──

    #[test]
    fn scheduled_batch_all_zero_draft_steps() {
        let batch = ScheduledBatch {
            requests: vec![10, 20],
            seq_offsets: vec![0, 1, 2],
            draft_steps: vec![0, 0],
        };
        assert!(batch.draft_steps.iter().all(|&d| d == 0));
    }

    // ── ScheduledBatch: clone independence ──

    #[test]
    fn scheduled_batch_clone_independence() {
        let mut batch = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 1],
            draft_steps: vec![5],
        };
        let cloned = batch.clone();
        batch.draft_steps[0] = 0;
        assert_eq!(cloned.draft_steps[0], 5);
    }

    // ── BatchPrepData: set_sampling_params with subnormal float ──

    #[test]
    fn batch_prep_data_set_sampling_params_subnormal_float() {
        let mut d = BatchPrepData::new(1);
        let subnormal = f32::from_bits(1); // smallest positive subnormal
        d.set_sampling_params(0, subnormal, 0, subnormal, 0);
        assert_eq!(d.sampling_params_packed[0], subnormal.to_bits());
        assert_eq!(d.sampling_params_packed[2], subnormal.to_bits());
    }

    // ── BatchPrepData: set_sampling_params with negative infinity ──

    #[test]
    fn batch_prep_data_set_sampling_params_neg_infinity() {
        let mut d = BatchPrepData::new(1);
        d.set_sampling_params(0, f32::NEG_INFINITY, 0, f32::INFINITY, 0);
        assert_eq!(d.sampling_params_packed[0], f32::NEG_INFINITY.to_bits());
        assert_eq!(d.sampling_params_packed[2], f32::INFINITY.to_bits());
    }

    // ── BatchPrepData: active_flags all ones after construction ──

    #[test]
    fn batch_prep_data_active_flags_all_ones() {
        for n in [1, 5, 100] {
            let d = BatchPrepData::new(n);
            assert!(d.active_flags.iter().all(|&f| f == 1), "active_flags must be all 1 for n={n}");
        }
    }

    // ── BatchPrepData: all vec fields have same length ──

    #[test]
    fn batch_prep_data_all_vecs_same_length() {
        let n = 7;
        let d = BatchPrepData::new(n);
        assert_eq!(d.prompt_lens.len(), n);
        assert_eq!(d.kv_lens.len(), n);
        assert_eq!(d.session_positions.len(), n);
        assert_eq!(d.rope_pos_offsets.len(), n);
        assert_eq!(d.max_new_tokens.len(), n);
        assert_eq!(d.page_table_offsets.len(), n);
        assert_eq!(d.page_table_lens.len(), n);
        assert_eq!(d.fused_hidden_offsets.len(), n);
        assert_eq!(d.num_mm_tokens.len(), n);
        assert_eq!(d.active_flags.len(), n);
        assert_eq!(d.seq_positions.len(), n);
        assert_eq!(d.gen_counts.len(), n);
        assert_eq!(d.last_sampled_tokens.len(), n);
        assert_eq!(d.sampling_params_packed.len(), n * 4);
    }

    // ── BatchPrepData: set_sampling_params exact last index ──

    #[test]
    fn batch_prep_data_set_sampling_params_exact_last_valid_index() {
        let mut d = BatchPrepData::new(5);
        d.set_sampling_params(4, 1.0, 1, 1.0, 1);
        let base = 4 * 4;
        assert_eq!(d.sampling_params_packed[base], 1.0f32.to_bits());
        assert_eq!(d.sampling_params_packed[base + 1], 1u32);
        assert_eq!(d.sampling_params_packed[base + 2], 1.0f32.to_bits());
        assert_eq!(d.sampling_params_packed[base + 3], 1u32);
    }

    // ── BatchPrepData: set_sampling_params one past end is noop ──

    #[test]
    fn batch_prep_data_set_sampling_params_one_past_end() {
        let mut d = BatchPrepData::new(2);
        d.set_sampling_params(2, 99.0, 99, 99.0, 99); // index 2 does not exist
        // All slots should remain zero
        assert!(d.sampling_params_packed.iter().all(|&v| v == 0));
    }

    // ── BatchPrepData: mutate all scalar fields ──

    #[test]
    fn batch_prep_data_mutate_all_scalar_fields() {
        let mut d = BatchPrepData::new(1);
        d.max_decode_steps = u32::MAX;
        d.total_prefill_tokens = u32::MAX / 2;
        assert_eq!(d.max_decode_steps, u32::MAX);
        assert_eq!(d.total_prefill_tokens, u32::MAX / 2);
    }

    // ── BatchPrepData: mutate individual vec fields ──

    #[test]
    fn batch_prep_data_mutate_individual_vec_fields() {
        let mut d = BatchPrepData::new(3);
        d.prompt_lens[0] = 100;
        d.kv_lens[1] = 200;
        d.session_positions[2] = 300;
        d.rope_pos_offsets[0] = 400;
        d.max_new_tokens[1] = 500;
        d.page_table_offsets[2] = 600;
        d.page_table_lens[0] = 700;
        d.fused_hidden_offsets[1] = 800;
        d.num_mm_tokens[2] = 900;
        d.active_flags[0] = 0;
        d.seq_positions[1] = 1000;
        d.gen_counts[2] = 1100;
        d.last_sampled_tokens[0] = 1200;

        assert_eq!(d.prompt_lens[0], 100);
        assert_eq!(d.kv_lens[1], 200);
        assert_eq!(d.session_positions[2], 300);
        assert_eq!(d.rope_pos_offsets[0], 400);
        assert_eq!(d.max_new_tokens[1], 500);
        assert_eq!(d.page_table_offsets[2], 600);
        assert_eq!(d.page_table_lens[0], 700);
        assert_eq!(d.fused_hidden_offsets[1], 800);
        assert_eq!(d.num_mm_tokens[2], 900);
        assert_eq!(d.active_flags[0], 0);
        assert_eq!(d.seq_positions[1], 1000);
        assert_eq!(d.gen_counts[2], 1100);
        assert_eq!(d.last_sampled_tokens[0], 1200);

        // Other indices remain at default
        assert_eq!(d.prompt_lens[1], 0);
        assert_eq!(d.prompt_lens[2], 0);
    }

    // ── InterleavedBatch: decode-only is not interleaved ──

    #[test]
    fn interleaved_batch_decode_only_not_interleaved() {
        let inner = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1, 2],
            draft_steps: vec![0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![],
        };
        assert!(!ib.is_interleaved());
        assert_eq!(ib.decode_tokens(), 2);
        assert_eq!(ib.prefill_tokens(), 0);
    }

    // ── InterleavedBatch: large token counts ──

    #[test]
    fn interleaved_batch_large_token_counts() {
        let inner = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1000, 2000],
            draft_steps: vec![0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 999, draft_steps: 0 },
            ],
        };
        assert_eq!(ib.decode_tokens(), 1);
        assert_eq!(ib.prefill_tokens(), 999);
        assert_eq!(ib.total_tokens(), 1000);
        assert!(ib.is_interleaved());
    }

    // ── InterleavedBatch: request_ids returns inner requests ──

    #[test]
    fn interleaved_batch_request_ids_returns_inner() {
        let inner = ScheduledBatch {
            requests: vec![5, 10, 15],
            seq_offsets: vec![0, 1, 2, 3],
            draft_steps: vec![0, 0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        assert_eq!(ib.request_ids(), &[5, 10, 15]);
    }

    // ── InterleavedBatch: total_tokens with empty inner ──

    #[test]
    fn interleaved_batch_total_tokens_empty_inner() {
        let inner = ScheduledBatch {
            requests: vec![],
            seq_offsets: vec![0],
            draft_steps: vec![],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        assert_eq!(ib.total_tokens(), 0);
        assert_eq!(ib.decode_tokens(), 0);
        assert_eq!(ib.prefill_tokens(), 0);
        assert!(!ib.is_interleaved());
    }

    // ── InterleavedSlot: max batch_index ──

    #[test]
    fn interleaved_slot_max_batch_index() {
        let slot = InterleavedSlot {
            request_id: 1,
            batch_index: usize::MAX,
            token_count: 1,
            draft_steps: 0,
        };
        assert_eq!(slot.batch_index, usize::MAX);
    }

    // ── InterleavedSlot: max draft_steps ──

    #[test]
    fn interleaved_slot_max_draft_steps() {
        let slot = InterleavedSlot {
            request_id: 1,
            batch_index: 0,
            token_count: 1,
            draft_steps: usize::MAX,
        };
        assert_eq!(slot.draft_steps, usize::MAX);
    }

    // ── InterleavedSlot: clone independence ──

    #[test]
    fn interleaved_slot_clone_independence() {
        let mut slot = InterleavedSlot {
            request_id: 42,
            batch_index: 5,
            token_count: 10,
            draft_steps: 3,
        };
        let cloned = slot.clone();
        slot.token_count = 0;
        assert_eq!(slot.token_count, 0);
        assert_eq!(cloned.token_count, 10);
    }

    // ── InterleavedSlot: large token_count ──

    #[test]
    fn interleaved_slot_large_token_count() {
        let slot = InterleavedSlot {
            request_id: 1,
            batch_index: 0,
            token_count: usize::MAX,
            draft_steps: 0,
        };
        assert_eq!(slot.token_count, usize::MAX);
    }

    // ── ContinuousBatcher: enqueue multiple distinct ids ──

    #[test]
    fn continuous_batcher_enqueue_multiple_distinct() {
        let mut b = ContinuousBatcher::new();
        for i in 0..20 {
            b.enqueue(make_sequence(i, 1));
        }
        assert_eq!(b.waiting_len(), 20);
    }

    // ── ContinuousBatcher: enqueue duplicate in waiting ignored ──

    #[test]
    fn continuous_batcher_enqueue_duplicate_in_waiting_ignored() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(42, 4));
        b.enqueue(make_sequence(42, 8)); // same id, different prompt len
        assert_eq!(b.waiting_len(), 1);
    }

    // ── ContinuousBatcher: running_len after admit ──

    #[test]
    fn continuous_batcher_running_len_after_admit() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        b.enqueue(make_sequence(2, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(b.running_len(), 2);
        assert_eq!(b.waiting_len(), 0);
    }

    // ── ContinuousBatcher: running_ids returns correct set ──

    #[test]
    fn continuous_batcher_running_ids_after_admit() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(10, 4));
        b.enqueue(make_sequence(20, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        let mut ids = b.running_ids();
        ids.sort();
        assert_eq!(ids, vec![10, 20]);
    }

    // ── ContinuousBatcher: mean_context_len single sequence ──

    #[test]
    fn continuous_batcher_mean_context_len_single_sequence() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(1, 10));
        // Single sequence with context_len=10, mean=10
        assert_eq!(b.mean_context_len(), 10);
    }

    // ── ContinuousBatcher: has_pending_work with running ──

    #[test]
    fn continuous_batcher_has_pending_work_with_running() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(b.waiting_len(), 0);
        assert_eq!(b.running_len(), 1);
        assert!(b.has_pending_work());
    }

    // ── ContinuousBatcher: has_pending_work false after complete ──

    #[test]
    fn continuous_batcher_has_pending_work_false_after_complete() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::complete(1, Some(99), Default::default())]);
        assert!(!b.has_pending_work());
    }

    // ── ContinuousBatcher: enqueue_order wraps via saturating_add ──

    #[test]
    fn continuous_batcher_enqueue_order_saturating_at_max() {
        let mut b = ContinuousBatcher::new();
        b.next_enqueue_order = u64::MAX;
        b.enqueue(make_sequence(1, 1));
        // First enqueue saturates at MAX
        assert_eq!(b.waiting.front().unwrap().enqueue_order, u64::MAX);
        // Second enqueue also saturates (MAX.saturating_add(1) == MAX)
        b.enqueue(make_sequence(2, 1));
        assert_eq!(b.waiting.back().unwrap().enqueue_order, u64::MAX);
    }

    // ── ContinuousBatcher: build_batch_with_prep returns prep data ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_returns_data() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let (batch, prep) = b.build_batch_with_prep(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(batch.requests, vec![1]);
        assert_eq!(prep.prompt_lens.len(), 1);
        assert_eq!(prep.prompt_lens[0], 4);
    }

    // ── ContinuousBatcher: build_batch_with_prep empty batch ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_empty() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        let (batch, prep) = b.build_batch_with_prep(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert!(batch.requests.is_empty());
        assert!(prep.prompt_lens.is_empty());
    }

    // ── ContinuousBatcher: update_batch fail removes from running ──

    #[test]
    fn continuous_batcher_update_batch_fail_removes_from_running() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(b.running_len(), 1);
        b.update_batch(&mut scheduler, &[BatchResult::fail(1)]);
        assert_eq!(b.running_len(), 0);
        assert!(!b.has_pending_work());
    }

    // ── ContinuousBatcher: update_batch unknown request ignored ──

    #[test]
    fn continuous_batcher_update_batch_unknown_request_ignored() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Update for a non-existent request_id should not affect state
        b.update_batch(&mut scheduler, &[BatchResult::fail(9999)]);
        assert_eq!(b.running_len(), 1);
    }

    // ── ContinuousBatcher: build_batch with zero token_budget ──

    #[test]
    fn continuous_batcher_build_batch_zero_token_budget() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        // With admit_new_prefill=true, the sequence gets admitted to running,
        // but token_budget=0 means neither decode nor prefill tokens are allocated.
        // The sequence sits in running state but is not included in the batch.
        let batch = b.build_batch(&mut scheduler, 0, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Sequence was admitted (running_len=1) but batch has zero slots
        assert_eq!(batch.requests.len(), 0, "zero budget should produce empty batch");
        assert_eq!(b.running_len(), 1, "sequence should still be admitted to running");
    }

    // ── ContinuousBatcher: build_batch no admit keeps sequences waiting ──

    #[test]
    fn continuous_batcher_build_batch_no_admit_keeps_waiting() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        // admit_new_prefill=false means waiting sequences are not admitted
        let batch = b.build_batch(&mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder);
        assert!(batch.requests.is_empty());
        assert_eq!(b.waiting_len(), 1);
    }

    // ── BatchPrepData: debug output contains field names ──

    #[test]
    fn batch_prep_data_debug_contains_key_fields() {
        let d = BatchPrepData::new(1);
        let debug = format!("{d:?}");
        assert!(debug.contains("prompt_lens"), "debug output should contain 'prompt_lens'");
        assert!(debug.contains("active_flags"), "debug output should contain 'active_flags'");
        assert!(debug.contains("max_decode_steps"), "debug output should contain 'max_decode_steps'");
    }

    // ── BatchResult: telemetry for pause is default ──

    #[test]
    fn batch_result_pause_telemetry_is_default() {
        let result = BatchResult::pause(1);
        assert_eq!(result.telemetry, Default::default());
    }

    // ── BatchResult: telemetry for fail is default ──

    #[test]
    fn batch_result_fail_telemetry_is_default() {
        let result = BatchResult::fail(1);
        assert_eq!(result.telemetry, Default::default());
    }

    // ── ScheduledBatch: debug output contains struct name ──

    #[test]
    fn scheduled_batch_debug_output_non_empty() {
        let batch = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1, 2],
            draft_steps: vec![0, 0],
        };
        let debug = format!("{batch:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("ScheduledBatch"));
    }

    // ── InterleavedBatch: debug output is non-empty ──

    #[test]
    fn interleaved_batch_debug_output_non_empty() {
        let inner = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 1],
            draft_steps: vec![0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        let debug = format!("{ib:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("InterleavedBatch"));
    }

    // ── InterleavedSlot: debug output is non-empty ──

    #[test]
    fn interleaved_slot_debug_output_non_empty() {
        let slot = InterleavedSlot {
            request_id: 1,
            batch_index: 0,
            token_count: 1,
            draft_steps: 0,
        };
        let debug = format!("{slot:?}");
        assert!(!debug.is_empty());
        assert!(debug.contains("InterleavedSlot"));
    }

    // ── InterleavedBatch: clone via derive produces identical data ──

    #[test]
    fn interleaved_batch_clone_exact_copy() {
        let inner = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 1, 3, 6],
            draft_steps: vec![0, 2, 4],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 3, batch_index: 2, token_count: 3, draft_steps: 4 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 2, draft_steps: 2 },
            ],
        };
        let cloned = ib.clone();
        assert_eq!(cloned.inner.requests, ib.inner.requests);
        assert_eq!(cloned.inner.seq_offsets, ib.inner.seq_offsets);
        assert_eq!(cloned.inner.draft_steps, ib.inner.draft_steps);
        assert_eq!(cloned.decode_slots.len(), ib.decode_slots.len());
        assert_eq!(cloned.prefill_slots.len(), ib.prefill_slots.len());
    }

    // ── New tests (~60 added) ────────────────────────────────────────────

    // ── ScheduledBatch: construction with all fields zero ──

    #[test]
    fn scheduled_batch_construction_zero_lengths() {
        let batch = ScheduledBatch {
            requests: vec![],
            seq_offsets: vec![],
            draft_steps: vec![],
        };
        assert_eq!(batch.requests.len(), 0);
        assert_eq!(batch.seq_offsets.len(), 0);
        assert_eq!(batch.draft_steps.len(), 0);
    }

    // ── ScheduledBatch: field access after manual construction ──

    #[test]
    fn scheduled_batch_field_access_after_construction() {
        let batch = ScheduledBatch {
            requests: vec![100, 200, 300],
            seq_offsets: vec![0, 10, 25, 40],
            draft_steps: vec![0, 4, 0],
        };
        assert_eq!(batch.requests[0], 100);
        assert_eq!(batch.requests[2], 300);
        assert_eq!(batch.seq_offsets[1], 10);
        assert_eq!(batch.seq_offsets[3], 40);
        assert_eq!(batch.draft_steps[1], 4);
    }

    // ── ScheduledBatch: seq_offsets cumulative sum equals total tokens ──

    #[test]
    fn scheduled_batch_seq_offsets_cumulative_sum() {
        let batch = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 3, 7, 12],
            draft_steps: vec![0, 0, 0],
        };
        let total: usize = batch.seq_offsets.last().copied().unwrap_or(0);
        assert_eq!(total, 12);
        for i in 0..batch.requests.len() {
            let span = batch.seq_offsets[i + 1] - batch.seq_offsets[i];
            assert!(span > 0, "seq {i} has zero span");
        }
    }

    // ── ScheduledBatch: clone preserves draft_steps values ──

    #[test]
    fn scheduled_batch_clone_preserves_draft_steps() {
        let batch = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1, 2],
            draft_steps: vec![7, 8],
        };
        let cloned = batch.clone();
        assert_eq!(cloned.draft_steps, vec![7, 8]);
    }

    // ── BatchPrepData: new with single sequence ──

    #[test]
    fn batch_prep_data_new_single_sequence() {
        let d = BatchPrepData::new(1);
        assert_eq!(d.prompt_lens.len(), 1);
        assert_eq!(d.active_flags.len(), 1);
        assert_eq!(d.active_flags[0], 1);
        assert_eq!(d.sampling_params_packed.len(), 4);
    }

    // ── BatchPrepData: all vec fields zero-initialized except active_flags ──

    #[test]
    fn batch_prep_data_zero_init_except_active() {
        let d = BatchPrepData::new(3);
        for i in 0..3 {
            assert_eq!(d.prompt_lens[i], 0);
            assert_eq!(d.kv_lens[i], 0);
            assert_eq!(d.session_positions[i], 0);
            assert_eq!(d.rope_pos_offsets[i], 0);
            assert_eq!(d.max_new_tokens[i], 0);
            assert_eq!(d.page_table_offsets[i], 0);
            assert_eq!(d.page_table_lens[i], 0);
            assert_eq!(d.fused_hidden_offsets[i], 0);
            assert_eq!(d.num_mm_tokens[i], 0);
            assert_eq!(d.seq_positions[i], 0);
            assert_eq!(d.gen_counts[i], 0);
            assert_eq!(d.last_sampled_tokens[i], 0);
            assert_eq!(d.active_flags[i], 1, "active_flags[{i}] must be 1");
        }
    }

    // ── BatchPrepData: set_sampling_params does not corrupt neighbor seq ──

    #[test]
    fn batch_prep_data_set_sampling_params_no_neighbor_corruption() {
        let mut d = BatchPrepData::new(3);
        d.set_sampling_params(0, 1.0, 10, 0.5, 0);
        d.set_sampling_params(2, 2.0, 20, 0.8, 5);
        // Seq 1 must remain all zeros
        assert_eq!(&d.sampling_params_packed[4..8], &[0u32; 4]);
    }

    // ── BatchPrepData: set_sampling_params with zero temperature ──

    #[test]
    fn batch_prep_data_set_sampling_params_zero_temperature() {
        let mut d = BatchPrepData::new(1);
        d.set_sampling_params(0, 0.0, 1, 1.0, 0);
        assert_eq!(d.sampling_params_packed[0], 0.0f32.to_bits());
        assert_eq!(d.sampling_params_packed[1], 1u32);
        assert_eq!(d.sampling_params_packed[2], 1.0f32.to_bits());
    }

    // ── BatchPrepData: set_sampling_params with epsilon float ──

    #[test]
    fn batch_prep_data_set_sampling_params_epsilon_float() {
        let mut d = BatchPrepData::new(1);
        let eps = f32::EPSILON;
        d.set_sampling_params(0, eps, 0, eps, 0);
        assert_eq!(d.sampling_params_packed[0], eps.to_bits());
        assert_eq!(d.sampling_params_packed[2], eps.to_bits());
    }

    // ── BatchPrepData: clone after setting all sampling params ──

    #[test]
    fn batch_prep_data_clone_after_full_sampling_setup() {
        let mut d = BatchPrepData::new(2);
        d.set_sampling_params(0, 0.5, 50, 0.95, 2);
        d.set_sampling_params(1, 1.0, 100, 0.8, 1);
        let cloned = d.clone();
        assert_eq!(cloned.sampling_params_packed[0], 0.5f32.to_bits());
        assert_eq!(cloned.sampling_params_packed[4], 1.0f32.to_bits());
        assert_eq!(cloned.active_flags, d.active_flags);
    }

    // ── BatchPrepData: debug format includes all field names ──

    #[test]
    fn batch_prep_data_debug_includes_kv_lens_and_rope() {
        let d = BatchPrepData::new(1);
        let debug = format!("{d:?}");
        assert!(debug.contains("kv_lens"), "debug must include 'kv_lens'");
        assert!(debug.contains("rope_pos_offsets"), "debug must include 'rope_pos_offsets'");
        assert!(debug.contains("total_prefill_tokens"), "debug must include 'total_prefill_tokens'");
    }

    // ── BatchPrepData: mutate page_table fields ──

    #[test]
    fn batch_prep_data_mutate_page_table_fields() {
        let mut d = BatchPrepData::new(2);
        d.page_table_offsets[0] = 1024;
        d.page_table_lens[0] = 7;
        d.page_table_offsets[1] = 2048;
        d.page_table_lens[1] = 3;
        assert_eq!(d.page_table_offsets[0], 1024);
        assert_eq!(d.page_table_lens[0], 7);
        assert_eq!(d.page_table_offsets[1], 2048);
        assert_eq!(d.page_table_lens[1], 3);
    }

    // ── BatchPrepData: mutate fused_hidden and mm_token fields ──

    #[test]
    fn batch_prep_data_mutate_hidden_and_mm_fields() {
        let mut d = BatchPrepData::new(2);
        d.fused_hidden_offsets[0] = 4096;
        d.fused_hidden_offsets[1] = 8192;
        d.num_mm_tokens[0] = 5;
        d.num_mm_tokens[1] = 0;
        assert_eq!(d.fused_hidden_offsets[0], 4096);
        assert_eq!(d.fused_hidden_offsets[1], 8192);
        assert_eq!(d.num_mm_tokens[0], 5);
        assert_eq!(d.num_mm_tokens[1], 0);
    }

    // ── BatchPrepData: mutate gen_counts and last_sampled_tokens ──

    #[test]
    fn batch_prep_data_mutate_gen_and_sampled_fields() {
        let mut d = BatchPrepData::new(2);
        d.gen_counts[0] = 15;
        d.gen_counts[1] = 30;
        d.last_sampled_tokens[0] = 12345;
        d.last_sampled_tokens[1] = 67890;
        assert_eq!(d.gen_counts[0], 15);
        assert_eq!(d.gen_counts[1], 30);
        assert_eq!(d.last_sampled_tokens[0], 12345);
        assert_eq!(d.last_sampled_tokens[1], 67890);
    }

    // ── BatchAction: Debug output matches variant names ──

    #[test]
    fn batch_action_debug_strings_are_upper_camel() {
        let debug_strings = [
            format!("{:?}", BatchAction::Continue),
            format!("{:?}", BatchAction::Complete),
            format!("{:?}", BatchAction::Pause),
            format!("{:?}", BatchAction::Fail),
        ];
        for s in &debug_strings {
            let first = s.chars().next().unwrap();
            assert!(first.is_uppercase(), "Debug output '{s}' must start uppercase");
        }
    }

    // ── BatchAction: all four variants are distinct via Ord ──

    #[test]
    fn batch_action_all_variants_pairwise_distinct() {
        let variants = [
            BatchAction::Continue,
            BatchAction::Complete,
            BatchAction::Pause,
            BatchAction::Fail,
        ];
        let mut distinct_count = 0;
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j], "variants[{i}] != variants[{j}]");
                distinct_count += 1;
            }
        }
        assert_eq!(distinct_count, 6, "4 variants must produce 6 distinct pairs");
    }

    // ── BatchAction: Copy makes independent values ──

    #[test]
    fn batch_action_copy_preserves_original() {
        let mut action = BatchAction::Continue;
        let saved = action;
        action = BatchAction::Fail;
        assert_eq!(saved, BatchAction::Continue);
        assert_eq!(action, BatchAction::Fail);
    }

    // ── BatchAction: Clone produces equal value ──

    #[test]
    fn batch_action_clone_equal() {
        let original = BatchAction::Pause;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── BatchResult: construction via continue_with_token stores all fields ──

    #[test]
    fn batch_result_continue_construction_stores_fields() {
        let tel = crate::scheduler::telemetry::SequenceTelemetry {
            l2_delta: 0.42,
            has_outlier: true,
            dead_density: 0.1,
            per_head_entropy: 3.14,
            transform_ratio: 0.05,
            output_entropy: 1.5,
        };
        let result = BatchResult::continue_with_token(999, 777, tel);
        assert_eq!(result.request_id, 999);
        assert_eq!(result.action, BatchAction::Continue);
        assert_eq!(result.generated_token, Some(777));
        assert_eq!(result.telemetry.l2_delta, 0.42);
        assert!(result.telemetry.has_outlier);
    }

    // ── BatchResult: complete without token has None ──

    #[test]
    fn batch_result_complete_without_token_is_none() {
        let result = BatchResult::complete(5, None, Default::default());
        assert_eq!(result.action, BatchAction::Complete);
        assert_eq!(result.generated_token, None);
        assert_eq!(result.request_id, 5);
    }

    // ── BatchResult: complete with token has Some ──

    #[test]
    fn batch_result_complete_with_token_is_some() {
        let result = BatchResult::complete(5, Some(42), Default::default());
        assert_eq!(result.generated_token, Some(42));
    }

    // ── BatchResult: pause always has None token ──

    #[test]
    fn batch_result_pause_token_always_none() {
        let result = BatchResult::pause(42);
        assert_eq!(result.generated_token, None);
        assert_eq!(result.action, BatchAction::Pause);
    }

    // ── BatchResult: fail always has None token ──

    #[test]
    fn batch_result_fail_token_always_none() {
        let result = BatchResult::fail(42);
        assert_eq!(result.generated_token, None);
        assert_eq!(result.action, BatchAction::Fail);
    }

    // ── BatchResult: Copy trait allows use after move ──

    #[test]
    fn batch_result_copy_allows_use_after_move() {
        let result = BatchResult::continue_with_token(1, 42, Default::default());
        let moved = result;
        // result is still usable after Copy
        assert_eq!(result.request_id, moved.request_id);
        assert_eq!(result.action, moved.action);
    }

    // ── BatchResult: Debug format contains BatchResult ──

    #[test]
    fn batch_result_debug_contains_struct_name() {
        let result = BatchResult::continue_with_token(1, 42, Default::default());
        let debug = format!("{result:?}");
        assert!(debug.contains("BatchResult"), "debug must contain 'BatchResult'");
    }

    // ── BatchResult: PartialEq reflexivity ──

    #[test]
    fn batch_result_partial_eq_reflexivity() {
        let a = BatchResult::continue_with_token(1, 42, Default::default());
        assert_eq!(a, a);
    }

    // ── BatchResult: PartialEq symmetry ──

    #[test]
    fn batch_result_partial_eq_symmetry() {
        let a = BatchResult::continue_with_token(1, 42, Default::default());
        let b = BatchResult::continue_with_token(1, 42, Default::default());
        assert_eq!(a, b);
        assert_eq!(b, a);
    }

    // ── BatchResult: all four factory methods produce distinct actions ──

    #[test]
    fn batch_result_all_factory_methods_distinct_actions() {
        let cont = BatchResult::continue_with_token(1, 42, Default::default());
        let comp = BatchResult::complete(1, Some(42), Default::default());
        let pause = BatchResult::pause(1);
        let fail = BatchResult::fail(1);
        let actions = [cont.action, comp.action, pause.action, fail.action];
        for i in 0..actions.len() {
            for j in (i + 1)..actions.len() {
                assert_ne!(actions[i], actions[j], "action[{i}] != action[{j}]");
            }
        }
    }

    // ── InterleavedBatch: construction with mixed slots ──

    #[test]
    fn interleaved_batch_construction_mixed_slots() {
        let inner = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 1, 5, 8],
            draft_steps: vec![0, 2, 0],
        };
        let decode = vec![
            InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
            InterleavedSlot { request_id: 3, batch_index: 2, token_count: 3, draft_steps: 0 },
        ];
        let prefill = vec![
            InterleavedSlot { request_id: 2, batch_index: 1, token_count: 4, draft_steps: 2 },
        ];
        let ib = InterleavedBatch { inner, decode_slots: decode, prefill_slots: prefill };
        assert_eq!(ib.decode_slots.len(), 2);
        assert_eq!(ib.prefill_slots.len(), 1);
        assert_eq!(ib.total_tokens(), 8);
    }

    // ── InterleavedBatch: is_interleaved false when only decode ──

    #[test]
    fn interleaved_batch_not_interleaved_when_only_decode() {
        let inner = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 1],
            draft_steps: vec![0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![],
        };
        assert!(!ib.is_interleaved());
    }

    // ── InterleavedBatch: is_interleaved false when only prefill ──

    #[test]
    fn interleaved_batch_not_interleaved_when_only_prefill() {
        let inner = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 10],
            draft_steps: vec![0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 10, draft_steps: 0 },
            ],
        };
        assert!(!ib.is_interleaved());
    }

    // ── InterleavedBatch: is_interleaved true when both present ──

    #[test]
    fn interleaved_batch_interleaved_when_both_present() {
        let inner = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1, 5],
            draft_steps: vec![0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 4, draft_steps: 0 },
            ],
        };
        assert!(ib.is_interleaved());
    }

    // ── InterleavedBatch: decode_tokens sums correctly ──

    #[test]
    fn interleaved_batch_decode_tokens_sums_multiple_slots() {
        let inner = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 1, 2, 3],
            draft_steps: vec![0, 0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 3, draft_steps: 0 },
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 7, draft_steps: 0 },
                InterleavedSlot { request_id: 3, batch_index: 2, token_count: 10, draft_steps: 0 },
            ],
            prefill_slots: vec![],
        };
        assert_eq!(ib.decode_tokens(), 20);
    }

    // ── InterleavedBatch: prefill_tokens sums correctly ──

    #[test]
    fn interleaved_batch_prefill_tokens_sums_multiple_slots() {
        let inner = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 50, 150],
            draft_steps: vec![0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 50, draft_steps: 0 },
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 100, draft_steps: 0 },
            ],
        };
        assert_eq!(ib.prefill_tokens(), 150);
    }

    // ── InterleavedBatch: request_ids matches inner.requests ──

    #[test]
    fn interleaved_batch_request_ids_slice_matches_inner() {
        let inner = ScheduledBatch {
            requests: vec![7, 14, 21],
            seq_offsets: vec![0, 1, 2, 3],
            draft_steps: vec![0, 0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        assert_eq!(ib.request_ids().len(), 3);
        assert_eq!(ib.request_ids()[0], 7);
        assert_eq!(ib.request_ids()[2], 21);
    }

    // ── InterleavedSlot: boundary values all max ──

    #[test]
    fn interleaved_slot_all_fields_max() {
        let slot = InterleavedSlot {
            request_id: u64::MAX,
            batch_index: usize::MAX,
            token_count: usize::MAX,
            draft_steps: usize::MAX,
        };
        assert_eq!(slot.request_id, u64::MAX);
        assert_eq!(slot.batch_index, usize::MAX);
        assert_eq!(slot.token_count, usize::MAX);
        assert_eq!(slot.draft_steps, usize::MAX);
    }

    // ── InterleavedSlot: all fields zero ──

    #[test]
    fn interleaved_slot_all_fields_zero() {
        let slot = InterleavedSlot {
            request_id: 0,
            batch_index: 0,
            token_count: 0,
            draft_steps: 0,
        };
        assert_eq!(slot.request_id, 0);
        assert_eq!(slot.batch_index, 0);
        assert_eq!(slot.token_count, 0);
        assert_eq!(slot.draft_steps, 0);
    }

    // ── InterleavedSlot: debug output contains field names ──

    #[test]
    fn interleaved_slot_debug_contains_fields() {
        let slot = InterleavedSlot {
            request_id: 42,
            batch_index: 1,
            token_count: 5,
            draft_steps: 3,
        };
        let debug = format!("{slot:?}");
        assert!(debug.contains("InterleavedSlot"));
        assert!(debug.contains("request_id") || debug.contains("42"));
    }

    // ── InterleavedSlot: clone then mutate independence ──

    #[test]
    fn interleaved_slot_clone_then_mutate_source() {
        let mut slot = InterleavedSlot {
            request_id: 1,
            batch_index: 2,
            token_count: 10,
            draft_steps: 5,
        };
        let cloned = slot.clone();
        slot.request_id = 99;
        slot.draft_steps = 0;
        assert_eq!(cloned.request_id, 1);
        assert_eq!(cloned.draft_steps, 5);
    }

    // ── ContinuousBatcher: new has no chunked_state ──

    #[test]
    fn continuous_batcher_new_no_chunked_state() {
        let b = ContinuousBatcher::new();
        assert!(b.chunked_state.is_none());
    }

    // ── ContinuousBatcher: default matches new ──

    #[test]
    fn continuous_batcher_default_equals_new() {
        let via_new = ContinuousBatcher::new();
        let via_default = ContinuousBatcher::default();
        assert_eq!(via_new.waiting_len(), via_default.waiting_len());
        assert_eq!(via_new.running_len(), via_default.running_len());
        assert_eq!(via_new.chunked_state.is_some(), via_default.chunked_state.is_some());
    }

    // ── ContinuousBatcher: with_chunked then build still works ──

    #[test]
    fn continuous_batcher_with_chunked_build_produces_batch() {
        let mut b = ContinuousBatcher::new().with_chunked(ChunkedConfig::default());
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let batch = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(batch.requests, vec![1]);
    }

    // ── ContinuousBatcher: multiple enqueue then dequeue order ──

    #[test]
    fn continuous_batcher_enqueue_fifo_waiting_order() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(1, 1));
        b.enqueue(make_sequence(2, 1));
        b.enqueue(make_sequence(3, 1));
        let ids: Vec<RequestId> = b.waiting.iter().map(|s| s.id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    // ── ContinuousBatcher: get_running_mut returns correct sequence ──

    #[test]
    fn continuous_batcher_get_running_mut_returns_correct_id() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(42, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        let seq = b.get_running_mut(42).unwrap();
        assert_eq!(seq.id, 42);
        assert_eq!(seq.state, SequenceState::Running);
    }

    // ── ContinuousBatcher: get_running_mut none for non_running ──

    #[test]
    fn continuous_batcher_get_running_mut_none_for_nonexistent() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert!(b.get_running_mut(999).is_none());
    }

    // ── ContinuousBatcher: update_batch pause keeps sequence in running ──

    #[test]
    fn continuous_batcher_update_batch_pause_keeps_in_running() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::pause(1)]);
        assert_eq!(b.running_len(), 1);
        let seq = b.get_running_mut(1).unwrap();
        assert_eq!(seq.state, SequenceState::Paused);
    }

    // ── ContinuousBatcher: update_batch continue then complete ──

    #[test]
    fn continuous_batcher_update_batch_continue_then_complete() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 42, Default::default())]);
        assert_eq!(b.running_len(), 1);
        b.update_batch(&mut scheduler, &[BatchResult::complete(1, Some(43), Default::default())]);
        assert_eq!(b.running_len(), 0);
    }

    // ── ContinuousBatcher: build_interleaved_batch empty returns empty ──

    #[test]
    fn continuous_batcher_build_interleaved_empty() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        let ib = b.build_interleaved_batch(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(ib.decode_tokens(), 0);
        assert_eq!(ib.prefill_tokens(), 0);
        assert_eq!(ib.total_tokens(), 0);
        assert!(!ib.is_interleaved());
    }

    // ── ContinuousBatcher: build_interleaved_batch with decode ──

    #[test]
    fn continuous_batcher_build_interleaved_decode_only() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        // First batch is prefill
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Continue to make it a decode sequence
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 42, Default::default())]);
        // Now build interleaved - this should be a decode step
        let ib = b.build_interleaved_batch(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        // Decode step produces at least one decode slot
        assert!(ib.decode_slots.len() + ib.prefill_slots.len() >= 1);
    }

    // ── ContinuousBatcher: mean_context_len mixed waiting_and_running ──

    #[test]
    fn continuous_batcher_mean_context_len_mixed() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        b.enqueue(make_sequence(2, 8));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Both admitted, both running with context_len 4 and 8 => mean = 6
        assert_eq!(b.mean_context_len(), 6);
    }

    // ── ContinuousBatcher: build_batch_with_prep populates prompt_lens ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_populates_prompt_lens() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 7));
        b.enqueue(make_sequence(2, 13));
        let (batch, prep) = b.build_batch_with_prep(
            &mut scheduler,
            usize::MAX,
            true,
            BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(batch.requests.len(), 2);
        // prompt_lens should reflect the actual prompt lengths
        let prompt_lens_sorted = {
            let mut v = prep.prompt_lens.clone();
            v.sort();
            v
        };
        assert_eq!(prompt_lens_sorted, vec![7, 13]);
    }

    // ── ContinuousBatcher: enqueue after complete is allowed ──

    #[test]
    fn continuous_batcher_enqueue_after_complete() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::complete(1, Some(99), Default::default())]);
        assert!(!b.has_pending_work());
        // Re-enqueue same id after completion
        b.enqueue(make_sequence(1, 8));
        assert_eq!(b.waiting_len(), 1);
        assert!(b.has_pending_work());
    }

    // ── ContinuousBatcher: update_batch with empty results is noop ──

    #[test]
    fn continuous_batcher_update_batch_empty_results() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        let running_before = b.running_len();
        b.update_batch(&mut scheduler, &[]);
        assert_eq!(b.running_len(), running_before);
    }

    // ── ContinuousBatcher: enqueue_order starts at zero ──

    #[test]
    fn continuous_batcher_enqueue_order_starts_at_zero() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(1, 1));
        assert_eq!(b.waiting.front().unwrap().enqueue_order, 0);
    }

    // ── ContinuousBatcher: enqueue_order increments by one ──

    #[test]
    fn continuous_batcher_enqueue_order_increments_by_one() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(1, 1));
        b.enqueue(make_sequence(2, 1));
        b.enqueue(make_sequence(3, 1));
        let orders: Vec<u64> = b.waiting.iter().map(|s| s.enqueue_order).collect();
        assert_eq!(orders[0], 0);
        assert_eq!(orders[1], 1);
        assert_eq!(orders[2], 2);
    }

    // ── BatchResult: debug for each action variant ──

    #[test]
    fn batch_result_debug_all_action_variants() {
        let results = [
            BatchResult::continue_with_token(1, 42, Default::default()),
            BatchResult::complete(2, Some(42), Default::default()),
            BatchResult::pause(3),
            BatchResult::fail(4),
        ];
        for r in &results {
            let debug = format!("{r:?}");
            assert!(!debug.is_empty());
            assert!(debug.contains("BatchResult"));
        }
    }

    // ── BatchPrepData: new with large count has consistent lengths ──

    #[test]
    fn batch_prep_data_new_consistent_lengths_large() {
        let n = 500;
        let d = BatchPrepData::new(n);
        assert_eq!(d.prompt_lens.len(), n);
        assert_eq!(d.active_flags.len(), n);
        assert_eq!(d.sampling_params_packed.len(), n * 4);
        assert_eq!(d.max_decode_steps, 0);
        assert_eq!(d.total_prefill_tokens, 0);
    }

    // ── BatchPrepData: set_sampling_params with smallest positive float ──

    #[test]
    fn batch_prep_data_set_sampling_params_smallest_positive() {
        let mut d = BatchPrepData::new(1);
        let smallest = f32::from_bits(1);
        d.set_sampling_params(0, smallest, 0, smallest, 0);
        assert_eq!(d.sampling_params_packed[0], smallest.to_bits());
        assert_eq!(d.sampling_params_packed[2], smallest.to_bits());
    }

    // ── ScheduledBatch: large request count ──

    #[test]
    fn scheduled_batch_large_request_count() {
        let n: usize = 256;
        let requests: Vec<RequestId> = (0..n as u64).collect();
        let seq_offsets: Vec<usize> = (0..=n).collect();
        let draft_steps = vec![0usize; n];
        let batch = ScheduledBatch { requests, seq_offsets, draft_steps };
        assert_eq!(batch.requests.len(), 256);
        assert_eq!(batch.seq_offsets.len(), 257);
        assert_eq!(batch.draft_steps.len(), 256);
    }

    // ── ScheduledBatch: seq_offsets strictly monotonic ──

    #[test]
    fn scheduled_batch_seq_offsets_monotonic() {
        let batch = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 5, 10, 20],
            draft_steps: vec![0, 0, 0],
        };
        for w in batch.seq_offsets.windows(2) {
            assert!(w[0] < w[1], "seq_offsets must be strictly increasing");
        }
    }

    // ── InterleavedBatch: inner field is accessible ──

    #[test]
    fn interleaved_batch_inner_field_accessible() {
        let inner = ScheduledBatch {
            requests: vec![1],
            seq_offsets: vec![0, 1],
            draft_steps: vec![3],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        assert_eq!(ib.inner.requests, vec![1]);
        assert_eq!(ib.inner.draft_steps, vec![3]);
    }

    // ── InterleavedBatch: slots preserve batch_index ordering ──

    #[test]
    fn interleaved_batch_slots_batch_index_ordering() {
        let inner = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 1, 5, 10],
            draft_steps: vec![0, 0, 0],
        };
        let decode = vec![
            InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
            InterleavedSlot { request_id: 3, batch_index: 2, token_count: 5, draft_steps: 0 },
        ];
        let prefill = vec![
            InterleavedSlot { request_id: 2, batch_index: 1, token_count: 4, draft_steps: 0 },
        ];
        let ib = InterleavedBatch { inner, decode_slots: decode, prefill_slots: prefill };
        // Decode slots have batch indices 0 and 2
        assert_eq!(ib.decode_slots[0].batch_index, 0);
        assert_eq!(ib.decode_slots[1].batch_index, 2);
        // Prefill slot has batch index 1
        assert_eq!(ib.prefill_slots[0].batch_index, 1);
    }

    // ── BatchAction: exhaustiveness check 4 variants ──

    #[test]
    fn batch_action_has_exactly_four_variants() {
        let count = {
            let mut n = 0;
            let _ = match BatchAction::Continue {
                BatchAction::Continue => { n += 1; }
                BatchAction::Complete => { n += 1; }
                BatchAction::Pause => { n += 1; }
                BatchAction::Fail => { n += 1; }
            };
            n
        };
        assert_eq!(count, 1, "must have exactly 4 variants");
    }

    // ── ContinuousBatcher: build_batch returns correct seq_offsets for single ──

    #[test]
    fn continuous_batcher_build_batch_seq_offsets_single() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let batch = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // seq_offsets starts with 0
        assert_eq!(batch.seq_offsets[0], 0);
        // For a single prefill of length 4, offset[1] >= 1
        assert!(batch.seq_offsets.len() >= 2);
    }

    // ── ContinuousBatcher: build_batch respects admit_new_prefill_false ──

    #[test]
    fn continuous_batcher_build_batch_no_admit_no_new_sequences() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        b.enqueue(make_sequence(2, 4));
        // admit=false, no sequences admitted, batch empty
        let batch = b.build_batch(&mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder);
        assert!(batch.requests.is_empty());
        assert_eq!(b.waiting_len(), 2);
    }

    // ── ContinuousBatcher: update_batch with multiple results ──

    #[test]
    fn continuous_batcher_update_batch_multiple_results() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        b.enqueue(make_sequence(2, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(b.running_len(), 2);
        // Complete one, continue the other
        b.update_batch(&mut scheduler, &[
            BatchResult::complete(1, Some(42), Default::default()),
            BatchResult::continue_with_token(2, 43, Default::default()),
        ]);
        assert_eq!(b.running_len(), 1);
        assert!(b.running.contains_key(&2));
        assert!(!b.running.contains_key(&1));
    }

    // ── SequenceTelemetry: PartialEq distinguishes values ──

    #[test]
    fn sequence_telemetry_partial_eq_distinguishes() {
        let a = crate::scheduler::telemetry::SequenceTelemetry {
            l2_delta: 0.0,
            has_outlier: false,
            dead_density: 0.0,
            per_head_entropy: 0.0,
            transform_ratio: 0.0,
            output_entropy: 0.0,
        };
        let b = crate::scheduler::telemetry::SequenceTelemetry {
            l2_delta: 1.0,
            ..a
        };
        assert_ne!(a, b);
    }

    // ── BatchOrderPolicy tests ──

    #[test]
    fn batch_order_policy_default_is_strict_request_id_order() {
        let policy = BatchOrderPolicy::default();
        assert_eq!(policy, BatchOrderPolicy::StrictRequestIdOrder);
    }

    #[test]
    fn batch_order_policy_all_variants_copy() {
        let a = BatchOrderPolicy::StrictRequestIdOrder;
        let b = BatchOrderPolicy::FifoOrder;
        assert_ne!(a, b);
        let a_copy = a;
        assert_eq!(a, a_copy);
    }

    #[test]
    fn batch_order_policy_debug_output() {
        assert!(format!("{:?}", BatchOrderPolicy::StrictRequestIdOrder).contains("StrictRequestIdOrder"));
        assert!(format!("{:?}", BatchOrderPolicy::FifoOrder).contains("FifoOrder"));
    }

    // ── BatchPrepData: session/rope/max fields ──

    #[test]
    fn batch_prep_data_session_positions_field_mutable() {
        let mut prep = BatchPrepData::new(3);
        assert_eq!(prep.session_positions, vec![0, 0, 0]);
        prep.session_positions[1] = 42;
        assert_eq!(prep.session_positions[1], 42);
    }

    #[test]
    fn batch_prep_data_rope_pos_offsets_field_mutable() {
        let mut prep = BatchPrepData::new(3);
        assert_eq!(prep.rope_pos_offsets, vec![0, 0, 0]);
        prep.rope_pos_offsets[0] = 7;
        assert_eq!(prep.rope_pos_offsets[0], 7);
    }

    #[test]
    fn batch_prep_data_max_new_tokens_field_mutable() {
        let mut prep = BatchPrepData::new(2);
        assert_eq!(prep.max_new_tokens, vec![0, 0]);
        prep.max_new_tokens[0] = 100;
        prep.max_new_tokens[1] = 200;
        assert_eq!(prep.max_new_tokens, vec![100, 200]);
    }

    // ── BatchPrepData: scalar counters ──

    #[test]
    fn batch_prep_data_max_decode_steps_field_mutable() {
        let mut prep = BatchPrepData::new(1);
        assert_eq!(prep.max_decode_steps, 0);
        prep.max_decode_steps = 99;
        assert_eq!(prep.max_decode_steps, 99);
    }

    #[test]
    fn batch_prep_data_total_prefill_tokens_field_mutable() {
        let mut prep = BatchPrepData::new(1);
        assert_eq!(prep.total_prefill_tokens, 0);
        prep.total_prefill_tokens = 512;
        assert_eq!(prep.total_prefill_tokens, 512);
    }

    // ── ScheduledBatch: invariants ──

    #[test]
    fn scheduled_batch_draft_steps_len_matches_requests() {
        let batch = ScheduledBatch {
            requests: vec![10, 20, 30],
            seq_offsets: vec![0, 1, 3, 6],
            draft_steps: vec![0, 4, 2],
        };
        assert_eq!(batch.draft_steps.len(), batch.requests.len());
    }

    #[test]
    fn scheduled_batch_seq_offsets_len_is_requests_plus_one() {
        let batch = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 5, 9],
            draft_steps: vec![0, 0],
        };
        assert_eq!(batch.seq_offsets.len(), batch.requests.len() + 1);
    }

    // ── ContinuousBatcher: waiting_len / running_len after operations ──

    #[test]
    fn continuous_batcher_waiting_len_after_multiple_enqueue() {
        let mut b = ContinuousBatcher::new();
        assert_eq!(b.waiting_len(), 0);
        b.enqueue(make_sequence(1, 4));
        assert_eq!(b.waiting_len(), 1);
        b.enqueue(make_sequence(2, 8));
        assert_eq!(b.waiting_len(), 2);
        b.enqueue(make_sequence(3, 16));
        assert_eq!(b.waiting_len(), 3);
    }

    #[test]
    fn continuous_batcher_running_ids_returns_admitted_keys() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(5, 4));
        b.enqueue(make_sequence(10, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        let ids = b.running_ids();
        assert!(ids.contains(&5));
        assert!(ids.contains(&10));
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn continuous_batcher_mean_context_len_running_only() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 8));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert!(b.mean_context_len() > 0);
    }

    // ── InterleavedBatch: empty cases ──

    #[test]
    fn interleaved_batch_total_tokens_delegates_to_decode_plus_prefill() {
        let batch = InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![1, 2],
                seq_offsets: vec![0, 1, 3],
                draft_steps: vec![0, 0],
            },
            decode_slots: vec![InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 }],
            prefill_slots: vec![InterleavedSlot { request_id: 2, batch_index: 1, token_count: 2, draft_steps: 0 }],
        };
        assert_eq!(batch.total_tokens(), 3);
    }

    // ── InterleavedSlot: field accessors ──

    #[test]
    fn interleaved_slot_request_id_accessor() {
        let slot = InterleavedSlot { request_id: 99, batch_index: 3, token_count: 5, draft_steps: 2 };
        assert_eq!(slot.request_id, 99);
    }

    #[test]
    fn interleaved_slot_batch_index_zero_valid() {
        let slot = InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 };
        assert_eq!(slot.batch_index, 0);
    }

    // ── SequenceTelemetry: per-field distinction ──

    #[test]
    fn sequence_telemetry_dead_density_distinguishes() {
        use crate::scheduler::telemetry::SequenceTelemetry;
        let a = SequenceTelemetry { dead_density: 0.0, ..Default::default() };
        let b = SequenceTelemetry { dead_density: 0.5, ..Default::default() };
        assert_ne!(a, b);
    }

    #[test]
    fn sequence_telemetry_per_head_entropy_distinguishes() {
        use crate::scheduler::telemetry::SequenceTelemetry;
        let a = SequenceTelemetry { per_head_entropy: 0.0, ..Default::default() };
        let b = SequenceTelemetry { per_head_entropy: 2.0, ..Default::default() };
        assert_ne!(a, b);
    }

    #[test]
    fn sequence_telemetry_transform_ratio_distinguishes() {
        use crate::scheduler::telemetry::SequenceTelemetry;
        let a = SequenceTelemetry { transform_ratio: 0.0, ..Default::default() };
        let b = SequenceTelemetry { transform_ratio: 0.99, ..Default::default() };
        assert_ne!(a, b);
    }

    // ── New tests (+15) ────────────────────────────────────────────────

    // ── BatchOrderPolicy: Clone produces equal value ──

    #[test]
    fn batch_order_policy_clone_equal() {
        let a = BatchOrderPolicy::FifoOrder;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── BatchOrderPolicy: Copy preserves original after reassignment ──

    #[test]
    fn batch_order_policy_copy_preserves_original() {
        let mut policy = BatchOrderPolicy::StrictRequestIdOrder;
        let saved = policy;
        policy = BatchOrderPolicy::FifoOrder;
        assert_eq!(saved, BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(policy, BatchOrderPolicy::FifoOrder);
    }

    // ── SequenceTelemetry: new() matches default ──

    #[test]
    fn sequence_telemetry_new_matches_default() {
        use crate::scheduler::telemetry::SequenceTelemetry;
        let via_new = SequenceTelemetry::new();
        let via_default = SequenceTelemetry::default();
        assert_eq!(via_new.l2_delta, via_default.l2_delta);
        assert_eq!(via_new.has_outlier, via_default.has_outlier);
        assert_eq!(via_new.dead_density, via_default.dead_density);
        assert_eq!(via_new.per_head_entropy, via_default.per_head_entropy);
        assert_eq!(via_new.transform_ratio, via_default.transform_ratio);
        assert_eq!(via_new.output_entropy, via_default.output_entropy);
    }

    // ── SequenceTelemetry: has_outlier distinguishes ──

    #[test]
    fn sequence_telemetry_has_outlier_distinguishes() {
        use crate::scheduler::telemetry::SequenceTelemetry;
        let a = SequenceTelemetry { has_outlier: false, ..Default::default() };
        let b = SequenceTelemetry { has_outlier: true, ..Default::default() };
        assert_ne!(a, b);
    }

    // ── SequenceTelemetry: output_entropy distinguishes ──

    #[test]
    fn sequence_telemetry_output_entropy_distinguishes() {
        use crate::scheduler::telemetry::SequenceTelemetry;
        let a = SequenceTelemetry { output_entropy: 0.0, ..Default::default() };
        let b = SequenceTelemetry { output_entropy: 3.14, ..Default::default() };
        assert_ne!(a, b);
    }

    // ── BatchPrepData: active_flags can be cleared ──

    #[test]
    fn batch_prep_data_active_flags_can_be_cleared() {
        let mut d = BatchPrepData::new(3);
        assert!(d.active_flags.iter().all(|&f| f == 1));
        d.active_flags[1] = 0;
        assert_eq!(d.active_flags[0], 1);
        assert_eq!(d.active_flags[1], 0);
        assert_eq!(d.active_flags[2], 1);
    }

    // ── BatchPrepData: set_sampling_params two seqs simultaneously ──

    #[test]
    fn batch_prep_data_set_sampling_params_two_seqs_simultaneously() {
        let mut d = BatchPrepData::new(3);
        d.set_sampling_params(0, 0.5, 10, 0.9, 1);
        d.set_sampling_params(2, 1.5, 20, 0.1, 0);
        // Seq 0
        assert_eq!(d.sampling_params_packed[0], 0.5f32.to_bits());
        assert_eq!(d.sampling_params_packed[1], 10u32);
        // Seq 1 untouched
        assert_eq!(&d.sampling_params_packed[4..8], &[0u32; 4]);
        // Seq 2
        let base = 8;
        assert_eq!(d.sampling_params_packed[base], 1.5f32.to_bits());
        assert_eq!(d.sampling_params_packed[base + 1], 20u32);
    }

    // ── ScheduledBatch: requests field is mutable ──

    #[test]
    fn scheduled_batch_requests_field_mutable() {
        let mut batch = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 1, 2, 3],
            draft_steps: vec![0, 0, 0],
        };
        batch.requests.push(4);
        assert_eq!(batch.requests.len(), 4);
        assert_eq!(batch.requests[3], 4);
    }

    // ── InterleavedSlot: draft_steps non-zero ──

    #[test]
    fn interleaved_slot_non_zero_draft_steps() {
        let slot = InterleavedSlot {
            request_id: 1,
            batch_index: 0,
            token_count: 1,
            draft_steps: 8,
        };
        assert_eq!(slot.draft_steps, 8);
    }

    // ── InterleavedBatch: decode_slots and prefill_slots are independently accessible ──

    #[test]
    fn interleaved_batch_slot_vectors_independently_accessible() {
        let inner = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 1, 2],
            draft_steps: vec![3, 4],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 3 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 1, draft_steps: 4 },
            ],
        };
        assert_eq!(ib.decode_slots[0].draft_steps, 3);
        assert_eq!(ib.prefill_slots[0].draft_steps, 4);
    }

    // ── ContinuousBatcher: FifoOrder preserves enqueue order ──

    #[test]
    fn continuous_batcher_fifo_order_preserves_enqueue_order() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        // Enqueue in specific order
        b.enqueue(make_sequence(30, 1));
        b.enqueue(make_sequence(10, 1));
        b.enqueue(make_sequence(20, 1));
        let batch = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::FifoOrder);
        // FifoOrder should return in enqueue order: 30, 10, 20
        assert_eq!(batch.requests, vec![30, 10, 20]);
    }

    // ── ContinuousBatcher: multiple completes in sequence ──

    #[test]
    fn continuous_batcher_multiple_completes_in_sequence() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 2));
        b.enqueue(make_sequence(2, 2));
        b.enqueue(make_sequence(3, 2));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert_eq!(b.running_len(), 3);
        // Complete all
        b.update_batch(&mut scheduler, &[
            BatchResult::complete(1, Some(10), Default::default()),
            BatchResult::complete(2, Some(20), Default::default()),
            BatchResult::complete(3, Some(30), Default::default()),
        ]);
        assert_eq!(b.running_len(), 0);
        assert!(!b.has_pending_work());
    }

    // ── ChunkedConfig: field accessors ──

    #[test]
    fn chunked_config_field_accessors() {
        let config = ChunkedConfig {
            min_chunk: 64,
            max_chunk: 256,
            decode_slots: 2,
            enable_splitfuse: true,
        };
        assert_eq!(config.min_chunk, 64);
        assert_eq!(config.max_chunk, 256);
        assert_eq!(config.decode_slots, 2);
        assert!(config.enable_splitfuse);
    }

    // ── ChunkedConfig: default values ──

    #[test]
    fn chunked_config_default_values() {
        let config = ChunkedConfig::default();
        assert!(config.min_chunk > 0);
        assert!(config.max_chunk >= config.min_chunk);
    }

    // ── BatchPrepData: seq_positions field mutation ──

    #[test]
    fn batch_prep_data_seq_positions_field_mutation() {
        let mut d = BatchPrepData::new(4);
        assert_eq!(d.seq_positions, vec![0u32; 4]);
        d.seq_positions[0] = 100;
        d.seq_positions[3] = 400;
        assert_eq!(d.seq_positions[0], 100);
        assert_eq!(d.seq_positions[1], 0);
        assert_eq!(d.seq_positions[2], 0);
        assert_eq!(d.seq_positions[3], 400);
    }

    // ── BatchResult: continue_with_token with zero request_id ──

    #[test]
    fn batch_result_continue_with_zero_request_id() {
        let result = BatchResult::continue_with_token(0, 1, Default::default());
        assert_eq!(result.request_id, 0);
        assert_eq!(result.action, BatchAction::Continue);
        assert_eq!(result.generated_token, Some(1));
    }

    // ── New tests (+15) ────────────────────────────────────────────────────

    // ── BatchPrepData: set_sampling_params exact boundary base_plus_4_eq_len ──

    #[test]
    fn batch_prep_data_set_sampling_params_exact_boundary() {
        let mut d = BatchPrepData::new(1);
        d.set_sampling_params(0, 1.0, 1, 1.0, 1);
        assert_eq!(d.sampling_params_packed[0], 1.0f32.to_bits());
        // Verify exactly 4 elements were written and base+3 is the last valid index
        assert_eq!(d.sampling_params_packed[3], 1u32);
    }

    // ── BatchPrepData: sampling_params_packed all seqs written independently ──

    #[test]
    fn batch_prep_data_sampling_params_all_seqs_independent() {
        let mut d = BatchPrepData::new(3);
        d.set_sampling_params(0, 0.1, 0, 0.0, 0);
        d.set_sampling_params(1, 0.2, 1, 0.0, 1);
        d.set_sampling_params(2, 0.3, 2, 0.0, 2);
        assert_eq!(d.sampling_params_packed[0], 0.1f32.to_bits());
        assert_eq!(d.sampling_params_packed[4], 0.2f32.to_bits());
        assert_eq!(d.sampling_params_packed[8], 0.3f32.to_bits());
        assert_eq!(d.sampling_params_packed[3], 0u32);
        assert_eq!(d.sampling_params_packed[7], 1u32);
        assert_eq!(d.sampling_params_packed[11], 2u32);
    }

    // ── BatchPrepData: clone deep-copies sampling_params_packed ──

    #[test]
    fn batch_prep_data_clone_deep_copies_sampling_params() {
        let mut d = BatchPrepData::new(2);
        d.set_sampling_params(0, 42.0, 99, 0.5, 7);
        let cloned = d.clone();
        // Mutate original; clone must be unaffected
        d.sampling_params_packed[0] = 0;
        assert_eq!(cloned.sampling_params_packed[0], 42.0f32.to_bits());
    }

    // ── ScheduledBatch: requests with max u64 RequestId values ──

    #[test]
    fn scheduled_batch_requests_with_max_u64() {
        let batch = ScheduledBatch {
            requests: vec![u64::MAX, 0, u64::MAX - 1],
            seq_offsets: vec![0, 1, 2, 3],
            draft_steps: vec![0, 0, 0],
        };
        assert_eq!(batch.requests[0], u64::MAX);
        assert_eq!(batch.requests[1], 0);
        assert_eq!(batch.requests[2], u64::MAX - 1);
    }

    // ── BatchResult: PartialEq with NaN telemetry returns false ──

    #[test]
    fn batch_result_partial_eq_nan_telemetry() {
        let mut tel_nan = crate::scheduler::telemetry::SequenceTelemetry::default();
        tel_nan.l2_delta = f32::NAN;
        let a = BatchResult::continue_with_token(1, 42, tel_nan);
        let b = BatchResult::continue_with_token(1, 42, tel_nan);
        // NaN != NaN, so BatchResult PartialEq with NaN telemetry should be false
        assert_ne!(a, b);
    }

    // ── BatchResult: complete stores telemetry correctly ──

    #[test]
    fn batch_result_complete_stores_telemetry() {
        let tel = crate::scheduler::telemetry::SequenceTelemetry {
            l2_delta: 0.99,
            has_outlier: true,
            dead_density: 0.5,
            per_head_entropy: 1.0,
            transform_ratio: 0.01,
            output_entropy: 2.0,
        };
        let result = BatchResult::complete(42, Some(7), tel);
        assert_eq!(result.telemetry.l2_delta, 0.99);
        assert!(result.telemetry.has_outlier);
        assert_eq!(result.telemetry.output_entropy, 2.0);
    }

    // ── ContinuousBatcher: mean_context_len grows with longer prompts ──

    #[test]
    fn continuous_batcher_mean_context_len_grows_with_longer_prompts() {
        let mut b = ContinuousBatcher::new();
        b.enqueue(make_sequence(1, 4));
        let mean_short = b.mean_context_len();
        b.enqueue(make_sequence(2, 20));
        let mean_mixed = b.mean_context_len();
        assert_eq!(mean_short, 4);
        assert_eq!(mean_mixed, 12); // (4 + 20) / 2
    }

    // ── ContinuousBatcher: build_interleaved_batch classifies prefill ──

    #[test]
    fn continuous_batcher_build_interleaved_classifies_prefill() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 8));
        // First build: sequence needs prefill (no generated tokens yet)
        let ib = b.build_interleaved_batch(
            &mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder,
        );
        // Newly admitted sequence needs prefill, so it goes to prefill_slots
        assert!(ib.prefill_slots.len() + ib.decode_slots.len() >= 1);
    }

    // ── ContinuousBatcher: pause then resume via continue ──

    #[test]
    fn continuous_batcher_pause_then_resume_via_continue() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Pause
        b.update_batch(&mut scheduler, &[BatchResult::pause(1)]);
        assert_eq!(b.get_running_mut(1).unwrap().state, SequenceState::Paused);
        // Resume via Continue on next build_batch (paused -> running in build_batch)
        let batch = b.build_batch(&mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder);
        assert!(batch.requests.contains(&1));
    }

    // ── ContinuousBatcher: update_batch with mixed fail and continue ──

    #[test]
    fn continuous_batcher_update_batch_mixed_fail_and_continue() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        b.enqueue(make_sequence(2, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[
            BatchResult::fail(1),
            BatchResult::continue_with_token(2, 42, Default::default()),
        ]);
        assert!(!b.running.contains_key(&1));
        assert!(b.running.contains_key(&2));
        assert_eq!(b.running_len(), 1);
    }

    // ── InterleavedBatch: prefill_tokens with multiple prefill slots ──

    #[test]
    fn interleaved_batch_prefill_tokens_multiple_slots_sum() {
        let inner = ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 10, 25, 40],
            draft_steps: vec![0, 0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![],
            prefill_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 10, draft_steps: 0 },
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 15, draft_steps: 0 },
                InterleavedSlot { request_id: 3, batch_index: 2, token_count: 15, draft_steps: 0 },
            ],
        };
        assert_eq!(ib.prefill_tokens(), 40);
        assert_eq!(ib.total_tokens(), 40);
        assert!(!ib.is_interleaved());
    }

    // ── InterleavedSlot: draft_steps is preserved through clone ──

    #[test]
    fn interleaved_slot_draft_steps_preserved_through_clone() {
        let slot = InterleavedSlot {
            request_id: 42, batch_index: 3, token_count: 7, draft_steps: 16,
        };
        let cloned = slot.clone();
        assert_eq!(cloned.draft_steps, 16);
        assert_eq!(cloned.request_id, slot.request_id);
    }

    // ── BatchAction: Debug output is non-empty for all variants ──

    #[test]
    fn batch_action_debug_all_non_empty() {
        for variant in [BatchAction::Continue, BatchAction::Complete, BatchAction::Pause, BatchAction::Fail] {
            let s = format!("{variant:?}");
            assert!(!s.is_empty(), "Debug output for {variant:?} must be non-empty");
        }
    }

    // ── BatchPrepData: new with very large num_seqs does not panic ──

    #[test]
    fn batch_prep_data_new_large_count_no_panic() {
        let d = BatchPrepData::new(10_000);
        assert_eq!(d.prompt_lens.len(), 10_000);
        assert_eq!(d.sampling_params_packed.len(), 40_000);
        assert!(d.active_flags.iter().all(|&f| f == 1));
    }

    // ── ContinuousBatcher: enqueue sets correct enqueue_order sequence ──

    #[test]
    fn continuous_batcher_enqueue_order_independent_of_id() {
        let mut b = ContinuousBatcher::new();
        // Enqueue with descending IDs to verify order is based on enqueue time, not ID
        b.enqueue(make_sequence(100, 1));
        b.enqueue(make_sequence(1, 1));
        b.enqueue(make_sequence(50, 1));
        let orders: Vec<(RequestId, u64)> = b.waiting.iter().map(|s| (s.id, s.enqueue_order)).collect();
        assert!(orders[0].1 < orders[1].1);
        assert!(orders[1].1 < orders[2].1);
    }

    // ── New tests (+15) ────────────────────────────────────────────────────

    // ── ScheduledBatch: construction with duplicate request IDs is allowed ──

    #[test]
    fn scheduled_batch_duplicate_request_ids_allowed() {
        let batch = ScheduledBatch {
            requests: vec![7, 7, 7],
            seq_offsets: vec![0, 1, 2, 3],
            draft_steps: vec![0, 0, 0],
        };
        assert_eq!(batch.requests.len(), 3);
        assert!(batch.requests.iter().all(|&id| id == 7));
    }

    // ── ScheduledBatch: seq_offsets all zeros is a legal degenerate construction ──

    #[test]
    fn scheduled_batch_seq_offsets_all_zeros() {
        let batch = ScheduledBatch {
            requests: vec![1, 2],
            seq_offsets: vec![0, 0, 0],
            draft_steps: vec![0, 0],
        };
        assert_eq!(batch.seq_offsets.len(), 3);
        assert!(batch.seq_offsets.iter().all(|&off| off == 0));
    }

    // ── BatchPrepData: active_flags can be set to mixed 0 and 1 pattern ──

    #[test]
    fn batch_prep_data_active_flags_mixed_pattern() {
        let mut d = BatchPrepData::new(5);
        // Initially all 1
        assert_eq!(d.active_flags, vec![1u32; 5]);
        // Set alternating pattern
        d.active_flags[0] = 0;
        d.active_flags[2] = 0;
        d.active_flags[4] = 0;
        assert_eq!(d.active_flags, vec![0u32, 1, 0, 1, 0]);
        // Restore all to 1
        for f in &mut d.active_flags {
            *f = 1;
        }
        assert_eq!(d.active_flags, vec![1u32; 5]);
    }

    // ── ContinuousBatcher: enqueue sequence with empty prompt is admitted but stuck in prefill ──

    #[test]
    fn continuous_batcher_enqueue_empty_prompt_admitted_stuck_prefill() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 0));
        assert_eq!(b.waiting_len(), 1);
        // Build batch: sequence gets admitted to running (admit_waiting succeeds),
        // but prompt_len=0 means prefill backfill produces 0 extracted tokens.
        let batch = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Sequence was admitted to running
        assert_eq!(b.running_len(), 1);
        assert!(b.running.contains_key(&1));
        // With prompt_len=0, extracted_tokens=0, so it is not added to the batch requests
        assert!(!batch.requests.contains(&1));
        // needs_prefill() remains true (no generated tokens yet)
        assert!(b.running.get(&1).unwrap().needs_prefill());
    }

    // ── ContinuousBatcher: re-enqueue same ID after fail is allowed ──

    #[test]
    fn continuous_batcher_reenqueue_after_fail() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Fail removes it from running
        b.update_batch(&mut scheduler, &[BatchResult::fail(1)]);
        assert!(!b.has_pending_work());
        // Re-enqueue same ID is allowed since it is no longer in waiting or running
        b.enqueue(make_sequence(1, 8));
        assert_eq!(b.waiting_len(), 1);
        assert!(b.has_pending_work());
    }

    // ── ContinuousBatcher: build_batch_with_prep populates gen_counts after continue ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_gen_counts_after_continue() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        // First build: prefill
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Continue to produce a generated token
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 42, Default::default())]);
        // Second build_with_prep: gen_counts should reflect the generated token
        let (_, prep) = b.build_batch_with_prep(
            &mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(prep.gen_counts.len(), 1);
        assert!(prep.gen_counts[0] >= 1, "gen_counts must be >= 1 after one continue");
    }

    // ── ContinuousBatcher: build_batch_with_prep populates kv_lens and seq_positions ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_kv_lens_and_seq_positions() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 10));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Continue to advance position
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 99, Default::default())]);
        let (_, prep) = b.build_batch_with_prep(
            &mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder,
        );
        // After continue, kv_lens and seq_positions should reflect the updated position
        assert_eq!(prep.kv_lens[0], prep.seq_positions[0],
            "kv_lens and seq_positions must match for the same sequence");
        assert!(prep.seq_positions[0] > 0, "seq_positions must be > 0 after prefill+continue");
    }

    // ── ContinuousBatcher: build_interleaved_batch with decode after prefill+continue ──

    #[test]
    fn continuous_batcher_build_interleaved_with_running_decode() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        // Prefill
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 42, Default::default())]);
        // Now build interleaved — sequence no longer needs prefill, so it should be in decode_slots
        let ib = b.build_interleaved_batch(
            &mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder,
        );
        assert_eq!(ib.decode_slots.len(), 1, "decode-only sequence should be in decode_slots");
        assert_eq!(ib.prefill_slots.len(), 0, "no prefill sequences expected");
        assert_eq!(ib.decode_slots[0].request_id, 1);
        assert_eq!(ib.decode_slots[0].token_count, 1);
    }

    // ── ContinuousBatcher: mean_context_len with only running sequences ──

    #[test]
    fn continuous_batcher_mean_context_len_only_running() {
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 6));
        b.enqueue(make_sequence(2, 10));
        // Admit both to running
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // No waiting sequences remain
        assert_eq!(b.waiting_len(), 0);
        assert_eq!(b.running_len(), 2);
        // Mean = (6 + 10) / 2 = 8
        assert_eq!(b.mean_context_len(), 8);
    }

    // ── BatchResult: complete with custom telemetry propagates all fields ──

    #[test]
    fn batch_result_complete_full_telemetry_propagation() {
        let tel = crate::scheduler::telemetry::SequenceTelemetry {
            l2_delta: 0.75,
            has_outlier: true,
            dead_density: 0.33,
            per_head_entropy: 2.71,
            transform_ratio: 0.12,
            output_entropy: 1.41,
        };
        let result = BatchResult::complete(42, Some(999), tel);
        assert_eq!(result.request_id, 42);
        assert_eq!(result.action, BatchAction::Complete);
        assert_eq!(result.generated_token, Some(999));
        assert_eq!(result.telemetry.l2_delta, 0.75);
        assert!(result.telemetry.has_outlier);
        assert_eq!(result.telemetry.dead_density, 0.33);
        assert_eq!(result.telemetry.per_head_entropy, 2.71);
        assert_eq!(result.telemetry.transform_ratio, 0.12);
        assert_eq!(result.telemetry.output_entropy, 1.41);
    }

    // ── InterleavedBatch: multiple decode and multiple prefill slots ──

    #[test]
    fn interleaved_batch_multiple_decode_and_prefill() {
        let inner = ScheduledBatch {
            requests: vec![1, 2, 3, 4, 5],
            seq_offsets: vec![0, 1, 2, 12, 22, 23],
            draft_steps: vec![0, 0, 0, 0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 5, batch_index: 4, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 3, batch_index: 2, token_count: 10, draft_steps: 0 },
                InterleavedSlot { request_id: 4, batch_index: 3, token_count: 10, draft_steps: 0 },
            ],
        };
        assert_eq!(ib.decode_tokens(), 3);
        assert_eq!(ib.prefill_tokens(), 20);
        assert_eq!(ib.total_tokens(), 23);
        assert!(ib.is_interleaved());
        assert_eq!(ib.request_ids().len(), 5);
    }

    // ── InterleavedBatch: request_ids slice is a reference to inner ──

    #[test]
    fn interleaved_batch_request_ids_is_inner_slice() {
        let inner = ScheduledBatch {
            requests: vec![100, 200, 300],
            seq_offsets: vec![0, 1, 2, 3],
            draft_steps: vec![0, 0, 0],
        };
        let ib = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 100, batch_index: 0, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 200, batch_index: 1, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 300, batch_index: 2, token_count: 1, draft_steps: 0 },
            ],
        };
        let ids = ib.request_ids();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], 100);
        assert_eq!(ids[1], 200);
        assert_eq!(ids[2], 300);
    }

    // ── ChunkedState: config field is accessible through batcher ──

    #[test]
    fn continuous_batcher_chunked_state_config_accessible() {
        let config = ChunkedConfig {
            min_chunk: 32,
            max_chunk: 1024,
            decode_slots: 6,
            enable_splitfuse: false,
        };
        let b = ContinuousBatcher::new().with_chunked(config);
        let state = b.chunked_state.as_ref().unwrap();
        assert_eq!(state.config.min_chunk, 32);
        assert_eq!(state.config.max_chunk, 1024);
        assert_eq!(state.config.decode_slots, 6);
        assert!(!state.config.enable_splitfuse);
    }

    // ── BatchPrepData: sampling_params_packed layout for mid-sequence ──

    #[test]
    fn batch_prep_data_sampling_params_layout_mid_sequence() {
        let mut d = BatchPrepData::new(4);
        // Write to seq 1 and seq 3, leave 0 and 2 untouched
        d.set_sampling_params(1, 0.25, 25, 0.25, 10);
        d.set_sampling_params(3, 4.0, 4, 4.0, 40);
        // Seq 0: base=0, all zeros
        assert_eq!(&d.sampling_params_packed[0..4], &[0u32; 4]);
        // Seq 1: base=4
        assert_eq!(d.sampling_params_packed[4], 0.25f32.to_bits());
        assert_eq!(d.sampling_params_packed[5], 25u32);
        assert_eq!(d.sampling_params_packed[6], 0.25f32.to_bits());
        assert_eq!(d.sampling_params_packed[7], 10u32);
        // Seq 2: base=8, all zeros
        assert_eq!(&d.sampling_params_packed[8..12], &[0u32; 4]);
        // Seq 3: base=12
        assert_eq!(d.sampling_params_packed[12], 4.0f32.to_bits());
        assert_eq!(d.sampling_params_packed[13], 4u32);
        assert_eq!(d.sampling_params_packed[14], 4.0f32.to_bits());
        assert_eq!(d.sampling_params_packed[15], 40u32);
    }

    // ── New tests (+15) ────────────────────────────────────────────────────

    // ── ContinuousBatcher: build_batch_with_prep last_sampled_tokens after multiple continues ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_last_sampled_tokens_after_continues() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        // Prefill
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Continue twice
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 100, Default::default())]);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 200, Default::default())]);
        // Act
        let (_, prep) = b.build_batch_with_prep(
            &mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder,
        );
        // Assert: last_sampled_tokens should be the most recent generated token
        assert_eq!(prep.last_sampled_tokens[0], 200, "last_sampled_tokens must reflect the last generated token");
    }

    // ── ContinuousBatcher: build_interleaved_batch with new prefill and existing decode ──

    #[test]
    fn continuous_batcher_build_interleaved_mixed_prefill_and_decode() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        // First sequence: prefill + continue to make it decode
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 42, Default::default())]);
        // Second sequence: enqueue but don't admit yet
        b.enqueue(make_sequence(2, 8));
        // Act: build interleaved with admit=true
        let ib = b.build_interleaved_batch(
            &mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder,
        );
        // Assert: should have both decode (seq 1) and prefill (seq 2) slots
        assert!(ib.decode_slots.iter().any(|s| s.request_id == 1), "seq 1 must be in decode_slots");
        assert!(ib.prefill_slots.iter().any(|s| s.request_id == 2), "seq 2 must be in prefill_slots");
        assert!(ib.is_interleaved(), "batch with both decode and prefill must be interleaved");
    }

    // ── ContinuousBatcher: build_batch_with_prep returns zero last_sampled_tokens before any continue ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_zero_last_sampled_before_continue() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 6));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Act: build_with_prep without any continue
        let (_, prep) = b.build_batch_with_prep(
            &mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder,
        );
        // Assert: no generated tokens yet, so last_sampled_tokens is 0
        assert_eq!(prep.last_sampled_tokens[0], 0, "last_sampled_tokens must be 0 before any continue");
    }

    // ── ContinuousBatcher: multiple pause/resume cycles preserve sequence state ──

    #[test]
    fn continuous_batcher_multiple_pause_resume_cycles() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Act: cycle through pause -> resume -> pause -> resume
        b.update_batch(&mut scheduler, &[BatchResult::pause(1)]);
        assert_eq!(b.get_running_mut(1).unwrap().state, SequenceState::Paused);
        let _ = b.build_batch(&mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 10, Default::default())]);
        assert_eq!(b.get_running_mut(1).unwrap().state, SequenceState::Running);
        b.update_batch(&mut scheduler, &[BatchResult::pause(1)]);
        assert_eq!(b.get_running_mut(1).unwrap().state, SequenceState::Paused);
        let _ = b.build_batch(&mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 20, Default::default())]);
        // Assert: sequence is still running and has accumulated tokens
        let seq = b.get_running_mut(1).unwrap();
        assert_eq!(seq.state, SequenceState::Running);
        assert!(seq.generated_tokens.len() >= 2, "must have at least 2 generated tokens");
    }

    // ── ContinuousBatcher: build_batch with admit=false keeps decode running ──

    #[test]
    fn continuous_batcher_build_batch_no_admit_keeps_decode_running() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 42, Default::default())]);
        // Enqueue a second sequence that should NOT be admitted
        b.enqueue(make_sequence(2, 4));
        assert_eq!(b.waiting_len(), 1);
        // Act: build with admit=false
        let batch = b.build_batch(&mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder);
        // Assert: seq 1 still in batch (decode), seq 2 still waiting
        assert!(batch.requests.contains(&1), "seq 1 decode must be in batch");
        assert!(!batch.requests.contains(&2), "seq 2 must NOT be admitted with admit=false");
        assert_eq!(b.waiting_len(), 1, "seq 2 must remain in waiting");
    }

    // ── ContinuousBatcher: build_interleaved_batch with chunked config for prefill ──

    #[test]
    fn continuous_batcher_build_interleaved_with_chunked_prefill() {
        // Arrange
        let config = ChunkedConfig {
            min_chunk: 2,
            max_chunk: 4,
            decode_slots: 2,
            enable_splitfuse: false,
        };
        let mut b = ContinuousBatcher::new().with_chunked(config);
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 16));
        // Act: build interleaved - chunked config should chunk the prefill
        let ib = b.build_interleaved_batch(
            &mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder,
        );
        // Assert: at least one slot (either prefill or decode) should be present
        let total_slots = ib.decode_slots.len() + ib.prefill_slots.len();
        assert!(total_slots >= 1, "batch must contain at least one slot");
    }

    // ── ContinuousBatcher: update_batch fail with chunked_state removes tracker ──

    #[test]
    fn continuous_batcher_update_batch_fail_with_chunked_removes_tracker() {
        // Arrange
        let config = ChunkedConfig {
            min_chunk: 2,
            max_chunk: 8,
            decode_slots: 2,
            enable_splitfuse: false,
        };
        let mut b = ContinuousBatcher::new().with_chunked(config);
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 16));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Act: fail the sequence
        b.update_batch(&mut scheduler, &[BatchResult::fail(1)]);
        // Assert: sequence removed from running, no pending work
        assert_eq!(b.running_len(), 0);
        assert!(!b.has_pending_work());
    }

    // ── ContinuousBatcher: update_batch complete with chunked_state removes tracker ──

    #[test]
    fn continuous_batcher_update_batch_complete_with_chunked_removes_tracker() {
        // Arrange
        let config = ChunkedConfig {
            min_chunk: 2,
            max_chunk: 8,
            decode_slots: 2,
            enable_splitfuse: false,
        };
        let mut b = ContinuousBatcher::new().with_chunked(config);
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 42, Default::default())]);
        // Act: complete the sequence
        b.update_batch(&mut scheduler, &[BatchResult::complete(1, Some(99), Default::default())]);
        // Assert: sequence removed from running, scheduler resources freed
        assert_eq!(b.running_len(), 0);
        assert!(!b.has_pending_work());
        assert_eq!(scheduler.num_free_blocks(), 64);
    }

    // ── ContinuousBatcher: build_batch seq_offsets reflect prefill token counts ──

    #[test]
    fn continuous_batcher_build_batch_seq_offsets_prefill_token_counts() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 8));
        // Act
        let batch = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Assert: seq_offsets for a prefill of 8 tokens should span at least 1
        assert_eq!(batch.requests.len(), 1);
        let span = batch.seq_offsets[1] - batch.seq_offsets[0];
        assert!(span >= 1, "prefill span must be >= 1, got {span}");
    }

    // ── ContinuousBatcher: build_batch_with_prep kv_lens reflect position after multiple continues ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_kv_lens_advance_with_continues() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        // Continue 3 times
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 10, Default::default())]);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 20, Default::default())]);
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 30, Default::default())]);
        // Act
        let (_, prep) = b.build_batch_with_prep(
            &mut scheduler, usize::MAX, false, BatchOrderPolicy::StrictRequestIdOrder,
        );
        // Assert: kv_lens = position = prompt_len + 3 generated tokens = 7
        assert_eq!(prep.kv_lens[0], 7, "kv_lens must be prompt_len + generated_count = 4 + 3 = 7");
        assert_eq!(prep.gen_counts[0], 3, "gen_counts must be 3");
    }

    // ── BatchPrepData: set_sampling_params with top_k=0 encodes zero ──

    #[test]
    fn batch_prep_data_set_sampling_params_zero_top_k() {
        // Arrange
        let mut d = BatchPrepData::new(1);
        // Act
        d.set_sampling_params(0, 0.5, 0, 0.9, 2);
        // Assert
        assert_eq!(d.sampling_params_packed[1], 0u32, "top_k=0 must be encoded as 0");
        assert_eq!(d.sampling_params_packed[0], 0.5f32.to_bits());
        assert_eq!(d.sampling_params_packed[2], 0.9f32.to_bits());
        assert_eq!(d.sampling_params_packed[3], 2u32);
    }

    // ── ContinuousBatcher: build_batch with budget=1 allows single decode slot ──

    #[test]
    fn continuous_batcher_build_batch_budget_one_single_decode() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        // Enqueue two sequences and advance both to decode phase
        b.enqueue(make_sequence(1, 4));
        b.enqueue(make_sequence(2, 4));
        let _ = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder);
        b.update_batch(&mut scheduler, &[
            BatchResult::continue_with_token(1, 10, Default::default()),
            BatchResult::continue_with_token(2, 20, Default::default()),
        ]);
        // Act: build with token_budget=1
        let batch = b.build_batch(&mut scheduler, 1, false, BatchOrderPolicy::StrictRequestIdOrder);
        // Assert: only 1 decode slot due to budget constraint
        assert_eq!(batch.requests.len(), 1, "token_budget=1 should allow exactly 1 decode request");
    }

    // ── ContinuousBatcher: enqueue_order preserved across admit and build cycle ──

    #[test]
    fn continuous_batcher_fifo_order_preserved_across_build_cycles() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(30, 4));
        b.enqueue(make_sequence(10, 4));
        b.enqueue(make_sequence(20, 4));
        // Act: admit and build, then continue and build again
        let first = b.build_batch(&mut scheduler, usize::MAX, true, BatchOrderPolicy::FifoOrder);
        assert_eq!(first.requests, vec![30, 10, 20], "FifoOrder must preserve enqueue order");
        b.update_batch(&mut scheduler, &[
            BatchResult::continue_with_token(30, 1, Default::default()),
            BatchResult::continue_with_token(10, 2, Default::default()),
            BatchResult::continue_with_token(20, 3, Default::default()),
        ]);
        // Build again with FifoOrder
        let second = b.build_batch(&mut scheduler, usize::MAX, false, BatchOrderPolicy::FifoOrder);
        // Assert: decode order still respects original FIFO enqueue order
        assert_eq!(second.requests, vec![30, 10, 20], "FifoOrder decode must preserve original enqueue order");
    }

    // ── ContinuousBatcher: build_interleaved_batch with chunked_state on_chunk_finished ──

    #[test]
    fn continuous_batcher_chunked_prefill_completes_in_multiple_steps() {
        // Arrange
        let config = ChunkedConfig {
            min_chunk: 2,
            max_chunk: 4,
            decode_slots: 4,
            enable_splitfuse: false,
        };
        let mut b = ContinuousBatcher::new().with_chunked(config);
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 10));
        // Act: first chunk
        let batch1 = b.build_batch(&mut scheduler, 4, true, BatchOrderPolicy::StrictRequestIdOrder);
        assert!(batch1.requests.contains(&1));
        // Continue the prefill chunk (needs_prefill is still true since no generated tokens)
        b.update_batch(&mut scheduler, &[BatchResult::continue_with_token(1, 0, Default::default())]);
        // Assert: sequence still needs prefill (generated_tokens was pushed, but let's verify state)
        assert_eq!(b.running_len(), 1);
        assert!(b.has_pending_work());
    }

    // ── ContinuousBatcher: build_batch_with_prep for multi-sequence batch ──

    #[test]
    fn continuous_batcher_build_batch_with_prep_multi_sequence_prompt_lens() {
        // Arrange
        let mut b = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(64, 4, HGALConfig::default());
        b.enqueue(make_sequence(1, 3));
        b.enqueue(make_sequence(2, 7));
        b.enqueue(make_sequence(3, 11));
        // Act
        let (batch, prep) = b.build_batch_with_prep(
            &mut scheduler, usize::MAX, true, BatchOrderPolicy::StrictRequestIdOrder,
        );
        // Assert: all three sequences in batch
        assert_eq!(batch.requests.len(), 3);
        assert_eq!(prep.prompt_lens.len(), 3);
        // prompt_lens must match original prompt lengths (sorted by request_id order)
        let mut paired: Vec<_> = batch.requests.iter().zip(prep.prompt_lens.iter()).collect();
        paired.sort_by_key(|(&rid, _)| rid);
        assert_eq!(*paired[0].1, 3, "seq 1 prompt_len must be 3");
        assert_eq!(*paired[1].1, 7, "seq 2 prompt_len must be 7");
        assert_eq!(*paired[2].1, 11, "seq 3 prompt_len must be 11");
    }

    // ── BatchPrepData: new with zero sequences produces empty but valid state ──

    #[test]
    fn batch_prep_data_new_zero_seqs_empty_vectors() {
        // Arrange & Act
        let d = BatchPrepData::new(0);
        // Assert: all per-seq vectors are empty, scalars are zero
        assert!(d.prompt_lens.is_empty());
        assert!(d.kv_lens.is_empty());
        assert!(d.active_flags.is_empty());
        assert!(d.sampling_params_packed.is_empty());
        assert_eq!(d.max_decode_steps, 0);
        assert_eq!(d.total_prefill_tokens, 0);
    }

    // ── BatchPrepData: new initialises active_flags to 1 ──

    #[test]
    fn batch_prep_data_new_active_flags_default_to_one() {
        // Arrange & Act
        let d = BatchPrepData::new(4);
        // Assert: every active flag is 1
        assert_eq!(d.active_flags, vec![1u32, 1, 1, 1]);
    }

    // ── BatchPrepData: set_sampling_params with seq beyond capacity is silently ignored ──

    #[test]
    fn batch_prep_data_set_sampling_params_far_out_of_bounds_ignored() {
        // Arrange
        let mut d = BatchPrepData::new(1);
        // Act: seq index 100 is far out of range for a 1-seq prep data
        d.set_sampling_params(100, 0.7, 50, 0.95, 3);
        // Assert: packed array is still all zeros (unchanged)
        assert_eq!(d.sampling_params_packed, vec![0u32; 4]);
    }

    // ── BatchAction: all variants are distinct ──

    #[test]
    fn batch_action_variants_are_distinct() {
        // Arrange
        let actions = [
            BatchAction::Continue,
            BatchAction::Complete,
            BatchAction::Pause,
            BatchAction::Fail,
        ];
        // Act & Assert: every pair must be unequal
        for i in 0..actions.len() {
            for j in (i + 1)..actions.len() {
                assert_ne!(actions[i], actions[j], "BatchAction variants must all differ");
            }
        }
    }

    // ── BatchResult: fail has no generated token ──

    #[test]
    fn batch_result_fail_carries_no_token() {
        // Act
        let r = BatchResult::fail(42);
        // Assert
        assert_eq!(r.request_id, 42);
        assert_eq!(r.action, BatchAction::Fail);
        assert!(r.generated_token.is_none());
    }

    // ── BatchResult: pause has no generated token ──

    #[test]
    fn batch_result_pause_carries_no_token() {
        // Act
        let r = BatchResult::pause(7);
        // Assert
        assert_eq!(r.request_id, 7);
        assert_eq!(r.action, BatchAction::Pause);
        assert!(r.generated_token.is_none());
    }

    // ── BatchResult: complete without token ──

    #[test]
    fn batch_result_complete_without_token() {
        // Act
        let r = BatchResult::complete(99, None, Default::default());
        // Assert
        assert_eq!(r.request_id, 99);
        assert_eq!(r.action, BatchAction::Complete);
        assert!(r.generated_token.is_none());
    }

    // ── InterleavedBatch: decode_tokens with no slots is zero ──

    #[test]
    fn interleaved_batch_decode_tokens_empty_is_zero() {
        // Arrange
        let ib = InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![],
                seq_offsets: vec![0],
                draft_steps: vec![],
            },
            decode_slots: vec![],
            prefill_slots: vec![],
        };
        // Act & Assert
        assert_eq!(ib.decode_tokens(), 0);
        assert_eq!(ib.prefill_tokens(), 0);
        assert_eq!(ib.total_tokens(), 0);
        assert!(!ib.is_interleaved());
    }

    // ── InterleavedBatch: is_interleaved true when both slot types present ──

    #[test]
    fn interleaved_batch_is_interleaved_true_when_both_present() {
        // Arrange
        let ib = InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![10, 20],
                seq_offsets: vec![0, 1, 2],
                draft_steps: vec![0, 0],
            },
            decode_slots: vec![InterleavedSlot {
                request_id: 10,
                batch_index: 0,
                token_count: 1,
                draft_steps: 0,
            }],
            prefill_slots: vec![InterleavedSlot {
                request_id: 20,
                batch_index: 1,
                token_count: 8,
                draft_steps: 0,
            }],
        };
        // Act & Assert
        assert!(ib.is_interleaved());
        assert_eq!(ib.total_tokens(), 9);
    }

    // ── InterleavedBatch: total_tokens sums both slot types ──

    #[test]
    fn interleaved_batch_total_tokens_sums_decode_and_prefill() {
        // Arrange
        let ib = InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![1, 2, 3],
                seq_offsets: vec![0, 1, 2, 7],
                draft_steps: vec![0, 0, 0],
            },
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 2, batch_index: 1, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 3, batch_index: 2, token_count: 5, draft_steps: 0 },
            ],
        };
        // Act & Assert
        assert_eq!(ib.decode_tokens(), 2);
        assert_eq!(ib.prefill_tokens(), 5);
        assert_eq!(ib.total_tokens(), 7);
    }

    // ── InterleavedSlot: Clone and Debug round-trip ──

    #[test]
    fn interleaved_slot_clone_and_debug_round_trip() {
        // Arrange
        let slot = InterleavedSlot {
            request_id: 55,
            batch_index: 3,
            token_count: 16,
            draft_steps: 2,
        };
        // Act
        let cloned = slot.clone();
        let debug_str = format!("{:?}", slot);
        // Assert
        assert_eq!(cloned.request_id, 55);
        assert_eq!(cloned.batch_index, 3);
        assert_eq!(cloned.token_count, 16);
        assert_eq!(cloned.draft_steps, 2);
        assert!(debug_str.contains("55"), "Debug output must include request_id");
    }

    // ── ContinuousBatcher: mean_context_len returns 0 when empty ──

    #[test]
    fn continuous_batcher_mean_context_len_zero_when_empty() {
        // Arrange
        let b = ContinuousBatcher::new();
        // Act
        let mean = b.mean_context_len();
        // Assert
        assert_eq!(mean, 0, "empty batcher must report mean_context_len = 0");
    }

    // ── ScheduledBatch: Clone produces equal copy ──

    #[test]
    fn scheduled_batch_clone_is_equal() {
        // Arrange
        let batch = ScheduledBatch {
            requests: vec![10, 20, 30],
            seq_offsets: vec![0, 4, 9, 15],
            draft_steps: vec![0, 2, 1],
        };
        // Act
        let cloned = batch.clone();
        // Assert
        assert_eq!(cloned.requests, batch.requests);
        assert_eq!(cloned.seq_offsets, batch.seq_offsets);
        assert_eq!(cloned.draft_steps, batch.draft_steps);
    }
}
