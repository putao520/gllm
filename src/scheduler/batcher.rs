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
                    let prefill_done = self.chunked_state.as_ref().map_or(true, |c| c.is_request_complete(&sequence.id));
                    
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
            .unwrap_or(0)
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
                        .unwrap_or_default();
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

    #[allow(deprecated)]
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
                        .unwrap_or((u64::MAX, *request_id))
                });
            }
            BatchOrderPolicy::ThroughputFirst => {
                // Prefer longer contexts first to approximate throughput-oriented batching.
                ordered_ids.sort_by_key(|request_id| {
                    self.running
                        .get(request_id)
                        .map(|sequence| (usize::MAX - sequence.context_len(), *request_id))
                        .unwrap_or((usize::MAX, *request_id))
                });
            }
        }
        ordered_ids
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
}
