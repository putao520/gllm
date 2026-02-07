use std::collections::{BTreeMap, VecDeque};

use gllm_kernels::kernel_types::RequestId;

use super::paged_scheduler::PagedScheduler;
use super::sequence::{Sequence, SequenceState};

#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub requests: Vec<RequestId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchAction {
    Continue,
    Complete,
    Pause,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchResult {
    pub request_id: RequestId,
    pub action: BatchAction,
    pub generated_token: Option<u32>,
}

impl BatchResult {
    pub fn continue_with_token(request_id: RequestId, generated_token: u32) -> Self {
        Self {
            request_id,
            action: BatchAction::Continue,
            generated_token: Some(generated_token),
        }
    }

    pub fn complete(request_id: RequestId, generated_token: Option<u32>) -> Self {
        Self {
            request_id,
            action: BatchAction::Complete,
            generated_token,
        }
    }

    pub fn pause(request_id: RequestId) -> Self {
        Self {
            request_id,
            action: BatchAction::Pause,
            generated_token: None,
        }
    }

    pub fn fail(request_id: RequestId) -> Self {
        Self {
            request_id,
            action: BatchAction::Fail,
            generated_token: None,
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
}

impl ContinuousBatcher {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            running: BTreeMap::new(),
        }
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
        sequence.state = SequenceState::Waiting;
        self.waiting.push_back(sequence);
    }

    pub fn build_batch(
        &mut self,
        scheduler: &mut PagedScheduler,
        max_batch_size: usize,
        admit_new_prefill: bool,
    ) -> ScheduledBatch {
        if admit_new_prefill {
            self.admit_waiting(scheduler);
        }

        let mut requests = Vec::new();
        let mut failed = Vec::new();

        for sequence in self.running.values_mut() {
            if requests.len() >= max_batch_size {
                break;
            }

            if sequence.state == SequenceState::Paused {
                sequence.state = SequenceState::Running;
            }
            if sequence.state != SequenceState::Running {
                continue;
            }

            if sequence.needs_prefill() {
                requests.push(sequence.id);
                continue;
            }

            match scheduler.allocate_next_token(sequence.id) {
                Ok(Some(new_page)) => {
                    sequence.kv_pages.push(new_page);
                    requests.push(sequence.id);
                }
                Ok(None) => requests.push(sequence.id),
                Err(err) => {
                    if err.contains("Out of memory") {
                        // 内存不足：标记为暂停，让其他序列有机会
                        sequence.state = SequenceState::Paused;
                        continue;
                    }
                    sequence.state = SequenceState::Failed;
                    failed.push(sequence.id);
                }
            }
        }

        for request_id in failed {
            self.running.remove(&request_id);
            scheduler.free_sequence(request_id);
        }

        ScheduledBatch { requests }
    }

    pub fn update_batch(&mut self, scheduler: &mut PagedScheduler, results: &[BatchResult]) {
        let mut finished = Vec::new();

        for result in results {
            let Some(sequence) = self.running.get_mut(&result.request_id) else {
                continue;
            };

            match result.action {
                BatchAction::Continue => {
                    if let Some(token) = result.generated_token {
                        sequence.push_generated_token(token);
                    }
                    sequence.state = SequenceState::Running;
                }
                BatchAction::Pause => {
                    sequence.state = SequenceState::Paused;
                }
                BatchAction::Complete => {
                    if let Some(token) = result.generated_token {
                        sequence.push_generated_token(token);
                    }
                    sequence.state = SequenceState::Completed;
                    finished.push(result.request_id);
                }
                BatchAction::Fail => {
                    sequence.state = SequenceState::Failed;
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

            match scheduler.add_sequence(sequence.to_sequence_group()) {
                Ok(()) => {
                    let pages = scheduler
                        .block_tables
                        .get(&request_id)
                        .map(|table| table.blocks.clone())
                        .unwrap_or_default();
                    sequence.mark_running(pages);
                    self.running.insert(request_id, sequence);
                }
                Err(_) => {
                    // 分配失败：放回队列末尾，等待下次 build_batch 时重试
                    // 关键：不在本次循环中重试，避免无限循环
                    sequence.state = SequenceState::Waiting;
                    self.waiting.push_back(sequence);
                    // 继续处理下一个序列，不中断
                }
            }
        }
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
        let first = batcher.build_batch(&mut scheduler, usize::MAX, true);
        assert_eq!(first.requests, vec![10]);
        batcher.update_batch(&mut scheduler, &[BatchResult::continue_with_token(10, 100)]);

        batcher.enqueue(make_sequence(2, 2));
        let second = batcher.build_batch(&mut scheduler, usize::MAX, true);
        assert_eq!(second.requests, vec![2, 10]);
    }

    #[test]
    fn update_batch_complete_releases_scheduler_resources() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(8, 4, HGALConfig::default());

        batcher.enqueue(make_sequence(1, 4));
        let first = batcher.build_batch(&mut scheduler, usize::MAX, true);
        assert_eq!(first.requests, vec![1]);

        batcher.update_batch(&mut scheduler, &[BatchResult::complete(1, Some(7))]);
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

        let prefill = batcher.build_batch(&mut scheduler, usize::MAX, true);
        assert_eq!(prefill.requests, vec![3, 5, 7]);
        batcher.update_batch(
            &mut scheduler,
            &[
                BatchResult::continue_with_token(3, 1),
                BatchResult::continue_with_token(5, 1),
                BatchResult::continue_with_token(7, 1),
            ],
        );

        let decode = batcher.build_batch(&mut scheduler, usize::MAX, true);
        assert_eq!(decode.requests, vec![3, 5, 7]);
    }

    #[test]
    fn build_batch_keeps_waiting_when_capacity_is_not_enough() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(1, 4, HGALConfig::default());

        batcher.enqueue(make_sequence(1, 4));
        batcher.enqueue(make_sequence(2, 4));

        let first = batcher.build_batch(&mut scheduler, usize::MAX, true);
        assert_eq!(first.requests, vec![1]);
        assert_eq!(batcher.waiting.len(), 1);
        assert!(batcher.running.contains_key(&1));
        assert!(!batcher.running.contains_key(&2));
    }
}
