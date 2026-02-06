use super::paged_scheduler::PagedScheduler;
use super::types::SequenceGroup;
use gllm_kernels::kernel_types::RequestId;
use std::collections::VecDeque;

#[derive(Debug)]
pub struct ScheduledBatch {
    pub requests: Vec<RequestId>,
}

pub struct ContinuousBatcher {
    waiting: VecDeque<SequenceGroup>,
    running: Vec<RequestId>,
}

impl ContinuousBatcher {
    pub fn new() -> Self {
        Self {
            waiting: VecDeque::new(),
            running: Vec::new(),
        }
    }

    pub fn add_request(&mut self, group: SequenceGroup) {
        self.waiting.push_back(group);
    }

    pub fn schedule(&mut self, scheduler: &mut PagedScheduler) -> ScheduledBatch {
        let (mut batch_ids, mut next_running) = if self.running.is_empty() {
            self.schedule_prefill_phase(scheduler)
        } else {
            self.schedule_decode_phase(scheduler)
        };

        // Canonical ordering is required for deterministic scheduling.
        batch_ids.sort_unstable();
        next_running.sort_unstable();
        self.running = next_running;

        ScheduledBatch {
            requests: batch_ids,
        }
    }

    fn schedule_decode_phase(
        &mut self,
        scheduler: &mut PagedScheduler,
    ) -> (Vec<RequestId>, Vec<RequestId>) {
        let mut batch_ids = Vec::new();
        let mut next_running = Vec::new();

        self.running.sort_unstable();
        for request_id in self.running.iter().copied() {
            if scheduler.allocate_next_token(request_id).is_ok() {
                batch_ids.push(request_id);
                next_running.push(request_id);
            }
        }

        (batch_ids, next_running)
    }

    fn schedule_prefill_phase(
        &mut self,
        scheduler: &mut PagedScheduler,
    ) -> (Vec<RequestId>, Vec<RequestId>) {
        let mut batch_ids = Vec::new();
        let mut next_running = Vec::new();
        let mut remaining = self.waiting.len();

        while remaining > 0 {
            if let Some(group) = self.waiting.pop_front() {
                let request_id = group.id;
                match scheduler.add_sequence(group.clone()) {
                    Ok(_) => {
                        batch_ids.push(request_id);
                        next_running.push(request_id);
                    }
                    Err(_) => {
                        self.waiting.push_front(group);
                        break;
                    }
                }
            }
            remaining -= 1;
        }

        (batch_ids, next_running)
    }

    /// Mark a request as finished and free resources
    pub fn finish_request(&mut self, scheduler: &mut PagedScheduler, id: RequestId) {
        if let Some(pos) = self.running.iter().position(|&r| r == id) {
            self.running.remove(pos);
        }
        scheduler.free_sequence(id);
    }

    pub fn has_pending_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::scheduler::hgal::HGALConfig;
    use crate::scheduler::types::GroupState;

    use super::*;

    fn make_group(id: RequestId, context_len: usize) -> SequenceGroup {
        SequenceGroup {
            id,
            pages: Vec::new(),
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len,
        }
    }

    #[test]
    fn schedule_is_phase_isolated_and_decode_first() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());

        batcher.add_request(make_group(10, 4));
        let prefill_batch = batcher.schedule(&mut scheduler);
        assert_eq!(prefill_batch.requests, vec![10]);

        batcher.add_request(make_group(2, 2));
        let decode_batch = batcher.schedule(&mut scheduler);
        assert_eq!(decode_batch.requests, vec![10]);
        assert_eq!(batcher.waiting.len(), 1);

        batcher.finish_request(&mut scheduler, 10);
        let prefill_after_decode = batcher.schedule(&mut scheduler);
        assert_eq!(prefill_after_decode.requests, vec![2]);
    }

    #[test]
    fn decode_phase_never_falls_back_to_prefill_in_same_schedule_call() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(16, 4, HGALConfig::default());

        batcher.add_request(make_group(10, 4));
        let prefill_batch = batcher.schedule(&mut scheduler);
        assert_eq!(prefill_batch.requests, vec![10]);

        batcher.add_request(make_group(2, 2));
        scheduler.free_sequence(10);

        // Decode queue is non-empty at call start, so this round must stay decode-only.
        let decode_only_batch = batcher.schedule(&mut scheduler);
        assert!(decode_only_batch.requests.is_empty());
        assert_eq!(batcher.waiting.len(), 1);

        // Prefill is allowed only in the next round after running becomes empty.
        let prefill_batch = batcher.schedule(&mut scheduler);
        assert_eq!(prefill_batch.requests, vec![2]);
    }

    #[test]
    fn schedule_returns_sorted_ids_for_prefill_and_decode() {
        let mut batcher = ContinuousBatcher::new();
        let mut scheduler = PagedScheduler::new(32, 4, HGALConfig::default());

        batcher.add_request(make_group(7, 1));
        batcher.add_request(make_group(3, 1));
        batcher.add_request(make_group(5, 1));

        let prefill_batch = batcher.schedule(&mut scheduler);
        assert_eq!(prefill_batch.requests, vec![3, 5, 7]);

        let decode_batch = batcher.schedule(&mut scheduler);
        assert_eq!(decode_batch.requests, vec![3, 5, 7]);
    }
}
