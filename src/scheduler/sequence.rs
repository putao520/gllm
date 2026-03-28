use std::time::Instant;

use super::types::{PageId, RequestId};

use super::types::{GroupState, SequenceGroup};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    Waiting,
    Running,
    Paused,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: RequestId,
    pub enqueue_order: u64,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub state: SequenceState,
    pub kv_pages: Vec<PageId>,
    pub position: usize,
    pub telemetry: crate::scheduler::telemetry::SequenceTelemetry,
    pub draft_budget: usize,
}

impl Sequence {
    pub fn new(id: RequestId, prompt_tokens: Vec<u32>) -> Self {
        let position = prompt_tokens.len();
        Self {
            id,
            enqueue_order: 0,
            prompt_tokens,
            generated_tokens: Vec::new(),
            state: SequenceState::Waiting,
            kv_pages: Vec::new(),
            position,
            telemetry: crate::scheduler::telemetry::SequenceTelemetry::new(),
            draft_budget: 0,
        }
    }

    pub fn context_len(&self) -> usize {
        self.position
    }

    pub fn needs_prefill(&self) -> bool {
        self.generated_tokens.is_empty()
    }

    pub fn to_sequence_group(&self) -> SequenceGroup {
        SequenceGroup {
            id: self.id,
            pages: self.kv_pages.clone(),
            state: GroupState::Running,
            access_count: 0,
            last_access: Instant::now(),
            is_pinned: false,
            context_len: self.context_len(),
        }
    }

    pub fn mark_running(&mut self, kv_pages: Vec<PageId>) {
        self.state = SequenceState::Running;
        self.kv_pages = kv_pages;
    }

    pub fn push_generated_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
        self.position = self.position.saturating_add(1);
    }
}
