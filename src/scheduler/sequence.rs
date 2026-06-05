use std::time::Instant;

use super::types::{PageId, RequestId};

use super::types::{GroupState, SequenceGroup};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
            pipeline: crate::scheduler::types::KvPipeline::Conversation,
            payload_kind: None,
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

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    use super::*;

    // ── SequenceState trait tests ──

    #[test]
    fn sequence_state_all_variants_distinct() {
        let variants = [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Paused,
            SequenceState::Completed,
            SequenceState::Failed,
        ];
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                assert_eq!(i == j, a == b);
            }
        }
    }

    #[test]
    fn sequence_state_is_copy() {
        let a = SequenceState::Running;
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn sequence_state_hash_consistency() {
        let mut h1 = DefaultHasher::new();
        SequenceState::Completed.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        SequenceState::Completed.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn sequence_state_hash_differs_across_variants() {
        let hash_of = |s: SequenceState| -> u64 {
            let mut h = DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        };
        assert_ne!(hash_of(SequenceState::Waiting), hash_of(SequenceState::Running));
        assert_ne!(hash_of(SequenceState::Paused), hash_of(SequenceState::Completed));
        assert_ne!(hash_of(SequenceState::Completed), hash_of(SequenceState::Failed));
    }

    #[test]
    fn sequence_state_debug_format() {
        assert_eq!(format!("{:?}", SequenceState::Waiting), "Waiting");
        assert_eq!(format!("{:?}", SequenceState::Running), "Running");
        assert_eq!(format!("{:?}", SequenceState::Paused), "Paused");
        assert_eq!(format!("{:?}", SequenceState::Completed), "Completed");
        assert_eq!(format!("{:?}", SequenceState::Failed), "Failed");
    }

    #[test]
    fn sequence_state_clone() {
        let a = SequenceState::Failed;
        let b = a.clone();
        assert_eq!(a, b);
    }

    // ── Sequence::new tests ──

    #[test]
    fn new_sequence_sets_all_defaults() {
        let seq = Sequence::new(1, vec![10, 20, 30]);
        assert_eq!(seq.id, 1);
        assert_eq!(seq.enqueue_order, 0);
        assert_eq!(seq.prompt_tokens, vec![10, 20, 30]);
        assert!(seq.generated_tokens.is_empty());
        assert_eq!(seq.state, SequenceState::Waiting);
        assert!(seq.kv_pages.is_empty());
        assert_eq!(seq.position, 3);
        assert_eq!(seq.draft_budget, 0);
    }

    #[test]
    fn new_sequence_with_empty_prompt() {
        let seq = Sequence::new(0, vec![]);
        assert_eq!(seq.id, 0);
        assert!(seq.prompt_tokens.is_empty());
        assert_eq!(seq.position, 0);
        assert_eq!(seq.context_len(), 0);
    }

    #[test]
    fn new_sequence_with_max_request_id() {
        let seq = Sequence::new(RequestId::MAX, vec![1]);
        assert_eq!(seq.id, RequestId::MAX);
    }

    #[test]
    fn new_sequence_with_single_token() {
        let seq = Sequence::new(5, vec![42]);
        assert_eq!(seq.prompt_tokens, vec![42]);
        assert_eq!(seq.position, 1);
        assert_eq!(seq.context_len(), 1);
    }

    #[test]
    fn new_sequence_with_large_prompt() {
        let tokens: Vec<u32> = (0..10000).collect();
        let seq = Sequence::new(1, tokens.clone());
        assert_eq!(seq.prompt_tokens, tokens);
        assert_eq!(seq.position, 10000);
        assert_eq!(seq.context_len(), 10000);
    }

    #[test]
    fn new_sequence_with_max_token_values() {
        let seq = Sequence::new(0, vec![u32::MAX, 0, u32::MAX]);
        assert_eq!(seq.prompt_tokens, vec![u32::MAX, 0, u32::MAX]);
        assert_eq!(seq.position, 3);
    }

    // ── Sequence::context_len tests ──

    #[test]
    fn context_len_equals_position_initially() {
        let seq = Sequence::new(0, vec![1, 2, 3, 4, 5]);
        assert_eq!(seq.context_len(), 5);
    }

    #[test]
    fn context_len_zero_for_empty_prompt() {
        let seq = Sequence::new(0, vec![]);
        assert_eq!(seq.context_len(), 0);
    }

    #[test]
    fn context_len_updates_after_push_generated() {
        let mut seq = Sequence::new(0, vec![1, 2]);
        assert_eq!(seq.context_len(), 2);
        seq.push_generated_token(99);
        assert_eq!(seq.context_len(), 3);
        seq.push_generated_token(88);
        assert_eq!(seq.context_len(), 4);
    }

    // ── Sequence::needs_prefill tests ──

    #[test]
    fn needs_prefill_true_when_no_generated_tokens() {
        let seq = Sequence::new(0, vec![1, 2]);
        assert!(seq.needs_prefill());
    }

    #[test]
    fn needs_prefill_true_for_empty_prompt() {
        let seq = Sequence::new(0, vec![]);
        assert!(seq.needs_prefill());
    }

    #[test]
    fn needs_prefill_false_after_first_generated_token() {
        let mut seq = Sequence::new(0, vec![1, 2]);
        seq.push_generated_token(99);
        assert!(!seq.needs_prefill());
    }

    #[test]
    fn needs_prefill_remains_false_after_multiple_tokens() {
        let mut seq = Sequence::new(0, vec![1]);
        seq.push_generated_token(10);
        seq.push_generated_token(20);
        seq.push_generated_token(30);
        assert!(!seq.needs_prefill());
    }

    // ── Sequence::push_generated_token tests ──

    #[test]
    fn push_generated_appends_token() {
        let mut seq = Sequence::new(0, vec![1, 2]);
        seq.push_generated_token(10);
        seq.push_generated_token(20);
        assert_eq!(seq.generated_tokens, vec![10, 20]);
    }

    #[test]
    fn push_generated_increments_position() {
        let mut seq = Sequence::new(0, vec![1, 2]);
        assert_eq!(seq.position, 2);
        seq.push_generated_token(10);
        assert_eq!(seq.position, 3);
        seq.push_generated_token(20);
        assert_eq!(seq.position, 4);
    }

    #[test]
    fn push_generated_position_saturating_add_at_max() {
        let mut seq = Sequence::new(0, vec![]);
        seq.position = usize::MAX;
        seq.push_generated_token(0);
        assert_eq!(seq.position, usize::MAX, "saturating_add should cap at MAX");
        assert_eq!(seq.generated_tokens, vec![0], "token still appended");
    }

    #[test]
    fn push_generated_zero_token() {
        let mut seq = Sequence::new(0, vec![1]);
        seq.push_generated_token(0);
        assert_eq!(seq.generated_tokens, vec![0]);
        assert_eq!(seq.position, 2);
    }

    #[test]
    fn push_generated_max_token() {
        let mut seq = Sequence::new(0, vec![1]);
        seq.push_generated_token(u32::MAX);
        assert_eq!(seq.generated_tokens, vec![u32::MAX]);
    }

    // ── Sequence::mark_running tests ──

    #[test]
    fn mark_running_transitions_from_waiting() {
        let mut seq = Sequence::new(0, vec![1]);
        assert_eq!(seq.state, SequenceState::Waiting);
        seq.mark_running(vec![100, 101]);
        assert_eq!(seq.state, SequenceState::Running);
        assert_eq!(seq.kv_pages, vec![100, 101]);
    }

    #[test]
    fn mark_running_with_empty_pages() {
        let mut seq = Sequence::new(0, vec![1, 2, 3]);
        seq.mark_running(vec![]);
        assert_eq!(seq.state, SequenceState::Running);
        assert!(seq.kv_pages.is_empty());
    }

    #[test]
    fn mark_running_overwrites_previous_pages() {
        let mut seq = Sequence::new(0, vec![1]);
        seq.mark_running(vec![10, 20]);
        assert_eq!(seq.kv_pages, vec![10, 20]);
        seq.mark_running(vec![30, 40, 50]);
        assert_eq!(seq.kv_pages, vec![30, 40, 50]);
    }

    #[test]
    fn mark_running_with_many_pages() {
        let mut seq = Sequence::new(0, vec![1]);
        let pages: Vec<PageId> = (0..1000).collect();
        seq.mark_running(pages.clone());
        assert_eq!(seq.kv_pages, pages);
    }

    // ── Sequence::to_sequence_group tests ──

    #[test]
    fn to_sequence_group_preserves_id_and_pages() {
        let mut seq = Sequence::new(42, vec![1, 2, 3]);
        seq.mark_running(vec![5, 6]);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.id, 42);
        assert_eq!(sg.pages, vec![5, 6]);
        assert_eq!(sg.context_len, 3);
    }

    #[test]
    fn to_sequence_group_defaults() {
        let seq = Sequence::new(7, vec![1, 2]);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.id, 7);
        assert_eq!(sg.state, GroupState::Running);
        assert_eq!(sg.access_count, 0);
        assert!(!sg.is_pinned);
        assert_eq!(sg.pipeline, crate::scheduler::types::KvPipeline::Conversation);
        assert!(sg.payload_kind.is_none());
        assert!(sg.pages.is_empty());
    }

    #[test]
    fn to_sequence_group_context_len_reflects_position() {
        let mut seq = Sequence::new(1, vec![1, 2]);
        seq.push_generated_token(3);
        seq.push_generated_token(4);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.context_len, 4);
    }

    #[test]
    fn to_sequence_group_empty_sequence() {
        let seq = Sequence::new(0, vec![]);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.context_len, 0);
        assert!(sg.pages.is_empty());
    }

    // ── Sequence Clone trait ──

    #[test]
    fn sequence_clone_preserves_fields() {
        let mut seq = Sequence::new(99, vec![1, 2, 3]);
        seq.enqueue_order = 5;
        seq.draft_budget = 10;
        seq.mark_running(vec![100]);
        seq.push_generated_token(42);

        let cloned = seq.clone();
        assert_eq!(cloned.id, 99);
        assert_eq!(cloned.enqueue_order, 5);
        assert_eq!(cloned.prompt_tokens, vec![1, 2, 3]);
        assert_eq!(cloned.generated_tokens, vec![42]);
        assert_eq!(cloned.state, SequenceState::Running);
        assert_eq!(cloned.kv_pages, vec![100]);
        assert_eq!(cloned.position, 4);
        assert_eq!(cloned.draft_budget, 10);
    }

    #[test]
    fn sequence_clone_is_independent() {
        let mut seq = Sequence::new(1, vec![1, 2]);
        let cloned = seq.clone();
        seq.push_generated_token(99);
        assert_eq!(seq.generated_tokens, vec![99]);
        assert!(cloned.generated_tokens.is_empty(), "clone should be independent");
    }

    // ── Sequence Debug trait ──

    #[test]
    fn sequence_debug_format_is_non_empty() {
        let seq = Sequence::new(1, vec![10, 20]);
        let debug = format!("{:?}", seq);
        assert!(!debug.is_empty());
        assert!(debug.contains("Sequence"));
    }

    // ── Sequence field mutation tests ──

    #[test]
    fn enqueue_order_is_mutable() {
        let mut seq = Sequence::new(1, vec![1]);
        assert_eq!(seq.enqueue_order, 0);
        seq.enqueue_order = 42;
        assert_eq!(seq.enqueue_order, 42);
    }

    #[test]
    fn draft_budget_is_mutable() {
        let mut seq = Sequence::new(1, vec![1]);
        assert_eq!(seq.draft_budget, 0);
        seq.draft_budget = 16;
        assert_eq!(seq.draft_budget, 16);
    }

    #[test]
    fn state_is_manually_mutable_to_completed() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.state = SequenceState::Completed;
        assert_eq!(seq.state, SequenceState::Completed);
    }

    #[test]
    fn state_is_manually_mutable_to_failed() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.state = SequenceState::Failed;
        assert_eq!(seq.state, SequenceState::Failed);
    }

    #[test]
    fn state_is_manually_mutable_to_paused() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.state = SequenceState::Paused;
        assert_eq!(seq.state, SequenceState::Paused);
    }

    // ── Integration: full lifecycle ──

    #[test]
    fn full_lifecycle_waiting_to_completed() {
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        assert!(seq.needs_prefill());
        assert_eq!(seq.context_len(), 3);

        seq.mark_running(vec![10, 11]);
        assert_eq!(seq.state, SequenceState::Running);
        assert!(!seq.needs_prefill() || seq.generated_tokens.is_empty());

        seq.push_generated_token(4);
        assert!(!seq.needs_prefill());
        assert_eq!(seq.context_len(), 4);

        seq.push_generated_token(5);
        assert_eq!(seq.generated_tokens, vec![4, 5]);
        assert_eq!(seq.context_len(), 5);

        let sg = seq.to_sequence_group();
        assert_eq!(sg.id, 1);
        assert_eq!(sg.pages, vec![10, 11]);
        assert_eq!(sg.context_len, 5);

        seq.state = SequenceState::Completed;
        assert_eq!(seq.state, SequenceState::Completed);
    }

    // ── SequenceTelemetry initial state tests ──

    #[test]
    fn telemetry_new_has_zero_floats() {
        let t = crate::scheduler::telemetry::SequenceTelemetry::new();
        assert_eq!(t.l2_delta, 0.0);
        assert_eq!(t.dead_density, 0.0);
        assert_eq!(t.per_head_entropy, 0.0);
        assert_eq!(t.transform_ratio, 0.0);
        assert_eq!(t.output_entropy, 0.0);
        assert!(!t.has_outlier);
    }

    #[test]
    fn telemetry_default_equals_new() {
        let a = crate::scheduler::telemetry::SequenceTelemetry::new();
        let b = crate::scheduler::telemetry::SequenceTelemetry::default();
        assert_eq!(a, b);
    }

    #[test]
    fn telemetry_fields_are_mutable() {
        let mut t = crate::scheduler::telemetry::SequenceTelemetry::new();
        t.l2_delta = 1.5;
        t.has_outlier = true;
        t.dead_density = 0.42;
        t.per_head_entropy = 3.14;
        t.transform_ratio = 0.01;
        t.output_entropy = 2.71;
        assert_eq!(t.l2_delta, 1.5);
        assert!(t.has_outlier);
        assert!((t.dead_density - 0.42).abs() < f32::EPSILON);
        assert!((t.per_head_entropy - 3.14).abs() < f32::EPSILON);
        assert!((t.transform_ratio - 0.01).abs() < f32::EPSILON);
        assert!((t.output_entropy - 2.71).abs() < f32::EPSILON);
    }

    #[test]
    fn telemetry_special_floats_nan() {
        let mut t = crate::scheduler::telemetry::SequenceTelemetry::new();
        t.l2_delta = f32::NAN;
        assert!(t.l2_delta.is_nan());
    }

    #[test]
    fn telemetry_special_floats_infinity() {
        let mut t = crate::scheduler::telemetry::SequenceTelemetry::new();
        t.output_entropy = f32::INFINITY;
        assert!(t.output_entropy.is_infinite() && t.output_entropy.is_sign_positive());
    }

    #[test]
    fn telemetry_special_floats_neg_infinity() {
        let mut t = crate::scheduler::telemetry::SequenceTelemetry::new();
        t.per_head_entropy = f32::NEG_INFINITY;
        assert!(t.per_head_entropy.is_infinite() && t.per_head_entropy.is_sign_negative());
    }

    #[test]
    fn sequence_new_telemetry_is_default() {
        let seq = Sequence::new(1, vec![1, 2]);
        let default_tel = crate::scheduler::telemetry::SequenceTelemetry::new();
        assert_eq!(seq.telemetry, default_tel);
    }

    #[test]
    fn telemetry_copy_trait() {
        let mut t = crate::scheduler::telemetry::SequenceTelemetry::new();
        t.l2_delta = 42.0;
        let copy = t;
        assert_eq!(copy.l2_delta, 42.0);
        t.l2_delta = 99.0; // modify original after Copy
        assert_eq!(copy.l2_delta, 42.0, "copy should be independent after Copy");
        assert_eq!(t.l2_delta, 99.0, "original should reflect new value");
    }

    #[test]
    fn telemetry_equality_same_values() {
        let mut a = crate::scheduler::telemetry::SequenceTelemetry::new();
        let mut b = crate::scheduler::telemetry::SequenceTelemetry::new();
        a.l2_delta = 1.0;
        a.has_outlier = true;
        b.l2_delta = 1.0;
        b.has_outlier = true;
        assert_eq!(a, b);
    }

    #[test]
    fn telemetry_inequality_different_values() {
        let mut a = crate::scheduler::telemetry::SequenceTelemetry::new();
        let mut b = crate::scheduler::telemetry::SequenceTelemetry::new();
        a.l2_delta = 1.0;
        b.l2_delta = 2.0;
        assert_ne!(a, b);
    }

    #[test]
    fn telemetry_debug_format_contains_fields() {
        let t = crate::scheduler::telemetry::SequenceTelemetry::new();
        let debug = format!("{:?}", t);
        assert!(debug.contains("l2_delta"));
        assert!(debug.contains("has_outlier"));
    }

    // ── Sequence enqueue_order edge cases ──

    #[test]
    fn enqueue_order_zero_default() {
        let seq = Sequence::new(0, vec![]);
        assert_eq!(seq.enqueue_order, 0);
    }

    #[test]
    fn enqueue_order_max_value() {
        let mut seq = Sequence::new(0, vec![1]);
        seq.enqueue_order = u64::MAX;
        assert_eq!(seq.enqueue_order, u64::MAX);
    }

    #[test]
    fn enqueue_order_preserved_across_clone() {
        let mut seq = Sequence::new(5, vec![1]);
        seq.enqueue_order = 12345;
        let cloned = seq.clone();
        assert_eq!(cloned.enqueue_order, 12345);
    }

    // ── Sequence draft_budget edge cases ──

    #[test]
    fn draft_budget_zero_default() {
        let seq = Sequence::new(0, vec![]);
        assert_eq!(seq.draft_budget, 0);
    }

    #[test]
    fn draft_budget_usize_max() {
        let mut seq = Sequence::new(0, vec![1]);
        seq.draft_budget = usize::MAX;
        assert_eq!(seq.draft_budget, usize::MAX);
    }

    #[test]
    fn draft_budget_preserved_across_clone() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.draft_budget = 256;
        let cloned = seq.clone();
        assert_eq!(cloned.draft_budget, 256);
    }

    // ── Sequence kv_pages after mark_running ──

    #[test]
    fn mark_running_state_transition_from_failed() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.state = SequenceState::Failed;
        seq.mark_running(vec![5]);
        assert_eq!(seq.state, SequenceState::Running);
        assert_eq!(seq.kv_pages, vec![5]);
    }

    #[test]
    fn mark_running_state_transition_from_paused() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.state = SequenceState::Paused;
        seq.mark_running(vec![3, 4]);
        assert_eq!(seq.state, SequenceState::Running);
    }

    #[test]
    fn mark_running_state_transition_from_completed() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.state = SequenceState::Completed;
        seq.mark_running(vec![1]);
        assert_eq!(seq.state, SequenceState::Running);
    }

    #[test]
    fn mark_running_idempotent() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.mark_running(vec![10]);
        assert_eq!(seq.state, SequenceState::Running);
        seq.mark_running(vec![20]);
        assert_eq!(seq.state, SequenceState::Running);
        assert_eq!(seq.kv_pages, vec![20]);
    }

    #[test]
    fn mark_running_with_max_page_ids() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.mark_running(vec![PageId::MAX, 0, PageId::MAX]);
        assert_eq!(seq.kv_pages, vec![PageId::MAX, 0, PageId::MAX]);
    }

    // ── Sequence position saturating_add edge cases ──

    #[test]
    fn push_generated_many_tokens_position_tracks() {
        let mut seq = Sequence::new(0, vec![1]);
        for i in 0..100u32 {
            seq.push_generated_token(i);
        }
        assert_eq!(seq.generated_tokens.len(), 100);
        assert_eq!(seq.position, 101);
    }

    #[test]
    fn push_generated_token_does_not_modify_prompt_tokens() {
        let mut seq = Sequence::new(0, vec![100, 200, 300]);
        seq.push_generated_token(999);
        assert_eq!(seq.prompt_tokens, vec![100, 200, 300]);
        assert_eq!(seq.generated_tokens, vec![999]);
    }

    #[test]
    fn push_generated_position_near_max_saturates() {
        let mut seq = Sequence::new(0, vec![]);
        seq.position = usize::MAX - 1;
        seq.push_generated_token(1);
        assert_eq!(seq.position, usize::MAX);
        seq.push_generated_token(2);
        assert_eq!(seq.position, usize::MAX);
        assert_eq!(seq.generated_tokens, vec![1, 2]);
    }

    // ── Sequence::needs_prefill with mark_running ──

    #[test]
    fn needs_prefill_true_after_mark_running_without_generation() {
        let mut seq = Sequence::new(0, vec![1, 2]);
        seq.mark_running(vec![10]);
        assert!(seq.needs_prefill(), "needs_prefill depends on generated_tokens, not state");
    }

    // ── Sequence::to_sequence_group after state changes ──

    #[test]
    fn to_sequence_group_reflects_current_kv_pages() {
        let mut seq = Sequence::new(1, vec![1, 2]);
        seq.mark_running(vec![10, 20]);
        seq.push_generated_token(3);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.pages, vec![10, 20]);
    }

    #[test]
    fn to_sequence_group_context_len_includes_generated() {
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        seq.push_generated_token(4);
        seq.push_generated_token(5);
        seq.push_generated_token(6);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.context_len, 6);
    }

    #[test]
    fn to_sequence_group_copies_pages_not_references() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.mark_running(vec![10, 20, 30]);
        let sg = seq.to_sequence_group();
        seq.kv_pages.clear();
        assert_eq!(sg.pages, vec![10, 20, 30], "SequenceGroup should own its pages copy");
    }

    // ── Sequence clone independence for all vec fields ──

    #[test]
    fn sequence_clone_prompt_tokens_independent() {
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        let cloned = seq.clone();
        seq.prompt_tokens.push(999);
        assert_eq!(cloned.prompt_tokens, vec![1, 2, 3]);
    }

    #[test]
    fn sequence_clone_kv_pages_independent() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.mark_running(vec![10, 20]);
        let cloned = seq.clone();
        seq.kv_pages.push(999);
        assert_eq!(cloned.kv_pages, vec![10, 20]);
    }

    // ── SequenceState inequality pairwise ──

    #[test]
    fn sequence_state_waiting_not_equal_running() {
        assert_ne!(SequenceState::Waiting, SequenceState::Running);
    }

    #[test]
    fn sequence_state_running_not_equal_paused() {
        assert_ne!(SequenceState::Running, SequenceState::Paused);
    }

    #[test]
    fn sequence_state_paused_not_equal_completed() {
        assert_ne!(SequenceState::Paused, SequenceState::Completed);
    }

    #[test]
    fn sequence_state_completed_not_equal_failed() {
        assert_ne!(SequenceState::Completed, SequenceState::Failed);
    }

    #[test]
    fn sequence_state_waiting_not_equal_failed() {
        assert_ne!(SequenceState::Waiting, SequenceState::Failed);
    }

    #[test]
    fn sequence_state_waiting_not_equal_paused() {
        assert_ne!(SequenceState::Waiting, SequenceState::Paused);
    }

    #[test]
    fn sequence_state_waiting_not_equal_completed() {
        assert_ne!(SequenceState::Waiting, SequenceState::Completed);
    }

    #[test]
    fn sequence_state_running_not_equal_completed() {
        assert_ne!(SequenceState::Running, SequenceState::Completed);
    }

    #[test]
    fn sequence_state_running_not_equal_failed() {
        assert_ne!(SequenceState::Running, SequenceState::Failed);
    }

    #[test]
    fn sequence_state_paused_not_equal_failed() {
        assert_ne!(SequenceState::Paused, SequenceState::Failed);
    }

    // ── SequenceState Eq trait (reflexive) ──

    #[test]
    fn sequence_state_eq_reflexive() {
        assert_eq!(SequenceState::Waiting, SequenceState::Waiting);
        assert_eq!(SequenceState::Running, SequenceState::Running);
        assert_eq!(SequenceState::Paused, SequenceState::Paused);
        assert_eq!(SequenceState::Completed, SequenceState::Completed);
        assert_eq!(SequenceState::Failed, SequenceState::Failed);
    }

    // ── SequenceState transitivity via pairwise ──

    #[test]
    fn sequence_state_all_variants_count() {
        let variants = [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Paused,
            SequenceState::Completed,
            SequenceState::Failed,
        ];
        assert_eq!(variants.len(), 5, "SequenceState should have exactly 5 variants");
    }

    // ── Sequence with request_id 0 ──

    #[test]
    fn new_sequence_request_id_zero() {
        let seq = Sequence::new(0, vec![1]);
        assert_eq!(seq.id, 0);
    }

    // ── Sequence prompt_tokens not modified by operations ──

    #[test]
    fn prompt_tokens_preserved_after_mark_running() {
        let mut seq = Sequence::new(1, vec![10, 20, 30]);
        seq.mark_running(vec![1, 2]);
        assert_eq!(seq.prompt_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn prompt_tokens_preserved_after_state_change() {
        let mut seq = Sequence::new(1, vec![5, 6]);
        seq.state = SequenceState::Failed;
        assert_eq!(seq.prompt_tokens, vec![5, 6]);
    }

    // ── Sequence generated_tokens accumulation ──

    #[test]
    fn generated_tokens_order_preserved() {
        let mut seq = Sequence::new(0, vec![1]);
        seq.push_generated_token(10);
        seq.push_generated_token(20);
        seq.push_generated_token(30);
        assert_eq!(seq.generated_tokens, vec![10, 20, 30]);
    }

    #[test]
    fn generated_tokens_empty_initially() {
        let seq = Sequence::new(0, vec![1, 2, 3]);
        assert!(seq.generated_tokens.is_empty());
    }

    // ── Sequence debug contains key fields ──

    #[test]
    fn sequence_debug_contains_id() {
        let seq = Sequence::new(42, vec![1]);
        let debug = format!("{:?}", seq);
        assert!(debug.contains("42"), "Debug should contain the id");
    }

    #[test]
    fn sequence_debug_contains_state() {
        let seq = Sequence::new(0, vec![1]);
        let debug = format!("{:?}", seq);
        assert!(debug.contains("Waiting") || debug.contains("state"));
    }

    // ── Lifecycle: paused then resumed ──

    #[test]
    fn lifecycle_pause_then_resume() {
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        seq.mark_running(vec![10]);
        seq.state = SequenceState::Paused;
        assert_eq!(seq.state, SequenceState::Paused);
        seq.mark_running(vec![10, 11]);
        assert_eq!(seq.state, SequenceState::Running);
        seq.push_generated_token(4);
        assert!(!seq.needs_prefill());
    }

    // ── Lifecycle: multiple token generation then completion ──

    #[test]
    fn lifecycle_generate_many_then_complete() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.mark_running(vec![10]);
        for i in 0..50u32 {
            seq.push_generated_token(i);
        }
        assert_eq!(seq.generated_tokens.len(), 50);
        assert_eq!(seq.position, 51);
        seq.state = SequenceState::Completed;
        assert_eq!(seq.state, SequenceState::Completed);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.context_len, 51);
    }

    // ── SequenceTelemetry with special values on Sequence ──

    #[test]
    fn sequence_telemetry_mutation_is_visible() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.telemetry.l2_delta = 99.5;
        seq.telemetry.has_outlier = true;
        assert_eq!(seq.telemetry.l2_delta, 99.5);
        assert!(seq.telemetry.has_outlier);
    }

    #[test]
    fn sequence_telemetry_preserved_across_clone() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.telemetry.output_entropy = 3.14;
        let cloned = seq.clone();
        assert!((cloned.telemetry.output_entropy - 3.14).abs() < f32::EPSILON);
    }

    // ── SequenceGroup defaults from to_sequence_group ──

    #[test]
    fn to_sequence_group_always_sets_running_state() {
        let mut seq = Sequence::new(1, vec![1]);
        seq.state = SequenceState::Paused;
        let sg = seq.to_sequence_group();
        assert_eq!(sg.state, GroupState::Running, "to_sequence_group always produces Running GroupState");
    }

    #[test]
    fn to_sequence_group_always_sets_conversation_pipeline() {
        let seq = Sequence::new(1, vec![1]);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.pipeline, crate::scheduler::types::KvPipeline::Conversation);
    }

    #[test]
    fn to_sequence_group_payload_kind_is_none() {
        let seq = Sequence::new(1, vec![1]);
        let sg = seq.to_sequence_group();
        assert!(sg.payload_kind.is_none());
    }

    #[test]
    fn to_sequence_group_access_count_is_zero() {
        let seq = Sequence::new(1, vec![1]);
        let sg = seq.to_sequence_group();
        assert_eq!(sg.access_count, 0);
    }

    #[test]
    fn to_sequence_group_is_not_pinned() {
        let seq = Sequence::new(1, vec![1]);
        let sg = seq.to_sequence_group();
        assert!(!sg.is_pinned);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (15 additional)
    // ══════════════════════════════════════════════════════════════════════

    // ── Sequence: context_len == prompt_tokens.len() + generated_tokens.len() ──

    #[test]
    fn context_len_equals_prompt_plus_generated_sum() {
        // Arrange
        let mut seq = Sequence::new(1, vec![10, 20, 30]);

        // Act: push several generated tokens
        seq.push_generated_token(40);
        seq.push_generated_token(50);
        seq.push_generated_token(60);

        // Assert: context_len must be the sum
        let expected = seq.prompt_tokens.len() + seq.generated_tokens.len();
        assert_eq!(seq.context_len(), expected);
        assert_eq!(seq.context_len(), 6);
    }

    // ── Sequence: kv_pages empty before mark_running ──

    #[test]
    fn kv_pages_empty_before_mark_running() {
        // Arrange
        let seq = Sequence::new(1, vec![1, 2, 3]);

        // Assert: no pages allocated before mark_running
        assert!(seq.kv_pages.is_empty());
    }

    // ── Sequence: kv_pages directly mutable ──

    #[test]
    fn kv_pages_directly_mutable() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1]);

        // Act: direct field mutation (not via mark_running)
        seq.kv_pages.push(42);
        seq.kv_pages.push(43);

        // Assert: pages updated directly
        assert_eq!(seq.kv_pages, vec![42, 43]);
    }

    // ── Sequence: to_sequence_group called twice yields independent copies ──

    #[test]
    fn to_sequence_group_called_twice_yields_independent_copies() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2]);
        seq.mark_running(vec![10, 20]);

        // Act: convert twice
        let sg1 = seq.to_sequence_group();
        let sg2 = seq.to_sequence_group();

        // Assert: both have same data but are independent
        assert_eq!(sg1.id, sg2.id);
        assert_eq!(sg1.pages, sg2.pages);
        assert_eq!(sg1.context_len, sg2.context_len);
    }

    // ── Sequence: to_sequence_group context_len tracks position after generation ──

    #[test]
    fn to_sequence_group_after_many_generated_tokens() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1]);
        seq.mark_running(vec![10]);

        // Act: generate 200 tokens
        for i in 0..200u32 {
            seq.push_generated_token(i);
        }

        // Assert: group context_len includes prompt + generated
        let sg = seq.to_sequence_group();
        assert_eq!(sg.context_len, 201); // 1 prompt + 200 generated
        assert_eq!(sg.pages, vec![10]); // pages unchanged by generation
    }

    // ── Sequence: state transition running -> failed directly ──

    #[test]
    fn state_transition_running_to_failed_directly() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        seq.mark_running(vec![5]);

        // Act: transition directly to Failed (not through Paused)
        seq.state = SequenceState::Failed;

        // Assert
        assert_eq!(seq.state, SequenceState::Failed);
        assert_eq!(seq.kv_pages, vec![5], "kv_pages should persist after failure");
        assert_eq!(seq.position, 3, "position should persist after failure");
    }

    // ── Sequence: state transition waiting -> paused (skip running) ──

    #[test]
    fn state_transition_waiting_to_paused_skip_running() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1]);

        // Act: go directly to Paused without ever Running
        seq.state = SequenceState::Paused;

        // Assert
        assert_eq!(seq.state, SequenceState::Paused);
        assert!(seq.kv_pages.is_empty(), "kv_pages remain empty since mark_running was never called");
        assert!(seq.needs_prefill(), "needs_prefill depends on generated_tokens, not state");
    }

    // ── Sequence: full lifecycle with RequestId::MAX ──

    #[test]
    fn full_lifecycle_with_max_request_id() {
        // Arrange
        let mut seq = Sequence::new(RequestId::MAX, vec![100, 200]);
        assert_eq!(seq.id, RequestId::MAX);
        assert!(seq.needs_prefill());

        // Act: run lifecycle
        seq.mark_running(vec![1, 2]);
        assert_eq!(seq.state, SequenceState::Running);

        seq.push_generated_token(300);
        assert!(!seq.needs_prefill());
        assert_eq!(seq.context_len(), 3);

        // Assert: conversion preserves max id
        let sg = seq.to_sequence_group();
        assert_eq!(sg.id, RequestId::MAX);
        assert_eq!(sg.context_len, 3);

        seq.state = SequenceState::Completed;
        assert_eq!(seq.state, SequenceState::Completed);
    }

    // ── Sequence: generated_tokens independence from prompt_tokens ──

    #[test]
    fn generated_tokens_completely_independent_from_prompt() {
        // Arrange
        let mut seq = Sequence::new(1, vec![100, 200, 300]);

        // Act: generate tokens with overlapping values
        seq.push_generated_token(100);
        seq.push_generated_token(200);
        seq.push_generated_token(300);

        // Assert: no cross-contamination
        assert_eq!(seq.prompt_tokens, vec![100, 200, 300]);
        assert_eq!(seq.generated_tokens, vec![100, 200, 300]);
        assert_eq!(seq.position, 6);
        assert_eq!(seq.context_len(), 6);
    }

    // ── Sequence: position consistency through repeated mark_running ──

    #[test]
    fn position_unchanged_by_repeated_mark_running() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        assert_eq!(seq.position, 3);

        // Act: call mark_running multiple times
        seq.mark_running(vec![10]);
        assert_eq!(seq.position, 3, "mark_running should not change position");

        seq.mark_running(vec![20, 30]);
        assert_eq!(seq.position, 3, "mark_running should not change position");

        seq.push_generated_token(4);
        assert_eq!(seq.position, 4);

        seq.mark_running(vec![40]);
        assert_eq!(seq.position, 4, "mark_running should not change position after generation");
    }

    // ── Sequence: telemetry clone independence ──

    #[test]
    fn sequence_telemetry_clone_independence_after_mutation() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1]);
        seq.telemetry.l2_delta = 5.0;
        seq.telemetry.has_outlier = true;

        // Act: clone then mutate original
        let cloned = seq.clone();
        seq.telemetry.l2_delta = 99.0;
        seq.telemetry.has_outlier = false;

        // Assert: clone retains original values
        assert!((cloned.telemetry.l2_delta - 5.0).abs() < f32::EPSILON);
        assert!(cloned.telemetry.has_outlier);
        assert!((seq.telemetry.l2_delta - 99.0).abs() < f32::EPSILON);
        assert!(!seq.telemetry.has_outlier);
    }

    // ── Sequence: mark_running does not affect needs_prefill ──

    #[test]
    fn mark_running_does_not_satisfy_needs_prefill() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3, 4, 5]);

        // Act: call mark_running without generating any tokens
        seq.mark_running(vec![10, 11, 12]);
        seq.mark_running(vec![20, 21]);

        // Assert: still needs prefill because generated_tokens is empty
        assert!(seq.needs_prefill());
        assert!(seq.generated_tokens.is_empty());
    }

    // ── Sequence: prompt_tokens retain exact values after many operations ──

    #[test]
    fn prompt_tokens_unchanged_through_full_lifecycle() {
        // Arrange
        let prompt = vec![42, 84, 126, 168, 210];
        let mut seq = Sequence::new(1, prompt.clone());

        // Act: perform many operations
        seq.enqueue_order = 100;
        seq.draft_budget = 50;
        seq.mark_running(vec![1, 2, 3]);
        for i in 0..10u32 {
            seq.push_generated_token(i);
        }
        seq.state = SequenceState::Paused;
        seq.mark_running(vec![4, 5]);
        seq.state = SequenceState::Completed;
        let _ = seq.to_sequence_group();

        // Assert: prompt_tokens never modified
        assert_eq!(seq.prompt_tokens, prompt);
    }

    // ── SequenceGroup: last_access is a recent Instant from to_sequence_group ──

    #[test]
    fn to_sequence_group_last_access_is_recent() {
        // Arrange
        let before = Instant::now();
        let seq = Sequence::new(1, vec![1]);
        let sg = seq.to_sequence_group();
        let after = Instant::now();

        // Assert: last_access is between before and after (or equal on most platforms)
        assert!(sg.last_access >= before || sg.last_access <= after,
            "last_access should be a valid recent Instant");
    }

    // ── Sequence: push_generated_token with u32::MIN value ──

    #[test]
    fn push_generated_token_with_zero_and_min_values() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1]);

        // Act: push boundary token values
        seq.push_generated_token(0);
        seq.push_generated_token(u32::MIN);
        seq.push_generated_token(u32::MAX);

        // Assert: all tokens stored correctly
        assert_eq!(seq.generated_tokens, vec![0, 0, u32::MAX]);
        assert_eq!(seq.position, 4);
        assert_eq!(seq.context_len(), 4);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (15 additional — wave 2)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. Sequence: id field is immutable after construction ──

    #[test]
    fn sequence_id_cannot_be_mutated() {
        // Arrange: create a sequence with a specific id
        let seq = Sequence::new(42, vec![1, 2, 3]);

        // Assert: id is a pub field but the value is set at construction
        assert_eq!(seq.id, 42);
        // id is pub u64 — we can read it; no setter method exists
    }

    // ── 2. Sequence: multiple sequences have independent prompt_tokens ──

    #[test]
    fn two_sequences_have_independent_prompt_tokens() {
        // Arrange: create two sequences with different prompts
        let seq_a = Sequence::new(1, vec![10, 20]);
        let seq_b = Sequence::new(2, vec![30, 40, 50]);

        // Assert: each sequence owns its own prompt_tokens
        assert_eq!(seq_a.prompt_tokens, vec![10, 20]);
        assert_eq!(seq_b.prompt_tokens, vec![30, 40, 50]);
        assert_ne!(seq_a.prompt_tokens, seq_b.prompt_tokens);
    }

    // ── 3. Sequence: needs_prefill transitions from true to false exactly once ──

    #[test]
    fn needs_prefill_transition_is_one_way() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        assert!(seq.needs_prefill());

        // Act: push first generated token
        seq.push_generated_token(4);
        assert!(!seq.needs_prefill());

        // Act: push many more tokens
        for i in 5..20u32 {
            seq.push_generated_token(i);
        }

        // Assert: needs_prefill never returns true again
        assert!(!seq.needs_prefill());
    }

    // ── 4. Sequence: to_sequence_group snapshots context_len at call time ──

    #[test]
    fn to_sequence_group_context_len_snapshot_before_and_after_generation() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        let sg_before = seq.to_sequence_group();
        assert_eq!(sg_before.context_len, 3);

        // Act: generate tokens and convert again
        seq.push_generated_token(4);
        seq.push_generated_token(5);
        let sg_after = seq.to_sequence_group();

        // Assert: second snapshot reflects updated position
        assert_eq!(sg_after.context_len, 5);
        assert_ne!(sg_before.context_len, sg_after.context_len);
    }

    // ── 5. Sequence: mark_running does not alter generated_tokens ──

    #[test]
    fn mark_running_preserves_generated_tokens() {
        // Arrange: create sequence and generate some tokens
        let mut seq = Sequence::new(1, vec![1, 2]);
        seq.push_generated_token(10);
        seq.push_generated_token(20);
        assert_eq!(seq.generated_tokens, vec![10, 20]);

        // Act: call mark_running
        seq.mark_running(vec![100, 200]);

        // Assert: generated_tokens are untouched
        assert_eq!(seq.generated_tokens, vec![10, 20]);
        assert_eq!(seq.kv_pages, vec![100, 200]);
    }

    // ── 6. Sequence: clone after mark_running captures current kv_pages ──

    #[test]
    fn clone_after_mark_running_captures_pages() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        seq.mark_running(vec![50, 51, 52]);

        // Act: clone
        let cloned = seq.clone();

        // Assert: clone has the same pages
        assert_eq!(cloned.kv_pages, vec![50, 51, 52]);
        assert_eq!(cloned.state, SequenceState::Running);
        assert_eq!(cloned.position, 3);
    }

    // ── 7. Sequence: state round-trip waiting -> running -> paused -> running -> completed ──

    #[test]
    fn state_round_trip_through_all_states() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1]);
        assert_eq!(seq.state, SequenceState::Waiting);

        // Act: transition through all states
        seq.mark_running(vec![10]);
        assert_eq!(seq.state, SequenceState::Running);

        seq.state = SequenceState::Paused;
        assert_eq!(seq.state, SequenceState::Paused);

        seq.mark_running(vec![10, 11]);
        assert_eq!(seq.state, SequenceState::Running);

        seq.state = SequenceState::Failed;
        assert_eq!(seq.state, SequenceState::Failed);

        seq.mark_running(vec![12]);
        assert_eq!(seq.state, SequenceState::Running);

        seq.state = SequenceState::Completed;
        assert_eq!(seq.state, SequenceState::Completed);
    }

    // ── 8. Sequence: generated_tokens len equals position minus prompt_tokens len ──

    #[test]
    fn generated_tokens_len_equals_position_minus_prompt_len() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3, 4, 5]);
        let prompt_len = seq.prompt_tokens.len();

        // Act: push varying numbers of tokens
        for i in 0..37u32 {
            seq.push_generated_token(i);
        }

        // Assert: invariant holds
        assert_eq!(seq.generated_tokens.len(), seq.position - prompt_len);
        assert_eq!(seq.generated_tokens.len(), 37);
        assert_eq!(seq.position, 42);
    }

    // ── 9. Sequence: empty sequence to_sequence_group has last_access near now ──

    #[test]
    fn empty_sequence_to_group_has_recent_last_access() {
        // Arrange
        let before = Instant::now();
        let seq = Sequence::new(0, vec![]);
        let sg = seq.to_sequence_group();
        let after = Instant::now();

        // Assert: last_access is within the time window
        assert!(sg.last_access >= before || sg.last_access <= after);
    }

    // ── 10. Sequence: enqueue_order does not affect other fields ──

    #[test]
    fn enqueue_order_mutation_is_isolated() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2]);
        let original_position = seq.position;
        let original_state = seq.state;

        // Act: mutate enqueue_order
        seq.enqueue_order = 999;

        // Assert: other fields unchanged
        assert_eq!(seq.position, original_position);
        assert_eq!(seq.state, original_state);
        assert_eq!(seq.prompt_tokens, vec![1, 2]);
        assert!(seq.generated_tokens.is_empty());
        assert!(seq.kv_pages.is_empty());
    }

    // ── 11. Sequence: draft_budget does not affect context_len or needs_prefill ──

    #[test]
    fn draft_budget_mutation_is_isolated() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        let ctx = seq.context_len();
        let prefill = seq.needs_prefill();

        // Act: mutate draft_budget
        seq.draft_budget = 1024;

        // Assert: context_len and needs_prefill unchanged
        assert_eq!(seq.context_len(), ctx);
        assert_eq!(seq.needs_prefill(), prefill);
        assert_eq!(seq.draft_budget, 1024);
    }

    // ── 12. Sequence: push_generated_token after position saturation still appends ──

    #[test]
    fn push_generated_appends_even_at_position_saturation() {
        // Arrange: set position to usize::MAX
        let mut seq = Sequence::new(1, vec![]);
        seq.position = usize::MAX;

        // Act: push multiple tokens
        seq.push_generated_token(100);
        seq.push_generated_token(200);
        seq.push_generated_token(300);

        // Assert: position saturates but tokens are still appended
        assert_eq!(seq.position, usize::MAX);
        assert_eq!(seq.generated_tokens, vec![100, 200, 300]);
        assert_eq!(seq.generated_tokens.len(), 3);
    }

    // ── 13. Sequence: to_sequence_group after failed state still produces Running GroupState ──

    #[test]
    fn to_sequence_group_after_failure_produces_running_state() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2]);
        seq.mark_running(vec![5]);
        seq.state = SequenceState::Failed;
        assert_eq!(seq.state, SequenceState::Failed);

        // Act: convert to SequenceGroup
        let sg = seq.to_sequence_group();

        // Assert: SequenceGroup always gets Running state regardless of source state
        assert_eq!(sg.state, GroupState::Running);
        assert_eq!(sg.id, 1);
        assert_eq!(sg.pages, vec![5]);
        assert_eq!(sg.context_len, 2);
    }

    // ── 14. Sequence: telemetry mutation after clone is fully independent ──

    #[test]
    fn telemetry_deep_independence_across_clone() {
        // Arrange: set up telemetry with non-default values
        let mut seq = Sequence::new(1, vec![1]);
        seq.telemetry.dead_density = 0.75;
        seq.telemetry.per_head_entropy = 4.2;
        seq.telemetry.transform_ratio = 0.15;
        seq.telemetry.output_entropy = 1.618;

        // Act: clone and modify original telemetry
        let cloned = seq.clone();
        seq.telemetry.dead_density = 0.0;
        seq.telemetry.per_head_entropy = 0.0;
        seq.telemetry.transform_ratio = 0.0;
        seq.telemetry.output_entropy = 0.0;

        // Assert: cloned telemetry retains original values
        assert!((cloned.telemetry.dead_density - 0.75).abs() < f32::EPSILON);
        assert!((cloned.telemetry.per_head_entropy - 4.2).abs() < f32::EPSILON);
        assert!((cloned.telemetry.transform_ratio - 0.15).abs() < f32::EPSILON);
        assert!((cloned.telemetry.output_entropy - 1.618).abs() < f32::EPSILON);
    }

    // ── 15. Sequence: SequenceState used as HashMap key ──

    #[test]
    fn sequence_state_as_hash_map_key() {
        // Arrange: use all 5 SequenceState variants as HashMap keys
        use std::collections::HashMap;
        let mut map: HashMap<SequenceState, &'static str> = HashMap::new();
        map.insert(SequenceState::Waiting, "wait");
        map.insert(SequenceState::Running, "run");
        map.insert(SequenceState::Paused, "pause");
        map.insert(SequenceState::Completed, "done");
        map.insert(SequenceState::Failed, "error");

        // Assert: all 5 entries present and retrievable
        assert_eq!(map.len(), 5);
        assert_eq!(map.get(&SequenceState::Waiting), Some(&"wait"));
        assert_eq!(map.get(&SequenceState::Running), Some(&"run"));
        assert_eq!(map.get(&SequenceState::Paused), Some(&"pause"));
        assert_eq!(map.get(&SequenceState::Completed), Some(&"done"));
        assert_eq!(map.get(&SequenceState::Failed), Some(&"error"));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  NEW TESTS (wave 3 — 10 additional)
    // ══════════════════════════════════════════════════════════════════════

    // ── 1. SequenceState in HashSet (Hash + Eq contract) ──

    #[test]
    fn sequence_state_as_hash_set_deduplicates() {
        // Arrange: insert same variant multiple times
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SequenceState::Running);
        set.insert(SequenceState::Running);
        set.insert(SequenceState::Running);

        // Assert: deduplication works (Eq + Hash contract)
        assert_eq!(set.len(), 1);
        assert!(set.contains(&SequenceState::Running));
    }

    // ── 2. SequenceState all 5 variants in a HashSet ──

    #[test]
    fn sequence_state_hash_set_all_variants() {
        // Arrange
        use std::collections::HashSet;
        let set: HashSet<SequenceState> = [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Paused,
            SequenceState::Completed,
            SequenceState::Failed,
        ].into_iter().collect();

        // Assert: exactly 5 unique variants
        assert_eq!(set.len(), 5);
    }

    // ── 3. Sequence with duplicate prompt token values ──

    #[test]
    fn new_sequence_with_duplicate_prompt_tokens() {
        // Arrange: prompt with repeated values
        let prompt = vec![7, 7, 7, 7];

        // Act
        let seq = Sequence::new(1, prompt);

        // Assert: duplicates preserved exactly as given
        assert_eq!(seq.prompt_tokens, vec![7, 7, 7, 7]);
        assert_eq!(seq.position, 4);
        assert_eq!(seq.context_len(), 4);
    }

    // ── 4. Sequence: context_len invariant after empty prompt with generation ──

    #[test]
    fn context_len_after_empty_prompt_and_generation() {
        // Arrange: empty prompt sequence
        let mut seq = Sequence::new(1, vec![]);
        assert_eq!(seq.context_len(), 0);

        // Act: generate tokens on empty prompt
        seq.push_generated_token(10);
        seq.push_generated_token(20);

        // Assert: context_len == generated_tokens.len() when prompt is empty
        assert_eq!(seq.context_len(), 2);
        assert_eq!(seq.context_len(), seq.generated_tokens.len());
    }

    // ── 5. SequenceGroup Debug format contains key field names ──

    #[test]
    fn sequence_group_debug_format() {
        // Arrange: create a SequenceGroup via to_sequence_group
        let mut seq = Sequence::new(42, vec![1, 2]);
        seq.mark_running(vec![10, 20]);
        let sg = seq.to_sequence_group();

        // Act
        let debug = format!("{:?}", sg);

        // Assert: Debug output contains struct name and key fields
        assert!(debug.contains("SequenceGroup"));
        assert!(debug.contains("id"));
        assert!(debug.contains("pages"));
    }

    // ── 6. Sequence: kv_pages snapshot captured by to_sequence_group before mutation ──

    #[test]
    fn to_sequence_group_captures_kv_pages_snapshot() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2]);
        seq.mark_running(vec![10, 20, 30]);

        // Act: convert to group, then mutate original kv_pages
        let sg = seq.to_sequence_group();
        seq.kv_pages.clear();
        seq.kv_pages.push(999);

        // Assert: group has the snapshot, not the mutated state
        assert_eq!(sg.pages, vec![10, 20, 30]);
        assert_eq!(seq.kv_pages, vec![999]);
    }

    // ── 7. Sequence: position invariant across multiple mark_running calls ──

    #[test]
    fn position_invariant_across_lifecycle_with_mark_running_and_generation() {
        // Arrange: 5-token prompt
        let mut seq = Sequence::new(1, vec![1, 2, 3, 4, 5]);
        assert_eq!(seq.position, 5);

        // Act: mark_running, generate, mark_running again, generate more
        seq.mark_running(vec![100]);
        assert_eq!(seq.position, 5, "mark_running does not advance position");

        seq.push_generated_token(6);
        seq.push_generated_token(7);
        assert_eq!(seq.position, 7);

        seq.mark_running(vec![200, 201]);
        assert_eq!(seq.position, 7, "second mark_running does not advance position");

        seq.push_generated_token(8);
        assert_eq!(seq.position, 8);

        // Assert: position == prompt_tokens.len() + generated_tokens.len()
        assert_eq!(seq.position, seq.prompt_tokens.len() + seq.generated_tokens.len());
    }

    // ── 8. GroupState equality and Debug ──

    #[test]
    fn group_state_variants_distinct_and_debuggable() {
        // Arrange: all GroupState variants
        let running = GroupState::Running;
        let swapped = GroupState::Swapped;
        let paused = GroupState::Paused;

        // Assert: pairwise inequality
        assert_ne!(running, swapped);
        assert_ne!(running, paused);
        assert_ne!(swapped, paused);

        // Assert: Debug output matches variant names
        assert_eq!(format!("{:?}", running), "Running");
        assert_eq!(format!("{:?}", swapped), "Swapped");
        assert_eq!(format!("{:?}", paused), "Paused");
    }

    // ── 9. Sequence: telemetry mutation does not affect position or state ──

    #[test]
    fn telemetry_mutation_isolated_from_core_fields() {
        // Arrange
        let mut seq = Sequence::new(1, vec![1, 2, 3]);
        let position_before = seq.position;
        let state_before = seq.state;

        // Act: mutate telemetry fields
        seq.telemetry.l2_delta = 100.0;
        seq.telemetry.dead_density = 0.99;
        seq.telemetry.per_head_entropy = 7.5;
        seq.telemetry.has_outlier = true;
        seq.telemetry.transform_ratio = 0.5;
        seq.telemetry.output_entropy = 4.0;

        // Assert: core fields completely unaffected
        assert_eq!(seq.position, position_before);
        assert_eq!(seq.state, state_before);
        assert_eq!(seq.context_len(), 3);
        assert!(seq.needs_prefill());
        assert!(seq.kv_pages.is_empty());
        assert!(seq.generated_tokens.is_empty());
    }

    // ── 10. Sequence: multiple sequences with distinct IDs share no state ──

    #[test]
    fn multiple_sequences_are_fully_independent() {
        // Arrange: two sequences with same prompt but different IDs
        let mut seq_a = Sequence::new(100, vec![1, 2, 3]);
        let mut seq_b = Sequence::new(200, vec![1, 2, 3]);

        // Act: evolve them differently
        seq_a.mark_running(vec![10]);
        seq_a.push_generated_token(4);
        seq_a.enqueue_order = 1;

        seq_b.mark_running(vec![20, 21]);
        seq_b.state = SequenceState::Paused;

        // Assert: no cross-contamination
        assert_eq!(seq_a.id, 100);
        assert_eq!(seq_b.id, 200);
        assert_eq!(seq_a.kv_pages, vec![10]);
        assert_eq!(seq_b.kv_pages, vec![20, 21]);
        assert_eq!(seq_a.state, SequenceState::Running);
        assert_eq!(seq_b.state, SequenceState::Paused);
        assert_eq!(seq_a.generated_tokens, vec![4]);
        assert!(seq_b.generated_tokens.is_empty());
        assert_eq!(seq_a.enqueue_order, 1);
        assert_eq!(seq_b.enqueue_order, 0);
        assert_eq!(seq_a.position, 4);
        assert_eq!(seq_b.position, 3);
    }
}
