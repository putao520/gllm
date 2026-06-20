use std::collections::HashMap;

use crate::scheduler::chunked_prefill::ChunkedPrefillScheduler;
use crate::scheduler::memory_manager::GlobalMemoryManager;
use crate::scheduler::paged_scheduler::PagedScheduler;
use crate::scheduler::policy::PolicyVariant;
use crate::scheduler::types::RequestId;

use super::super::executor::RequestData;
use crate::scheduler::batcher::ContinuousBatcher;

pub struct DispatchCoordinator {
    pub scheduler: PagedScheduler,
    pub batcher: ContinuousBatcher,
    pub chunked_prefill_scheduler: ChunkedPrefillScheduler,
    pub requests: HashMap<RequestId, RequestData>,
    pub memory_manager: GlobalMemoryManager,
    pub policy: PolicyVariant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::chunked_prefill::ChunkedPrefillConfig;
    use crate::scheduler::hgal::HGALConfig;
    use crate::scheduler::request_state::RequestPhase;
    use crate::scheduler::vllm2024::ChunkedConfig;

    /// Helper: build a minimal `DispatchCoordinator` with default components.
    fn make_coordinator() -> DispatchCoordinator {
        DispatchCoordinator {
            scheduler: PagedScheduler::new(32, 4, HGALConfig::default()),
            batcher: ContinuousBatcher::new()
                .with_chunked(ChunkedConfig::default()),
            chunked_prefill_scheduler: ChunkedPrefillScheduler::new(
                ChunkedPrefillConfig::default(),
            ),
            requests: HashMap::new(),
            memory_manager: GlobalMemoryManager::new_with_capacities(32, 0, 0),
            policy: PolicyVariant::default(),
        }
    }

    // ── Construction ──────────────────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_construction_has_empty_requests() {
        // Arrange: construct a fresh coordinator
        let coord = make_coordinator();

        // Act & Assert: requests map starts empty
        assert!(coord.requests.is_empty());
        assert_eq!(coord.requests.len(), 0);
    }

    #[test]
    fn dispatch_coordinator_construction_policy_is_absolute() {
        // Arrange
        let coord = make_coordinator();

        // Act & Assert: default policy is Absolute variant
        assert!(matches!(coord.policy, PolicyVariant::Absolute));
    }

    #[test]
    fn dispatch_coordinator_construction_batcher_is_empty() {
        // Arrange
        let coord = make_coordinator();

        // Act & Assert: batcher starts with no pending work
        assert!(!coord.batcher.has_pending_work());
        assert_eq!(coord.batcher.waiting_len(), 0);
        assert_eq!(coord.batcher.running_len(), 0);
    }

    // ── Requests HashMap manipulation ─────────────────────────────────────

    #[test]
    fn dispatch_coordinator_requests_insert_and_lookup() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 100,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };

        // Act: insert request
        coord.requests.insert(42, req_data);

        // Assert: lookup succeeds
        assert!(coord.requests.contains_key(&42));
        assert_eq!(coord.requests.get(&42).unwrap().prompt_tokens, vec![1, 2, 3]);
        assert_eq!(coord.requests.get(&42).unwrap().phase, RequestPhase::Prefill);
        assert_eq!(coord.requests.get(&42).unwrap().max_new_tokens, 100);
    }

    #[test]
    fn dispatch_coordinator_requests_remove() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![10, 20],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 50,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(7, req_data);

        // Act
        let removed = coord.requests.remove(&7);

        // Assert
        assert!(removed.is_some());
        assert!(coord.requests.is_empty());
        assert!(!coord.requests.contains_key(&7));
    }

    #[test]
    fn dispatch_coordinator_requests_multiple_entries() {
        // Arrange
        let mut coord = make_coordinator();
        for i in 0..5 {
            let req_data = RequestData {
                prompt_tokens: vec![i],
                output_tokens: vec![],
                sampling_config: Default::default(),
                phase: RequestPhase::Prefill,
                max_new_tokens: i as usize * 10,
                finished: false,
                session_id: None,
                thinking_budget: None,
                fused_prefill_hidden: None,
            };
            coord.requests.insert(i as u64, req_data);
        }

        // Act & Assert
        assert_eq!(coord.requests.len(), 5);
        for i in 0u64..5 {
            assert!(coord.requests.contains_key(&i));
            assert_eq!(coord.requests[&i].max_new_tokens, i as usize * 10);
        }
    }

    #[test]
    fn dispatch_coordinator_requests_update_finished_flag() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 10,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(1, req_data);

        // Act: mark as finished
        coord.requests.get_mut(&1).unwrap().finished = true;

        // Assert
        assert!(coord.requests[&1].finished);
    }

    #[test]
    fn dispatch_coordinator_requests_phase_transition() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1, 2, 3, 4],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 20,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(100, req_data);

        // Act: transition to decode
        let req = coord.requests.get_mut(&100).unwrap();
        req.phase = RequestPhase::Decode;

        // Assert
        assert_eq!(coord.requests[&100].phase, RequestPhase::Decode);
        assert_eq!(coord.requests[&100].phase, RequestPhase::Decode);
    }

    // ── Field access patterns ─────────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_chunked_prefill_config_accessible() {
        // Arrange
        let coord = make_coordinator();

        // Act
        let config = coord.chunked_prefill_scheduler.config();

        // Assert: verify default ChunkedPrefillConfig values
        assert_eq!(config.chunk_size, 512);
        assert!(config.enabled);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_accessible() {
        // Arrange
        let coord = make_coordinator();

        // Act & Assert: memory_manager was created with 32 L1 blocks
        // (We cannot query capacity directly, but we know the object exists
        // and is a valid GlobalMemoryManager from construction.)
        let _ = &coord.memory_manager;
    }

    #[test]
    fn dispatch_coordinator_policy_decide_with_default_state() {
        // Arrange
        let coord = make_coordinator();
        let state = crate::scheduler::jit_types::SystemState::default();

        // Act
        let decision = coord.policy.decide(&state);

        // Assert: with zero pressure and zero fragmentation, safe mode
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);
        assert!(decision.max_batch_size > 0);
    }

    #[test]
    fn dispatch_coordinator_policy_decide_emergency_mode() {
        // Arrange
        let coord = make_coordinator();
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.95,
            kv_fragmentation: 0.0,
            current_running_len: 4,
            current_batch_size: 4,
            ..Default::default()
        };

        // Act
        let decision = coord.policy.decide(&state);

        // Assert: emergency mode triggers
        assert!(!decision.admit_new_prefill);
        assert!(decision.force_swap_out_count > 0);
    }

    #[test]
    fn dispatch_coordinator_batcher_chunked_state_present() {
        // Arrange
        let coord = make_coordinator();

        // Act & Assert: batcher was constructed with chunked config
        assert!(coord.batcher.chunked_state.is_some());
    }

    #[test]
    fn dispatch_coordinator_request_data_fused_prefill_hidden_none() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 50,
            finished: false,
            session_id: None,
            thinking_budget: Some(0),
            fused_prefill_hidden: None,
        };
        coord.requests.insert(1, req_data);

        // Act & Assert: fused_prefill_hidden starts as None
        assert!(coord.requests[&1].fused_prefill_hidden.is_none());
        // thinking_budget is Some(0) = disabled
        assert_eq!(coord.requests[&1].thinking_budget, Some(0));
    }

    #[test]
    fn dispatch_coordinator_requests_session_id_tracking() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1, 2],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 30,
            finished: false,
            session_id: Some(999),
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(55, req_data);

        // Act & Assert
        assert_eq!(coord.requests[&55].session_id, Some(999));
    }

    // ── PolicyVariant exhaustive matching ─────────────────────────────────

    #[test]
    fn dispatch_coordinator_policy_clone_preserves_variant() {
        // Arrange
        let coord = make_coordinator();

        // Act: clone the policy (PolicyVariant derives Clone)
        let cloned_policy = coord.policy.clone();

        // Assert: cloned variant matches original
        assert!(matches!(cloned_policy, PolicyVariant::Absolute));
    }

    #[test]
    fn dispatch_coordinator_policy_decide_at_boundary_pressure() {
        // Arrange
        let coord = make_coordinator();
        // At exactly 0.9, the condition is `> 0.9` (strict), so safe mode applies
        let state_at_boundary = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.9,
            kv_fragmentation: 0.0,
            current_running_len: 4,
            current_batch_size: 4,
            ..Default::default()
        };

        // Act
        let decision = coord.policy.decide(&state_at_boundary);

        // Assert: at exactly 0.9, condition is `> 0.9` so still safe mode
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);

        // Now just above the threshold (0.91) triggers emergency
        let state_above = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.91,
            kv_fragmentation: 0.0,
            current_running_len: 4,
            current_batch_size: 4,
            ..Default::default()
        };
        let decision_above = coord.policy.decide(&state_above);
        assert!(!decision_above.admit_new_prefill);
        assert!(decision_above.force_swap_out_count > 0);
    }

    #[test]
    fn dispatch_coordinator_policy_decide_safe_mode_low_pressure() {
        // Arrange
        let coord = make_coordinator();
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.1,
            kv_fragmentation: 0.05,
            current_running_len: 2,
            current_batch_size: 2,
            ..Default::default()
        };

        // Act
        let decision = coord.policy.decide(&state);

        // Assert: safe mode — prefill admitted, no forced swaps
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);
        assert!(decision.max_batch_size >= 2);
    }

    #[test]
    fn dispatch_coordinator_policy_decide_defrag_mode() {
        // Arrange
        let coord = make_coordinator();
        // Medium pressure + high fragmentation triggers defrag
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.5,
            kv_fragmentation: 0.6,
            current_running_len: 3,
            current_batch_size: 3,
            ..Default::default()
        };

        // Act
        let decision = coord.policy.decide(&state);

        // Assert: defrag mode — no new prefill, forced swap of 1
        assert!(!decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 1);
    }

    // ── RequestPhase exhaustive coverage ──────────────────────────────────

    #[test]
    fn dispatch_coordinator_request_phase_chunked_prefill() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::ChunkedPrefill,
            max_new_tokens: 40,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(200, req_data);

        // Act & Assert: ChunkedPrefill phase is stored and readable
        assert_eq!(coord.requests[&200].phase, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn dispatch_coordinator_request_phase_all_variants_round_trip() {
        // Arrange
        let mut coord = make_coordinator();
        let phases = [
            RequestPhase::Prefill,
            RequestPhase::Decode,
            RequestPhase::ChunkedPrefill,
        ];

        // Act: insert one request per phase
        for (idx, &phase) in phases.iter().enumerate() {
            let req_id = (idx + 1) as u64;
            let req_data = RequestData {
                prompt_tokens: vec![idx as u32],
                output_tokens: vec![],
                sampling_config: Default::default(),
                phase,
                max_new_tokens: 10,
                finished: false,
                session_id: None,
                thinking_budget: None,
                fused_prefill_hidden: None,
            };
            coord.requests.insert(req_id, req_data);
        }

        // Assert: all three phases stored and distinct
        assert_eq!(coord.requests.len(), 3);
        assert_eq!(coord.requests[&1].phase, RequestPhase::Prefill);
        assert_eq!(coord.requests[&2].phase, RequestPhase::Decode);
        assert_eq!(coord.requests[&3].phase, RequestPhase::ChunkedPrefill);
    }

    // ── SamplingConfig field verification ─────────────────────────────────

    #[test]
    fn dispatch_coordinator_request_sampling_config_custom_values() {
        // Arrange
        let mut coord = make_coordinator();
        let custom_sampling = crate::engine::executor_types::SamplingConfig {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
        };
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: custom_sampling,
            phase: RequestPhase::Prefill,
            max_new_tokens: 20,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(300, req_data);

        // Act & Assert: custom sampling config preserved
        let sc = &coord.requests[&300].sampling_config;
        assert_eq!(sc.temperature, 0.7);
        assert_eq!(sc.top_k, 50);
        assert_eq!(sc.top_p, 0.9);
    }

    #[test]
    fn dispatch_coordinator_request_sampling_config_default_values() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 20,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(301, req_data);

        // Act & Assert: default sampling config values
        let sc = &coord.requests[&301].sampling_config;
        assert_eq!(sc.temperature, 1.0);
        assert_eq!(sc.top_k, 0);
        assert_eq!(sc.top_p, 1.0);
    }

    // ── Output tokens accumulation ────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_request_output_tokens_accumulation() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1, 2],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Decode,
            max_new_tokens: 10,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(400, req_data);

        // Act: simulate decode step producing tokens
        let req = coord.requests.get_mut(&400).unwrap();
        req.output_tokens.push(100);
        req.output_tokens.push(200);
        req.output_tokens.push(300);

        // Assert
        assert_eq!(coord.requests[&400].output_tokens, vec![100, 200, 300]);
        assert_eq!(coord.requests[&400].output_tokens.len(), 3);
    }

    #[test]
    fn dispatch_coordinator_request_output_tokens_reaches_max_then_finishes() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![10, 20],
            sampling_config: Default::default(),
            phase: RequestPhase::Decode,
            max_new_tokens: 3,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(401, req_data);

        // Act: add one more token, reaching max_new_tokens
        let req = coord.requests.get_mut(&401).unwrap();
        req.output_tokens.push(30);
        if req.output_tokens.len() >= req.max_new_tokens {
            req.finished = true;
        }

        // Assert: finished flag set when output reaches max
        assert_eq!(coord.requests[&401].output_tokens.len(), 3);
        assert!(coord.requests[&401].finished);
    }

    // ── Fused prefill hidden: Some(vec) ───────────────────────────────────

    #[test]
    fn dispatch_coordinator_request_fused_prefill_hidden_with_data() {
        // Arrange
        let mut coord = make_coordinator();
        let hidden_data: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let req_data = RequestData {
            prompt_tokens: vec![1, 2],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 10,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: Some(hidden_data.clone()),
        };
        coord.requests.insert(500, req_data);

        // Act & Assert: fused_prefill_hidden is Some with correct data
        let hidden = coord.requests[&500].fused_prefill_hidden.as_ref().unwrap();
        assert_eq!(hidden.len(), 4);
        assert_eq!(hidden[0], 0.1);
        assert_eq!(hidden[3], 0.4);
    }

    #[test]
    fn dispatch_coordinator_request_fused_prefill_hidden_consumed_after_prefill() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 10,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: Some(vec![0.5, 0.6]),
        };
        coord.requests.insert(501, req_data);

        // Act: after prefill, fused_prefill_hidden is consumed (set to None)
        let req = coord.requests.get_mut(&501).unwrap();
        req.fused_prefill_hidden = None;
        req.phase = RequestPhase::Decode;

        // Assert: hidden data cleared, phase transitioned
        assert!(coord.requests[&501].fused_prefill_hidden.is_none());
        assert_eq!(coord.requests[&501].phase, RequestPhase::Decode);
    }

    // ── PagedScheduler integration ────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_scheduler_page_size_matches_construction() {
        // Arrange
        let coord = make_coordinator();

        // Act: read the page_size from the inner scheduler
        let page_size = coord.scheduler.page_size();

        // Assert: constructed with block_size=4
        assert_eq!(page_size, 4);
    }

    // ── Thinking budget: Some(n) active budget ────────────────────────────

    #[test]
    fn dispatch_coordinator_request_thinking_budget_active() {
        // Arrange
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 100,
            finished: false,
            session_id: None,
            thinking_budget: Some(256),
            fused_prefill_hidden: None,
        };
        coord.requests.insert(600, req_data);

        // Act & Assert: thinking_budget is Some(256) = active budget
        assert_eq!(coord.requests[&600].thinking_budget, Some(256));
    }

    // ── Batch removal of finished requests ────────────────────────────────

    #[test]
    fn dispatch_coordinator_batch_remove_finished_requests() {
        // Arrange: insert 4 requests, mark 2 as finished
        let mut coord = make_coordinator();
        for i in 0..4u64 {
            let req_data = RequestData {
                prompt_tokens: vec![i as u32],
                output_tokens: vec![],
                sampling_config: Default::default(),
                phase: RequestPhase::Prefill,
                max_new_tokens: 10,
                finished: i % 2 == 0, // requests 0, 2 are finished
                session_id: None,
                thinking_budget: None,
                fused_prefill_hidden: None,
            };
            coord.requests.insert(i, req_data);
        }

        // Act: remove all finished requests
        let finished_ids: Vec<RequestId> = coord
            .requests
            .iter()
            .filter(|(_, r)| r.finished)
            .map(|(id, _)| *id)
            .collect();
        for id in &finished_ids {
            coord.requests.remove(id);
        }

        // Assert: only unfinished requests remain
        assert_eq!(coord.requests.len(), 2);
        assert!(!coord.requests[&1].finished);
        assert!(!coord.requests[&3].finished);
        assert!(!coord.requests.contains_key(&0));
        assert!(!coord.requests.contains_key(&2));
    }

    // ── New tests: 15 additional tests ────────────────────────────────────

    #[test]
    fn dispatch_coordinator_requests_overwrite_existing_key() {
        // Arrange: insert a request, then overwrite with different data
        let mut coord = make_coordinator();
        let req_v1 = RequestData {
            prompt_tokens: vec![1, 2, 3],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 50,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(99, req_v1);

        let req_v2 = RequestData {
            prompt_tokens: vec![10, 20, 30, 40],
            output_tokens: vec![100],
            sampling_config: Default::default(),
            phase: RequestPhase::Decode,
            max_new_tokens: 200,
            finished: true,
            session_id: Some(7),
            thinking_budget: Some(64),
            fused_prefill_hidden: None,
        };

        // Act: overwrite key 99
        coord.requests.insert(99, req_v2);

        // Assert: the new data replaced the old
        assert_eq!(coord.requests.len(), 1);
        let r = &coord.requests[&99];
        assert_eq!(r.prompt_tokens, vec![10, 20, 30, 40]);
        assert_eq!(r.output_tokens, vec![100]);
        assert_eq!(r.phase, RequestPhase::Decode);
        assert_eq!(r.phase, RequestPhase::Decode);
        assert_eq!(r.max_new_tokens, 200);
        assert!(r.finished);
        assert_eq!(r.session_id, Some(7));
        assert_eq!(r.thinking_budget, Some(64));
    }

    #[test]
    fn dispatch_coordinator_requests_remove_nonexistent_returns_none() {
        // Arrange
        let mut coord = make_coordinator();

        // Act: remove a key that was never inserted
        let result = coord.requests.remove(&12345u64);

        // Assert: returns None, map still empty
        assert!(result.is_none());
        assert!(coord.requests.is_empty());
    }

    #[test]
    fn dispatch_coordinator_memory_manager_l1_allocate_and_tier_usage() {
        // Arrange: coordinator was built with 32 L1 blocks
        let mut coord = make_coordinator();

        // Act: allocate one page from L1
        let usage_before = coord.memory_manager.tier_usage(
            crate::scheduler::memory_manager::Tier::L1,
        );
        assert_eq!(usage_before.capacity, 32);
        assert_eq!(usage_before.used, 0);

        let alloc_result = coord.memory_manager.allocate_page(
            crate::scheduler::memory_manager::Tier::L1,
        );

        // Assert: allocation succeeds and usage reflects it
        assert!(alloc_result.is_ok());
        let usage_after = coord.memory_manager.tier_usage(
            crate::scheduler::memory_manager::Tier::L1,
        );
        assert_eq!(usage_after.used, 1);
        assert_eq!(usage_after.available(), 31);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_l2_l3_zero_capacity() {
        // Arrange: coordinator was built with l2=0, l3=0
        let coord = make_coordinator();

        // Act & Assert: L2 and L3 have zero capacity
        let l2 = coord.memory_manager.tier_usage(
            crate::scheduler::memory_manager::Tier::L2,
        );
        assert_eq!(l2.capacity, 0);
        assert_eq!(l2.used, 0);
        assert_eq!(l2.available(), 0);

        let l3 = coord.memory_manager.tier_usage(
            crate::scheduler::memory_manager::Tier::L3,
        );
        assert_eq!(l3.capacity, 0);
        assert_eq!(l3.used, 0);
        assert_eq!(l3.available(), 0);
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_should_chunk_boundary() {
        // Arrange: default config has chunk_size=512
        let coord = make_coordinator();

        // Act & Assert: at exactly chunk_size, should_chunk returns false (strict >)
        assert!(!coord.chunked_prefill_scheduler.should_chunk(512));
        assert!(!coord.chunked_prefill_scheduler.should_chunk(511));
        // Just above the boundary returns true
        assert!(coord.chunked_prefill_scheduler.should_chunk(513));
        // Much larger also true
        assert!(coord.chunked_prefill_scheduler.should_chunk(4096));
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_should_chunk_zero_seq_len() {
        // Arrange
        let coord = make_coordinator();

        // Act & Assert: zero-length sequence does not need chunking
        assert!(!coord.chunked_prefill_scheduler.should_chunk(0));
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_config_defaults_verified() {
        // Arrange
        let coord = make_coordinator();
        let config = coord.chunked_prefill_scheduler.config();

        // Assert: all default ChunkedPrefillConfig values
        assert_eq!(config.chunk_size, 512);
        assert!(config.enabled);
        assert_eq!(config.max_chunks_per_request, 0);
        assert!((config.decode_ratio_cap - 0.6).abs() < f32::EPSILON);
        assert!((config.compact_waste_threshold - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.compact_min_active, 4);
        assert_eq!(config.max_batch_tokens, 4096);
    }

    #[test]
    fn dispatch_coordinator_policy_decide_high_entropy_reduces_batch() {
        // Arrange: low pressure but very high logits entropy
        let coord = make_coordinator();
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.1,
            kv_fragmentation: 0.0,
            current_running_len: 4,
            current_batch_size: 4,
            logits_entropy: 12.0, // > 8.0 threshold
            ..Default::default()
        };

        // Act
        let decision = coord.policy.decide(&state);

        // Assert: safe mode still (low pressure), but entropy caps batch
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);
        // High entropy caps to current_running_len.max(1).min(32) = 4
        assert_eq!(decision.max_batch_size, 4);
    }

    #[test]
    fn dispatch_coordinator_policy_decide_high_sparsity_reduces_batch() {
        // Arrange: low pressure, normal entropy, but high attention sparsity
        let coord = make_coordinator();
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.1,
            kv_fragmentation: 0.0,
            current_running_len: 4,
            current_batch_size: 4,
            logits_entropy: 0.0,
            attention_sparsity: 0.8, // > 0.7 threshold
            ..Default::default()
        };

        // Act
        let decision = coord.policy.decide(&state);

        // Assert: safe mode, sparsity cap reduces batch
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);
        // sparsity cap: 32 * (1.0 - 0.8*0.5) = 32 * 0.6 = 19
        let expected = (32.0_f32 * (1.0 - 0.8 * 0.5)) as usize;
        assert_eq!(decision.max_batch_size, expected.max(4));
    }

    #[test]
    fn dispatch_coordinator_policy_decide_zero_pressure_zero_fragmentation() {
        // Arrange: absolutely minimal system state
        let coord = make_coordinator();
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.0,
            kv_fragmentation: 0.0,
            current_running_len: 0,
            current_batch_size: 0,
            ..Default::default()
        };

        // Act
        let decision = coord.policy.decide(&state);

        // Assert: safe mode with full batch capacity
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);
        assert_eq!(decision.max_batch_size, 32);
    }

    #[test]
    fn dispatch_coordinator_policy_decide_emergency_swap_out_scales_with_pressure() {
        // Arrange: higher pressure produces more forced swaps
        let coord = make_coordinator();

        let state_low = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.91,
            kv_fragmentation: 0.0,
            current_running_len: 4,
            current_batch_size: 4,
            ..Default::default()
        };
        let state_high = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.99,
            kv_fragmentation: 0.0,
            current_running_len: 4,
            current_batch_size: 4,
            ..Default::default()
        };

        // Act
        let decision_low = coord.policy.decide(&state_low);
        let decision_high = coord.policy.decide(&state_high);

        // Assert: both are emergency mode, higher pressure has more swaps
        assert!(!decision_low.admit_new_prefill);
        assert!(!decision_high.admit_new_prefill);
        assert!(decision_high.force_swap_out_count >= decision_low.force_swap_out_count);
        assert!(decision_high.force_swap_out_count > 0);
    }

    #[test]
    fn dispatch_coordinator_sampling_config_zero_temperature() {
        // Arrange: temperature=0 means greedy decoding
        let mut coord = make_coordinator();
        let sampling = crate::engine::executor_types::SamplingConfig {
            temperature: 0.0,
            top_k: 1,
            top_p: 0.0,
        };
        let req_data = RequestData {
            prompt_tokens: vec![5, 10, 15],
            output_tokens: vec![],
            sampling_config: sampling,
            phase: RequestPhase::Prefill,
            max_new_tokens: 50,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(700, req_data);

        // Act & Assert: greedy sampling config preserved exactly
        let sc = &coord.requests[&700].sampling_config;
        assert_eq!(sc.temperature, 0.0);
        assert_eq!(sc.top_k, 1);
        assert_eq!(sc.top_p, 0.0);
    }

    #[test]
    fn dispatch_coordinator_request_prompt_tokens_empty_vec() {
        // Arrange: a request with empty prompt tokens (edge case)
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 0,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(800, req_data);

        // Act & Assert: empty prompt tokens stored correctly
        assert!(coord.requests[&800].prompt_tokens.is_empty());
        assert_eq!(coord.requests[&800].max_new_tokens, 0);
        assert!(coord.requests[&800].output_tokens.is_empty());
    }

    #[test]
    fn dispatch_coordinator_request_large_output_tokens_accumulation() {
        // Arrange: simulate a request generating many output tokens
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Decode,
            max_new_tokens: 1000,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(900, req_data);

        // Act: push 500 output tokens
        let req = coord.requests.get_mut(&900).unwrap();
        for i in 0..500u32 {
            req.output_tokens.push(i);
        }

        // Assert: all 500 tokens accumulated correctly
        assert_eq!(coord.requests[&900].output_tokens.len(), 500);
        assert_eq!(coord.requests[&900].output_tokens[0], 0);
        assert_eq!(coord.requests[&900].output_tokens[499], 499);
        // Not yet at max_new_tokens
        assert!(!coord.requests[&900].finished);
    }

    #[test]
    fn dispatch_coordinator_batcher_default_no_chunked_state() {
        // Arrange: construct a batcher without with_chunked
        let batcher = ContinuousBatcher::new();

        // Act & Assert: no chunked state present
        assert!(batcher.chunked_state.is_none());
        assert!(!batcher.has_pending_work());
        assert_eq!(batcher.waiting_len(), 0);
        assert_eq!(batcher.running_len(), 0);
    }

    #[test]
    fn dispatch_coordinator_batcher_running_ids_empty_initially() {
        // Arrange
        let coord = make_coordinator();

        // Act
        let ids = coord.batcher.running_ids();

        // Assert: no running sequences initially
        assert!(ids.is_empty());
    }

    // ── 15 additional tests ────────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_memory_manager_plan_prefill_fully_resident() {
        // Arrange: 32 L1 blocks, prompt that fits within available pages
        let mut coord = make_coordinator();
        for _ in 0..10 {
            coord.memory_manager.allocate_page(
                crate::scheduler::memory_manager::Tier::L1,
            ).unwrap();
        }

        // Act: plan prefill for 20 tokens with page_size=4 (5 pages needed)
        let plan = coord.memory_manager.plan_prefill(20, 512, 4);

        // Assert: 5 pages fit in remaining 22 L1 slots
        assert!(matches!(
            plan,
            crate::scheduler::memory_manager::PrefillPlan::FullyResident { pages } if pages == 5
        ));
    }

    #[test]
    fn dispatch_coordinator_memory_manager_plan_prefill_pipelined() {
        // Arrange: 32 L1 blocks, all allocated, prompt exceeds capacity
        let mut coord = make_coordinator();
        for _ in 0..32 {
            coord.memory_manager.allocate_page(
                crate::scheduler::memory_manager::Tier::L1,
            ).unwrap();
        }

        // Act: plan prefill for 200 tokens with page_size=4
        let plan = coord.memory_manager.plan_prefill(200, 64, 4);

        // Assert: L1 is full so Pipelined with l1_pages=0
        assert!(matches!(
            plan,
            crate::scheduler::memory_manager::PrefillPlan::Pipelined { l1_pages, .. } if l1_pages == 0
        ));
    }

    #[test]
    fn dispatch_coordinator_memory_manager_plan_prefill_zero_tokens() {
        // Arrange
        let mut coord = make_coordinator();

        // Act: zero prompt tokens
        let plan = coord.memory_manager.plan_prefill(0, 512, 4);

        // Assert: FullyResident with 0 pages
        assert!(matches!(
            plan,
            crate::scheduler::memory_manager::PrefillPlan::FullyResident { pages } if pages == 0
        ));
    }

    #[test]
    fn dispatch_coordinator_memory_manager_session_register_and_finalize() {
        // Arrange
        let mut coord = make_coordinator();

        // Act: register a session
        let session = coord.memory_manager.register_session(42);

        // Assert: session created with correct id and zero finalized_position
        assert_eq!(session.session_id, 42);
        assert!(session.pages.is_empty());
        assert_eq!(session.finalized_position, 0);

        // Act: finalize session tokens
        coord.memory_manager.finalize_session_tokens(42, 128);
        let pos = coord.memory_manager.session_finalized_position(42);

        // Assert: finalized position updated
        assert_eq!(pos, Some(128));

        // Act: finalize with lower value must not decrease
        coord.memory_manager.finalize_session_tokens(42, 50);
        let pos_after = coord.memory_manager.session_finalized_position(42);

        // Assert: position stays at 128 (monotonic)
        assert_eq!(pos_after, Some(128));
    }

    #[test]
    fn dispatch_coordinator_memory_manager_session_unknown_returns_none() {
        // Arrange
        let coord = make_coordinator();

        // Act: query unknown session
        let pos = coord.memory_manager.session_finalized_position(999);

        // Assert: unknown session returns None
        assert!(pos.is_none());
    }

    #[test]
    fn dispatch_coordinator_memory_manager_virtual_page_bind_resolve_unmap() {
        // Arrange
        let mut coord = make_coordinator();
        let tier = crate::scheduler::memory_manager::Tier::L1;
        let phys = coord.memory_manager.allocate_page(tier).unwrap();
        let vpage = crate::scheduler::memory_manager::VirtualPageId::new(1, 0);

        // Act: bind virtual to physical
        coord.memory_manager.bind_virtual_page(vpage, tier, phys).unwrap();

        // Assert: resolve returns correct mapping
        let (resolved_tier, resolved_phys) = coord.memory_manager.resolve(vpage).unwrap();
        assert_eq!(resolved_tier, tier);
        assert_eq!(resolved_phys, phys);

        // Act: unmap virtual page
        let old_loc = coord.memory_manager.unmap_virtual_page(vpage);

        // Assert: unmap returns the old location
        assert!(old_loc.is_some());
        assert_eq!(old_loc.unwrap().physical_id, phys);

        // Assert: resolve now fails
        assert!(coord.memory_manager.resolve(vpage).is_err());
    }

    #[test]
    fn dispatch_coordinator_memory_manager_migrate_page_l1_to_l2() {
        // Arrange: create a coordinator with both L1 and L2 capacity
        let mut coord = DispatchCoordinator {
            scheduler: PagedScheduler::new(64, 4, HGALConfig::default()),
            batcher: ContinuousBatcher::new()
                .with_chunked(ChunkedConfig::default()),
            chunked_prefill_scheduler: ChunkedPrefillScheduler::new(
                ChunkedPrefillConfig::default(),
            ),
            requests: HashMap::new(),
            memory_manager: GlobalMemoryManager::new_with_capacities(32, 32, 0),
            policy: PolicyVariant::default(),
        };
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let l2 = crate::scheduler::memory_manager::Tier::L2;
        let phys = coord.memory_manager.allocate_page(l1).unwrap();

        // Act: migrate from L1 to L2
        let new_phys = coord.memory_manager.migrate_page(l1, l2, phys).unwrap();

        // Assert: L1 usage decreased, L2 usage increased, migration succeeded
        let l1_usage = coord.memory_manager.tier_usage(l1);
        let l2_usage = coord.memory_manager.tier_usage(l2);
        assert_eq!(l1_usage.used, 0);
        assert_eq!(l2_usage.used, 1);
        // PhysicalIds are per-tier counters starting at 0, so they may coincide
        let _ = new_phys;
    }

    #[test]
    fn dispatch_coordinator_memory_manager_allocate_exhausts_l1() {
        // Arrange: coordinator with 32 L1 blocks
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;

        // Act: allocate all 32 L1 pages
        for _ in 0..32 {
            assert!(coord.memory_manager.allocate_page(l1).is_ok());
        }

        // Act: 33rd allocation should fail
        let result = coord.memory_manager.allocate_page(l1);

        // Assert: capacity exceeded
        assert!(result.is_err());
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.used, 32);
        assert_eq!(usage.available(), 0);
    }

    #[test]
    fn dispatch_coordinator_batcher_enqueue_and_lengths() {
        // Arrange
        let mut coord = make_coordinator();
        let seq1 = crate::scheduler::sequence::Sequence::new(1, vec![10, 20, 30]);
        let seq2 = crate::scheduler::sequence::Sequence::new(2, vec![40, 50]);

        // Act: enqueue two sequences
        coord.batcher.enqueue(seq1);
        coord.batcher.enqueue(seq2);

        // Assert: both in waiting, none running
        assert_eq!(coord.batcher.waiting_len(), 2);
        assert_eq!(coord.batcher.running_len(), 0);
        assert!(coord.batcher.has_pending_work());
    }

    #[test]
    fn dispatch_coordinator_batcher_duplicate_enqueue_deduplication() {
        // Arrange
        let mut coord = make_coordinator();
        let seq = crate::scheduler::sequence::Sequence::new(42, vec![1, 2, 3]);

        // Act: enqueue same ID twice
        coord.batcher.enqueue(seq.clone());
        coord.batcher.enqueue(seq);

        // Assert: only one entry (deduplication)
        assert_eq!(coord.batcher.waiting_len(), 1);
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_next_chunk_size_adaptive() {
        // Arrange
        let mut coord = make_coordinator();

        // Act: low L1 ratio returns default chunk_size
        coord.chunked_prefill_scheduler.update_l1_ratio(0.1);
        let chunk = coord.chunked_prefill_scheduler.next_chunk_size(2000, 4096);

        // Assert: at low L1 ratio, chunk size is the default (512)
        assert_eq!(chunk, 512);

        // Act: high L1 ratio returns up to max_seq_len capped by remaining
        coord.chunked_prefill_scheduler.update_l1_ratio(0.9);
        let chunk_high = coord.chunked_prefill_scheduler.next_chunk_size(2000, 4096);

        // Assert: capped by remaining tokens
        assert_eq!(chunk_high, 2000);
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_update_concurrency_reduces_chunk() {
        // Arrange
        let mut coord = make_coordinator();

        // Act: high concurrency with high L1 ratio
        coord.chunked_prefill_scheduler.update_l1_ratio(0.9);
        coord.chunked_prefill_scheduler.update_concurrency(16);
        let chunk = coord.chunked_prefill_scheduler.next_chunk_size(2000, 4096);

        // Assert: chunk is bounded between base and remaining
        assert!(chunk >= 512);
        assert!(chunk <= 2000);
    }

    #[test]
    fn dispatch_coordinator_scheduler_total_pages_matches_construction() {
        // Arrange: constructed with total_blocks=32
        let coord = make_coordinator();

        // Act
        let total = coord.scheduler.total_pages();

        // Assert
        assert_eq!(total, 32);
    }

    #[test]
    fn dispatch_coordinator_tier_usage_available_saturating() {
        // Arrange: degenerate edge where used exceeds capacity
        let usage = crate::scheduler::memory_manager::TierUsage {
            used: 50,
            capacity: 30,
        };

        // Act & Assert: available saturates at 0
        assert_eq!(usage.available(), 0);

        // Normal case
        let normal = crate::scheduler::memory_manager::TierUsage {
            used: 10,
            capacity: 30,
        };
        assert_eq!(normal.available(), 20);
    }

    #[test]
    fn dispatch_coordinator_request_data_full_lifecycle() {
        // Arrange: simulate a request through all lifecycle phases
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![100, 200, 300, 400, 500],
            output_tokens: vec![],
            sampling_config: crate::engine::executor_types::SamplingConfig {
                temperature: 0.85,
                top_k: 40,
                top_p: 0.95,
            },
            phase: RequestPhase::Prefill,
            max_new_tokens: 5,
            finished: false,
            session_id: Some(77),
            thinking_budget: Some(128),
            fused_prefill_hidden: Some(vec![0.1; 2560]),
        };
        coord.requests.insert(1001, req_data);

        // Assert: Phase 1 - Prefill
        let r = &coord.requests[&1001];
        assert_eq!(r.phase, RequestPhase::Prefill);
        assert_eq!(r.phase, RequestPhase::Prefill);
        assert!(r.fused_prefill_hidden.is_some());
        assert_eq!(r.fused_prefill_hidden.as_ref().unwrap().len(), 2560);
        assert_eq!(r.thinking_budget, Some(128));
        assert_eq!(r.session_id, Some(77));
        assert_eq!(r.sampling_config.temperature, 0.85);

        // Act: Phase 2 - transition to ChunkedPrefill
        let req = coord.requests.get_mut(&1001).unwrap();
        req.phase = RequestPhase::ChunkedPrefill;
        assert_eq!(coord.requests[&1001].phase, RequestPhase::ChunkedPrefill);

        // Act: Phase 3 - consume fused hidden, transition to Decode
        let req = coord.requests.get_mut(&1001).unwrap();
        req.fused_prefill_hidden = None;
        req.phase = RequestPhase::Decode;
        assert!(coord.requests[&1001].fused_prefill_hidden.is_none());
        assert_eq!(coord.requests[&1001].phase, RequestPhase::Decode);

        // Act: Phase 4 - generate tokens and finish
        let req = coord.requests.get_mut(&1001).unwrap();
        for i in 0..5u32 {
            req.output_tokens.push(1000 + i);
        }
        if req.output_tokens.len() >= req.max_new_tokens {
            req.finished = true;
        }
        assert_eq!(
            coord.requests[&1001].output_tokens,
            vec![1000, 1001, 1002, 1003, 1004]
        );
        assert!(coord.requests[&1001].finished);

        // Act: Phase 5 - remove finished request
        let removed = coord.requests.remove(&1001);
        assert!(removed.is_some());
        assert!(coord.requests.is_empty());
    }

    // -- 15 new tests: uncovered data structures and edge cases --

    #[test]
    fn dispatch_coordinator_memory_manager_free_page_and_reallocate() {
        // Arrange: allocate all 32 L1 pages, free one, reallocate it
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let mut allocated = Vec::new();
        for _ in 0..32 {
            allocated.push(coord.memory_manager.allocate_page(l1).unwrap());
        }
        let last_id = allocated[31];

        // Act: free the last page
        coord.memory_manager.free_page(l1, last_id).unwrap();

        // Assert: usage drops to 31
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.used, 31);
        assert_eq!(usage.available(), 1);

        // Act: reallocate -- should succeed and reuse the freed slot
        let new_id = coord.memory_manager.allocate_page(l1).unwrap();
        let usage_after = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage_after.used, 32);
        assert_eq!(usage_after.available(), 0);

        // The reallocated ID should be the same physical page (freed pool reuse)
        assert_eq!(new_id, last_id);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_free_unknown_page_returns_error() {
        // Arrange
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;

        // Act: free a page that was never allocated
        let result = coord.memory_manager.free_page(l1, 9999);

        // Assert: returns error for unknown physical page
        assert!(result.is_err());
    }

    #[test]
    fn dispatch_coordinator_memory_manager_migrate_same_tier_is_noop() {
        // Arrange: allocate a page in L1
        let mut coord = DispatchCoordinator {
            scheduler: PagedScheduler::new(64, 4, HGALConfig::default()),
            batcher: ContinuousBatcher::new()
                .with_chunked(ChunkedConfig::default()),
            chunked_prefill_scheduler: ChunkedPrefillScheduler::new(
                ChunkedPrefillConfig::default(),
            ),
            requests: HashMap::new(),
            memory_manager: GlobalMemoryManager::new_with_capacities(32, 32, 0),
            policy: PolicyVariant::default(),
        };
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let phys = coord.memory_manager.allocate_page(l1).unwrap();

        // Act: migrate from L1 to L1 (same tier)
        let result = coord.memory_manager.migrate_page(l1, l1, phys).unwrap();

        // Assert: same-tier migration returns the same physical ID with no usage change
        assert_eq!(result, phys);
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.used, 1);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_migrate_unknown_source_returns_error() {
        // Arrange: coordinator with L1 and L2 capacity
        let mut coord = DispatchCoordinator {
            scheduler: PagedScheduler::new(64, 4, HGALConfig::default()),
            batcher: ContinuousBatcher::new()
                .with_chunked(ChunkedConfig::default()),
            chunked_prefill_scheduler: ChunkedPrefillScheduler::new(
                ChunkedPrefillConfig::default(),
            ),
            requests: HashMap::new(),
            memory_manager: GlobalMemoryManager::new_with_capacities(32, 32, 0),
            policy: PolicyVariant::default(),
        };
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let l2 = crate::scheduler::memory_manager::Tier::L2;

        // Act: migrate a page that was never allocated in L1
        let result = coord.memory_manager.migrate_page(l1, l2, 42);

        // Assert: returns error for unknown source physical page
        assert!(result.is_err());
    }

    #[test]
    fn dispatch_coordinator_memory_manager_remap_virtual_page() {
        // Arrange: allocate two physical pages and bind a virtual page to the first
        let mut coord = make_coordinator();
        let tier = crate::scheduler::memory_manager::Tier::L1;
        let phys_a = coord.memory_manager.allocate_page(tier).unwrap();
        let phys_b = coord.memory_manager.allocate_page(tier).unwrap();
        let vpage = crate::scheduler::memory_manager::VirtualPageId::new(10, 0);

        coord.memory_manager.bind_virtual_page(vpage, tier, phys_a).unwrap();
        let (resolved_tier, resolved_phys) = coord.memory_manager.resolve(vpage).unwrap();
        assert_eq!(resolved_tier, tier);
        assert_eq!(resolved_phys, phys_a);

        // Act: remap the virtual page to the second physical page
        coord.memory_manager.remap_virtual_page(vpage, tier, phys_b).unwrap();

        // Assert: resolve now returns the new physical page
        let (new_tier, new_phys) = coord.memory_manager.resolve(vpage).unwrap();
        assert_eq!(new_tier, tier);
        assert_eq!(new_phys, phys_b);
    }

    #[test]
    fn dispatch_coordinator_sequence_construction_and_fields() {
        // Arrange: construct sequences via Sequence::new
        let seq = crate::scheduler::sequence::Sequence::new(42, vec![10, 20, 30, 40]);

        // Act & Assert: verify initial field values
        assert_eq!(seq.id, 42);
        assert_eq!(seq.prompt_tokens, vec![10, 20, 30, 40]);
        assert!(seq.generated_tokens.is_empty());
        assert_eq!(seq.state, crate::scheduler::sequence::SequenceState::Waiting);
        assert_eq!(seq.enqueue_order, 0);
        assert_eq!(seq.context_len(), 4);
        assert!(seq.needs_prefill());
        assert!(seq.kv_pages.is_empty());
        assert_eq!(seq.draft_budget, 0);
    }

    #[test]
    fn dispatch_coordinator_sequence_state_all_variants() {
        // Arrange: verify all SequenceState variants exist and are distinct
        use crate::scheduler::sequence::SequenceState;
        let states = [
            SequenceState::Waiting,
            SequenceState::Running,
            SequenceState::Paused,
            SequenceState::Completed,
            SequenceState::Failed,
        ];

        // Act & Assert: all five variants are pairwise distinct
        for i in 0..states.len() {
            for j in 0..states.len() {
                if i == j {
                    assert_eq!(states[i], states[j]);
                } else {
                    assert_ne!(states[i], states[j]);
                }
            }
        }
    }

    #[test]
    fn dispatch_coordinator_batch_manifest_should_compact_logic() {
        // Arrange: BatchManifest with high waste ratio and enough active slots
        let config = ChunkedPrefillConfig::default();
        let manifest = crate::scheduler::chunked_prefill::BatchManifest {
            slots: vec![
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 1,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 0,
                    token_end: 1,
                    compact_target: 0,
                },
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 2,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 1,
                    token_end: 2,
                    compact_target: 1,
                },
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 3,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 2,
                    token_end: 3,
                    compact_target: 2,
                },
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 4,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 3,
                    token_end: 4,
                    compact_target: 3,
                },
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 5,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 4,
                    token_end: 5,
                    compact_target: -1,
                },
            ],
            total_tokens: 5,
            decode_tokens: 5,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.5,
        };

        // Act: should_compact returns true when waste > threshold and active >= min
        assert!(manifest.should_compact(&config));

        // Assert: low waste ratio manifest should not trigger compact
        let low_waste = crate::scheduler::chunked_prefill::BatchManifest {
            waste_ratio: 0.1,
            ..manifest.clone()
        };
        assert!(!low_waste.should_compact(&config));
    }

    #[test]
    fn dispatch_coordinator_batch_slot_type_variants_distinct() {
        // Arrange
        use crate::scheduler::chunked_prefill::SlotType;
        let decode = SlotType::Decode;
        let prefill = SlotType::PrefillChunk;

        // Act & Assert: the two variants are distinct
        assert_ne!(decode, prefill);
        assert_eq!(decode, SlotType::Decode);
        assert_eq!(prefill, SlotType::PrefillChunk);
    }

    #[test]
    fn dispatch_coordinator_batch_prep_data_construction_and_sampling() {
        // Arrange: construct BatchPrepData for 3 sequences
        let mut prep = crate::scheduler::batcher::BatchPrepData::new(3);

        // Act: set sampling params for sequence 1
        prep.set_sampling_params(1, 0.5f32, 100, 0.9f32, 2);

        // Assert: packed layout verified
        assert_eq!(prep.sampling_params_packed.len(), 12); // 3 * 4
        let base = 1 * 4;
        assert_eq!(prep.sampling_params_packed[base], 0.5f32.to_bits());
        assert_eq!(prep.sampling_params_packed[base + 1], 100);
        assert_eq!(prep.sampling_params_packed[base + 2], 0.9f32.to_bits());
        assert_eq!(prep.sampling_params_packed[base + 3], 2);

        // Assert: unset sequence 0 remains zeroed
        assert_eq!(prep.sampling_params_packed[0], 0);
        assert_eq!(prep.sampling_params_packed[3], 0);

        // Assert: default fields
        assert_eq!(prep.max_decode_steps, 0);
        assert_eq!(prep.total_prefill_tokens, 0);
        assert_eq!(prep.active_flags, vec![1, 1, 1]); // default all active
        assert_eq!(prep.prompt_lens, vec![0, 0, 0]);
    }

    #[test]
    fn dispatch_coordinator_batch_result_action_variants() {
        // Arrange: verify all BatchAction variants
        use crate::scheduler::batcher::{BatchAction, BatchResult};
        let telemetry = crate::scheduler::telemetry::SequenceTelemetry::new();

        // Act: construct each variant
        let cont = BatchResult::continue_with_token(1, 42, telemetry.clone());
        let comp = BatchResult::complete(2, Some(99), telemetry.clone());
        let pause = BatchResult::pause(3);
        let fail = BatchResult::fail(4);

        // Assert: each variant has correct action type
        assert_eq!(cont.action, BatchAction::Continue);
        assert_eq!(cont.generated_token, Some(42));
        assert_eq!(comp.action, BatchAction::Complete);
        assert_eq!(comp.generated_token, Some(99));
        assert_eq!(pause.action, BatchAction::Pause);
        assert!(pause.generated_token.is_none());
        assert_eq!(fail.action, BatchAction::Fail);
        assert!(fail.generated_token.is_none());
    }

    #[test]
    fn dispatch_coordinator_scheduler_decision_default_values() {
        // Arrange & Act
        let decision = crate::scheduler::jit_types::SchedulerDecision::default();

        // Assert: default decision is maximally conservative
        assert_eq!(decision.max_batch_size, 1);
        assert!(!decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);
    }

    #[test]
    fn dispatch_coordinator_policy_custom_config_reduces_batch() {
        // Arrange: custom PolicyConfig with smaller batch sizes
        use crate::scheduler::policy::{AbsolutePolicy, PolicyConfig, SchedulingPolicy};
        let config = PolicyConfig {
            pressure_emergency: 0.9,
            pressure_aggressive_ceiling: 1.0,
            frag_defrag_threshold: 0.5,
            queue_aggressive_trigger: usize::MAX,
            batch_safe: 8,
            batch_normal: 8,
            batch_aggressive: 8,
        };
        let policy = AbsolutePolicy::with_config(config);
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.1,
            kv_fragmentation: 0.0,
            current_running_len: 3,
            current_batch_size: 3,
            ..Default::default()
        };

        // Act
        let decision = policy.decide(&state);

        // Assert: safe mode with custom batch cap of 8 (not default 32)
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);
        assert_eq!(decision.max_batch_size, 8);
    }

    #[test]
    fn dispatch_coordinator_adaptive_chunk_policy_boundary_ratios() {
        // Arrange: test the three regions of AdaptiveChunkPolicy
        use crate::scheduler::chunked_prefill::AdaptiveChunkPolicy;
        let mut policy = AdaptiveChunkPolicy::new(512);
        let remaining = 4096;
        let max_seq = 8192;

        // Act & Assert: below 0.25 -> base chunk size
        policy.l1_available_ratio = 0.1;
        policy.concurrent_requests = 1;
        let low = policy.compute_chunk_size(remaining, max_seq);
        assert_eq!(low, 512);

        // Act & Assert: exactly 0.25 -> still base (boundary is < 0.25)
        policy.l1_available_ratio = 0.25;
        let at_boundary = policy.compute_chunk_size(remaining, max_seq);
        assert_eq!(at_boundary, 512);

        // Act & Assert: above 0.75 -> max_seq_len (capped by remaining)
        policy.l1_available_ratio = 0.8;
        let high = policy.compute_chunk_size(remaining, max_seq);
        assert_eq!(high, remaining); // capped by remaining (4096 < max_seq 8192)
    }

    #[test]
    fn dispatch_coordinator_eviction_policy_select_victims_empty_metadata() {
        // Arrange: empty metadata map
        let coord = make_coordinator();
        let metadata = std::collections::HashMap::new();
        let semantic = std::collections::HashMap::new();

        // Act: request 3 victims from empty metadata
        let victims = coord.memory_manager.select_victims(&metadata, &semantic, 3);

        // Assert: no victims selected from empty metadata
        assert!(victims.is_empty());
    }

    #[test]
    fn dispatch_coordinator_batcher_mean_context_len_empty() {
        // Arrange: empty batcher
        let coord = make_coordinator();

        // Act: mean context length with no sequences
        let mean = coord.batcher.mean_context_len();

        // Assert: returns 0 for empty batcher
        assert_eq!(mean, 0);
    }

    // ── 15 additional tests (wave-12x34) ──────────────────────────────────

    #[test]
    fn dispatch_coordinator_send_sync_bounds() {
        // Arrange & Act: static assertions that DispatchCoordinator is Send + Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<DispatchCoordinator>();
        assert_sync::<DispatchCoordinator>();

        // Assert: compiles successfully — DispatchCoordinator is Send + Sync
        // (no runtime assertion needed; compile-time check is the test)
    }

    #[test]
    fn dispatch_coordinator_scheduler_decision_debug_trait() {
        // Arrange: construct a default SchedulerDecision
        let decision = crate::scheduler::jit_types::SchedulerDecision::default();

        // Act: format via Debug trait
        let debug_str = format!("{:?}", decision);

        // Assert: Debug output contains field names
        assert!(
            debug_str.contains("max_batch_size"),
            "SchedulerDecision Debug should contain 'max_batch_size', got: {}", debug_str
        );
        assert!(
            debug_str.contains("admit_new_prefill"),
            "SchedulerDecision Debug should contain 'admit_new_prefill', got: {}", debug_str
        );
    }

    #[test]
    fn dispatch_coordinator_clone_trait_policy_independent() {
        // Arrange
        let coord = make_coordinator();
        let cloned_policy = coord.policy.clone();

        // Act: modify the clone's decision by calling decide (immutable, but proves independence)
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.5,
            ..Default::default()
        };
        let decision = cloned_policy.decide(&state);

        // Assert: original policy is unaffected
        let original_decision = coord.policy.decide(&state);
        assert_eq!(decision.max_batch_size, original_decision.max_batch_size);
        assert_eq!(decision.admit_new_prefill, original_decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, original_decision.force_swap_out_count);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_tier_l3_zero_capacity_no_allocation() {
        // Arrange: coordinator with L3 capacity = 0
        let mut coord = make_coordinator();
        let l3 = crate::scheduler::memory_manager::Tier::L3;

        // Act: attempt to allocate from L3
        let result = coord.memory_manager.allocate_page(l3);

        // Assert: allocation fails on zero-capacity tier
        assert!(result.is_err());
        let usage = coord.memory_manager.tier_usage(l3);
        assert_eq!(usage.capacity, 0);
        assert_eq!(usage.used, 0);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_session_pages_tracked() {
        // Arrange
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;

        // Act: register session and allocate pages bound to it
        let session = coord.memory_manager.register_session(55);
        assert!(session.pages.is_empty());

        let phys_a = coord.memory_manager.allocate_page(l1).unwrap();
        let phys_b = coord.memory_manager.allocate_page(l1).unwrap();
        let vpage_a = crate::scheduler::memory_manager::VirtualPageId::new(55, 0);
        let vpage_b = crate::scheduler::memory_manager::VirtualPageId::new(55, 1);
        coord.memory_manager.bind_virtual_page(vpage_a, l1, phys_a).unwrap();
        coord.memory_manager.bind_virtual_page(vpage_b, l1, phys_b).unwrap();

        // Assert: both virtual pages resolve correctly
        let (t1, p1) = coord.memory_manager.resolve(vpage_a).unwrap();
        assert_eq!(t1, l1);
        assert_eq!(p1, phys_a);
        let (t2, p2) = coord.memory_manager.resolve(vpage_b).unwrap();
        assert_eq!(t2, l1);
        assert_eq!(p2, phys_b);
    }

    #[test]
    fn dispatch_coordinator_sequence_state_transitions() {
        // Arrange
        use crate::scheduler::sequence::{Sequence, SequenceState};
        let mut seq = Sequence::new(10, vec![1, 2, 3]);
        assert_eq!(seq.state, SequenceState::Waiting);

        // Act & Assert: transition Waiting -> Running (still needs_prefill until tokens generated)
        seq.state = SequenceState::Running;
        assert_eq!(seq.state, SequenceState::Running);
        assert!(seq.needs_prefill()); // no generated tokens yet, still needs prefill

        // Act: generate a token, now needs_prefill returns false
        seq.generated_tokens.push(42);
        seq.position += 1;
        assert!(!seq.needs_prefill());

        // Act & Assert: transition Running -> Paused
        seq.state = SequenceState::Paused;
        assert_eq!(seq.state, SequenceState::Paused);

        // Act & Assert: transition Paused -> Completed
        seq.state = SequenceState::Completed;
        assert_eq!(seq.state, SequenceState::Completed);

        // Act & Assert: transition to Failed
        seq.state = SequenceState::Failed;
        assert_eq!(seq.state, SequenceState::Failed);
    }

    #[test]
    fn dispatch_coordinator_sequence_generated_tokens_and_context_len() {
        // Arrange
        use crate::scheduler::sequence::Sequence;
        let mut seq = Sequence::new(10, vec![1, 2, 3]);
        assert_eq!(seq.context_len(), 3); // prompt only

        // Act: add generated tokens (must also update position for context_len)
        seq.generated_tokens.push(100);
        seq.position += 1;
        seq.generated_tokens.push(200);
        seq.position += 1;
        seq.generated_tokens.push(300);
        seq.position += 1;

        // Assert: context_len = prompt + generated (via position)
        assert_eq!(seq.context_len(), 6);
        assert_eq!(seq.generated_tokens.len(), 3);
        assert_eq!(seq.generated_tokens, vec![100, 200, 300]);
    }

    #[test]
    fn dispatch_coordinator_sequence_draft_budget_default_zero() {
        // Arrange
        use crate::scheduler::sequence::Sequence;
        let mut seq = Sequence::new(99, vec![5, 6, 7]);

        // Assert: default draft_budget is 0
        assert_eq!(seq.draft_budget, 0);

        // Act: set draft budget (speculative decoding)
        seq.draft_budget = 5;

        // Assert
        assert_eq!(seq.draft_budget, 5);
    }

    #[test]
    fn dispatch_coordinator_batch_manifest_clone_independence() {
        // Arrange
        let manifest = crate::scheduler::chunked_prefill::BatchManifest {
            slots: vec![
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 1,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 0,
                    token_end: 4,
                    compact_target: 0,
                },
            ],
            total_tokens: 4,
            decode_tokens: 4,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };

        // Act: clone the manifest
        let cloned = manifest.clone();

        // Assert: cloned fields match original
        assert_eq!(cloned.total_tokens, manifest.total_tokens);
        assert_eq!(cloned.decode_tokens, manifest.decode_tokens);
        assert_eq!(cloned.prefill_tokens, manifest.prefill_tokens);
        assert_eq!(cloned.waste_ratio, manifest.waste_ratio);
        assert_eq!(cloned.slots.len(), manifest.slots.len());
        assert_eq!(cloned.slots[0].request_id, 1);
    }

    #[test]
    fn dispatch_coordinator_batch_result_request_id_extraction() {
        // Arrange
        use crate::scheduler::batcher::BatchResult;
        let telemetry = crate::scheduler::telemetry::SequenceTelemetry::new();

        // Act: create results with different request IDs
        let cont = BatchResult::continue_with_token(111, 42, telemetry.clone());
        let comp = BatchResult::complete(222, Some(99), telemetry.clone());
        let pause = BatchResult::pause(333);
        let fail = BatchResult::fail(444);

        // Assert: each variant preserves its request_id
        assert_eq!(cont.request_id, 111);
        assert_eq!(comp.request_id, 222);
        assert_eq!(pause.request_id, 333);
        assert_eq!(fail.request_id, 444);
    }

    #[test]
    fn dispatch_coordinator_batch_prep_data_active_flags_override() {
        // Arrange
        let mut prep = crate::scheduler::batcher::BatchPrepData::new(4);

        // Assert: default all active
        assert_eq!(prep.active_flags, vec![1, 1, 1, 1]);

        // Act: deactivate sequences 1 and 3
        prep.active_flags[1] = 0;
        prep.active_flags[3] = 0;

        // Assert: flags reflect deactivation
        assert_eq!(prep.active_flags, vec![1, 0, 1, 0]);
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_should_chunk_disabled_config() {
        // Arrange: disabled config
        let disabled_config = ChunkedPrefillConfig {
            enabled: false,
            ..ChunkedPrefillConfig::default()
        };
        let scheduler = ChunkedPrefillScheduler::new(disabled_config);

        // Act & Assert: even with large seq_len, disabled means no chunking
        assert!(!scheduler.should_chunk(100000));
    }

    #[test]
    fn dispatch_coordinator_adaptive_chunk_policy_high_concurrency_shrinks() {
        // Arrange: high concurrency reduces chunk size
        use crate::scheduler::chunked_prefill::AdaptiveChunkPolicy;
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.5;
        policy.concurrent_requests = 1;
        let remaining = 4096;
        let max_seq = 8192;

        // Act: single request
        let chunk_single = policy.compute_chunk_size(remaining, max_seq);

        // Act: high concurrency (32 requests)
        policy.concurrent_requests = 32;
        let chunk_many = policy.compute_chunk_size(remaining, max_seq);

        // Assert: more concurrency produces smaller or equal chunks
        assert!(chunk_many <= chunk_single);
        assert!(chunk_many >= 512); // never below base
    }

    #[test]
    fn dispatch_coordinator_eviction_select_victims_respects_max_count() {
        // Arrange: build metadata for 5 pages with varying access info
        let coord = make_coordinator();
        let mut metadata = std::collections::HashMap::new();
        let semantic = std::collections::HashMap::new();
        let now = std::time::Instant::now();
        for i in 0..5usize {
            metadata.insert(
                i,
                crate::scheduler::types::PageMetadata {
                    page_id: i,
                    sequence_id: Some(i as u64),
                    state: crate::scheduler::types::PageState::Standby,
                    recency: 100 + i as usize * 50,
                    is_lir: false,
                    swap_in_time: None,
                    warm_until: None,
                    access_count: 10 - i as usize,
                    last_access: now - std::time::Duration::from_secs((i + 1) as u64),
                },
            );
        }

        // Act: request only 2 victims
        let victims = coord.memory_manager.select_victims(&metadata, &semantic, 2);

        // Assert: at most 2 victims returned
        assert!(victims.len() <= 2);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_free_page_returns_error_for_wrong_tier() {
        // Arrange: allocate from L1, try to free from L2
        let mut coord = DispatchCoordinator {
            scheduler: PagedScheduler::new(64, 4, HGALConfig::default()),
            batcher: ContinuousBatcher::new()
                .with_chunked(ChunkedConfig::default()),
            chunked_prefill_scheduler: ChunkedPrefillScheduler::new(
                ChunkedPrefillConfig::default(),
            ),
            requests: HashMap::new(),
            memory_manager: GlobalMemoryManager::new_with_capacities(32, 32, 0),
            policy: PolicyVariant::default(),
        };
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let l2 = crate::scheduler::memory_manager::Tier::L2;
        let phys = coord.memory_manager.allocate_page(l1).unwrap();

        // Act: free the page from the wrong tier (L2 instead of L1)
        let result = coord.memory_manager.free_page(l2, phys);

        // Assert: error because phys was allocated in L1, not L2
        assert!(result.is_err());
        // Original allocation in L1 is still intact
        let l1_usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(l1_usage.used, 1);
    }

    // ── 15 new tests (wave-12x35) ─────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_sequence_context_len_advances_on_push_generated() {
        // Arrange: sequence with 3 prompt tokens, context_len starts at 3
        use crate::scheduler::sequence::Sequence;
        let mut seq = Sequence::new(1, vec![100, 200, 300]);
        assert_eq!(seq.context_len(), 3);

        // Act: push 4 generated tokens via push_generated_token
        seq.push_generated_token(400);
        seq.push_generated_token(500);
        seq.push_generated_token(600);
        seq.push_generated_token(700);

        // Assert: context_len advances by exactly 4
        assert_eq!(seq.context_len(), 7);
        assert_eq!(seq.generated_tokens, vec![400, 500, 600, 700]);
        assert_eq!(seq.position, 7);
    }

    #[test]
    fn dispatch_coordinator_sequence_context_len_saturating_on_overflow() {
        // Arrange: sequence with position at usize::MAX - 1
        use crate::scheduler::sequence::Sequence;
        let mut seq = Sequence::new(2, vec![10]);
        seq.position = usize::MAX - 1;

        // Act: push one token — position wraps to usize::MAX
        seq.push_generated_token(99);
        assert_eq!(seq.context_len(), usize::MAX);

        // Act: push another — saturating_add prevents overflow
        seq.push_generated_token(100);
        assert_eq!(seq.context_len(), usize::MAX);
    }

    #[test]
    fn dispatch_coordinator_batch_prep_data_default_all_vecs_zeroed() {
        // Arrange: construct BatchPrepData with 2 sequences
        let prep = crate::scheduler::batcher::BatchPrepData::new(2);

        // Assert: all vec fields are properly sized and zeroed
        assert_eq!(prep.prompt_lens, vec![0, 0]);
        assert_eq!(prep.kv_lens, vec![0, 0]);
        assert_eq!(prep.session_positions, vec![0, 0]);
        assert_eq!(prep.rope_pos_offsets, vec![0, 0]);
        assert_eq!(prep.max_new_tokens, vec![0, 0]);
        assert_eq!(prep.page_table_offsets, vec![0, 0]);
        assert_eq!(prep.page_table_lens, vec![0, 0]);
        assert_eq!(prep.fused_hidden_offsets, vec![0, 0]);
        assert_eq!(prep.num_mm_tokens, vec![0, 0]);
        assert_eq!(prep.active_flags, vec![1, 1]);
        assert_eq!(prep.seq_positions, vec![0, 0]);
        assert_eq!(prep.gen_counts, vec![0, 0]);
        assert_eq!(prep.last_sampled_tokens, vec![0, 0]);
        assert_eq!(prep.sampling_params_packed, vec![0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(prep.max_decode_steps, 0);
        assert_eq!(prep.total_prefill_tokens, 0);
    }

    #[test]
    fn dispatch_coordinator_batch_prep_data_zero_seqs_empty_vecs() {
        // Arrange: construct BatchPrepData with 0 sequences
        let prep = crate::scheduler::batcher::BatchPrepData::new(0);

        // Assert: all vecs are empty, scalars zero
        assert!(prep.prompt_lens.is_empty());
        assert!(prep.active_flags.is_empty());
        assert!(prep.sampling_params_packed.is_empty());
        assert_eq!(prep.max_decode_steps, 0);
        assert_eq!(prep.total_prefill_tokens, 0);
    }

    #[test]
    fn dispatch_coordinator_batch_result_all_variants_debug_output() {
        // Arrange: construct all four BatchResult variants
        use crate::scheduler::batcher::BatchResult;
        let telemetry = crate::scheduler::telemetry::SequenceTelemetry::new();
        let cont = BatchResult::continue_with_token(1, 42, telemetry.clone());
        let comp = BatchResult::complete(2, Some(99), telemetry.clone());
        let pause = BatchResult::pause(3);
        let fail = BatchResult::fail(4);

        // Act: format each via Debug
        let cont_dbg = format!("{:?}", cont);
        let comp_dbg = format!("{:?}", comp);
        let pause_dbg = format!("{:?}", pause);
        let fail_dbg = format!("{:?}", fail);

        // Assert: all debug outputs contain request_id and action
        assert!(cont_dbg.contains("request_id"), "Continue debug: {}", cont_dbg);
        assert!(comp_dbg.contains("request_id"), "Complete debug: {}", comp_dbg);
        assert!(pause_dbg.contains("request_id"), "Pause debug: {}", pause_dbg);
        assert!(fail_dbg.contains("request_id"), "Fail debug: {}", fail_dbg);
        assert!(cont_dbg.contains("Continue"), "Continue action in debug: {}", cont_dbg);
        assert!(comp_dbg.contains("Complete"), "Complete action in debug: {}", comp_dbg);
    }

    #[test]
    fn dispatch_coordinator_batch_result_all_variants_partial_eq() {
        // Arrange: verify PartialEq for each BatchResult variant
        use crate::scheduler::batcher::{BatchAction, BatchResult};
        let telemetry = crate::scheduler::telemetry::SequenceTelemetry::new();

        // Act & Assert: Continue variants
        let c1 = BatchResult::continue_with_token(1, 42, telemetry.clone());
        let c2 = BatchResult::continue_with_token(1, 42, telemetry.clone());
        assert_eq!(c1, c2);

        // Assert: different token makes them unequal
        let c3 = BatchResult::continue_with_token(1, 99, telemetry.clone());
        assert_ne!(c1, c3);

        // Assert: Complete with token
        let comp1 = BatchResult::complete(5, Some(10), telemetry.clone());
        let comp2 = BatchResult::complete(5, Some(10), telemetry.clone());
        assert_eq!(comp1, comp2);

        // Assert: Pause and Fail are distinct
        assert_ne!(c1.action, BatchAction::Pause);
        assert_ne!(c1.action, BatchAction::Fail);
        assert_ne!(BatchAction::Pause, BatchAction::Fail);
    }

    #[test]
    fn dispatch_coordinator_scheduler_decision_partial_eq_via_policy() {
        // Arrange: two identical SystemState inputs produce equal decisions
        let coord = make_coordinator();
        let state_a = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.3,
            kv_fragmentation: 0.1,
            current_running_len: 5,
            current_batch_size: 5,
            ..Default::default()
        };
        let state_b = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.3,
            kv_fragmentation: 0.1,
            current_running_len: 5,
            current_batch_size: 5,
            ..Default::default()
        };

        // Act: decide twice with identical inputs
        let decision_a = coord.policy.decide(&state_a);
        let decision_b = coord.policy.decide(&state_b);

        // Assert: decisions are equal (deterministic policy)
        assert_eq!(decision_a, decision_b);
        assert_eq!(decision_a.max_batch_size, decision_b.max_batch_size);
        assert_eq!(decision_a.admit_new_prefill, decision_b.admit_new_prefill);
        assert_eq!(decision_a.force_swap_out_count, decision_b.force_swap_out_count);
    }

    #[test]
    fn dispatch_coordinator_policy_variant_clone_independence_after_decide() {
        // Arrange: clone the policy, use both independently
        let coord = make_coordinator();
        let cloned_policy = coord.policy.clone();

        let state_high_pressure = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.95,
            kv_fragmentation: 0.0,
            current_running_len: 4,
            current_batch_size: 4,
            ..Default::default()
        };
        let state_low_pressure = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.1,
            kv_fragmentation: 0.0,
            current_running_len: 2,
            current_batch_size: 2,
            ..Default::default()
        };

        // Act: cloned decides high pressure, original decides low pressure
        let cloned_decision = cloned_policy.decide(&state_high_pressure);
        let original_decision = coord.policy.decide(&state_low_pressure);

        // Assert: decisions are different — clone independence verified
        assert!(!cloned_decision.admit_new_prefill);
        assert!(original_decision.admit_new_prefill);
        assert_ne!(cloned_decision, original_decision);
    }

    #[test]
    fn dispatch_coordinator_adaptive_chunk_policy_exact_boundary_0_25_and_0_75() {
        // Arrange: test exact boundary ratios of AdaptiveChunkPolicy
        use crate::scheduler::chunked_prefill::AdaptiveChunkPolicy;
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.concurrent_requests = 1;
        let remaining = 8192;
        let max_seq = 16384;

        // Act & Assert: at 0.25 exactly (< 0.25 is false, so middle range)
        policy.l1_available_ratio = 0.25;
        let at_025 = policy.compute_chunk_size(remaining, max_seq);
        // t = (0.25 - 0.25) / 0.50 = 0.0 → scaled = 512
        assert_eq!(at_025, 512);

        // Act & Assert: at 0.75 exactly (> 0.75 is false, so middle range)
        policy.l1_available_ratio = 0.75;
        let at_075 = policy.compute_chunk_size(remaining, max_seq);
        // t = (0.75 - 0.25) / 0.50 = 1.0 → scaled = 16384, capped by remaining=8192
        assert_eq!(at_075, 8192);

        // Act & Assert: just above 0.75 → upper range
        policy.l1_available_ratio = 0.7501;
        let above_075 = policy.compute_chunk_size(remaining, max_seq);
        assert_eq!(above_075, 8192); // capped by remaining

        // Act & Assert: just below 0.25 → lower range
        policy.l1_available_ratio = 0.2499;
        let below_025 = policy.compute_chunk_size(remaining, max_seq);
        assert_eq!(below_025, 512); // base chunk
    }

    #[test]
    fn dispatch_coordinator_adaptive_chunk_policy_remaining_less_than_base() {
        // Arrange: remaining tokens less than base chunk size
        use crate::scheduler::chunked_prefill::AdaptiveChunkPolicy;
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.9; // high ratio
        policy.concurrent_requests = 1;

        // Act: remaining=100 which is < base=512
        let chunk = policy.compute_chunk_size(100, 4096);

        // Assert: returns remaining (100), not base
        assert_eq!(chunk, 100);
    }

    #[test]
    fn dispatch_coordinator_adaptive_chunk_policy_concurrency_thresholds() {
        // Arrange: test the three concurrency tiers
        use crate::scheduler::chunked_prefill::AdaptiveChunkPolicy;
        let mut policy = AdaptiveChunkPolicy::new(512);
        policy.l1_available_ratio = 0.5; // middle range: t=0.5 → scaled=4608
        let remaining = 8192;
        let max_seq = 8192;

        // Act & Assert: <=4 requests → factor 1.0
        policy.concurrent_requests = 4;
        let low_conc = policy.compute_chunk_size(remaining, max_seq);
        assert!(low_conc >= 512);

        // Act & Assert: 5-8 requests → factor 0.75
        policy.concurrent_requests = 5;
        let mid_conc = policy.compute_chunk_size(remaining, max_seq);
        assert!(mid_conc >= 512);
        assert!(mid_conc <= low_conc);

        // Act & Assert: >8 requests → factor 0.5
        policy.concurrent_requests = 9;
        let high_conc = policy.compute_chunk_size(remaining, max_seq);
        assert!(high_conc >= 512);
        assert!(high_conc <= mid_conc);
    }

    #[test]
    fn dispatch_coordinator_sequence_push_generated_updates_position() {
        // Arrange: verify position increments match context_len after each push
        use crate::scheduler::sequence::Sequence;
        let mut seq = Sequence::new(7, vec![1, 2]); // position=2
        assert_eq!(seq.position, 2);
        assert_eq!(seq.context_len(), 2);

        // Act & Assert: push one token
        seq.push_generated_token(10);
        assert_eq!(seq.position, 3);
        assert_eq!(seq.context_len(), 3);
        assert!(!seq.needs_prefill());

        // Act & Assert: push another
        seq.push_generated_token(20);
        assert_eq!(seq.position, 4);
        assert_eq!(seq.context_len(), 4);
    }

    #[test]
    fn dispatch_coordinator_sequence_mark_running_assigns_kv_pages() {
        // Arrange
        use crate::scheduler::sequence::{Sequence, SequenceState};
        let mut seq = Sequence::new(10, vec![1, 2, 3]);
        assert!(seq.kv_pages.is_empty());
        assert_eq!(seq.state, SequenceState::Waiting);

        // Act: mark as running with KV pages
        let pages = vec![0, 1, 2, 3];
        seq.mark_running(pages.clone());

        // Assert: state transitioned and KV pages assigned
        assert_eq!(seq.state, SequenceState::Running);
        assert_eq!(seq.kv_pages, pages);
        assert_eq!(seq.kv_pages.len(), 4);
    }

    #[test]
    fn dispatch_coordinator_sequence_to_sequence_group_preserves_fields() {
        // Arrange
        use crate::scheduler::sequence::Sequence;
        use crate::scheduler::types::GroupState;
        let mut seq = Sequence::new(42, vec![10, 20, 30, 40]);
        let pages = vec![5, 6, 7];
        seq.kv_pages = pages.clone();

        // Act: convert to SequenceGroup
        let group = seq.to_sequence_group();

        // Assert: fields are preserved
        assert_eq!(group.id, 42);
        assert_eq!(group.pages, pages);
        assert_eq!(group.state, GroupState::Running);
        assert_eq!(group.context_len, 4);
        assert!(!group.is_pinned);
    }

    #[test]
    fn dispatch_coordinator_batch_prep_data_sampling_params_bounds_check() {
        // Arrange: BatchPrepData with 1 sequence
        let mut prep = crate::scheduler::batcher::BatchPrepData::new(1);

        // Act: set sampling params for out-of-bounds index 2
        // This should silently do nothing (no panic)
        prep.set_sampling_params(2, 0.5, 50, 0.9, 2);

        // Assert: only 4 packed slots exist (1 seq * 4 params)
        assert_eq!(prep.sampling_params_packed.len(), 4);

        // Act: set valid params for index 0
        prep.set_sampling_params(0, 0.8, 100, 0.95, 3);

        // Assert: first 4 slots are populated
        assert_eq!(prep.sampling_params_packed[0], 0.8f32.to_bits());
        assert_eq!(prep.sampling_params_packed[1], 100);
        assert_eq!(prep.sampling_params_packed[2], 0.95f32.to_bits());
        assert_eq!(prep.sampling_params_packed[3], 3);
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_mark_as_chunked_sets_phase() {
        // Arrange: create a coordinator and a request in prefill
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        let coord = make_coordinator();
        let mut req_state = RequestState::new(1, RequestPhase::Prefill, 2048, 0);

        // Act: mark as chunked
        coord.chunked_prefill_scheduler.mark_as_chunked(&mut req_state);

        // Assert: phase is now ChunkedPrefill
        assert_eq!(req_state.phase, RequestPhase::ChunkedPrefill);
    }

    // -- 15 new tests (wave-12x36) --

    #[test]
    fn dispatch_coordinator_construction_all_fields_accessible() {
        // Arrange: construct coordinator and verify every field is accessible
        let coord = make_coordinator();

        // Act & Assert: each pub field can be borrowed without panic
        let _scheduler_page_size = coord.scheduler.page_size();
        let _batcher_has_work = coord.batcher.has_pending_work();
        let _chunked_config = coord.chunked_prefill_scheduler.config();
        let _requests_len = coord.requests.len();
        let _l1_usage = coord.memory_manager.tier_usage(
            crate::scheduler::memory_manager::Tier::L1,
        );
        let _policy_is_absolute = matches!(coord.policy, PolicyVariant::Absolute);

        // Assert: all fields return valid values (no panic = accessible)
        assert_eq!(_scheduler_page_size, 4);
        assert!(!_batcher_has_work);
        assert_eq!(_requests_len, 0);
        assert_eq!(_l1_usage.capacity, 32);
        assert!(_policy_is_absolute);
    }

    #[test]
    fn dispatch_coordinator_scheduler_decision_all_fields_covered() {
        // Arrange: construct a SchedulerDecision with non-default values
        let decision = crate::scheduler::jit_types::SchedulerDecision {
            max_batch_size: 16,
            admit_new_prefill: true,
            force_swap_out_count: 3,
        };

        // Act & Assert: all three fields are readable and correct
        assert_eq!(decision.max_batch_size, 16);
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 3);

        // Assert: PartialEq compares all fields
        let same = crate::scheduler::jit_types::SchedulerDecision {
            max_batch_size: 16,
            admit_new_prefill: true,
            force_swap_out_count: 3,
        };
        assert_eq!(decision, same);

        // Assert: changing any field breaks equality
        let diff_batch = crate::scheduler::jit_types::SchedulerDecision {
            max_batch_size: 15,
            ..decision
        };
        assert_ne!(decision, diff_batch);

        let diff_admit = crate::scheduler::jit_types::SchedulerDecision {
            admit_new_prefill: false,
            ..decision
        };
        assert_ne!(decision, diff_admit);

        let diff_swap = crate::scheduler::jit_types::SchedulerDecision {
            force_swap_out_count: 0,
            ..decision
        };
        assert_ne!(decision, diff_swap);
    }

    #[test]
    fn dispatch_coordinator_policy_variant_clone_independence_between_copies() {
        // Arrange: create a policy, clone it, make separate decisions
        let coord = make_coordinator();
        let clone_a = coord.policy.clone();
        let clone_b = coord.policy.clone();

        // Act: both clones decide independently with different states
        let state_low = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.1,
            ..Default::default()
        };
        let state_high = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.95,
            ..Default::default()
        };

        let dec_a = clone_a.decide(&state_low);
        let dec_b = clone_b.decide(&state_high);

        // Assert: decisions are different, proving independent internal state
        assert!(dec_a.admit_new_prefill);
        assert!(!dec_b.admit_new_prefill);

        // Assert: original policy still works and is independent
        let dec_orig = coord.policy.decide(&state_low);
        assert_eq!(dec_orig, dec_a);
    }

    #[test]
    fn dispatch_coordinator_adaptive_chunk_policy_factor_zero_and_max() {
        // Arrange: test AdaptiveChunkPolicy with extreme factor values
        use crate::scheduler::chunked_prefill::AdaptiveChunkPolicy;

        // Act & Assert: base chunk size of 0 (degenerate)
        let mut policy_zero = AdaptiveChunkPolicy::new(0);
        policy_zero.l1_available_ratio = 0.1;
        policy_zero.concurrent_requests = 1;
        let chunk_zero = policy_zero.compute_chunk_size(100, 4096);
        // With base=0 and remaining=100 > base=0: result = max(0, min(0,100)) = 0
        // But remaining <= base? 100 <= 0 is false, so result.max(0).min(100)
        // adaptive = 0, concurrency_factor = 1.0, result = 0
        // remaining(100) > base(0): result.max(0).min(100) = 0
        assert_eq!(chunk_zero, 0);

        // Act & Assert: very large max_seq_len
        let mut policy_large = AdaptiveChunkPolicy::new(512);
        policy_large.l1_available_ratio = 0.5;
        policy_large.concurrent_requests = 1;
        let chunk_large = policy_large.compute_chunk_size(10000, usize::MAX);
        // t = (0.5 - 0.25) / 0.50 = 0.5; scaled = 512 + 0.5*(usize::MAX - 512)
        // capped by remaining=10000
        assert!(chunk_large >= 512);
        assert!(chunk_large <= 10000);
    }

    #[test]
    fn dispatch_coordinator_batch_manifest_empty_slots_no_compact() {
        // Arrange: BatchManifest with zero slots
        let config = ChunkedPrefillConfig::default();
        let manifest = crate::scheduler::chunked_prefill::BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };

        // Act & Assert: empty manifest should not trigger compact
        assert!(!manifest.should_compact(&config));
        assert_eq!(manifest.total_tokens, 0);
        assert_eq!(manifest.decode_tokens, 0);
        assert_eq!(manifest.prefill_tokens, 0);
    }

    #[test]
    fn dispatch_coordinator_batch_result_complete_without_token() {
        // Arrange: BatchResult::complete with None token (end-of-sequence, no token produced)
        use crate::scheduler::batcher::BatchResult;
        let telemetry = crate::scheduler::telemetry::SequenceTelemetry::new();

        // Act
        let result = BatchResult::complete(42, None, telemetry);

        // Assert: action is Complete but no generated token
        assert_eq!(result.request_id, 42);
        assert_eq!(result.action, crate::scheduler::batcher::BatchAction::Complete);
        assert!(result.generated_token.is_none());

        // Assert: distinct from continue_with_token
        let cont = BatchResult::continue_with_token(42, 0, crate::scheduler::telemetry::SequenceTelemetry::new());
        assert_ne!(result.action, cont.action);
    }

    #[test]
    fn dispatch_coordinator_sequence_state_full_transition_chain() {
        // Arrange: test Pending(Waiting) -> Running -> Completed transition chain
        use crate::scheduler::sequence::{Sequence, SequenceState};

        let mut seq = Sequence::new(1, vec![10, 20, 30]);
        assert_eq!(seq.state, SequenceState::Waiting);

        // Act: transition Waiting -> Running
        seq.mark_running(vec![0, 1, 2]);
        assert_eq!(seq.state, SequenceState::Running);
        assert!(seq.needs_prefill()); // no generated tokens yet

        // Act: generate tokens (transition out of prefill)
        seq.push_generated_token(40);
        assert!(!seq.needs_prefill());

        // Act: transition Running -> Completed
        seq.state = SequenceState::Completed;
        assert_eq!(seq.state, SequenceState::Completed);
        assert_eq!(seq.context_len(), 4);
        assert_eq!(seq.generated_tokens, vec![40]);
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_phase_transition_prefill_to_chunked() {
        // Arrange: verify ChunkedPrefill phase transitions
        use crate::scheduler::request_state::{RequestPhase, RequestState};

        let mut req_state = RequestState::new(1, RequestPhase::Prefill, 2048, 0);
        assert_eq!(req_state.phase, RequestPhase::Prefill);

        // Act: transition Prefill -> ChunkedPrefill
        let coord = make_coordinator();
        coord.chunked_prefill_scheduler.mark_as_chunked(&mut req_state);

        // Assert: phase changed
        assert_eq!(req_state.phase, RequestPhase::ChunkedPrefill);

        // Act: transition ChunkedPrefill -> Decode (manual)
        req_state.phase = RequestPhase::Decode;

        // Assert: decode phase set
        assert_eq!(req_state.phase, RequestPhase::Decode);
        assert_ne!(req_state.phase, RequestPhase::Prefill);
        assert_ne!(req_state.phase, RequestPhase::ChunkedPrefill);
    }

    #[test]
    fn dispatch_coordinator_sequence_push_mark_running_roundtrip() {
        // Arrange: Sequence -> mark_running -> push_generated -> to_sequence_group roundtrip
        use crate::scheduler::sequence::Sequence;
        use crate::scheduler::types::GroupState;

        let mut seq = Sequence::new(88, vec![1, 2, 3, 4]);
        assert_eq!(seq.context_len(), 4);
        assert!(seq.kv_pages.is_empty());

        // Act: mark running
        seq.mark_running(vec![10, 20]);
        assert_eq!(seq.state, crate::scheduler::sequence::SequenceState::Running);
        assert_eq!(seq.kv_pages, vec![10, 20]);

        // Act: generate tokens
        for i in 100..105u32 {
            seq.push_generated_token(i);
        }

        // Act: convert to group
        let group = seq.to_sequence_group();

        // Assert: roundtrip preserves all data
        assert_eq!(group.id, 88);
        assert_eq!(group.pages, vec![10, 20]);
        assert_eq!(group.state, GroupState::Running);
        assert_eq!(group.context_len, 9); // 4 prompt + 5 generated
        assert!(!group.is_pinned);
    }

    #[test]
    fn dispatch_coordinator_batch_prep_data_default_trait_all_zeroed() {
        // Arrange: verify that BatchPrepData::new(0) and new(N) produce correct defaults
        let prep_empty = crate::scheduler::batcher::BatchPrepData::new(0);
        assert_eq!(prep_empty.max_decode_steps, 0);
        assert_eq!(prep_empty.total_prefill_tokens, 0);
        assert!(prep_empty.prompt_lens.is_empty());
        assert!(prep_empty.sampling_params_packed.is_empty());

        // Act: create with N=5
        let prep_5 = crate::scheduler::batcher::BatchPrepData::new(5);

        // Assert: all vec fields have length 5, sampling_params has length 20 (5*4)
        assert_eq!(prep_5.prompt_lens.len(), 5);
        assert_eq!(prep_5.kv_lens.len(), 5);
        assert_eq!(prep_5.session_positions.len(), 5);
        assert_eq!(prep_5.rope_pos_offsets.len(), 5);
        assert_eq!(prep_5.max_new_tokens.len(), 5);
        assert_eq!(prep_5.page_table_offsets.len(), 5);
        assert_eq!(prep_5.page_table_lens.len(), 5);
        assert_eq!(prep_5.fused_hidden_offsets.len(), 5);
        assert_eq!(prep_5.num_mm_tokens.len(), 5);
        assert_eq!(prep_5.active_flags.len(), 5);
        assert_eq!(prep_5.seq_positions.len(), 5);
        assert_eq!(prep_5.gen_counts.len(), 5);
        assert_eq!(prep_5.last_sampled_tokens.len(), 5);
        assert_eq!(prep_5.sampling_params_packed.len(), 20);

        // Assert: active_flags are all 1
        assert_eq!(prep_5.active_flags, vec![1, 1, 1, 1, 1]);

        // Assert: all other vec fields are zeroed
        assert_eq!(prep_5.prompt_lens, vec![0, 0, 0, 0, 0]);
        assert_eq!(prep_5.kv_lens, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn dispatch_coordinator_eviction_victim_selection_empty_candidates() {
        // Arrange: coordinator with empty metadata, requesting various victim counts
        let coord = make_coordinator();
        let metadata = std::collections::HashMap::new();
        let semantic = std::collections::HashMap::new();

        // Act & Assert: requesting 0, 1, and many victims from empty set
        let v0 = coord.memory_manager.select_victims(&metadata, &semantic, 0);
        assert!(v0.is_empty());

        let v1 = coord.memory_manager.select_victims(&metadata, &semantic, 1);
        assert!(v1.is_empty());

        let v_many = coord.memory_manager.select_victims(&metadata, &semantic, 100);
        assert!(v_many.is_empty());
    }

    #[test]
    fn dispatch_coordinator_tier_allocation_math_with_large_capacity() {
        // Arrange: create a coordinator with large L1 capacity
        let mut coord = DispatchCoordinator {
            scheduler: PagedScheduler::new(1024, 4, HGALConfig::default()),
            batcher: ContinuousBatcher::new()
                .with_chunked(ChunkedConfig::default()),
            chunked_prefill_scheduler: ChunkedPrefillScheduler::new(
                ChunkedPrefillConfig::default(),
            ),
            requests: HashMap::new(),
            memory_manager: GlobalMemoryManager::new_with_capacities(1000, 0, 0),
            policy: PolicyVariant::default(),
        };
        let l1 = crate::scheduler::memory_manager::Tier::L1;

        // Act: allocate 500 pages
        for _ in 0..500 {
            coord.memory_manager.allocate_page(l1).unwrap();
        }

        // Assert: usage math is exact
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.capacity, 1000);
        assert_eq!(usage.used, 500);
        assert_eq!(usage.available(), 500);

        // Act: free 200 pages
        for i in 0..200usize {
            coord.memory_manager.free_page(l1, i).unwrap();
        }

        // Assert: usage reflects frees
        let usage_after = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage_after.used, 300);
        assert_eq!(usage_after.available(), 700);
    }

    #[test]
    fn dispatch_coordinator_context_length_advancement_large_token_count() {
        // Arrange: sequence with a large prompt, then advance near usize limits
        use crate::scheduler::sequence::Sequence;

        let mut seq = Sequence::new(1, vec![0; 1000]);
        assert_eq!(seq.context_len(), 1000);
        assert_eq!(seq.position, 1000);

        // Act: push many generated tokens
        for i in 0..5000u32 {
            seq.push_generated_token(i);
        }

        // Assert: context_len = 1000 prompt + 5000 generated = 6000
        assert_eq!(seq.context_len(), 6000);
        assert_eq!(seq.position, 6000);
        assert_eq!(seq.generated_tokens.len(), 5000);
        assert!(!seq.needs_prefill());

        // Act: convert to sequence group
        let group = seq.to_sequence_group();
        assert_eq!(group.context_len, 6000);
    }

    #[test]
    fn dispatch_coordinator_draft_budget_tracking_speculative() {
        // Arrange: simulate speculative decoding draft budget tracking
        use crate::scheduler::sequence::Sequence;

        let mut seq = Sequence::new(1, vec![10, 20, 30]);
        assert_eq!(seq.draft_budget, 0);

        // Act: set draft budget for speculative decoding (e.g., 5 candidate tokens)
        seq.draft_budget = 5;

        // Assert: budget set
        assert_eq!(seq.draft_budget, 5);

        // Act: generate tokens, decrementing budget
        for i in 100..105u32 {
            seq.push_generated_token(i);
            seq.draft_budget = seq.draft_budget.saturating_sub(1);
        }

        // Assert: budget exhausted
        assert_eq!(seq.draft_budget, 0);
        assert_eq!(seq.generated_tokens.len(), 5);
        assert_eq!(seq.context_len(), 8); // 3 prompt + 5 generated

        // Act: verify budget doesn't go below 0 with saturating_sub
        seq.draft_budget = seq.draft_budget.saturating_sub(10);
        assert_eq!(seq.draft_budget, 0);
    }

    // -- 15 new tests (wave-12x37) --

    #[test]
    fn dispatch_coordinator_page_table_map_resolve_remove() {
        // Arrange: construct a PageTable, map a virtual page, resolve it, then remove
        use crate::scheduler::memory_manager::{PageTable, VirtualPageId, Tier};
        let mut table = PageTable::new();
        let vpage = VirtualPageId::new(10, 3);

        // Act: map virtual -> physical
        table.map(vpage, Tier::L1, 7);

        // Assert: resolve returns the mapped location
        let resolved = table.resolve(vpage);
        assert!(resolved.is_some());
        let r = resolved.unwrap();
        assert_eq!(r.physical_id, 7);
        assert_eq!(r.tier, Tier::L1);

        // Act: remove the mapping
        let removed = table.remove(&vpage);

        // Assert: remove returns the old location
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().physical_id, 7);

        // Assert: resolve now returns None
        assert!(table.resolve(vpage).is_none());
    }

    #[test]
    fn dispatch_coordinator_page_table_resolve_unmapped_returns_none() {
        // Arrange: empty PageTable
        use crate::scheduler::memory_manager::{PageTable, VirtualPageId};
        let table = PageTable::new();
        let vpage = VirtualPageId::new(99, 0);

        // Act & Assert: unmapped virtual page returns None
        assert!(table.resolve(vpage).is_none());
    }

    #[test]
    fn dispatch_coordinator_page_table_remap_updates_location() {
        // Arrange: map a virtual page to one location, then remap to another
        use crate::scheduler::memory_manager::{PageTable, VirtualPageId, Tier};
        let mut table = PageTable::new();
        let vpage = VirtualPageId::new(5, 2);
        table.map(vpage, Tier::L1, 10);

        // Act: remap to different location
        table.remap(vpage, Tier::L2, 20).unwrap();

        // Assert: resolve now returns the new location
        let resolved = table.resolve(vpage).unwrap();
        assert_eq!(resolved.physical_id, 20);
        assert_eq!(resolved.tier, Tier::L2);
    }

    #[test]
    fn dispatch_coordinator_compact_scatter_meta_default_values() {
        // Arrange: construct CompactScatterMeta and verify defaults
        use crate::scheduler::request_state::CompactScatterMeta;
        let meta = CompactScatterMeta {
            original_slot: 3,
            compacted_slot: 1,
            active: 1,
        };

        // Assert: fields are readable and match construction
        assert_eq!(meta.original_slot, 3);
        assert_eq!(meta.compacted_slot, 1);
        assert_eq!(meta.active, 1);
    }

    #[test]
    fn dispatch_coordinator_request_telemetry_default_zeroed() {
        // Arrange: construct RequestTelemetry with defaults
        use crate::scheduler::request_state::RequestTelemetry;
        let telemetry = RequestTelemetry::default();

        // Assert: entropy/centroid/range_group start at zero, residual defaults are 1.0
        assert_eq!(telemetry.entropy, 0.0);
        assert_eq!(telemetry.centroid, 0.0);
        assert_eq!(telemetry.residual_delta, 1.0);
        assert_eq!(telemetry.residual_cosine, 1.0);
        assert_eq!(telemetry.range_group, 0);
    }

    #[test]
    fn dispatch_coordinator_request_state_with_target_layer() {
        // Arrange: construct RequestState then set target layer
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        let req = RequestState::new(42, RequestPhase::Prefill, 128, 0)
            .with_target_layer(7);

        // Assert: target_layer is set
        assert_eq!(req.target_layer, 7);
        assert_eq!(req.request_id, 42);
        assert_eq!(req.seq_len, 128);
    }

    #[test]
    fn dispatch_coordinator_request_state_exit_flag_transitions() {
        // Arrange: RequestState starts with exit_flag = 0
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        let req = RequestState::new(1, RequestPhase::Decode, 50, 0);
        assert!(!req.is_exited());

        // Act: mark exited
        req.mark_exited();

        // Assert: is_exited returns true
        assert!(req.is_exited());

        // Act: reset exit
        req.reset_exit();

        // Assert: is_exited returns false again
        assert!(!req.is_exited());
    }

    #[test]
    fn dispatch_coordinator_request_state_is_full_model_default_true() {
        // Arrange: default target_layer = 0 means full model
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        let req = RequestState::new(1, RequestPhase::Prefill, 100, 0);

        // Assert: target_layer=0 means full model
        assert!(req.is_full_model());

        // Act: set target layer to non-zero (early exit)
        let req_with_layer = RequestState::new(2, RequestPhase::Prefill, 100, 0)
            .with_target_layer(12);

        // Assert: not full model
        assert!(!req_with_layer.is_full_model());
    }

    #[test]
    fn dispatch_coordinator_group_state_all_variants_distinct() {
        // Arrange: verify all GroupState variants
        use crate::scheduler::types::GroupState;
        let variants = [
            GroupState::Running,
            GroupState::Swapped,
            GroupState::Paused,
        ];

        // Assert: all three variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    #[test]
    fn dispatch_coordinator_page_state_all_variants_distinct() {
        // Arrange: verify all PageState variants
        use crate::scheduler::types::PageState;
        let variants = [
            PageState::Free,
            PageState::Active,
            PageState::Standby,
            PageState::SwappedOut,
            PageState::Warm,
            PageState::Protected,
            PageState::Swapped,
        ];

        // Assert: all seven variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    #[test]
    fn dispatch_coordinator_kv_pipeline_variants_and_equality() {
        // Arrange: verify KvPipeline variants
        use crate::scheduler::types::KvPipeline;
        let working = KvPipeline::Working;
        let conversation = KvPipeline::Conversation;

        // Assert: distinct variants
        assert_ne!(working, conversation);

        // Assert: same variants are equal
        assert_eq!(working, KvPipeline::Working);
        assert_eq!(conversation, KvPipeline::Conversation);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_error_display_trait() {
        // Arrange: construct each MemoryManagerError variant and format via Display
        use crate::scheduler::memory_manager::{MemoryManagerError, Tier, VirtualPageId};
        let tier_err = MemoryManagerError::TierCapacityExceeded { tier: Tier::L1 };
        let phys_err = MemoryManagerError::UnknownPhysicalPage { tier: Tier::L2, physical_id: 42 };
        let virt_err = MemoryManagerError::UnknownVirtualPage { virtual_id: VirtualPageId::new(1, 2) };
        let session_err = MemoryManagerError::UnknownSession { session_id: 99 };

        // Act & Assert: each Display output contains identifying text
        let tier_str = format!("{}", tier_err);
        assert!(tier_str.contains("L1"), "TierCapacityExceeded Display should mention tier: {}", tier_str);

        let phys_str = format!("{}", phys_err);
        assert!(phys_str.contains("42"), "UnknownPhysicalPage Display should mention id: {}", phys_str);

        let virt_str = format!("{}", virt_err);
        assert!(virt_str.contains("virtual"), "UnknownVirtualPage Display: {}", virt_str);

        let session_str = format!("{}", session_err);
        assert!(session_str.contains("99"), "UnknownSession Display: {}", session_str);
    }

    #[test]
    fn dispatch_coordinator_virtual_page_id_new_and_fields() {
        // Arrange: construct VirtualPageId with specific values
        use crate::scheduler::memory_manager::VirtualPageId;
        let vpage = VirtualPageId::new(42, 7);

        // Assert: fields match construction
        assert_eq!(vpage.sequence_id, 42);
        assert_eq!(vpage.logical_index, 7);

        // Assert: different values produce distinct instances
        let other = VirtualPageId::new(42, 8);
        assert_ne!(vpage, other);
    }

    #[test]
    fn dispatch_coordinator_session_kv_cache_construction_fields() {
        // Arrange: register a session and verify SessionKvCache fields
        let mut coord = make_coordinator();
        let session = coord.memory_manager.register_session(123);

        // Assert: initial state has correct session_id and empty pages
        assert_eq!(session.session_id, 123);
        assert!(session.pages.is_empty());
        assert_eq!(session.finalized_position, 0);
    }

    #[test]
    fn dispatch_coordinator_sequence_telemetry_default_all_zeroed() {
        // Arrange: construct SequenceTelemetry via new()
        use crate::scheduler::telemetry::SequenceTelemetry;
        let telem = SequenceTelemetry::new();

        // Assert: all fields are zeroed or false
        assert_eq!(telem.l2_delta, 0.0);
        assert!(!telem.has_outlier);
        assert_eq!(telem.dead_density, 0.0);
        assert_eq!(telem.per_head_entropy, 0.0);
        assert_eq!(telem.transform_ratio, 0.0);
        assert_eq!(telem.output_entropy, 0.0);
    }

    // ── 13 new tests (wave-12x38) ──────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_requests_iter_collects_all_keys() {
        // Arrange: insert 6 requests with non-contiguous keys
        let mut coord = make_coordinator();
        let keys = [10u64, 20, 30, 40, 50, 60];
        for &k in &keys {
            let req_data = RequestData {
                prompt_tokens: vec![k as u32],
                output_tokens: vec![],
                sampling_config: Default::default(),
                phase: RequestPhase::Prefill,
                max_new_tokens: 10,
                finished: false,
                session_id: None,
                thinking_budget: None,
                fused_prefill_hidden: None,
            };
            coord.requests.insert(k, req_data);
        }

        // Act: collect all keys via iteration
        let mut collected_keys: Vec<u64> = coord.requests.keys().copied().collect();
        collected_keys.sort();

        // Assert: all keys present and in sorted order
        assert_eq!(collected_keys, vec![10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn dispatch_coordinator_requests_retain_removes_unfinished() {
        // Arrange: insert 4 requests, 2 with finished=true, 2 with finished=false
        let mut coord = make_coordinator();
        for i in 0..4u64 {
            let req_data = RequestData {
                prompt_tokens: vec![i as u32],
                output_tokens: vec![],
                sampling_config: Default::default(),
                phase: RequestPhase::Decode,
                max_new_tokens: 10,
                finished: i < 2, // first two are finished
                session_id: None,
                thinking_budget: None,
                fused_prefill_hidden: None,
            };
            coord.requests.insert(i, req_data);
        }

        // Act: retain only unfinished requests
        coord.requests.retain(|_, r| !r.finished);

        // Assert: only keys 2 and 3 remain
        assert_eq!(coord.requests.len(), 2);
        assert!(!coord.requests.contains_key(&0));
        assert!(!coord.requests.contains_key(&1));
        assert!(coord.requests.contains_key(&2));
        assert!(coord.requests.contains_key(&3));
    }

    #[test]
    fn dispatch_coordinator_request_data_output_tokens_replace() {
        // Arrange: create a request with output tokens
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1, 2],
            output_tokens: vec![10, 20, 30],
            sampling_config: Default::default(),
            phase: RequestPhase::Decode,
            max_new_tokens: 100,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(1, req_data);

        // Act: completely replace output_tokens
        let req = coord.requests.get_mut(&1).unwrap();
        req.output_tokens = vec![99, 88, 77, 66];

        // Assert: replacement is exact
        assert_eq!(coord.requests[&1].output_tokens, vec![99, 88, 77, 66]);
        assert_eq!(coord.requests[&1].output_tokens.len(), 4);
    }

    #[test]
    fn dispatch_coordinator_request_data_clear_output_tokens() {
        // Arrange: request with accumulated output tokens
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![10, 20, 30, 40, 50],
            sampling_config: Default::default(),
            phase: RequestPhase::Decode,
            max_new_tokens: 100,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(1, req_data);

        // Act: clear output tokens
        coord.requests.get_mut(&1).unwrap().output_tokens.clear();

        // Assert: output tokens now empty but request still exists
        assert!(coord.requests[&1].output_tokens.is_empty());
        assert!(coord.requests.contains_key(&1));
    }

    #[test]
    fn dispatch_coordinator_memory_manager_track_and_untrack_page() {
        // Arrange: allocate a page, then track/untrack it
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let phys = coord.memory_manager.allocate_page(l1).unwrap();

        // Act: track the page (already tracked by allocation, but verify API works)
        let track_result = coord.memory_manager.track_page(l1, phys);
        // Already tracked, so this should return false or be a no-op
        // (track_page returns bool — true if newly tracked)

        // Assert: page is allocated and tracked
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.used, 1);

        // Act: untrack the page
        coord.memory_manager.untrack_page(l1, phys);

        // Assert: usage decremented
        let usage_after = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage_after.used, 0);
    }

    #[test]
    fn dispatch_coordinator_virtual_page_id_as_hash_map_key() {
        // Arrange: use VirtualPageId as HashMap key
        use std::collections::HashMap;
        use crate::scheduler::memory_manager::VirtualPageId;
        let mut map: HashMap<VirtualPageId, u64> = HashMap::new();

        let vp1 = VirtualPageId::new(1, 0);
        let vp2 = VirtualPageId::new(1, 1);
        let vp3 = VirtualPageId::new(2, 0);

        // Act: insert different virtual page IDs
        map.insert(vp1, 100);
        map.insert(vp2, 200);
        map.insert(vp3, 300);

        // Assert: all entries present and distinct
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&VirtualPageId::new(1, 0)), Some(&100));
        assert_eq!(map.get(&VirtualPageId::new(1, 1)), Some(&200));
        assert_eq!(map.get(&VirtualPageId::new(2, 0)), Some(&300));

        // Assert: overwriting a key replaces the value
        map.insert(VirtualPageId::new(1, 0), 999);
        assert_eq!(map.get(&VirtualPageId::new(1, 0)), Some(&999));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn dispatch_coordinator_page_table_multiple_entries_independent() {
        // Arrange: map multiple virtual pages in a single PageTable
        use crate::scheduler::memory_manager::{PageTable, VirtualPageId, Tier};
        let mut table = PageTable::new();

        let vp_a = VirtualPageId::new(1, 0);
        let vp_b = VirtualPageId::new(2, 0);
        let vp_c = VirtualPageId::new(3, 5);

        // Act: map three entries
        table.map(vp_a, Tier::L1, 10);
        table.map(vp_b, Tier::L2, 20);
        table.map(vp_c, Tier::L1, 30);

        // Assert: each resolves independently
        let a = table.resolve(vp_a).unwrap();
        assert_eq!(a.physical_id, 10);
        assert_eq!(a.tier, Tier::L1);

        let b = table.resolve(vp_b).unwrap();
        assert_eq!(b.physical_id, 20);
        assert_eq!(b.tier, Tier::L2);

        let c = table.resolve(vp_c).unwrap();
        assert_eq!(c.physical_id, 30);
        assert_eq!(c.tier, Tier::L1);

        // Act: remove one entry, others remain
        table.remove(&vp_b);
        assert!(table.resolve(vp_b).is_none());
        assert!(table.resolve(vp_a).is_some());
        assert!(table.resolve(vp_c).is_some());
    }

    #[test]
    fn dispatch_coordinator_scheduled_batch_default_fields() {
        // Arrange: construct a ScheduledBatch manually
        let batch = crate::scheduler::batcher::ScheduledBatch {
            requests: vec![1, 2, 3],
            seq_offsets: vec![0, 10, 20],
            draft_steps: vec![0, 0, 0],
        };

        // Assert: fields are readable and match construction
        assert_eq!(batch.requests, vec![1, 2, 3]);
        assert_eq!(batch.seq_offsets, vec![0, 10, 20]);
        assert_eq!(batch.draft_steps, vec![0, 0, 0]);
        assert_eq!(batch.requests.len(), 3);
    }

    #[test]
    fn dispatch_coordinator_scheduled_batch_empty_is_valid() {
        // Arrange: an empty ScheduledBatch
        let batch = crate::scheduler::batcher::ScheduledBatch {
            requests: vec![],
            seq_offsets: vec![],
            draft_steps: vec![],
        };

        // Assert: empty batch has zero length and is valid
        assert!(batch.requests.is_empty());
        assert!(batch.seq_offsets.is_empty());
        assert!(batch.draft_steps.is_empty());
        assert_eq!(batch.requests.len(), batch.seq_offsets.len());
        assert_eq!(batch.requests.len(), batch.draft_steps.len());
    }

    #[test]
    fn dispatch_coordinator_request_state_is_active_reflects_compact_scatter() {
        // Arrange: default RequestState has compact_scatter.active = 1
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        let mut req = RequestState::new(1, RequestPhase::Prefill, 100, 0);

        // Assert: fresh request is active (active=1 from construction)
        assert!(req.is_active());

        // Act: deactivate by setting active=0
        req.compact_scatter.active = 0;

        // Assert: no longer active
        assert!(!req.is_active());

        // Act: re-activate
        req.compact_scatter.active = 1;

        // Assert: active again
        assert!(req.is_active());
    }

    #[test]
    fn dispatch_coordinator_request_state_set_original_slot() {
        // Arrange
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        let mut req = RequestState::new(1, RequestPhase::Prefill, 100, 0);

        // Assert: default original_slot
        assert_eq!(req.compact_scatter.original_slot, 0);

        // Act: set original slot
        req.set_original_slot(5);

        // Assert: original_slot updated
        assert_eq!(req.compact_scatter.original_slot, 5);
    }

    #[test]
    fn dispatch_coordinator_request_state_update_telemetry_values() {
        // Arrange
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        let mut req = RequestState::new(1, RequestPhase::Decode, 50, 0);
        assert_eq!(req.telemetry.entropy, 0.0);
        assert_eq!(req.telemetry.centroid, 0.0);

        // Act: update telemetry
        req.update_telemetry(2.5, 0.8, 0.3, 0.95);

        // Assert: telemetry fields updated
        assert!((req.telemetry.entropy - 2.5).abs() < f32::EPSILON);
        assert!((req.telemetry.centroid - 0.8).abs() < f32::EPSILON);
        assert!((req.telemetry.residual_delta - 0.3).abs() < f32::EPSILON);
        assert!((req.telemetry.residual_cosine - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn dispatch_coordinator_policy_config_default_values() {
        // Arrange: construct PolicyConfig with defaults and verify all fields
        use crate::scheduler::policy::PolicyConfig;
        let config = PolicyConfig::absolute();

        // Assert: all default values are within expected ranges
        assert!(config.pressure_emergency > 0.0 && config.pressure_emergency < 1.0);
        assert!(config.pressure_aggressive_ceiling > config.pressure_emergency);
        assert!(config.frag_defrag_threshold > 0.0 && config.frag_defrag_threshold < 1.0);
        assert!(config.batch_safe > 0);
        assert!(config.batch_normal > 0);
        assert!(config.batch_aggressive > 0);
    }

    // ── 13 new tests (wave-12x39) ──────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_memory_manager_entropy_evict_low_entropy_pages() {
        // Arrange: allocate pages, simulate low-entropy measurements
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let _p0 = coord.memory_manager.allocate_page(l1).unwrap();
        let _p1 = coord.memory_manager.allocate_page(l1).unwrap();
        let _p2 = coord.memory_manager.allocate_page(l1).unwrap();

        // Act: entropy_evict with threshold 0.5 — all pages have entropy 0.0 < 0.5
        let mut entropies = std::collections::HashMap::new();
        entropies.insert(0, 0.01);
        entropies.insert(1, 0.49);
        // page 2 has high entropy, should not be evicted
        entropies.insert(2, 0.8);
        let freed = coord.memory_manager.entropy_evict(&entropies, 0.5, l1);

        // Assert: two low-entropy pages freed, one high-entropy page remains
        assert_eq!(freed, 2);
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.used, 1);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_entropy_evict_no_low_entropy() {
        // Arrange: allocate pages, all with high entropy
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let _p0 = coord.memory_manager.allocate_page(l1).unwrap();
        let _p1 = coord.memory_manager.allocate_page(l1).unwrap();

        // Act: all pages have high entropy
        let mut entropies = std::collections::HashMap::new();
        entropies.insert(0, 1.5);
        entropies.insert(1, 3.0);
        let freed = coord.memory_manager.entropy_evict(&entropies, 0.5, l1);

        // Assert: nothing evicted
        assert_eq!(freed, 0);
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.used, 2);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_pipeline_allocate_and_track() {
        // Arrange: allocate a page via pipeline API
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;

        // Act: allocate in Working pipeline for request 42
        let pid = coord.memory_manager.allocate_page_in_pipeline(
            crate::scheduler::types::KvPipeline::Working,
            42,
            l1,
        );

        // Assert: allocation succeeds and L1 usage is 1
        assert!(pid.is_ok());
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.used, 1);

        // Act: allocate in Conversation pipeline for same request
        let pid2 = coord.memory_manager.allocate_page_in_pipeline(
            crate::scheduler::types::KvPipeline::Conversation,
            42,
            l1,
        );
        assert!(pid2.is_ok());
        let usage2 = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage2.used, 2);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_release_working_pipeline() {
        // Arrange: allocate pages in both Working and Conversation pipelines
        let mut coord = make_coordinator();
        let l1 = crate::scheduler::memory_manager::Tier::L1;

        let _wp = coord.memory_manager.allocate_page_in_pipeline(
            crate::scheduler::types::KvPipeline::Working,
            10,
            l1,
        ).unwrap();
        let _cp = coord.memory_manager.allocate_page_in_pipeline(
            crate::scheduler::types::KvPipeline::Conversation,
            10,
            l1,
        ).unwrap();
        assert_eq!(coord.memory_manager.tier_usage(l1).used, 2);

        // Act: release Working pipeline pages for request 10
        coord.memory_manager.release_working_pipeline(10);

        // Assert: Working page freed, Conversation page retained
        let usage = coord.memory_manager.tier_usage(l1);
        assert_eq!(usage.used, 1);
    }

    #[test]
    fn dispatch_coordinator_compact_op_kind_eligibility() {
        // Arrange: test CompactOpCategory compact eligibility
        use crate::scheduler::compact::CompactOpCategory;

        // Assert: only GEMM is compact-eligible
        assert!(CompactOpCategory::Gemm.is_compact_eligible());
        assert!(!CompactOpCategory::Attention.is_compact_eligible());
        assert!(!CompactOpCategory::Norm.is_compact_eligible());
        assert!(!CompactOpCategory::Elementwise.is_compact_eligible());

        // Assert: only GEMM is compute-bound
        assert!(CompactOpCategory::Gemm.is_compute_bound());
        assert!(!CompactOpCategory::Attention.is_compute_bound());
    }

    #[test]
    fn dispatch_coordinator_compact_evaluate_empty_batch() {
        // Arrange: empty BatchManifest
        use crate::scheduler::compact::{evaluate_compact, CompactConfig, CompactReason, CompactOpCategory};
        let manifest = crate::scheduler::chunked_prefill::BatchManifest {
            slots: vec![],
            total_tokens: 0,
            decode_tokens: 0,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0,
        };

        // Act
        let decision = evaluate_compact(&manifest, CompactOpCategory::Gemm, &CompactConfig::default());

        // Assert: empty batch returns EmptyBatch reason
        assert!(!decision.should_compact);
        assert_eq!(decision.active_count, 0);
        assert_eq!(decision.total_count, 0);
        assert!(matches!(decision.reason, CompactReason::EmptyBatch));
    }

    #[test]
    fn dispatch_coordinator_compact_evaluate_attention_not_eligible() {
        // Arrange: high-waste manifest but Attention op (not eligible)
        use crate::scheduler::compact::{evaluate_compact, CompactConfig, CompactReason, CompactOpCategory};
        let manifest = crate::scheduler::chunked_prefill::BatchManifest {
            slots: vec![
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 1,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 0,
                    token_end: 0, // inactive
                    compact_target: -1,
                },
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 2,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 0,
                    token_end: 4,
                    compact_target: 0,
                },
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 3,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 0,
                    token_end: 4,
                    compact_target: 1,
                },
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 4,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 0,
                    token_end: 4,
                    compact_target: 2,
                },
                crate::scheduler::chunked_prefill::BatchSlot {
                    request_id: 5,
                    slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                    token_start: 0,
                    token_end: 4,
                    compact_target: 3,
                },
            ],
            total_tokens: 16,
            decode_tokens: 16,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.8,
        };

        // Act: evaluate with Attention (memory-bound, not eligible)
        let decision = evaluate_compact(&manifest, CompactOpCategory::Attention, &CompactConfig::default());

        // Assert: should not compact because Attention is not eligible
        assert!(!decision.should_compact);
        assert!(matches!(decision.reason, CompactReason::CostExceedsBenefit { .. } | CompactReason::BelowThreshold { .. } | CompactReason::TooFewActive { .. }));
    }

    #[test]
    fn dispatch_coordinator_compact_evaluate_gemm_triggered() {
        // Arrange: high-waste GEMM manifest with sufficient active elements
        use crate::scheduler::compact::{evaluate_compact, CompactConfig, CompactReason, CompactOpCategory};
        let slots: Vec<_> = (0..10)
            .map(|i| crate::scheduler::chunked_prefill::BatchSlot {
                request_id: i,
                slot_type: crate::scheduler::chunked_prefill::SlotType::Decode,
                token_start: 0,
                token_end: if i < 5 { 4 } else { 0 }, // 5 active, 5 inactive
                compact_target: if i < 5 { i as i32 } else { -1 },
            })
            .collect();
        let manifest = crate::scheduler::chunked_prefill::BatchManifest {
            slots,
            total_tokens: 20,
            decode_tokens: 20,
            prefill_tokens: 0,
            compact_required: true,
            waste_ratio: 0.5, // 50% waste > threshold 25%
        };

        // Act: evaluate with GEMM (compute-bound, eligible)
        let decision = evaluate_compact(&manifest, CompactOpCategory::Gemm, &CompactConfig::default());

        // Assert: compact triggered
        assert!(decision.should_compact);
        assert_eq!(decision.active_count, 5);
        assert_eq!(decision.total_count, 10);
        assert!(matches!(decision.reason, CompactReason::Triggered { .. }));
    }

    #[test]
    fn dispatch_coordinator_compact_config_default_values() {
        // Arrange: verify CompactConfig default values match SPEC §10.6.3
        use crate::scheduler::compact::CompactConfig;
        let config = CompactConfig::default();

        // Assert: defaults match SPEC
        assert!((config.waste_threshold - 0.25).abs() < f32::EPSILON);
        assert_eq!(config.min_active_count, 4);
        assert!((config.cycles_per_element - 2.0).abs() < f32::EPSILON);
        assert!((config.flops_to_mem_ratio - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn dispatch_coordinator_page_metadata_default_values() {
        // Arrange: verify PageMetadata default values
        use crate::scheduler::types::PageMetadata;
        let meta = PageMetadata::default();

        // Assert: defaults are sensible
        assert_eq!(meta.page_id, 0);
        assert!(meta.sequence_id.is_none());
        assert_eq!(meta.recency, 0);
        assert_eq!(meta.access_count, 0);
        assert!(!meta.is_lir);
        assert_eq!(meta.state, crate::scheduler::types::PageState::Standby);
        assert!(meta.warm_until.is_none());
        assert!(meta.swap_in_time.is_none());
    }

    #[test]
    fn dispatch_coordinator_unified_virtual_page_kv_construction() {
        // Arrange: construct a KV context page
        use crate::scheduler::types::{UnifiedVirtualPage, PagePayloadKind, MemoryResidency};
        let page = UnifiedVirtualPage::kv(
            42,
            10,
            crate::scheduler::types::KvPipeline::Conversation,
            3,
            gllm_kernels::types::DType::BF16,
        );

        // Assert: KV page fields
        assert_eq!(page.page_id, 42);
        assert_eq!(page.payload_kind, PagePayloadKind::KvContext);
        assert_eq!(page.residency, MemoryResidency::DeviceLocal);
        assert_eq!(page.owner, Some(10));
        assert_eq!(page.pipeline, Some(crate::scheduler::types::KvPipeline::Conversation));
        assert_eq!(page.logical_index, 3);
        assert!(page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.expert_id.is_none());
        assert!(page.layer_idx.is_none());
    }

    #[test]
    fn dispatch_coordinator_unified_virtual_page_expert_construction() {
        // Arrange: construct a MoE expert weight page
        use crate::scheduler::types::{UnifiedVirtualPage, PagePayloadKind};
        let page = UnifiedVirtualPage::expert(7, 3, 12, gllm_kernels::types::DType::F32);

        // Assert: expert page fields
        assert_eq!(page.page_id, 7);
        assert_eq!(page.payload_kind, PagePayloadKind::ExpertWeight);
        assert!(page.is_evictable());
        assert_eq!(page.expert_id, Some(3));
        assert_eq!(page.layer_idx, Some(12));
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
    }

    #[test]
    fn dispatch_coordinator_unified_virtual_page_system_prompt_not_evictable() {
        // Arrange: construct a system prompt page
        use crate::scheduler::types::{UnifiedVirtualPage, PagePayloadKind, MemoryResidency};
        let page = UnifiedVirtualPage::system_prompt(99, gllm_kernels::types::DType::F16);

        // Assert: system prompt page is not evictable
        assert_eq!(page.page_id, 99);
        assert_eq!(page.payload_kind, PagePayloadKind::PromptSystem);
        assert!(!page.is_evictable());
        assert!(page.is_on_device());
        assert!(page.owner.is_none());

        // Assert: RAG page is evictable and on host
        let rag = UnifiedVirtualPage::rag(100, 5, gllm_kernels::types::DType::F32);
        assert_eq!(rag.payload_kind, PagePayloadKind::KnowledgeRAG);
        assert!(rag.is_evictable());
        assert!(!rag.is_on_device());
        assert_eq!(rag.residency, MemoryResidency::HostLocal);
        assert_eq!(rag.owner, Some(5));
    }

    // ── Wave 13 Additional Tests ──────────────────────────────────────

    #[test]
    fn dispatch_coordinator_claim_session_prefix_returns_virtual_pages() {
        // Arrange: register a session and finalize tokens, then claim a zero-length prefix
        use crate::scheduler::memory_manager::{GlobalMemoryManager, MemoryManagerError, SessionId};
        let mut mm = GlobalMemoryManager::new_with_capacities(64, 0, 0);
        let session_id: SessionId = 100;
        mm.register_session(session_id);
        mm.finalize_session_tokens(session_id, 5);

        // Act: claim a zero-length prefix (always valid when session exists)
        let result = mm.claim_session_prefix(session_id, 42, 0);

        // Assert: returns empty Vec (success)
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);

        // Assert: claiming more than 0 pages fails because pages.len() == 0
        let overflow = mm.claim_session_prefix(session_id, 42, 1);
        assert!(overflow.is_err());
        assert!(matches!(
            overflow.unwrap_err(),
            MemoryManagerError::SessionPagesInsufficient { session_id: sid, prefix_tokens: 1, available_pages: 0 } if sid == 100
        ));
    }

    #[test]
    fn dispatch_coordinator_claim_session_prefix_fails_on_unknown_session() {
        // Arrange: no session registered
        use crate::scheduler::memory_manager::{GlobalMemoryManager, MemoryManagerError, SessionId};
        let mut mm = GlobalMemoryManager::new_with_capacities(16, 0, 0);

        // Act: try to claim from nonexistent session
        let result = mm.claim_session_prefix(999, 1, 1);

        // Assert: returns UnknownSession error
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, MemoryManagerError::UnknownSession { session_id } if session_id == 999)
        );
    }

    #[test]
    fn dispatch_coordinator_prepare_next_turn_releases_working_pages() {
        // Arrange: allocate pages in Working pipeline, and Conversation pipeline
        use crate::scheduler::memory_manager::Tier;
        use crate::scheduler::types::KvPipeline;
        let mut coord = make_coordinator();
        let _working_pid = coord
            .memory_manager
            .allocate_page_in_pipeline(KvPipeline::Working, 10, Tier::L1)
            .unwrap();
        let _conv_pid = coord
            .memory_manager
            .allocate_page_in_pipeline(KvPipeline::Conversation, 10, Tier::L1)
            .unwrap();
        let usage_before = coord.memory_manager.tier_usage(Tier::L1);
        assert_eq!(usage_before.used, 2);

        // Act: prepare_next_turn should release only Working pages
        coord.memory_manager.prepare_next_turn(10);

        // Assert: Working page freed, Conversation page kept → 1 page remains
        let usage_after = coord.memory_manager.tier_usage(Tier::L1);
        assert_eq!(usage_after.used, 1);
    }

    #[test]
    fn dispatch_coordinator_track_in_pipeline_registers_existing_page() {
        // Arrange: allocate a page normally, then register it in a pipeline
        use crate::scheduler::memory_manager::Tier;
        use crate::scheduler::types::KvPipeline;
        let mut coord = make_coordinator();
        let pid = coord.memory_manager.allocate_page(Tier::L1).unwrap();

        // Act: track the already-allocated page into Working pipeline
        coord
            .memory_manager
            .track_in_pipeline(KvPipeline::Working, 55, pid);

        // Assert: page is still allocated (usage = 1), and releasing Working pipeline frees it (usage = 0)
        assert_eq!(coord.memory_manager.tier_usage(Tier::L1).used, 1);
        coord.memory_manager.release_working_pipeline(55);
        assert_eq!(coord.memory_manager.tier_usage(Tier::L1).used, 0);
    }

    #[test]
    fn dispatch_coordinator_request_state_bus_port_attach_and_check() {
        // Arrange: create a RequestState and attach multiple bus ports
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        use crate::routing::BusPortTag;
        let req = RequestState::new(1, RequestPhase::Decode, 50, 0)
            .with_bus_port(BusPortTag::RagInjection)
            .with_bus_port(BusPortTag::Guardrail);

        // Assert: attached ports are present, non-attached are absent
        assert!(req.has_bus_port(BusPortTag::RagInjection));
        assert!(req.has_bus_port(BusPortTag::Guardrail));
        assert!(!req.has_bus_port(BusPortTag::EarlyExit));
        assert!(!req.has_bus_port(BusPortTag::IntentRecall));
        assert!(!req.has_bus_port(BusPortTag::ShadowKv));
    }

    #[test]
    fn dispatch_coordinator_request_state_bus_port_custom_tag() {
        // Arrange: attach a Custom bus port tag
        use crate::scheduler::request_state::{RequestPhase, RequestState};
        use crate::routing::BusPortTag;
        let req = RequestState::new(2, RequestPhase::Prefill, 100, 0)
            .with_bus_port(BusPortTag::Custom(5));

        // Assert: Custom(5) is present, Custom(6) is not
        assert!(req.has_bus_port(BusPortTag::Custom(5)));
        assert!(!req.has_bus_port(BusPortTag::Custom(6)));
    }

    #[test]
    fn dispatch_coordinator_request_state_table_add_and_len() {
        // Arrange: create a CPU-backed RequestStateTable
        use crate::backend::detection::BackendType;
        use crate::scheduler::request_state::{RequestPhase, RequestState, RequestStateTable};
        let mut table = RequestStateTable::new(BackendType::Cpu);
        assert!(table.states.is_empty());

        // Act: add two request states
        table.add(RequestState::new(10, RequestPhase::Prefill, 200, 0));
        table.add(RequestState::new(20, RequestPhase::Decode, 50, 100));

        // Assert: states are stored in order
        assert_eq!(table.states.len(), 2);
        assert_eq!(table.states[0].request_id, 10);
        assert_eq!(table.states[1].request_id, 20);
        assert_eq!(table.states[0].phase, RequestPhase::Prefill);
        assert_eq!(table.states[1].phase, RequestPhase::Decode);
    }

    #[test]
    fn dispatch_coordinator_prefill_plan_pipelined_chunk_schedule() {
        // Arrange: use a memory manager with limited L1 capacity and a large prompt
        // that exceeds L1, forcing a Pipelined plan with a non-trivial chunk_schedule
        use crate::scheduler::memory_manager::PrefillPlan;
        let mut coord = make_coordinator();
        // L1 capacity = 32 pages, all free initially. Prompt needs > 32 pages.
        // page_size = 16 tokens, chunk_size = 256 tokens, so 1024 tokens = 64 pages > 32 L1.
        let plan = coord.memory_manager.plan_prefill(1024, 256, 16);

        // Assert: plan is Pipelined with a chunk_schedule
        match plan {
            PrefillPlan::Pipelined {
                l1_pages,
                l2_prefetch,
                chunk_schedule,
            } => {
                // 1024 tokens / 16 tokens_per_page = 64 total pages
                // L1 has 32 capacity, 0 used, so l1_pages <= 32
                assert!(l1_pages <= 32);
                // chunk_schedule: 1024 tokens / 256 per chunk = 4 chunks
                // Each chunk = 256/16 = 16 pages
                assert_eq!(chunk_schedule.len(), 4);
                assert!(chunk_schedule.iter().all(|&c| c == 16));
                // l2_prefetch: L2 capacity is 0, so prefetch is 0
                assert_eq!(l2_prefetch, 0);
            }
            PrefillPlan::FullyResident { .. } => {
                panic!("expected Pipelined plan, got FullyResident");
            }
        }
    }

    #[test]
    fn dispatch_coordinator_page_location_direct_construction() {
        // Arrange: construct a PageLocation directly
        use crate::scheduler::memory_manager::{PageLocation, Tier};
        let loc = PageLocation {
            physical_id: 42,
            tier: Tier::L1,
        };

        // Assert: fields are readable and match construction
        assert_eq!(loc.physical_id, 42);
        assert_eq!(loc.tier, Tier::L1);
    }

    #[test]
    fn dispatch_coordinator_scheduled_batch_with_nonzero_draft_steps() {
        // Arrange: construct a ScheduledBatch with MTP-style draft steps
        use crate::scheduler::batcher::ScheduledBatch;
        let batch = ScheduledBatch {
            requests: vec![100, 200],
            seq_offsets: vec![0, 64],
            draft_steps: vec![3, 5],
        };

        // Assert: draft_steps can carry non-zero values for speculative decoding
        assert_eq!(batch.draft_steps, vec![3, 5]);
        assert_ne!(batch.draft_steps[0], 0);
        assert_ne!(batch.draft_steps[1], 0);
    }

    #[test]
    fn dispatch_coordinator_compact_decision_below_threshold_reason() {
        // Arrange: build a manifest where waste is below threshold (all active)
        use crate::scheduler::chunked_prefill::{BatchManifest, BatchSlot, SlotType};
        use crate::scheduler::compact::{
            evaluate_compact, CompactConfig, CompactReason, CompactOpCategory,
        };
        let manifest = BatchManifest {
            slots: vec![
                BatchSlot { request_id: 1, slot_type: SlotType::Decode, token_start: 0, token_end: 4, compact_target: 0 },
                BatchSlot { request_id: 2, slot_type: SlotType::Decode, token_start: 0, token_end: 4, compact_target: 1 },
                BatchSlot { request_id: 3, slot_type: SlotType::Decode, token_start: 0, token_end: 4, compact_target: 2 },
                BatchSlot { request_id: 4, slot_type: SlotType::Decode, token_start: 0, token_end: 4, compact_target: 3 },
            ],
            total_tokens: 16,
            decode_tokens: 16,
            prefill_tokens: 0,
            compact_required: false,
            waste_ratio: 0.0, // 0% waste → below threshold
        };

        // Act
        let decision = evaluate_compact(&manifest, CompactOpCategory::Gemm, &CompactConfig::default());

        // Assert: not triggered, reason is BelowThreshold (waste = 0% < 25%)
        assert!(!decision.should_compact);
        assert!(matches!(
            decision.reason,
            CompactReason::BelowThreshold {
                waste_ratio,
                threshold,
            } if waste_ratio < threshold
        ));
    }

    #[test]
    fn dispatch_coordinator_request_kind_variants_coverage() {
        // Arrange: verify all RequestKind variants exist and are distinct
        use crate::scheduler::types::RequestKind;
        let chat = RequestKind::Chat;
        let embedding = RequestKind::Embedding;
        let rerank = RequestKind::Rerank;

        // Assert: variants are distinct and comparable
        assert_ne!(chat, embedding);
        assert_ne!(embedding, rerank);
        assert_ne!(chat, rerank);
    }

    #[test]
    fn dispatch_coordinator_memory_can_allocate_before_and_after_exhaustion() {
        // Arrange: create a coordinator with L1 capacity (32 pages from make_coordinator)
        use crate::scheduler::memory_manager::Tier;
        let mut coord = make_coordinator();

        // Assert: initially has available capacity
        let usage_start = coord.memory_manager.tier_usage(Tier::L1);
        assert_eq!(usage_start.available(), 32);

        // Act: exhaust L1 by allocating all pages
        let mut allocated = Vec::new();
        while let Ok(pid) = coord.memory_manager.allocate_page(Tier::L1) {
            allocated.push(pid);
        }

        // Assert: no available capacity after exhaustion
        let usage_full = coord.memory_manager.tier_usage(Tier::L1);
        assert_eq!(usage_full.available(), 0);
        assert_eq!(allocated.len(), 32);
        // Next allocation should fail
        assert!(coord.memory_manager.allocate_page(Tier::L1).is_err());

        // Act: free one page
        let freed = allocated.pop().unwrap();
        let free_result = coord.memory_manager.free_page(Tier::L1, freed);

        // Assert: available capacity restored
        assert!(free_result.is_ok());
        let usage_after = coord.memory_manager.tier_usage(Tier::L1);
        assert_eq!(usage_after.available(), 1);
        // Can allocate again
        assert!(coord.memory_manager.allocate_page(Tier::L1).is_ok());
    }

    // ── 13 new tests (wave-12x40) ──────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_interleaved_batch_construction_and_totals() {
        // Arrange: construct an InterleavedBatch with decode and prefill slots
        use crate::scheduler::batcher::{InterleavedBatch, InterleavedSlot, ScheduledBatch};
        let inner = ScheduledBatch {
            requests: vec![10, 20, 30],
            seq_offsets: vec![0, 64, 128],
            draft_steps: vec![0, 0, 0],
        };
        let batch = InterleavedBatch {
            inner,
            decode_slots: vec![
                InterleavedSlot { request_id: 10, batch_index: 0, token_count: 1, draft_steps: 0 },
                InterleavedSlot { request_id: 20, batch_index: 1, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![
                InterleavedSlot { request_id: 30, batch_index: 2, token_count: 512, draft_steps: 0 },
            ],
        };

        // Act & Assert: token counts are correct
        assert_eq!(batch.decode_tokens(), 2);
        assert_eq!(batch.prefill_tokens(), 512);
        assert_eq!(batch.total_tokens(), 514);
        assert!(batch.is_interleaved());
        assert_eq!(batch.request_ids(), &[10, 20, 30]);
    }

    #[test]
    fn dispatch_coordinator_interleaved_batch_decode_only_not_interleaved() {
        // Arrange: InterleavedBatch with only decode slots (no prefill)
        use crate::scheduler::batcher::{InterleavedBatch, InterleavedSlot, ScheduledBatch};
        let batch = InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![1],
                seq_offsets: vec![0],
                draft_steps: vec![0],
            },
            decode_slots: vec![
                InterleavedSlot { request_id: 1, batch_index: 0, token_count: 1, draft_steps: 0 },
            ],
            prefill_slots: vec![],
        };

        // Act & Assert: not interleaved because no prefill slots
        assert!(!batch.is_interleaved());
        assert_eq!(batch.decode_tokens(), 1);
        assert_eq!(batch.prefill_tokens(), 0);
        assert_eq!(batch.total_tokens(), 1);
    }

    #[test]
    fn dispatch_coordinator_batch_order_policy_variants_and_default() {
        // Arrange: verify all BatchOrderPolicy variants and Default
        use crate::scheduler::types::BatchOrderPolicy;
        let variants = [
            BatchOrderPolicy::StrictRequestIdOrder,
            BatchOrderPolicy::FifoOrder,
        ];

        // Assert: all variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }

        // Assert: Default is StrictRequestIdOrder
        let default = BatchOrderPolicy::default();
        assert_eq!(default, BatchOrderPolicy::StrictRequestIdOrder);
    }

    #[test]
    fn dispatch_coordinator_weight_tier_all_variants_distinct() {
        // Arrange: verify all WeightTier variants
        use crate::scheduler::types::WeightTier;
        let variants = [WeightTier::Hot, WeightTier::Warm, WeightTier::Cold];

        // Assert: all three variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    #[test]
    fn dispatch_coordinator_pipelined_virtual_page_id_construction() {
        // Arrange: construct PipelinedVirtualPageId with specific values
        use crate::scheduler::types::{KvPipeline, PipelinedVirtualPageId};
        let pvp = PipelinedVirtualPageId {
            pipeline: KvPipeline::Working,
            sequence_id: 42,
            logical_index: 7,
        };

        // Assert: fields match construction
        assert_eq!(pvp.pipeline, KvPipeline::Working);
        assert_eq!(pvp.sequence_id, 42);
        assert_eq!(pvp.logical_index, 7);

        // Assert: Conversation variant is distinct from Working
        let pvp_conv = PipelinedVirtualPageId {
            pipeline: KvPipeline::Conversation,
            sequence_id: 42,
            logical_index: 7,
        };
        assert_ne!(pvp.pipeline, pvp_conv.pipeline);
    }

    #[test]
    fn dispatch_coordinator_memory_residency_all_variants_distinct() {
        // Arrange: verify all MemoryResidency variants
        use crate::scheduler::types::MemoryResidency;
        let variants = [
            MemoryResidency::DeviceLocal,
            MemoryResidency::HostLocal,
            MemoryResidency::DiskSwap,
        ];

        // Assert: all three variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    #[test]
    fn dispatch_coordinator_eviction_priority_construction_fields() {
        // Arrange: construct an EvictionPriority with custom values
        use crate::scheduler::types::{EvictionPriority, PagePayloadKind};
        let priority = EvictionPriority {
            score: -100,
            payload_kind: PagePayloadKind::KvContext,
            is_pinned: false,
            access_count: 50,
            recency: 200,
            layer_idx: Some(12),
            expert_id: Some(3),
        };

        // Assert: all fields readable and match construction
        assert_eq!(priority.score, -100);
        assert_eq!(priority.payload_kind, PagePayloadKind::KvContext);
        assert!(!priority.is_pinned);
        assert_eq!(priority.access_count, 50);
        assert_eq!(priority.recency, 200);
        assert_eq!(priority.layer_idx, Some(12));
        assert_eq!(priority.expert_id, Some(3));
    }

    #[test]
    fn dispatch_coordinator_sequence_group_construction_and_pinned_field() {
        // Arrange: construct a SequenceGroup and verify is_pinned defaults to false
        use crate::scheduler::types::{GroupState, KvPipeline, SequenceGroup};
        let group = SequenceGroup {
            id: 99,
            pages: vec![1, 2, 3],
            state: GroupState::Running,
            access_count: 0,
            last_access: std::time::Instant::now(),
            is_pinned: false,
            context_len: 10,
            pipeline: KvPipeline::Conversation,
            payload_kind: None,
        };

        // Assert: fields match construction
        assert_eq!(group.id, 99);
        assert_eq!(group.pages, vec![1, 2, 3]);
        assert_eq!(group.state, GroupState::Running);
        assert!(!group.is_pinned);
        assert_eq!(group.context_len, 10);
        assert_eq!(group.pipeline, KvPipeline::Conversation);
        assert!(group.payload_kind.is_none());
    }

    #[test]
    fn dispatch_coordinator_request_data_max_new_tokens_boundary() {
        // Arrange: create a request with max_new_tokens = usize::MAX
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Decode,
            max_new_tokens: usize::MAX,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(1, req_data);

        // Act: push many tokens (but not usize::MAX)
        let req = coord.requests.get_mut(&1).unwrap();
        for i in 0..1000u32 {
            req.output_tokens.push(i);
        }

        // Assert: not finished (1000 << usize::MAX)
        assert!(!coord.requests[&1].finished);
        assert_eq!(coord.requests[&1].output_tokens.len(), 1000);
        assert_eq!(coord.requests[&1].max_new_tokens, usize::MAX);
    }

    #[test]
    fn dispatch_coordinator_dispatch_coordinator_struct_update_syntax() {
        // Arrange: create a coordinator then use struct update syntax
        let coord = make_coordinator();
        let updated = DispatchCoordinator {
            policy: PolicyVariant::Absolute,
            requests: HashMap::from([(999, RequestData {
                prompt_tokens: vec![1],
                output_tokens: vec![],
                sampling_config: Default::default(),
                phase: RequestPhase::Prefill,
                max_new_tokens: 10,
                finished: false,
                session_id: None,
                thinking_budget: None,
                fused_prefill_hidden: None,
            })]),
            ..coord
        };

        // Assert: updated field takes precedence, rest is shared
        assert_eq!(updated.requests.len(), 1);
        assert!(updated.requests.contains_key(&999));
        assert!(matches!(updated.policy, PolicyVariant::Absolute));
        assert!(!updated.batcher.has_pending_work());
    }

    #[test]
    fn dispatch_coordinator_page_payload_kind_all_variants_distinct() {
        // Arrange: verify all PagePayloadKind variants
        use crate::scheduler::types::PagePayloadKind;
        let variants = [
            PagePayloadKind::KvContext,
            PagePayloadKind::ExpertWeight,
            PagePayloadKind::PromptSystem,
            PagePayloadKind::KnowledgeRAG,
            PagePayloadKind::DenseLayerWeight,
        ];

        // Assert: all five variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    #[test]
    fn dispatch_coordinator_unified_virtual_page_dense_layer_not_evictable() {
        // Arrange: construct a DenseLayerWeight page
        use crate::scheduler::types::{UnifiedVirtualPage, PagePayloadKind};
        let page = UnifiedVirtualPage::dense_layer(42, 5, gllm_kernels::types::DType::F32);

        // Assert: dense layer pages are not evictable but are on device
        assert_eq!(page.page_id, 42);
        assert_eq!(page.payload_kind, PagePayloadKind::DenseLayerWeight);
        assert!(!page.is_evictable());
        assert!(page.is_on_device());
        assert_eq!(page.logical_index, 5);
        assert!(page.owner.is_none());
        assert!(page.pipeline.is_none());
        assert_eq!(page.layer_idx, Some(5));
    }

    #[test]
    fn dispatch_coordinator_interleaved_slot_construction_fields() {
        // Arrange: construct an InterleavedSlot for speculative decoding
        use crate::scheduler::batcher::InterleavedSlot;
        let slot = InterleavedSlot {
            request_id: 42,
            batch_index: 3,
            token_count: 512,
            draft_steps: 4,
        };

        // Assert: all fields match construction
        assert_eq!(slot.request_id, 42);
        assert_eq!(slot.batch_index, 3);
        assert_eq!(slot.token_count, 512);
        assert_eq!(slot.draft_steps, 4);
    }

    // ── Wave-33 additional tests ────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_compression_codec_round_trip() {
        // Arrange: all 5 CompressionCodec variants
        use crate::kv_cache::CompressionCodec;
        let variants = [
            CompressionCodec::None,
            CompressionCodec::Lz4,
            CompressionCodec::BitPackRle,
            CompressionCodec::NvcompAns,
            CompressionCodec::ZstdDict,
        ];

        // Act & Assert: each variant round-trips through u8
        for v in &variants {
            let byte = v.as_u8();
            let recovered = CompressionCodec::from_u8(byte);
            assert_eq!(recovered, Some(*v), "round-trip failed for {:?}", v);
        }

        // Assert: invalid byte returns None
        assert_eq!(CompressionCodec::from_u8(255), None);
    }

    #[test]
    fn dispatch_coordinator_storage_tier_ordering_hbm_dominates() {
        // Arrange: three storage tiers
        use crate::kv_cache::StorageTier;
        let hbm = StorageTier::GpuHbm;
        let dram = StorageTier::CpuDram;
        let nvme = StorageTier::Nvme;

        // Act & Assert: HBM > DRAM > NVMe (higher priority)
        assert!(hbm > dram, "GpuHbm should be higher priority than CpuDram");
        assert!(dram > nvme, "CpuDram should be higher priority than Nvme");
        assert!(hbm > nvme, "GpuHbm should be higher priority than Nvme");

        // Assert: round-trip through u8
        assert_eq!(StorageTier::from_u8(0), Some(StorageTier::GpuHbm));
        assert_eq!(StorageTier::from_u8(1), Some(StorageTier::CpuDram));
        assert_eq!(StorageTier::from_u8(2), Some(StorageTier::Nvme));
        assert_eq!(StorageTier::from_u8(3), None);
    }

    #[test]
    fn dispatch_coordinator_kv_page_header_default_is_inactive() {
        // Arrange: construct default KvPageHeader
        use crate::kv_cache::KvPageHeader;
        let header = KvPageHeader::default();

        // Assert: no refs means inactive
        assert!(!header.is_active(), "default header should have ref_count=0");
        assert!(!header.has_sink_token(), "default header should not have sink token");
        assert!(!header.needs_requantize(), "default header should not need requantize");
        assert!(!header.is_position_agnostic(), "default header should not be position-agnostic");
        assert!(header.is_low_entropy(), "default header should have zero entropy_avg");
        assert!(!header.is_high_dead_ratio(), "default header dead_ratio=0 < 128");
    }

    #[test]
    fn dispatch_coordinator_kv_page_header_new_sets_page_id() {
        // Arrange: create header with specific page_id
        use crate::kv_cache::KvPageHeader;
        let header = KvPageHeader::new(99);

        // Assert: page_id set correctly, rest defaults
        assert_eq!(header.page_id, 99);
        assert!(!header.is_active());
        assert_eq!(header.head_entropy_spread(), 0, "max and min both 0, spread=0");
    }

    #[test]
    fn dispatch_coordinator_kv_page_header_position_agnostic_toggle() {
        // Arrange: create a default header
        use crate::kv_cache::KvPageHeader;
        let mut header = KvPageHeader::default();
        assert!(!header.is_position_agnostic());

        // Act: set position-agnostic to true
        header.set_position_agnostic(true);

        // Assert: now position-agnostic
        assert!(header.is_position_agnostic());

        // Act: set back to false
        header.set_position_agnostic(false);

        // Assert: no longer position-agnostic
        assert!(!header.is_position_agnostic());
    }

    #[test]
    fn dispatch_coordinator_kv_cache_slot_flip_round_trip() {
        // Arrange: both slot variants
        use crate::kv_cache::KvCacheSlot;
        let front = KvCacheSlot::Front;
        let back = KvCacheSlot::Back;

        // Act & Assert: flip is an involution (flip twice = identity)
        assert_eq!(front.flip(), KvCacheSlot::Back);
        assert_eq!(back.flip(), KvCacheSlot::Front);
        assert_eq!(front.flip().flip(), KvCacheSlot::Front);
        assert_eq!(back.flip().flip(), KvCacheSlot::Back);
    }

    #[test]
    fn dispatch_coordinator_fault_recovery_stats_default_and_recording() {
        // Arrange: default stats
        use std::time::Duration;
        use crate::scheduler::fault_recovery::FaultRecoveryStats;
        use crate::scheduler::memory_manager::Tier;
        let mut stats = FaultRecoveryStats::default();

        // Assert: all counters zero, avg returns 0.0
        assert_eq!(stats.successful_recoveries, 0);
        assert_eq!(stats.aborted_faults, 0);
        assert_eq!(stats.retried_faults, 0);
        assert_eq!(stats.avg_recovery_latency_us(), 0.0);

        // Act: record L2 recovery with 100us latency
        stats.record_recovery(Tier::L2, Duration::from_micros(100));

        // Assert: L2 count incremented
        assert_eq!(stats.successful_recoveries, 1);
        assert_eq!(stats.l2_to_l1_count, 1);
        assert_eq!(stats.l3_to_l1_count, 0);
        assert!((stats.avg_recovery_latency_us() - 100.0).abs() < f64::EPSILON);

        // Act: record L3 recovery with 500us latency
        stats.record_recovery(Tier::L3, Duration::from_micros(500));

        // Assert: L3 count and multi-hop both incremented
        assert_eq!(stats.successful_recoveries, 2);
        assert_eq!(stats.l3_to_l1_count, 1);
        assert_eq!(stats.multi_hop_count, 1);
        assert!((stats.avg_recovery_latency_us() - 300.0).abs() < f64::EPSILON);

        // Act: record abort and retry
        stats.record_abort();
        stats.record_retry();

        // Assert: counters updated
        assert_eq!(stats.aborted_faults, 1);
        assert_eq!(stats.retried_faults, 1);
    }

    #[test]
    fn dispatch_coordinator_weight_page_table_register_and_lookup() {
        // Arrange: create empty table
        use crate::scheduler::fault_recovery::WeightPageTable;
        use crate::scheduler::memory_manager::Tier;
        let mut table = WeightPageTable::new();
        assert_eq!(table.layer_count(), 0);
        assert_eq!(table.total_pages(), 0);

        // Act: register layer 0 with 3 physical pages
        table.register_layer(0, vec![10, 20, 30]);

        // Assert: layer and page counts updated
        assert_eq!(table.layer_count(), 1);
        assert_eq!(table.total_pages(), 3);

        // Assert: lookup succeeds
        assert_eq!(table.get_layer_pages(0), Some(&[10usize, 20, 30][..]));
        assert_eq!(table.layer_for_page(20), Some(0));
        assert_eq!(table.position_for_page(30), Some(2));
        assert_eq!(table.page_tier(10), Some(Tier::L1));

        // Assert: unregistered layer returns None
        assert!(table.get_layer_pages(1).is_none());
        assert_eq!(table.layer_for_page(99), None);
    }

    #[test]
    fn dispatch_coordinator_weight_page_table_update_layer_tier() {
        // Arrange: register layer 0 in L1, then tier-migrate to L3
        use crate::scheduler::fault_recovery::WeightPageTable;
        use crate::scheduler::memory_manager::Tier;
        let mut table = WeightPageTable::new();
        table.register_layer(5, vec![100, 200]);

        // Assert: initial tier is L1
        assert_eq!(table.page_tier(100), Some(Tier::L1));

        // Act: batch-migrate entire layer to L3
        table.update_layer_tier(5, Tier::L3);

        // Assert: both pages now report L3
        assert_eq!(table.page_tier(100), Some(Tier::L3));
        assert_eq!(table.page_tier(200), Some(Tier::L3));
    }

    #[test]
    fn dispatch_coordinator_eviction_tier_classify_variants() {
        // Arrange: import types for classification
        use crate::scheduler::eviction_worker::{EvictionWorker, EvictionTier};
        use crate::scheduler::types::PagePayloadKind;

        // Act & Assert: payload kind drives tier classification
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::ExpertWeight), 999),
            EvictionTier::ColdExpert,
        );
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::DenseLayerWeight), 999),
            EvictionTier::PinnedDense,
        );
        // Low score → StandbyKv
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 50),
            EvictionTier::StandbyKv,
        );
        // High score → Protected
        assert_eq!(
            EvictionWorker::classify_eviction_tier(Some(PagePayloadKind::KvContext), 500),
            EvictionTier::Protected,
        );
        // None payload with high score → Protected
        assert_eq!(
            EvictionWorker::classify_eviction_tier(None, 500),
            EvictionTier::Protected,
        );
    }

    #[test]
    fn dispatch_coordinator_three_tier_swap_stats_default_and_avg() {
        // Arrange: default stats
        use crate::scheduler::three_tier_swap::ThreeTierSwapStats;
        let stats = ThreeTierSwapStats::default();

        // Assert: all counters zero, avg methods return 0.0
        assert_eq!(stats.evictions_gpu_to_dram, 0);
        assert_eq!(stats.swap_ins_dram_to_gpu, 0);
        assert_eq!(stats.total_migrations(), 0);
        assert_eq!(stats.avg_eviction_latency_us(), 0.0);
        assert_eq!(stats.avg_swap_in_latency_us(), 0.0);
    }

    #[test]
    fn dispatch_coordinator_swap_in_worker_stats_avg_latency() {
        // Arrange: stats with two successful promotions
        use crate::scheduler::swap_in_worker::SwapInWorkerStats;
        let stats = SwapInWorkerStats {
            total_requests: 10,
            submitted: 8,
            skipped: 2,
            promoted_ok: 2,
            promoted_failed: 1,
            two_hop_promotions: 1,
            total_latency_us: 600,
            rounds: 3,
        };

        // Act
        let avg = stats.avg_latency_us();

        // Assert: 600 / 2 = 300.0
        assert!((avg - 300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn dispatch_coordinator_migration_actor_config_swap_file_path() {
        // Arrange: create config with custom session_id
        use crate::scheduler::migration_actor::MigrationActorConfig;
        let config = MigrationActorConfig {
            nvme_swap_dir: std::path::PathBuf::from("/tmp/gllm_swap"),
            queue_capacity: 128,
            session_id: "test-session".to_string(),
            page_size: 8192,
            max_swap_pages: 2048,
        };

        // Act
        let path = config.swap_file_path();

        // Assert: path joins dir + "<session_id>.swap"
        assert_eq!(path, std::path::PathBuf::from("/tmp/gllm_swap/test-session.swap"));
    }

    // ── 13 new tests (wave-12x41) ──────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_oom_halt_error_fatal_and_soft() {
        // Arrange: construct both error variants
        use crate::kv_cache::OomHaltError;

        let fatal = OomHaltError::fatal_halt("GPU OOM");
        let soft = OomHaltError::soft_halt("budget exceeded");

        // Assert: fatal is fatal, soft is not
        assert!(fatal.fatal);
        assert!(!soft.fatal);
        assert!(fatal.message.contains("GPU OOM"));
        assert!(soft.message.contains("budget exceeded"));

        // Assert: Display contains the message and fatal flag
        let fatal_str = format!("{}", fatal);
        assert!(fatal_str.contains("GPU OOM"), "fatal display: {}", fatal_str);
        assert!(fatal_str.contains("fatal=true"), "fatal display: {}", fatal_str);

        let soft_str = format!("{}", soft);
        assert!(soft_str.contains("budget exceeded"), "soft display: {}", soft_str);
        assert!(soft_str.contains("fatal=false"), "soft display: {}", soft_str);
    }

    #[test]
    fn dispatch_coordinator_precision_tier_all_variants_distinct() {
        // Arrange: verify all PrecisionTier variants
        use crate::kv_cache::PrecisionTier;
        let variants = [
            PrecisionTier::FP16,
            PrecisionTier::FP8,
            PrecisionTier::KIVI4,
            PrecisionTier::KIVI2,
            PrecisionTier::Sparse,
            PrecisionTier::Dictionary,
            PrecisionTier::Evicted,
        ];

        // Assert: all seven variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }

        // Assert: discriminant ordering matches SPEC (0..=6)
        assert_eq!(variants[0] as u8, 0);
        assert_eq!(variants[6] as u8, 6);
    }

    #[test]
    fn dispatch_coordinator_select_codec_returns_correct_codec_per_tier() {
        // Arrange: test all PrecisionTier paths through select_codec
        use crate::kv_cache::{select_codec, select_cold_codec, CompressionCodec, PrecisionTier};

        // Act & Assert: FP16 + gpu + nvcomp → NvcompAns
        assert_eq!(
            select_codec(PrecisionTier::FP16, true, true),
            CompressionCodec::NvcompAns
        );

        // Act & Assert: FP16 + gpu + no nvcomp → Lz4
        assert_eq!(
            select_codec(PrecisionTier::FP16, true, false),
            CompressionCodec::Lz4
        );

        // Act & Assert: FP16 + cpu → Lz4 (gpu_path=false, nvcomp irrelevant)
        assert_eq!(
            select_codec(PrecisionTier::FP16, false, true),
            CompressionCodec::Lz4
        );

        // Act & Assert: FP8 + gpu + nvcomp → NvcompAns
        assert_eq!(
            select_codec(PrecisionTier::FP8, true, true),
            CompressionCodec::NvcompAns
        );

        // Act & Assert: KIVI4 → BitPackRle
        assert_eq!(
            select_codec(PrecisionTier::KIVI4, false, false),
            CompressionCodec::BitPackRle
        );

        // Act & Assert: KIVI2 → BitPackRle
        assert_eq!(
            select_codec(PrecisionTier::KIVI2, true, true),
            CompressionCodec::BitPackRle
        );

        // Act & Assert: Sparse → None
        assert_eq!(
            select_codec(PrecisionTier::Sparse, false, false),
            CompressionCodec::None
        );

        // Act & Assert: Dictionary → None
        assert_eq!(
            select_codec(PrecisionTier::Dictionary, true, true),
            CompressionCodec::None
        );

        // Act & Assert: Evicted → None
        assert_eq!(
            select_codec(PrecisionTier::Evicted, false, false),
            CompressionCodec::None
        );

        // Act & Assert: select_cold_codec always returns ZstdDict
        for tier in [
            PrecisionTier::FP16,
            PrecisionTier::FP8,
            PrecisionTier::KIVI4,
            PrecisionTier::Evicted,
        ] {
            assert_eq!(select_cold_codec(tier), CompressionCodec::ZstdDict);
        }
    }

    #[test]
    fn dispatch_coordinator_kv_page_header_precision_tier_set_and_get() {
        // Arrange: default header starts at FP16
        use crate::kv_cache::{KvPageHeader, PrecisionTier};
        let mut header = KvPageHeader::default();
        assert_eq!(header.precision_tier(), PrecisionTier::FP16);

        // Act: set to FP8
        header.set_precision_tier(PrecisionTier::FP8);
        assert_eq!(header.precision_tier(), PrecisionTier::FP8);

        // Act: set to KIVI4
        header.set_precision_tier(PrecisionTier::KIVI4);
        assert_eq!(header.precision_tier(), PrecisionTier::KIVI4);

        // Act: set to Evicted
        header.set_precision_tier(PrecisionTier::Evicted);
        assert_eq!(header.precision_tier(), PrecisionTier::Evicted);
    }

    #[test]
    fn dispatch_coordinator_fault_action_variants_distinct() {
        // Arrange: construct all FaultAction variants
        use crate::scheduler::fault_recovery::FaultAction;
        let load = FaultAction::LoadFromTier {
            source_tier: crate::scheduler::memory_manager::Tier::L2,
            target_tier: crate::scheduler::memory_manager::Tier::L1,
        };
        let abort = FaultAction::Abort {
            reason: "page gone".to_string(),
        };
        let retry = FaultAction::Retry;

        // Assert: each variant is distinct via PartialEq
        assert_ne!(load, abort);
        assert_ne!(load, retry);
        assert_ne!(abort, retry);

        // Assert: self-equality
        assert_eq!(load, FaultAction::LoadFromTier {
            source_tier: crate::scheduler::memory_manager::Tier::L2,
            target_tier: crate::scheduler::memory_manager::Tier::L1,
        });
        assert_eq!(retry, FaultAction::Retry);
    }

    #[test]
    fn dispatch_coordinator_step_fault_plan_default_is_empty() {
        // Arrange: construct default StepFaultPlan
        use crate::scheduler::fault_recovery::StepFaultPlan;
        let plan = StepFaultPlan::default();

        // Assert: empty plan has no faults
        assert!(!plan.has_faults());
        assert_eq!(plan.total_faults(), 0);
        assert!(plan.pending_faults.is_empty());
        assert_eq!(plan.pages_in_l1, 0);
        assert_eq!(plan.l2_faults, 0);
        assert_eq!(plan.l3_faults, 0);
    }

    #[test]
    fn dispatch_coordinator_step_fault_plan_new_matches_default() {
        // Arrange: construct via new() and via default()
        use crate::scheduler::fault_recovery::StepFaultPlan;
        let from_new = StepFaultPlan::new();
        let from_default = StepFaultPlan::default();

        // Assert: both construction paths produce equivalent empty plans
        assert_eq!(from_new.pages_in_l1, from_default.pages_in_l1);
        assert_eq!(from_new.l2_faults, from_default.l2_faults);
        assert_eq!(from_new.l3_faults, from_default.l3_faults);
        assert_eq!(from_new.pending_faults.len(), from_default.pending_faults.len());
        assert!(!from_new.has_faults());
    }

    #[test]
    fn dispatch_coordinator_eviction_candidate_construction_fields() {
        // Arrange: construct an EvictionCandidate with specific values
        use crate::scheduler::eviction_worker::EvictionCandidate;
        use crate::kv_cache::{CompressionCodec, StorageTier};
        let candidate = EvictionCandidate {
            page_id: 42,
            score: -150,
            current_tier: StorageTier::GpuHbm,
            codec: CompressionCodec::Lz4,
            page_bytes: 4096,
            group_id: Some(99),
        };

        // Assert: all fields match construction
        assert_eq!(candidate.page_id, 42);
        assert_eq!(candidate.score, -150);
        assert_eq!(candidate.current_tier, StorageTier::GpuHbm);
        assert_eq!(candidate.codec, CompressionCodec::Lz4);
        assert_eq!(candidate.page_bytes, 4096);
        assert_eq!(candidate.group_id, Some(99));
    }

    #[test]
    fn dispatch_coordinator_tier_migration_reason_all_variants() {
        // Arrange: verify all TierMigrationReason variants are distinct
        use crate::scheduler::three_tier_swap::TierMigrationReason;
        let variants = [
            TierMigrationReason::EvictionPressure,
            TierMigrationReason::SequenceDemand,
            TierMigrationReason::Prefetch,
            TierMigrationReason::ColdCascade,
        ];

        // Assert: all four variants are pairwise distinct
        for i in 0..variants.len() {
            for j in 0..variants.len() {
                if i == j {
                    assert_eq!(variants[i], variants[j]);
                } else {
                    assert_ne!(variants[i], variants[j]);
                }
            }
        }
    }

    #[test]
    fn dispatch_coordinator_tier_migration_construction_fields() {
        // Arrange: construct a TierMigration
        use crate::scheduler::three_tier_swap::{TierMigration, TierMigrationReason};
        use crate::kv_cache::{CompressionCodec, StorageTier};
        let migration = TierMigration {
            page_id: 7,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            codec: CompressionCodec::BitPackRle,
            page_bytes: 8192,
            reason: TierMigrationReason::EvictionPressure,
        };

        // Assert: all fields match construction
        assert_eq!(migration.page_id, 7);
        assert_eq!(migration.from_tier, StorageTier::GpuHbm);
        assert_eq!(migration.to_tier, StorageTier::CpuDram);
        assert_eq!(migration.codec, CompressionCodec::BitPackRle);
        assert_eq!(migration.page_bytes, 8192);
        assert_eq!(migration.reason, TierMigrationReason::EvictionPressure);
    }

    #[test]
    fn dispatch_coordinator_swap_in_worker_config_default_values() {
        // Arrange: construct default SwapInWorkerConfig
        use crate::scheduler::swap_in_worker::SwapInWorkerConfig;
        let config = SwapInWorkerConfig::default();

        // Assert: all default values match SPEC constants
        assert_eq!(config.max_prefetch_per_round, 16);
        assert_eq!(config.tick_interval, std::time::Duration::from_millis(5));
        assert!((config.min_confidence - 0.1).abs() < f32::EPSILON);
        assert_eq!(config.max_in_flight, 64);
        assert_eq!(config.page_bytes, 4096);
    }

    #[test]
    fn dispatch_coordinator_page_addr_entry_construction_fields() {
        // Arrange: construct a PageAddrEntry for a GPU-resident page
        use crate::scheduler::migration_actor::PageAddrEntry;
        use crate::kv_cache::{CompressionCodec, StorageTier};
        let entry = PageAddrEntry {
            gpu_ptr: Some(0xDEADBEEF),
            host_buffer: None,
            current_tier: StorageTier::GpuHbm,
            original_bytes: 4096,
            codec: CompressionCodec::None,
        };

        // Assert: all fields match construction
        assert_eq!(entry.gpu_ptr, Some(0xDEADBEEF));
        assert!(entry.host_buffer.is_none());
        assert_eq!(entry.current_tier, StorageTier::GpuHbm);
        assert_eq!(entry.original_bytes, 4096);
        assert_eq!(entry.codec, CompressionCodec::None);
    }

    #[test]
    fn dispatch_coordinator_kv_page_header_active_and_entropy_behavior() {
        // Arrange: create a header and manipulate activity and entropy fields
        use crate::kv_cache::KvPageHeader;
        let mut header = KvPageHeader::new(10);

        // Assert: starts inactive
        assert!(!header.is_active());

        // Act: increment ref_count to make it active
        header.ref_count = 3;
        assert!(header.is_active());

        // Assert: low entropy by default (entropy_avg = 0)
        assert!(header.is_low_entropy());

        // Act: set non-zero entropy
        header.entropy_avg = 42;
        assert!(!header.is_low_entropy());

        // Act: set high dead_ratio
        header.dead_ratio = 200; // > 127
        assert!(header.is_high_dead_ratio());

        // Assert: low dead_ratio is not "high"
        header.dead_ratio = 50;
        assert!(!header.is_high_dead_ratio());

        // Act: set sink_mask
        header.sink_mask = 0b101;
        assert!(header.has_sink_token());

        // Act: clear sink_mask
        header.sink_mask = 0;
        assert!(!header.has_sink_token());

        // Act: set head_entropy spread
        header.head_entropy_max = 200;
        header.head_entropy_min = 50;
        assert_eq!(header.head_entropy_spread(), 150);

        // Assert: saturating behavior
        header.head_entropy_max = 10;
        header.head_entropy_min = 200;
        assert_eq!(header.head_entropy_spread(), 0); // saturating_sub
    }

    // ── 13 new tests (wave-12x42) ──────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_requests_lookup_missing_key_returns_none() {
        // Arrange: insert one request, then query multiple non-existent keys
        let mut coord = make_coordinator();
        let req_data = RequestData {
            prompt_tokens: vec![1],
            output_tokens: vec![],
            sampling_config: Default::default(),
            phase: RequestPhase::Prefill,
            max_new_tokens: 10,
            finished: false,
            session_id: None,
            thinking_budget: None,
            fused_prefill_hidden: None,
        };
        coord.requests.insert(1, req_data);

        // Act & Assert: non-existent keys return None
        assert!(coord.requests.get(&0).is_none());
        assert!(coord.requests.get(&2).is_none());
        assert!(coord.requests.get(&u64::MAX).is_none());
        assert!(!coord.requests.contains_key(&0));
        assert!(!coord.requests.contains_key(&u64::MAX));
        // Existing key still resolves
        assert!(coord.requests.get(&1).is_some());
    }

    #[test]
    fn dispatch_coordinator_system_state_extreme_entropy_and_sparsity() {
        // Arrange: coordinator with default policy
        let coord = make_coordinator();

        // Act: decide with maximum entropy and maximum sparsity simultaneously
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.1,
            kv_fragmentation: 0.0,
            current_running_len: 8,
            current_batch_size: 8,
            logits_entropy: f32::MAX,
            attention_sparsity: 1.0,
            ..Default::default()
        };
        let decision = coord.policy.decide(&state);

        // Assert: safe mode (low pressure) but batch severely capped
        assert!(decision.admit_new_prefill);
        assert_eq!(decision.force_swap_out_count, 0);
        // Both high entropy and high sparsity cap the batch below default 32
        assert!(decision.max_batch_size < 32);
        assert!(decision.max_batch_size > 0);
    }

    #[test]
    fn dispatch_coordinator_interleaved_batch_prefill_only_is_interleaved() {
        // Arrange: InterleavedBatch with only prefill slots (no decode)
        use crate::scheduler::batcher::{InterleavedBatch, InterleavedSlot, ScheduledBatch};
        let batch = InterleavedBatch {
            inner: ScheduledBatch {
                requests: vec![5, 6],
                seq_offsets: vec![0, 256],
                draft_steps: vec![0, 0],
            },
            decode_slots: vec![],
            prefill_slots: vec![
                InterleavedSlot { request_id: 5, batch_index: 0, token_count: 256, draft_steps: 0 },
                InterleavedSlot { request_id: 6, batch_index: 1, token_count: 128, draft_steps: 0 },
            ],
        };

        // Act & Assert: is_interleaved = false (requires both decode + prefill slots)
        assert!(!batch.is_interleaved());
        assert_eq!(batch.decode_tokens(), 0);
        assert_eq!(batch.prefill_tokens(), 384);
        assert_eq!(batch.total_tokens(), 384);
        assert_eq!(batch.request_ids(), &[5, 6]);
    }

    #[test]
    fn dispatch_coordinator_eviction_candidate_zero_score_and_large_bytes() {
        // Arrange: construct an EvictionCandidate with boundary values
        use crate::scheduler::eviction_worker::EvictionCandidate;
        use crate::kv_cache::{CompressionCodec, StorageTier};
        let candidate = EvictionCandidate {
            page_id: 0,
            score: 0,
            current_tier: StorageTier::CpuDram,
            codec: CompressionCodec::None,
            page_bytes: usize::MAX,
            group_id: None,
        };

        // Assert: zero score and max bytes stored correctly
        assert_eq!(candidate.page_id, 0);
        assert_eq!(candidate.score, 0);
        assert_eq!(candidate.current_tier, StorageTier::CpuDram);
        assert_eq!(candidate.codec, CompressionCodec::None);
        assert_eq!(candidate.page_bytes, usize::MAX);
        assert!(candidate.group_id.is_none());
    }

    #[test]
    fn dispatch_coordinator_batch_slot_negative_compact_target_inactive() {
        // Arrange: construct a BatchSlot with compact_target = -1 (inactive)
        use crate::scheduler::chunked_prefill::{BatchSlot, SlotType};
        let slot = BatchSlot {
            request_id: 42,
            slot_type: SlotType::Decode,
            token_start: 0,
            token_end: 0,
            compact_target: -1,
        };

        // Assert: negative compact_target indicates an inactive/invalid slot
        assert_eq!(slot.request_id, 42);
        assert!(slot.compact_target < 0);
        assert_eq!(slot.token_end - slot.token_start, 0);
    }

    #[test]
    fn dispatch_coordinator_memory_manager_allocate_all_tiers_exhaust_independently() {
        // Arrange: create coordinator with L1=4, L2=4, L3=4
        let mut coord = DispatchCoordinator {
            scheduler: PagedScheduler::new(12, 4, HGALConfig::default()),
            batcher: ContinuousBatcher::new()
                .with_chunked(ChunkedConfig::default()),
            chunked_prefill_scheduler: ChunkedPrefillScheduler::new(
                ChunkedPrefillConfig::default(),
            ),
            requests: HashMap::new(),
            memory_manager: GlobalMemoryManager::new_with_capacities(4, 4, 4),
            policy: PolicyVariant::default(),
        };
        let l1 = crate::scheduler::memory_manager::Tier::L1;
        let l2 = crate::scheduler::memory_manager::Tier::L2;
        let l3 = crate::scheduler::memory_manager::Tier::L3;

        // Act: exhaust L1
        for _ in 0..4 {
            assert!(coord.memory_manager.allocate_page(l1).is_ok());
        }
        assert!(coord.memory_manager.allocate_page(l1).is_err());

        // Assert: L2 and L3 still have capacity
        assert!(coord.memory_manager.allocate_page(l2).is_ok());
        assert!(coord.memory_manager.allocate_page(l3).is_ok());

        // Act: exhaust L2
        for _ in 0..3 {
            assert!(coord.memory_manager.allocate_page(l2).is_ok());
        }
        assert!(coord.memory_manager.allocate_page(l2).is_err());

        // Assert: L3 still has remaining
        for _ in 0..3 {
            assert!(coord.memory_manager.allocate_page(l3).is_ok());
        }
        assert!(coord.memory_manager.allocate_page(l3).is_err());

        // Assert: all tiers fully used
        assert_eq!(coord.memory_manager.tier_usage(l1).available(), 0);
        assert_eq!(coord.memory_manager.tier_usage(l2).available(), 0);
        assert_eq!(coord.memory_manager.tier_usage(l3).available(), 0);
    }

    #[test]
    fn dispatch_coordinator_policy_variant_default_equals_clone() {
        // Arrange: get default and clone from coordinator
        let coord = make_coordinator();
        let from_default = PolicyVariant::default();
        let from_clone = coord.policy.clone();

        // Act: both decide with identical state
        let state = crate::scheduler::jit_types::SystemState {
            memory_pressure: 0.5,
            kv_fragmentation: 0.3,
            current_running_len: 5,
            current_batch_size: 5,
            ..Default::default()
        };
        let decision_default = from_default.decide(&state);
        let decision_clone = from_clone.decide(&state);

        // Assert: identical decisions
        assert_eq!(decision_default, decision_clone);
    }

    #[test]
    fn dispatch_coordinator_scheduler_decision_debug_all_fields_present() {
        // Arrange: construct a SchedulerDecision with non-trivial values
        let decision = crate::scheduler::jit_types::SchedulerDecision {
            max_batch_size: 7,
            admit_new_prefill: false,
            force_swap_out_count: 2,
        };

        // Act: format via Debug
        let debug = format!("{:?}", decision);

        // Assert: all three fields appear in debug output
        assert!(debug.contains("7"), "max_batch_size value should appear: {}", debug);
        assert!(debug.contains("false"), "admit_new_prefill=false should appear: {}", debug);
        assert!(debug.contains("2"), "force_swap_out_count value should appear: {}", debug);
    }

    #[test]
    fn dispatch_coordinator_chunked_prefill_scheduler_disabled_config_construction() {
        // Arrange: build a ChunkedPrefillScheduler with disabled config
        let config = ChunkedPrefillConfig {
            enabled: false,
            ..ChunkedPrefillConfig::default()
        };
        let scheduler = ChunkedPrefillScheduler::new(config);

        // Act: query config back
        let retrieved = scheduler.config();

        // Assert: disabled flag preserved
        assert!(!retrieved.enabled);
        // Assert: other fields still at defaults
        assert_eq!(retrieved.chunk_size, 512);
        assert!(!scheduler.should_chunk(10000));
        assert!(!scheduler.should_chunk(0));
    }

    #[test]
    fn dispatch_coordinator_virtual_page_id_same_sequence_different_logical_index() {
        // Arrange: two virtual pages with same sequence_id but different logical indices
        use crate::scheduler::memory_manager::VirtualPageId;
        let vp_a = VirtualPageId::new(10, 0);
        let vp_b = VirtualPageId::new(10, 1);
        let vp_c = VirtualPageId::new(10, 0); // same as vp_a

        // Assert: same sequence_id, different logical_index => distinct
        assert_ne!(vp_a, vp_b);
        // Assert: same sequence_id and logical_index => equal
        assert_eq!(vp_a, vp_c);
        // Assert: different sequence_id, same logical_index => distinct
        let vp_d = VirtualPageId::new(11, 0);
        assert_ne!(vp_a, vp_d);
    }

    #[test]
    fn dispatch_coordinator_batch_prep_data_large_batch_construction() {
        // Arrange: construct BatchPrepData with 256 sequences
        let prep = crate::scheduler::batcher::BatchPrepData::new(256);

        // Assert: all vec fields have length 256
        assert_eq!(prep.prompt_lens.len(), 256);
        assert_eq!(prep.kv_lens.len(), 256);
        assert_eq!(prep.session_positions.len(), 256);
        assert_eq!(prep.active_flags.len(), 256);
        assert_eq!(prep.sampling_params_packed.len(), 256 * 4);

        // Assert: active_flags all 1 (all active by default)
        assert!(prep.active_flags.iter().all(|&f| f == 1));

        // Assert: all other numeric vecs are zeroed
        assert!(prep.prompt_lens.iter().all(|&v| v == 0));
        assert!(prep.kv_lens.iter().all(|&v| v == 0));
        assert!(prep.sampling_params_packed.iter().all(|&v| v == 0));
    }

    #[test]
    fn dispatch_coordinator_request_data_all_fields_populated_and_readable() {
        // Arrange: construct a RequestData with every field set to non-default values
        let mut coord = make_coordinator();
        let sampling = crate::engine::executor_types::SamplingConfig {
            temperature: 0.33,
            top_k: 77,
            top_p: 0.88,
        };
        let req_data = RequestData {
            prompt_tokens: vec![100, 200, 300],
            output_tokens: vec![400, 500],
            sampling_config: sampling,
            phase: RequestPhase::Decode,
            max_new_tokens: 999,
            finished: false,
            session_id: Some(12345),
            thinking_budget: Some(512),
            fused_prefill_hidden: Some(vec![0.25, 0.75]),
        };
        coord.requests.insert(777, req_data);

        // Act: read back all fields
        let r = &coord.requests[&777];

        // Assert: every field matches construction
        assert_eq!(r.prompt_tokens, vec![100, 200, 300]);
        assert_eq!(r.output_tokens, vec![400, 500]);
        assert_eq!(r.sampling_config.temperature, 0.33);
        assert_eq!(r.sampling_config.top_k, 77);
        assert_eq!(r.sampling_config.top_p, 0.88);
        assert_eq!(r.phase, RequestPhase::Decode);
        assert_eq!(r.phase, RequestPhase::Decode);
        assert_eq!(r.max_new_tokens, 999);
        assert!(!r.finished);
        assert_eq!(r.session_id, Some(12345));
        assert_eq!(r.thinking_budget, Some(512));
        assert_eq!(r.fused_prefill_hidden.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn dispatch_coordinator_tier_usage_repr_and_equality() {
        // Arrange: construct two identical TierUsage instances
        use crate::scheduler::memory_manager::TierUsage;
        let a = TierUsage { used: 10, capacity: 32 };
        let b = TierUsage { used: 10, capacity: 32 };

        // Assert: equality
        assert_eq!(a, b);

        // Assert: available computes correctly
        assert_eq!(a.available(), 22);

        // Assert: different used breaks equality
        let c = TierUsage { used: 11, capacity: 32 };
        assert_ne!(a, c);

        // Assert: different capacity breaks equality
        let d = TierUsage { used: 10, capacity: 64 };
        assert_ne!(a, d);

        // Assert: Debug output contains field names
        let debug = format!("{:?}", a);
        assert!(debug.contains("used"), "TierUsage Debug should contain 'used': {}", debug);
        assert!(debug.contains("capacity"), "TierUsage Debug should contain 'capacity': {}", debug);
    }

    // ── 10 new tests (wave-12x43) ──────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_weight_page_table_tier_distribution_multi_layer() {
        // Arrange: register 3 layers across different tiers
        use crate::scheduler::fault_recovery::WeightPageTable;
        use crate::scheduler::memory_manager::Tier;
        let mut table = WeightPageTable::new();
        table.register_layer(0, vec![0, 1, 2]); // L1 by default
        table.register_layer(1, vec![3, 4]);     // L1 by default
        // Migrate layer 1 to L2
        table.update_layer_tier(1, Tier::L2);

        // Act
        let (l1, l2, l3) = table.tier_distribution();

        // Assert: 3 pages in L1 (layer 0), 2 pages in L2 (layer 1), 0 in L3
        assert_eq!(l1, 3);
        assert_eq!(l2, 2);
        assert_eq!(l3, 0);
        assert_eq!(table.total_pages(), 5);
        assert_eq!(table.layer_count(), 2);
    }

    #[test]
    fn dispatch_coordinator_weight_page_table_layer_needs_recovery() {
        // Arrange: register layer 0 in L1, layer 1 migrated to L2
        use crate::scheduler::fault_recovery::WeightPageTable;
        use crate::scheduler::memory_manager::Tier;
        let mut table = WeightPageTable::new();
        table.register_layer(0, vec![10, 11]);
        table.register_layer(1, vec![20, 21]);
        table.update_layer_tier(1, Tier::L2);

        // Act & Assert: layer 0 is fully in L1 → no recovery needed
        assert!(!table.layer_needs_recovery(0));

        // Act & Assert: layer 1 has pages in L2 → needs recovery
        assert!(table.layer_needs_recovery(1));

        // Act & Assert: nonexistent layer → no recovery needed
        assert!(!table.layer_needs_recovery(99));

        // Act: migrate layer 1 back to L1 → no longer needs recovery
        table.update_layer_tier(1, Tier::L1);
        assert!(!table.layer_needs_recovery(1));
    }

    #[test]
    fn dispatch_coordinator_weight_page_table_update_physical_id_swaps_mapping() {
        // Arrange: register layer with physical pages
        use crate::scheduler::fault_recovery::WeightPageTable;
        use crate::scheduler::memory_manager::Tier;
        let mut table = WeightPageTable::new();
        table.register_layer(5, vec![100, 200, 300]);

        // Assert: initial reverse mapping
        assert_eq!(table.layer_for_page(200), Some(5));
        assert_eq!(table.position_for_page(200), Some(1));
        assert_eq!(table.page_tier(200), Some(Tier::L1));

        // Act: update physical id at position 1 from 200 to 999 in L2
        let old = table.update_physical_id(5, 1, 999, Tier::L2);

        // Assert: old physical id returned
        assert_eq!(old, Some(200));

        // Assert: new physical id resolves correctly
        assert_eq!(table.layer_for_page(999), Some(5));
        assert_eq!(table.position_for_page(999), Some(1));
        assert_eq!(table.page_tier(999), Some(Tier::L2));

        // Assert: old physical id no longer resolves
        assert_eq!(table.layer_for_page(200), None);
        assert_eq!(table.page_tier(200), None);

        // Assert: other entries unaffected
        assert_eq!(table.layer_for_page(100), Some(5));
        assert_eq!(table.layer_for_page(300), Some(5));

        // Act: update at out-of-bounds position returns None
        let oob = table.update_physical_id(5, 10, 888, Tier::L1);
        assert!(oob.is_none());
    }

    #[test]
    fn dispatch_coordinator_fault_recovery_error_display_trait() {
        // Arrange: construct all four FaultRecoveryError variants
        use crate::scheduler::fault_recovery::FaultRecoveryError;
        use crate::scheduler::memory_manager::Tier;

        let page_not_found = FaultRecoveryError::PageNotFound { page_id: 42, tier: Tier::L2 };
        let target_full = FaultRecoveryError::TargetTierFull { tier: Tier::L1 };
        let migration_failed = FaultRecoveryError::MigrationFailed {
            page_id: 7,
            reason: "DMA timeout".to_string(),
        };
        let max_retries = FaultRecoveryError::MaxRetriesExceeded { page_id: 99 };

        // Act: format each via Display
        let pnf_str = format!("{}", page_not_found);
        let tf_str = format!("{}", target_full);
        let mf_str = format!("{}", migration_failed);
        let mr_str = format!("{}", max_retries);

        // Assert: each output contains its identifying text
        assert!(pnf_str.contains("42"), "PageNotFound display: {}", pnf_str);
        assert!(pnf_str.contains("L2"), "PageNotFound display should mention tier: {}", pnf_str);
        assert!(tf_str.contains("L1"), "TargetTierFull display: {}", tf_str);
        assert!(tf_str.contains("insufficient"), "TargetTierFull display: {}", tf_str);
        assert!(mf_str.contains("7"), "MigrationFailed display: {}", mf_str);
        assert!(mf_str.contains("DMA timeout"), "MigrationFailed display: {}", mf_str);
        assert!(mr_str.contains("99"), "MaxRetriesExceeded display: {}", mr_str);
        assert!(mr_str.contains("retries"), "MaxRetriesExceeded display: {}", mr_str);
    }

    #[test]
    fn dispatch_coordinator_layer_donor_info_owned_and_reference() {
        // Arrange: construct owned and reference entries
        use crate::kv_cache::LayerDonorInfo;
        let owned = LayerDonorInfo::owned(5, 0);
        let reference = LayerDonorInfo::reference(8, 1, 5);

        // Assert: owned entry has no donor, zero borrowers, not shared
        assert_eq!(owned.layer, 5);
        assert_eq!(owned.attn_bucket, 0);
        assert!(owned.donor_layer.is_none());
        assert_eq!(owned.borrower_refcount, 0);
        assert!(!owned.is_shared());

        // Assert: reference entry points to donor layer 5, is shared
        assert_eq!(reference.layer, 8);
        assert_eq!(reference.attn_bucket, 1);
        assert_eq!(reference.donor_layer, Some(5));
        assert_eq!(reference.borrower_refcount, 0);
        assert!(reference.is_shared());
    }

    #[test]
    fn dispatch_coordinator_dead_ratio_round_trip() {
        // Arrange: test the boundary and midpoint values for dead ratio conversion
        use crate::kv_cache::{f32_to_dead_ratio, dead_ratio_to_f32};

        // Act & Assert: zero and one boundaries
        assert_eq!(f32_to_dead_ratio(0.0), 0);
        assert_eq!(dead_ratio_to_f32(0), 0.0);
        assert_eq!(f32_to_dead_ratio(1.0), 255);
        assert!((dead_ratio_to_f32(255) - 1.0).abs() < f32::EPSILON);

        // Act & Assert: midpoint
        let mid = f32_to_dead_ratio(0.5);
        assert!(mid > 0 && mid < 255);
        let recovered = dead_ratio_to_f32(mid);
        assert!((recovered - 0.5).abs() < 0.01, "round-trip for 0.5: recovered={}", recovered);

        // Act & Assert: clamping — values outside [0,1] are clamped
        assert_eq!(f32_to_dead_ratio(-1.0), 0);
        assert_eq!(f32_to_dead_ratio(2.0), 255);
    }

    #[test]
    fn dispatch_coordinator_eviction_worker_config_default_values() {
        // Arrange: construct default EvictionWorkerConfig
        use crate::scheduler::eviction_worker::EvictionWorkerConfig;
        let config = EvictionWorkerConfig::default();

        // Assert: default values match SPEC constants
        assert_eq!(config.tick_interval, std::time::Duration::from_millis(10));
        assert_eq!(config.max_evict_per_round, 8);
        assert!((config.hbm_pressure_threshold - 0.90).abs() < f32::EPSILON);
        assert!((config.dram_pressure_threshold - 0.80).abs() < f32::EPSILON);
        assert_eq!(config.importance_threshold, 100);
        assert_eq!(config.hbm_evict_age_ticks, 50);
        assert_eq!(config.dram_evict_age_ticks, 500);
    }

    #[test]
    fn dispatch_coordinator_migration_result_and_done_construction() {
        // Arrange: construct MigrationResult variants and a MigrationDone
        use crate::scheduler::migration_actor::{MigrationDone, MigrationResult};
        use crate::kv_cache::StorageTier;

        let ok_result = MigrationResult::Ok {
            compressed_bytes: 2048,
            checksum: 0xABCD,
        };
        let failed_result = MigrationResult::Failed {
            reason: "DMA error".to_string(),
        };

        // Assert: Ok variant fields
        match &ok_result {
            MigrationResult::Ok { compressed_bytes, checksum } => {
                assert_eq!(*compressed_bytes, 2048);
                assert_eq!(*checksum, 0xABCD);
            }
            MigrationResult::Failed { .. } => panic!("expected Ok variant"),
        }

        // Assert: Failed variant fields
        match &failed_result {
            MigrationResult::Failed { reason } => {
                assert_eq!(reason, "DMA error");
            }
            MigrationResult::Ok { .. } => panic!("expected Failed variant"),
        }

        // Act: construct MigrationDone wrapping the Ok result
        let done = MigrationDone {
            page_id: 42,
            from_tier: StorageTier::GpuHbm,
            to_tier: StorageTier::CpuDram,
            result: ok_result,
        };

        // Assert: MigrationDone fields match construction
        assert_eq!(done.page_id, 42);
        assert_eq!(done.from_tier, StorageTier::GpuHbm);
        assert_eq!(done.to_tier, StorageTier::CpuDram);
    }

    #[test]
    fn dispatch_coordinator_fwht_insertion_point_variants_distinct() {
        // Arrange: all three FWHT insertion points
        use crate::kv_cache::turboquant::FwhtInsertionPoint;
        let attn = FwhtInsertionPoint::AttentionEpilogue;
        let ffn = FwhtInsertionPoint::FfnEpilogue;
        let kv = FwhtInsertionPoint::KvWrite;

        // Assert: all three are pairwise distinct
        assert_ne!(attn, ffn);
        assert_ne!(ffn, kv);
        assert_ne!(attn, kv);

        // Assert: self-equality
        assert_eq!(attn, FwhtInsertionPoint::AttentionEpilogue);
        assert_eq!(ffn, FwhtInsertionPoint::FfnEpilogue);
        assert_eq!(kv, FwhtInsertionPoint::KvWrite);

        // Assert: Debug output contains variant names
        assert!(format!("{:?}", attn).contains("AttentionEpilogue"));
        assert!(format!("{:?}", ffn).contains("FfnEpilogue"));
        assert!(format!("{:?}", kv).contains("KvWrite"));
    }

    #[test]
    fn dispatch_coordinator_migration_actor_config_default_values() {
        // Arrange: construct default MigrationActorConfig
        use crate::scheduler::migration_actor::MigrationActorConfig;
        let config = MigrationActorConfig::default();

        // Assert: default values match SPEC
        assert_eq!(config.queue_capacity, 256);
        assert_eq!(config.page_size, 4096);
        assert_eq!(config.max_swap_pages, 4096);
        assert_eq!(config.session_id, "default");

        // Assert: swap_file_path joins correctly
        let path = config.swap_file_path();
        assert!(path.to_string_lossy().ends_with("default.swap"));
        assert!(path.to_string_lossy().contains("swap"));
    }

    // ── 10 new tests (wave-12x44) ──────────────────────────────────────────

    #[test]
    fn dispatch_coordinator_scheduler_error_display_all_variants() {
        // Arrange: construct all six SchedulerError variants
        use crate::scheduler::paged_scheduler::SchedulerError;

        let oom = SchedulerError::OutOfMemory {
            operation: "prefill",
            needed_blocks: 10,
            free_blocks: 3,
        };
        let missing_group = SchedulerError::MissingGroup {
            request_id: 42,
            context: "swap_out",
        };
        let invariant = SchedulerError::AllocatorInvariant {
            operation: "allocate",
        };
        let overflow = SchedulerError::StorageKeyOverflow {
            field: "page_index",
        };
        let missing_donor = SchedulerError::MissingDonorPage {
            request_id: 7,
            consumer_layer: 3,
            donor_layer: 0,
        };
        let no_donor = SchedulerError::NoDonorForConsumer {
            layer: 5,
            bucket: 2,
        };
        let pattern_mismatch = SchedulerError::AttentionPatternMismatch {
            pattern_len: 10,
            num_layers: 12,
        };

        // Act: format each via Display
        let oom_str = format!("{}", oom);
        let mg_str = format!("{}", missing_group);
        let inv_str = format!("{}", invariant);
        let of_str = format!("{}", overflow);
        let md_str = format!("{}", missing_donor);
        let nd_str = format!("{}", no_donor);
        let pm_str = format!("{}", pattern_mismatch);

        // Assert: each contains its identifying data
        assert!(oom_str.contains("prefill"), "OOM display: {}", oom_str);
        assert!(oom_str.contains("10"), "OOM needed_blocks: {}", oom_str);
        assert!(oom_str.contains("3"), "OOM free_blocks: {}", oom_str);

        assert!(mg_str.contains("42"), "MissingGroup display: {}", mg_str);
        assert!(mg_str.contains("swap_out"), "MissingGroup context: {}", mg_str);

        assert!(inv_str.contains("allocate"), "Invariant display: {}", inv_str);

        assert!(of_str.contains("page_index"), "Overflow display: {}", of_str);

        assert!(md_str.contains("7"), "MissingDonor display: {}", md_str);
        assert!(md_str.contains("3"), "MissingDonor consumer: {}", md_str);
        assert!(md_str.contains("0"), "MissingDonor donor: {}", md_str);

        assert!(nd_str.contains("5"), "NoDonor layer: {}", nd_str);
        assert!(nd_str.contains("2"), "NoDonor bucket: {}", nd_str);

        assert!(pm_str.contains("10"), "PatternMismatch len: {}", pm_str);
        assert!(pm_str.contains("12"), "PatternMismatch layers: {}", pm_str);
    }

    #[test]
    fn dispatch_coordinator_backend_error_display_all_variants() {
        // Arrange: construct all BackendError variants
        use crate::engine::executor_types::BackendError;

        let cuda = BackendError::Cuda("device lost".to_string());
        let hip = BackendError::Hip("memory fault".to_string());
        let metal = BackendError::Metal("shader compile fail".to_string());
        let cpu = BackendError::Cpu("illegal instruction".to_string());
        let unimpl = BackendError::Unimplemented("fused attention");
        let other = BackendError::Other("unknown failure".to_string());

        // Act: format each via Display
        let cuda_str = format!("{}", cuda);
        let hip_str = format!("{}", hip);
        let metal_str = format!("{}", metal);
        let cpu_str = format!("{}", cpu);
        let unimpl_str = format!("{}", unimpl);
        let other_str = format!("{}", other);

        // Assert: each output contains identifying text
        assert!(cuda_str.contains("CUDA"), "CUDA prefix: {}", cuda_str);
        assert!(cuda_str.contains("device lost"), "CUDA message: {}", cuda_str);

        assert!(hip_str.contains("HIP"), "HIP prefix: {}", hip_str);
        assert!(hip_str.contains("memory fault"), "HIP message: {}", hip_str);

        assert!(metal_str.contains("Metal"), "Metal prefix: {}", metal_str);
        assert!(metal_str.contains("shader compile fail"), "Metal message: {}", metal_str);

        assert!(cpu_str.contains("CPU"), "CPU prefix: {}", cpu_str);
        assert!(cpu_str.contains("illegal instruction"), "CPU message: {}", cpu_str);

        assert!(unimpl_str.contains("unimplemented"), "Unimpl prefix: {}", unimpl_str);
        assert!(unimpl_str.contains("fused attention"), "Unimpl what: {}", unimpl_str);

        assert!(other_str.contains("backend error"), "Other prefix: {}", other_str);
        assert!(other_str.contains("unknown failure"), "Other message: {}", other_str);
    }

    #[test]
    fn dispatch_coordinator_tier_enum_variants_and_ordering() {
        // Arrange: get all Tier variants
        use crate::scheduler::memory_manager::Tier;
        let l1 = Tier::L1;
        let l2 = Tier::L2;
        let l3 = Tier::L3;

        // Assert: all pairwise distinct
        assert_ne!(l1, l2);
        assert_ne!(l2, l3);
        assert_ne!(l1, l3);

        // Assert: self-equality
        assert_eq!(l1, Tier::L1);
        assert_eq!(l2, Tier::L2);
        assert_eq!(l3, Tier::L3);

        // Assert: ordering L1 < L2 < L3
        assert!(l1 < l2, "L1 should be less than L2");
        assert!(l2 < l3, "L2 should be less than L3");
        assert!(l1 < l3, "L1 should be less than L3");

        // Assert: Debug output contains variant names
        let debug_l1 = format!("{:?}", l1);
        let debug_l2 = format!("{:?}", l2);
        let debug_l3 = format!("{:?}", l3);
        assert!(debug_l1.contains("L1"), "L1 debug: {}", debug_l1);
        assert!(debug_l2.contains("L2"), "L2 debug: {}", debug_l2);
        assert!(debug_l3.contains("L3"), "L3 debug: {}", debug_l3);

        // Assert: Copy and Clone
        let l1_copy = l1;
        assert_eq!(l1_copy, Tier::L1);

        // Assert: Hash (usable as HashMap key)
        let mut map = std::collections::HashMap::new();
        map.insert(l1, 100);
        map.insert(l2, 200);
        assert_eq!(map.get(&l1), Some(&100));
        assert_eq!(map.get(&l2), Some(&200));
        assert_eq!(map.get(&l3), None);
    }

    #[test]
    fn dispatch_coordinator_kv_cache_error_display_and_construction() {
        // Arrange: construct KvCacheError::Exhausted
        use crate::kv_cache::KvCacheError;

        let exhausted = KvCacheError::Exhausted {
            requested: 512,
            available: 256,
        };

        // Act: format via Display
        let err_str = format!("{}", exhausted);

        // Assert: display contains both numbers
        assert!(err_str.contains("512"), "requested should appear: {}", err_str);
        assert!(err_str.contains("256"), "available should appear: {}", err_str);

        // Assert: Debug output
        let debug_str = format!("{:?}", exhausted);
        assert!(debug_str.contains("Exhausted"), "Debug should contain variant name: {}", debug_str);

        // Assert: std::error::Error trait is implemented
        let _: &dyn std::error::Error = &exhausted;
    }

    #[test]
    fn dispatch_coordinator_attention_mask_type_variants_distinct() {
        // Arrange: construct both AttentionMaskType variants
        use crate::engine::executor_types::AttentionMaskType;

        let bidir = AttentionMaskType::Bidirectional;
        let causal = AttentionMaskType::Causal;

        // Assert: pairwise distinct
        assert_ne!(bidir, causal);

        // Assert: self-equality
        assert_eq!(bidir, AttentionMaskType::Bidirectional);
        assert_eq!(causal, AttentionMaskType::Causal);

        // Assert: Debug output contains variant names
        assert!(format!("{:?}", bidir).contains("Bidirectional"));
        assert!(format!("{:?}", causal).contains("Causal"));

        // Assert: Copy + Clone
        let bidir_copy = bidir;
        assert_eq!(bidir_copy, AttentionMaskType::Bidirectional);

        // Assert: usable as HashMap key (Hash)
        let mut map = std::collections::HashMap::new();
        map.insert(bidir, "bert");
        map.insert(causal, "gpt");
        assert_eq!(map.get(&AttentionMaskType::Bidirectional), Some(&"bert"));
        assert_eq!(map.get(&AttentionMaskType::Causal), Some(&"gpt"));
    }

    #[test]
    fn dispatch_coordinator_kv_cache_handle_construction_equality_hash() {
        // Arrange: construct multiple KvCacheHandle instances
        use crate::engine::executor_types::KvCacheHandle;

        let h1 = KvCacheHandle(42);
        let h2 = KvCacheHandle(42);
        let h3 = KvCacheHandle(99);
        let h4 = KvCacheHandle(0);

        // Assert: same inner value => equal
        assert_eq!(h1, h2, "same u64 value should be equal");

        // Assert: different inner value => not equal
        assert_ne!(h1, h3);
        assert_ne!(h2, h4);

        // Assert: zero handle
        assert_eq!(h4.0, 0);

        // Assert: Debug output contains inner value
        let debug = format!("{:?}", h1);
        assert!(debug.contains("42"), "Debug should contain inner u64: {}", debug);

        // Assert: Copy + Clone
        let h1_clone = h1.clone();
        assert_eq!(h1_clone, h1);

        // Assert: Hash (usable as HashMap key)
        let mut map = std::collections::HashMap::new();
        map.insert(h1, "first");
        map.insert(h3, "second");
        assert_eq!(map.get(&KvCacheHandle(42)), Some(&"first"));
        assert_eq!(map.get(&KvCacheHandle(99)), Some(&"second"));
        assert_eq!(map.get(&KvCacheHandle(0)), None);
    }

    #[test]
    fn dispatch_coordinator_logits_handle_construction_and_access() {
        // Arrange: construct LogitsHandle with known data
        use crate::engine::executor_types::LogitsHandle;

        let handle = LogitsHandle {
            data: vec![0.1, 0.5, 0.3, 0.05, 0.05],
        };

        // Assert: data field accessible and matches
        assert_eq!(handle.data.len(), 5);
        assert_eq!(handle.data[0], 0.1);
        assert_eq!(handle.data[1], 0.5);

        // Assert: sum of probabilities (approximate check)
        let sum: f32 = handle.data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "probabilities should sum to ~1.0, got {}", sum);

        // Assert: Debug trait
        let debug = format!("{:?}", handle);
        assert!(debug.contains("LogitsHandle"), "Debug should contain type name: {}", debug);

        // Assert: Clone
        let cloned = handle.clone();
        assert_eq!(cloned.data.len(), 5);
        assert_eq!(cloned.data[2], 0.3);

        // Assert: empty data is valid
        let empty = LogitsHandle { data: vec![] };
        assert!(empty.data.is_empty());
    }

    #[test]
    fn dispatch_coordinator_observer_error_display_trait() {
        // Arrange: construct ObserverError
        use crate::scheduler::observer::ObserverError;

        let err = ObserverError::BackendUnavailable("GPU driver crashed".to_string());

        // Act: format via Display
        let err_str = format!("{}", err);

        // Assert: output contains the message
        assert!(err_str.contains("backend unavailable"), "should contain prefix: {}", err_str);
        assert!(err_str.contains("GPU driver crashed"), "should contain detail: {}", err_str);

        // Assert: Debug output
        let debug = format!("{:?}", err);
        assert!(debug.contains("BackendUnavailable"), "Debug variant name: {}", debug);

        // Assert: std::error::Error trait is implemented
        let _: &dyn std::error::Error = &err;
    }

    #[test]
    fn dispatch_coordinator_page_fault_construction_all_fields() {
        // Arrange: construct a PageFault with both optional fields
        use crate::scheduler::fault_recovery::PageFault;
        use crate::scheduler::memory_manager::Tier;

        let expert_fault = PageFault {
            page_id: 42,
            current_tier: Tier::L2,
            target_tier: Tier::L1,
            fault_time: std::time::Instant::now(),
            expert_key: Some((3, 7)),
            dense_layer_idx: None,
        };

        // Assert: basic fields
        assert_eq!(expert_fault.page_id, 42);
        assert_eq!(expert_fault.current_tier, Tier::L2);
        assert_eq!(expert_fault.target_tier, Tier::L1);
        assert_eq!(expert_fault.expert_key, Some((3, 7)));
        assert!(expert_fault.dense_layer_idx.is_none());

        // Arrange: dense layer fault (no expert, has layer idx)
        let dense_fault = PageFault {
            page_id: 100,
            current_tier: Tier::L3,
            target_tier: Tier::L1,
            fault_time: std::time::Instant::now(),
            expert_key: None,
            dense_layer_idx: Some(15),
        };

        // Assert: dense layer fault fields
        assert_eq!(dense_fault.page_id, 100);
        assert_eq!(dense_fault.current_tier, Tier::L3);
        assert!(dense_fault.expert_key.is_none());
        assert_eq!(dense_fault.dense_layer_idx, Some(15));

        // Assert: Debug output contains field info
        let debug = format!("{:?}", expert_fault);
        assert!(debug.contains("42"), "Debug should contain page_id: {}", debug);
    }

    #[test]
    fn dispatch_coordinator_compact_scatter_meta_construction_and_equality() {
        // Arrange: construct CompactScatterMeta instances
        use crate::scheduler::request_state::CompactScatterMeta;

        let meta_a = CompactScatterMeta {
            original_slot: 3,
            compacted_slot: 0,
            active: 1,
        };
        let meta_b = CompactScatterMeta {
            original_slot: 3,
            compacted_slot: 0,
            active: 1,
        };
        let meta_inactive = CompactScatterMeta {
            original_slot: 5,
            compacted_slot: 1,
            active: 0,
        };

        // Assert: equality with same values
        assert_eq!(meta_a, meta_b, "identical field values should be equal");

        // Assert: inequality with different values
        assert_ne!(meta_a, meta_inactive);

        // Assert: active flag distinguishes entries
        let mut meta_different_active = meta_a;
        meta_different_active.active = 0;
        assert_ne!(meta_a, meta_different_active);

        // Assert: original_slot field
        assert_eq!(meta_a.original_slot, 3);
        assert_eq!(meta_inactive.original_slot, 5);

        // Assert: compacted_slot field
        assert_eq!(meta_a.compacted_slot, 0);
        assert_eq!(meta_inactive.compacted_slot, 1);

        // Assert: Debug output contains field names
        let debug = format!("{:?}", meta_a);
        assert!(debug.contains("original_slot"), "Debug should show original_slot: {}", debug);
        assert!(debug.contains("compacted_slot"), "Debug should show compacted_slot: {}", debug);
        assert!(debug.contains("active"), "Debug should show active: {}", debug);

        // Assert: Copy trait
        let meta_copy = meta_a;
        assert_eq!(meta_copy, meta_a);
    }
}
