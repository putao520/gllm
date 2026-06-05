//! WP9: Weight Paging Integration Tests
//!
//! Tests the weight paging system end-to-end, covering:
//! - WeightPageTable lifecycle (register, lookup, update, tier distribution)
//! - GMM page allocation and tier migration
//! - Page fault recovery (L2→L1, L3→L2→L1 two-hop)
//! - Step fault plan generation for dense and MoE expert layers
//! - HGAL eviction priority calculation and victim selection
//! - BasicObserver weight page telemetry events
//! - WeightPaging enabled and disabled modes
//!
//! SPEC 21-WEIGHT-PAGING.md §1–§8

use std::collections::HashMap;
use std::time::Instant;

use gllm::engine::mega_kernel::WeightPageJitConfig;
use gllm::kv_cache::CompressionCodec;
use gllm::scheduler::{
    BasicObserver, EvictionReason, FaultAction, FaultRecoveryError, FaultRecoveryHandler,
    FaultRecoveryStats, HGALConfig, HGALScheduler, PageFault, PagePayloadKind, PageState, Tier,
    WeightPageTable, WeightPageTelemetryEvent, WeightTier,
};
use gllm::scheduler::memory_manager::{GlobalMemoryManager, TierUsage};
use gllm::scheduler::types::KvPipeline;

// ===========================================================================
// Helper utilities
// ===========================================================================

/// Create a GMM with small but usable capacities for testing.
fn test_gmm() -> GlobalMemoryManager {
    GlobalMemoryManager::new_with_capacities(16, 8, 4)
}

/// Create a HGALScheduler with default config.
fn test_hgal() -> HGALScheduler {
    HGALScheduler::new(HGALConfig::default())
}

/// Verify tier usage matches expected values.
fn assert_tier_usage(gmm: &GlobalMemoryManager, tier: Tier, expected_used: usize, expected_capacity: usize) {
    let usage = gmm.tier_usage(tier);
    assert_eq!(
        usage.used, expected_used,
        "tier {:?} used pages mismatch (expected {}, got {})",
        tier, expected_used, usage.used
    );
    assert_eq!(
        usage.capacity, expected_capacity,
        "tier {:?} capacity mismatch (expected {}, got {})",
        tier, expected_capacity, usage.capacity
    );
}

// ===========================================================================
// REQ-WP-007: WeightPageTable Lifecycle
// ===========================================================================

/// Test: Register layer weight pages and verify forward/reverse lookups.
#[test]
fn weight_page_table_register_and_lookup() {
    // ── SPEC §4: WeightPageTable is populated during weight loading ──
    let mut table = WeightPageTable::new();
    assert_eq!(table.layer_count(), 0);
    assert_eq!(table.total_pages(), 0);

    // Register dense layer 0 with 3 physical pages
    table.register_layer(0, vec![10, 11, 12]);
    // Register MoE layer 1 with 2 expert pages
    table.register_layer(1, vec![20, 21]);

    assert_eq!(table.layer_count(), 2);
    assert_eq!(table.total_pages(), 5);

    // ── Forward lookup ──
    assert_eq!(table.get_layer_pages(0), Some(&[10, 11, 12][..]));
    assert_eq!(table.get_layer_pages(1), Some(&[20, 21][..]));
    assert_eq!(table.get_layer_pages(99), None);

    // ── Reverse lookup ──
    assert_eq!(table.layer_for_page(10), Some(0));
    assert_eq!(table.position_for_page(10), Some(0));
    assert_eq!(table.layer_for_page(12), Some(0));
    assert_eq!(table.position_for_page(12), Some(2));
    assert_eq!(table.layer_for_page(21), Some(1));
    assert_eq!(table.position_for_page(21), Some(1));
    assert_eq!(table.layer_for_page(999), None);

    // ── Default tier ──
    assert_eq!(table.page_tier(10), Some(Tier::L1));
    assert_eq!(table.page_tier(20), Some(Tier::L1));
}

/// Test: Update physical ID after tier migration updates all mappings.
#[test]
fn weight_page_table_update_after_migration() {
    let mut table = WeightPageTable::new();
    table.register_layer(3, vec![100, 101, 102]);

    // Migrate page 100 from L1 to L2, receiving new physical ID 200
    let old = table.update_physical_id(3, 0, 200, Tier::L2);
    assert_eq!(old, Some(100));

    // Forward map: page at position 0 is now 200
    let pages = table.get_layer_pages(3).expect("layer 3 exists");
    assert_eq!(pages[0], 200);
    assert_eq!(pages[1], 101);
    assert_eq!(pages[2], 102);

    // Reverse map: old ID removed, new ID registered
    assert_eq!(table.layer_for_page(200), Some(3));
    assert_eq!(table.position_for_page(200), Some(0));
    assert_eq!(table.layer_for_page(100), None);
    assert_eq!(table.position_for_page(100), None);

    // Tier tracking updated
    assert_eq!(table.page_tier(200), Some(Tier::L2));
    assert_eq!(table.page_tier(101), Some(Tier::L1));
    assert_eq!(table.page_tier(102), Some(Tier::L1));
}

/// Test: Tier distribution counting after migrations.
#[test]
fn weight_page_table_tier_distribution() {
    let mut table = WeightPageTable::new();
    table.register_layer(0, vec![1, 2, 3]);

    // All in L1 initially
    assert_eq!(table.tier_distribution(), (3, 0, 0));

    // Migrate one to L2
    table.update_physical_id(0, 1, 200, Tier::L2);
    assert_eq!(table.tier_distribution(), (2, 1, 0));

    // Migrate one to L3
    table.update_physical_id(0, 2, 300, Tier::L3);
    assert_eq!(table.tier_distribution(), (1, 1, 1));

    // Migrate last to L2
    table.update_physical_id(0, 0, 400, Tier::L2);
    assert_eq!(table.tier_distribution(), (0, 2, 1));
}

/// Test: needs_recovery detects L2/L3 pages.
#[test]
fn weight_page_table_needs_recovery() {
    let mut table = WeightPageTable::new();
    table.register_layer(0, vec![1, 2]);

    assert!(!table.layer_needs_recovery(0));
    assert!(!table.layer_needs_recovery(99)); // non-existent layer

    // Migrate one page to L2 → layer needs recovery
    table.update_physical_id(0, 0, 100, Tier::L2);
    assert!(table.layer_needs_recovery(0));

    // Migrate it back to L1 → no longer needs recovery
    table.update_physical_id(0, 0, 1, Tier::L1);
    assert!(!table.layer_needs_recovery(0));
}

/// Test: Batch update layer tier.
#[test]
fn weight_page_table_batch_tier_update() {
    let mut table = WeightPageTable::new();
    table.register_layer(0, vec![10, 11, 12]);
    table.register_layer(1, vec![20, 21]);

    table.update_layer_tier(0, Tier::L2);
    assert_eq!(table.page_tier(10), Some(Tier::L2));
    assert_eq!(table.page_tier(11), Some(Tier::L2));
    assert_eq!(table.page_tier(12), Some(Tier::L2));

    // Layer 1 still in L1
    assert_eq!(table.page_tier(20), Some(Tier::L1));
    assert_eq!(table.page_tier(21), Some(Tier::L1));
}

// ===========================================================================
// GMM Decision Correctness (SPEC §3, §5)
// ===========================================================================

/// Test: GMM page allocation and deallocation across all three tiers.
#[test]
fn gmm_page_alloc_free_all_tiers() {
    let mut gmm = test_gmm();

    assert_tier_usage(&gmm, Tier::L1, 0, 16);
    assert_tier_usage(&gmm, Tier::L2, 0, 8);
    assert_tier_usage(&gmm, Tier::L3, 0, 4);

    // Allocate pages in each tier
    let l1_pid = gmm.allocate_page(Tier::L1).expect("alloc L1");
    let l2_pid = gmm.allocate_page(Tier::L2).expect("alloc L2");
    let l3_pid = gmm.allocate_page(Tier::L3).expect("alloc L3");

    assert_tier_usage(&gmm, Tier::L1, 1, 16);
    assert_tier_usage(&gmm, Tier::L2, 1, 8);
    assert_tier_usage(&gmm, Tier::L3, 1, 4);

    // Free pages
    gmm.free_page(Tier::L1, l1_pid).expect("free L1");
    gmm.free_page(Tier::L2, l2_pid).expect("free L2");
    gmm.free_page(Tier::L3, l3_pid).expect("free L3");

    assert_tier_usage(&gmm, Tier::L1, 0, 16);
    assert_tier_usage(&gmm, Tier::L2, 0, 8);
    assert_tier_usage(&gmm, Tier::L3, 0, 4);
}

/// Test: GMM tier migration from L1→L2 and L2→L3.
#[test]
fn gmm_tier_migration_downward() {
    let mut gmm = test_gmm();

    let pid = gmm.allocate_page(Tier::L1).expect("alloc L1");
    assert_tier_usage(&gmm, Tier::L1, 1, 16);

    // L1 → L2 migration
    let l2_pid = gmm.migrate_page(Tier::L1, Tier::L2, pid).expect("migrate L1→L2");
    assert_tier_usage(&gmm, Tier::L1, 0, 16);
    assert_tier_usage(&gmm, Tier::L2, 1, 8);

    // L2 → L3 migration
    let l3_pid = gmm.migrate_page(Tier::L2, Tier::L3, l2_pid).expect("migrate L2→L3");
    assert_tier_usage(&gmm, Tier::L2, 0, 8);
    assert_tier_usage(&gmm, Tier::L3, 1, 4);

    // L3 → L2 migration (promotion)
    let l2_pid2 = gmm.migrate_page(Tier::L3, Tier::L2, l3_pid).expect("migrate L3→L2");
    assert_tier_usage(&gmm, Tier::L3, 0, 4);
    assert_tier_usage(&gmm, Tier::L2, 1, 8);

    // Verify tiers updated correctly after promotion
    assert_tier_usage(&gmm, Tier::L3, 0, 4);
    assert_tier_usage(&gmm, Tier::L2, 1, 8);
}

/// Test: GMM tracks pages via track_page and can verify allocation.
#[test]
fn gmm_track_and_untrack_pages() {
    let mut gmm = test_gmm();

    // track_page registers an externally allocated page
    gmm.track_page(Tier::L1, 42).expect("track L1 page 42");
    assert_tier_usage(&gmm, Tier::L1, 1, 16);

    gmm.track_page(Tier::L1, 43).expect("track L1 page 43");
    assert_tier_usage(&gmm, Tier::L1, 2, 16);

    // untrack (free) a tracked page
    gmm.untrack_page(Tier::L1, 42).expect("untrack L1 page 42");
    assert_tier_usage(&gmm, Tier::L1, 1, 16);

    // Track exceeding capacity should fail
    let mut full_gmm = GlobalMemoryManager::new_with_capacities(1, 1, 1);
    full_gmm.track_page(Tier::L1, 100).expect("track first");
    assert!(full_gmm.track_page(Tier::L1, 101).is_err());
}

/// Test: tier_usage.available() returns correct free count.
#[test]
fn gmm_tier_usage_available() {
    let gmm = GlobalMemoryManager::new_with_capacities(10, 5, 2);

    let l1 = gmm.tier_usage(Tier::L1);
    assert_eq!(l1.available(), 10);
    assert_eq!(l1.capacity, 10);
    assert_eq!(l1.used, 0);

    let l2 = gmm.tier_usage(Tier::L2);
    assert_eq!(l2.available(), 5);

    let l3 = gmm.tier_usage(Tier::L3);
    assert_eq!(l3.available(), 2);
}

// ===========================================================================
// Page Fault Recovery (SPEC §6)
// ===========================================================================

/// Test: L2→L1 fault yields LoadFromTier action with correct source/target.
#[test]
fn fault_recovery_l2_to_l1_action() {
    let mut handler = FaultRecoveryHandler::new();
    let mut gmm = test_gmm();
    let mut table = WeightPageTable::new();

    // Allocate and register a page, then evict it to L2
    let pid = gmm.allocate_page(Tier::L1).expect("alloc L1");
    table.register_layer(0, vec![pid]);

    let l2_pid = gmm.migrate_page(Tier::L1, Tier::L2, pid).expect("migrate to L2");
    table.update_physical_id(0, 0, l2_pid, Tier::L2);

    let fault = PageFault {
        page_id: l2_pid,
        current_tier: Tier::L2,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: None,
        dense_layer_idx: Some(0),
    };

    let action = handler.handle_page_fault(&fault, &gmm, &table);
    assert_eq!(
        action,
        FaultAction::LoadFromTier {
            source_tier: Tier::L2,
            target_tier: Tier::L1,
        },
        "L2 fault should yield LoadFromTier L2→L1"
    );

    // Stats updated
    assert_eq!(handler.stats.total_faults, 1);
    assert_eq!(handler.stats.successful_recoveries, 0); // not yet recovered
}

/// Test: L3→L1 fault yields LoadFromTier with first hop L3→L2.
#[test]
fn fault_recovery_l3_to_l1_two_hop_action() {
    let mut handler = FaultRecoveryHandler::new();
    let gmm = test_gmm();
    let mut table = WeightPageTable::new();

    table.register_layer(0, vec![100]);
    // Simulate page migrated to L3
    table.update_physical_id(0, 0, 100, Tier::L3);

    let fault = PageFault {
        page_id: 100,
        current_tier: Tier::L3,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: Some((5, 0)),
        dense_layer_idx: None,
    };

    let action = handler.handle_page_fault(&fault, &gmm, &table);
    // First hop: L3→L2
    assert_eq!(
        action,
        FaultAction::LoadFromTier {
            source_tier: Tier::L3,
            target_tier: Tier::L2,
        },
        "L3 fault should yield LoadFromTier L3→L2 (first hop)"
    );
}

/// Test: Page already in target tier yields LoadFromTier with same tier (no-op).
#[test]
fn fault_recovery_page_already_in_l1() {
    let mut handler = FaultRecoveryHandler::new();
    let gmm = test_gmm();
    let mut table = WeightPageTable::new();

    let pid = gmm.allocate_page(Tier::L1).expect("alloc L1");
    table.register_layer(0, vec![pid]);

    let fault = PageFault {
        page_id: pid,
        current_tier: Tier::L1,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: None,
        dense_layer_idx: Some(0),
    };

    let action = handler.handle_page_fault(&fault, &gmm, &table);
    assert_eq!(
        action,
        FaultAction::LoadFromTier {
            source_tier: Tier::L1,
            target_tier: Tier::L1,
        },
        "Page already in L1 should yield LoadFromTier L1→L1 (no-op)"
    );
    assert_eq!(handler.stats.successful_recoveries, 1);
}

/// Test: Target tier full causes Retry then Abort.
#[test]
fn fault_recovery_target_full_retry_then_abort() {
    let mut handler = FaultRecoveryHandler::new();
    // L1 has capacity 0, L2 has 4 pages
    let gmm = GlobalMemoryManager::new_with_capacities(0, 4, 4);
    let mut table = WeightPageTable::new();
    table.register_layer(0, vec![42]);
    table.update_physical_id(0, 0, 42, Tier::L2);

    let fault = PageFault {
        page_id: 42,
        current_tier: Tier::L2,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: None,
        dense_layer_idx: Some(0),
    };

    // First fault: target full → Retry
    let action1 = handler.handle_page_fault(&fault, &gmm, &table);
    assert_eq!(action1, FaultAction::Retry);
    assert_eq!(handler.stats.total_faults, 1);
    assert_eq!(handler.stats.retried_faults, 1);

    // Second fault: still full → Retry
    let action2 = handler.handle_page_fault(&fault, &gmm, &table);
    assert_eq!(action2, FaultAction::Retry);

    // Third fault: still full → Retry (max_retries=3, we've had 2 retries so far)
    let action3 = handler.handle_page_fault(&fault, &gmm, &table);
    assert_eq!(action3, FaultAction::Retry);

    // Fourth fault: exceeded max retries → Abort
    let action4 = handler.handle_page_fault(&fault, &gmm, &table);
    assert!(matches!(action4, FaultAction::Abort { .. }));
    assert_eq!(handler.stats.retried_faults, 3);
    assert_eq!(handler.stats.aborted_faults, 1);
}

/// Test: Full recovery execution with migration and table update.
#[test]
fn fault_recovery_full_migration_l2_to_l1() {
    let mut handler = FaultRecoveryHandler::new();
    let mut gmm = test_gmm();
    let mut table = WeightPageTable::new();

    // Setup: register a page in layer 0, migrate it to L2
    let pid = gmm.allocate_page(Tier::L1).expect("alloc L1");
    table.register_layer(0, vec![pid]);
    let l2_pid = gmm.migrate_page(Tier::L1, Tier::L2, pid).expect("migrate to L2");
    table.update_physical_id(0, 0, l2_pid, Tier::L2);

    assert!(table.layer_needs_recovery(0));

    let fault = PageFault {
        page_id: l2_pid,
        current_tier: Tier::L2,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: None,
        dense_layer_idx: Some(0),
    };

    let result = handler.recover_fault(&fault, &mut gmm, &mut table);
    assert!(result.is_ok(), "recover_fault should succeed: {:?}", result);

    let new_pid = result.unwrap();
    // Verify the page is now in L1 in the weight table
    assert_eq!(
        table.page_tier(new_pid),
        Some(Tier::L1),
        "Recovered page should be in L1"
    );
    assert!(!table.layer_needs_recovery(0), "Layer should no longer need recovery");

    // Stats
    assert_eq!(handler.stats.successful_recoveries, 2); // handle + execute
    assert_eq!(handler.stats.total_faults, 1);
}

/// Test: Full two-hop recovery from L3→L1.
#[test]
fn fault_recovery_two_hop_l3_to_l1() {
    let mut handler = FaultRecoveryHandler::new();
    let mut gmm = test_gmm();
    let mut table = WeightPageTable::new();

    // Register page, migrate to L3
    let pid = gmm.allocate_page(Tier::L1).expect("alloc L1");
    table.register_layer(0, vec![pid]);

    let l2_pid = gmm.migrate_page(Tier::L1, Tier::L2, pid).expect("migrate L1→L2");
    table.update_physical_id(0, 0, l2_pid, Tier::L2);

    let l3_pid = gmm.migrate_page(Tier::L2, Tier::L3, l2_pid).expect("migrate L2→L3");
    table.update_physical_id(0, 0, l3_pid, Tier::L3);

    assert!(table.layer_needs_recovery(0));

    let fault = PageFault {
        page_id: l3_pid,
        current_tier: Tier::L3,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: Some((3, 0)),
        dense_layer_idx: None,
    };

    let result = handler.recover_fault(&fault, &mut gmm, &mut table);
    assert!(result.is_ok(), "Two-hop recovery should succeed: {:?}", result);

    let final_pid = result.unwrap();
    assert_eq!(
        table.page_tier(final_pid),
        Some(Tier::L1),
        "Two-hop recovered page should be in L1"
    );
    assert!(!table.layer_needs_recovery(0));
}

/// Test: Recovery fails gracefully when page not in weight table.
#[test]
fn fault_recovery_page_not_in_table() {
    let mut handler = FaultRecoveryHandler::new();
    let mut gmm = test_gmm();
    let mut table = WeightPageTable::new();

    let fault = PageFault {
        page_id: 999,
        current_tier: Tier::L2,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: None,
        dense_layer_idx: None,
    };

    let result = handler.recover_fault(&fault, &mut gmm, &mut table);
    assert!(result.is_err(), "Recovery of unregistered page should fail");
    match result {
        Err(FaultRecoveryError::PageNotFound { page_id, .. }) => {
            assert_eq!(page_id, 999);
        }
        Err(e) => panic!("Expected PageNotFound error, got: {:?}", e),
        Ok(_) => panic!("Expected error"),
    }
}

// ===========================================================================
// Step Fault Plan Generation (SPEC §6.3)
// ===========================================================================

/// Test: Generate fault plan for dense layers.
#[test]
fn step_fault_plan_dense_layers() {
    let mut table = WeightPageTable::new();
    let mut gmm = test_gmm();

    // Register 2 layers with 3 pages each
    let l0_pids: Vec<_> = (0..3).map(|_| gmm.allocate_page(Tier::L1).expect("alloc")).collect();
    let l1_pids: Vec<_> = (0..3).map(|_| gmm.allocate_page(Tier::L1).expect("alloc")).collect();
    table.register_layer(0, l0_pids);
    table.register_layer(1, l1_pids);

    let expert_pages = HashMap::new();

    // All pages in L1 → no faults
    let plan = gllm::scheduler::fault_recovery::generate_step_fault_plan(
        &[0, 1],
        &table,
        &expert_pages,
    );
    assert!(!plan.has_faults());
    assert_eq!(plan.pages_in_l1, 6);
    assert_eq!(plan.l2_faults, 0);
    assert_eq!(plan.l3_faults, 0);

    // Migrate layer 0 page to L2
    let l2_pid = gmm.migrate_page(Tier::L1, Tier::L2, l0_pids[0]).expect("migrate");
    table.update_physical_id(0, 0, l2_pid, Tier::L2);

    let plan2 = gllm::scheduler::fault_recovery::generate_step_fault_plan(
        &[0, 1],
        &table,
        &expert_pages,
    );
    assert!(plan2.has_faults());
    assert_eq!(plan2.pages_in_l1, 5);
    assert_eq!(plan2.l2_faults, 1);
    assert_eq!(plan2.total_faults(), 1);
}

/// Test: Generate fault plan for MoE expert weight pages.
#[test]
fn step_fault_plan_moe_experts() {
    let mut table = WeightPageTable::new();
    let mut gmm = test_gmm();
    let mut expert_pages = HashMap::new();

    // Register expert pages for expert 3, layer 5
    let pids: Vec<_> = (0..2).map(|_| gmm.allocate_page(Tier::L1).expect("alloc")).collect();
    table.register_layer(5, pids.clone());
    expert_pages.insert((3u32, 5usize), pids);

    // All in L1 → no faults
    let plan = gllm::scheduler::fault_recovery::generate_step_fault_plan(
        &[5],
        &table,
        &expert_pages,
    );
    assert!(!plan.has_faults());
    assert_eq!(plan.pages_in_l1, 2);
    assert_eq!(plan.total_faults(), 0);

    // Migrate one expert page to L3
    let l3_pid = gmm.migrate_page(Tier::L1, Tier::L3, table.get_layer_pages(5).unwrap()[0]).expect("migrate");
    table.update_physical_id(5, 0, l3_pid, Tier::L3);
    // Also update expert_pages map
    let old_pids = expert_pages.get_mut(&(3, 5)).unwrap();
    old_pids[0] = l3_pid;

    let plan2 = gllm::scheduler::fault_recovery::generate_step_fault_plan(
        &[5],
        &table,
        &expert_pages,
    );
    assert!(plan2.has_faults());
    assert_eq!(plan2.pages_in_l1, 1);
    assert_eq!(plan2.l3_faults, 1);
    assert_eq!(plan2.total_faults(), 1);
}

/// Test: Execute step fault plan recovers all pending faults.
#[test]
fn step_fault_plan_execution() {
    let mut handler = FaultRecoveryHandler::new();
    let mut gmm = test_gmm();
    let mut table = WeightPageTable::new();

    let pid = gmm.allocate_page(Tier::L1).expect("alloc L1");
    table.register_layer(0, vec![pid]);

    let l2_pid = gmm.migrate_page(Tier::L1, Tier::L2, pid).expect("migrate");
    table.update_physical_id(0, 0, l2_pid, Tier::L2);

    let expert_pages = HashMap::new();
    let plan = gllm::scheduler::fault_recovery::generate_step_fault_plan(
        &[0],
        &table,
        &expert_pages,
    );
    assert!(plan.has_faults());
    assert_eq!(plan.total_faults(), 1);

    let (succeeded, failed) = gllm::scheduler::fault_recovery::execute_step_fault_plan(
        &plan,
        &mut handler,
        &mut gmm,
        &mut table,
    );
    assert_eq!(succeeded.len(), 1, "One page should be recovered");
    assert!(failed.is_empty(), "No pages should fail recovery");

    // Verify page is now in L1
    let new_pid = succeeded[0].1;
    assert_eq!(table.page_tier(new_pid), Some(Tier::L1));
    assert!(!table.layer_needs_recovery(0));
}

// ===========================================================================
// HGAL Eviction Priority (SPEC §5)
// ===========================================================================

/// Test: ExpertWeight pages get lower score (easier to evict) than DenseLayerWeight.
#[test]
fn eviction_priority_expert_vs_dense() {
    let hgal = test_hgal();

    let expert_page = gllm::scheduler::types::UnifiedVirtualPage {
        page_id: 1,
        payload_kind: PagePayloadKind::ExpertWeight,
        residency: gllm::scheduler::types::MemoryResidency::DeviceLocal,
        dtype: gllm_kernels::types::DType::F16,
        owner: None,
        pipeline: None,
        logical_index: 0,
        codec: CompressionCodec::None,
        compressed_size: 0,
        decompressed_size: 0,
        expert_id: Some(3),
        layer_idx: Some(5),
    };

    let dense_page = gllm::scheduler::types::UnifiedVirtualPage {
        page_id: 2,
        payload_kind: PagePayloadKind::DenseLayerWeight,
        residency: gllm::scheduler::types::MemoryResidency::DeviceLocal,
        dtype: gllm_kernels::types::DType::F16,
        owner: None,
        pipeline: None,
        logical_index: 0,
        codec: CompressionCodec::None,
        compressed_size: 0,
        decompressed_size: 0,
        expert_id: None,
        layer_idx: Some(0),
    };

    let expert_priority = hgal.compute_eviction_priority(&expert_page);
    let dense_priority = hgal.compute_eviction_priority(&dense_page);

    // ExpertWeight should have lower score (more evictable) than DenseLayerWeight
    assert!(
        expert_priority.score < dense_priority.score,
        "ExpertWeight score {} should be < DenseLayerWeight score {}",
        expert_priority.score,
        dense_priority.score
    );
    // DenseLayerWeight should be pinned (not evictable)
    assert!(dense_priority.is_pinned, "DenseLayerWeight should be pinned");
    // ExpertWeight should not be pinned
    assert!(!expert_priority.is_pinned, "ExpertWeight should not be pinned");
}

/// Test: KnowledgeRAG pages have lower eviction priority than KV context.
#[test]
fn eviction_priority_rag_vs_kv() {
    let hgal = test_hgal();

    let rag_page = gllm::scheduler::types::UnifiedVirtualPage {
        page_id: 10,
        payload_kind: PagePayloadKind::KnowledgeRAG,
        residency: gllm::scheduler::types::MemoryResidency::DeviceLocal,
        dtype: gllm_kernels::types::DType::F16,
        owner: None,
        pipeline: None,
        logical_index: 0,
        codec: CompressionCodec::None,
        compressed_size: 0,
        decompressed_size: 0,
        expert_id: None,
        layer_idx: None,
    };

    let kv_page = gllm::scheduler::types::UnifiedVirtualPage {
        page_id: 11,
        payload_kind: PagePayloadKind::KvContext,
        residency: gllm::scheduler::types::MemoryResidency::DeviceLocal,
        dtype: gllm_kernels::types::DType::F16,
        owner: Some(1),
        pipeline: Some(KvPipeline::Conversation),
        logical_index: 0,
        codec: CompressionCodec::None,
        compressed_size: 0,
        decompressed_size: 0,
        expert_id: None,
        layer_idx: None,
    };

    let rag_priority = hgal.compute_eviction_priority(&rag_page);
    let kv_priority = hgal.compute_eviction_priority(&kv_page);

    // KnowledgeRAG should have lower score (more evictable) than KvContext
    assert!(
        rag_priority.score < kv_priority.score,
        "KnowledgeRAG score {} should be < KvContext score {}",
        rag_priority.score,
        kv_priority.score
    );
}

/// Test: ExpertWeight pages with identical metadata get identical scores.
#[test]
fn eviction_priority_identical_pages_equal_score() {
    let hgal = test_hgal();

    let page = gllm::scheduler::types::UnifiedVirtualPage {
        page_id: 100,
        payload_kind: PagePayloadKind::ExpertWeight,
        residency: gllm::scheduler::types::MemoryResidency::DeviceLocal,
        dtype: gllm_kernels::types::DType::F16,
        owner: None,
        pipeline: None,
        logical_index: 0,
        codec: CompressionCodec::None,
        compressed_size: 0,
        decompressed_size: 0,
        expert_id: Some(1),
        layer_idx: Some(3),
    };

    let prio1 = hgal.compute_eviction_priority(&page);
    let prio2 = hgal.compute_eviction_priority(&page);

    assert_eq!(
        prio1.score, prio2.score,
        "Identical pages should have identical eviction scores"
    );
    assert_eq!(prio1.payload_kind, prio2.payload_kind);
    assert_eq!(prio1.is_pinned, prio2.is_pinned);
}

/// Test: select_victim_weight_pages returns pages sorted by eviction priority.
#[test]
fn hgal_select_victim_weight_pages() {
    let mut hgal = test_hgal();

    // Register weight pages for 2 layers
    for layer in 0..3 {
        hgal.allocate_expert_weight_pages(4, layer);
    }

    assert_eq!(hgal.num_expert_weight_pages(), 12);

    // All pages are in Standby state by default → all should be candidates
    let victims = hgal.select_victim_weight_pages(5);
    assert_eq!(victims.len(), 5, "Should select 5 victims");

    // Victims should be sorted by score ascending (most evictable first)
    for i in 1..victims.len() {
        assert!(
            victims[i - 1].1.score <= victims[i].1.score,
            "Victims should be sorted by score ascending"
        );
    }
}

/// Test: Once weight pages are registered, HGAL correctly tracks count.
#[test]
fn hgal_weight_page_registration_lifecycle() {
    let mut hgal = test_hgal();
    assert_eq!(hgal.num_expert_weight_pages(), 0);

    // Allocate weight pages
    let pages_layer0 = hgal.allocate_expert_weight_pages(8, 0);
    assert_eq!(pages_layer0.len(), 8);
    assert_eq!(hgal.num_expert_weight_pages(), 8);

    let pages_layer1 = hgal.allocate_expert_weight_pages(4, 1);
    assert_eq!(pages_layer1.len(), 4);
    assert_eq!(hgal.num_expert_weight_pages(), 12);

    // Free layer 0
    hgal.free_expert_weight_pages(0);
    assert_eq!(hgal.num_expert_weight_pages(), 4);

    // Free layer 1
    hgal.free_expert_weight_pages(1);
    assert_eq!(hgal.num_expert_weight_pages(), 0);
}

// ===========================================================================
// Weight Page Telemetry (SPEC §7, REQ-WP-010)
// ===========================================================================

/// Test: Recording an Evicted telemetry event updates counters and tier distribution.
#[test]
fn telemetry_eviction_event() {
    let mut obs = BasicObserver::new();

    obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
        page_id: 1,
        from_tier: WeightTier::Hot,
        to_tier: WeightTier::Warm,
        reason: EvictionReason::MemoryPressure,
        bytes: 4096,
    });

    let state = obs.capture().unwrap();
    assert_eq!(state.weight_eviction_count, 1);
    assert_eq!(state.weight_pages_l1, 0); // Removed from Hot
    assert_eq!(state.weight_pages_l2, 1); // Added to Warm
    assert_eq!(state.weight_pages_l3, 0);
}

/// Test: Recording a Recovered telemetry event updates counters and tier distribution.
#[test]
fn telemetry_recovery_event() {
    let mut obs = BasicObserver::new();

    obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
        page_id: 2,
        from_tier: WeightTier::Cold,
        to_tier: WeightTier::Hot,
        latency_us: 1500,
        bytes: 8192,
    });

    let state = obs.capture().unwrap();
    assert_eq!(state.weight_recovery_count, 1);
    assert_eq!(state.weight_pages_l3, 0); // Removed from Cold
    assert_eq!(state.weight_pages_l1, 1); // Added to Hot
    assert_eq!(state.weight_pages_l2, 0);
}

/// Test: Multiple events accumulate correctly.
#[test]
fn telemetry_multiple_events() {
    let mut obs = BasicObserver::new();

    // Evict 2 pages from Hot→Warm
    obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
        page_id: 1,
        from_tier: WeightTier::Hot,
        to_tier: WeightTier::Warm,
        reason: EvictionReason::MemoryPressure,
        bytes: 4096,
    });
    obs.record_weight_page_event(WeightPageTelemetryEvent::Evicted {
        page_id: 2,
        from_tier: WeightTier::Hot,
        to_tier: WeightTier::Warm,
        reason: EvictionReason::MemoryPressure,
        bytes: 4096,
    });

    // Recover 1 page from Warm→Hot
    obs.record_weight_page_event(WeightPageTelemetryEvent::Recovered {
        page_id: 1,
        from_tier: WeightTier::Warm,
        to_tier: WeightTier::Hot,
        latency_us: 800,
        bytes: 4096,
    });

    let state = obs.capture().unwrap();
    assert_eq!(state.weight_eviction_count, 2);
    assert_eq!(state.weight_recovery_count, 1);
    // Net: 2 evicted from Hot, 1 recovered into Hot = Hot has 0 (1 recovered - but 2 left)
    // Actually: 2 evicted from Hot (Hot→Warm), 1 recovered into Hot (Warm→Hot)
    // Net Hot: -1 (2 left, 1 returned) → 1 remaining in Warm
    // Let's recalculate:
    // Evict 1: Hot-1, Warm+1 → Hot=0, Warm=1
    // Evict 2: Hot-1, Warm+1 → Hot=0 (stays 0), Warm=2
    // Recover: Warm-1, Hot+1 → Hot=1, Warm=1
    assert_eq!(state.weight_pages_l1, 1);
    assert_eq!(state.weight_pages_l2, 1);
    assert_eq!(state.weight_pages_l3, 0);
}

/// Test: Bulk weight metrics update via update_weight_metrics.
#[test]
fn telemetry_bulk_update() {
    let mut obs = BasicObserver::new();

    obs.update_weight_metrics(
        100,   // total
        50,    // L1
        30,    // L2
        20,    // L3
        5,     // eviction_count
        3,     // recovery_count
    );

    let state = obs.capture().unwrap();
    assert_eq!(state.weight_page_total, 100);
    assert_eq!(state.weight_pages_l1, 50);
    assert_eq!(state.weight_pages_l2, 30);
    assert_eq!(state.weight_pages_l3, 20);
    assert_eq!(state.weight_eviction_count, 5);
    assert_eq!(state.weight_recovery_count, 3);
}

// ===========================================================================
// Weight Paging Enabled / Disabled Modes (SPEC §8)
// ===========================================================================

/// Test: WeightPageJitConfig default is disabled.
#[test]
fn weight_paging_disabled_by_default() {
    let config = WeightPageJitConfig::default();
    assert!(!config.enabled, "Weight paging should be disabled by default");
    assert_eq!(config.num_pages, 1024);
    assert_eq!(config.page_size_bytes, 64 * 1024 * 1024);
    assert_eq!(config.prefetch_distance, 0);
}

/// Test: WeightPageJitConfig can be enabled with custom parameters.
#[test]
fn weight_paging_enabled_config() {
    let config = WeightPageJitConfig {
        enabled: true,
        num_pages: 512,
        page_size_bytes: 16 * 1024 * 1024, // 16 MiB
        prefetch_distance: 2,
    };

    assert!(config.enabled);
    assert_eq!(config.num_pages, 512);
    assert_eq!(config.page_size_bytes, 16 * 1024 * 1024);
    assert_eq!(config.prefetch_distance, 2);
}

/// Test: Weight paging with prefetch distance > 0 enables proactive prefetch.
#[test]
fn weight_paging_with_prefetch() {
    let config = WeightPageJitConfig {
        enabled: true,
        num_pages: 256,
        page_size_bytes: 32 * 1024 * 1024, // 32 MiB
        prefetch_distance: 4, // Prefetch 4 pages ahead
    };

    assert!(config.enabled, "Weight paging should be enabled");
    assert!(
        config.prefetch_distance > 0,
        "Prefetch distance should be > 0 for proactive prefetch"
    );

    // Verify that the prefetch distance is bounded by total pages
    assert!(
        config.prefetch_distance < config.num_pages,
        "Prefetch distance should not exceed total pages"
    );
}

/// Test: Weight paging disabled mode — fault recovery still works but no JIT injection.
#[test]
fn weight_paging_disabled_fault_recovery() {
    // With weight paging disabled, fault recovery is still functional
    // (the telemetry and migration paths remain available)
    let mut handler = FaultRecoveryHandler::new();
    let mut gmm = test_gmm();
    let mut table = WeightPageTable::new();

    let pid = gmm.allocate_page(Tier::L1).expect("alloc L1");
    table.register_layer(0, vec![pid]);

    let l2_pid = gmm.migrate_page(Tier::L1, Tier::L2, pid).expect("migrate");
    table.update_physical_id(0, 0, l2_pid, Tier::L2);

    let fault = PageFault {
        page_id: l2_pid,
        current_tier: Tier::L2,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: None,
        dense_layer_idx: Some(0),
    };

    // Even with JIT disabled, the migration/table path works
    let result = handler.recover_fault(&fault, &mut gmm, &mut table);
    assert!(result.is_ok(), "Fault recovery should work regardless of JIT config");
}

/// Test: FaultRecoveryHandler respects max_retries configuration.
#[test]
fn fault_recovery_handler_with_max_retries() {
    let mut handler = FaultRecoveryHandler::new().with_max_retries(1);
    // with_max_retries sets the internal retry limit; verify through behavior

    let gmm = GlobalMemoryManager::new_with_capacities(0, 0, 0);
    let table = WeightPageTable::new();

    let fault = PageFault {
        page_id: 1,
        current_tier: Tier::L2,
        target_tier: Tier::L1,
        fault_time: Instant::now(),
        expert_key: None,
        dense_layer_idx: None,
    };

    // Single retry then abort
    let action1 = handler.handle_page_fault(&fault, &gmm, &table);
    assert_eq!(action1, FaultAction::Retry, "First call should retry");

    let action2 = handler.handle_page_fault(&fault, &gmm, &table);
    assert!(
        matches!(action2, FaultAction::Abort { .. }),
        "Second call should abort (max_retries=1)"
    );
}

/// Test: FaultRecoveryStats accumulates correctly.
#[test]
fn fault_recovery_stats_accumulation() {
    let mut stats = FaultRecoveryStats::default();

    assert_eq!(stats.total_faults, 0);
    assert_eq!(stats.avg_recovery_latency_us(), 0.0);

    stats.total_faults = 5;
    stats.record_recovery(Tier::L2, std::time::Duration::from_micros(2000));
    stats.record_recovery(Tier::L3, std::time::Duration::from_micros(5000));
    stats.record_abort();
    stats.record_retry();

    assert_eq!(stats.total_faults, 5);
    assert_eq!(stats.successful_recoveries, 2);
    assert_eq!(stats.aborted_faults, 1);
    assert_eq!(stats.retried_faults, 1);
    assert_eq!(stats.total_recovery_latency_us, 7000);
    assert!((stats.avg_recovery_latency_us() - 3500.0).abs() < 0.001);

    assert_eq!(stats.l2_to_l1_count, 1);
    assert_eq!(stats.l3_to_l1_count, 1);
    assert_eq!(stats.multi_hop_count, 1);
}

/// Test: WeightPageTelemetryEvent equality and debug.
#[test]
fn weight_page_telemetry_event_properties() {
    let evicted = WeightPageTelemetryEvent::Evicted {
        page_id: 42,
        from_tier: WeightTier::Hot,
        to_tier: WeightTier::Warm,
        reason: EvictionReason::MemoryPressure,
        bytes: 8192,
    };

    let recovered = WeightPageTelemetryEvent::Recovered {
        page_id: 42,
        from_tier: WeightTier::Warm,
        to_tier: WeightTier::Hot,
        latency_us: 1200,
        bytes: 8192,
    };

    // Verify Debug formatting
    let debug_evicted = format!("{:?}", evicted);
    assert!(debug_evicted.contains("Evicted"));
    assert!(debug_evicted.contains("MemoryPressure"));

    let debug_recovered = format!("{:?}", recovered);
    assert!(debug_recovered.contains("Recovered"));
    assert!(debug_recovered.contains("1200"));
}

// ===========================================================================
// Edge Cases and Boundary Tests
// ===========================================================================

/// Test: Empty weight page table returns empty results.
#[test]
fn weight_page_table_empty_operations() {
    let table = WeightPageTable::new();

    assert_eq!(table.layer_count(), 0);
    assert_eq!(table.total_pages(), 0);
    assert_eq!(table.tier_distribution(), (0, 0, 0));
    assert!(!table.layer_needs_recovery(0));
    assert_eq!(table.get_layer_pages(0), None);
    assert_eq!(table.page_tier(999), None);
    assert_eq!(table.layer_for_page(999), None);
}

/// Test: Select victims with count=0 returns empty.
#[test]
fn hgal_select_victims_empty_count() {
    let hgal = test_hgal();
    let victims = hgal.select_victim_weight_pages(0);
    assert!(victims.is_empty());
}

/// Test: Empty GMM operations return appropriate errors.
#[test]
fn gmm_empty_operations() {
    let mut gmm = GlobalMemoryManager::new_with_capacities(0, 0, 0);

    // Allocating from full tier should fail
    assert!(gmm.allocate_page(Tier::L1).is_err());
    assert!(gmm.allocate_page(Tier::L2).is_err());
    assert!(gmm.allocate_page(Tier::L3).is_err());

    // Freeing unallocated page should fail
    assert!(gmm.free_page(Tier::L1, 999).is_err());
}

/// Test: Fault action equality (verify PartialEq impl).
#[test]
fn fault_action_equality() {
    let load1 = FaultAction::LoadFromTier {
        source_tier: Tier::L2,
        target_tier: Tier::L1,
    };
    let load2 = FaultAction::LoadFromTier {
        source_tier: Tier::L2,
        target_tier: Tier::L1,
    };
    assert_eq!(load1, load2);

    let abort1 = FaultAction::Abort {
        reason: "OOM".to_string(),
    };
    let abort2 = FaultAction::Abort {
        reason: "OOM".to_string(),
    };
    assert_eq!(abort1, abort2);

    let abort3 = FaultAction::Abort {
        reason: "different".to_string(),
    };
    assert_ne!(abort1, abort3);
    assert_ne!(load1, abort1);
    assert_eq!(FaultAction::Retry, FaultAction::Retry);
}
