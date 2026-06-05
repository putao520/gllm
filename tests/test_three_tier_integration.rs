//! ThreeTierSwapCoordinator Integration Tests (REQ-COMP-007/008/011/016)
//!
//! Validates the end-to-end data flow from page registration through
//! eviction scoring, swap-in urgency ordering, and HGAL state sync.

use gllm::scheduler::dma_helpers::CpuDmaBackendSized;
use gllm::scheduler::eviction_worker::EvictionWorkerConfig;
use gllm::scheduler::migration_actor::MigrationActorConfig;
use gllm::scheduler::swap_in_worker::SwapInWorkerConfig;
use gllm::scheduler::three_tier_swap::{
    ThreeTierSwapConfig, ThreeTierSwapCoordinator,
};
use gllm::scheduler::types::{PageId, PageMetadata, PageState};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

fn make_coordinator(page_bytes: usize, dir: &TempDir) -> ThreeTierSwapCoordinator {
    let backend: Arc<dyn gllm::scheduler::dma_helpers::DmaBackend> =
        Arc::new(CpuDmaBackendSized);
    let memory_manager = Arc::new(Mutex::new(
        gllm::scheduler::memory_manager::GlobalMemoryManager::new_with_capacities(
            1024 * 1024 * 1024,
            4 * 1024 * 1024 * 1024,
            100 * 1024 * 1024 * 1024,
        ),
    ));
    let observer = Arc::new(Mutex::new(
        gllm::scheduler::observer::BasicObserver::new(),
    ));
    let config = ThreeTierSwapConfig {
        eviction: EvictionWorkerConfig {
            page_bytes,
            hbm_pressure_threshold: 0.85,
            dram_pressure_threshold: 0.90,
            importance_threshold: 30,
            hbm_evict_age_ticks: 0,
            ..Default::default()
        },
        swap_in: SwapInWorkerConfig::default(),
        migration: MigrationActorConfig {
            nvme_swap_dir: dir.path().to_path_buf(),
            queue_capacity: 64,
            session_id: "test-session".to_string(),
            page_size: page_bytes,
            max_swap_pages: 4096,
        },
        auto_start: true,
    };
    ThreeTierSwapCoordinator::new(config, backend, memory_manager, observer)
}

fn make_page_metadata(page_id: PageId, state: PageState) -> PageMetadata {
    PageMetadata {
        page_id,
        sequence_id: Some(1),
        recency: 0,
        access_count: 1,
        last_access: std::time::Instant::now() - std::time::Duration::from_secs(5),
        swap_in_time: None,
        is_lir: false,
        state,
        warm_until: None,
    }
}

#[test]
fn coordinator_registers_pages_in_addr_table() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(4096, &dir);

    coord.register_page(1, Some(0x1000), 4096);
    coord.register_page(2, Some(0x2000), 4096);
    coord.register_page(3, None, 4096);

    let stats = coord.stats();
    assert_eq!(stats.pages_on_hbm, 2);
    assert_eq!(stats.pages_on_dram, 1);
}

#[test]
fn coordinator_registers_pages_from_hgal() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(4096, &dir);

    let mut hgal_pages = HashMap::new();
    for pid in 0..5 {
        hgal_pages.insert(pid, make_page_metadata(pid, PageState::Active));
    }
    coord.register_pages_from_hgal(&hgal_pages, 4096);

    let stats = coord.stats();
    assert!(stats.pages_on_dram + stats.pages_on_hbm >= 5);
}

#[test]
fn build_batch_identifies_swap_in_for_active_pages_not_in_hbm() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(4096, &dir);

    for pid in 0..3 {
        coord.register_page(pid, None, 4096);
    }
    let mut hgal_pages = HashMap::new();
    for pid in 0..3 {
        hgal_pages.insert(pid, make_page_metadata(pid, PageState::Warm));
    }
    coord.register_pages_from_hgal(&hgal_pages, 4096);

    let active_pages: Vec<PageId> = vec![0, 1, 2];
    let plan = coord.build_batch(&active_pages, 0.5);

    assert_eq!(plan.swap_in_requests.len(), 3, "all warm pages needed by active sequences should be queued for swap-in");
    for req in &plan.swap_in_requests {
        assert!(active_pages.contains(&req.page_id));
        assert!(req.urgency > 0.0, "swap-in urgency should be positive for demanded pages");
    }
}

#[test]
fn build_batch_identifies_eviction_candidates_under_pressure() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(4096, &dir);

    for pid in 0..10 {
        coord.register_page(pid, Some(0x1000 + pid as u64 * 4096), 4096);
    }
    let mut hgal_pages = HashMap::new();
    for pid in 0..10 {
        let mut meta = make_page_metadata(pid, PageState::Standby);
        meta.access_count = if pid < 5 { 100 } else { 1 };
        hgal_pages.insert(pid, meta);
    }
    coord.register_pages_from_hgal(&hgal_pages, 4096);

    let active_pages: Vec<PageId> = vec![0, 1, 2, 3, 4];
    let plan = coord.build_batch(&active_pages, 0.95);

    assert!(!plan.eviction_candidates.is_empty(), "high pressure should produce eviction candidates");
    for candidate in &plan.eviction_candidates {
        assert!(!active_pages.contains(&candidate.page_id), "active pages must never be evicted");
    }
}

#[test]
fn tier_changed_pages_detects_divergence() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(4096, &dir);

    coord.register_page(10, Some(0x5000), 4096);
    let mut hgal_pages = HashMap::new();
    hgal_pages.insert(10, make_page_metadata(10, PageState::Active));
    coord.register_pages_from_hgal(&hgal_pages, 4096);

    let changed = coord.tier_changed_pages();
    assert!(changed.is_empty(), "no divergence when addr_table and page_metadata agree");

    coord.release_page(10);
    coord.register_page(10, None, 4096);
    coord.register_pages_from_hgal(&hgal_pages, 4096);

    let changed = coord.tier_changed_pages();
    assert!(!changed.is_empty(), "page registered without gpu_ptr (DRAM) but metadata says Active (HBM) should diverge");
    assert_eq!(changed[0].0, 10);
}

#[test]
fn stats_track_tier_distribution() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(4096, &dir);

    for pid in 0..4 {
        coord.register_page(pid, Some(pid as u64 * 4096), 4096);
    }
    for pid in 4..7 {
        coord.register_page(pid, None, 4096);
    }

    let stats = coord.stats();
    assert_eq!(stats.pages_on_hbm, 4);
    assert_eq!(stats.pages_on_dram, 3);
}

#[test]
fn release_page_removes_from_tracking() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(4096, &dir);

    coord.register_page(42, Some(0xA000), 4096);
    let stats_before = coord.stats();
    assert!(stats_before.pages_on_hbm >= 1);

    coord.release_page(42);
    let stats_after = coord.stats();
    assert_eq!(stats_after.pages_on_hbm, stats_before.pages_on_hbm - 1);
}

#[test]
fn build_batch_returns_empty_plan_when_no_pages() {
    let dir = tempfile::tempdir().unwrap();
    let coord = make_coordinator(4096, &dir);

    let plan = coord.build_batch(&[], 0.0);
    assert!(plan.eviction_candidates.is_empty());
    assert!(plan.swap_in_requests.is_empty());
    assert!(plan.tier_migrations.is_empty());
}
