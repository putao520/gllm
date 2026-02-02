use gllm::engine::vllm2024::{
    CacheLevel, L3Backend, LMCacheConfig, LmcacheState, SwiftKVConfig, SwiftKvState,
};
use gllm_kernels::backend_trait::Backend;
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::kernel_types::KvCacheConfig;

#[test]
fn lmcache_stores_and_reuses_handles() {
    let mut state = LmcacheState::new(LMCacheConfig {
        l1_capacity_mb: 1,
        l2_capacity_mb: 1,
        l3_backend: L3Backend::Disabled,
        cache_prefix_len: 16,
    });
    let key = LmcacheState::cache_key("model", "hello world", 16);
    let backend = CpuBackend::new();
    let kv = backend
        .alloc_kv_cache(&KvCacheConfig {
            num_layers: 1,
            num_heads: 1,
            head_dim: 1,
            max_seq_len: 8,
            dtype_size: 4,
            page_size: 8,
            swap_config: None,
        })
        .expect("alloc kv");

    state.put(key.clone(), 8, Some(kv), None);

    let hit = state.get(&key).expect("cache hit");
    assert_eq!(hit.level, CacheLevel::L1);
    assert_eq!(hit.kv_handle, Some(kv));
    assert_eq!(hit.logits_handle, None);
}

#[test]
fn swift_kv_sikv_reduces_pages() {
    let mut swift = SwiftKvState::new(SwiftKVConfig {
        enabled: true,
        window_size: 2,
        enable_across_kv: false,
        similarity_threshold: 0.9,
        precision_guard: 2.0,
    });
    let pages = vec![vec![1.0f32, 0.0], vec![0.0, 1.0], vec![1.0, 1.0], vec![0.5, 0.5]];
    let outcome = swift.distill_cpu(&pages);
    assert_eq!(outcome.result.distilled_pages, 2);
    assert!(!outcome.precision_fallback);
}

#[test]
fn swift_kv_akv_shares_similar_layers() {
    let mut swift = SwiftKvState::new(SwiftKVConfig {
        enabled: true,
        window_size: 1,
        enable_across_kv: true,
        similarity_threshold: 0.8,
        precision_guard: 2.0,
    });
    let pages = vec![vec![1.0f32, 1.0], vec![1.0f32, 1.0]];
    let outcome = swift.distill_cpu(&pages);
    assert_eq!(outcome.result.distilled_pages, 1);
}

#[test]
fn swift_kv_precision_guard_fallbacks() {
    let mut swift = SwiftKvState::new(SwiftKVConfig {
        enabled: true,
        window_size: 2,
        enable_across_kv: false,
        similarity_threshold: 0.9,
        precision_guard: 0.01, // very strict
    });
    let pages = vec![vec![1.0f32, 0.0], vec![-1.0f32, 0.0]];
    let outcome = swift.distill_cpu(&pages);
    assert!(outcome.precision_fallback);
    assert_eq!(outcome.result.distilled_pages, pages.len());
}
