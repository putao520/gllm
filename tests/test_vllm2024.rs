use gllm::scheduler::vllm2024::{
    CacheLevel, L3Backend, LMCacheConfig, LmcacheState, SwiftKVConfig, SwiftKvState,
};
use gllm_kernels::backend_trait::Backend;
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::kernel_types::KvCacheConfig;

/// TEST-VLLM-001: LMCache 存储和复用 handles
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建 LMCache 状态
/// 2. 存储 KV handle
/// 3. 获取缓存的 handle
///
/// **期望结果**: 缓存命中，返回正确的 handle
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

/// TEST-VLLM-002: SwiftKV SiKV 减少页面
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建 SwiftKV 状态
/// 2. 执行 distill_cpu()
/// 3. 验证蒸馏后页面数减少
///
/// **期望结果**: 页面数减少，无需精度回退
#[test]
fn swift_kv_sikv_reduces_pages() {
    let mut swift = SwiftKvState::new(SwiftKVConfig {
        enabled: true,
        window_size: 2,
        enable_across_kv: false,
        similarity_threshold: 0.9,
        precision_guard: 2.0,
    });
    let pages = vec![
        vec![1.0f32, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
    ];
    let outcome = swift.distill_cpu(&pages);
    assert_eq!(outcome.result.distilled_pages, 2);
    assert!(!outcome.precision_fallback);
}

/// TEST-VLLM-003: SwiftKV AKV 共享相似层
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 启用 across_kv 模式
/// 2. 执行 distill_cpu()
/// 3. 验证相似层被合并
///
/// **期望结果**: 相似层被合并，页面数减少
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

/// TEST-VLLM-004: SwiftKV 精度保护回退
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 边界测试
///
/// **测试步骤**:
/// 1. 设置严格的精度保护阈值
/// 2. 执行 distill_cpu()
/// 3. 验证精度回退
///
/// **期望结果**: 精度不足时触发回退
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
