use gllm::kv_cache::{KvCacheDoubleBuffer, KvCacheError, KvCacheSlot, KvCacheState};
use gllm::compat::backend_trait::Backend;
use gllm::compat::CpuBackend;
use gllm::engine::KvCacheConfig;

fn make_state(backend: &CpuBackend<f32>, max_seq_len: usize) -> KvCacheState {
    let config = KvCacheConfig {
        num_layers: 1,
        num_heads: 1,
        head_dim: 1,
        max_seq_len,
        dtype_size: std::mem::size_of::<f32>(),
        page_size: 0,
        swap_config: None,
    };
    let handle = backend.alloc_kv_cache(&config).expect("alloc kv cache");
    KvCacheState::new(handle, config)
}

/// TEST-KVCACHE-001: KV Cache Slot 翻转
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 执行 Front.flip()
/// 2. 执行 Back.flip()
///
/// **期望结果**: Front 变 Back，Back 变 Front
#[test]
fn kv_cache_slot_flip() {
    assert_eq!(KvCacheSlot::Front.flip(), KvCacheSlot::Back);
    assert_eq!(KvCacheSlot::Back.flip(), KvCacheSlot::Front);
}

/// TEST-KVCACHE-002: KV Cache 状态推进和边界检查
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试 + 边界测试
///
/// **测试步骤**:
/// 1. 推进状态 (advance)
/// 2. 验证 used() 和 remaining()
/// 3. 尝试超额推进 (应返回错误)
/// 4. 重置状态
///
/// **期望结果**: 状态正确推进，超额推进返回 Exhausted 错误
#[test]
fn kv_cache_state_advances_and_bounds() {
    let backend = CpuBackend::new();
    let mut state = make_state(&backend, 4);
    assert_eq!(state.remaining(), 4);

    state.advance(2).expect("advance");
    assert_eq!(state.used(), 2);
    assert_eq!(state.remaining(), 2);

    state.advance(0).expect("zero advance");
    assert_eq!(state.used(), 2);

    let err = state.advance(3).expect_err("should exhaust");
    match err {
        KvCacheError::Exhausted {
            requested,
            available,
        } => {
            assert_eq!(requested, 3);
            assert_eq!(available, 2);
        }
    }
    assert_eq!(state.used(), 2);

    state.reset();
    assert_eq!(state.used(), 0);
    assert_eq!(state.remaining(), 4);
}

/// TEST-KVCACHE-003: KV Cache 双缓冲交换和重置
///
/// **关联需求**: REQ-TEST-005
/// **测试类型**: 正向测试
///
/// **测试步骤**:
/// 1. 创建双缓冲 (front/back)
/// 2. 验证初始状态
/// 3. 执行 swap()
/// 4. 验证交换后状态
/// 5. 执行 reset_all()
///
/// **期望结果**: 交换后 front/back 互换，重置后归零
#[test]
fn kv_cache_double_buffer_swap_and_reset() {
    let backend = CpuBackend::new();
    let mut front = make_state(&backend, 8);
    let mut back = make_state(&backend, 8);
    front.advance(1).expect("advance front");
    back.advance(2).expect("advance back");

    let mut buffer = KvCacheDoubleBuffer::new(front, back);
    assert_eq!(buffer.front().used(), 1);
    assert_eq!(buffer.back().used(), 2);
    assert_eq!(buffer.slot(KvCacheSlot::Front).used(), 1);
    assert_eq!(buffer.slot(KvCacheSlot::Back).used(), 2);

    buffer.swap();
    assert_eq!(buffer.front().used(), 2);
    assert_eq!(buffer.back().used(), 1);

    buffer.reset_all();
    assert_eq!(buffer.front().used(), 0);
    assert_eq!(buffer.back().used(), 0);
}
