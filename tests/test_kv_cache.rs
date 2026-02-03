use gllm::kv_cache::{KvCacheDoubleBuffer, KvCacheError, KvCacheSlot, KvCacheState};
use gllm_kernels::backend_trait::Backend;
use gllm_kernels::cpu_backend::CpuBackend;
use gllm_kernels::kernel_types::KvCacheConfig;

fn make_state(backend: &CpuBackend, max_seq_len: usize) -> KvCacheState {
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

#[test]
fn kv_cache_slot_flip() {
    assert_eq!(KvCacheSlot::Front.flip(), KvCacheSlot::Back);
    assert_eq!(KvCacheSlot::Back.flip(), KvCacheSlot::Front);
}

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
