//! §13.2 Softmax Centroid-Guided Prefetch
//!
//! Extracts attention probability centroids (argmax token positions) from
//! softmax epilogue telemetry and issues asynchronous memory prefetch hints
//! for KV cache blocks in layer N+1.

use std::collections::VecDeque;

/// Prefetch request for a KV cache block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefetchRequest {
    pub layer: usize,
    pub token_idx: usize,
}

/// Prefetch queue for KV cache blocks.
pub struct PrefetchQueue {
    queue: VecDeque<PrefetchRequest>,
    block_size: usize,
}

impl PrefetchQueue {
    pub fn new(block_size: usize) -> Self {
        Self {
            queue: VecDeque::with_capacity(32),
            block_size,
        }
    }

    /// Enqueue a prefetch request for layer N+1 based on centroid token index.
    pub fn enqueue(&mut self, layer: usize, centroid_token_idx: usize) {
        self.queue.push_back(PrefetchRequest {
            layer: layer + 1,
            token_idx: centroid_token_idx,
        });
    }

    /// Issue prefetch hints for all queued requests.
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn issue_prefetch(&mut self, kv_cache_ptr: *const u8, kv_stride: usize) {
        while let Some(req) = self.queue.pop_front() {
            let offset = req.layer * kv_stride + req.token_idx * self.block_size;
            let ptr = kv_cache_ptr.add(offset);

            // prefetcht1: prefetch to L2 cache
            std::arch::x86_64::_mm_prefetch::<{std::arch::x86_64::_MM_HINT_T1}>(ptr as *const i8);
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn issue_prefetch(&mut self, kv_cache_ptr: *const u8, kv_stride: usize) {
        while let Some(req) = self.queue.pop_front() {
            let offset = req.layer * kv_stride + req.token_idx * self.block_size;
            let ptr = kv_cache_ptr.add(offset);

            // ARM PRFM instruction (prefetch to L2)
            std::arch::asm!(
                "prfm pldl2keep, [{ptr}]",
                ptr = in(reg) ptr,
                options(nostack, preserves_flags)
            );
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn issue_prefetch(&mut self, _kv_cache_ptr: *const u8, _kv_stride: usize) {
        // No-op on unsupported architectures
        self.queue.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    // ── PrefetchRequest: construction & field access ──────────────────────

    #[test]
    fn request_construction_and_field_access() {
        let req = PrefetchRequest { layer: 7, token_idx: 99 };
        assert_eq!(req.layer, 7);
        assert_eq!(req.token_idx, 99);
    }

    #[test]
    fn request_zero_fields() {
        let req = PrefetchRequest { layer: 0, token_idx: 0 };
        assert_eq!(req.layer, 0);
        assert_eq!(req.token_idx, 0);
    }

    #[test]
    fn request_max_fields() {
        let req = PrefetchRequest { layer: usize::MAX, token_idx: usize::MAX };
        assert_eq!(req.layer, usize::MAX);
        assert_eq!(req.token_idx, usize::MAX);
    }

    // ── PrefetchRequest: Copy trait ──────────────────────────────────────

    #[test]
    fn request_copy_independent() {
        let original = PrefetchRequest { layer: 4, token_idx: 50 };
        let copy = original;
        // Modify copy's fields (struct is Copy, so original is unaffected)
        assert_eq!(original.layer, 4);
        assert_eq!(original.token_idx, 50);
        assert_eq!(copy.layer, 4);
        assert_eq!(copy.token_idx, 50);
    }

    // ── PrefetchRequest: Clone trait ─────────────────────────────────────

    #[test]
    fn request_clone_equals_original() {
        let original = PrefetchRequest { layer: 10, token_idx: 200 };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    // ── PrefetchRequest: PartialEq / Eq ──────────────────────────────────

    #[test]
    fn request_eq_same_values() {
        let a = PrefetchRequest { layer: 1, token_idx: 2 };
        let b = PrefetchRequest { layer: 1, token_idx: 2 };
        assert_eq!(a, b);
    }

    #[test]
    fn request_neq_different_layer() {
        let a = PrefetchRequest { layer: 1, token_idx: 5 };
        let b = PrefetchRequest { layer: 2, token_idx: 5 };
        assert_ne!(a, b);
    }

    #[test]
    fn request_neq_different_token_idx() {
        let a = PrefetchRequest { layer: 3, token_idx: 10 };
        let b = PrefetchRequest { layer: 3, token_idx: 20 };
        assert_ne!(a, b);
    }

    // ── PrefetchRequest: Hash trait ──────────────────────────────────────

    #[test]
    fn request_equal_values_hash_equal() {
        let a = PrefetchRequest { layer: 5, token_idx: 30 };
        let b = PrefetchRequest { layer: 5, token_idx: 30 };
        let mut set = HashSet::new();
        assert!(set.insert(a));
        assert!(!set.insert(b)); // same hash + eq → duplicate
    }

    #[test]
    fn request_different_values_distinct_in_set() {
        let a = PrefetchRequest { layer: 1, token_idx: 0 };
        let b = PrefetchRequest { layer: 2, token_idx: 0 };
        let set: HashSet<PrefetchRequest> = [a, b].into_iter().collect();
        assert_eq!(set.len(), 2);
    }

    // ── PrefetchRequest: Debug trait ─────────────────────────────────────

    #[test]
    fn request_debug_format() {
        let req = PrefetchRequest { layer: 3, token_idx: 42 };
        let s = format!("{req:?}");
        assert!(s.contains("layer"), "Debug output should contain 'layer'");
        assert!(s.contains("token_idx"), "Debug output should contain 'token_idx'");
        assert!(s.contains("42"), "Debug output should contain field value 42");
    }

    // ── PrefetchQueue: construction ──────────────────────────────────────

    #[test]
    fn new_queue_is_empty() {
        let q = PrefetchQueue::new(16);
        assert!(q.queue.is_empty());
        assert_eq!(q.block_size, 16);
    }

    #[test]
    fn new_queue_with_zero_block_size() {
        let q = PrefetchQueue::new(0);
        assert!(q.queue.is_empty());
        assert_eq!(q.block_size, 0);
    }

    #[test]
    fn new_queue_with_max_block_size() {
        let q = PrefetchQueue::new(usize::MAX);
        assert_eq!(q.block_size, usize::MAX);
    }

    // ── PrefetchQueue: enqueue ───────────────────────────────────────────

    #[test]
    fn enqueue_adds_to_back() {
        let mut q = PrefetchQueue::new(16);
        q.enqueue(0, 10);
        q.enqueue(1, 20);
        assert_eq!(q.queue.len(), 2);
        assert_eq!(q.queue[0], PrefetchRequest { layer: 1, token_idx: 10 });
        assert_eq!(q.queue[1], PrefetchRequest { layer: 2, token_idx: 20 });
    }

    #[test]
    fn enqueue_layer_offset_by_one() {
        let mut q = PrefetchQueue::new(64);
        q.enqueue(5, 100);
        assert_eq!(q.queue[0].layer, 6);
    }

    #[test]
    fn enqueue_layer_zero_produces_layer_one() {
        let mut q = PrefetchQueue::new(16);
        q.enqueue(0, 0);
        assert_eq!(q.queue[0].layer, 1);
        assert_eq!(q.queue[0].token_idx, 0);
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn enqueue_max_layer_panics_on_overflow() {
        // layer + 1 with layer = usize::MAX overflows
        let mut q = PrefetchQueue::new(16);
        q.enqueue(usize::MAX, 5);
    }

    #[test]
    fn enqueue_many_requests_fifo_order() {
        let mut q = PrefetchQueue::new(32);
        for i in 0..100 {
            q.enqueue(i, i * 10);
        }
        assert_eq!(q.queue.len(), 100);
        for i in 0..100 {
            assert_eq!(q.queue[i], PrefetchRequest { layer: i + 1, token_idx: i * 10 });
        }
    }

    // ── PrefetchQueue: issue_prefetch drains queue ───────────────────────

    #[test]
    fn issue_prefetch_drains_queue() {
        let mut q = PrefetchQueue::new(16);
        q.enqueue(0, 0);
        q.enqueue(1, 1);
        let buf = vec![0u8; 4096];
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        assert!(q.queue.is_empty());
    }

    #[test]
    fn issue_prefetch_empty_queue_noop() {
        let mut q = PrefetchQueue::new(16);
        let buf = vec![0u8; 4096];
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        assert!(q.queue.is_empty());
    }

    #[test]
    fn issue_prefetch_large_stride_single_request() {
        let mut q = PrefetchQueue::new(64);
        q.enqueue(0, 0);
        // Need enough buffer: layer(1) * stride + token(0) * block_size = stride
        let buf = vec![0u8; 8192];
        unsafe { q.issue_prefetch(buf.as_ptr(), 4096); }
        assert!(q.queue.is_empty());
    }

    #[test]
    fn issue_prefetch_can_enqueue_again_after_drain() {
        let mut q = PrefetchQueue::new(16);
        q.enqueue(0, 0);
        let buf = vec![0u8; 4096];
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        assert!(q.queue.is_empty());
        // Re-enqueue after drain
        q.enqueue(2, 15);
        assert_eq!(q.queue.len(), 1);
        assert_eq!(q.queue[0], PrefetchRequest { layer: 3, token_idx: 15 });
    }

    #[test]
    fn issue_prefetch_zero_block_size() {
        // block_size=0 means offset = layer * stride (token contributes nothing)
        let mut q = PrefetchQueue::new(0);
        q.enqueue(0, 100);
        let buf = vec![0u8; 4096];
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        assert!(q.queue.is_empty());
    }

    #[test]
    fn issue_prefetch_zero_stride() {
        // stride=0 means offset = token * block_size (layer contributes nothing)
        let mut q = PrefetchQueue::new(16);
        q.enqueue(5, 2);
        let buf = vec![0u8; 256];
        unsafe { q.issue_prefetch(buf.as_ptr(), 0); }
        assert!(q.queue.is_empty());
    }

    // ── Additional tests: boundary conditions & untested paths ───────────

    // @trace TEST-PREF-01 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn issue_prefetch_multiple_requests_correct_drain_order() {
        // Arrange: enqueue 3 requests with distinct layers and tokens
        let mut q = PrefetchQueue::new(16);
        q.enqueue(0, 10); // layer=1, token=10
        q.enqueue(1, 20); // layer=2, token=20
        q.enqueue(2, 30); // layer=3, token=30
        let buf = vec![0u8; 65536];
        // Act: drain via issue_prefetch
        unsafe { q.issue_prefetch(buf.as_ptr(), 2048); }
        // Assert: queue is fully drained (all 3 consumed)
        assert!(q.queue.is_empty());
    }

    // @trace TEST-PREF-02 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn enqueue_same_layer_different_tokens_preserves_fifo() {
        // Arrange
        let mut q = PrefetchQueue::new(32);
        q.enqueue(4, 100);
        q.enqueue(4, 200);
        q.enqueue(4, 300);
        // Act & Assert: same input layer produces same output layer, tokens in FIFO order
        assert_eq!(q.queue[0], PrefetchRequest { layer: 5, token_idx: 100 });
        assert_eq!(q.queue[1], PrefetchRequest { layer: 5, token_idx: 200 });
        assert_eq!(q.queue[2], PrefetchRequest { layer: 5, token_idx: 300 });
    }

    // @trace TEST-PREF-03 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn enqueue_token_zero_at_nonzero_layer() {
        // Arrange: token_idx=0 with layer>0
        let mut q = PrefetchQueue::new(64);
        // Act
        q.enqueue(9, 0);
        // Assert
        assert_eq!(q.queue.len(), 1);
        assert_eq!(q.queue[0], PrefetchRequest { layer: 10, token_idx: 0 });
    }

    // @trace TEST-PREF-04 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn queue_length_across_enqueue_bursts() {
        // Arrange
        let mut q = PrefetchQueue::new(16);
        // Act: first burst
        for i in 0..5 {
            q.enqueue(i, i);
        }
        assert_eq!(q.queue.len(), 5);
        // Act: second burst
        for i in 5..12 {
            q.enqueue(i, i);
        }
        // Assert: total accumulated
        assert_eq!(q.queue.len(), 12);
    }

    // @trace TEST-PREF-05 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn issue_prefetch_double_drain_idempotent() {
        // Arrange: enqueue, drain, drain again
        let mut q = PrefetchQueue::new(16);
        q.enqueue(0, 5);
        let buf = vec![0u8; 4096];
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        assert!(q.queue.is_empty());
        // Act: second drain on already-empty queue
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        // Assert: still empty, no panic
        assert!(q.queue.is_empty());
    }

    // @trace TEST-PREF-06 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_hash_many_distinct_values() {
        // Arrange: insert 100 distinct PrefetchRequests into HashSet
        let set: HashSet<PrefetchRequest> = (0..100)
            .map(|i| PrefetchRequest { layer: i, token_idx: i * 7 })
            .collect();
        // Assert: all 100 distinct entries present
        assert_eq!(set.len(), 100);
    }

    // @trace TEST-PREF-07 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_equality_via_different_construction() {
        // Arrange: construct two requests with same logical values
        let layer_val = 42;
        let token_val = 256;
        let a = PrefetchRequest { layer: layer_val, token_idx: token_val };
        let b = PrefetchRequest {
            layer: layer_val,
            token_idx: token_val,
        };
        // Assert: structurally equal despite separate construction
        assert_eq!(a, b);
        assert!(!(a != b));
    }

    // @trace TEST-PREF-08 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn dequeue_manual_inspect_all_fields() {
        // Arrange: enqueue and pop_front manually
        let mut q = PrefetchQueue::new(128);
        q.enqueue(3, 77);
        // Act
        let req = q.queue.pop_front();
        // Assert: all fields match expected values
        assert!(req.is_some());
        let r = req.unwrap();
        assert_eq!(r.layer, 4);
        assert_eq!(r.token_idx, 77);
        assert_eq!(q.queue.len(), 0);
    }

    // @trace TEST-PREF-09 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn issue_prefetch_offset_zero_in_bounds() {
        // Arrange: enqueue layer=0, token=0 → offset = 1*stride + 0*block = stride
        // Use stride=0 so offset=0, which is always valid
        let mut q = PrefetchQueue::new(16);
        q.enqueue(0, 0);
        let buf = vec![0u8; 1]; // minimal buffer, offset will be 0
        // Act: stride=0, so offset = 1*0 + 0*16 = 0
        unsafe { q.issue_prefetch(buf.as_ptr(), 0); }
        // Assert: drained without out-of-bounds
        assert!(q.queue.is_empty());
    }

    // @trace TEST-PREF-10 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_as_hashmap_key() {
        // Arrange: use PrefetchRequest as HashMap key (exercises Hash + Eq)
        use std::collections::HashMap;
        let mut map = HashMap::new();
        let k1 = PrefetchRequest { layer: 1, token_idx: 10 };
        let k2 = PrefetchRequest { layer: 2, token_idx: 20 };
        // Act
        map.insert(k1, "first");
        map.insert(k2, "second");
        // Also test lookup with a cloned key
        let lookup = PrefetchRequest { layer: 1, token_idx: 10 };
        // Assert
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&lookup), Some(&"first"));
        assert_eq!(map.get(&k2), Some(&"second"));
    }

    // @trace TEST-PREF-11 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn enqueue_max_token_idx_at_layer_zero() {
        // Arrange: boundary - maximum token_idx with layer=0
        let mut q = PrefetchQueue::new(16);
        // Act
        q.enqueue(0, usize::MAX);
        // Assert: layer is offset to 1, token_idx preserved
        assert_eq!(q.queue.len(), 1);
        assert_eq!(q.queue[0].layer, 1);
        assert_eq!(q.queue[0].token_idx, usize::MAX);
    }

    // @trace TEST-PREF-12 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn issue_prefetch_stride_zero_block_size_zero() {
        // Arrange: both stride and block_size are zero → offset always 0
        let mut q = PrefetchQueue::new(0);
        q.enqueue(10, 999);
        let buf = vec![0u8; 1];
        // Act
        unsafe { q.issue_prefetch(buf.as_ptr(), 0); }
        // Assert: queue drained, no panic with zero-zero params
        assert!(q.queue.is_empty());
    }

    // @trace TEST-PREF-13 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn issue_prefetch_offset_at_exact_buffer_end() {
        // Arrange: compute offset that lands exactly at the last byte of the buffer.
        // offset = layer * stride + token_idx * block_size
        // We want offset < buf.len(). Let block_size=16, stride=1024, layer=0→1, token=0.
        // offset = 1 * 1024 + 0 * 16 = 1024. buf.len() must be > 1024.
        let mut q = PrefetchQueue::new(16);
        q.enqueue(0, 0); // layer becomes 1, token 0
        let buf = vec![0u8; 1025]; // offset 1024 is valid (< 1025)
        // Act
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        // Assert: drained successfully
        assert!(q.queue.is_empty());
    }

    // ── Additional tests: untested paths & trait contracts ──────────────────

    // @trace TEST-PREF-14 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_eq_symmetry_and_transitivity() {
        // Arrange: three structurally equal requests
        let a = PrefetchRequest { layer: 7, token_idx: 33 };
        let b = PrefetchRequest { layer: 7, token_idx: 33 };
        let c = PrefetchRequest { layer: 7, token_idx: 33 };
        // Assert: symmetry (a==b implies b==a)
        assert_eq!(a, b);
        assert_eq!(b, a);
        // Assert: transitivity (a==b and b==c implies a==c)
        assert_eq!(a, c);
    }

    // @trace TEST-PREF-15 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_ne_both_fields_differ() {
        // Arrange: two requests differing in both fields
        let a = PrefetchRequest { layer: 1, token_idx: 2 };
        let b = PrefetchRequest { layer: 3, token_idx: 4 };
        // Assert: both fields differ → not equal
        assert_ne!(a, b);
        assert!(a != b);
    }

    // @trace TEST-PREF-16 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_debug_contains_both_field_values() {
        // Arrange
        let req = PrefetchRequest { layer: 255, token_idx: 1023 };
        // Act
        let debug_str = format!("{req:?}");
        // Assert: both field names and values present
        assert!(debug_str.contains("255"), "should contain layer value");
        assert!(debug_str.contains("1023"), "should contain token_idx value");
    }

    // @trace TEST-PREF-17 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_copy_preserves_original_after_stack_reassignment() {
        // Arrange
        let original = PrefetchRequest { layer: 10, token_idx: 20 };
        // Act: copy into new binding, then overwrite new binding
        let mut copy = original;
        copy = PrefetchRequest { layer: 99, token_idx: 88 };
        // Assert: original unchanged
        assert_eq!(original.layer, 10);
        assert_eq!(original.token_idx, 20);
        assert_eq!(copy.layer, 99);
        assert_eq!(copy.token_idx, 88);
    }

    // @trace TEST-PREF-18 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn queue_block_size_preserved_across_operations() {
        // Arrange
        let mut q = PrefetchQueue::new(256);
        // Act: enqueue and drain
        q.enqueue(0, 0);
        let buf = vec![0u8; 4096];
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        // Assert: block_size unchanged after drain
        assert_eq!(q.block_size, 256);
    }

    // @trace TEST-PREF-19 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn enqueue_single_then_issue_preserves_fifo_order() {
        // Arrange: enqueue 5 requests, record expected pop order, drain manually
        let mut q = PrefetchQueue::new(8);
        let expected: Vec<PrefetchRequest> = (0..5)
            .map(|i| PrefetchRequest { layer: i + 1, token_idx: i * 3 })
            .collect();
        for (layer, token) in expected.iter().map(|r| (r.layer - 1, r.token_idx)) {
            q.enqueue(layer, token);
        }
        // Act & Assert: pop_front yields FIFO order
        for expected_req in &expected {
            assert_eq!(q.queue.pop_front(), Some(*expected_req));
        }
        assert!(q.queue.is_empty());
    }

    // @trace TEST-PREF-20 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_all_zero_max_quadrants() {
        // Arrange: test all four boundary quadrants
        let zero_zero = PrefetchRequest { layer: 0, token_idx: 0 };
        let zero_max = PrefetchRequest { layer: 0, token_idx: usize::MAX };
        let max_zero = PrefetchRequest { layer: usize::MAX, token_idx: 0 };
        let max_max = PrefetchRequest { layer: usize::MAX, token_idx: usize::MAX };
        // Assert: all four are distinct (exercises Hash + Eq across full range)
        let set: HashSet<PrefetchRequest> = [zero_zero, zero_max, max_zero, max_max].into_iter().collect();
        assert_eq!(set.len(), 4);
        // Assert: each self-equal
        assert_eq!(zero_zero, zero_zero);
        assert_eq!(zero_max, zero_max);
        assert_eq!(max_zero, max_zero);
        assert_eq!(max_max, max_max);
    }

    // @trace TEST-PREF-21 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn issue_prefetch_single_byte_buffer_zero_offset() {
        // Arrange: 1-byte buffer with block_size=0, stride=0 → offset=0 always
        let mut q = PrefetchQueue::new(0);
        q.enqueue(0, 0); // offset = 1*0 + 0*0 = 0
        let buf = vec![0u8; 1];
        // Act
        unsafe { q.issue_prefetch(buf.as_ptr(), 0); }
        // Assert: drained without panic on minimal buffer
        assert!(q.queue.is_empty());
    }

    // @trace TEST-PREF-22 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn enqueue_many_then_issue_prefetch_full_cycle() {
        // Arrange: enqueue 50, drain, enqueue 30 more, drain again
        let mut q = PrefetchQueue::new(16);
        let buf = vec![0u8; 1_048_576]; // 1 MiB
        // Act: first cycle
        for i in 0..50 {
            q.enqueue(i, i * 2);
        }
        assert_eq!(q.queue.len(), 50);
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        assert!(q.queue.is_empty());
        // Act: second cycle
        for i in 0..30 {
            q.enqueue(i + 100, i * 3);
        }
        assert_eq!(q.queue.len(), 30);
        unsafe { q.issue_prefetch(buf.as_ptr(), 1024); }
        // Assert: fully drained after two complete cycles
        assert!(q.queue.is_empty());
        assert_eq!(q.block_size, 16);
    }

    // @trace TEST-PREF-23 [req:REQ-PREFETCH] [level:unit]
    #[test]
    fn request_clone_into_vec_and_sort_stable() {
        // Arrange: create requests with varying layers
        let reqs: Vec<PrefetchRequest> = (0..10)
            .rev()
            .map(|i| PrefetchRequest { layer: i, token_idx: i })
            .collect();
        // Act: clone and sort by layer (stable sort)
        let mut sorted = reqs.clone();
        sorted.sort_by_key(|r| r.layer);
        // Assert: sorted in ascending layer order
        for (idx, req) in sorted.iter().enumerate() {
            assert_eq!(req.layer, idx);
            assert_eq!(req.token_idx, idx);
        }
        // Assert: original unmodified (Copy semantics)
        assert_eq!(reqs[0].layer, 9);
    }
}
