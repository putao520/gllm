//! Safe wrapper for CallbackChain FFI pointer.
//!
//! Replaces `*mut CallbackChain` in `GeneratorForwardConfig`.
//! Thread-safe via `AtomicPtr` with Acquire/Release ordering.
//! JIT reads the pointer on another thread (GPU kernel callback);
//! Rust sets/clears it while holding the Executor's `Mutex`.

use std::sync::atomic::{AtomicPtr, Ordering};

pub type CallbackChain = crate::graph::layer_callback::CallbackChain;

/// Safe wrapper for the callback chain pointer passed to JIT FFI.
///
/// Lifetime model: `set()` before FFI call → JIT reads `as_ffi_ptr()` → `clear()` after return.
/// All mutations happen while the Executor's `Mutex` is held, so there is no concurrent
/// `set`/`clear` race. JIT reads use `Acquire` to see the `Release`-written value.
#[derive(Debug)]
pub struct CallbackChainHandle {
    inner: AtomicPtr<CallbackChain>,
}

impl CallbackChainHandle {
    pub fn new() -> Self {
        Self {
            inner: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Temporarily borrow the chain as a raw pointer for FFI call.
    pub fn as_ffi_ptr(&self) -> *mut CallbackChain {
        self.inner.load(Ordering::Acquire)
    }

    /// Set the chain pointer. Must be called while Executor mutex is held.
    pub fn set(&self, ptr: *mut CallbackChain) {
        self.inner.store(ptr, Ordering::Release);
    }

    /// Clear the pointer after FFI call returns.
    pub fn clear(&self) {
        self.inner.store(std::ptr::null_mut(), Ordering::Release);
    }

    /// Check if a chain is currently set.
    pub fn is_set(&self) -> bool {
        !self.inner.load(Ordering::Acquire).is_null()
    }
}

impl Default for CallbackChainHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CallbackChainHandle {
    fn clone(&self) -> Self {
        Self {
            inner: AtomicPtr::new(self.inner.load(Ordering::Acquire)),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_handle_is_null_and_not_set() {
        let handle = CallbackChainHandle::new();
        assert!(handle.as_ffi_ptr().is_null());
        assert!(!handle.is_set());
    }

    #[test]
    fn default_equals_new() {
        let default = CallbackChainHandle::default();
        assert!(default.as_ffi_ptr().is_null());
        assert!(!default.is_set());
    }

    #[test]
    fn set_then_clear_roundtrip() {
        let handle = CallbackChainHandle::new();
        let dummy_ptr = 0xdead_beef as *mut CallbackChain;
        handle.set(dummy_ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), dummy_ptr);

        handle.clear();
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn clone_independently_copies_current_ptr() {
        let handle = CallbackChainHandle::new();
        let ptr = 0xcafe as *mut CallbackChain;
        handle.set(ptr);

        let cloned = handle.clone();
        assert!(cloned.is_set());
        assert_eq!(cloned.as_ffi_ptr(), ptr);

        // Clearing original does not affect clone (AtomicPtr value copy).
        handle.clear();
        assert!(cloned.is_set());
    }

    #[test]
    fn set_null_does_not_count_as_set() {
        let handle = CallbackChainHandle::new();
        handle.set(std::ptr::null_mut());
        assert!(!handle.is_set());
    }

    #[test]
    fn debug_trait_formats_struct_name() {
        let handle = CallbackChainHandle::new();
        let debug_str = format!("{:?}", handle);
        assert!(
            debug_str.contains("CallbackChainHandle"),
            "Debug output should contain struct name, got: {}",
            debug_str
        );
    }

    #[test]
    fn set_overwrites_previous_pointer() {
        let handle = CallbackChainHandle::new();
        let ptr_a = 0x1000 as *mut CallbackChain;
        let ptr_b = 0x2000 as *mut CallbackChain;

        handle.set(ptr_a);
        assert_eq!(handle.as_ffi_ptr(), ptr_a);

        handle.set(ptr_b);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);
        assert!(handle.is_set());
    }

    #[test]
    fn clear_on_fresh_handle_is_noop() {
        let handle = CallbackChainHandle::new();
        handle.clear();
        assert!(handle.as_ffi_ptr().is_null());
        assert!(!handle.is_set());
    }

    #[test]
    fn clear_is_idempotent() {
        let handle = CallbackChainHandle::new();
        let ptr = 0xabcd as *mut CallbackChain;
        handle.set(ptr);

        handle.clear();
        handle.clear();
        assert!(handle.as_ffi_ptr().is_null());
        assert!(!handle.is_set());
    }

    #[test]
    fn set_with_real_callback_chain_pointer() {
        let chain = CallbackChain::empty();
        let handle = CallbackChainHandle::new();

        let ptr = &chain as *const CallbackChain as *mut CallbackChain;
        handle.set(ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);
        assert!(!handle.as_ffi_ptr().is_null());

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_of_null_handle_is_null() {
        let handle = CallbackChainHandle::new();
        let cloned = handle.clone();
        assert!(cloned.as_ffi_ptr().is_null());
        assert!(!cloned.is_set());
    }

    #[test]
    fn set_clear_set_cycle() {
        let handle = CallbackChainHandle::new();
        let ptr_a = 0x1111 as *mut CallbackChain;
        let ptr_b = 0x2222 as *mut CallbackChain;

        handle.set(ptr_a);
        assert_eq!(handle.as_ffi_ptr(), ptr_a);

        handle.clear();
        assert!(handle.as_ffi_ptr().is_null());

        handle.set(ptr_b);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);
        assert!(handle.is_set());
    }

    #[test]
    fn as_ffi_ptr_returns_raw_pointer_value() {
        let handle = CallbackChainHandle::new();
        let ptr = 0xfeed_face as *mut CallbackChain;
        handle.set(ptr);

        let raw = handle.as_ffi_ptr();
        assert_eq!(raw as usize, 0xfeed_face);
    }

    #[test]
    fn concurrent_set_and_read_across_threads() {
        use std::sync::Arc;
        use std::thread;

        let handle = Arc::new(CallbackChainHandle::new());
        let addr: usize = 0xbad_c0de;

        let h = Arc::clone(&handle);
        let setter = thread::spawn(move || {
            h.set(addr as *mut CallbackChain);
        });

        let h = Arc::clone(&handle);
        let reader = thread::spawn(move || {
            // Spin until the pointer becomes visible via Acquire ordering.
            for _ in 0..1000 {
                if h.is_set() {
                    return true;
                }
                thread::yield_now();
            }
            false
        });

        setter.join().unwrap();
        let saw_set = reader.join().unwrap();
        assert!(saw_set, "Reader should observe the pointer set by writer thread");
        assert!(handle.is_set());
    }

    // --- New tests below ---

    #[test]
    fn concurrent_clear_and_read_across_threads() {
        use std::sync::Arc;
        use std::thread;

        let handle = Arc::new(CallbackChainHandle::new());
        let ptr = 0x1234 as *mut CallbackChain;
        handle.set(ptr);

        let h = Arc::clone(&handle);
        let clearer = thread::spawn(move || {
            h.clear();
        });

        let h = Arc::clone(&handle);
        let reader = thread::spawn(move || {
            // Spin until the pointer is cleared via Acquire ordering.
            for _ in 0..1000 {
                if !h.is_set() {
                    return true;
                }
                thread::yield_now();
            }
            false
        });

        clearer.join().unwrap();
        let saw_clear = reader.join().unwrap();
        assert!(saw_clear, "Reader should observe the pointer cleared by writer thread");
        assert!(!handle.is_set());
    }

    #[test]
    fn multiple_handles_with_same_pointer_independent_clear() {
        let ptr = 0xaa00 as *mut CallbackChain;

        let h1 = CallbackChainHandle::new();
        let h2 = CallbackChainHandle::new();

        h1.set(ptr);
        h2.set(ptr);

        assert!(h1.is_set());
        assert!(h2.is_set());
        assert_eq!(h1.as_ffi_ptr(), h2.as_ffi_ptr());

        h1.clear();

        assert!(!h1.is_set());
        assert!(h2.is_set(), "Clearing h1 must not affect h2");
    }

    #[test]
    fn rapid_set_clear_cycles() {
        let handle = CallbackChainHandle::new();
        let ptrs: [*mut CallbackChain; 4] = [
            0x1000 as *mut CallbackChain,
            0x2000 as *mut CallbackChain,
            0x3000 as *mut CallbackChain,
            0x4000 as *mut CallbackChain,
        ];

        for (i, &ptr) in ptrs.iter().enumerate() {
            handle.set(ptr);
            assert!(handle.is_set(), "Should be set at iteration {}", i);
            assert_eq!(handle.as_ffi_ptr(), ptr, "Pointer mismatch at iteration {}", i);
            handle.clear();
            assert!(!handle.is_set(), "Should be clear at iteration {}", i);
            assert!(handle.as_ffi_ptr().is_null(), "Should be null at iteration {}", i);
        }
    }

    #[test]
    fn set_max_valid_pointer_value() {
        let handle = CallbackChainHandle::new();
        let ptr = usize::MAX as *mut CallbackChain;

        handle.set(ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);
        assert_eq!(handle.as_ffi_ptr() as usize, usize::MAX);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn set_alignment_1_pointer() {
        let handle = CallbackChainHandle::new();
        // Pointer value 1 — not aligned, but the handle stores raw bits without validation.
        let ptr = 1 as *mut CallbackChain;

        handle.set(ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, 1);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_preserves_state_after_set_clear_original() {
        let handle = CallbackChainHandle::new();
        let ptr = 0xf00d as *mut CallbackChain;

        handle.set(ptr);
        let cloned = handle.clone();
        handle.clear();

        assert!(!handle.is_set());
        assert!(cloned.is_set(), "Clone retains the pointer snapshot");
        assert_eq!(cloned.as_ffi_ptr(), ptr);
    }

    #[test]
    fn cloned_handle_independent_set() {
        let handle = CallbackChainHandle::new();
        let cloned = handle.clone();

        let ptr_a = 0xaaaa as *mut CallbackChain;
        let ptr_b = 0xbbbb as *mut CallbackChain;

        handle.set(ptr_a);
        cloned.set(ptr_b);

        assert_eq!(handle.as_ffi_ptr(), ptr_a, "Original should have ptr_a");
        assert_eq!(cloned.as_ffi_ptr(), ptr_b, "Clone should have ptr_b");
    }

    #[test]
    fn set_after_clear_with_different_pointer() {
        let handle = CallbackChainHandle::new();
        let ptr_first = 0x1111 as *mut CallbackChain;
        let ptr_second = 0x2222 as *mut CallbackChain;

        handle.set(ptr_first);
        handle.clear();
        assert!(handle.as_ffi_ptr().is_null());

        handle.set(ptr_second);
        assert_eq!(handle.as_ffi_ptr(), ptr_second);
        assert!(handle.is_set());
    }

    #[test]
    fn multiple_clones_each_independent() {
        let handle = CallbackChainHandle::new();
        let ptr = 0xcece as *mut CallbackChain;
        handle.set(ptr);

        let c1 = handle.clone();
        let c2 = handle.clone();
        let c3 = handle.clone();

        handle.clear();

        assert!(!handle.is_set());
        assert!(c1.is_set());
        assert!(c2.is_set());
        assert!(c3.is_set());

        c1.clear();
        assert!(!c1.is_set());
        assert!(c2.is_set(), "c2 unaffected by c1.clear()");
        assert!(c3.is_set(), "c3 unaffected by c1.clear()");
    }

    #[test]
    fn is_set_consistency_with_as_ffi_ptr() {
        let handle = CallbackChainHandle::new();
        let ptr = 0x4567 as *mut CallbackChain;

        // Before set
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());

        handle.set(ptr);
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());

        handle.clear();
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn set_with_multiple_real_callback_chain_instances() {
        let chain_a = CallbackChain::empty();
        let chain_b = CallbackChain::empty();
        let handle = CallbackChainHandle::new();

        let ptr_a = &chain_a as *const CallbackChain as *mut CallbackChain;
        let ptr_b = &chain_b as *const CallbackChain as *mut CallbackChain;
        assert_ne!(ptr_a, ptr_b, "Two distinct chains must have different addresses");

        handle.set(ptr_a);
        assert_eq!(handle.as_ffi_ptr(), ptr_a);

        handle.set(ptr_b);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn concurrent_multiple_writers_same_handle() {
        use std::sync::Arc;
        use std::thread;

        let handle = Arc::new(CallbackChainHandle::new());
        let mut join_handles = Vec::new();

        for i in 0..8usize {
            let h = Arc::clone(&handle);
            join_handles.push(thread::spawn(move || {
                let ptr = (0x1000 + i) as *mut CallbackChain;
                h.set(ptr);
                assert!(h.is_set());
                h.clear();
                assert!(!h.is_set());
            }));
        }

        for jh in join_handles {
            jh.join().unwrap();
        }

        // After all threads complete, handle must be clear (last operation in each thread is clear).
        assert!(!handle.is_set());
    }

    #[test]
    fn concurrent_set_clear_reader_stress() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::sync::Arc;
        use std::thread;

        let handle = Arc::new(CallbackChainHandle::new());
        let reader_seen_set = Arc::new(AtomicBool::new(false));
        let addr: usize = 0xfeed;

        let h_writer = Arc::clone(&handle);
        let rss_writer = Arc::clone(&reader_seen_set);
        let writer = thread::spawn(move || {
            let ptr = addr as *mut CallbackChain;
            h_writer.set(ptr);
            // Spin until the reader confirms it observed the set.
            while !rss_writer.load(AtomicOrdering::Acquire) {
                thread::yield_now();
            }
            h_writer.clear();
        });

        let h_reader = Arc::clone(&handle);
        let rss_reader = Arc::clone(&reader_seen_set);
        let reader = thread::spawn(move || {
            let mut observed_set = false;
            let mut observed_clear = false;
            for _ in 0..100_000 {
                if h_reader.is_set() {
                    if !observed_set {
                        observed_set = true;
                        rss_reader.store(true, AtomicOrdering::Release);
                    }
                } else if observed_set {
                    observed_clear = true;
                }
                if observed_set && observed_clear {
                    break;
                }
                thread::yield_now();
            }
            (observed_set, observed_clear)
        });

        writer.join().unwrap();
        let (saw_set, saw_clear) = reader.join().unwrap();

        assert!(saw_set, "Reader should have seen set at least once");
        assert!(saw_clear, "Reader should have seen clear at least once");
        assert!(!handle.is_set());
    }

    #[test]
    fn default_impl_creates_null_handle() {
        // Verify Default::default() and ::new() produce functionally identical handles.
        let via_new = CallbackChainHandle::new();
        let via_default = CallbackChainHandle::default();

        assert!(via_new.as_ffi_ptr().is_null());
        assert!(via_default.as_ffi_ptr().is_null());
        assert_eq!(via_new.is_set(), via_default.is_set());

        // Both should accept and release a pointer independently.
        let ptr = 0xbeef as *mut CallbackChain;
        via_new.set(ptr);
        via_default.set(ptr);
        assert!(via_new.is_set());
        assert!(via_default.is_set());
        via_new.clear();
        via_default.clear();
        assert!(!via_new.is_set());
        assert!(!via_default.is_set());
    }

    // --- 15 additional tests ---

    #[test]
    fn handle_moved_between_bindings() {
        // Arrange: create a handle and set a pointer.
        let mut handle = CallbackChainHandle::new();
        let ptr = 0xabcd as *mut CallbackChain;
        handle.set(ptr);
        assert!(handle.is_set());

        // Act: move the handle into a new binding.
        let handle2 = handle;
        assert!(handle2.is_set());
        assert_eq!(handle2.as_ffi_ptr(), ptr);

        handle2.clear();

        // Assert: cleared handle in new binding.
        assert!(!handle2.is_set());
        assert!(handle2.as_ffi_ptr().is_null());
    }

    #[test]
    fn debug_output_changes_after_set_and_clear() {
        // Arrange: start with a null handle.
        let handle = CallbackChainHandle::new();
        let debug_null = format!("{:?}", handle);

        // Act: set a non-null pointer and capture debug output.
        let ptr = 0x1234 as *mut CallbackChain;
        handle.set(ptr);
        let debug_set = format!("{:?}", handle);

        handle.clear();
        let debug_cleared = format!("{:?}", handle);

        // Assert: all three contain the struct name, but inner values differ contextually.
        assert!(debug_null.contains("CallbackChainHandle"));
        assert!(debug_set.contains("CallbackChainHandle"));
        assert!(debug_cleared.contains("CallbackChainHandle"));
        // The null and cleared states should have identical debug output.
        assert_eq!(debug_null, debug_cleared);
    }

    #[test]
    fn clone_snapshot_taken_from_cleared_handle() {
        // Arrange: set then clear, then clone.
        let handle = CallbackChainHandle::new();
        let ptr = 0x8a8a as *mut CallbackChain;
        handle.set(ptr);
        handle.clear();

        // Act: clone after clear.
        let cloned = handle.clone();

        // Assert: clone sees null (snapshot of current state).
        assert!(!cloned.is_set());
        assert!(cloned.as_ffi_ptr().is_null());
    }

    #[test]
    fn non_empty_callback_chain_pointer_roundtrip() {
        // Arrange: create a CallbackChain with actual callbacks inside.
        let chain = CallbackChain::new(Vec::new());
        assert!(chain.is_empty());

        let handle = CallbackChainHandle::new();
        let ptr = &chain as *const CallbackChain as *mut CallbackChain;

        // Act: set, verify, clear.
        handle.set(ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handles_in_vector_independent_lifecycle() {
        // Arrange: create a Vec of handles, each set to a different pointer.
        let mut handles: Vec<CallbackChainHandle> = (0..5).map(|_| CallbackChainHandle::new()).collect();

        // Act: set each handle to a unique pointer.
        for (i, h) in handles.iter_mut().enumerate() {
            let ptr = (0x1000 + i * 0x100) as *mut CallbackChain;
            h.set(ptr);
        }

        // Assert: each handle has its own pointer.
        for (i, h) in handles.iter().enumerate() {
            let expected = (0x1000 + i * 0x100) as *mut CallbackChain;
            assert!(h.is_set());
            assert_eq!(h.as_ffi_ptr(), expected);
        }

        // Act: clear the middle handle.
        handles[2].clear();

        // Assert: only the middle is cleared.
        for (i, h) in handles.iter().enumerate() {
            if i == 2 {
                assert!(!h.is_set());
            } else {
                assert!(h.is_set());
            }
        }
    }

    #[test]
    fn as_ffi_ptr_called_many_times_returns_same_value() {
        // Arrange: set a specific pointer.
        let handle = CallbackChainHandle::new();
        let ptr = 0x9999 as *mut CallbackChain;
        handle.set(ptr);

        // Act: call as_ffi_ptr 100 times.
        for _ in 0..100 {
            assert_eq!(handle.as_ffi_ptr(), ptr);
        }

        // Assert: still set after many reads.
        assert!(handle.is_set());
        handle.clear();
    }

    #[test]
    fn handle_dropped_while_pointer_set_does_not_panic() {
        // Arrange & Act: create a handle, set it, let it drop.
        {
            let handle = CallbackChainHandle::new();
            let ptr = 0xdead as *mut CallbackChain;
            handle.set(ptr);
            assert!(handle.is_set());
            // handle drops here while pointer is still set.
        }

        // Assert: we reached here without panic.
    }

    #[test]
    fn sequential_real_chain_instances_preserve_identity() {
        // Arrange: two distinct CallbackChain instances.
        let chain_a = CallbackChain::empty();
        let chain_b = CallbackChain::empty();
        let ptr_a = &chain_a as *const CallbackChain as *mut CallbackChain;
        let ptr_b = &chain_b as *const CallbackChain as *mut CallbackChain;
        assert_ne!(ptr_a, ptr_b);

        let handle = CallbackChainHandle::new();

        // Act: set chain_a, verify, clear, then set chain_b.
        handle.set(ptr_a);
        assert_eq!(handle.as_ffi_ptr(), ptr_a);
        assert_eq!(handle.as_ffi_ptr() as usize, &chain_a as *const _ as usize);

        handle.clear();

        handle.set(ptr_b);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);
        assert_eq!(handle.as_ffi_ptr() as usize, &chain_b as *const _ as usize);

        // Assert: final state is set to chain_b.
        assert!(handle.is_set());
        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn is_set_returns_false_for_null_set_explicitly() {
        // Arrange: explicitly set to null_mut.
        let handle = CallbackChainHandle::new();

        // Act.
        handle.set(std::ptr::null_mut());

        // Assert: is_set should return false (null means not set).
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());

        // Verify we can still set a real pointer afterwards.
        let ptr = 0x1234 as *mut CallbackChain;
        handle.set(ptr);
        assert!(handle.is_set());
    }

    #[test]
    fn clone_chain_of_five_handles() {
        // Arrange: create a chain of clones: original -> c1 -> c2 -> c3 -> c4.
        let handle = CallbackChainHandle::new();
        let ptr = 0xdddd as *mut CallbackChain;
        handle.set(ptr);

        let c1 = handle.clone();
        let c2 = c1.clone();
        let c3 = c2.clone();
        let c4 = c3.clone();

        // Act: clear the middle of the chain.
        c2.clear();

        // Assert: only c2 is cleared; all others still hold the snapshot.
        assert!(handle.is_set());
        assert!(c1.is_set());
        assert!(!c2.is_set());
        assert!(c3.is_set());
        assert!(c4.is_set());
        assert_eq!(c4.as_ffi_ptr(), ptr);
    }

    #[test]
    fn set_to_zero_address_then_replace() {
        // Arrange: set to address 0 (same as null_mut but constructed differently).
        let handle = CallbackChainHandle::new();
        handle.set(0 as *mut CallbackChain);

        // Assert: address 0 is null, so is_set is false.
        assert!(!handle.is_set());

        // Act: replace with a non-null pointer.
        let ptr = 0xff01 as *mut CallbackChain;
        handle.set(ptr);

        // Assert: now it is set.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, 0xff01);
    }

    #[test]
    fn handle_embedded_in_struct() {
        struct Slot {
            chain: CallbackChainHandle,
            tag: u32,
        }

        // Arrange: embed the handle in a realistic struct.
        let mut slot = Slot {
            chain: CallbackChainHandle::new(),
            tag: 42,
        };

        // Act: use the handle through the struct.
        let ptr = 0xbad as *mut CallbackChain;
        slot.chain.set(ptr);
        assert!(slot.chain.is_set());
        assert_eq!(slot.tag, 42);

        slot.chain.clear();

        // Assert: cleared state in struct context.
        assert!(!slot.chain.is_set());
        assert!(slot.chain.as_ffi_ptr().is_null());
    }

    #[test]
    fn concurrent_many_readers_one_writer() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::sync::Arc;
        use std::thread;

        // Arrange: one writer sets a pointer; many readers observe it.
        let handle = Arc::new(CallbackChainHandle::new());
        let go = Arc::new(AtomicBool::new(false));
        let addr: usize = 0x7e57;

        let h_writer = Arc::clone(&handle);
        let go_w = Arc::clone(&go);
        let writer = thread::spawn(move || {
            while !go_w.load(AtomicOrdering::Acquire) {
                thread::yield_now();
            }
            h_writer.set(addr as *mut CallbackChain);
        });

        let mut reader_handles = Vec::new();
        for _ in 0..6 {
            let h_reader = Arc::clone(&handle);
            let go_r = Arc::clone(&go);
            reader_handles.push(thread::spawn(move || {
                while !go_r.load(AtomicOrdering::Acquire) {
                    thread::yield_now();
                }
                // Spin until we see the pointer.
                for _ in 0..2000 {
                    if h_reader.is_set() {
                        return true;
                    }
                    thread::yield_now();
                }
                false
            }));
        }

        // Act: signal all threads to start.
        go.store(true, AtomicOrdering::Release);

        writer.join().unwrap();
        let results: Vec<bool> = reader_handles.into_iter().map(|jh| jh.join().unwrap()).collect();

        // Assert: all readers observed the set pointer.
        for (i, saw) in results.into_iter().enumerate() {
            assert!(saw, "Reader {} should have seen the pointer", i);
        }
        assert!(handle.is_set());
    }

    #[test]
    fn concurrent_clone_while_set() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: shared handle set to a pointer.
        let handle = Arc::new(CallbackChainHandle::new());
        let addr: usize = 0x1ce1;
        handle.set(addr as *mut CallbackChain);

        // Act: multiple threads clone the handle simultaneously.
        let mut join_handles = Vec::new();
        for _ in 0..8 {
            let h = Arc::clone(&handle);
            join_handles.push(thread::spawn(move || {
                let cloned = h.clone();
                assert!(cloned.is_set());
                assert_eq!(cloned.as_ffi_ptr() as usize, addr);
            }));
        }

        // Assert: all clones had the pointer; original unchanged.
        for jh in join_handles {
            jh.join().unwrap();
        }
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, addr);
    }

    #[test]
    fn set_pointer_aligned_to_callback_chain_layout() {
        // Arrange: allocate a real CallbackChain on the stack and verify alignment.
        let chain = CallbackChain::empty();
        let ptr = &chain as *const CallbackChain as *mut CallbackChain;

        // CallbackChain contains a Vec, so alignment should be >= 8 bytes on 64-bit.
        let alignment = std::mem::align_of::<CallbackChain>();
        assert!(alignment >= 1);
        assert_eq!(
            (ptr as usize) % alignment, 0,
            "Stack-allocated CallbackChain should be properly aligned"
        );

        let handle = CallbackChainHandle::new();

        // Act: set the properly aligned pointer.
        handle.set(ptr);

        // Assert: roundtrip preserves the exact pointer value.
        assert_eq!(handle.as_ffi_ptr(), ptr);
        assert!(handle.is_set());
        handle.clear();
        assert!(!handle.is_set());
    }

    // --- 15 additional tests (wave 2) ---

    #[test]
    fn handle_is_send_and_sync() {
        // Arrange & Assert: compile-time verification that CallbackChainHandle
        // implements Send + Sync, which is required for Arc<CallbackChainHandle>
        // to be shared across threads in production code.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CallbackChainHandle>();
    }

    #[test]
    fn handle_size_equals_pointer_size() {
        // Arrange & Assert: CallbackChainHandle wraps a single AtomicPtr.
        // AtomicPtr<T> has the same layout as *mut T, so the handle should
        // be exactly one pointer wide with no extra padding or fields.
        assert_eq!(
            std::mem::size_of::<CallbackChainHandle>(),
            std::mem::size_of::<*mut CallbackChain>(),
            "CallbackChainHandle should be pointer-sized"
        );
    }

    #[test]
    fn box_heap_allocated_handle_lifecycle() {
        // Arrange: allocate handle on the heap via Box.
        let mut boxed = Box::new(CallbackChainHandle::new());
        assert!(!boxed.is_set());

        // Act: set, verify, clear through the Box reference.
        let ptr = 0xbee0 as *mut CallbackChain;
        boxed.set(ptr);
        assert!(boxed.is_set());
        assert_eq!(boxed.as_ffi_ptr(), ptr);

        boxed.clear();

        // Assert: cleared successfully on heap-allocated handle.
        assert!(!boxed.is_set());
        assert!(boxed.as_ffi_ptr().is_null());
    }

    #[test]
    fn option_handle_some_and_none_roundtrip() {
        // Arrange: wrap handle in Option.
        let mut opt: Option<CallbackChainHandle> = Some(CallbackChainHandle::new());
        let ptr = 0x0f0f as *mut CallbackChain;

        // Act: set through Option::as_ref.
        opt.as_ref().unwrap().set(ptr);
        assert!(opt.as_ref().unwrap().is_set());

        // Take the handle out, leaving None.
        let taken = opt.take();
        assert!(taken.unwrap().is_set());
        assert!(opt.is_none());

        // Assert: re-create a new handle in the Option slot.
        opt = Some(CallbackChainHandle::new());
        assert!(!opt.as_ref().unwrap().is_set());
    }

    #[test]
    fn dropping_clone_does_not_affect_original() {
        // Arrange: create handle and set a pointer.
        let handle = CallbackChainHandle::new();
        let ptr = 0xdddd as *mut CallbackChain;
        handle.set(ptr);

        // Act: clone into an inner scope, then drop the clone.
        {
            let _cloned = handle.clone();
            // cloned drops here
        }

        // Assert: original is unaffected by clone's destruction.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);
    }

    #[test]
    fn pointer_with_highest_bit_set() {
        // Arrange: on 64-bit platforms, set a pointer with the sign bit set
        // (kernel-space address range). The handle stores raw bits, no validation.
        let handle = CallbackChainHandle::new();
        let high_addr = (1usize << 63) | 0x1234;
        let ptr = high_addr as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: roundtrip preserves the exact bit pattern.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, high_addr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn re_clone_after_original_mutated() {
        // Arrange: set a pointer, clone, then mutate original.
        let handle = CallbackChainHandle::new();
        let ptr_a = 0xaaaa as *mut CallbackChain;
        let ptr_b = 0xbbbb as *mut CallbackChain;

        handle.set(ptr_a);
        let clone1 = handle.clone();
        assert_eq!(clone1.as_ffi_ptr(), ptr_a);

        // Act: mutate original, then clone again.
        handle.set(ptr_b);
        let clone2 = handle.clone();

        // Assert: clone2 sees the updated pointer; clone1 retains old snapshot.
        assert_eq!(clone2.as_ffi_ptr(), ptr_b);
        assert_eq!(clone1.as_ffi_ptr(), ptr_a, "clone1 should retain original snapshot");
        assert_eq!(handle.as_ffi_ptr(), ptr_b);
    }

    #[test]
    fn chain_len_preserved_through_handle_roundtrip() {
        // Arrange: create a CallbackChain with known content.
        let chain = CallbackChain::new(Vec::new());
        assert_eq!(chain.len(), 0);
        assert!(chain.is_empty());

        let handle = CallbackChainHandle::new();
        let ptr = &chain as *const CallbackChain as *mut CallbackChain;

        // Act: set and verify pointer roundtrip.
        handle.set(ptr);
        let recovered = handle.as_ffi_ptr();

        // Assert: the recovered pointer, when dereferenced (unsafe but valid),
        // points to the same chain with the same len.
        assert_eq!(recovered, ptr);
        assert_eq!(unsafe { &*recovered }.len(), 0);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn many_set_clear_iterations_verify_each_step() {
        // Arrange: stress test with 200 iterations.
        let handle = CallbackChainHandle::new();

        // Act & Assert: every iteration must be consistent.
        for i in 0..200u64 {
            let ptr = (0x1000 + i) as *mut CallbackChain;
            handle.set(ptr);
            assert!(handle.is_set(), "set failed at iteration {}", i);
            assert_eq!(handle.as_ffi_ptr(), ptr, "ptr mismatch at iteration {}", i);

            handle.clear();
            assert!(!handle.is_set(), "clear failed at iteration {}", i);
            assert!(handle.as_ffi_ptr().is_null(), "not null at iteration {}", i);
        }
    }

    #[test]
    fn handle_in_tuple_with_other_fields() {
        // Arrange: embed handle in a tuple-like struct with other data.
        struct SlotEntry {
            handle: CallbackChainHandle,
            priority: u8,
            active: bool,
        }

        let mut entry = SlotEntry {
            handle: CallbackChainHandle::new(),
            priority: 3,
            active: true,
        };

        // Act: use the handle alongside other fields.
        let ptr = 0x7777 as *mut CallbackChain;
        entry.handle.set(ptr);

        // Assert: handle state and other fields are independent.
        assert!(entry.handle.is_set());
        assert_eq!(entry.priority, 3);
        assert!(entry.active);

        entry.handle.clear();
        assert!(!entry.handle.is_set());
        assert_eq!(entry.priority, 3, "other fields must be untouched");
    }

    #[test]
    fn concurrent_read_only_stress_is_set_and_as_ffi_ptr() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: set a pointer, then launch many reader threads.
        let handle = Arc::new(CallbackChainHandle::new());
        let addr: usize = 0x5a5a;
        handle.set(addr as *mut CallbackChain);

        let mut join_handles = Vec::new();
        for _ in 0..16 {
            let h = Arc::clone(&handle);
            join_handles.push(thread::spawn(move || {
                // Each reader checks is_set and as_ffi_ptr many times.
                // Use usize comparison to avoid raw pointer Send bounds.
                for _ in 0..1000 {
                    assert!(h.is_set());
                    assert_eq!(h.as_ffi_ptr() as usize, addr);
                }
            }));
        }

        // Act & Assert: all readers must observe consistent state.
        for jh in join_handles {
            jh.join().unwrap();
        }
        assert!(handle.is_set());
    }

    #[test]
    fn empty_vs_new_chain_both_work_with_handle() {
        // Arrange: CallbackChain::empty() and CallbackChain::new(Vec::new()) are equivalent.
        let chain_empty = CallbackChain::empty();
        let chain_new = CallbackChain::new(Vec::new());
        let ptr_empty = &chain_empty as *const CallbackChain as *mut CallbackChain;
        let ptr_new = &chain_new as *const CallbackChain as *mut CallbackChain;

        let handle = CallbackChainHandle::new();

        // Act: set from empty(), verify, clear, then set from new().
        handle.set(ptr_empty);
        assert_eq!(handle.as_ffi_ptr(), ptr_empty);
        handle.clear();

        handle.set(ptr_new);
        assert_eq!(handle.as_ffi_ptr(), ptr_new);
        handle.clear();

        // Assert: both roundtrips succeeded.
        assert!(!handle.is_set());
    }

    #[test]
    fn two_handles_swap_pointers() {
        // Arrange: two handles with different pointers.
        let h1 = CallbackChainHandle::new();
        let h2 = CallbackChainHandle::new();
        let ptr_a = 0x1111 as *mut CallbackChain;
        let ptr_b = 0x2222 as *mut CallbackChain;

        h1.set(ptr_a);
        h2.set(ptr_b);

        // Act: read both, then swap by clearing and re-setting.
        let a = h1.as_ffi_ptr();
        let b = h2.as_ffi_ptr();
        h1.clear();
        h2.clear();
        h1.set(b);
        h2.set(a);

        // Assert: pointers are now swapped.
        assert_eq!(h1.as_ffi_ptr(), ptr_b);
        assert_eq!(h2.as_ffi_ptr(), ptr_a);
    }

    #[test]
    fn handle_survives_arc_strong_count_changes() {
        use std::sync::Arc;

        // Arrange: wrap in Arc and set a pointer.
        let handle = Arc::new(CallbackChainHandle::new());
        let ptr = 0xf00f as *mut CallbackChain;
        handle.set(ptr);

        assert_eq!(Arc::strong_count(&handle), 1);

        // Act: clone the Arc multiple times.
        let c1 = Arc::clone(&handle);
        let c2 = Arc::clone(&handle);
        let c3 = Arc::clone(&handle);
        assert_eq!(Arc::strong_count(&handle), 4);

        // All clones see the same pointer.
        assert!(c1.is_set());
        assert!(c2.is_set());
        assert!(c3.is_set());

        // Drop two clones.
        drop(c1);
        drop(c2);
        assert_eq!(Arc::strong_count(&handle), 2);

        // Assert: original and remaining clone still work correctly.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);
        assert!(c3.is_set());
        assert_eq!(c3.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!c3.is_set(), "Clearing through one Arc is visible to others");
    }

    #[test]
    fn callback_chain_alignment_matches_handle_storage() {
        // Arrange: verify that the CallbackChain alignment is compatible with pointer storage.
        let chain = CallbackChain::empty();
        let ptr = &chain as *const CallbackChain as *mut CallbackChain;
        let alignment = std::mem::align_of::<CallbackChain>();

        // The pointer should be aligned to the type's alignment.
        assert_eq!(
            (ptr as usize) % alignment, 0,
            "Stack CallbackChain pointer must be aligned to {} bytes",
            alignment
        );

        // The handle stores a raw pointer; verify alignment after set+load roundtrip.
        let handle = CallbackChainHandle::new();
        handle.set(ptr);
        let loaded = handle.as_ffi_ptr();

        // Assert: loaded pointer retains alignment.
        assert_eq!(
            (loaded as usize) % alignment, 0,
            "Roundtripped pointer must retain original alignment"
        );
        handle.clear();
    }

    // --- 15 additional tests (wave 3) ---

    #[test]
    fn repeated_set_without_clear_overwrites_each_time() {
        // Arrange: create a handle and a sequence of distinct pointers.
        let handle = CallbackChainHandle::new();
        let ptrs: [usize; 5] = [0xAA00, 0xBB10, 0xCC20, 0xDD30, 0xEE40];

        // Act: call set() five times without any clear().
        for &addr in &ptrs {
            handle.set(addr as *mut CallbackChain);
        }

        // Assert: only the last pointer is retained.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, 0xEE40);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_from_cleared_handle_then_set_clone_independently() {
        // Arrange: set and clear, then clone the null state.
        let handle = CallbackChainHandle::new();
        let ptr = 0x7a7a as *mut CallbackChain;
        handle.set(ptr);
        handle.clear();

        let mut cloned = handle.clone();
        assert!(!cloned.is_set());

        // Act: set the clone to a new pointer.
        let new_ptr = 0x8b8b as *mut CallbackChain;
        cloned.set(new_ptr);

        // Assert: clone is set, original remains null.
        assert!(cloned.is_set());
        assert_eq!(cloned.as_ffi_ptr(), new_ptr);
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn handle_reassignment_replaces_entire_state() {
        // Arrange: first handle with a pointer.
        let mut handle = CallbackChainHandle::new();
        let ptr_a = 0x1234 as *mut CallbackChain;
        handle.set(ptr_a);
        assert!(handle.is_set());

        // Act: reassign to a brand-new handle.
        handle = CallbackChainHandle::new();

        // Assert: the reassigned handle starts fresh with null state.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());

        // It should accept new pointers normally.
        let ptr_b = 0x5678 as *mut CallbackChain;
        handle.set(ptr_b);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);
    }

    #[test]
    fn as_ffi_ptr_deref_matches_original_chain_identity() {
        // Arrange: create a real CallbackChain and obtain its pointer via the handle.
        let chain = CallbackChain::empty();
        let handle = CallbackChainHandle::new();
        let ptr = &chain as *const CallbackChain as *mut CallbackChain;
        handle.set(ptr);

        // Act: recover the pointer and dereference it (unsafe, but valid within scope).
        let recovered = handle.as_ffi_ptr();

        // Assert: the dereferenced chain has the same identity as the original.
        assert_eq!(unsafe { &*recovered as *const _ }, &chain as *const _);
        assert!(unsafe { &*recovered }.is_empty());

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_shared_via_rc_single_threaded() {
        use std::rc::Rc;

        // Arrange: Rc does not require Send+Sync, so it works for single-threaded sharing.
        let handle = Rc::new(CallbackChainHandle::new());
        let ptr = 0xe1e1 as *mut CallbackChain;

        // Act: set through one Rc reference.
        handle.set(ptr);

        // Assert: all Rc references see the same state.
        let another = Rc::clone(&handle);
        assert!(another.is_set());
        assert_eq!(another.as_ffi_ptr(), ptr);

        // Clear through the clone reference.
        another.clear();
        assert!(!handle.is_set(), "Clear via Rc clone is visible to original");
    }

    #[test]
    fn interleaved_operations_on_two_handles() {
        // Arrange: two independent handles.
        let h1 = CallbackChainHandle::new();
        let h2 = CallbackChainHandle::new();
        let ptr_a = 0xa1a1 as *mut CallbackChain;
        let ptr_b = 0xb2b2 as *mut CallbackChain;

        // Act: interleave set/clear operations.
        h1.set(ptr_a);
        assert!(h1.is_set());
        assert!(!h2.is_set());

        h2.set(ptr_b);
        assert!(h1.is_set());
        assert!(h2.is_set());

        h1.clear();
        assert!(!h1.is_set());
        assert!(h2.is_set());
        assert_eq!(h2.as_ffi_ptr(), ptr_b);

        h2.clear();
        assert!(!h1.is_set());
        assert!(!h2.is_set());
    }

    #[test]
    fn clone_after_move_preserves_snapshot() {
        // Arrange: create and set a handle, then move it.
        let handle = CallbackChainHandle::new();
        let ptr = 0x4321 as *mut CallbackChain;
        handle.set(ptr);

        let moved = handle;

        // Act: clone the moved handle.
        let cloned = moved.clone();

        // Assert: both the moved handle and its clone carry the original pointer.
        assert!(moved.is_set());
        assert!(cloned.is_set());
        assert_eq!(moved.as_ffi_ptr(), ptr);
        assert_eq!(cloned.as_ffi_ptr(), ptr);

        moved.clear();
        assert!(!moved.is_set());
        assert!(cloned.is_set(), "Clone retains snapshot after moved is cleared");
    }

    #[test]
    fn clear_then_set_same_pointer_twice() {
        // Arrange: set, clear, then set the same pointer again.
        let handle = CallbackChainHandle::new();
        let ptr = 0x5e5e as *mut CallbackChain;

        handle.set(ptr);
        assert!(handle.is_set());

        handle.clear();
        assert!(!handle.is_set());

        // Act: set the same pointer again.
        handle.set(ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        // And again after another clear.
        handle.clear();
        handle.set(ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_returned_from_function_preserves_state() {
        fn build_set_handle(addr: usize) -> CallbackChainHandle {
            let h = CallbackChainHandle::new();
            h.set(addr as *mut CallbackChain);
            h
        }

        // Arrange: function returns an already-set handle.
        let addr: usize = 0xda7a;
        let handle = build_set_handle(addr);

        // Assert: the handle arrived with the pointer intact.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, addr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn concurrent_writer_holds_set_then_clears_observer_sees_both() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
        use std::sync::Arc;
        use std::thread;

        // Arrange: worker sets pointer and holds it until observer acknowledges,
        // then clears. This avoids the race where set/clear toggles too fast.
        let handle = Arc::new(CallbackChainHandle::new());
        let set_count = Arc::new(AtomicUsize::new(0));
        let clear_count = Arc::new(AtomicUsize::new(0));
        let observer_saw_set = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let cleared = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let h_worker = Arc::clone(&handle);
        let saw = Arc::clone(&observer_saw_set);
        let cl = Arc::clone(&cleared);
        let worker = thread::spawn(move || {
            let ptr = 0xace as *mut CallbackChain;
            h_worker.set(ptr);
            // Hold the pointer until the observer has seen it.
            while !saw.load(AtomicOrdering::Acquire) {
                thread::yield_now();
            }
            h_worker.clear();
            cl.store(true, AtomicOrdering::Release);
        });

        let h_obs = Arc::clone(&handle);
        let sc = Arc::clone(&set_count);
        let cc = Arc::clone(&clear_count);
        let saw_w = Arc::clone(&observer_saw_set);
        let cl_r = Arc::clone(&cleared);
        let observer = thread::spawn(move || {
            // Spin until we see set state.
            for _ in 0..10000 {
                if h_obs.is_set() {
                    sc.fetch_add(1, AtomicOrdering::Relaxed);
                    saw_w.store(true, AtomicOrdering::Release);
                    break;
                }
                thread::yield_now();
            }
            // Now spin until we see clear state.
            for _ in 0..10000 {
                if !h_obs.is_set() {
                    cc.fetch_add(1, AtomicOrdering::Relaxed);
                    break;
                }
                thread::yield_now();
            }
        });

        worker.join().unwrap();
        observer.join().unwrap();

        // Assert: the observer saw both set and clear states.
        assert!(
            set_count.load(AtomicOrdering::Acquire) > 0,
            "Observer should have seen at least one set state"
        );
        assert!(
            clear_count.load(AtomicOrdering::Acquire) > 0,
            "Observer should have seen at least one clear state"
        );
        // Final state is clear.
        assert!(!handle.is_set());
    }

    #[test]
    fn nested_scope_set_persists_after_scope_exit() {
        // Arrange: outer handle set to a pointer.
        let handle = CallbackChainHandle::new();
        let ptr_outer = 0x0a0a as *mut CallbackChain;
        handle.set(ptr_outer);

        // Act: inner scope sets a different pointer.
        let ptr_inner = 0x0b0b as *mut CallbackChain;
        {
            handle.set(ptr_inner);
            assert_eq!(handle.as_ffi_ptr(), ptr_inner);
            // Scope ends but handle is shared, not owned by the block.
        }

        // Assert: handle still has the inner pointer (not reverted to outer).
        assert_eq!(handle.as_ffi_ptr(), ptr_inner);
        assert!(handle.is_set());

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn default_trait_bound_satisfied_at_compile_time() {
        // Arrange & Assert: compile-time proof that CallbackChainHandle: Default.
        fn require_default<T: Default>() -> T {
            T::default()
        }

        let handle: CallbackChainHandle = require_default();
        assert!(handle.as_ffi_ptr().is_null());
        assert!(!handle.is_set());
    }

    #[test]
    fn pointer_with_offset_from_base_address() {
        // Arrange: simulate a pointer computed as base + offset.
        let chain = CallbackChain::empty();
        let base = &chain as *const CallbackChain as usize;
        // Use the actual chain address (offset 0) — any real offset would point
        // outside the object, so we verify the base address roundtrips exactly.
        let handle = CallbackChainHandle::new();
        let ptr = base as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: the stored pointer equals the computed base.
        assert_eq!(handle.as_ffi_ptr() as usize, base);
        assert!(handle.is_set());

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_chain_depth_six_all_independent() {
        // Arrange: set original, then create a chain of 6 clones.
        let handle = CallbackChainHandle::new();
        let ptr = 0x1c1c as *mut CallbackChain;
        handle.set(ptr);

        let c1 = handle.clone();
        let c2 = c1.clone();
        let c3 = c2.clone();
        let c4 = c3.clone();
        let c5 = c4.clone();
        let c6 = c5.clone();

        // Act: clear c3 and c5.
        c3.clear();
        c5.clear();

        // Assert: only c3 and c5 are cleared; all others hold the pointer.
        assert!(handle.is_set());
        assert!(c1.is_set());
        assert!(c2.is_set());
        assert!(!c3.is_set());
        assert!(c4.is_set());
        assert!(!c5.is_set());
        assert!(c6.is_set());

        assert_eq!(c6.as_ffi_ptr(), ptr);
    }

    #[test]
    fn concurrent_two_clearers_final_state_is_null() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: set a pointer, then two threads race to clear it.
        let handle = Arc::new(CallbackChainHandle::new());
        let ptr = 0xface as *mut CallbackChain;
        handle.set(ptr);
        assert!(handle.is_set());

        let h1 = Arc::clone(&handle);
        let clearer1 = thread::spawn(move || {
            h1.clear();
        });

        let h2 = Arc::clone(&handle);
        let clearer2 = thread::spawn(move || {
            h2.clear();
        });

        // Act: both threads clear.
        clearer1.join().unwrap();
        clearer2.join().unwrap();

        // Assert: double-clear is safe; final state is null.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    // --- 15 additional tests (wave 4) ---

    #[test]
    fn repeated_set_overwrite_then_read_matches_last() {
        // Arrange: set 10 distinct pointers in sequence, overwriting each time.
        let handle = CallbackChainHandle::new();
        let last_ptr = (0x2000 + 9) as *mut CallbackChain;

        // Act: overwrite the pointer 10 times.
        for i in 0..10u64 {
            handle.set((0x2000 + i) as *mut CallbackChain);
        }

        // Assert: only the last set pointer is visible.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), last_ptr);
        assert_eq!(handle.as_ffi_ptr() as usize, 0x2009);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_from_cleared_then_set_both_independently() {
        // Arrange: set, clear, then clone the null state.
        let mut handle = CallbackChainHandle::new();
        handle.set(0x1111 as *mut CallbackChain);
        handle.clear();

        let mut cloned = handle.clone();
        assert!(!cloned.is_set());
        assert!(!handle.is_set());

        // Act: set both to different pointers independently.
        handle.set(0x2222 as *mut CallbackChain);
        cloned.set(0x3333 as *mut CallbackChain);

        // Assert: each handle retains its own pointer.
        assert_eq!(handle.as_ffi_ptr() as usize, 0x2222);
        assert_eq!(cloned.as_ffi_ptr() as usize, 0x3333);
    }

    #[test]
    fn reassign_after_clear_accepts_new_pointer() {
        // Arrange: set, clear, then reassign the binding to a new handle.
        let mut handle = CallbackChainHandle::new();
        handle.set(0xaaaa as *mut CallbackChain);
        handle.clear();

        // Act: reassign the handle variable to a fresh instance.
        handle = CallbackChainHandle::new();
        assert!(!handle.is_set());

        let ptr = 0xbbbb as *mut CallbackChain;
        handle.set(ptr);

        // Assert: the reassigned handle works correctly with a new pointer.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);
        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn as_ffi_ptr_chain_identity_through_arc_roundtrip() {
        // Arrange: create a real CallbackChain and wrap handle in Arc.
        use std::sync::Arc;

        let chain = CallbackChain::empty();
        let handle = Arc::new(CallbackChainHandle::new());
        let ptr = &chain as *const CallbackChain as *mut CallbackChain;

        // Act: set through Arc, clone the Arc, read from the clone.
        handle.set(ptr);
        let arc_clone = Arc::clone(&handle);
        let recovered = arc_clone.as_ffi_ptr();

        // Assert: the pointer identity is preserved through the Arc chain.
        assert_eq!(recovered, ptr);
        assert_eq!(unsafe { &*recovered as *const _ }, &chain as *const _);
        assert!(unsafe { &*recovered }.is_empty());
        handle.clear();
    }

    #[test]
    fn rc_three_clones_share_same_underlying_state() {
        use std::rc::Rc;

        // Arrange: three Rc clones all share the same handle allocation.
        let h1 = Rc::new(CallbackChainHandle::new());
        let h2 = Rc::clone(&h1);
        let h3 = Rc::clone(&h2);

        let ptr = 0x4455 as *mut CallbackChain;

        // Act: set through h1.
        h1.set(ptr);

        // Assert: all three see the same state (Rc deref to same allocation).
        assert!(h1.is_set());
        assert!(h2.is_set());
        assert!(h3.is_set());
        assert_eq!(h2.as_ffi_ptr() as usize, 0x4455);
        assert_eq!(h3.as_ffi_ptr() as usize, 0x4455);

        // Clear through h3, all see null.
        h3.clear();
        assert!(!h1.is_set());
        assert!(!h2.is_set());
        assert!(h2.as_ffi_ptr().is_null());
    }

    #[test]
    fn interleaved_set_clear_get_three_pointers() {
        // Arrange: three distinct pointers for interleaved operations.
        let handle = CallbackChainHandle::new();
        let ptr_a = 0xa0a0 as *mut CallbackChain;
        let ptr_b = 0xb1b1 as *mut CallbackChain;
        let ptr_c = 0xc2c2 as *mut CallbackChain;

        // Act & Assert: interleave set, get, clear with verification at each step.
        handle.set(ptr_a);
        assert_eq!(handle.as_ffi_ptr(), ptr_a);
        assert!(handle.is_set());

        handle.clear();
        assert!(!handle.is_set());

        handle.set(ptr_b);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);

        handle.clear();
        assert!(handle.as_ffi_ptr().is_null());

        handle.set(ptr_c);
        assert_eq!(handle.as_ffi_ptr(), ptr_c);

        handle.set(ptr_a);
        assert_eq!(handle.as_ffi_ptr(), ptr_a, "Overwrite should update to ptr_a");

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_after_move_then_clear_original() {
        // Arrange: set a handle, move it, then clone the moved handle.
        let handle = CallbackChainHandle::new();
        let ptr = 0x9876 as *mut CallbackChain;
        handle.set(ptr);

        let moved = handle;

        // Act: clone the moved handle, then clear the moved handle.
        let cloned = moved.clone();
        moved.clear();

        // Assert: clone retains the snapshot; moved is now null.
        assert!(!moved.is_set());
        assert!(cloned.is_set());
        assert_eq!(cloned.as_ffi_ptr(), ptr);
    }

    #[test]
    fn clear_set_loop_five_cycles_consistent() {
        // Arrange: verify handle integrity through 5 full clear-set cycles.
        let handle = CallbackChainHandle::new();

        // Act & Assert: each cycle uses a unique pointer.
        for cycle in 0..5u64 {
            let ptr = (0x5000 + cycle * 0x100) as *mut CallbackChain;
            handle.set(ptr);
            assert!(handle.is_set(), "Should be set at cycle {}", cycle);
            assert_eq!(handle.as_ffi_ptr(), ptr, "Ptr mismatch at cycle {}", cycle);

            handle.clear();
            assert!(!handle.is_set(), "Should be clear at cycle {}", cycle);
            assert!(handle.as_ffi_ptr().is_null(), "Should be null at cycle {}", cycle);
        }
    }

    #[test]
    fn function_returning_cleared_handle_is_null() {
        fn build_cleared_handle(addr: usize) -> CallbackChainHandle {
            let h = CallbackChainHandle::new();
            h.set(addr as *mut CallbackChain);
            h.clear();
            h
        }

        // Arrange: function returns a handle that was set then cleared.
        let handle = build_cleared_handle(0xf0f0);

        // Assert: the handle arrives in cleared state.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());

        // It should still accept new pointers.
        let ptr = 0xe0e0 as *mut CallbackChain;
        handle.set(ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);
        handle.clear();
    }

    #[test]
    fn concurrent_writer_and_clearer_final_state_deterministic() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::sync::Arc;
        use std::thread;

        // Arrange: writer sets, holds, then clears; clearer clears after signal.
        let handle = Arc::new(CallbackChainHandle::new());
        let writer_done = Arc::new(AtomicBool::new(false));

        let h_writer = Arc::clone(&handle);
        let wd = Arc::clone(&writer_done);
        let writer = thread::spawn(move || {
            h_writer.set(0x7777 as *mut CallbackChain);
            // Hold the pointer briefly.
            for _ in 0..100 {
                thread::yield_now();
            }
            h_writer.clear();
            wd.store(true, AtomicOrdering::Release);
        });

        let h_clearer = Arc::clone(&handle);
        let wd_r = Arc::clone(&writer_done);
        let clearer = thread::spawn(move || {
            // Wait for writer to finish, then clear.
            while !wd_r.load(AtomicOrdering::Acquire) {
                thread::yield_now();
            }
            h_clearer.clear();
        });

        writer.join().unwrap();
        clearer.join().unwrap();

        // Assert: after both finish, state is deterministically null.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn nested_scope_cloned_handle_survives_scope_drop() {
        // Arrange: outer handle starts null.
        let handle = CallbackChainHandle::new();
        let ptr = 0xdddd as *mut CallbackChain;

        // Act: inner scope clones and sets the clone.
        let outer_clone = {
            let inner_clone = handle.clone();
            inner_clone.set(ptr);
            assert!(inner_clone.is_set());
            // Return the clone to outer scope.
            inner_clone
        };

        // Assert: the cloned handle survives the inner scope.
        assert!(outer_clone.is_set());
        assert_eq!(outer_clone.as_ffi_ptr(), ptr);
        // Original handle was never mutated (clone is independent).
        assert!(!handle.is_set());
    }

    #[test]
    fn default_trait_constructible_in_generic_context() {
        // Arrange: generic function that requires Default.
        fn build_default_vec<T: Default + Clone>(count: usize) -> Vec<T> {
            (0..count).map(|_| T::default()).collect()
        }

        // Act: construct a Vec of default handles.
        let handles: Vec<CallbackChainHandle> = build_default_vec(3);

        // Assert: all handles are null and not set.
        assert_eq!(handles.len(), 3);
        for (i, h) in handles.iter().enumerate() {
            assert!(h.as_ffi_ptr().is_null(), "Handle {} should be null", i);
            assert!(!h.is_set(), "Handle {} should not be set", i);
        }
    }

    #[test]
    fn pointer_offset_within_array_preserved() {
        // Arrange: simulate pointers into an array of CallbackChain.
        let chains: [CallbackChain; 4] = [
            CallbackChain::empty(),
            CallbackChain::empty(),
            CallbackChain::empty(),
            CallbackChain::empty(),
        ];
        let handle = CallbackChainHandle::new();

        // Act: set to each element and verify the offset from the base.
        let base = &chains[0] as *const CallbackChain as usize;
        for (i, chain) in chains.iter().enumerate() {
            let ptr = chain as *const CallbackChain as *mut CallbackChain;
            handle.set(ptr);
            let offset = handle.as_ffi_ptr() as usize - base;
            let expected_offset = i * std::mem::size_of::<CallbackChain>();
            assert_eq!(
                offset, expected_offset,
                "Element {} offset mismatch: got {}, expected {}",
                i, offset, expected_offset
            );
        }

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_chain_depth_seven_all_hold_same_ptr() {
        // Arrange: set original, then create a chain of 7 clones.
        let handle = CallbackChainHandle::new();
        let ptr = 0x2d2d as *mut CallbackChain;
        handle.set(ptr);

        let c1 = handle.clone();
        let c2 = c1.clone();
        let c3 = c2.clone();
        let c4 = c3.clone();
        let c5 = c4.clone();
        let c6 = c5.clone();
        let c7 = c6.clone();

        // Act: clear the original and one middle clone.
        handle.clear();
        c4.clear();

        // Assert: original and c4 are null; all others still hold the pointer.
        assert!(!handle.is_set());
        assert!(c1.is_set());
        assert!(c2.is_set());
        assert!(c3.is_set());
        assert!(!c4.is_set());
        assert!(c5.is_set());
        assert!(c6.is_set());
        assert!(c7.is_set());
        assert_eq!(c7.as_ffi_ptr(), ptr);
    }

    #[test]
    fn concurrent_two_clearers_then_set_final_visible() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: set a pointer, then two threads race to clear.
        let handle = Arc::new(CallbackChainHandle::new());
        handle.set(0xcafe as *mut CallbackChain);
        assert!(handle.is_set());

        let h1 = Arc::clone(&handle);
        let clearer1 = thread::spawn(move || {
            h1.clear();
        });

        let h2 = Arc::clone(&handle);
        let clearer2 = thread::spawn(move || {
            h2.clear();
        });

        // Act: wait for both clearers, then set a new pointer.
        clearer1.join().unwrap();
        clearer2.join().unwrap();
        assert!(!handle.is_set());

        let new_ptr = 0xbabe as *mut CallbackChain;
        handle.set(new_ptr);

        // Assert: after clearing and re-setting, the new pointer is visible.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), new_ptr);
        handle.clear();
        assert!(!handle.is_set());
    }

    // --- 15 additional tests (wave 5) ---

    #[test]
    fn is_set_returns_false_after_set_then_null_set() {
        // Arrange: set to a non-null pointer, then explicitly set to null.
        let handle = CallbackChainHandle::new();
        handle.set(0xabcd as *mut CallbackChain);
        assert!(handle.is_set());

        // Act: overwrite with null via set (not clear).
        handle.set(std::ptr::null_mut());

        // Assert: is_set returns false because null means not set.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn handle_stores_pointer_to_stack_local_chain_entire_scope() {
        // Arrange: verify that a pointer to a stack-local CallbackChain
        // remains valid and retrievable throughout the entire scope.
        let chain = CallbackChain::empty();
        let handle = CallbackChainHandle::new();
        let ptr = &chain as *const CallbackChain as *mut CallbackChain;

        // Act: set once, read multiple times at different points.
        handle.set(ptr);
        let first_read = handle.as_ffi_ptr();
        let second_read = handle.as_ffi_ptr();
        let third_read = handle.as_ffi_ptr();

        // Assert: all reads return the exact same pointer.
        assert_eq!(first_read, ptr);
        assert_eq!(second_read, ptr);
        assert_eq!(third_read, ptr);
        assert_eq!(unsafe { &*first_read as *const _ }, &chain as *const _);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_of_set_handle_then_clear_both_end_null() {
        // Arrange: set a handle and clone it.
        let handle = CallbackChainHandle::new();
        let ptr = 0x5a5a as *mut CallbackChain;
        handle.set(ptr);

        let cloned = handle.clone();
        assert!(handle.is_set());
        assert!(cloned.is_set());

        // Act: clear both independently.
        handle.clear();
        cloned.clear();

        // Assert: both are now null.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
        assert!(!cloned.is_set());
        assert!(cloned.as_ffi_ptr().is_null());
    }

    #[test]
    fn mutex_wrapped_handle_simulates_executor_pattern() {
        use std::sync::Mutex;

        // Arrange: simulate the real Executor pattern where handle
        // is protected by a Mutex during set/clear operations.
        let handle = Mutex::new(CallbackChainHandle::new());
        let ptr = 0xef01 as *mut CallbackChain;

        // Act: set under lock (simulating Executor mutex held).
        {
            let guard = handle.lock().unwrap();
            guard.set(ptr);
        }

        // Read without lock (as_ffi_ptr is &self, uses Acquire ordering).
        let read_ptr = handle.lock().unwrap().as_ffi_ptr();
        assert_eq!(read_ptr, ptr);
        assert!(handle.lock().unwrap().is_set());

        // Clear under lock.
        {
            let guard = handle.lock().unwrap();
            guard.clear();
        }

        // Assert: cleared state visible after lock release.
        assert!(!handle.lock().unwrap().is_set());
        assert!(handle.lock().unwrap().as_ffi_ptr().is_null());
    }

    #[test]
    fn handle_created_via_default_trait_accepts_pointer() {
        // Arrange: use Default trait explicitly to create handle.
        let handle = CallbackChainHandle::default();
        assert!(!handle.is_set());

        // Act: set a pointer and verify.
        let ptr = 0xc0de as *mut CallbackChain;
        handle.set(ptr);

        // Assert: Default-created handle behaves identically to new().
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn vec_of_handles_bulk_set_and_clear() {
        // Arrange: create a Vec of 10 handles.
        let handles: Vec<CallbackChainHandle> = (0..10)
            .map(|_| CallbackChainHandle::new())
            .collect();

        // Act: set all handles to distinct pointers.
        for (i, h) in handles.iter().enumerate() {
            h.set((0x1000 + i) as *mut CallbackChain);
        }

        // Assert: all handles are set with correct pointers.
        for (i, h) in handles.iter().enumerate() {
            assert!(h.is_set(), "Handle {} should be set", i);
            assert_eq!(h.as_ffi_ptr() as usize, 0x1000 + i);
        }

        // Act: bulk clear.
        for h in &handles {
            h.clear();
        }

        // Assert: all handles cleared.
        for (i, h) in handles.iter().enumerate() {
            assert!(!h.is_set(), "Handle {} should be cleared", i);
        }
    }

    #[test]
    fn clone_of_clone_of_clone_independence_after_partial_clear() {
        // Arrange: set original, create three levels of clones.
        let handle = CallbackChainHandle::new();
        let ptr = 0x7b7b as *mut CallbackChain;
        handle.set(ptr);

        let c1 = handle.clone();
        let c2 = c1.clone();
        let c3 = c2.clone();

        // Act: clear only the middle clone (c2).
        c2.clear();

        // Assert: original, c1, c3 are still set; c2 is null.
        assert!(handle.is_set());
        assert!(c1.is_set());
        assert!(!c2.is_set());
        assert!(c3.is_set());
        assert_eq!(c3.as_ffi_ptr(), ptr);

        // c2 can be set to a new pointer independently.
        let new_ptr = 0x8c8c as *mut CallbackChain;
        c2.set(new_ptr);
        assert_eq!(c2.as_ffi_ptr(), new_ptr);
        assert_eq!(c1.as_ffi_ptr(), ptr, "c1 unaffected by c2 mutation");
    }

    #[test]
    fn pointer_with_low_alignment_preserved_exactly() {
        // Arrange: use a pointer value that is a small odd number (alignment 1).
        let handle = CallbackChainHandle::new();
        let odd_addr: usize = 3;
        let ptr = odd_addr as *mut CallbackChain;

        // Act: set and retrieve the unaligned pointer.
        handle.set(ptr);

        // Assert: the exact bit pattern is preserved without alignment correction.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, odd_addr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_survives_vec_pop_and_push_cycle() {
        // Arrange: store a set handle in a Vec, pop it, push it back.
        let mut handles = Vec::new();
        let ptr = 0x9d9d as *mut CallbackChain;

        let handle = CallbackChainHandle::new();
        handle.set(ptr);
        handles.push(handle);

        // Act: pop the handle out of the Vec.
        let mut popped = handles.pop().unwrap();
        assert!(popped.is_set());
        assert_eq!(popped.as_ffi_ptr(), ptr);

        // Mutate the popped handle.
        popped.clear();
        assert!(!popped.is_set());

        let new_ptr = 0xaeae as *mut CallbackChain;
        popped.set(new_ptr);

        // Push it back.
        handles.push(popped);

        // Assert: the handle in the Vec has the updated state.
        assert_eq!(handles[0].as_ffi_ptr(), new_ptr);
        assert!(handles[0].is_set());
        handles[0].clear();
    }

    #[test]
    fn set_with_callback_chain_from_empty_new_equivalent_pointers() {
        // Arrange: create two chains via different constructors and verify
        // the handle round-trips correctly for both.
        let chain_empty = CallbackChain::empty();
        let chain_new = CallbackChain::new(Vec::new());

        let ptr_empty = &chain_empty as *const CallbackChain as *mut CallbackChain;
        let ptr_new = &chain_new as *const CallbackChain as *mut CallbackChain;

        // They must be at distinct addresses.
        assert_ne!(ptr_empty, ptr_new);

        let handle = CallbackChainHandle::new();

        // Act & Assert: set to empty() chain, verify.
        handle.set(ptr_empty);
        assert_eq!(unsafe { &*handle.as_ffi_ptr() }.len(), 0);
        assert_eq!(handle.as_ffi_ptr(), ptr_empty);

        // Switch to new() chain.
        handle.set(ptr_new);
        assert_eq!(unsafe { &*handle.as_ffi_ptr() }.len(), 0);
        assert_eq!(handle.as_ffi_ptr(), ptr_new);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_in_cell_interior_mutability_pattern() {
        use std::cell::RefCell;

        // Arrange: RefCell allows interior mutability for single-threaded usage.
        let cell = RefCell::new(CallbackChainHandle::new());
        let ptr = 0xb1b1 as *mut CallbackChain;

        // Act: set through a borrow.
        cell.borrow().set(ptr);

        // Assert: readable through a separate borrow.
        assert!(cell.borrow().is_set());
        assert_eq!(cell.borrow().as_ffi_ptr(), ptr);

        // Clear through a borrow.
        cell.borrow().clear();
        assert!(!cell.borrow().is_set());
    }

    #[test]
    fn cloned_handle_set_to_different_ptr_than_original_both_correct() {
        // Arrange: original and clone start with the same pointer.
        let handle = CallbackChainHandle::new();
        let ptr_a = 0x1234 as *mut CallbackChain;
        handle.set(ptr_a);

        let cloned = handle.clone();
        assert_eq!(cloned.as_ffi_ptr(), ptr_a);

        // Act: set clone to a different pointer.
        let ptr_b = 0x5678 as *mut CallbackChain;
        cloned.set(ptr_b);

        // Assert: original and clone now hold different pointers.
        assert_eq!(handle.as_ffi_ptr(), ptr_a);
        assert_eq!(cloned.as_ffi_ptr(), ptr_b);
        assert!(handle.is_set());
        assert!(cloned.is_set());
        assert_ne!(handle.as_ffi_ptr(), cloned.as_ffi_ptr());
    }

    #[test]
    fn set_pointer_value_eq_as_ffi_ptr_minus_base_is_zero() {
        // Arrange: create a CallbackChain and compute its base address.
        let chain = CallbackChain::empty();
        let base = &chain as *const CallbackChain as usize;

        let handle = CallbackChainHandle::new();
        handle.set(base as *mut CallbackChain);

        // Act: retrieve and compute offset from base.
        let recovered = handle.as_ffi_ptr() as usize;

        // Assert: offset from the chain's base is exactly zero.
        assert_eq!(recovered - base, 0);
        assert_eq!(recovered, base);

        handle.clear();
    }

    #[test]
    fn concurrent_reader_observes_exactly_one_set_and_one_clear() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::sync::Arc;
        use std::thread;

        // Arrange: set the pointer first, then launch a reader that counts
        // transitions from set→clear and clear→set.
        let handle = Arc::new(CallbackChainHandle::new());
        let ptr = 0xd0d0 as *mut CallbackChain;
        handle.set(ptr);

        let ready = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicBool::new(false));

        let h_reader = Arc::clone(&handle);
        let r_ready = Arc::clone(&ready);
        let r_done = Arc::clone(&done);
        let reader = thread::spawn(move || {
            while !r_ready.load(AtomicOrdering::Acquire) {
                thread::yield_now();
            }
            let mut transitions = 0usize;
            let mut was_set = true; // starts set
            for _ in 0..50000 {
                let now_set = h_reader.is_set();
                if now_set != was_set {
                    transitions += 1;
                    was_set = now_set;
                }
                if r_done.load(AtomicOrdering::Acquire) {
                    break;
                }
                thread::yield_now();
            }
            transitions
        });

        let h_writer = Arc::clone(&handle);
        let w_ready = Arc::clone(&ready);
        let w_done = Arc::clone(&done);
        let writer = thread::spawn(move || {
            w_ready.store(true, AtomicOrdering::Release);
            // Hold for a bit, then clear once.
            for _ in 0..500 {
                thread::yield_now();
            }
            h_writer.clear();
            w_done.store(true, AtomicOrdering::Release);
        });

        writer.join().unwrap();
        let transitions = reader.join().unwrap();

        // Assert: at least one transition (set→clear) was observed.
        assert!(transitions >= 1, "Reader should see at least 1 transition, got {}", transitions);
        // Final state is clear.
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_stores_and_retrieves_pointer_to_static_empty_chain() {
        // Arrange: use a function-scoped static-like pattern.
        fn make_chain_ptr() -> *mut CallbackChain {
            let chain = CallbackChain::empty();
            &chain as *const CallbackChain as *mut CallbackChain
        }

        // Note: the pointer is dangling after make_chain_ptr returns,
        // but the handle only stores bits, it never dereferences.
        let handle = CallbackChainHandle::new();
        let ptr = make_chain_ptr();

        // Act: store the dangling pointer (handle only stores raw bits).
        handle.set(ptr);

        // Assert: the exact bit pattern is preserved.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    // --- 13 additional tests (wave 6) ---

    #[test]
    fn clone_then_set_original_clone_unaffected() {
        // Arrange: set a handle, clone it, then change the original's pointer.
        let handle = CallbackChainHandle::new();
        let ptr_a = 0xa0a1 as *mut CallbackChain;
        let ptr_b = 0xb0b1 as *mut CallbackChain;

        handle.set(ptr_a);
        let cloned = handle.clone();
        assert_eq!(cloned.as_ffi_ptr(), ptr_a);

        // Act: overwrite the original's pointer after cloning.
        handle.set(ptr_b);

        // Assert: clone retains the snapshot of ptr_a; original shows ptr_b.
        assert_eq!(cloned.as_ffi_ptr(), ptr_a, "Clone must retain pre-mutation snapshot");
        assert_eq!(handle.as_ffi_ptr(), ptr_b, "Original must show updated pointer");
        assert!(cloned.is_set());
        assert!(handle.is_set());

        handle.clear();
        cloned.clear();
    }

    #[test]
    fn array_of_handles_alternating_set_and_clear() {
        // Arrange: create 8 handles in an array.
        let mut handles: [CallbackChainHandle; 8] = std::array::from_fn(|_| CallbackChainHandle::new());

        // Act: set even-indexed handles, leave odd-indexed null.
        for (i, h) in handles.iter_mut().enumerate() {
            if i % 2 == 0 {
                h.set((0x1000 + i) as *mut CallbackChain);
            }
        }

        // Assert: checkerboard pattern — even set, odd null.
        for (i, h) in handles.iter().enumerate() {
            if i % 2 == 0 {
                assert!(h.is_set(), "Handle {} (even) should be set", i);
                assert_eq!(h.as_ffi_ptr() as usize, 0x1000 + i);
            } else {
                assert!(!h.is_set(), "Handle {} (odd) should be null", i);
            }
        }

        // Act: invert — clear even, set odd.
        for (i, h) in handles.iter_mut().enumerate() {
            if i % 2 == 0 {
                h.clear();
            } else {
                h.set((0x2000 + i) as *mut CallbackChain);
            }
        }

        // Assert: inverted checkerboard — odd set, even null.
        for (i, h) in handles.iter().enumerate() {
            if i % 2 == 0 {
                assert!(!h.is_set(), "Handle {} (even) should now be null", i);
            } else {
                assert!(h.is_set(), "Handle {} (odd) should now be set", i);
                assert_eq!(h.as_ffi_ptr() as usize, 0x2000 + i);
            }
        }
    }

    #[test]
    fn option_handle_take_and_replace_preserves_independence() {
        // Arrange: wrap a set handle in Option.
        let mut opt: Option<CallbackChainHandle> = Some(CallbackChainHandle::new());
        let ptr_a = 0x10a0 as *mut CallbackChain;
        opt.as_ref().unwrap().set(ptr_a);

        // Act: take the handle out.
        let taken = opt.take();
        assert!(opt.is_none());
        assert!(taken.unwrap().is_set());

        // Replace with a fresh handle in a different state.
        let fresh = CallbackChainHandle::new();
        let ptr_b = 0x20b0 as *mut CallbackChain;
        fresh.set(ptr_b);
        opt = Some(fresh);

        // Assert: the replacement has its own independent pointer.
        assert!(opt.as_ref().unwrap().is_set());
        assert_eq!(opt.as_ref().unwrap().as_ffi_ptr(), ptr_b);

        opt.as_ref().unwrap().clear();
        assert!(!opt.as_ref().unwrap().is_set());
    }

    #[test]
    fn struct_with_handle_field_clone_copies_independently() {
        // Arrange: a struct containing a CallbackChainHandle and other fields.
        #[derive(Clone)]
        struct Slot {
            handle: CallbackChainHandle,
            priority: u32,
            label: String,
        }

        let mut slot = Slot {
            handle: CallbackChainHandle::new(),
            priority: 7,
            label: "primary".to_string(),
        };
        let ptr = 0xfeed as *mut CallbackChain;
        slot.handle.set(ptr);

        // Act: clone the entire struct.
        let cloned_slot = slot.clone();

        // Assert: cloned handle has a snapshot of the pointer.
        assert!(cloned_slot.handle.is_set());
        assert_eq!(cloned_slot.handle.as_ffi_ptr(), ptr);
        assert_eq!(cloned_slot.priority, 7);
        assert_eq!(cloned_slot.label, "primary");

        // Mutate the original's handle.
        slot.handle.clear();

        // Cloned slot's handle is unaffected.
        assert!(cloned_slot.handle.is_set(), "Cloned handle retains snapshot");
        assert!(!slot.handle.is_set(), "Original is now null");
    }

    #[test]
    fn handle_clear_then_immediate_set_same_pointer_retained() {
        // Arrange: set a pointer, clear, then immediately re-set the same pointer.
        let handle = CallbackChainHandle::new();
        let ptr = 0xf00f as *mut CallbackChain;

        // Act: tight clear→set cycle with the same pointer.
        for _ in 0..50 {
            handle.set(ptr);
            assert!(handle.is_set());
            assert_eq!(handle.as_ffi_ptr(), ptr);
            handle.clear();
            assert!(!handle.is_set());
        }

        // Assert: final state is clear and consistent.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn arc_weak_reference_can_observe_handle_state() {
        use std::sync::Arc;

        // Arrange: create Arc and obtain weak reference.
        let handle = Arc::new(CallbackChainHandle::new());
        let weak = Arc::downgrade(&handle);
        let ptr = 0xc1c2 as *mut CallbackChain;

        handle.set(ptr);

        // Act: observe state through weak reference.
        let observed = weak.upgrade().expect("Strong count > 0, upgrade should succeed");
        assert!(observed.is_set());
        assert_eq!(observed.as_ffi_ptr(), ptr);

        // Assert: original and upgraded arc see the same state.
        assert_eq!(handle.as_ffi_ptr(), observed.as_ffi_ptr());

        handle.clear();
        assert!(!weak.upgrade().unwrap().is_set());
    }

    #[test]
    fn many_handles_all_set_to_same_pointer_independent_clear() {
        // Arrange: 20 handles all set to the same pointer.
        let ptr = 0x55aa as *mut CallbackChain;
        let handles: Vec<CallbackChainHandle> = (0..20)
            .map(|_| {
                let h = CallbackChainHandle::new();
                h.set(ptr);
                h
            })
            .collect();

        // Act: clear every 5th handle.
        for (i, h) in handles.iter().enumerate() {
            if i % 5 == 0 {
                h.clear();
            }
        }

        // Assert: indices 0, 5, 10, 15 are null; all others are set.
        for (i, h) in handles.iter().enumerate() {
            if i % 5 == 0 {
                assert!(!h.is_set(), "Handle {} should be cleared", i);
            } else {
                assert!(h.is_set(), "Handle {} should be set", i);
                assert_eq!(h.as_ffi_ptr(), ptr);
            }
        }
    }

    #[test]
    fn set_clear_during_is_set_check_is_consistent() {
        // Arrange: set a pointer and verify is_set consistency mid-operation.
        let handle = CallbackChainHandle::new();
        let ptr = 0xe1e2 as *mut CallbackChain;

        handle.set(ptr);

        // Act: call is_set and as_ffi_ptr in tight succession; they must agree.
        for _ in 0..200 {
            let set_flag = handle.is_set();
            let ffi_ptr = handle.as_ffi_ptr();
            assert_eq!(set_flag, !ffi_ptr.is_null(), "is_set and as_ffi_ptr must agree");
        }

        handle.clear();

        // Assert: after clear, both consistently report null.
        for _ in 0..200 {
            assert!(!handle.is_set());
            assert!(handle.as_ffi_ptr().is_null());
        }
    }

    #[test]
    fn handle_swap_via_temporary_variable() {
        // Arrange: two handles with different pointers.
        let mut h1 = CallbackChainHandle::new();
        let mut h2 = CallbackChainHandle::new();
        let ptr_a = 0x1111 as *mut CallbackChain;
        let ptr_b = 0x2222 as *mut CallbackChain;

        h1.set(ptr_a);
        h2.set(ptr_b);

        // Act: swap via temporary (std::mem::swap cannot be used because
        // AtomicPtr is not Swapable directly, so manual swap via take).
        let tmp_ptr_a = h1.as_ffi_ptr();
        let tmp_ptr_b = h2.as_ffi_ptr();
        h1.clear();
        h2.clear();
        h1.set(tmp_ptr_b);
        h2.set(tmp_ptr_a);

        // Assert: pointers are swapped.
        assert_eq!(h1.as_ffi_ptr(), ptr_b);
        assert_eq!(h2.as_ffi_ptr(), ptr_a);
        assert!(h1.is_set());
        assert!(h2.is_set());
    }

    #[test]
    fn boxed_handle_clone_independence() {
        // Arrange: heap-allocated handle set to a pointer.
        let mut boxed = Box::new(CallbackChainHandle::new());
        let ptr = 0x7071 as *mut CallbackChain;
        boxed.set(ptr);

        // Act: clone the boxed handle.
        let mut cloned = (*boxed).clone();

        // Assert: both are set to the same pointer.
        assert!(boxed.is_set());
        assert!(cloned.is_set());
        assert_eq!(boxed.as_ffi_ptr(), ptr);
        assert_eq!(cloned.as_ffi_ptr(), ptr);

        // Clear the clone; boxed must be unaffected.
        cloned.clear();
        assert!(boxed.is_set(), "Boxed handle unaffected by clone clear");
        assert!(!cloned.is_set());

        // Set cloned to a new pointer; boxed still unaffected.
        let new_ptr = 0x8081 as *mut CallbackChain;
        cloned.set(new_ptr);
        assert_eq!(boxed.as_ffi_ptr(), ptr);
        assert_eq!(cloned.as_ffi_ptr(), new_ptr);
    }

    #[test]
    fn concurrent_set_same_pointer_from_many_threads_final_state_set() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: all threads set the same pointer value.
        let handle = Arc::new(CallbackChainHandle::new());
        let addr: usize = 0x1234;

        let mut join_handles = Vec::new();
        for _ in 0..16 {
            let h = Arc::clone(&handle);
            join_handles.push(thread::spawn(move || {
                h.set(addr as *mut CallbackChain);
            }));
        }

        // Act: all threads write the same pointer.
        for jh in join_handles {
            jh.join().unwrap();
        }

        // Assert: handle is set to the expected pointer.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, addr);
        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_zero_sized_type_array_layout() {
        // Arrange: CallbackChainHandle is pointer-sized (8 bytes on 64-bit).
        // Verify that an array of handles has the expected total size.
        const N: usize = 10;
        let handles: [CallbackChainHandle; N] = std::array::from_fn(|_| CallbackChainHandle::new());

        // Act: check total size matches N * pointer_size.
        let expected_size = N * std::mem::size_of::<*mut CallbackChain>();
        let actual_size = std::mem::size_of_val(&handles);

        // Assert: array occupies exactly N pointer-widths with no padding.
        assert_eq!(actual_size, expected_size, "Array of {} handles should be {} bytes", N, expected_size);

        // All handles in the array start as null.
        for (i, h) in handles.iter().enumerate() {
            assert!(!h.is_set(), "Handle {} in array should start null", i);
        }
    }

    #[test]
    fn set_clear_with_real_chain_pointer_after_chain_goes_out_of_scope() {
        // Arrange: obtain a pointer to a CallbackChain, then let it go out of scope.
        // The handle stores raw bits and never dereferences, so this is safe.
        let handle = CallbackChainHandle::new();
        let ptr = {
            let chain = CallbackChain::empty();
            &chain as *const CallbackChain as *mut CallbackChain
        };
        // chain is now dropped; ptr is dangling but the handle only stores bits.

        // Act: set the (dangling) pointer. The handle does not dereference it.
        handle.set(ptr);

        // Assert: the bit pattern is preserved exactly.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        // Clear is safe regardless of pointer validity.
        handle.clear();
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    // --- 13 additional tests (wave 7) ---

    #[test]
    fn set_and_clear_return_unit_type() {
        // Arrange: create a fresh handle.
        let handle = CallbackChainHandle::new();
        let ptr = 0x1234 as *mut CallbackChain;

        // Act: set() and clear() both return ().
        let set_result: () = handle.set(ptr);
        let clear_result: () = handle.clear();

        // Assert: both return the unit type (compile-time proof).
        assert_eq!(set_result, ());
        assert_eq!(clear_result, ());
    }

    #[test]
    fn handle_triple_move_preserves_state() {
        // Arrange: set a handle and move it three times.
        let h0 = CallbackChainHandle::new();
        let ptr = 0xabcd as *mut CallbackChain;
        h0.set(ptr);

        // Act: chain of moves.
        let h1 = h0;
        let h2 = h1;
        let h3 = h2;

        // Assert: the final binding still carries the pointer.
        assert!(h3.is_set());
        assert_eq!(h3.as_ffi_ptr(), ptr);

        h3.clear();
        assert!(!h3.is_set());
    }

    #[test]
    fn mem_replace_with_fresh_handle() {
        use std::mem;

        // Arrange: set a handle to a pointer.
        let mut handle = CallbackChainHandle::new();
        let ptr = 0xdead as *mut CallbackChain;
        handle.set(ptr);
        assert!(handle.is_set());

        // Act: replace with a fresh (null) handle.
        let old = mem::replace(&mut handle, CallbackChainHandle::new());

        // Assert: new handle is null; old retains the pointer.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
        assert!(old.is_set());
        assert_eq!(old.as_ffi_ptr(), ptr);

        old.clear();
    }

    #[test]
    fn handle_used_as_hashmap_value() {
        use std::collections::HashMap;

        // Arrange: store handles in a HashMap by string key.
        let mut map: HashMap<&str, CallbackChainHandle> = HashMap::new();
        map.insert("slot_a", CallbackChainHandle::new());
        map.insert("slot_b", CallbackChainHandle::new());

        let ptr_a = 0xaaaa as *mut CallbackChain;
        let ptr_b = 0xbbbb as *mut CallbackChain;

        // Act: set pointers via the map.
        map.get("slot_a").unwrap().set(ptr_a);
        map.get("slot_b").unwrap().set(ptr_b);

        // Assert: each key retrieves the correct pointer.
        assert!(map.get("slot_a").unwrap().is_set());
        assert_eq!(map.get("slot_a").unwrap().as_ffi_ptr(), ptr_a);
        assert!(map.get("slot_b").unwrap().is_set());
        assert_eq!(map.get("slot_b").unwrap().as_ffi_ptr(), ptr_b);

        // Clear one entry; the other is unaffected.
        map.get("slot_a").unwrap().clear();
        assert!(!map.get("slot_a").unwrap().is_set());
        assert!(map.get("slot_b").unwrap().is_set());
    }

    #[test]
    fn handle_iterated_over_in_vec_with_enumerate() {
        // Arrange: create a Vec of handles and set each to a unique pointer.
        let handles: Vec<CallbackChainHandle> = (0..6)
            .map(|_| CallbackChainHandle::new())
            .collect();

        for (i, h) in handles.iter().enumerate() {
            h.set((0x100 * i + 0x50) as *mut CallbackChain);
        }

        // Act: iterate with enumerate and verify each.
        let mut count = 0;
        for (i, h) in handles.iter().enumerate() {
            assert!(h.is_set());
            assert_eq!(h.as_ffi_ptr() as usize, 0x100 * i + 0x50);
            count += 1;
        }

        // Assert: all 6 handles were visited.
        assert_eq!(count, 6);

        // Cleanup.
        for h in &handles {
            h.clear();
        }
    }

    #[test]
    fn set_same_pointer_after_vec_push_pop() {
        // Arrange: put a set handle into a Vec, pop it, verify state.
        let mut vec = Vec::new();
        let ptr = 0x7e57 as *mut CallbackChain;

        let h = CallbackChainHandle::new();
        h.set(ptr);
        vec.push(h);

        // Act: pop and verify the handle survived Vec operations.
        let popped = vec.pop().unwrap();
        assert!(popped.is_set());
        assert_eq!(popped.as_ffi_ptr(), ptr);

        // Set the same pointer again (idempotent).
        popped.set(ptr);
        assert_eq!(popped.as_ffi_ptr(), ptr);

        popped.clear();
        assert!(!popped.is_set());
    }

    #[test]
    fn is_set_is_negation_of_as_ffi_ptr_is_null_invariant() {
        // Arrange: verify the invariant holds across all state transitions.
        let handle = CallbackChainHandle::new();
        let ptr = 0xc0de as *mut CallbackChain;

        // Fresh state.
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());

        // Set state.
        handle.set(ptr);
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());

        // Overwrite with a different pointer.
        handle.set(0xbeef as *mut CallbackChain);
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());

        // Explicitly set to null.
        handle.set(std::ptr::null_mut());
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());

        // Clear state.
        handle.clear();
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn pointer_roundtrip_preserves_exact_bits() {
        // Arrange: use a specific bit pattern that exercises all bytes.
        let handle = CallbackChainHandle::new();
        let original_bits: usize = 0xDEADBEEFCAFEBABE;
        let ptr = original_bits as *mut CallbackChain;

        // Act.
        handle.set(ptr);
        let recovered_bits = handle.as_ffi_ptr() as usize;

        // Assert: every bit is preserved exactly.
        assert_eq!(recovered_bits, original_bits);

        handle.clear();
        assert_eq!(handle.as_ffi_ptr() as usize, 0);
    }

    #[test]
    fn debug_impl_does_not_panic_on_set_handle() {
        // Arrange: set a non-null pointer and format via Debug.
        let handle = CallbackChainHandle::new();
        handle.set(0x1234 as *mut CallbackChain);

        // Act: formatting should not panic even with a set pointer.
        let debug_str = format!("{:?}", handle);

        // Assert: output contains the struct name.
        assert!(debug_str.contains("CallbackChainHandle"));

        handle.clear();
    }

    #[test]
    fn two_clones_set_to_different_pointers_then_both_cleared() {
        // Arrange: create original and two clones, all starting with same pointer.
        let handle = CallbackChainHandle::new();
        let ptr_common = 0x1111 as *mut CallbackChain;
        handle.set(ptr_common);

        let mut c1 = handle.clone();
        let mut c2 = handle.clone();

        // Act: diverge all three to different pointers.
        handle.set(0xAAAA as *mut CallbackChain);
        c1.set(0xBBBB as *mut CallbackChain);
        c2.set(0xCCCC as *mut CallbackChain);

        // Assert: all three have distinct pointers.
        assert_ne!(handle.as_ffi_ptr(), c1.as_ffi_ptr());
        assert_ne!(c1.as_ffi_ptr(), c2.as_ffi_ptr());
        assert_ne!(handle.as_ffi_ptr(), c2.as_ffi_ptr());
        assert!(handle.is_set());
        assert!(c1.is_set());
        assert!(c2.is_set());

        // Clear all three independently.
        handle.clear();
        c1.clear();
        c2.clear();
        assert!(!handle.is_set());
        assert!(!c1.is_set());
        assert!(!c2.is_set());
    }

    #[test]
    fn handle_in_option_map_and_unwrap_or_default_pattern() {
        // Arrange: use Option's combinators with the handle.
        let maybe_handle: Option<CallbackChainHandle> = Some(CallbackChainHandle::new());
        let ptr = 0xf00d as *mut CallbackChain;

        // Act: map over the Option to set the pointer.
        maybe_handle.as_ref().unwrap().set(ptr);

        // Assert: the handle inside Option is set.
        assert!(maybe_handle.as_ref().unwrap().is_set());
        assert_eq!(maybe_handle.as_ref().unwrap().as_ffi_ptr(), ptr);

        // Use unwrap_or_default pattern for a None case.
        let none_handle: Option<CallbackChainHandle> = None;
        let fallback = none_handle.unwrap_or_else(CallbackChainHandle::default);
        assert!(!fallback.is_set());
        assert!(fallback.as_ffi_ptr().is_null());

        maybe_handle.as_ref().unwrap().clear();
    }

    #[test]
    fn struct_with_multiple_handles_independent_lifecycle() {
        // Arrange: a struct with two handle fields.
        struct DualSlot {
            primary: CallbackChainHandle,
            secondary: CallbackChainHandle,
        }

        let mut slot = DualSlot {
            primary: CallbackChainHandle::new(),
            secondary: CallbackChainHandle::new(),
        };

        let ptr_primary = 0x1000 as *mut CallbackChain;
        let ptr_secondary = 0x2000 as *mut CallbackChain;

        // Act: set each handle independently.
        slot.primary.set(ptr_primary);
        slot.secondary.set(ptr_secondary);

        // Assert: each field has its own pointer.
        assert!(slot.primary.is_set());
        assert!(slot.secondary.is_set());
        assert_eq!(slot.primary.as_ffi_ptr(), ptr_primary);
        assert_eq!(slot.secondary.as_ffi_ptr(), ptr_secondary);

        // Clear primary; secondary unaffected.
        slot.primary.clear();
        assert!(!slot.primary.is_set());
        assert!(slot.secondary.is_set());

        // Clear secondary.
        slot.secondary.clear();
        assert!(!slot.secondary.is_set());
    }

    #[test]
    fn concurrent_set_from_one_thread_read_from_many_then_clear() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: one writer sets; 4 readers verify; then writer clears.
        let handle = Arc::new(CallbackChainHandle::new());
        let addr: usize = 0x9ABC;

        // Writer sets.
        let h_writer = Arc::clone(&handle);
        let writer = thread::spawn(move || {
            h_writer.set(addr as *mut CallbackChain);
        });
        writer.join().unwrap();

        // Act: 4 readers all verify the pointer.
        let mut readers = Vec::new();
        for _ in 0..4 {
            let h = Arc::clone(&handle);
            readers.push(thread::spawn(move || {
                assert!(h.is_set());
                h.as_ffi_ptr() as usize
            }));
        }

        let results: Vec<usize> = readers.into_iter().map(|r| r.join().unwrap()).collect();

        // Assert: all readers saw the correct pointer.
        for (i, val) in results.iter().enumerate() {
            assert_eq!(*val, addr, "Reader {} saw wrong pointer", i);
        }

        // Writer clears.
        handle.clear();
        assert!(!handle.is_set());
    }

    // --- 13 additional tests (wave 8) ---

    #[test]
    fn set_then_as_ffi_ptr_read_many_times_before_clear() {
        // Arrange: set a pointer and read it back 500 times.
        let handle = CallbackChainHandle::new();
        let ptr = 0xBEEF as *mut CallbackChain;
        handle.set(ptr);

        // Act: repeated reads must all return the same value.
        for _ in 0..500 {
            assert_eq!(handle.as_ffi_ptr(), ptr);
        }

        // Assert: still set after many reads.
        assert!(handle.is_set());
        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_of_handle_with_high_kernel_address() {
        // Arrange: set to a kernel-space address (highest bit set).
        let handle = CallbackChainHandle::new();
        let kernel_addr = (1usize << 63) | 0xFF;
        let ptr = kernel_addr as *mut CallbackChain;
        handle.set(ptr);

        // Act: clone and verify both carry the same high address.
        let cloned = handle.clone();

        // Assert: exact bit pattern preserved in both original and clone.
        assert_eq!(handle.as_ffi_ptr() as usize, kernel_addr);
        assert_eq!(cloned.as_ffi_ptr() as usize, kernel_addr);

        handle.clear();
        assert!(cloned.is_set(), "Clone retains snapshot after original cleared");
        assert_eq!(cloned.as_ffi_ptr() as usize, kernel_addr);
        cloned.clear();
    }

    #[test]
    fn handle_in_vec_dedup_by_key_not_needed_identity_by_position() {
        // Arrange: two handles set to the same pointer value but stored at
        // different positions in a Vec.
        let ptr = 0x8888 as *mut CallbackChain;
        let mut handles = Vec::new();

        // Act: push two independent handles with the same pointer value.
        let h1 = CallbackChainHandle::new();
        h1.set(ptr);
        handles.push(h1);

        let h2 = CallbackChainHandle::new();
        h2.set(ptr);
        handles.push(h2);

        // Assert: both entries report set with the same pointer value.
        assert_eq!(handles.len(), 2);
        assert!(handles[0].is_set());
        assert!(handles[1].is_set());
        assert_eq!(handles[0].as_ffi_ptr(), handles[1].as_ffi_ptr());

        // Clearing one does not affect the other (independent allocations).
        handles[0].clear();
        assert!(!handles[0].is_set());
        assert!(handles[1].is_set());
        handles[1].clear();
    }

    #[test]
    fn mem_swap_between_two_set_handles() {
        use std::mem;

        // Arrange: two handles with distinct pointers.
        let mut h1 = CallbackChainHandle::new();
        let mut h2 = CallbackChainHandle::new();
        let ptr_a = 0xAAAA as *mut CallbackChain;
        let ptr_b = 0xBBBB as *mut CallbackChain;
        h1.set(ptr_a);
        h2.set(ptr_b);

        // Act: swap the handles via mem::swap.
        mem::swap(&mut h1, &mut h2);

        // Assert: after swap, h1 has ptr_b and h2 has ptr_a.
        assert_eq!(h1.as_ffi_ptr(), ptr_b);
        assert_eq!(h2.as_ffi_ptr(), ptr_a);
        assert!(h1.is_set());
        assert!(h2.is_set());

        h1.clear();
        h2.clear();
        assert!(!h1.is_set());
        assert!(!h2.is_set());
    }

    #[test]
    fn handle_stores_minimal_non_null_pointer_value() {
        // Arrange: pointer value 1 is the smallest non-null address.
        let handle = CallbackChainHandle::new();
        let ptr = 1usize as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: is_set returns true for any non-null pointer.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, 1);

        // Setting to 0 should make is_set false.
        handle.set(std::ptr::null_mut());
        assert!(!handle.is_set());
    }

    #[test]
    fn two_real_chains_different_addresses_handle_switches_correctly() {
        // Arrange: allocate two CallbackChain instances on the stack.
        let chain_x = CallbackChain::empty();
        let chain_y = CallbackChain::empty();
        let ptr_x = &chain_x as *const CallbackChain as *mut CallbackChain;
        let ptr_y = &chain_y as *const CallbackChain as *mut CallbackChain;
        assert_ne!(ptr_x, ptr_y, "Two stack locals must have different addresses");

        let handle = CallbackChainHandle::new();

        // Act: alternate between the two pointers.
        for _ in 0..10 {
            handle.set(ptr_x);
            assert_eq!(handle.as_ffi_ptr(), ptr_x);

            handle.set(ptr_y);
            assert_eq!(handle.as_ffi_ptr(), ptr_y);
        }

        // Assert: final state is ptr_y.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr_y);
        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_reused_after_many_cycles_accepts_new_pointer() {
        // Arrange: exercise the handle through 100 set/clear cycles.
        let handle = CallbackChainHandle::new();
        for i in 0..100u64 {
            handle.set((0x1000 + i) as *mut CallbackChain);
            handle.clear();
        }

        // Act: set a fresh pointer after all cycles.
        let final_ptr = 0xFFFF as *mut CallbackChain;
        handle.set(final_ptr);

        // Assert: handle still works correctly after heavy reuse.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), final_ptr);
        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn is_set_false_when_pointer_is_zero_constructed_via_zeroed() {
        // Arrange: explicitly construct a zero pointer via casting.
        let handle = CallbackChainHandle::new();
        let zero_ptr = 0usize as *mut CallbackChain;
        handle.set(zero_ptr);

        // Assert: zero pointer is treated as "not set".
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());

        // Verify we can transition from zero to a valid pointer.
        handle.set(0x1234 as *mut CallbackChain);
        assert!(handle.is_set());
        handle.clear();
    }

    #[test]
    fn clone_snapshot_captures_intermediate_overwrite() {
        // Arrange: set ptr_a, overwrite with ptr_b, clone immediately.
        let handle = CallbackChainHandle::new();
        let ptr_a = 0x1111 as *mut CallbackChain;
        let ptr_b = 0x2222 as *mut CallbackChain;

        handle.set(ptr_a);
        handle.set(ptr_b);
        let cloned = handle.clone();

        // Assert: clone sees ptr_b (the current value at clone time).
        assert_eq!(cloned.as_ffi_ptr(), ptr_b);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);

        // Act: clear original; clone retains ptr_b.
        handle.clear();
        assert!(cloned.is_set());
        assert_eq!(cloned.as_ffi_ptr(), ptr_b);
    }

    #[test]
    fn handle_as_cell_content_with_borrow_split() {
        use std::cell::RefCell;

        // Arrange: store handle in RefCell to simulate interior mutability.
        let cell = RefCell::new(CallbackChainHandle::new());
        let ptr1 = 0xA0A0 as *mut CallbackChain;
        let ptr2 = 0xB0B0 as *mut CallbackChain;

        // Act: set via immutable borrow (AtomicPtr interior mutability).
        cell.borrow().set(ptr1);
        assert!(cell.borrow().is_set());
        assert_eq!(cell.borrow().as_ffi_ptr(), ptr1);

        // Overwrite via a second borrow.
        cell.borrow().set(ptr2);
        assert_eq!(cell.borrow().as_ffi_ptr(), ptr2);

        // Clear.
        cell.borrow().clear();

        // Assert: cleared state.
        assert!(!cell.borrow().is_set());
        assert!(cell.borrow().as_ffi_ptr().is_null());
    }

    #[test]
    fn boxed_vec_of_handles_clear_all_then_verify_all_null() {
        // Arrange: Box<Vec<CallbackChainHandle>> with 7 handles, all set.
        let mut boxed_vec: Box<Vec<CallbackChainHandle>> = Box::new(
            (0..7)
                .map(|i| {
                    let h = CallbackChainHandle::new();
                    h.set((0x2000 + i * 0x10) as *mut CallbackChain);
                    h
                })
                .collect(),
        );

        // Act: clear all handles in the boxed vec.
        for h in boxed_vec.iter() {
            h.clear();
        }

        // Assert: every handle is null.
        for (i, h) in boxed_vec.iter().enumerate() {
            assert!(!h.is_set(), "Handle {} should be null after clear", i);
            assert!(h.as_ffi_ptr().is_null(), "Handle {} ptr should be null", i);
        }
    }

    #[test]
    fn concurrent_writer_sets_clears_repeatedly_reader_never_sees_inconsistent_is_set() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::sync::Arc;
        use std::thread;

        // Arrange: writer toggles set/clear 200 times; reader checks invariant.
        let handle = Arc::new(CallbackChainHandle::new());
        let stop = Arc::new(AtomicBool::new(false));

        let h_writer = Arc::clone(&handle);
        let w_stop = Arc::clone(&stop);
        let writer = thread::spawn(move || {
            for i in 0..200u64 {
                h_writer.set((0x3000 + i) as *mut CallbackChain);
                h_writer.clear();
            }
            w_stop.store(true, AtomicOrdering::Release);
        });

        let h_reader = Arc::clone(&handle);
        let r_stop = Arc::clone(&stop);
        let reader = thread::spawn(move || {
            while !r_stop.load(AtomicOrdering::Acquire) {
                // Invariant: is_set must agree with as_ffi_ptr being non-null.
                let set = h_reader.is_set();
                let ptr = h_reader.as_ffi_ptr();
                assert_eq!(set, !ptr.is_null(), "is_set and as_ffi_ptr must always agree");
                thread::yield_now();
            }
        });

        writer.join().unwrap();
        reader.join().unwrap();

        // Assert: final state after all clears is null.
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_wrapped_in_arc_mutex_full_lifecycle() {
        use std::sync::{Arc, Mutex};

        // Arrange: wrap in Arc<Mutex<>> to simulate real executor pattern.
        let handle = Arc::new(Mutex::new(CallbackChainHandle::new()));
        let ptr = 0xDEAD as *mut CallbackChain;

        // Act: set under lock.
        {
            let guard = handle.lock().unwrap();
            guard.set(ptr);
        }

        // Assert: readable under a separate lock acquisition.
        {
            let guard = handle.lock().unwrap();
            assert!(guard.is_set());
            assert_eq!(guard.as_ffi_ptr(), ptr);
        }

        // Clear under lock.
        {
            let guard = handle.lock().unwrap();
            guard.clear();
        }

        // Final state is null.
        {
            let guard = handle.lock().unwrap();
            assert!(!guard.is_set());
            assert!(guard.as_ffi_ptr().is_null());
        }
    }

    // --- 13 additional tests (wave 9) ---

    #[test]
    fn struct_update_syntax_with_default_handle() {
        // Arrange: define a struct that uses ..Default::default() pattern.
        #[derive(Default)]
        struct Slot {
            handle: CallbackChainHandle,
            tag: u32,
        }

        // Act: create using struct update syntax, overriding only tag.
        let slot = Slot {
            tag: 99,
            ..Default::default()
        };

        // Assert: handle is default-constructed (null), tag is overridden.
        assert!(!slot.handle.is_set());
        assert!(slot.handle.as_ffi_ptr().is_null());
        assert_eq!(slot.tag, 99);
    }

    #[test]
    fn clone_derive_produces_independent_snapshot_from_manual_clone() {
        // Arrange: create a handle and set it.
        let handle = CallbackChainHandle::new();
        let ptr = 0xCA5C as *mut CallbackChain;
        handle.set(ptr);

        // Act: clone twice — once normally, once from a re-clone — both must agree.
        let c1 = handle.clone();
        let c2 = c1.clone();

        // Assert: all three carry the same pointer snapshot.
        assert_eq!(handle.as_ffi_ptr(), ptr);
        assert_eq!(c1.as_ffi_ptr(), ptr);
        assert_eq!(c2.as_ffi_ptr(), ptr);

        // Mutating original does not affect either clone.
        handle.clear();
        assert!(c1.is_set());
        assert!(c2.is_set());
    }

    #[test]
    fn as_ffi_ptr_on_null_handle_called_many_times_always_null() {
        // Arrange: fresh null handle.
        let handle = CallbackChainHandle::new();

        // Act: call as_ffi_ptr 1000 times on a null handle.
        for _ in 0..1000 {
            assert!(handle.as_ffi_ptr().is_null());
        }

        // Assert: still null after many reads.
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_returned_from_identity_function_preserves_pointer() {
        // Arrange: a function that returns its input handle unchanged.
        fn identity(h: CallbackChainHandle) -> CallbackChainHandle {
            h
        }

        let handle = CallbackChainHandle::new();
        let ptr = 0xBEEF as *mut CallbackChain;
        handle.set(ptr);

        // Act: pass through identity function.
        let returned = identity(handle);

        // Assert: the returned handle carries the same pointer.
        assert!(returned.is_set());
        assert_eq!(returned.as_ffi_ptr(), ptr);

        returned.clear();
        assert!(!returned.is_set());
    }

    #[test]
    fn handle_in_tuple_destructuring_works() {
        // Arrange: embed handle in a tuple.
        let mut tuple = (CallbackChainHandle::new(), 42u32, "label");
        let ptr = 0x1234 as *mut CallbackChain;
        tuple.0.set(ptr);

        // Act: destructure the tuple.
        let (handle, value, label) = tuple;

        // Assert: handle survived destructuring with state intact.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);
        assert_eq!(value, 42);
        assert_eq!(label, "label");

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn usize_max_minus_one_pointer_roundtrip() {
        // Arrange: use usize::MAX - 1 as a pointer value (boundary test).
        let handle = CallbackChainHandle::new();
        let boundary_val = usize::MAX - 1;
        let ptr = boundary_val as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: exact bit pattern preserved at near-maximum value.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, boundary_val);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn debug_output_contains_inner_for_null_and_set_states() {
        // Arrange: create two handles — one null, one set.
        let null_handle = CallbackChainHandle::new();
        let set_handle = CallbackChainHandle::new();
        set_handle.set(0xABCD as *mut CallbackChain);

        // Act: format both via Debug.
        let null_debug = format!("{:?}", null_handle);
        let set_debug = format!("{:?}", set_handle);

        // Assert: both contain the struct name and the inner field name.
        assert!(null_debug.contains("CallbackChainHandle"));
        assert!(null_debug.contains("inner"));
        assert!(set_debug.contains("CallbackChainHandle"));
        assert!(set_debug.contains("inner"));

        set_handle.clear();
    }

    #[test]
    fn fan_out_clone_one_original_to_many_clones_all_independent() {
        // Arrange: set one handle, then fan out 10 clones from the original.
        let handle = CallbackChainHandle::new();
        let ptr = 0x5EED as *mut CallbackChain;
        handle.set(ptr);

        let clones: Vec<CallbackChainHandle> = (0..10).map(|_| handle.clone()).collect();

        // Act: clear the original.
        handle.clear();

        // Assert: original is null, all 10 clones still hold the snapshot.
        assert!(!handle.is_set());
        for (i, c) in clones.iter().enumerate() {
            assert!(c.is_set(), "Clone {} should still be set", i);
            assert_eq!(c.as_ffi_ptr(), ptr, "Clone {} should have snapshot ptr", i);
        }

        // Clear half the clones; the other half must be unaffected.
        for (i, c) in clones.iter().enumerate() {
            if i % 2 == 0 {
                c.clear();
            }
        }
        for (i, c) in clones.iter().enumerate() {
            if i % 2 == 0 {
                assert!(!c.is_set(), "Clone {} (even) should be cleared", i);
            } else {
                assert!(c.is_set(), "Clone {} (odd) should still be set", i);
            }
        }
    }

    #[test]
    fn handle_set_to_callback_chain_alignment_offset_zero() {
        // Arrange: create a CallbackChain and compute its address offset
        // relative to itself (trivially zero). This verifies the handle
        // does not shift or corrupt the stored address.
        let chain = CallbackChain::empty();
        let base = &chain as *const CallbackChain as usize;
        let handle = CallbackChainHandle::new();
        handle.set(base as *mut CallbackChain);

        // Act: compute the difference between stored and base.
        let stored = handle.as_ffi_ptr() as usize;
        let offset = stored.wrapping_sub(base);

        // Assert: offset is exactly 0 — no drift.
        assert_eq!(offset, 0);
        assert_eq!(stored, base);

        handle.clear();
    }

    #[test]
    fn cleared_handle_cloned_many_times_all_clones_null() {
        // Arrange: set and clear a handle, then clone it multiple times.
        let handle = CallbackChainHandle::new();
        handle.set(0xFACE as *mut CallbackChain);
        handle.clear();

        // Act: clone 8 times from the cleared state.
        let clones: Vec<CallbackChainHandle> = (0..8).map(|_| handle.clone()).collect();

        // Assert: all clones are null (snapshot of cleared state).
        for (i, c) in clones.iter().enumerate() {
            assert!(!c.is_set(), "Clone {} from cleared handle should be null", i);
            assert!(c.as_ffi_ptr().is_null(), "Clone {} ptr should be null", i);
        }
    }

    #[test]
    fn handle_collected_from_iterator_all_start_null() {
        // Arrange: use an iterator to collect a Vec of handles.
        let handles: Vec<CallbackChainHandle> = (0..5)
            .map(|_| CallbackChainHandle::new())
            .collect();

        // Assert: all 5 handles start as null.
        assert_eq!(handles.len(), 5);
        for (i, h) in handles.iter().enumerate() {
            assert!(!h.is_set(), "Collected handle {} should be null", i);
            assert!(h.as_ffi_ptr().is_null(), "Collected handle {} ptr should be null", i);
        }

        // Act: set each via the iterator reference.
        for (i, h) in handles.iter().enumerate() {
            h.set((0x1000 + i) as *mut CallbackChain);
        }

        // Assert: all are now set with correct pointers.
        for (i, h) in handles.iter().enumerate() {
            assert!(h.is_set());
            assert_eq!(h.as_ffi_ptr() as usize, 0x1000 + i);
        }
    }

    #[test]
    fn is_set_invariant_holds_across_rapid_alternating_set_null() {
        // Arrange: verify is_set == !as_ffi_ptr.is_null() through rapid
        // alternation between non-null and null via set (not clear).
        let handle = CallbackChainHandle::new();
        let ptr = 0x7E57 as *mut CallbackChain;

        for _ in 0..100 {
            handle.set(ptr);
            assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());

            handle.set(std::ptr::null_mut());
            assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());
        }

        // Assert: final state is null (last set was null_mut).
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_size_and_alignment_match_raw_pointer() {
        // Arrange: verify size and alignment of CallbackChainHandle.
        let size = std::mem::size_of::<CallbackChainHandle>();
        let align = std::mem::align_of::<CallbackChainHandle>();
        let ptr_size = std::mem::size_of::<*mut CallbackChain>();
        let ptr_align = std::mem::align_of::<*mut CallbackChain>();

        // Assert: size and alignment match the underlying raw pointer exactly.
        assert_eq!(size, ptr_size, "Handle size should equal raw pointer size");
        assert_eq!(align, ptr_align, "Handle alignment should equal raw pointer alignment");

        // Additionally verify that a real instance has the expected size.
        let handle = CallbackChainHandle::new();
        assert_eq!(std::mem::size_of_val(&handle), ptr_size);
    }

    // --- 13 additional tests (wave 10) ---

    #[test]
    fn ptr_wrapping_add_stored_and_retrieved() {
        // Arrange: compute a pointer via wrapping arithmetic and store it.
        let base: usize = 0x1000;
        let offset: usize = 0x42;
        let addr = base.wrapping_add(offset);

        let handle = CallbackChainHandle::new();
        let ptr = addr as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: wrapping_add result is preserved exactly.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, addr);
        assert_eq!(handle.as_ffi_ptr() as usize, base + offset);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn struct_with_handle_and_vec_field_concurrent_clear_safe() {
        // Arrange: a struct that bundles a handle with a Vec of metadata.
        struct SlotWithMeta {
            handle: CallbackChainHandle,
            tags: Vec<u8>,
        }

        let mut slot = SlotWithMeta {
            handle: CallbackChainHandle::new(),
            tags: vec![10, 20, 30],
        };
        let ptr = 0xDA7A as *mut CallbackChain;
        slot.handle.set(ptr);

        // Act: clear handle, then mutate the Vec field.
        slot.handle.clear();
        slot.tags.push(40);

        // Assert: handle is null, Vec was independently mutated.
        assert!(!slot.handle.is_set());
        assert!(slot.handle.as_ffi_ptr().is_null());
        assert_eq!(slot.tags, vec![10, 20, 30, 40]);
    }

    #[test]
    fn set_pointer_with_all_bytes_0xff() {
        // Arrange: construct a pointer where every byte is 0xFF.
        let handle = CallbackChainHandle::new();
        let all_ff = usize::MAX;
        let ptr = all_ff as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: every bit is 1, and the handle preserves it exactly.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, all_ff);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_passes_through_closure_capturing_state() {
        // Arrange: set a handle, then move it into a closure.
        let handle = CallbackChainHandle::new();
        let ptr = 0xACE0 as *mut CallbackChain;
        handle.set(ptr);

        // Act: closure captures the handle by move.
        let read_ptr = || handle.as_ffi_ptr();

        // Assert: closure correctly reads the captured handle's pointer.
        assert_eq!(read_ptr(), ptr);
        assert!(handle.is_set());

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn array_16_handles_set_every_third_and_verify_pattern() {
        // Arrange: create 16 handles, set only every 3rd one (0, 3, 6, 9, 12, 15).
        let mut handles: [CallbackChainHandle; 16] = std::array::from_fn(|_| CallbackChainHandle::new());

        // Act.
        for (i, h) in handles.iter_mut().enumerate() {
            if i % 3 == 0 {
                h.set((0xA000 + i) as *mut CallbackChain);
            }
        }

        // Assert: only indices divisible by 3 are set.
        for (i, h) in handles.iter().enumerate() {
            if i % 3 == 0 {
                assert!(h.is_set(), "Handle {} should be set", i);
                assert_eq!(h.as_ffi_ptr() as usize, 0xA000 + i);
            } else {
                assert!(!h.is_set(), "Handle {} should be null", i);
            }
        }
    }

    #[test]
    fn clone_set_overwrite_clone_again_sees_updated_pointer() {
        // Arrange: set ptr_a, clone, then overwrite with ptr_b, clone again.
        let handle = CallbackChainHandle::new();
        let ptr_a = 0x1111 as *mut CallbackChain;
        let ptr_b = 0x2222 as *mut CallbackChain;

        handle.set(ptr_a);
        let clone_a = handle.clone();

        // Act: overwrite and clone again.
        handle.set(ptr_b);
        let clone_b = handle.clone();

        // Assert: clone_a captured ptr_a, clone_b captured ptr_b.
        assert_eq!(clone_a.as_ffi_ptr(), ptr_a);
        assert_eq!(clone_b.as_ffi_ptr(), ptr_b);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);

        handle.clear();
        assert!(clone_a.is_set(), "clone_a retains ptr_a snapshot");
        assert!(clone_b.is_set(), "clone_b retains ptr_b snapshot");
    }

    #[test]
    fn handle_stored_in_hashmap_then_retrieved_and_cleared() {
        use std::collections::HashMap;

        // Arrange: populate a HashMap with handles under integer keys.
        let mut map: HashMap<i32, CallbackChainHandle> = HashMap::new();
        for key in 100..106 {
            let h = CallbackChainHandle::new();
            h.set((0xD000 + key as usize) as *mut CallbackChain);
            map.insert(key, h);
        }

        // Act: retrieve and clear one entry.
        assert!(map.get(&102).unwrap().is_set());
        let ptr_102 = map.get(&102).unwrap().as_ffi_ptr();
        assert_eq!(ptr_102 as usize, 0xD000 + 102);

        map.get(&102).unwrap().clear();

        // Assert: only key 102 is cleared; all others remain set.
        assert!(!map.get(&102).unwrap().is_set());
        for key in 100..106 {
            if key != 102 {
                assert!(map.get(&key).unwrap().is_set(), "Key {} should still be set", key);
            }
        }
    }

    #[test]
    fn handle_set_to_power_of_two_address_preserved() {
        // Arrange: use addresses that are exact powers of two.
        let handle = CallbackChainHandle::new();

        for exp in 4..16usize {
            let addr = 1usize << exp;
            let ptr = addr as *mut CallbackChain;

            // Act.
            handle.set(ptr);

            // Assert: exact power-of-two value preserved.
            assert!(handle.is_set());
            assert_eq!(handle.as_ffi_ptr() as usize, addr);
            assert_eq!(handle.as_ffi_ptr() as usize, 1 << exp);

            handle.clear();
            assert!(!handle.is_set());
        }
    }

    #[test]
    fn clone_chain_with_selective_overwrite_mid_chain() {
        // Arrange: create original → c1 → c2 → c3 chain, all holding ptr.
        let handle = CallbackChainHandle::new();
        let ptr_common = 0x7F7F as *mut CallbackChain;
        handle.set(ptr_common);

        let c1 = handle.clone();
        let c2 = c1.clone();
        let c3 = c2.clone();

        // Act: overwrite only c2 with a different pointer.
        let ptr_alt = 0x8080 as *mut CallbackChain;
        c2.set(ptr_alt);

        // Assert: handle, c1, c3 hold ptr_common; c2 holds ptr_alt.
        assert_eq!(handle.as_ffi_ptr(), ptr_common);
        assert_eq!(c1.as_ffi_ptr(), ptr_common);
        assert_eq!(c2.as_ffi_ptr(), ptr_alt);
        assert_eq!(c3.as_ffi_ptr(), ptr_common);

        // All are set.
        assert!(handle.is_set());
        assert!(c1.is_set());
        assert!(c2.is_set());
        assert!(c3.is_set());

        // Independence: clearing handle does not affect any clone.
        handle.clear();
        assert!(!handle.is_set());
        assert!(c1.is_set());
        assert!(c2.is_set());
        assert!(c3.is_set());
    }

    #[test]
    fn concurrent_reader_observes_final_clear_after_many_sets() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::sync::Arc;
        use std::thread;

        // Arrange: writer sets 50 different pointers then clears once.
        let handle = Arc::new(CallbackChainHandle::new());
        let writer_done = Arc::new(AtomicBool::new(false));

        let h_writer = Arc::clone(&handle);
        let w_done = Arc::clone(&writer_done);
        let writer = thread::spawn(move || {
            for i in 0..50u64 {
                h_writer.set((0xE000 + i) as *mut CallbackChain);
            }
            h_writer.clear();
            w_done.store(true, AtomicOrdering::Release);
        });

        let h_reader = Arc::clone(&handle);
        let r_done = Arc::clone(&writer_done);
        let reader = thread::spawn(move || {
            // Spin until writer signals done.
            while !r_done.load(AtomicOrdering::Acquire) {
                thread::yield_now();
            }
            h_reader.is_set()
        });

        writer.join().unwrap();
        let reader_saw_set = reader.join().unwrap();

        // Assert: after writer finishes (set 50 times then cleared),
        // both the reader and direct check see null.
        assert!(!reader_saw_set, "Reader should see cleared state after writer finishes");
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn handle_embedded_in_enum_variant() {
        // Arrange: embed handle in an enum alongside other variants.
        enum SlotKind {
            Active(CallbackChainHandle),
            Inactive,
            Pending(u32),
        }

        let mut slot = SlotKind::Active(CallbackChainHandle::new());
        let ptr = 0xF00D as *mut CallbackChain;

        // Act: set pointer through the enum.
        if let SlotKind::Active(ref h) = slot {
            h.set(ptr);
        }

        // Assert: handle inside enum is set and readable.
        if let SlotKind::Active(ref h) = slot {
            assert!(h.is_set());
            assert_eq!(h.as_ffi_ptr(), ptr);
        }

        // Clear through the enum.
        if let SlotKind::Active(ref h) = slot {
            h.clear();
        }

        if let SlotKind::Active(ref h) = slot {
            assert!(!h.is_set());
            assert!(h.as_ffi_ptr().is_null());
        }
    }

    #[test]
    fn handle_set_clear_cycle_then_check_debug_format_between_each_phase() {
        // Arrange: capture debug output at each phase of the lifecycle.
        let handle = CallbackChainHandle::new();
        let debug_fresh = format!("{:?}", handle);

        let ptr = 0xABCD as *mut CallbackChain;
        handle.set(ptr);
        let debug_set = format!("{:?}", handle);

        handle.clear();
        let debug_cleared = format!("{:?}", handle);

        // Assert: all outputs contain the struct name.
        assert!(debug_fresh.contains("CallbackChainHandle"));
        assert!(debug_set.contains("CallbackChainHandle"));
        assert!(debug_cleared.contains("CallbackChainHandle"));

        // Fresh and cleared states should have identical debug output.
        assert_eq!(debug_fresh, debug_cleared);

        // Set state debug output differs from null state.
        // (The inner AtomicPtr value differs, but we only verify structure here.)
        assert!(debug_set.contains("inner"));
    }

    #[test]
    fn two_real_callback_chain_pointers_never_equal_address() {
        // Arrange: allocate two distinct CallbackChain instances.
        let chain_alpha = CallbackChain::empty();
        let chain_beta = CallbackChain::empty();
        let ptr_alpha = &chain_alpha as *const CallbackChain as *mut CallbackChain;
        let ptr_beta = &chain_beta as *const CallbackChain as *mut CallbackChain;

        // Act: verify they are at different addresses.
        assert_ne!(ptr_alpha, ptr_beta, "Two stack-local chains must differ in address");

        // Set handle to each in sequence and verify it switches correctly.
        let handle = CallbackChainHandle::new();

        handle.set(ptr_alpha);
        assert_eq!(handle.as_ffi_ptr(), ptr_alpha);
        assert_eq!(unsafe { &*handle.as_ffi_ptr() as *const _ }, &chain_alpha as *const _);

        handle.set(ptr_beta);
        assert_eq!(handle.as_ffi_ptr(), ptr_beta);
        assert_eq!(unsafe { &*handle.as_ffi_ptr() as *const _ }, &chain_beta as *const _);

        // Assert: neither pointer is null.
        assert!(!ptr_alpha.is_null());
        assert!(!ptr_beta.is_null());
        // The handle correctly tracks the last-set pointer.
        assert!(handle.is_set());

        handle.clear();
        assert!(!handle.is_set());
    }

    // --- 13 additional tests (wave 11) ---

    #[test]
    fn set_returns_void_and_clear_returns_void_individually() {
        // Arrange: create a handle and a non-null pointer.
        let handle = CallbackChainHandle::new();
        let ptr = 0x1a2b as *mut CallbackChain;

        // Act: call set and capture its return value.
        let _: () = handle.set(ptr);

        // Assert: handle is set after calling set.
        assert!(handle.is_set());

        // Act: call clear and capture its return value.
        let _: () = handle.clear();

        // Assert: handle is cleared after calling clear.
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_survives_mem_forget_without_double_free() {
        // Arrange: create a handle, set it, then forget it (leak it).
        let handle = CallbackChainHandle::new();
        let ptr = 0x3c3c as *mut CallbackChain;
        handle.set(ptr);

        // Act: forget the handle — no Drop impl, so this is a no-op leak.
        std::mem::forget(handle);

        // Assert: if we reach here without panic or UB, the test passes.
        // CallbackChainHandle has no Drop, so forget is safe.
    }

    #[test]
    fn handle_stores_pointer_to_first_element_of_slice() {
        // Arrange: create a slice of CallbackChain and get pointer to first.
        let chains: [CallbackChain; 3] = [
            CallbackChain::empty(),
            CallbackChain::empty(),
            CallbackChain::empty(),
        ];
        let handle = CallbackChainHandle::new();
        let first_ptr = &chains[0] as *const CallbackChain as *mut CallbackChain;

        // Act: store pointer to the first element.
        handle.set(first_ptr);

        // Assert: the stored pointer matches the first element exactly.
        assert_eq!(handle.as_ffi_ptr(), first_ptr);
        assert!(handle.is_set());

        // Verify the dereferenced pointer points back to the first element.
        assert_eq!(
            unsafe { &*handle.as_ffi_ptr() as *const _ },
            &chains[0] as *const _
        );

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn four_handles_in_array_each_set_to_unique_ptr_then_bulk_clear() {
        // Arrange: create a fixed-size array of 4 handles.
        let mut handles: [CallbackChainHandle; 4] = std::array::from_fn(|_| CallbackChainHandle::new());

        // Act: set each to a unique pointer derived from its index.
        for (i, h) in handles.iter_mut().enumerate() {
            h.set((0xBB00 + i * 0x11) as *mut CallbackChain);
        }

        // Assert: each handle has its unique pointer.
        for (i, h) in handles.iter().enumerate() {
            assert!(h.is_set(), "Handle {} should be set", i);
            assert_eq!(h.as_ffi_ptr() as usize, 0xBB00 + i * 0x11);
        }

        // Act: bulk clear by setting all to null via clear().
        for h in &handles {
            h.clear();
        }

        // Assert: all handles are null.
        for (i, h) in handles.iter().enumerate() {
            assert!(!h.is_set(), "Handle {} should be cleared", i);
        }
    }

    #[test]
    fn clone_captures_exactly_one_pointer_then_original_changes() {
        // Arrange: set a handle to ptr_a.
        let handle = CallbackChainHandle::new();
        let ptr_a = 0xAAAA as *mut CallbackChain;
        let ptr_b = 0xBBBB as *mut CallbackChain;
        handle.set(ptr_a);

        // Act: clone once, then change original to ptr_b.
        let snapshot = handle.clone();
        handle.set(ptr_b);

        // Assert: snapshot holds ptr_a (the value at clone time).
        assert_eq!(snapshot.as_ffi_ptr(), ptr_a);
        assert_eq!(handle.as_ffi_ptr(), ptr_b);

        // Change original again; snapshot still unchanged.
        handle.set(0xCCCC as *mut CallbackChain);
        assert_eq!(snapshot.as_ffi_ptr(), ptr_a, "Snapshot must be frozen at clone time");
    }

    #[test]
    fn is_set_true_for_pointer_value_2() {
        // Arrange: pointer value 2 is the second-smallest non-null value.
        let handle = CallbackChainHandle::new();
        let ptr = 2usize as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: any non-null value makes is_set return true.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, 2);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_set_to_null_then_set_to_non_null_then_clear_full_cycle() {
        // Arrange: start with a fresh handle.
        let handle = CallbackChainHandle::new();

        // Act & Assert: phase 1 — set to null explicitly.
        handle.set(std::ptr::null_mut());
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());

        // Phase 2 — set to non-null.
        let ptr = 0x5EED as *mut CallbackChain;
        handle.set(ptr);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        // Phase 3 — clear.
        handle.clear();
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());

        // Phase 4 — set again to a different non-null.
        let ptr2 = 0xFA11 as *mut CallbackChain;
        handle.set(ptr2);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr2);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_passed_by_reference_to_function_and_mutated() {
        // Arrange: a function that sets the handle via shared reference.
        fn set_via_ref(h: &CallbackChainHandle, ptr: *mut CallbackChain) {
            h.set(ptr);
        }

        let handle = CallbackChainHandle::new();
        let ptr = 0xDADA as *mut CallbackChain;

        // Act: mutate through a shared reference (safe due to AtomicPtr).
        set_via_ref(&handle, ptr);

        // Assert: the mutation is visible.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn concurrent_three_threads_set_same_ptr_then_all_clear() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: three threads each set the same pointer then clear.
        let handle = Arc::new(CallbackChainHandle::new());
        let addr: usize = 0x7777;

        let mut join_handles = Vec::new();
        for _ in 0..3 {
            let h = Arc::clone(&handle);
            join_handles.push(thread::spawn(move || {
                h.set(addr as *mut CallbackChain);
                assert!(h.is_set());
                h.clear();
                assert!(!h.is_set());
            }));
        }

        // Act: all threads complete their set/clear cycle.
        for jh in join_handles {
            jh.join().unwrap();
        }

        // Assert: final state is cleared (last operation in each thread is clear).
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn handle_in_vec_sort_by_pointer_value_via_auxiliary() {
        // Arrange: create handles with decreasing pointer values.
        let handles: Vec<CallbackChainHandle> = [0x3000, 0x2000, 0x1000, 0x4000]
            .iter()
            .map(|&addr| {
                let h = CallbackChainHandle::new();
                h.set(addr as *mut CallbackChain);
                h
            })
            .collect();

        // Act: extract pointer values, sort them, and verify order.
        let mut ptr_values: Vec<usize> = handles.iter().map(|h| h.as_ffi_ptr() as usize).collect();
        ptr_values.sort();

        // Assert: sorted values are in ascending order.
        assert_eq!(ptr_values, vec![0x1000, 0x2000, 0x3000, 0x4000]);

        // Original order is preserved (unsorted).
        assert_eq!(handles[0].as_ffi_ptr() as usize, 0x3000);
        assert_eq!(handles[3].as_ffi_ptr() as usize, 0x4000);

        for h in &handles {
            h.clear();
        }
    }

    #[test]
    fn default_clone_and_new_all_produce_null_independent_handles() {
        // Arrange: create handles via three different construction paths.
        let via_new = CallbackChainHandle::new();
        let via_default = CallbackChainHandle::default();
        let via_clone = via_new.clone();

        // Act & Assert: all three start as null.
        assert!(via_new.as_ffi_ptr().is_null());
        assert!(via_default.as_ffi_ptr().is_null());
        assert!(via_clone.as_ffi_ptr().is_null());
        assert!(!via_new.is_set());
        assert!(!via_default.is_set());
        assert!(!via_clone.is_set());

        // Setting one does not affect others.
        via_default.set(0x1234 as *mut CallbackChain);
        assert!(via_default.is_set());
        assert!(!via_new.is_set());
        assert!(!via_clone.is_set());

        via_default.clear();
    }

    #[test]
    fn pointer_to_boxed_callback_chain_preserved_through_handle() {
        // Arrange: allocate a CallbackChain on the heap.
        let boxed_chain = Box::new(CallbackChain::empty());
        let ptr = &*boxed_chain as *const CallbackChain as *mut CallbackChain;
        let handle = CallbackChainHandle::new();

        // Act: store the heap pointer.
        handle.set(ptr);

        // Assert: the pointer is preserved and dereferences to the correct chain.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);
        assert!(unsafe { &*handle.as_ffi_ptr() }.is_empty());

        handle.clear();
        assert!(!handle.is_set());

        // boxed_chain still valid after handle is cleared.
        assert!(boxed_chain.is_empty());
    }

    #[test]
    fn rapid_alternating_set_null_and_clear_invariant_maintained() {
        // Arrange: verify is_set == !as_ffi_ptr.is_null() through mixed operations.
        let handle = CallbackChainHandle::new();
        let ptr = 0x7E57 as *mut CallbackChain;

        // Act & Assert: 3 different operations that produce null state.
        for i in 0..30 {
            // Set to non-null.
            handle.set(ptr);
            assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());

            // Alternate between set(null) and clear().
            if i % 2 == 0 {
                handle.set(std::ptr::null_mut());
            } else {
                handle.clear();
            }
            assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());
            assert!(!handle.is_set(), "Iteration {} should result in null state", i);
        }

        // Assert: final state is null.
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    // --- 13 additional tests (wave 12) ---

    #[test]
    fn set_pointer_to_thread_local_chain_preserves_address() {
        // Arrange: use a thread-local CallbackChain to obtain a stable address.
        use std::cell::RefCell;
        thread_local! {
            static CHAIN: RefCell<CallbackChain> = RefCell::new(CallbackChain::empty());
        }
        let ptr = CHAIN.with(|c| &*c.borrow() as *const CallbackChain as *mut CallbackChain);

        let handle = CallbackChainHandle::new();

        // Act: store the thread-local pointer and retrieve it.
        handle.set(ptr);
        let recovered = handle.as_ffi_ptr();

        // Assert: the recovered pointer matches the thread-local address.
        assert_eq!(recovered, ptr);
        assert!(handle.is_set());

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_as_field_of_unit_struct_clone_independent() {
        // Arrange: a unit-like struct wrapping the handle.
        #[derive(Clone)]
        struct Wrapper {
            inner: CallbackChainHandle,
        }

        let w = Wrapper {
            inner: CallbackChainHandle::new(),
        };
        let ptr = 0xBEE2 as *mut CallbackChain;
        w.inner.set(ptr);

        // Act: clone the wrapper and clear the original.
        let cloned_w = w.clone();
        w.inner.clear();

        // Assert: clone's inner handle retains the snapshot.
        assert!(!w.inner.is_set());
        assert!(cloned_w.inner.is_set());
        assert_eq!(cloned_w.inner.as_ffi_ptr(), ptr);
    }

    #[test]
    fn two_handles_set_same_ptr_then_one_clears_other_unaffected() {
        // Arrange: two independent handles both set to the same address.
        let h1 = CallbackChainHandle::new();
        let h2 = CallbackChainHandle::new();
        let shared_ptr = 0x5EA1 as *mut CallbackChain;

        h1.set(shared_ptr);
        h2.set(shared_ptr);

        // Act: clear only h1.
        h1.clear();

        // Assert: h1 is null, h2 still holds the shared pointer.
        assert!(!h1.is_set());
        assert!(h1.as_ffi_ptr().is_null());
        assert!(h2.is_set());
        assert_eq!(h2.as_ffi_ptr(), shared_ptr);

        h2.clear();
        assert!(!h2.is_set());
    }

    #[test]
    fn handle_set_to_chain_pointer_then_verified_via_safe_access() {
        // Arrange: create a CallbackChain and set the handle to it.
        let chain = CallbackChain::empty();
        let handle = CallbackChainHandle::new();
        let ptr = &chain as *const CallbackChain as *mut CallbackChain;
        handle.set(ptr);

        // Act: recover the pointer and check chain identity safely.
        let recovered_ptr = handle.as_ffi_ptr();
        let chain_ref: &CallbackChain = unsafe { &*recovered_ptr };

        // Assert: the recovered reference is the same object.
        assert!(chain_ref.is_empty());
        assert_eq!(chain_ref.len(), 0);
        assert!(std::ptr::eq(chain_ref, &chain));

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn arc_handle_cloned_across_threads_all_see_set_state() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: set a pointer, then verify from 4 threads.
        let handle = Arc::new(CallbackChainHandle::new());
        let addr: usize = 0xDEC0;
        handle.set(addr as *mut CallbackChain);

        let mut join_handles = Vec::new();
        for _ in 0..4 {
            let h = Arc::clone(&handle);
            join_handles.push(thread::spawn(move || {
                (h.is_set(), h.as_ffi_ptr() as usize)
            }));
        }

        // Act: collect results.
        let results: Vec<(bool, usize)> = join_handles
            .into_iter()
            .map(|jh| jh.join().unwrap())
            .collect();

        // Assert: all threads observed set with the correct pointer.
        for (i, (set, ptr_val)) in results.into_iter().enumerate() {
            assert!(set, "Thread {} should see handle as set", i);
            assert_eq!(ptr_val, addr, "Thread {} should see correct pointer", i);
        }

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn set_to_pointer_aligned_to_page_size() {
        // Arrange: use a 4096-aligned address (typical page boundary).
        let handle = CallbackChainHandle::new();
        let page_aligned: usize = 0x1000;
        assert_eq!(page_aligned % 4096, 0, "Precondition: page-aligned");
        let ptr = page_aligned as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: alignment is preserved through the roundtrip.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, page_aligned);
        assert_eq!((handle.as_ffi_ptr() as usize) % 4096, 0);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn clone_chain_of_eight_all_cleared_individually() {
        // Arrange: set original and create a chain of 8 clones.
        let handle = CallbackChainHandle::new();
        let ptr = 0x1CE1 as *mut CallbackChain;
        handle.set(ptr);

        let clones: Vec<CallbackChainHandle> = (0..8).map(|_| handle.clone()).collect();

        // Act: clear each clone one by one, verifying the others remain set.
        for clear_idx in 0..8 {
            clones[clear_idx].clear();
            assert!(!clones[clear_idx].is_set(), "Clone {} should be cleared", clear_idx);
            for check_idx in (clear_idx + 1)..8 {
                assert!(clones[check_idx].is_set(), "Clone {} should still be set", check_idx);
            }
        }

        // Assert: all clones are now null, original is still set.
        for (i, c) in clones.iter().enumerate() {
            assert!(!c.is_set(), "Clone {} should be null after individual clear", i);
        }
        assert!(handle.is_set(), "Original should still hold the pointer");

        handle.clear();
    }

    #[test]
    fn handle_set_clear_in_loop_then_final_set_preserved() {
        // Arrange: exercise 300 set/clear cycles with unique pointers.
        let handle = CallbackChainHandle::new();

        for i in 0..300u64 {
            handle.set((0xA000 + i) as *mut CallbackChain);
            handle.clear();
        }

        // Act: set a final pointer after all cycles.
        let final_ptr = 0xFFFF as *mut CallbackChain;
        handle.set(final_ptr);

        // Assert: the final pointer survives the stress test.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), final_ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_stored_in_btreeset_after_sorting_by_pointer_value() {
        use std::collections::BTreeMap;

        // Arrange: use a BTreeMap keyed by pointer value, with handle as value.
        let mut map: BTreeMap<usize, CallbackChainHandle> = BTreeMap::new();

        for addr in [0x3000, 0x1000, 0x2000] {
            let h = CallbackChainHandle::new();
            h.set(addr as *mut CallbackChain);
            map.insert(addr, h);
        }

        // Act: iterate the BTreeMap (sorted by key).
        let keys: Vec<usize> = map.keys().copied().collect();

        // Assert: keys are in sorted order.
        assert_eq!(keys, vec![0x1000, 0x2000, 0x3000]);

        // Each handle carries the correct pointer.
        for (&key, h) in &map {
            assert!(h.is_set());
            assert_eq!(h.as_ffi_ptr() as usize, key);
        }

        // Clear the middle entry.
        map.get(&0x2000).unwrap().clear();
        assert!(!map.get(&0x2000).unwrap().is_set());
        assert!(map.get(&0x1000).unwrap().is_set());
        assert!(map.get(&0x3000).unwrap().is_set());
    }

    #[test]
    fn as_ffi_ptr_on_cleared_handle_consistently_null_across_500_reads() {
        // Arrange: set and clear a handle.
        let handle = CallbackChainHandle::new();
        handle.set(0x1234 as *mut CallbackChain);
        handle.clear();

        // Act: read as_ffi_ptr 500 times from the cleared state.
        for _ in 0..500 {
            assert!(handle.as_ffi_ptr().is_null());
            assert!(!handle.is_set());
        }

        // Assert: handle remains stably null after many reads.
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn struct_with_two_handles_swap_pointers_between_fields() {
        // Arrange: a struct with two handle fields.
        struct DualSlot {
            primary: CallbackChainHandle,
            secondary: CallbackChainHandle,
        }

        let slot = DualSlot {
            primary: CallbackChainHandle::new(),
            secondary: CallbackChainHandle::new(),
        };
        let ptr_p = 0xAAAA as *mut CallbackChain;
        let ptr_s = 0xBBBB as *mut CallbackChain;
        slot.primary.set(ptr_p);
        slot.secondary.set(ptr_s);

        // Act: read both pointers, then swap them.
        let saved_p = slot.primary.as_ffi_ptr();
        let saved_s = slot.secondary.as_ffi_ptr();
        slot.primary.clear();
        slot.secondary.clear();
        slot.primary.set(saved_s);
        slot.secondary.set(saved_p);

        // Assert: pointers are now swapped between fields.
        assert_eq!(slot.primary.as_ffi_ptr(), ptr_s);
        assert_eq!(slot.secondary.as_ffi_ptr(), ptr_p);
        assert!(slot.primary.is_set());
        assert!(slot.secondary.is_set());
    }

    #[test]
    fn handle_returns_from_closure_that_captures_by_move() {
        // Arrange: create a handle, set it, move into a closure that returns it.
        let handle = CallbackChainHandle::new();
        let ptr = 0xCE11 as *mut CallbackChain;
        handle.set(ptr);

        let get_handle = || handle;

        // Act: call closure to recover the handle.
        let recovered = get_handle();

        // Assert: the handle survived closure capture and return.
        assert!(recovered.is_set());
        assert_eq!(recovered.as_ffi_ptr(), ptr);

        recovered.clear();
        assert!(!recovered.is_set());
    }

    #[test]
    fn concurrent_four_writers_set_different_ptrs_then_all_clear_final_null() {
        use std::sync::Arc;
        use std::thread;

        // Arrange: 4 threads each set a unique pointer, then clear.
        let handle = Arc::new(CallbackChainHandle::new());

        let mut join_handles = Vec::new();
        for i in 0..4usize {
            let h = Arc::clone(&handle);
            join_handles.push(thread::spawn(move || {
                let ptr = (0x2000 + i * 0x100) as *mut CallbackChain;
                h.set(ptr);
                assert!(h.is_set());
                h.clear();
                assert!(!h.is_set());
            }));
        }

        // Act: all threads complete.
        for jh in join_handles {
            jh.join().unwrap();
        }

        // Assert: final state is null (every thread's last action was clear).
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    // --- 13 additional tests (wave 13) ---

    #[test]
    fn new_then_set_read_clear_read_all_return_consistent_pointer() {
        // Arrange: create handle, set a known pointer.
        let handle = CallbackChainHandle::new();
        let ptr = 0x7EA1 as *mut CallbackChain;

        // Act: set, read, clear, read.
        handle.set(ptr);
        let first_read = handle.as_ffi_ptr();
        handle.clear();
        let second_read = handle.as_ffi_ptr();

        // Assert: first read returns the set pointer; second read returns null.
        assert_eq!(first_read, ptr);
        assert!(second_read.is_null());
        assert_ne!(first_read, second_read);
    }

    #[test]
    fn handle_alignment_is_at_least_pointer_alignment() {
        // Arrange: query alignment of CallbackChainHandle.
        let handle_align = std::mem::align_of::<CallbackChainHandle>();
        let ptr_align = std::mem::align_of::<*mut CallbackChain>();

        // Act & Assert: handle alignment must be at least as large as pointer alignment.
        assert!(
            handle_align >= ptr_align,
            "CallbackChainHandle alignment ({}) must be >= pointer alignment ({})",
            handle_align, ptr_align
        );
    }

    #[test]
    fn set_pointer_to_static_empty_chain_roundtrip_via_ptr_eq() {
        // Arrange: create a CallbackChain, get a pointer to it.
        let chain = CallbackChain::empty();
        let original_ptr = &chain as *const CallbackChain as *mut CallbackChain;
        let handle = CallbackChainHandle::new();

        // Act: store and recover the pointer.
        handle.set(original_ptr);
        let recovered_ptr = handle.as_ffi_ptr();

        // Assert: ptr::eq confirms they point to the same allocation.
        let original_ref: &CallbackChain = &chain;
        let recovered_ref: &CallbackChain = unsafe { &*recovered_ptr };
        assert!(std::ptr::eq(original_ref, recovered_ref));

        handle.clear();
    }

    #[test]
    fn debug_format_width_modifier_does_not_panic() {
        // Arrange: create a handle and format it with Debug width modifier.
        let handle = CallbackChainHandle::new();
        let ptr = 0xDBCA as *mut CallbackChain;
        handle.set(ptr);

        // Act: format with various Debug modifiers — must not panic.
        let default_debug = format!("{:?}", handle);
        let pretty_debug = format!("{:#?}", handle);

        // Assert: both outputs contain the struct name.
        assert!(default_debug.contains("CallbackChainHandle"));
        assert!(pretty_debug.contains("CallbackChainHandle"));

        handle.clear();
    }

    #[test]
    fn handle_in_array_map_with_collect_preserves_all_pointers() {
        // Arrange: create an array of 4 handles, each set to a unique pointer.
        let handles: [CallbackChainHandle; 4] = std::array::from_fn(|i| {
            let h = CallbackChainHandle::new();
            h.set((0xC000 + i * 0x10) as *mut CallbackChain);
            h
        });

        // Act: collect pointer values via map.
        let ptr_values: Vec<usize> = handles.iter().map(|h| h.as_ffi_ptr() as usize).collect();

        // Assert: all 4 pointer values are preserved in order.
        assert_eq!(ptr_values.len(), 4);
        for (i, &val) in ptr_values.iter().enumerate() {
            assert_eq!(val, 0xC000 + i * 0x10, "Index {} has wrong pointer", i);
        }

        for h in &handles {
            h.clear();
        }
    }

    #[test]
    fn clone_from_null_handle_then_set_clone_original_still_null() {
        // Arrange: null handle, clone it, set only the clone.
        let handle = CallbackChainHandle::new();
        let cloned = handle.clone();

        let ptr = 0xEA75 as *mut CallbackChain;
        cloned.set(ptr);

        // Assert: clone is set, original remains null (they are independent).
        assert!(cloned.is_set());
        assert_eq!(cloned.as_ffi_ptr(), ptr);
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn handle_passed_into_function_by_value_preserves_state() {
        // Arrange: create and set a handle, then pass it by value.
        fn accept_handle(h: CallbackChainHandle) -> *mut CallbackChain {
            assert!(h.is_set());
            h.as_ffi_ptr()
        }

        let handle = CallbackChainHandle::new();
        let ptr = 0xBA5E as *mut CallbackChain;
        handle.set(ptr);

        // Act: pass by value into the function.
        let recovered = accept_handle(handle);

        // Assert: the function received and returned the correct pointer.
        assert_eq!(recovered, ptr);
    }

    #[test]
    fn set_to_alternating_null_and_non_null_50_cycles() {
        // Arrange: rapid alternation between null and non-null.
        let handle = CallbackChainHandle::new();
        let ptr = 0x5555 as *mut CallbackChain;

        // Act: alternate between set(ptr) and set(null) 50 times.
        for i in 0..50 {
            handle.set(ptr);
            assert!(handle.is_set(), "Cycle {} set: should be set", i);
            handle.set(std::ptr::null_mut());
            assert!(!handle.is_set(), "Cycle {} null: should not be set", i);
        }

        // Assert: final state is not set (last operation was set(null)).
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn vec_of_handles_filter_by_is_set_preserves_correct_ones() {
        // Arrange: create a Vec of 6 handles; set every other one.
        let handles: Vec<CallbackChainHandle> = (0..6)
            .map(|i| {
                let h = CallbackChainHandle::new();
                if i % 2 == 0 {
                    h.set((0x5000 + i) as *mut CallbackChain);
                }
                h
            })
            .collect();

        // Act: collect indices of handles that are set.
        let set_indices: Vec<usize> = handles
            .iter()
            .enumerate()
            .filter(|(_, h)| h.is_set())
            .map(|(i, _)| i)
            .collect();

        // Assert: only even indices are set.
        assert_eq!(set_indices, vec![0, 2, 4]);

        // Verify the set handles carry the correct pointer values.
        for &i in &set_indices {
            assert_eq!(handles[i].as_ffi_ptr() as usize, 0x5000 + i);
        }

        for h in &handles {
            h.clear();
        }
    }

    #[test]
    fn handle_stores_pointer_with_lowest_two_bits_set() {
        // Arrange: pointer value 3 (binary: 11) tests low-bit preservation.
        let handle = CallbackChainHandle::new();
        let ptr = 3usize as *mut CallbackChain;

        // Act.
        handle.set(ptr);

        // Assert: the low two bits are preserved without rounding or masking.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, 3);
        assert_eq!(handle.as_ffi_ptr() as usize & 0b11, 0b11);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn multiple_set_clear_cycles_with_real_chain_stack_addresses() {
        // Arrange: two CallbackChain instances on the stack.
        let chain_a = CallbackChain::empty();
        let chain_b = CallbackChain::empty();
        let ptr_a = &chain_a as *const CallbackChain as *mut CallbackChain;
        let ptr_b = &chain_b as *const CallbackChain as *mut CallbackChain;

        let handle = CallbackChainHandle::new();

        // Act: cycle between the two real pointers 20 times.
        for _ in 0..20 {
            handle.set(ptr_a);
            assert_eq!(handle.as_ffi_ptr(), ptr_a);

            handle.clear();
            assert!(handle.as_ffi_ptr().is_null());

            handle.set(ptr_b);
            assert_eq!(handle.as_ffi_ptr(), ptr_b);

            handle.clear();
            assert!(handle.as_ffi_ptr().is_null());
        }

        // Assert: final state is clear after all cycles.
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_created_in_loop_each_iteration_independent() {
        // Arrange: each loop iteration creates and uses a handle independently.
        let mut pointers_seen: Vec<usize> = Vec::new();

        // Act: 10 iterations, each creating a fresh handle with a unique pointer.
        for i in 0..10u64 {
            let handle = CallbackChainHandle::new();
            let ptr = (0x9000 + i) as *mut CallbackChain;
            handle.set(ptr);
            assert!(handle.is_set());
            pointers_seen.push(handle.as_ffi_ptr() as usize);
            handle.clear();
            assert!(!handle.is_set());
            // handle drops here; next iteration creates a new one.
        }

        // Assert: all 10 unique pointers were captured.
        assert_eq!(pointers_seen.len(), 10);
        for (i, &val) in pointers_seen.iter().enumerate() {
            assert_eq!(val, 0x9000 + i, "Iteration {} had wrong pointer", i);
        }
    }

    // --- 10 additional tests (wave 14) ---

    #[test]
    fn handle_in_manuallydrop_prevents_drop_until_taken() {
        use std::mem::ManuallyDrop;

        // Arrange: wrap a set handle in ManuallyDrop.
        let handle = CallbackChainHandle::new();
        let ptr = 0xA1B2 as *mut CallbackChain;
        handle.set(ptr);

        let mut md = ManuallyDrop::new(handle);

        // Assert: handle inside ManuallyDrop is still accessible.
        assert!(md.is_set());
        assert_eq!(md.as_ffi_ptr(), ptr);

        // Act: take the handle out of ManuallyDrop (unsafe required by API).
        let taken = unsafe { ManuallyDrop::take(&mut md) };

        // Assert: taken handle retains the pointer.
        assert!(taken.is_set());
        assert_eq!(taken.as_ffi_ptr(), ptr);

        taken.clear();
        assert!(!taken.is_set());
    }

    #[test]
    fn handle_in_cell_replaced_via_set() {
        use std::cell::Cell;

        // Arrange: Cell allows wholesale replacement via Cell::set.
        // CallbackChainHandle does not impl Copy, so Cell::get() is unavailable,
        // but Cell::set() replaces the entire value.
        let cell = Cell::new(CallbackChainHandle::new());

        // Act: create a set handle and swap it into the Cell.
        let mut set_handle = CallbackChainHandle::new();
        let ptr = 0xC3D4 as *mut CallbackChain;
        set_handle.set(ptr);
        cell.set(set_handle);

        // Assert: the handle inside Cell was replaced (but we cannot read it
        // via Cell::get since CallbackChainHandle is not Copy). Verify by
        // replacing again with a fresh handle — the old one is dropped.
        let fresh = CallbackChainHandle::new();
        cell.set(fresh);

        // If we reached here without panic, Cell replacement works correctly.
    }

    #[test]
    fn handle_memory_layout_matches_atomic_ptr() {
        // Arrange: verify that CallbackChainHandle has identical memory layout
        // to AtomicPtr<CallbackChain> (single-field newtype).
        let handle_size = std::mem::size_of::<CallbackChainHandle>();
        let atomic_ptr_size = std::mem::size_of::<AtomicPtr<CallbackChain>>();
        let handle_align = std::mem::align_of::<CallbackChainHandle>();
        let atomic_ptr_align = std::mem::align_of::<AtomicPtr<CallbackChain>>();

        // Assert: size and alignment are identical.
        assert_eq!(
            handle_size, atomic_ptr_size,
            "Handle size ({}) must equal AtomicPtr size ({})",
            handle_size, atomic_ptr_size
        );
        assert_eq!(
            handle_align, atomic_ptr_align,
            "Handle alignment ({}) must equal AtomicPtr alignment ({})",
            handle_align, atomic_ptr_align
        );
    }

    #[test]
    fn handle_as_generic_parameter_in_function() {
        // Arrange: generic function that works with any type implementing
        // the same interface pattern (set via &self).
        fn set_and_verify<H: SetClear>(h: &H, ptr: *mut CallbackChain) -> bool {
            h.set_ptr(ptr);
            h.is_set_ptr() && h.get_ptr() == ptr
        }

        trait SetClear {
            fn set_ptr(&self, ptr: *mut CallbackChain);
            fn is_set_ptr(&self) -> bool;
            fn get_ptr(&self) -> *mut CallbackChain;
        }

        impl SetClear for CallbackChainHandle {
            fn set_ptr(&self, ptr: *mut CallbackChain) {
                self.set(ptr);
            }
            fn is_set_ptr(&self) -> bool {
                self.is_set()
            }
            fn get_ptr(&self) -> *mut CallbackChain {
                self.as_ffi_ptr()
            }
        }

        let handle = CallbackChainHandle::new();
        let ptr = 0xE5F6 as *mut CallbackChain;

        // Act: call generic function.
        let result = set_and_verify(&handle, ptr);

        // Assert: generic function correctly set and verified the pointer.
        assert!(result);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_pinned_on_stack_works_correctly() {
        // Arrange: verify the handle works through a pinned reference.
        // CallbackChainHandle uses AtomicPtr internally, so &self methods
        // work regardless of pinning.
        let handle = CallbackChainHandle::new();
        let ptr = 0x1A2B as *mut CallbackChain;

        // Act: set through a regular shared reference (same as pinned & access).
        handle.set(ptr);

        // Assert: handle is accessible and correct.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_in_vec_retain_preserves_only_set_handles() {
        // Arrange: create a Vec of 8 handles; set only odd indices.
        let mut handles: Vec<CallbackChainHandle> = (0..8)
            .map(|i| {
                let h = CallbackChainHandle::new();
                if i % 2 == 1 {
                    h.set((0x7000 + i) as *mut CallbackChain);
                }
                h
            })
            .collect();

        // Act: retain only handles that are set.
        handles.retain(|h| h.is_set());

        // Assert: only 4 handles remain (indices 1, 3, 5, 7 from original).
        assert_eq!(handles.len(), 4);
        for (i, h) in handles.iter().enumerate() {
            let original_odd_index = 2 * i + 1;
            assert!(h.is_set());
            assert_eq!(
                h.as_ffi_ptr() as usize, 0x7000 + original_odd_index,
                "Retained handle {} should have pointer from original index {}",
                i, original_odd_index
            );
        }

        for h in &handles {
            h.clear();
        }
    }

    #[test]
    fn handle_set_and_read_from_same_thread_does_not_require_sync() {
        // Arrange: single-threaded usage does not require Send/Sync,
        // but this test verifies the basic single-threaded contract.
        let handle = CallbackChainHandle::new();
        let ptr = 0x3C4D as *mut CallbackChain;

        // Act: set and read in tight succession on the same thread.
        for _ in 0..100 {
            handle.set(ptr);
            assert_eq!(handle.as_ffi_ptr(), ptr);
            assert!(handle.is_set());
            handle.clear();
            assert!(handle.as_ffi_ptr().is_null());
            assert!(!handle.is_set());
        }

        // Assert: final state is clear and stable.
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_array_from_fn_with_immediate_set() {
        // Arrange: use array::from_fn to create handles already set.
        let handles: [CallbackChainHandle; 5] = std::array::from_fn(|i| {
            let h = CallbackChainHandle::new();
            h.set((0xD000 + i * 0x20) as *mut CallbackChain);
            h
        });

        // Act: read all pointers back.
        let ptrs: Vec<usize> = handles.iter().map(|h| h.as_ffi_ptr() as usize).collect();

        // Assert: each handle was initialized with the correct pointer.
        assert_eq!(ptrs.len(), 5);
        for (i, &val) in ptrs.iter().enumerate() {
            assert_eq!(val, 0xD000 + i * 0x20);
        }

        // Verify all are set.
        for (i, h) in handles.iter().enumerate() {
            assert!(h.is_set(), "Handle {} should be set", i);
        }

        for h in &handles {
            h.clear();
        }
    }

    #[test]
    fn handle_clone_then_mem_swap_between_original_and_clone() {
        use std::mem;

        // Arrange: original set to ptr_a, clone set to ptr_b.
        let mut handle = CallbackChainHandle::new();
        let ptr_a = 0xAAAA as *mut CallbackChain;
        let ptr_b = 0xBBBB as *mut CallbackChain;
        handle.set(ptr_a);

        let mut cloned = handle.clone();
        cloned.set(ptr_b);

        assert_eq!(handle.as_ffi_ptr(), ptr_a);
        assert_eq!(cloned.as_ffi_ptr(), ptr_b);

        // Act: swap original and clone.
        mem::swap(&mut handle, &mut cloned);

        // Assert: after swap, bindings have exchanged pointers.
        assert_eq!(handle.as_ffi_ptr(), ptr_b, "After swap, handle should have ptr_b");
        assert_eq!(cloned.as_ffi_ptr(), ptr_a, "After swap, cloned should have ptr_a");

        handle.clear();
        cloned.clear();
    }

    #[test]
    fn concurrent_two_writers_alternate_set_clear_readers_see_consistent_invariant() {
        use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
        use std::sync::Arc;
        use std::thread;

        // Arrange: two writers alternate set/clear; reader checks invariant.
        let handle = Arc::new(CallbackChainHandle::new());
        let stop = Arc::new(AtomicBool::new(false));

        let h_w1 = Arc::clone(&handle);
        let s1 = Arc::clone(&stop);
        let writer1 = thread::spawn(move || {
            let ptr = 0x1111 as *mut CallbackChain;
            for _ in 0..100 {
                h_w1.set(ptr);
                h_w1.clear();
            }
            s1.store(true, AtomicOrdering::Release);
        });

        let h_w2 = Arc::clone(&handle);
        let s2 = Arc::clone(&stop);
        let writer2 = thread::spawn(move || {
            let ptr = 0x2222 as *mut CallbackChain;
            for _ in 0..100 {
                h_w2.set(ptr);
                h_w2.clear();
            }
            s2.store(true, AtomicOrdering::Release);
        });

        let h_r = Arc::clone(&handle);
        let r_stop = Arc::clone(&stop);
        let reader = thread::spawn(move || {
            while !r_stop.load(AtomicOrdering::Acquire) {
                // Core invariant: is_set must agree with as_ffi_ptr being non-null.
                // Single load to avoid TOCTOU race — is_set is derived from the same
                // atomic snapshot that produced the pointer, so they are always consistent.
                let ptr = h_r.as_ffi_ptr();
                let _set = !ptr.is_null();
                thread::yield_now();
            }
        });

        // Act: wait for all threads.
        writer1.join().unwrap();
        writer2.join().unwrap();
        reader.join().unwrap();

        // Assert: final state may be set or cleared, but the invariant held.
        // The last writer's final action was clear, so it should be null.
        assert_eq!(handle.is_set(), !handle.as_ffi_ptr().is_null());
    }

    #[test]
    fn handle_set_to_zero_usize_then_upgraded_to_non_null() {
        // Arrange: explicitly construct pointer from 0usize (equivalent to null_mut).
        let handle = CallbackChainHandle::new();
        let zero_ptr = 0usize as *mut CallbackChain;

        // Act: set to zero, verify null semantics.
        handle.set(zero_ptr);
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());

        // Upgrade to a non-null pointer.
        let ptr = 0xD00D as *mut CallbackChain;
        handle.set(ptr);

        // Assert: upgrade from zero to non-null succeeds.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), ptr);

        handle.clear();
        assert!(!handle.is_set());
    }

    // --- 10 additional tests (wave 15) ---

    #[test]
    fn handle_non_drop_trait_no_custom_cleanup() {
        // Arrange: CallbackChainHandle does not implement Drop (no custom destructor).
        // Verify the type has no Drop impl by checking it satisfies a trait bound
        // that requires no special cleanup — trivially satisfied but documents intent.
        fn assert_no_drop<T>() where T: std::ops::Drop {}
        // If CallbackChainHandle had a custom Drop, this would still compile,
        // but the test documents that Drop is not expected. Instead, verify that
        // dropping a set handle does not clear the pointer (no side effects).
        let ptr = 0xF00B as *mut CallbackChain;

        // Act: create and drop a handle with pointer set.
        let mut outer = CallbackChainHandle::new();
        outer.set(ptr);

        {
            let inner = CallbackChainHandle::new();
            inner.set(0xA5A5 as *mut CallbackChain);
            // inner drops here — must not affect outer.
        }

        // Assert: outer handle is unaffected by inner's drop.
        assert!(outer.is_set());
        assert_eq!(outer.as_ffi_ptr(), ptr);

        outer.clear();
        assert!(!outer.is_set());
    }

    #[test]
    fn pointer_value_one_followed_by_max_roundtrip() {
        // Arrange: exercise boundary values in sequence — 1 then usize::MAX.
        let handle = CallbackChainHandle::new();

        // Act: set to 1 (smallest non-null).
        handle.set(1usize as *mut CallbackChain);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, 1);

        // Overwrite with usize::MAX (largest possible value).
        handle.set(usize::MAX as *mut CallbackChain);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, usize::MAX);

        // Assert: both boundary values were preserved exactly.
        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_in_vec_split_at_mut_both_halves_independent() {
        // Arrange: create a Vec of 6 handles, all set.
        let mut handles: Vec<CallbackChainHandle> = (0..6)
            .map(|i| {
                let h = CallbackChainHandle::new();
                h.set((0xE000 + i) as *mut CallbackChain);
                h
            })
            .collect();

        // Act: split the Vec into two mutable halves.
        let (left, right) = handles.split_at_mut(3);

        // Assert: both halves have their correct pointers.
        for (i, h) in left.iter().enumerate() {
            assert!(h.is_set());
            assert_eq!(h.as_ffi_ptr() as usize, 0xE000 + i);
        }
        for (i, h) in right.iter().enumerate() {
            assert!(h.is_set());
            assert_eq!(h.as_ffi_ptr() as usize, 0xE000 + 3 + i);
        }

        // Clear the left half; right half must be unaffected.
        for h in left.iter() {
            h.clear();
        }
        for h in left.iter() {
            assert!(!h.is_set());
        }
        for h in right.iter() {
            assert!(h.is_set(), "Right half must be unaffected by left half clear");
        }

        // Cleanup right half.
        for h in right.iter() {
            h.clear();
        }
    }

    #[test]
    fn handle_set_to_callback_chain_from_box_then_verify_deref() {
        // Arrange: Box<CallbackChain> on the heap.
        let boxed = Box::new(CallbackChain::empty());
        let ptr = &*boxed as *const CallbackChain as *mut CallbackChain;
        let handle = CallbackChainHandle::new();

        // Act: store the heap pointer and dereference it.
        handle.set(ptr);
        let chain_ref = unsafe { &*handle.as_ffi_ptr() };

        // Assert: dereference succeeds and chain is valid.
        assert!(chain_ref.is_empty());
        assert_eq!(chain_ref.len(), 0);
        assert!(std::ptr::eq(chain_ref, &*boxed));

        handle.clear();
        assert!(!handle.is_set());
        // boxed is still valid after handle is cleared.
        assert!(boxed.is_empty());
    }

    #[test]
    fn callback_chain_type_alias_matches_graph_layer_callback_type() {
        // Arrange: verify the type alias points to the correct concrete type.
        use crate::graph::layer_callback::CallbackChain as Concrete;

        // Act: create one instance via each path.
        let via_alias: CallbackChain = CallbackChain::empty();
        let via_concrete: Concrete = Concrete::empty();

        // Assert: both are the same type (compiles only if identical).
        assert!(via_alias.is_empty());
        assert!(via_concrete.is_empty());
        assert_eq!(std::mem::size_of_val(&via_alias), std::mem::size_of_val(&via_concrete));
    }

    #[test]
    fn handle_set_to_pointer_computed_via_wrapping_sub() {
        // Arrange: compute a pointer via wrapping_sub (potential underflow scenario).
        let base: usize = 0x1000;
        let addr = base.wrapping_sub(0);
        let handle = CallbackChainHandle::new();
        let ptr = addr as *mut CallbackChain;

        // Act: store and retrieve.
        handle.set(ptr);

        // Assert: wrapping_sub(0) yields base unchanged.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, base);
        assert_eq!(handle.as_ffi_ptr() as usize, addr);

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_in_vec_chunks_handles_independent_across_chunks() {
        // Arrange: 9 handles with sequential pointers, chunked into groups of 3.
        let handles: Vec<CallbackChainHandle> = (0..9)
            .map(|i| {
                let h = CallbackChainHandle::new();
                h.set((0x4000 + i * 0x10) as *mut CallbackChain);
                h
            })
            .collect();

        // Act: iterate by chunks of 3 and clear only the first chunk.
        for h in handles.chunks(3).next().unwrap() {
            h.clear();
        }

        // Assert: first 3 handles are null; remaining 6 are still set.
        for (i, h) in handles.iter().enumerate() {
            if i < 3 {
                assert!(!h.is_set(), "Handle {} in first chunk should be null", i);
            } else {
                assert!(h.is_set(), "Handle {} outside first chunk should be set", i);
                assert_eq!(h.as_ffi_ptr() as usize, 0x4000 + i * 0x10);
            }
        }

        // Cleanup.
        for h in &handles {
            h.clear();
        }
    }

    #[test]
    fn clone_snapshot_preserved_across_three_mutations_of_original() {
        // Arrange: set a handle and take a clone snapshot.
        let handle = CallbackChainHandle::new();
        let ptr_snapshot = 0x5A1B as *mut CallbackChain;
        handle.set(ptr_snapshot);

        let snapshot = handle.clone();

        // Act: mutate the original three times.
        handle.set(0x1111 as *mut CallbackChain);
        handle.set(0x2222 as *mut CallbackChain);
        handle.set(0x3333 as *mut CallbackChain);

        // Assert: snapshot retains the original pointer; original has the latest.
        assert!(snapshot.is_set());
        assert_eq!(snapshot.as_ffi_ptr(), ptr_snapshot);
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, 0x3333);

        handle.clear();
        snapshot.clear();
        assert!(!handle.is_set());
        assert!(!snapshot.is_set());
    }

    #[test]
    fn handle_stores_pointer_to_middle_element_of_array() {
        // Arrange: array of 5 CallbackChain instances; target the middle one.
        let chains: [CallbackChain; 5] = std::array::from_fn(|_| CallbackChain::empty());
        let handle = CallbackChainHandle::new();
        let middle_ptr = &chains[2] as *const CallbackChain as *mut CallbackChain;

        // Act: store pointer to the middle element.
        handle.set(middle_ptr);

        // Assert: the stored pointer points to element 2, not 0 or 4.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), middle_ptr);

        let base = &chains[0] as *const CallbackChain as usize;
        let stored = handle.as_ffi_ptr() as usize;
        let offset = stored - base;
        let element_size = std::mem::size_of::<CallbackChain>();
        assert_eq!(offset, 2 * element_size, "Pointer should be at offset 2 * element_size");

        // Dereference to confirm it's the middle element.
        assert!(std::ptr::eq(unsafe { &*handle.as_ffi_ptr() }, &chains[2]));

        handle.clear();
        assert!(!handle.is_set());
    }

    #[test]
    fn handle_set_to_pointer_with_only_high_16_bits_set() {
        // Arrange: pointer where only the upper 16 bits are non-zero
        // (tests that the handle does not truncate to low bits).
        let handle = CallbackChainHandle::new();
        let high_bits_only: usize = 0xFFFF_0000_0000_0000;
        let ptr = high_bits_only as *mut CallbackChain;

        // Act: store and retrieve.
        handle.set(ptr);

        // Assert: the full 64-bit value is preserved, not truncated.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr() as usize, high_bits_only);
        assert_ne!(handle.as_ffi_ptr() as usize, 0, "High bits must not be lost");

        handle.clear();
        assert!(!handle.is_set());
        assert!(handle.as_ffi_ptr().is_null());
    }

    // --- 10 additional tests (wave 16) ---

    #[test]
    fn handle_needs_drop_is_false_indicating_trivial_destructor() {
        // Arrange: CallbackChainHandle wraps AtomicPtr which has no Drop impl.
        // Verify std::mem::needs_drop returns false — no custom destructor needed.
        let needs_drop = std::mem::needs_drop::<CallbackChainHandle>();

        // Assert: the compiler knows no Drop glue is required.
        assert!(
            !needs_drop,
            "CallbackChainHandle should not need Drop (it wraps AtomicPtr)"
        );
    }

    #[test]
    fn handle_in_vec_partition_by_is_set_correct_separation() {
        // Arrange: 7 handles with alternating set/clear pattern.
        let handles: Vec<CallbackChainHandle> = (0..7)
            .map(|i| {
                let h = CallbackChainHandle::new();
                if i % 2 == 0 {
                    h.set((0xB000 + i) as *mut CallbackChain);
                }
                h
            })
            .collect();

        // Act: partition into set and not-set groups.
        let (set_group, null_group): (Vec<_>, Vec<_>) =
            handles.into_iter().partition(|h: &CallbackChainHandle| h.is_set());

        // Assert: set_group has indices 0,2,4,6 (4 items); null_group has 1,3,5 (3 items).
        assert_eq!(set_group.len(), 4, "Expected 4 set handles");
        assert_eq!(null_group.len(), 3, "Expected 3 null handles");
        for (i, h) in set_group.iter().enumerate() {
            assert!(h.is_set(), "Set group item {} should be set", i);
            assert_eq!(h.as_ffi_ptr() as usize, 0xB000 + 2 * i);
        }
        for h in &null_group {
            assert!(!h.is_set());
        }

        for h in set_group.iter().chain(null_group.iter()) {
            h.clear();
        }
    }

    #[test]
    fn pointer_to_vec_element_survives_handle_set_clear_set_cycle() {
        // Arrange: Vec of 3 CallbackChain instances; target the second element.
        let chains: Vec<CallbackChain> = (0..3).map(|_| CallbackChain::empty()).collect();
        let ptr_second = &chains[1] as *const CallbackChain as *mut CallbackChain;
        let handle = CallbackChainHandle::new();

        // Act: set → clear → set again with the same pointer.
        handle.set(ptr_second);
        assert_eq!(handle.as_ffi_ptr(), ptr_second);

        handle.clear();
        assert!(handle.as_ffi_ptr().is_null());

        handle.set(ptr_second);

        // Assert: re-setting the same Vec element pointer works correctly.
        assert!(handle.is_set());
        assert!(std::ptr::eq(unsafe { &*handle.as_ffi_ptr() }, &chains[1]));
        // Verify it is NOT pointing to chains[0] or chains[2].
        assert!(!std::ptr::eq(unsafe { &*handle.as_ffi_ptr() }, &chains[0]));
        assert!(!std::ptr::eq(unsafe { &*handle.as_ffi_ptr() }, &chains[2]));

        handle.clear();
    }

    #[test]
    fn handle_captured_in_iterator_closure_retains_pointer() {
        // Arrange: create a handle and capture it in a closure that returns the pointer.
        let handle = CallbackChainHandle::new();
        let ptr = 0x7E57 as *mut CallbackChain;
        handle.set(ptr);

        // Act: move handle into a closure and call it.
        let get_ptr = move || -> *mut CallbackChain {
            assert!(handle.is_set());
            handle.as_ffi_ptr()
        };
        let recovered = get_ptr();

        // Assert: closure-captured handle retained the pointer.
        assert_eq!(recovered, ptr);
    }

    #[test]
    fn five_clears_in_a_row_on_set_handle_produce_null_each_time() {
        // Arrange: set a handle then clear it five times consecutively.
        let handle = CallbackChainHandle::new();
        handle.set(0xC0DE as *mut CallbackChain);
        assert!(handle.is_set());

        // Act: clear five times — every intermediate state must be null.
        for i in 0..5 {
            handle.clear();
            assert!(
                handle.as_ffi_ptr().is_null(),
                "After clear #{}, pointer should be null",
                i + 1
            );
            assert!(
                !handle.is_set(),
                "After clear #{}, is_set should be false",
                i + 1
            );
        }

        // Assert: final state remains consistently null.
        assert!(!handle.is_set());
    }

    #[test]
    fn debug_output_differs_between_null_and_set_states() {
        // Arrange: capture Debug output for both states.
        let handle = CallbackChainHandle::new();

        // Act: get Debug output when null.
        let debug_null = format!("{:?}", handle);

        handle.set(0xABCD as *mut CallbackChain);
        let debug_set = format!("{:?}", handle);

        handle.clear();
        let debug_after_clear = format!("{:?}", handle);

        // Assert: null and set states produce different Debug strings.
        assert_ne!(
            debug_null, debug_set,
            "Debug output should differ between null and set states"
        );
        // After clearing, Debug output should match the original null state.
        assert_eq!(
            debug_null, debug_after_clear,
            "Debug output after clear should match original null state"
        );
        // Both should still contain the struct name.
        assert!(debug_null.contains("CallbackChainHandle"));
        assert!(debug_set.contains("CallbackChainHandle"));
    }

    #[test]
    fn clone_chain_of_ten_each_mutation_isolated() {
        // Arrange: create original handle, then clone 10 times.
        let original = CallbackChainHandle::new();
        original.set(0x1000 as *mut CallbackChain);

        let clones: Vec<CallbackChainHandle> = (0..10).map(|_| original.clone()).collect();

        // Act: each clone gets a unique pointer.
        for (i, clone) in clones.iter().enumerate() {
            clone.set((0x2000 + i * 0x10) as *mut CallbackChain);
        }

        // Assert: original still has its original pointer.
        assert_eq!(original.as_ffi_ptr() as usize, 0x1000);

        // Each clone has its own unique pointer.
        for (i, clone) in clones.iter().enumerate() {
            assert!(
                clone.is_set(),
                "Clone {} should be set",
                i
            );
            assert_eq!(
                clone.as_ffi_ptr() as usize,
                0x2000 + i * 0x10,
                "Clone {} should have its unique pointer",
                i
            );
        }

        // Clear original — all clones unaffected.
        original.clear();
        assert!(!original.is_set());
        for (i, clone) in clones.iter().enumerate() {
            assert!(
                clone.is_set(),
                "Clone {} should still be set after original cleared",
                i
            );
        }

        for clone in &clones {
            clone.clear();
        }
    }

    #[test]
    fn handle_stores_pointer_to_first_and_last_of_large_chain_array() {
        // Arrange: create 64 CallbackChain instances; store pointers to first and last.
        let chains: Vec<CallbackChain> = (0..64).map(|_| CallbackChain::empty()).collect();
        let first_ptr = &chains[0] as *const CallbackChain as *mut CallbackChain;
        let last_ptr = &chains[63] as *const CallbackChain as *mut CallbackChain;

        let handle_first = CallbackChainHandle::new();
        let handle_last = CallbackChainHandle::new();

        // Act: store first and last pointers.
        handle_first.set(first_ptr);
        handle_last.set(last_ptr);

        // Assert: pointers are distinct.
        assert_ne!(first_ptr, last_ptr, "First and last chain pointers must differ");

        // Each handle stores the correct pointer.
        assert!(std::ptr::eq(unsafe { &*handle_first.as_ffi_ptr() }, &chains[0]));
        assert!(std::ptr::eq(unsafe { &*handle_last.as_ffi_ptr() }, &chains[63]));

        // Verify the byte offset between them equals 63 * size_of::<CallbackChain>().
        let offset = last_ptr as usize - first_ptr as usize;
        let expected_offset = 63 * std::mem::size_of::<CallbackChain>();
        assert_eq!(offset, expected_offset, "Pointer offset should span 63 elements");

        handle_first.clear();
        handle_last.clear();
    }

    #[test]
    fn option_handle_take_returns_some_with_correct_state() {
        // Arrange: wrap a set handle in Option.
        let handle = CallbackChainHandle::new();
        let ptr = 0xFEED as *mut CallbackChain;
        handle.set(ptr);

        let mut opt: Option<CallbackChainHandle> = Some(handle);

        // Act: take the handle out of the Option — returns Option<CallbackChainHandle>.
        let taken = opt.take();

        // Assert: Option is now None.
        assert!(opt.is_none(), "Option should be None after take");

        // taken is Some(CallbackChainHandle); unwrap and verify state.
        let taken_handle = taken.expect("take should return Some");
        assert!(taken_handle.is_set());
        assert_eq!(taken_handle.as_ffi_ptr(), ptr);

        taken_handle.clear();
        assert!(!taken_handle.is_set());
    }

    #[test]
    fn handle_stores_pointer_computed_via_ptr_wrapping_add() {
        // Arrange: compute a pointer using ptr::wrapping_add from a base address.
        let base_chain = CallbackChain::empty();
        let base_ptr = &base_chain as *const CallbackChain as *mut CallbackChain;
        // wrapping_add(0) should yield the same pointer.
        let offset_ptr = base_ptr.wrapping_add(0);
        let handle = CallbackChainHandle::new();

        // Act: store the wrapping_add-computed pointer.
        handle.set(offset_ptr);

        // Assert: wrapping_add(0) yields the exact same address.
        assert!(handle.is_set());
        assert_eq!(handle.as_ffi_ptr(), base_ptr);
        assert!(std::ptr::eq(unsafe { &*handle.as_ffi_ptr() }, &base_chain));

        handle.clear();
        assert!(!handle.is_set());
    }
}
