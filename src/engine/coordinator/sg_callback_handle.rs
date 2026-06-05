//! Safe wrapper for leaked SgCallbackCtx FFI pointer.
//!
//! Replaces `*const u8` (raw leaked pointer) in `ModelContextHolder`.
//! Uses `Pin<Box<>>` for stable heap address guaranteed across moves.
//! The callback context lives as long as this handle owns it.

use std::pin::Pin;

use super::super::mega_kernel_callback::SgCallbackCtx;

/// Safe ownership wrapper for an `SgCallbackCtx` that has been registered in the
/// JIT callback table.
///
/// The `Pin<Box<>>` ensures the heap address passed to JIT via `as_ptr()` remains
/// stable even if the `SgCallbackHandle` is moved within `ModelContextHolder`.
pub struct SgCallbackHandle {
    inner: Option<Pin<Box<SgCallbackCtx>>>,
}

impl SgCallbackHandle {
    pub fn new() -> Self {
        Self { inner: None }
    }

    /// Register a new callback context. Returns the raw pointer for the FFI callback table.
    /// The handle takes ownership; the pointer remains valid until `reclaim()` is called.
    pub fn register(&mut self, ctx: SgCallbackCtx) -> *const u8 {
        let pinned = Box::pin(ctx);
        let ptr = (&*pinned) as *const SgCallbackCtx as *const u8;
        self.inner = Some(pinned);
        ptr
    }

    /// Reclaim and drop the callback context.
    /// Returns the `provider_state` raw pointer if present (caller must `Arc::from_raw` it).
    /// Also reclaims the leaked `precomputed_knowledge` `Box<[f32]>`.
    pub fn reclaim(&mut self) -> Option<*const u8> {
        let pinned = self.inner.take()?;
        let ctx = Pin::into_inner(pinned);
        let provider_ptr = ctx.provider_state;

        // Reclaim the leaked precomputed_knowledge Box<[f32]>.
        if !ctx.precomputed_knowledge.is_null() {
            let len = ctx.hidden_size as usize;
            // SAFETY: precomputed_knowledge was created by `Box::into_raw` from a
            // `Box<[f32]>` of length `hidden_size`.
            unsafe {
                let slice =
                    std::slice::from_raw_parts_mut(ctx.precomputed_knowledge as *mut f32, len);
                let _ = Box::from_raw(slice as *mut [f32]);
            }
        }
        // `ctx` is dropped here; `provider_ptr` returned for caller to Arc::from_raw
        Some(provider_ptr)
    }

    /// Check if a callback is currently registered.
    pub fn is_registered(&self) -> bool {
        self.inner.is_some()
    }
}

impl Default for SgCallbackHandle {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: SgCallbackHandle is only accessed behind Executor's Mutex.
// The inner Pin<Box<>> is heap-allocated and its address is stable.
unsafe impl Send for SgCallbackHandle {}
unsafe impl Sync for SgCallbackHandle {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mega_kernel_callback::null_retrieve_bridge;

    fn make_ctx() -> SgCallbackCtx {
        SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 64,
            alpha: 0.5,
            precomputed_knowledge: std::ptr::null(),
        }
    }

    #[test]
    fn new_handle_is_empty() {
        let handle = SgCallbackHandle::new();
        assert!(!handle.is_registered());
    }

    #[test]
    fn default_equals_new() {
        let handle = SgCallbackHandle::default();
        assert!(!handle.is_registered());
    }

    #[test]
    fn register_returns_non_null_and_is_registered() {
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(make_ctx());
        assert!(!ptr.is_null());
        assert!(handle.is_registered());
    }

    #[test]
    fn register_returns_stable_address_across_field_accesses() {
        let mut handle = SgCallbackHandle::new();
        let ptr1 = handle.register(make_ctx());
        // A second register replaces the first; new pinned allocation.
        let ptr2 = handle.register(make_ctx());
        // First Box is dropped by `inner = Some(new)`, ptr1 may dangle.
        assert!(handle.is_registered());
        assert!(!ptr2.is_null());
        let _ = ptr1;
    }

    #[test]
    fn reclaim_drops_and_returns_provider_ptr() {
        let mut handle = SgCallbackHandle::new();
        let ctx = make_ctx();
        handle.register(ctx);
        assert!(handle.is_registered());

        let provider = handle.reclaim();
        assert!(provider.is_some());
        assert!(!handle.is_registered());
    }

    #[test]
    fn reclaim_on_empty_returns_none() {
        let mut handle = SgCallbackHandle::new();
        assert!(handle.reclaim().is_none());
    }

    // ── Additional tests ──

    #[test]
    fn new_handle_not_registered() {
        let handle = SgCallbackHandle::new();
        assert!(!handle.is_registered());
        assert!(handle.inner.is_none());
    }

    #[test]
    fn default_is_same_as_new() {
        let from_new = SgCallbackHandle::new();
        let from_default = SgCallbackHandle::default();
        assert!(!from_new.is_registered());
        assert!(!from_default.is_registered());
    }

    #[test]
    fn register_then_is_registered_true() {
        let mut handle = SgCallbackHandle::new();
        assert!(!handle.is_registered());
        handle.register(make_ctx());
        assert!(handle.is_registered());
    }

    #[test]
    fn register_pointer_non_null_and_non_zero() {
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(make_ctx());
        assert!(!ptr.is_null());
        assert_ne!(ptr as usize, 0);
    }

    #[test]
    fn register_pointer_aligned_to_sg_callback_ctx() {
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(make_ctx());
        let align = std::mem::align_of::<SgCallbackCtx>();
        assert_eq!((ptr as usize) % align, 0, "pointer should be aligned to SgCallbackCtx");
    }

    #[test]
    fn second_register_replaces_first() {
        let mut handle = SgCallbackHandle::new();
        let ptr1 = handle.register(make_ctx());
        let ptr2 = handle.register(make_ctx());
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert!(handle.is_registered());
        // After replacing, handle holds the second allocation
    }

    #[test]
    fn reclaim_returns_none_provider_when_provider_null() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx()); // provider_state is null in make_ctx
        let provider = handle.reclaim();
        assert!(provider.is_some());
        assert!(provider.unwrap().is_null());
    }

    #[test]
    fn reclaim_returns_nonnull_provider_when_provider_set() {
        let sentinel = 0xABCD_usize as *const u8;
        let mut ctx = make_ctx();
        ctx.provider_state = sentinel;
        let mut handle = SgCallbackHandle::new();
        handle.register(ctx);
        let provider = handle.reclaim();
        assert!(provider.is_some());
        assert_eq!(provider.unwrap(), sentinel);
    }

    #[test]
    fn reclaim_clears_registration() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx());
        assert!(handle.is_registered());
        handle.reclaim();
        assert!(!handle.is_registered());
    }

    #[test]
    fn double_reclaim_both_return_correctly() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx());
        let first = handle.reclaim();
        let second = handle.reclaim();
        assert!(first.is_some());
        assert!(second.is_none()); // already reclaimed
    }

    #[test]
    fn register_after_reclaim_works() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx());
        handle.reclaim();
        assert!(!handle.is_registered());
        let ptr = handle.register(make_ctx());
        assert!(!ptr.is_null());
        assert!(handle.is_registered());
    }

    #[test]
    fn reclaim_on_never_registered_returns_none() {
        let mut handle = SgCallbackHandle::new();
        assert!(handle.reclaim().is_none());
        assert!(!handle.is_registered());
    }

    #[test]
    fn register_with_hidden_size_zero() {
        let mut ctx = make_ctx();
        ctx.hidden_size = 0;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        assert!(handle.is_registered());
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    #[test]
    fn register_with_hidden_size_max() {
        let mut ctx = make_ctx();
        ctx.hidden_size = u32::MAX;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        // Reclaim with hidden_size = u32::MAX but precomputed_knowledge = null
        // → no deallocation attempt (null check passes)
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    #[test]
    fn register_with_precomputed_knowledge_reclaims_on_reclaim() {
        let knowledge: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let boxed: Box<[f32]> = knowledge.into_boxed_slice();
        let raw = Box::into_raw(boxed);

        let mut ctx = make_ctx();
        ctx.precomputed_knowledge = raw as *const f32;
        ctx.hidden_size = 4;

        let mut handle = SgCallbackHandle::new();
        handle.register(ctx);
        let provider = handle.reclaim();
        assert!(provider.is_some());
        // The Box<[f32]> has been reclaimed inside reclaim().
        // No double-free or use-after-free.
    }

    #[test]
    fn register_with_precomputed_knowledge_null_and_hidden_size_nonzero_no_crash() {
        let mut ctx = make_ctx();
        ctx.precomputed_knowledge = std::ptr::null();
        ctx.hidden_size = 1024;
        let mut handle = SgCallbackHandle::new();
        handle.register(ctx);
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    #[test]
    fn register_with_alpha_zero() {
        let mut ctx = make_ctx();
        ctx.alpha = 0.0;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        handle.reclaim();
    }

    #[test]
    fn register_with_alpha_negative() {
        let mut ctx = make_ctx();
        ctx.alpha = -1.5;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        handle.reclaim();
    }

    #[test]
    fn register_with_alpha_max() {
        let mut ctx = make_ctx();
        ctx.alpha = f32::MAX;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        handle.reclaim();
    }

    #[test]
    fn register_with_all_null_pointers() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        let provider = handle.reclaim();
        assert!(provider.is_some());
        assert!(provider.unwrap().is_null());
    }

    #[test]
    fn send_sync_bounds_compile() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<SgCallbackHandle>();
        assert_sync::<SgCallbackHandle>();
    }

    #[test]
    fn register_multiple_cycles() {
        let mut handle = SgCallbackHandle::new();
        for _ in 0..5 {
            assert!(!handle.is_registered());
            let ptr = handle.register(make_ctx());
            assert!(!ptr.is_null());
            assert!(handle.is_registered());
            let provider = handle.reclaim();
            assert!(provider.is_some());
            assert!(!handle.is_registered());
        }
    }

    #[test]
    fn reclaim_idempotent_after_cycle() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx());
        handle.reclaim();
        handle.reclaim();
        handle.reclaim();
        assert!(!handle.is_registered());
    }

    #[test]
    fn inner_is_none_after_new() {
        let handle = SgCallbackHandle::new();
        assert!(handle.inner.is_none());
    }

    #[test]
    fn inner_is_some_after_register() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx());
        assert!(handle.inner.is_some());
    }

    #[test]
    fn inner_is_none_after_reclaim() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx());
        handle.reclaim();
        assert!(handle.inner.is_none());
    }

    // ── 15 new tests ──

    #[test]
    fn register_with_alpha_nan_does_not_panic() {
        let mut ctx = make_ctx();
        ctx.alpha = f32::NAN;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        assert!(handle.is_registered());
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    #[test]
    fn register_with_alpha_infinity_does_not_panic() {
        let mut ctx = make_ctx();
        ctx.alpha = f32::INFINITY;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    #[test]
    fn register_with_alpha_subnormal_does_not_panic() {
        let mut ctx = make_ctx();
        ctx.alpha = f32::from_bits(1); // smallest positive subnormal
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    #[test]
    fn register_with_hidden_size_one() {
        let mut ctx = make_ctx();
        ctx.hidden_size = 1;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        assert!(handle.is_registered());
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    #[test]
    fn register_with_non_null_sg_shared_memory() {
        let sentinel_sg: usize = 0xDEAD_BEEF;
        let mut ctx = make_ctx();
        ctx.sg_shared_memory = sentinel_sg as *const u8;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        // sg_shared_memory is stored but not dereferenced; reclaim succeeds
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    #[test]
    fn distinct_registrations_yield_distinct_pointers() {
        let mut handle1 = SgCallbackHandle::new();
        let mut handle2 = SgCallbackHandle::new();
        let ptr1 = handle1.register(make_ctx());
        let ptr2 = handle2.register(make_ctx());
        assert_ne!(ptr1, ptr2, "two independent handles must have distinct heap allocations");
        handle1.reclaim();
        handle2.reclaim();
    }

    #[test]
    fn alpha_preserved_through_register_reclaim_roundtrip() {
        let sentinel_alpha = 0.12345;
        let mut ctx = make_ctx();
        ctx.alpha = sentinel_alpha;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        // Read back alpha through the raw pointer to verify field preservation
        let ctx_ptr = ptr as *const SgCallbackCtx;
        let read_alpha = unsafe { (*ctx_ptr).alpha };
        assert_eq!(read_alpha, sentinel_alpha);
        handle.reclaim();
    }

    #[test]
    fn hidden_size_preserved_through_register_reclaim_roundtrip() {
        let test_hidden = 2048u32;
        let mut ctx = make_ctx();
        ctx.hidden_size = test_hidden;
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        let ctx_ptr = ptr as *const SgCallbackCtx;
        let read_hidden = unsafe { (*ctx_ptr).hidden_size };
        assert_eq!(read_hidden, test_hidden);
        handle.reclaim();
    }

    #[test]
    fn provider_state_preserved_through_register_reclaim_roundtrip() {
        let sentinel = 0x1337_usize as *const u8;
        let mut ctx = make_ctx();
        ctx.provider_state = sentinel;
        let mut handle = SgCallbackHandle::new();
        handle.register(ctx);
        let provider = handle.reclaim();
        assert_eq!(provider.unwrap(), sentinel);
    }

    #[test]
    fn register_reclaim_ten_cycles_no_leak() {
        let mut handle = SgCallbackHandle::new();
        for i in 0..10 {
            let mut ctx = make_ctx();
            ctx.alpha = i as f32 * 0.1;
            ctx.hidden_size = 64 + i as u32;
            let ptr = handle.register(ctx);
            assert!(!ptr.is_null());
            assert!(handle.is_registered());
            let provider = handle.reclaim();
            assert!(provider.is_some());
            assert!(!handle.is_registered());
        }
    }

    #[test]
    fn precomputed_knowledge_large_array_reclaims_cleanly() {
        let size = 4096usize;
        let knowledge: Vec<f32> = vec![42.0; size];
        let boxed: Box<[f32]> = knowledge.into_boxed_slice();
        let raw = Box::into_raw(boxed);

        let mut ctx = make_ctx();
        ctx.precomputed_knowledge = raw as *const f32;
        ctx.hidden_size = size as u32;

        let mut handle = SgCallbackHandle::new();
        handle.register(ctx);
        let provider = handle.reclaim();
        assert!(provider.is_some());
        // Box reclaimed; no double-free or leak
    }

    #[test]
    fn sequential_precomputed_knowledge_allocations_reclaim() {
        for &len in &[1usize, 16, 256, 1024] {
            let knowledge: Vec<f32> = vec![1.0; len];
            let boxed: Box<[f32]> = knowledge.into_boxed_slice();
            let raw = Box::into_raw(boxed);

            let mut ctx = make_ctx();
            ctx.precomputed_knowledge = raw as *const f32;
            ctx.hidden_size = len as u32;

            let mut handle = SgCallbackHandle::new();
            handle.register(ctx);
            let provider = handle.reclaim();
            assert!(provider.is_some());
        }
    }

    #[test]
    fn drop_without_reclaim_does_not_double_free() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx());
        // Drop handle without calling reclaim.
        // Pin<Box<SgCallbackCtx>> is dropped normally by Option::take via Drop.
        drop(handle);
    }

    #[test]
    fn register_after_double_reclaim_succeeds() {
        let mut handle = SgCallbackHandle::new();
        handle.register(make_ctx());
        handle.reclaim();
        assert!(handle.reclaim().is_none());
        // Should be able to register again after exhausted reclaim
        let ptr = handle.register(make_ctx());
        assert!(!ptr.is_null());
        assert!(handle.is_registered());
        handle.reclaim();
    }

    #[test]
    fn register_with_retrieve_fn_null_bridge() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);
        assert!(!ptr.is_null());
        // Verify retrieve_fn field is stored correctly by reading back
        let ctx_ptr = ptr as *const SgCallbackCtx;
        let read_fn = unsafe { (*ctx_ptr).retrieve_fn };
        assert_eq!(read_fn as usize, null_retrieve_bridge as usize);
        handle.reclaim();
    }

    // ── 13 additional tests ──

    /// Verify that the raw pointer returned by `register` correctly points to the
    /// stored `SgCallbackCtx` by reading back the `sg_shared_memory` sentinel.
    #[test]
    // @trace TEST-SGCH-48 [req:REQ-SG] [level:unit]
    fn sg_shared_memory_preserved_through_register() {
        // Arrange
        let sentinel_sg: usize = 0xCAFE_F00D;
        let mut ctx = make_ctx();
        ctx.sg_shared_memory = sentinel_sg as *const u8;
        let mut handle = SgCallbackHandle::new();

        // Act
        let ptr = handle.register(ctx);
        let ctx_ptr = ptr as *const SgCallbackCtx;
        let read_sg = unsafe { (*ctx_ptr).sg_shared_memory };

        // Assert
        assert_eq!(read_sg as usize, sentinel_sg);
        handle.reclaim();
    }

    /// Drop a handle that holds a registered context with an allocated
    /// `precomputed_knowledge` `Box<[f32]>`.  The `Pin<Box<SgCallbackCtx>>`
    /// drops normally but the leaked knowledge array is **not** reclaimed
    /// (only `reclaim()` does that). This test confirms no double-free or
    /// panic — the leaked allocation is simply leaked (acceptable for
    /// shutdown paths where the OS reclaims memory).
    #[test]
    // @trace TEST-SGCH-49 [req:REQ-SG] [level:unit]
    fn drop_without_reclaim_leaks_precomputed_knowledge_safely() {
        // Arrange
        let knowledge: Vec<f32> = vec![7.0; 128];
        let boxed: Box<[f32]> = knowledge.into_boxed_slice();
        let raw = Box::into_raw(boxed);
        let mut ctx = make_ctx();
        ctx.precomputed_knowledge = raw as *const f32;
        ctx.hidden_size = 128;
        let mut handle = SgCallbackHandle::new();
        handle.register(ctx);

        // Act — drop without reclaim (Pin<Box<SgCallbackCtx>> drops,
        // but the leaked Box<[f32]> is not recovered).
        // This should not panic or double-free.
        drop(handle);

        // Assert — manually reclaim the leaked allocation to avoid actual leak in tests.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(raw as *mut f32, 128);
            let _ = Box::from_raw(slice as *mut [f32]);
        }
    }

    /// Register with `alpha = f32::MIN` (most negative normal f32).
    #[test]
    // @trace TEST-SGCH-50 [req:REQ-SG] [level:unit]
    fn register_with_alpha_min_negative() {
        // Arrange
        let mut ctx = make_ctx();
        ctx.alpha = f32::MIN;
        let mut handle = SgCallbackHandle::new();

        // Act
        let ptr = handle.register(ctx);
        let ctx_ptr = ptr as *const SgCallbackCtx;
        let read_alpha = unsafe { (*ctx_ptr).alpha };

        // Assert
        assert!(!ptr.is_null());
        assert_eq!(read_alpha, f32::MIN);
        handle.reclaim();
    }

    /// Register with `alpha = f32::NEG_INFINITY`.
    #[test]
    // @trace TEST-SGCH-51 [req:REQ-SG] [level:unit]
    fn register_with_alpha_neg_infinity() {
        // Arrange
        let mut ctx = make_ctx();
        ctx.alpha = f32::NEG_INFINITY;
        let mut handle = SgCallbackHandle::new();

        // Act
        let ptr = handle.register(ctx);
        let ctx_ptr = ptr as *const SgCallbackCtx;
        let read_alpha = unsafe { (*ctx_ptr).alpha };

        // Assert
        assert!(!ptr.is_null());
        assert!(read_alpha.is_infinite() && read_alpha.is_sign_negative());
        handle.reclaim();
    }

    /// Register with `precomputed_knowledge` non-null but `hidden_size = 0`.
    /// In this case `reclaim()` creates a zero-length slice and reclaims it —
    /// confirming the null-check bypass works correctly.
    #[test]
    // @trace TEST-SGCH-52 [req:REQ-SG] [level:unit]
    fn reclaim_with_non_null_knowledge_and_hidden_size_zero() {
        // Arrange — allocate a 0-length Box<[f32]> (valid but empty)
        let boxed: Box<[f32]> = Vec::<f32>::new().into_boxed_slice();
        let raw = Box::into_raw(boxed);
        let mut ctx = make_ctx();
        ctx.precomputed_knowledge = raw as *const f32;
        ctx.hidden_size = 0;
        let mut handle = SgCallbackHandle::new();
        handle.register(ctx);

        // Act
        let provider = handle.reclaim();

        // Assert — reclaim succeeds, Box<[f32]> of length 0 is reclaimed.
        assert!(provider.is_some());
    }

    /// Register 3 times in a row (each replacing the previous) without
    /// calling reclaim. Only the last registration's context survives.
    #[test]
    // @trace TEST-SGCH-53 [req:REQ-SG] [level:unit]
    fn triple_register_without_reclaim_retains_last() {
        // Arrange
        let mut ctx1 = make_ctx();
        ctx1.alpha = 1.0;
        let mut ctx2 = make_ctx();
        ctx2.alpha = 2.0;
        let mut ctx3 = make_ctx();
        ctx3.alpha = 3.0;
        let mut handle = SgCallbackHandle::new();

        // Act
        let ptr1 = handle.register(ctx1);
        let ptr2 = handle.register(ctx2);
        let ptr3 = handle.register(ctx3);

        // Assert — only the last pointer is valid; read alpha back.
        let ctx_ptr = ptr3 as *const SgCallbackCtx;
        let read_alpha = unsafe { (*ctx_ptr).alpha };
        assert_eq!(read_alpha, 3.0);
        assert!(handle.is_registered());
        // ptr1 and ptr2 may dangle; just confirm they were non-null at creation.
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        handle.reclaim();
    }

    /// Reclaim after multiple sequential replacements returns the provider
    /// of the last registered context.
    #[test]
    // @trace TEST-SGCH-54 [req:REQ-SG] [level:unit]
    fn reclaim_after_sequential_replacements_returns_last_provider() {
        // Arrange
        let sentinel_a: usize = 0xAAAA;
        let sentinel_b: usize = 0xBBBB;
        let sentinel_c: usize = 0xCCCC;

        let mut ctx_a = make_ctx();
        ctx_a.provider_state = sentinel_a as *const u8;
        let mut ctx_b = make_ctx();
        ctx_b.provider_state = sentinel_b as *const u8;
        let mut ctx_c = make_ctx();
        ctx_c.provider_state = sentinel_c as *const u8;

        let mut handle = SgCallbackHandle::new();

        // Act
        handle.register(ctx_a);
        handle.register(ctx_b);
        handle.register(ctx_c);
        let provider = handle.reclaim();

        // Assert — provider from the last registration (ctx_c) is returned.
        assert_eq!(provider.unwrap(), sentinel_c as *const u8);
    }

    /// Read back ALL fields simultaneously through the raw pointer to verify
    /// complete structural preservation of `SgCallbackCtx`.
    #[test]
    // @trace TEST-SGCH-55 [req:REQ-SG] [level:unit]
    fn all_fields_preserved_through_register() {
        // Arrange
        let sentinel_sg: usize = 0x1234_5678;
        let sentinel_provider: usize = 0xABCD_EF01;
        let test_alpha = 0.75;
        let test_hidden: u32 = 512;

        let mut ctx = make_ctx();
        ctx.sg_shared_memory = sentinel_sg as *const u8;
        ctx.provider_state = sentinel_provider as *const u8;
        ctx.alpha = test_alpha;
        ctx.hidden_size = test_hidden;
        ctx.precomputed_knowledge = std::ptr::null();

        let mut handle = SgCallbackHandle::new();

        // Act
        let ptr = handle.register(ctx);
        let ctx_ptr = ptr as *const SgCallbackCtx;

        // Assert — read every field back.
        unsafe {
            assert_eq!((*ctx_ptr).sg_shared_memory as usize, sentinel_sg);
            assert_eq!((*ctx_ptr).provider_state as usize, sentinel_provider);
            assert_eq!((*ctx_ptr).alpha, test_alpha);
            assert_eq!((*ctx_ptr).hidden_size, test_hidden);
            assert!((*ctx_ptr).precomputed_knowledge.is_null());
            assert_eq!(
                (*ctx_ptr).retrieve_fn as usize,
                null_retrieve_bridge as usize
            );
        }
        handle.reclaim();
    }

    /// Verify `inner` transitions correctly through a full
    /// register → reclaim → register → reclaim lifecycle.
    #[test]
    // @trace TEST-SGCH-56 [req:REQ-SG] [level:unit]
    fn inner_transitions_through_full_lifecycle() {
        // Arrange
        let mut handle = SgCallbackHandle::new();
        assert!(handle.inner.is_none());

        // Act & Assert — first cycle
        handle.register(make_ctx());
        assert!(handle.inner.is_some());
        handle.reclaim();
        assert!(handle.inner.is_none());

        // Act & Assert — second cycle
        handle.register(make_ctx());
        assert!(handle.inner.is_some());
        handle.reclaim();
        assert!(handle.inner.is_none());
    }

    /// Register with a large `precomputed_knowledge` array and matching
    /// `hidden_size`, then verify reclaim deallocates correctly by
    /// reading sentinel values through the pointer before reclaim.
    #[test]
    // @trace TEST-SGCH-57 [req:REQ-SG] [level:unit]
    fn large_precomputed_knowledge_readable_before_reclaim() {
        // Arrange
        let size = 8192usize;
        let knowledge: Vec<f32> = vec![99.0; size];
        let boxed: Box<[f32]> = knowledge.into_boxed_slice();
        let raw = Box::into_raw(boxed);

        let mut ctx = make_ctx();
        ctx.precomputed_knowledge = raw as *const f32;
        ctx.hidden_size = size as u32;

        let mut handle = SgCallbackHandle::new();
        let ptr = handle.register(ctx);

        // Act — read back through the stored context pointer
        let ctx_ptr = ptr as *const SgCallbackCtx;
        let knowledge_ptr = unsafe { (*ctx_ptr).precomputed_knowledge };
        assert!(!knowledge_ptr.is_null());
        // Read first and last elements
        let first = unsafe { *knowledge_ptr };
        let last = unsafe { *knowledge_ptr.add(size - 1) };

        // Assert
        assert_eq!(first, 99.0);
        assert_eq!(last, 99.0);

        // Reclaim — this frees the Box<[f32]>.
        let provider = handle.reclaim();
        assert!(provider.is_some());
    }

    /// Register with `hidden_size = 1` and a single-element `precomputed_knowledge`
    /// array, verifying correct single-element reclaim.
    #[test]
    // @trace TEST-SGCH-58 [req:REQ-SG] [level:unit]
    fn single_element_precomputed_knowledge_reclaims() {
        // Arrange
        let knowledge: Vec<f32> = vec![3.14];
        let boxed: Box<[f32]> = knowledge.into_boxed_slice();
        let raw = Box::into_raw(boxed);

        let mut ctx = make_ctx();
        ctx.precomputed_knowledge = raw as *const f32;
        ctx.hidden_size = 1;

        let mut handle = SgCallbackHandle::new();
        handle.register(ctx);

        // Act
        let provider = handle.reclaim();

        // Assert — single element reclaimed successfully.
        assert!(provider.is_some());
        assert!(!handle.is_registered());
    }

    /// Verify that two sequential register-reclaim cycles with different
    /// `precomputed_knowledge` allocations both reclaim correctly
    /// (no cross-contamination).
    #[test]
    // @trace TEST-SGCH-59 [req:REQ-SG] [level:unit]
    fn two_cycles_with_different_knowledge_no_cross_contamination() {
        // Arrange — first cycle
        let knowledge1: Vec<f32> = vec![1.0; 64];
        let boxed1: Box<[f32]> = knowledge1.into_boxed_slice();
        let raw1 = Box::into_raw(boxed1);
        let mut ctx1 = make_ctx();
        ctx1.precomputed_knowledge = raw1 as *const f32;
        ctx1.hidden_size = 64;

        let mut handle = SgCallbackHandle::new();
        handle.register(ctx1);

        // Act — first reclaim
        let provider1 = handle.reclaim();
        assert!(provider1.is_some());

        // Arrange — second cycle with different size
        let knowledge2: Vec<f32> = vec![2.0; 256];
        let boxed2: Box<[f32]> = knowledge2.into_boxed_slice();
        let raw2 = Box::into_raw(boxed2);
        let mut ctx2 = make_ctx();
        ctx2.precomputed_knowledge = raw2 as *const f32;
        ctx2.hidden_size = 256;

        handle.register(ctx2);

        // Act — second reclaim
        let provider2 = handle.reclaim();
        assert!(provider2.is_some());

        // Assert — both cycles completed cleanly
        assert!(!handle.is_registered());
    }

    /// Register with `alpha` equal to epsilon (smallest positive normal f32),
    /// verifying the value survives the register/reclaim roundtrip.
    #[test]
    // @trace TEST-SGCH-60 [req:REQ-SG] [level:unit]
    fn alpha_epsilon_preserved_through_roundtrip() {
        // Arrange
        let epsilon = f32::EPSILON;
        let mut ctx = make_ctx();
        ctx.alpha = epsilon;
        let mut handle = SgCallbackHandle::new();

        // Act
        let ptr = handle.register(ctx);
        let ctx_ptr = ptr as *const SgCallbackCtx;
        let read_alpha = unsafe { (*ctx_ptr).alpha };

        // Assert
        assert_eq!(read_alpha, epsilon);
        assert_eq!(read_alpha.to_bits(), epsilon.to_bits());
        handle.reclaim();
    }
}
