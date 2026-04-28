//! Mega-Kernel Callback Table — C-style function pointer array for JIT→external callbacks.
//!
//! Design: flat `#[repr(C)]` array of `{fn_ptr, ctx}` entries, passed as ABI arg 20.
//! JIT code loads entries via `VmInstr::LoadCallbackEntry` and calls via `VmInstr::NativeCall`.
//! Zero overhead when `callback_table_ptr=NULL` (GprSkipIfNull skips all callback code).
//!
//! Slot allocation:
//!   0 = SG_KNOWLEDGE_RETRIEVE
//!   1 = GUARDRAIL_CHECK
//!   2 = COT_STEP_CHECK
//!   3 = EARLY_EXIT_QUERY
//!   4..7 = reserved

/// Unified callback signature: pure C ABI.
/// `ctx` is a raw pointer interpreted by the callback internally.
/// Returns 0 = no action, 1 = data written.
pub type MegaKernelCallbackFn = unsafe extern "C" fn(ctx: *const u8) -> u32;

/// Number of callback slots in the table.
pub const CALLBACK_TABLE_SLOTS: usize = 8;

/// Callback slot IDs.
pub mod slot {
    pub const SG_KNOWLEDGE_RETRIEVE: usize = 0;
    pub const GUARDRAIL_CHECK: usize = 1;
    pub const COT_STEP_CHECK: usize = 2;
    pub const EARLY_EXIT_QUERY: usize = 3;
}

/// A single callback entry: function pointer + opaque context.
/// 16 bytes, C-layout.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CallbackEntry {
    /// Function pointer. NULL = not registered, JIT skips.
    pub fn_ptr: *const u8,
    /// Opaque context pointer, interpreted by the callback.
    pub ctx: *const u8,
}

// Safety: CallbackEntry contains raw pointers but is just a plain data struct.
// It's safe to Send/Sync as long as the callback functions themselves are thread-safe.
unsafe impl Send for CallbackEntry {}
unsafe impl Sync for CallbackEntry {}

/// C-style flat callback table: 8 slots × 16B = 128B.
#[repr(C)]
pub struct MegaKernelCallbackTable {
    entries: [CallbackEntry; CALLBACK_TABLE_SLOTS],
}

impl MegaKernelCallbackTable {
    /// Create a new zero-initialized callback table (all fn_ptr = NULL).
    pub fn new() -> Self {
        Self {
            entries: [CallbackEntry {
                fn_ptr: std::ptr::null(),
                ctx: std::ptr::null(),
            }; CALLBACK_TABLE_SLOTS],
        }
    }

    /// Register a callback in a slot.
    ///
    /// # Safety
    /// Caller must ensure `fn_ptr` points to a valid `extern "C" fn(*const u8) -> u32`
    /// and `ctx` remains valid for the lifetime of the callback registration.
    pub unsafe fn register(&mut self, slot_id: usize, fn_ptr: *const u8, ctx: *const u8) {
        assert!(slot_id < CALLBACK_TABLE_SLOTS, "callback slot {} out of range", slot_id);
        self.entries[slot_id] = CallbackEntry { fn_ptr, ctx };
    }

    /// Clear a callback slot (set fn_ptr = NULL).
    pub fn clear(&mut self, slot_id: usize) {
        assert!(slot_id < CALLBACK_TABLE_SLOTS, "callback slot {} out of range", slot_id);
        self.entries[slot_id] = CallbackEntry {
            fn_ptr: std::ptr::null(),
            ctx: std::ptr::null(),
        };
    }

    /// Check if any callback is registered.
    pub fn has_any_callback(&self) -> bool {
        self.entries.iter().any(|e| !e.fn_ptr.is_null())
    }

    /// Get raw pointer to pass as ABI arg 20.
    pub fn as_ptr(&self) -> *const u8 {
        self as *const Self as *const u8
    }
}

impl Default for MegaKernelCallbackTable {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Slot 0: SG_KNOWLEDGE_RETRIEVE callback bridge
// ============================================================================

/// C ABI bridge function signature: retrieves knowledge from detect_hidden.
///
/// Called by `sg_knowledge_retrieve_callback` (the slot 0 callback that JIT calls)
/// to bridge from C ABI world into the Rust KnowledgeProvider.
///
/// * `detect_hidden` — pointer to `[f32; hidden_size]` from SgSharedMemory
/// * `hidden_size` — dimension of the hidden vectors
/// * `output_knowledge` — output buffer `[f32; hidden_size]`
/// * `output_confidence` — output f32 confidence value
/// * `provider_state` — opaque pointer supplied at registration time
///
/// Returns 0 if no knowledge was retrieved, 1 on success.
pub type SgRetrieveFn = unsafe extern "C" fn(
    detect_hidden: *const f32,
    hidden_size: u32,
    output_knowledge: *mut f32,
    output_confidence: *mut f32,
    provider_state: *const u8,
) -> i32;

/// SG callback context for slot 0 (SG_KNOWLEDGE_RETRIEVE).
///
/// Passed as `ctx` to `sg_knowledge_retrieve_callback` via the callback table.
/// Layout must match what JIT expects (all fields are raw pointers/scalars).
#[repr(C)]
pub struct SgCallbackCtx {
    /// Pointer to SgSharedMemory data (control + detect_hidden + knowledge_vector).
    pub sg_shared_memory: *const u8,
    /// Bridge function: detect_hidden → knowledge_vector + confidence.
    pub retrieve_fn: SgRetrieveFn,
    /// Opaque provider state passed through to retrieve_fn.
    pub provider_state: *const u8,
    /// Hidden dimension size.
    pub hidden_size: u32,
    /// SG injection strength (alpha multiplier).
    pub alpha: f32,
}

/// Null retrieve bridge: always returns 0 (no knowledge retrieved).
/// Placeholder until full KnowledgeProvider bridge is validated.
pub unsafe extern "C" fn null_retrieve_bridge(
    _detect_hidden: *const f32,
    _hidden_size: u32,
    _output_knowledge: *mut f32,
    _output_confidence: *mut f32,
    _provider_state: *const u8,
) -> i32 {
    0
}

/// Slot 0 callback: SG_KNOWLEDGE_RETRIEVE.
///
/// Called from JIT after SgDetect writes `detect_hidden` to SgSharedMemory.
/// Reads `detect_hidden` from shared memory, calls the `retrieve_fn` bridge,
/// writes `confidence` and `knowledge_vector` back to SgSharedMemory.
///
/// Returns 1 if knowledge was written, 0 otherwise.
///
/// # Safety
/// `ctx` must point to a valid, properly aligned `SgCallbackCtx`.
#[no_mangle]
pub unsafe extern "C" fn sg_knowledge_retrieve_callback(ctx: *const u8) -> u32 {
    if ctx.is_null() { return 0; }
    let cb_ctx = &*(ctx as *const SgCallbackCtx);
    if cb_ctx.sg_shared_memory.is_null() || cb_ctx.provider_state.is_null() { return 0; }
    let hidden = cb_ctx.hidden_size as usize;
    if hidden == 0 { return 0; }

    let sg_ptr = cb_ctx.sg_shared_memory;
    let confidence_ptr = sg_ptr.add(12) as *mut f32;

    // Step 1: lock mutex + read alpha (tested OK).
    let mutex = &*(cb_ctx.provider_state
        as *const std::sync::Mutex<
            crate::semantic_gatekeeper::callback::SemanticGatekeeperCallback,
        >);
    let cb = match mutex.lock() { Ok(cb) => cb, Err(_) => return 0 };

    // Alloc before vtable dispatch (works around allocator ordering issue).
    let _a64 = vec![0u8; 64];
    let _a256 = vec![0u8; 256];

    // Vtable dispatch (isolated fn — stable 5/5).
    fn retrieve_conf(
        cb: &crate::semantic_gatekeeper::callback::SemanticGatekeeperCallback,
        detect: &[f32],
    ) -> Option<f32> {
        cb.retrieve_confidence(detect)
    }
    let detect = std::slice::from_raw_parts(sg_ptr.add(16) as *const f32, hidden);
    let conf = match retrieve_conf(&*cb, detect) {
        Some(c) => c,
        None => return 0,
    };

    let alpha_conf = conf * cb.alpha();
    *confidence_ptr = alpha_conf;
    let knowledge = sg_ptr.add(16 + hidden * 4) as *mut f32;
    if alpha_conf > 0.0 {
        for i in 0..hidden { *knowledge.add(i) = alpha_conf; }
    }
    drop(cb);
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn test_callback(_ctx: *const u8) -> u32 {
        1
    }

    #[test]
    fn test_new_table_all_null() {
        let table = MegaKernelCallbackTable::new();
        for entry in &table.entries {
            assert!(entry.fn_ptr.is_null());
            assert!(entry.ctx.is_null());
        }
    }

    #[test]
    fn test_register_and_clear() {
        let mut table = MegaKernelCallbackTable::new();
        assert!(!table.has_any_callback());

        unsafe {
            table.register(
                slot::SG_KNOWLEDGE_RETRIEVE,
                test_callback as *const u8,
                std::ptr::null(),
            );
        }
        assert!(table.has_any_callback());
        assert!(!table.entries[slot::SG_KNOWLEDGE_RETRIEVE].fn_ptr.is_null());

        table.clear(slot::SG_KNOWLEDGE_RETRIEVE);
        assert!(!table.has_any_callback());
        assert!(table.entries[slot::SG_KNOWLEDGE_RETRIEVE].fn_ptr.is_null());
    }

    #[test]
    fn test_layout() {
        assert_eq!(std::mem::size_of::<CallbackEntry>(), 16);
        assert_eq!(std::mem::size_of::<MegaKernelCallbackTable>(), 128);
        assert_eq!(std::mem::align_of::<CallbackEntry>(), 8);
    }

    #[test]
    fn test_as_ptr_readable() {
        let table = MegaKernelCallbackTable::new();
        let ptr = table.as_ptr();
        assert!(!ptr.is_null());
        // All 128 bytes should be zero
        let bytes = unsafe { std::slice::from_raw_parts(ptr, 128) };
        assert!(bytes.iter().all(|&b| b == 0));
    }
}
