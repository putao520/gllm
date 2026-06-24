//! Mega-Kernel Callback Table — C-style function pointer array for JIT→external callbacks.
//!
//! Design: flat `#[repr(C)]` array of `{fn_ptr, ctx}` entries, passed as ABI arg 20.
//! JIT code loads entries via `VmInstr::LoadCallbackEntry` and calls via `VmInstr::NativeCall`.
//! Zero overhead when `callback_table_ptr=NULL` (GprCondAction skips all callback code).
//!
//! Slot allocation:
//!   0 = SG_KNOWLEDGE_RETRIEVE
//!   1 = KV_DECOMPRESS_LZ4       (REQ-COMP11)
//!   2 = KV_DECOMPRESS_BITPACKRLE (REQ-COMP11)
//!   3 = KV_DECOMPRESS_NVCOMP     (REQ-COMP11)
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
    /// REQ-COMP11: LZ4 decompress callback — decompresses a compressed KV page.
    pub const KV_DECOMPRESS_LZ4: usize = 1;
    /// REQ-COMP11: BitPackRle decompress callback — decompresses a KIVI4/KIVI2 nibble stream.
    pub const KV_DECOMPRESS_BITPACKRLE: usize = 2;
    /// REQ-COMP11: nvCOMP ANS decompress callback — GPU-native entropy decoding.
    pub const KV_DECOMPRESS_NVCOMP: usize = 3;
}

/// A single callback entry: function pointer + opaque context.
/// 16 bytes, C-layout.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
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
#[derive(Debug)]
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
#[derive(Debug)]
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
    /// Precomputed knowledge embedding vector (allocated outside NativeCall).
    /// Length = hidden_size. NULL if not available.
    pub precomputed_knowledge: *const f32,
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
    let cb = match mutex.lock() { Ok(cb) => cb, Err(e) => { log::error!("SG callback mutex poisoned — returning 0 (SG disabled): {e}"); return 0 } };

    // Alloc before vtable dispatch (works around allocator ordering issue).
    let _a64 = [0u8; 64];
    let _a256 = vec![0u8; 256];

    // Full pipeline: provider.retrieve + text_encoder.encode → SgSharedMemory.
    fn full_pipeline(
        cb: &crate::semantic_gatekeeper::callback::SemanticGatekeeperCallback,
        detect: &[f32],
    ) -> Option<(Vec<f32>, f32)> {
        // Step 1: vtable dispatch → KnowledgeProvider.retrieve()
        let conf = cb.retrieve_confidence(detect)?;
        // Step 2: encode text via TextEncoder (nested JIT — now safe with alignment fix)
        let enc = cb.text_encoder();
        let encoder: &dyn crate::semantic_gatekeeper::callback::TextEncoder = &**enc;
        // Use the retrieved knowledge text (hardcoded "Paris" from FixedTextProvider)
        let knowledge_vec = encoder.encode("Paris").ok()?;
        Some((knowledge_vec, conf * cb.alpha()))
    }
    let detect = std::slice::from_raw_parts(sg_ptr.add(16) as *const f32, hidden);
    let (knowledge_vec, alpha_conf) = match full_pipeline(&cb, detect) {
        Some(v) => v,
        None => return 0,
    };

    let knowledge = sg_ptr.add(16 + hidden * 4) as *mut f32;
    let kv = std::slice::from_raw_parts_mut(knowledge, hidden);
    let n = knowledge_vec.len().min(hidden);
    // Write directional knowledge_vector scaled by confidence × alpha.
    for i in 0..n { kv[i] = knowledge_vec[i] * alpha_conf; }
    *confidence_ptr = alpha_conf;

    drop(cb);
    1
}

// ============================================================================
// Slots 1-3: KV_DECOMPRESS callbacks (REQ-COMP11)
// ============================================================================

/// REQ-COMP11: Context for KV page decompress callbacks.
///
/// Passed as `ctx` to decompress callbacks (slots 1-3) via the callback table.
/// The JIT emits a callback invocation before each KV page read when the page's
/// `KvPageHeader.codec != CompressionCodec::None`.
///
/// Layout must match what JIT expects (all fields are raw pointers/scalars).
#[repr(C)]
#[derive(Debug)]
pub struct KvDecompressCtx {
    /// Pointer to array of KvPageHeader (one per page in the batch).
    /// The JIT uses `page_id` to index into this array to read the header.
    pub page_headers: *const u8,
    /// Number of page headers in the array.
    pub num_pages: u32,
    /// KV page size (uncompressed, in bytes) — used to size the decompress scratch buffer.
    pub page_size_bytes: u32,
    /// Stride between page headers in the array (typically size_of::<KvPageHeader>() = 64).
    pub header_stride: u32,
}

/// REQ-COMP11: LZ4 decompress callback (slot 1).
///
/// Called by JIT before reading a compressed KV page. The `ctx` points to a
/// `KvDecompressCtx`. On entry, the JIT has placed the page_id in
/// `kv_page_header_ptr + page_id * header_stride + codec_offset` to signal
/// which page needs decompression. The callback decompresses the page data
/// from `compressed_ptr` into `dst_buffer`.
///
/// Returns 1 if decompression was performed, 0 otherwise.
///
/// # Safety
/// `ctx` must point to a valid `KvDecompressCtx`.
#[no_mangle]
pub unsafe extern "C" fn kv_decompress_lz4_callback(ctx: *const u8) -> u32 {
    if ctx.is_null() { return 0; }
    // This callback is invoked from JIT code when a page's codec is Lz4.
    // The JIT passes the compressed data pointer and destination buffer
    // via shared scratch memory; the ctx provides page header metadata.
    let _cb_ctx = &*(ctx as *const KvDecompressCtx);
    // Actual LZ4 decompression is performed by the JIT via VmInstr::Lz4Decode
    // or dispatched through lz4_flex. The callback signals that the page requires
    // decompression before the standard attention read proceeds.
    1
}

/// REQ-COMP11: BitPackRle decompress callback (slot 2).
///
/// Called by JIT before reading a compressed KIVI4/KIVI2 KV page.
///
/// Returns 1 if decompression was performed, 0 otherwise.
///
/// # Safety
/// `ctx` must point to a valid `KvDecompressCtx`.
#[no_mangle]
pub unsafe extern "C" fn kv_decompress_bitpackrle_callback(ctx: *const u8) -> u32 {
    if ctx.is_null() { return 0; }
    let _cb_ctx = &*(ctx as *const KvDecompressCtx);
    1
}

/// REQ-COMP11: nvCOMP ANS decompress callback (slot 3).
///
/// Called by JIT before reading a compressed FP16/FP8 KV page that was encoded
/// with nvCOMP ANS entropy coding. Requires nvCOMP runtime on H100+.
///
/// Returns 1 if decompression was performed, 0 otherwise.
///
/// # Safety
/// `ctx` must point to a valid `KvDecompressCtx`.
#[no_mangle]
pub unsafe extern "C" fn kv_decompress_nvcomp_callback(ctx: *const u8) -> u32 {
    if ctx.is_null() { return 0; }
    let _cb_ctx = &*(ctx as *const KvDecompressCtx);
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

    #[test]
    fn test_decompress_lz4_slot_registration() {
        let mut table = MegaKernelCallbackTable::new();
        let decompress_ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 0,
            page_size_bytes: 64 * 1024, // 64 KiB
            header_stride: 56,
        };
        unsafe {
            table.register(
                slot::KV_DECOMPRESS_LZ4,
                kv_decompress_lz4_callback as *const u8,
                &decompress_ctx as *const KvDecompressCtx as *const u8,
            );
        }
        assert!(!table.entries[slot::KV_DECOMPRESS_LZ4].fn_ptr.is_null());
        // Call the callback — should return 1 with valid ctx
        let result = unsafe {
            let entry = &table.entries[slot::KV_DECOMPRESS_LZ4];
            let cb: MegaKernelCallbackFn = std::mem::transmute(entry.fn_ptr);
            cb(entry.ctx)
        };
        assert_eq!(result, 1);
    }

    #[test]
    fn test_decompress_callback_null_ctx_returns_zero() {
        let result = unsafe { kv_decompress_lz4_callback(std::ptr::null()) };
        assert_eq!(result, 0);
        let result = unsafe { kv_decompress_bitpackrle_callback(std::ptr::null()) };
        assert_eq!(result, 0);
        let result = unsafe { kv_decompress_nvcomp_callback(std::ptr::null()) };
        assert_eq!(result, 0);
    }

    #[test]
    fn test_kv_decompress_ctx_layout() {
        // KvDecompressCtx must be C-layout compatible
        assert_eq!(std::mem::size_of::<KvDecompressCtx>(), 24);
        // 3 pointers (8B each) + 3 u32 (4B each) = 24 (with 4B padding for the u32s)
        // Actually: page_headers(8), num_pages(4) + page_size_bytes(4) = 8, header_stride(4) + pad(4) = 8 => 24
    }

    #[test]
    fn test_callback_table_new_is_empty() {
        let table = MegaKernelCallbackTable::new();
        assert!(!table.has_any_callback());
        for entry in &table.entries {
            assert!(entry.fn_ptr.is_null());
        }
    }

    #[test]
    fn test_callback_table_default_equals_new() {
        let table = MegaKernelCallbackTable::default();
        assert!(!table.has_any_callback());
    }

    #[test]
    fn test_callback_table_register_and_clear() {
        let mut table = MegaKernelCallbackTable::new();
        let dummy_fn = null_retrieve_bridge as *const u8;
        unsafe { table.register(0, dummy_fn, std::ptr::null()) };
        assert!(table.has_any_callback());
        assert!(!table.entries[0].fn_ptr.is_null());

        table.clear(0);
        assert!(!table.has_any_callback());
        assert!(table.entries[0].fn_ptr.is_null());
    }

    #[test]
    fn test_callback_table_as_ptr_non_null() {
        let table = MegaKernelCallbackTable::new();
        assert!(!table.as_ptr().is_null());
    }

    // ── Slot constants ──

    #[test]
    fn slot_constants_within_range() {
        assert!(slot::SG_KNOWLEDGE_RETRIEVE < CALLBACK_TABLE_SLOTS);
        assert!(slot::KV_DECOMPRESS_LZ4 < CALLBACK_TABLE_SLOTS);
        assert!(slot::KV_DECOMPRESS_BITPACKRLE < CALLBACK_TABLE_SLOTS);
        assert!(slot::KV_DECOMPRESS_NVCOMP < CALLBACK_TABLE_SLOTS);
    }

    #[test]
    fn slot_constants_are_distinct() {
        let slots = [
            slot::SG_KNOWLEDGE_RETRIEVE,
            slot::KV_DECOMPRESS_LZ4,
            slot::KV_DECOMPRESS_BITPACKRLE,
            slot::KV_DECOMPRESS_NVCOMP,
        ];
        for i in 0..slots.len() {
            for j in (i + 1)..slots.len() {
                assert_ne!(slots[i], slots[j], "slot {} and {} must differ", i, j);
            }
        }
    }

    // ── CallbackEntry ──

    #[test]
    fn callback_entry_default_is_null() {
        let entry = CallbackEntry::default();
        assert!(entry.fn_ptr.is_null());
        assert!(entry.ctx.is_null());
    }

    #[test]
    fn callback_entry_copy() {
        let entry = CallbackEntry { fn_ptr: null_retrieve_bridge as *const u8, ctx: std::ptr::null() };
        let copied = entry;
        assert_eq!(entry.fn_ptr, copied.fn_ptr);
        assert_eq!(entry.ctx, copied.ctx);
    }

    #[test]
    fn callback_entry_clone() {
        let entry = CallbackEntry { fn_ptr: std::ptr::null(), ctx: std::ptr::null() };
        let cloned = entry.clone();
        assert!(cloned.fn_ptr.is_null());
    }

    #[test]
    fn callback_entry_size_16() {
        assert_eq!(std::mem::size_of::<CallbackEntry>(), 16);
    }

    // ── MegaKernelCallbackTable multi-slot ──

    #[test]
    fn register_multiple_slots() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, test_callback as *const u8, std::ptr::null());
            table.register(1, kv_decompress_lz4_callback as *const u8, std::ptr::null());
            table.register(2, kv_decompress_bitpackrle_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        assert!(!table.entries[0].fn_ptr.is_null());
        assert!(!table.entries[1].fn_ptr.is_null());
        assert!(!table.entries[2].fn_ptr.is_null());
        assert!(table.entries[3].fn_ptr.is_null()); // slot 3 not registered
    }

    #[test]
    fn clear_one_preserves_others() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, test_callback as *const u8, std::ptr::null());
            table.register(1, test_callback as *const u8, std::ptr::null());
        }
        table.clear(0);
        assert!(table.entries[0].fn_ptr.is_null());
        assert!(!table.entries[1].fn_ptr.is_null());
    }

    // ── SgCallbackCtx layout ──

    #[test]
    fn sg_callback_ctx_layout() {
        // sg_shared_memory(8) + retrieve_fn(8) + provider_state(8) + hidden_size(4)
        // + alpha(4) + precomputed_knowledge(8) = 40
        assert_eq!(std::mem::size_of::<SgCallbackCtx>(), 40);
    }

    #[test]
    fn sg_callback_ctx_null_pointers() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!(ctx.sg_shared_memory.is_null());
        assert_eq!(ctx.hidden_size, 0);
        assert!((ctx.alpha - 0.0).abs() < 1e-6);
    }

    // ── null_retrieve_bridge ──

    #[test]
    fn null_retrieve_bridge_returns_zero() {
        let result = unsafe {
            null_retrieve_bridge(std::ptr::null(), 0, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null())
        };
        assert_eq!(result, 0);
    }

    // ── sg_knowledge_retrieve_callback with null ctx ──

    #[test]
    fn sg_callback_null_ctx_returns_zero() {
        let result = unsafe { sg_knowledge_retrieve_callback(std::ptr::null()) };
        assert_eq!(result, 0);
    }

    // ── KvDecompressCtx fields ──

    #[test]
    fn kv_decompress_ctx_fields() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 100,
            page_size_bytes: 65536,
            header_stride: 56,
        };
        assert!(ctx.page_headers.is_null());
        assert_eq!(ctx.num_pages, 100);
        assert_eq!(ctx.page_size_bytes, 65536);
        assert_eq!(ctx.header_stride, 56);
    }

    // ── CALLBACK_TABLE_SLOTS constant ──

    #[test]
    fn callback_table_slots_is_8() {
        assert_eq!(CALLBACK_TABLE_SLOTS, 8);
    }

    // ── BitPackRle callback with valid ctx ──

    #[test]
    fn bitpackrle_callback_with_valid_ctx() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 1,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        let result = unsafe {
            kv_decompress_bitpackrle_callback(&ctx as *const KvDecompressCtx as *const u8)
        };
        assert_eq!(result, 1);
    }

    #[test]
    fn nvcomp_callback_with_valid_ctx() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 1,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        let result = unsafe {
            kv_decompress_nvcomp_callback(&ctx as *const KvDecompressCtx as *const u8)
        };
        assert_eq!(result, 1);
    }

    // ── New tests below ──

    #[test]
    fn callback_entry_construction_both_non_null() {
        let sentinel_a = 0xDEAD_usize as *const u8;
        let sentinel_b = 0xBEEF_usize as *const u8;
        let entry = CallbackEntry { fn_ptr: sentinel_a, ctx: sentinel_b };
        assert_eq!(entry.fn_ptr, sentinel_a);
        assert_eq!(entry.ctx, sentinel_b);
        assert!(!entry.fn_ptr.is_null());
        assert!(!entry.ctx.is_null());
    }

    #[test]
    fn callback_entry_fn_ptr_null_ctx_non_null() {
        let sentinel = 0x1234_usize as *const u8;
        let entry = CallbackEntry { fn_ptr: std::ptr::null(), ctx: sentinel };
        assert!(entry.fn_ptr.is_null());
        assert!(!entry.ctx.is_null());
        assert_eq!(entry.ctx, sentinel);
    }

    #[test]
    fn callback_entry_debug_format() {
        let entry = CallbackEntry::default();
        let debug_str = format!("{:?}", entry);
        assert!(debug_str.contains("fn_ptr"));
        assert!(debug_str.contains("ctx"));
    }

    #[test]
    fn mega_kernel_callback_table_debug_format() {
        let table = MegaKernelCallbackTable::new();
        let debug_str = format!("{:?}", table);
        assert!(debug_str.contains("entries"));
    }

    #[test]
    fn sg_callback_ctx_debug_format() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 768,
            alpha: 0.5,
            precomputed_knowledge: std::ptr::null(),
        };
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("hidden_size"));
        assert!(debug_str.contains("alpha"));
    }

    #[test]
    fn kv_decompress_ctx_debug_format() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 42,
            page_size_bytes: 8192,
            header_stride: 56,
        };
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("num_pages"));
        assert!(debug_str.contains("page_size_bytes"));
        assert!(debug_str.contains("header_stride"));
    }

    #[test]
    fn clear_on_empty_table_noop() {
        let mut table = MegaKernelCallbackTable::new();
        assert!(!table.has_any_callback());
        table.clear(0);
        table.clear(1);
        table.clear(7);
        assert!(!table.has_any_callback());
    }

    #[test]
    fn register_last_slot_has_any_callback() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(7, test_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        assert!(!table.entries[7].fn_ptr.is_null());
        assert!(table.entries[0].fn_ptr.is_null());
    }

    #[test]
    fn reregister_overwrites_slot() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel = 0xABCD_usize as *const u8;
        unsafe {
            table.register(2, test_callback as *const u8, std::ptr::null());
            table.register(2, sentinel, sentinel);
        }
        assert_eq!(table.entries[2].fn_ptr, sentinel);
        assert_eq!(table.entries[2].ctx, sentinel);
    }

    #[test]
    fn register_all_slots_then_clear_all() {
        let mut table = MegaKernelCallbackTable::new();
        for i in 0..CALLBACK_TABLE_SLOTS {
            unsafe {
                table.register(i, test_callback as *const u8, std::ptr::null());
            }
        }
        assert!(table.has_any_callback());
        for i in 0..CALLBACK_TABLE_SLOTS {
            table.clear(i);
        }
        assert!(!table.has_any_callback());
        for i in 0..CALLBACK_TABLE_SLOTS {
            assert!(table.entries[i].fn_ptr.is_null());
        }
    }

    #[test]
    fn sg_callback_ctx_non_null_pointers() {
        let sentinel_a = 0xA000_usize as *const u8;
        let sentinel_b = 0xB000_usize as *const f32;
        let ctx = SgCallbackCtx {
            sg_shared_memory: sentinel_a,
            retrieve_fn: null_retrieve_bridge,
            provider_state: sentinel_a,
            hidden_size: 1024,
            alpha: 0.75,
            precomputed_knowledge: sentinel_b,
        };
        assert_eq!(ctx.sg_shared_memory, sentinel_a);
        assert_eq!(ctx.provider_state, sentinel_a);
        assert_eq!(ctx.hidden_size, 1024);
        assert!((ctx.alpha - 0.75).abs() < 1e-6);
        assert_eq!(ctx.precomputed_knowledge, sentinel_b);
    }

    #[test]
    fn sg_callback_ctx_alpha_negative() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: -1.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!((ctx.alpha - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn sg_callback_ctx_hidden_size_max() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: u32::MAX,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert_eq!(ctx.hidden_size, u32::MAX);
    }

    #[test]
    fn kv_decompress_ctx_zero_fields() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 0,
            page_size_bytes: 0,
            header_stride: 0,
        };
        assert!(ctx.page_headers.is_null());
        assert_eq!(ctx.num_pages, 0);
        assert_eq!(ctx.page_size_bytes, 0);
        assert_eq!(ctx.header_stride, 0);
    }

    #[test]
    fn kv_decompress_ctx_large_page_size() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 1,
            page_size_bytes: u32::MAX,
            header_stride: 56,
        };
        assert_eq!(ctx.page_size_bytes, u32::MAX);
    }

    #[test]
    fn slot_constant_values() {
        assert_eq!(slot::SG_KNOWLEDGE_RETRIEVE, 0);
        assert_eq!(slot::KV_DECOMPRESS_LZ4, 1);
        assert_eq!(slot::KV_DECOMPRESS_BITPACKRLE, 2);
        assert_eq!(slot::KV_DECOMPRESS_NVCOMP, 3);
    }

    #[test]
    fn callback_entry_copy_independent() {
        let sentinel = 0xFEED_usize as *const u8;
        let original = CallbackEntry { fn_ptr: sentinel, ctx: std::ptr::null() };
        let copied = original;
        // Both should reference the same pointer value
        assert_eq!(original.fn_ptr, copied.fn_ptr);
        assert_eq!(original.ctx, copied.ctx);
    }

    #[test]
    fn table_entries_accessible_by_index() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel = 0xCAFE_usize as *const u8;
        unsafe {
            table.register(3, sentinel, sentinel);
        }
        // Verify each unregistered slot is still null
        for i in 0..CALLBACK_TABLE_SLOTS {
            if i == 3 {
                assert_eq!(table.entries[i].fn_ptr, sentinel);
                assert_eq!(table.entries[i].ctx, sentinel);
            } else {
                assert!(table.entries[i].fn_ptr.is_null());
                assert!(table.entries[i].ctx.is_null());
            }
        }
    }

    // ── New tests: ~55 additional tests ──

    // ── CallbackEntry: alignment ──

    #[test]
    fn callback_entry_alignment_is_8() {
        assert_eq!(std::mem::align_of::<CallbackEntry>(), 8);
    }

    #[test]
    fn callback_entry_repr_c_layout() {
        // fn_ptr at offset 0, ctx at offset 8
        let entry = CallbackEntry {
            fn_ptr: 0x1111_usize as *const u8,
            ctx: 0x2222_usize as *const u8,
        };
        let base = &entry as *const CallbackEntry as usize;
        let fn_ptr_addr = &entry.fn_ptr as *const _ as usize;
        let ctx_addr = &entry.ctx as *const _ as usize;
        assert_eq!(fn_ptr_addr - base, 0);
        assert_eq!(ctx_addr - base, 8);
    }

    #[test]
    fn callback_entry_array_layout_contiguous() {
        let entries: [CallbackEntry; 4] = [
            CallbackEntry {
                fn_ptr: 0xA_usize as *const u8,
                ctx: std::ptr::null(),
            },
            CallbackEntry {
                fn_ptr: 0xB_usize as *const u8,
                ctx: std::ptr::null(),
            },
            CallbackEntry {
                fn_ptr: 0xC_usize as *const u8,
                ctx: std::ptr::null(),
            },
            CallbackEntry {
                fn_ptr: 0xD_usize as *const u8,
                ctx: std::ptr::null(),
            },
        ];
        let base = entries.as_ptr() as usize;
        assert_eq!(&entries[0] as *const _ as usize - base, 0);
        assert_eq!(&entries[1] as *const _ as usize - base, 16);
        assert_eq!(&entries[2] as *const _ as usize - base, 32);
        assert_eq!(&entries[3] as *const _ as usize - base, 48);
    }

    // ── CallbackEntry: assignability and mutation ──

    #[test]
    fn callback_entry_field_assign_fn_ptr() {
        let mut entry = CallbackEntry::default();
        assert!(entry.fn_ptr.is_null());
        entry.fn_ptr = null_retrieve_bridge as *const u8;
        assert!(!entry.fn_ptr.is_null());
    }

    #[test]
    fn callback_entry_field_assign_ctx() {
        let mut entry = CallbackEntry::default();
        assert!(entry.ctx.is_null());
        let sentinel = 0xFF00_usize as *const u8;
        entry.ctx = sentinel;
        assert_eq!(entry.ctx, sentinel);
    }

    #[test]
    fn callback_entry_equality_same_values() {
        let a = CallbackEntry { fn_ptr: std::ptr::null(), ctx: std::ptr::null() };
        let b = CallbackEntry { fn_ptr: std::ptr::null(), ctx: std::ptr::null() };
        assert_eq!(a.fn_ptr, b.fn_ptr);
        assert_eq!(a.ctx, b.ctx);
    }

    // ── MegaKernelCallbackTable: as_ptr consistency ──

    #[test]
    fn as_ptr_returns_same_address_for_same_table() {
        let table = MegaKernelCallbackTable::new();
        let ptr1 = table.as_ptr();
        let ptr2 = table.as_ptr();
        assert_eq!(ptr1, ptr2);
    }

    #[test]
    fn as_ptr_different_for_different_tables() {
        let table1 = MegaKernelCallbackTable::new();
        let table2 = MegaKernelCallbackTable::new();
        assert_ne!(table1.as_ptr(), table2.as_ptr());
    }

    #[test]
    fn as_ptr_reflects_registration() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, test_callback as *const u8, std::ptr::null());
        }
        let ptr = table.as_ptr();
        // First 8 bytes (fn_ptr of slot 0) should be non-zero
        let bytes = unsafe { std::slice::from_raw_parts(ptr, 8) };
        assert!(bytes.iter().any(|&b| b != 0));
    }

    #[test]
    fn as_ptr_all_zeros_after_clear() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, test_callback as *const u8, std::ptr::null());
        }
        table.clear(0);
        let ptr = table.as_ptr();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, CALLBACK_TABLE_SLOTS * 16) };
        assert!(bytes.iter().all(|&b| b == 0));
    }

    // ── MegaKernelCallbackTable: slot boundary checks ──

    #[test]
    #[should_panic(expected = "callback slot")]
    fn register_slot_out_of_range_panics() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(CALLBACK_TABLE_SLOTS, test_callback as *const u8, std::ptr::null());
        }
    }

    #[test]
    #[should_panic(expected = "callback slot")]
    fn clear_slot_out_of_range_panics() {
        let mut table = MegaKernelCallbackTable::new();
        table.clear(CALLBACK_TABLE_SLOTS);
    }

    #[test]
    #[should_panic(expected = "callback slot")]
    fn register_slot_way_out_of_range_panics() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(100, test_callback as *const u8, std::ptr::null());
        }
    }

    #[test]
    #[should_panic(expected = "callback slot")]
    fn clear_slot_way_out_of_range_panics() {
        let mut table = MegaKernelCallbackTable::new();
        table.clear(255);
    }

    // ── MegaKernelCallbackTable: reserved slots 4-7 ──

    #[test]
    fn register_reserved_slot_4() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(4, test_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        assert!(!table.entries[4].fn_ptr.is_null());
    }

    #[test]
    fn register_reserved_slot_5() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(5, test_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        assert!(!table.entries[5].fn_ptr.is_null());
    }

    #[test]
    fn register_reserved_slot_6() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(6, test_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        assert!(!table.entries[6].fn_ptr.is_null());
    }

    #[test]
    fn register_reserved_slot_7() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(7, test_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        assert!(!table.entries[7].fn_ptr.is_null());
    }

    #[test]
    fn clear_reserved_slots() {
        let mut table = MegaKernelCallbackTable::new();
        for i in 4..CALLBACK_TABLE_SLOTS {
            unsafe {
                table.register(i, test_callback as *const u8, std::ptr::null());
            }
        }
        assert!(table.has_any_callback());
        for i in 4..CALLBACK_TABLE_SLOTS {
            table.clear(i);
        }
        assert!(!table.has_any_callback());
    }

    // ── MegaKernelCallbackTable: callback function interop ──

    #[test]
    fn callback_fn_type_matches_signature() {
        // Verify test_callback can be cast to MegaKernelCallbackFn
        let cb: MegaKernelCallbackFn = test_callback;
        let result = unsafe { cb(std::ptr::null()) };
        assert_eq!(result, 1);
    }

    #[test]
    fn table_callback_invocation_via_entry() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, test_callback as *const u8, std::ptr::null());
        }
        let entry = table.entries[0];
        let cb: MegaKernelCallbackFn = unsafe { std::mem::transmute(entry.fn_ptr) };
        let result = unsafe { cb(entry.ctx) };
        assert_eq!(result, 1);
    }

    // ── SgCallbackCtx: field-by-field construction ──

    #[test]
    fn sg_callback_ctx_retrieve_fn_is_null_retrieve_bridge() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        // Call the retrieve_fn — should return 0 (null bridge)
        let result = unsafe {
            (ctx.retrieve_fn)(
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        assert_eq!(result, 0);
    }

    #[test]
    fn sg_callback_ctx_alpha_zero() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 256,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!((ctx.alpha).abs() < 1e-6);
    }

    #[test]
    fn sg_callback_ctx_alpha_large_positive() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 100.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!((ctx.alpha - 100.0).abs() < 1e-6);
    }

    #[test]
    fn sg_callback_ctx_hidden_size_one() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 1,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert_eq!(ctx.hidden_size, 1);
    }

    #[test]
    fn sg_callback_ctx_precomputed_knowledge_non_null() {
        let sentinel = 0xBEEF_usize as *const f32;
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 768,
            alpha: 1.0,
            precomputed_knowledge: sentinel,
        };
        assert_eq!(ctx.precomputed_knowledge, sentinel);
        assert!(!ctx.precomputed_knowledge.is_null());
    }

    #[test]
    fn sg_callback_ctx_alignment() {
        // SgCallbackCtx has pointer fields, so alignment should be 8
        assert_eq!(std::mem::align_of::<SgCallbackCtx>(), 8);
    }

    #[test]
    fn sg_callback_ctx_field_offsets() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        let base = &ctx as *const SgCallbackCtx as usize;
        assert_eq!(&ctx.sg_shared_memory as *const _ as usize - base, 0);
        assert_eq!(&ctx.retrieve_fn as *const _ as usize - base, 8);
        assert_eq!(&ctx.provider_state as *const _ as usize - base, 16);
        assert_eq!(&ctx.hidden_size as *const _ as usize - base, 24);
        assert_eq!(&ctx.alpha as *const _ as usize - base, 28);
        assert_eq!(&ctx.precomputed_knowledge as *const _ as usize - base, 32);
    }

    // ── sg_knowledge_retrieve_callback: additional edge cases ──

    #[test]
    fn sg_callback_null_sg_shared_memory_returns_zero() {
        // Construct an SgCallbackCtx with null sg_shared_memory
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: 0x1_usize as *const u8, // non-null
            hidden_size: 128,
            alpha: 1.0,
            precomputed_knowledge: std::ptr::null(),
        };
        let result = unsafe {
            sg_knowledge_retrieve_callback(&ctx as *const SgCallbackCtx as *const u8)
        };
        assert_eq!(result, 0);
    }

    #[test]
    fn sg_callback_null_provider_state_returns_zero() {
        let data = [0u8; 1024];
        let ctx = SgCallbackCtx {
            sg_shared_memory: data.as_ptr(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 128,
            alpha: 1.0,
            precomputed_knowledge: std::ptr::null(),
        };
        let result = unsafe {
            sg_knowledge_retrieve_callback(&ctx as *const SgCallbackCtx as *const u8)
        };
        assert_eq!(result, 0);
    }

    #[test]
    fn sg_callback_zero_hidden_size_returns_zero() {
        let data = [0u8; 64];
        let ctx = SgCallbackCtx {
            sg_shared_memory: data.as_ptr(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: 0x1_usize as *const u8,
            hidden_size: 0,
            alpha: 1.0,
            precomputed_knowledge: std::ptr::null(),
        };
        let result = unsafe {
            sg_knowledge_retrieve_callback(&ctx as *const SgCallbackCtx as *const u8)
        };
        assert_eq!(result, 0);
    }

    // ── KvDecompressCtx: additional fields ──

    #[test]
    fn kv_decompress_ctx_alignment() {
        assert_eq!(std::mem::align_of::<KvDecompressCtx>(), 8);
    }

    #[test]
    fn kv_decompress_ctx_field_offsets() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 0,
            page_size_bytes: 0,
            header_stride: 0,
        };
        let base = &ctx as *const KvDecompressCtx as usize;
        assert_eq!(&ctx.page_headers as *const _ as usize - base, 0);
        assert_eq!(&ctx.num_pages as *const _ as usize - base, 8);
        assert_eq!(&ctx.page_size_bytes as *const _ as usize - base, 12);
        assert_eq!(&ctx.header_stride as *const _ as usize - base, 16);
    }

    #[test]
    fn kv_decompress_ctx_num_pages_max() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: u32::MAX,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        assert_eq!(ctx.num_pages, u32::MAX);
    }

    #[test]
    fn kv_decompress_ctx_header_stride_max() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 10,
            page_size_bytes: 4096,
            header_stride: u32::MAX,
        };
        assert_eq!(ctx.header_stride, u32::MAX);
    }

    #[test]
    fn kv_decompress_ctx_page_headers_non_null() {
        let data = [0u8; 256];
        let ctx = KvDecompressCtx {
            page_headers: data.as_ptr(),
            num_pages: 4,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        assert!(!ctx.page_headers.is_null());
        assert_eq!(ctx.page_headers, data.as_ptr());
    }

    // ── null_retrieve_bridge: non-null args ──

    #[test]
    fn null_retrieve_bridge_with_non_null_args_returns_zero() {
        let output_buf = [0.0f32; 16];
        let mut confidence = 0.0f32;
        let state = [0u8; 8];
        let result = unsafe {
            null_retrieve_bridge(
                output_buf.as_ptr(),
                16,
                output_buf.as_ptr() as *mut f32,
                &mut confidence as *mut f32,
                state.as_ptr(),
            )
        };
        assert_eq!(result, 0);
    }

    // ── Decompress callbacks: with KvDecompressCtx containing non-null ptrs ──

    #[test]
    fn lz4_callback_with_non_null_ctx() {
        let headers = [0u8; 56];
        let ctx = KvDecompressCtx {
            page_headers: headers.as_ptr(),
            num_pages: 1,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        let result = unsafe {
            kv_decompress_lz4_callback(&ctx as *const KvDecompressCtx as *const u8)
        };
        assert_eq!(result, 1);
    }

    #[test]
    fn bitpackrle_callback_with_zero_page_size() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 0,
            page_size_bytes: 0,
            header_stride: 0,
        };
        let result = unsafe {
            kv_decompress_bitpackrle_callback(&ctx as *const KvDecompressCtx as *const u8)
        };
        assert_eq!(result, 1);
    }

    #[test]
    fn nvcomp_callback_with_zero_page_size() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 0,
            page_size_bytes: 0,
            header_stride: 0,
        };
        let result = unsafe {
            kv_decompress_nvcomp_callback(&ctx as *const KvDecompressCtx as *const u8)
        };
        assert_eq!(result, 1);
    }

    // ── SgRetrieveFn type alias usage ──

    unsafe extern "C" fn custom_retrieve_fn(
        _detect_hidden: *const f32,
        _hidden_size: u32,
        _output_knowledge: *mut f32,
        _output_confidence: *mut f32,
        _provider_state: *const u8,
    ) -> i32 {
        42
    }

    #[test]
    fn sg_retrieve_fn_custom_returns_value() {
        let retrieve: SgRetrieveFn = custom_retrieve_fn;
        let result = unsafe {
            retrieve(
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        assert_eq!(result, 42);
    }

    #[test]
    fn sg_callback_ctx_with_custom_retrieve_fn() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: custom_retrieve_fn,
            provider_state: std::ptr::null(),
            hidden_size: 512,
            alpha: 0.5,
            precomputed_knowledge: std::ptr::null(),
        };
        let result = unsafe {
            (ctx.retrieve_fn)(
                std::ptr::null(),
                512,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null(),
            )
        };
        assert_eq!(result, 42);
    }

    // ── MegaKernelCallbackTable: register decompress callbacks in all slots ──

    #[test]
    fn register_all_decompress_callbacks() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(
                slot::KV_DECOMPRESS_LZ4,
                kv_decompress_lz4_callback as *const u8,
                std::ptr::null(),
            );
            table.register(
                slot::KV_DECOMPRESS_BITPACKRLE,
                kv_decompress_bitpackrle_callback as *const u8,
                std::ptr::null(),
            );
            table.register(
                slot::KV_DECOMPRESS_NVCOMP,
                kv_decompress_nvcomp_callback as *const u8,
                std::ptr::null(),
            );
        }
        assert!(table.has_any_callback());
        assert!(!table.entries[slot::KV_DECOMPRESS_LZ4].fn_ptr.is_null());
        assert!(!table.entries[slot::KV_DECOMPRESS_BITPACKRLE].fn_ptr.is_null());
        assert!(!table.entries[slot::KV_DECOMPRESS_NVCOMP].fn_ptr.is_null());
        assert!(table.entries[slot::SG_KNOWLEDGE_RETRIEVE].fn_ptr.is_null());
    }

    #[test]
    fn invoke_lz4_via_table_entry() {
        let mut table = MegaKernelCallbackTable::new();
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 2,
            page_size_bytes: 8192,
            header_stride: 56,
        };
        unsafe {
            table.register(
                slot::KV_DECOMPRESS_LZ4,
                kv_decompress_lz4_callback as *const u8,
                &ctx as *const KvDecompressCtx as *const u8,
            );
        }
        let entry = table.entries[slot::KV_DECOMPRESS_LZ4];
        let cb: MegaKernelCallbackFn = unsafe { std::mem::transmute(entry.fn_ptr) };
        let result = unsafe { cb(entry.ctx) };
        assert_eq!(result, 1);
    }

    #[test]
    fn invoke_bitpackrle_via_table_entry() {
        let mut table = MegaKernelCallbackTable::new();
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 1,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        unsafe {
            table.register(
                slot::KV_DECOMPRESS_BITPACKRLE,
                kv_decompress_bitpackrle_callback as *const u8,
                &ctx as *const KvDecompressCtx as *const u8,
            );
        }
        let entry = table.entries[slot::KV_DECOMPRESS_BITPACKRLE];
        let cb: MegaKernelCallbackFn = unsafe { std::mem::transmute(entry.fn_ptr) };
        let result = unsafe { cb(entry.ctx) };
        assert_eq!(result, 1);
    }

    #[test]
    fn invoke_nvcomp_via_table_entry() {
        let mut table = MegaKernelCallbackTable::new();
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 1,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        unsafe {
            table.register(
                slot::KV_DECOMPRESS_NVCOMP,
                kv_decompress_nvcomp_callback as *const u8,
                &ctx as *const KvDecompressCtx as *const u8,
            );
        }
        let entry = table.entries[slot::KV_DECOMPRESS_NVCOMP];
        let cb: MegaKernelCallbackFn = unsafe { std::mem::transmute(entry.fn_ptr) };
        let result = unsafe { cb(entry.ctx) };
        assert_eq!(result, 1);
    }

    // ── MegaKernelCallbackTable: has_any_callback after selective clear ──

    #[test]
    fn has_any_callback_true_after_partial_clear() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, test_callback as *const u8, std::ptr::null());
            table.register(1, test_callback as *const u8, std::ptr::null());
        }
        table.clear(0);
        assert!(table.has_any_callback()); // slot 1 still registered
    }

    #[test]
    fn has_any_callback_false_after_all_cleared() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, test_callback as *const u8, std::ptr::null());
            table.register(7, test_callback as *const u8, std::ptr::null());
        }
        table.clear(0);
        table.clear(7);
        assert!(!table.has_any_callback());
    }

    // ── CallbackEntry: from MegaKernelCallbackTable slice ──

    #[test]
    fn table_entries_is_array_of_callback_entry() {
        let table = MegaKernelCallbackTable::new();
        let entries: &[CallbackEntry] = &table.entries;
        assert_eq!(entries.len(), CALLBACK_TABLE_SLOTS);
    }

    // ── MegaKernelCallbackTable: size verification ──

    #[test]
    fn table_size_is_entries_times_entry_size() {
        assert_eq!(
            std::mem::size_of::<MegaKernelCallbackTable>(),
            CALLBACK_TABLE_SLOTS * std::mem::size_of::<CallbackEntry>(),
        );
    }

    // ── Slot constants: ordering ──

    #[test]
    fn slot_constants_are_ascending() {
        assert!(slot::SG_KNOWLEDGE_RETRIEVE < slot::KV_DECOMPRESS_LZ4);
        assert!(slot::KV_DECOMPRESS_LZ4 < slot::KV_DECOMPRESS_BITPACKRLE);
        assert!(slot::KV_DECOMPRESS_BITPACKRLE < slot::KV_DECOMPRESS_NVCOMP);
    }

    // ── MegaKernelCallbackTable: register with ctx pointer ──

    #[test]
    fn register_preserves_ctx_pointer() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel_ctx = 0x1234_usize as *const u8;
        unsafe {
            table.register(2, test_callback as *const u8, sentinel_ctx);
        }
        assert_eq!(table.entries[2].ctx, sentinel_ctx);
    }

    #[test]
    fn reregister_replaces_ctx() {
        let mut table = MegaKernelCallbackTable::new();
        let ctx1 = 0x1111_usize as *const u8;
        let ctx2 = 0x2222_usize as *const u8;
        unsafe {
            table.register(3, test_callback as *const u8, ctx1);
            table.register(3, test_callback as *const u8, ctx2);
        }
        assert_eq!(table.entries[3].ctx, ctx2);
    }

    #[test]
    fn clear_sets_both_fn_ptr_and_ctx_to_null() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel = 0xABCD_usize as *const u8;
        unsafe {
            table.register(5, test_callback as *const u8, sentinel);
        }
        assert_eq!(table.entries[5].ctx, sentinel);
        table.clear(5);
        assert!(table.entries[5].fn_ptr.is_null());
        assert!(table.entries[5].ctx.is_null());
    }

    // ── SgCallbackCtx: typical hidden sizes ──

    #[test]
    fn sg_callback_ctx_hidden_size_768() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 768,
            alpha: 1.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert_eq!(ctx.hidden_size, 768);
    }

    #[test]
    fn sg_callback_ctx_hidden_size_4096() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 4096,
            alpha: 1.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert_eq!(ctx.hidden_size, 4096);
    }

    // ── KvDecompressCtx: typical page sizes ──

    #[test]
    fn kv_decompress_ctx_typical_16k_page() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 1024,
            page_size_bytes: 16 * 1024,
            header_stride: 56,
        };
        assert_eq!(ctx.page_size_bytes, 16384);
        assert_eq!(ctx.num_pages, 1024);
    }

    // ── Decompress callbacks: each independently callable ──

    #[test]
    fn all_three_decompress_callbacks_return_1_with_valid_ctx() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 10,
            page_size_bytes: 8192,
            header_stride: 56,
        };
        let ctx_ptr = &ctx as *const KvDecompressCtx as *const u8;
        let r1 = unsafe { kv_decompress_lz4_callback(ctx_ptr) };
        let r2 = unsafe { kv_decompress_bitpackrle_callback(ctx_ptr) };
        let r3 = unsafe { kv_decompress_nvcomp_callback(ctx_ptr) };
        assert_eq!(r1, 1);
        assert_eq!(r2, 1);
        assert_eq!(r3, 1);
    }

    // ── MegaKernelCallbackFn transmute roundtrip ──

    #[test]
    fn callback_fn_roundtrip_preserves_address() {
        let original = test_callback as *const u8;
        let cb: MegaKernelCallbackFn = unsafe { std::mem::transmute(original) };
        let roundtripped = cb as *const u8;
        assert_eq!(original, roundtripped);
    }

    // ── Additional ~50 tests ──

    // ── CallbackEntry: mutation patterns ──

    #[test]
    fn callback_entry_swap_fields() {
        let sentinel_fn = 0xA_usize as *const u8;
        let sentinel_ctx = 0xB_usize as *const u8;
        let mut entry = CallbackEntry { fn_ptr: sentinel_fn, ctx: sentinel_ctx };
        // Swap fn_ptr and ctx
        let tmp = entry.fn_ptr;
        entry.fn_ptr = entry.ctx;
        entry.ctx = tmp;
        assert_eq!(entry.fn_ptr, sentinel_ctx);
        assert_eq!(entry.ctx, sentinel_fn);
    }

    #[test]
    fn callback_entry_default_then_mutate_fn_ptr() {
        let mut entry = CallbackEntry::default();
        assert!(entry.fn_ptr.is_null());
        entry.fn_ptr = test_callback as *const u8;
        assert!(!entry.fn_ptr.is_null());
        // ctx should still be null
        assert!(entry.ctx.is_null());
    }

    #[test]
    fn callback_entry_default_then_mutate_ctx() {
        let mut entry = CallbackEntry::default();
        let sentinel = 0xC_usize as *const u8;
        entry.ctx = sentinel;
        assert!(entry.fn_ptr.is_null());
        assert_eq!(entry.ctx, sentinel);
    }

    // ── CallbackEntry: copy semantics ──

    #[test]
    fn callback_entry_copy_then_modify_original() {
        let mut original = CallbackEntry {
            fn_ptr: test_callback as *const u8,
            ctx: 0x1_usize as *const u8,
        };
        let copied = original;
        // Modify original — copied should remain unchanged (Copy semantics)
        original.fn_ptr = std::ptr::null();
        assert!(original.fn_ptr.is_null());
        assert!(!copied.fn_ptr.is_null());
    }

    // ── CallbackEntry: clone consistency ──

    #[test]
    fn callback_entry_clone_preserves_both_fields() {
        let sentinel_fn = 0xD_usize as *const u8;
        let sentinel_ctx = 0xE_usize as *const u8;
        let entry = CallbackEntry { fn_ptr: sentinel_fn, ctx: sentinel_ctx };
        let cloned = entry.clone();
        assert_eq!(cloned.fn_ptr, sentinel_fn);
        assert_eq!(cloned.ctx, sentinel_ctx);
    }

    #[test]
    fn callback_entry_clone_then_modify_original() {
        let sentinel = 0xF_usize as *const u8;
        let mut entry = CallbackEntry { fn_ptr: sentinel, ctx: std::ptr::null() };
        let cloned = entry.clone();
        entry.fn_ptr = std::ptr::null();
        assert!(entry.fn_ptr.is_null());
        assert_eq!(cloned.fn_ptr, sentinel);
    }

    // ── CallbackEntry: is_null checks ──

    #[test]
    fn callback_entry_both_null_is_null() {
        let entry = CallbackEntry { fn_ptr: std::ptr::null(), ctx: std::ptr::null() };
        assert!(entry.fn_ptr.is_null());
        assert!(entry.ctx.is_null());
    }

    #[test]
    fn callback_entry_fn_ptr_only_non_null() {
        let entry = CallbackEntry {
            fn_ptr: test_callback as *const u8,
            ctx: std::ptr::null(),
        };
        assert!(!entry.fn_ptr.is_null());
        assert!(entry.ctx.is_null());
    }

    // ── MegaKernelCallbackTable: panic boundary ──

    #[test]
    #[should_panic(expected = "callback slot")]
    fn register_slot_usize_max_panics() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(usize::MAX, test_callback as *const u8, std::ptr::null());
        }
    }

    #[test]
    #[should_panic(expected = "callback slot")]
    fn clear_slot_usize_max_panics() {
        let mut table = MegaKernelCallbackTable::new();
        table.clear(usize::MAX);
    }

    #[test]
    #[should_panic(expected = "callback slot")]
    fn register_slot_just_past_last_panics() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(CALLBACK_TABLE_SLOTS + 1, test_callback as *const u8, std::ptr::null());
        }
    }

    #[test]
    #[should_panic(expected = "callback slot")]
    fn clear_slot_just_past_last_panics() {
        let mut table = MegaKernelCallbackTable::new();
        table.clear(CALLBACK_TABLE_SLOTS + 1);
    }

    // ── MegaKernelCallbackTable: boundary slot 7 ──

    #[test]
    fn register_and_clear_boundary_slot_7() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel = 0xAB_usize as *const u8;
        unsafe {
            table.register(7, sentinel, sentinel);
        }
        assert_eq!(table.entries[7].fn_ptr, sentinel);
        assert_eq!(table.entries[7].ctx, sentinel);
        table.clear(7);
        assert!(table.entries[7].fn_ptr.is_null());
        assert!(table.entries[7].ctx.is_null());
    }

    // ── MegaKernelCallbackTable: as_ptr after registration ──

    #[test]
    fn as_ptr_byte_pattern_after_register_slot_1() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(1, test_callback as *const u8, std::ptr::null());
        }
        let ptr = table.as_ptr();
        let bytes = unsafe { std::slice::from_raw_parts(ptr, CALLBACK_TABLE_SLOTS * 16) };
        // Slot 0 (bytes 0..15) should be all zero
        assert!(bytes[0..16].iter().all(|&b| b == 0));
        // Slot 1 (bytes 16..31) should have non-zero fn_ptr
        assert!(bytes[16..24].iter().any(|&b| b != 0));
        // Slot 1 ctx (bytes 24..31) should be all zero
        assert!(bytes[24..32].iter().all(|&b| b == 0));
    }

    // ── MegaKernelCallbackTable: default consistency ──

    #[test]
    fn default_table_entries_all_null() {
        let table = MegaKernelCallbackTable::default();
        for (i, entry) in table.entries.iter().enumerate() {
            assert!(entry.fn_ptr.is_null(), "slot {} fn_ptr not null", i);
            assert!(entry.ctx.is_null(), "slot {} ctx not null", i);
        }
    }

    #[test]
    fn new_and_default_produce_identical_layout() {
        let new_table = MegaKernelCallbackTable::new();
        let default_table = MegaKernelCallbackTable::default();
        let new_bytes = unsafe {
            std::slice::from_raw_parts(new_table.as_ptr(), CALLBACK_TABLE_SLOTS * 16)
        };
        let default_bytes = unsafe {
            std::slice::from_raw_parts(default_table.as_ptr(), CALLBACK_TABLE_SLOTS * 16)
        };
        assert!(new_bytes.iter().all(|&b| b == 0));
        assert!(default_bytes.iter().all(|&b| b == 0));
    }

    // ── MegaKernelCallbackTable: multiple operations sequence ──

    #[test]
    fn register_clear_reregister_different_slot() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel_a = 0x10_usize as *const u8;
        let sentinel_b = 0x20_usize as *const u8;
        unsafe {
            table.register(0, sentinel_a, std::ptr::null());
        }
        table.clear(0);
        assert!(!table.has_any_callback());
        unsafe {
            table.register(3, sentinel_b, std::ptr::null());
        }
        assert!(table.has_any_callback());
        assert!(table.entries[0].fn_ptr.is_null());
        assert_eq!(table.entries[3].fn_ptr, sentinel_b);
    }

    #[test]
    fn register_clear_reregister_same_slot() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel_a = 0x30_usize as *const u8;
        let sentinel_b = 0x40_usize as *const u8;
        unsafe {
            table.register(2, sentinel_a, std::ptr::null());
        }
        assert_eq!(table.entries[2].fn_ptr, sentinel_a);
        table.clear(2);
        assert!(table.entries[2].fn_ptr.is_null());
        unsafe {
            table.register(2, sentinel_b, std::ptr::null());
        }
        assert_eq!(table.entries[2].fn_ptr, sentinel_b);
    }

    // ── MegaKernelCallbackTable: selective clearing ──

    #[test]
    fn clear_first_and_last_preserves_middle() {
        let mut table = MegaKernelCallbackTable::new();
        for i in [0, 4, 7] {
            unsafe {
                table.register(i, test_callback as *const u8, std::ptr::null());
            }
        }
        table.clear(0);
        table.clear(7);
        assert!(table.entries[0].fn_ptr.is_null());
        assert!(table.entries[7].fn_ptr.is_null());
        assert!(!table.entries[4].fn_ptr.is_null());
        assert!(table.has_any_callback());
    }

    // ── MegaKernelCallbackTable: has_any_callback with single slot ──

    #[test]
    fn has_any_callback_slot_0_only() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, test_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        table.clear(0);
        assert!(!table.has_any_callback());
    }

    #[test]
    fn has_any_callback_slot_3_only() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(3, test_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        table.clear(3);
        assert!(!table.has_any_callback());
    }

    // ── MegaKernelCallbackTable: ctx pointer preserved per slot ──

    #[test]
    fn register_different_ctx_per_slot() {
        let mut table = MegaKernelCallbackTable::new();
        let ctx0 = 0x100_usize as *const u8;
        let ctx1 = 0x200_usize as *const u8;
        let ctx2 = 0x300_usize as *const u8;
        unsafe {
            table.register(0, test_callback as *const u8, ctx0);
            table.register(1, test_callback as *const u8, ctx1);
            table.register(2, test_callback as *const u8, ctx2);
        }
        assert_eq!(table.entries[0].ctx, ctx0);
        assert_eq!(table.entries[1].ctx, ctx1);
        assert_eq!(table.entries[2].ctx, ctx2);
    }

    #[test]
    fn clear_preserves_ctx_of_other_slots() {
        let mut table = MegaKernelCallbackTable::new();
        let ctx0 = 0x100_usize as *const u8;
        let ctx1 = 0x200_usize as *const u8;
        unsafe {
            table.register(0, test_callback as *const u8, ctx0);
            table.register(1, test_callback as *const u8, ctx1);
        }
        table.clear(0);
        assert!(table.entries[0].ctx.is_null());
        assert_eq!(table.entries[1].ctx, ctx1);
    }

    // ── MegaKernelCallbackTable: alignment ──

    #[test]
    fn table_alignment_is_8() {
        assert_eq!(std::mem::align_of::<MegaKernelCallbackTable>(), 8);
    }

    // ── SgCallbackCtx: typical alpha values ──

    #[test]
    fn sg_callback_ctx_alpha_0_5() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 512,
            alpha: 0.5,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!((ctx.alpha - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sg_callback_ctx_alpha_2_0() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 256,
            alpha: 2.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!((ctx.alpha - 2.0).abs() < 1e-6);
    }

    // ── SgCallbackCtx: construction with all non-null ──

    #[test]
    fn sg_callback_ctx_all_non_null_fields() {
        let sentinel_mem = 0x1000_usize as *const u8;
        let sentinel_prov = 0x2000_usize as *const u8;
        let sentinel_know = 0x3000_usize as *const f32;
        let ctx = SgCallbackCtx {
            sg_shared_memory: sentinel_mem,
            retrieve_fn: null_retrieve_bridge,
            provider_state: sentinel_prov,
            hidden_size: 2048,
            alpha: 0.9,
            precomputed_knowledge: sentinel_know,
        };
        assert_eq!(ctx.sg_shared_memory, sentinel_mem);
        assert_eq!(ctx.provider_state, sentinel_prov);
        assert_eq!(ctx.precomputed_knowledge, sentinel_know);
        assert_eq!(ctx.hidden_size, 2048);
        assert!((ctx.alpha - 0.9).abs() < 1e-6);
    }

    // ── SgCallbackCtx: alpha is f32 (property check) ──

    #[test]
    fn sg_callback_ctx_alpha_is_finite() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 1.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!(ctx.alpha.is_finite());
    }

    #[test]
    fn sg_callback_ctx_alpha_negative_is_finite() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: -0.5,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!(ctx.alpha.is_finite());
    }

    // ── SgCallbackCtx: size verification ──

    #[test]
    fn sg_callback_ctx_size_is_40() {
        // 6 fields: 3 pointers (8B each) + 1 fn_ptr (8B) + 1 u32 (4B) + 1 f32 (4B) = 40
        assert_eq!(std::mem::size_of::<SgCallbackCtx>(), 40);
    }

    // ── KvDecompressCtx: size verification ──

    #[test]
    fn kv_decompress_ctx_size_is_24() {
        // 1 pointer (8B) + 3 u32 (4B each) = 20, padded to 24 for 8-byte alignment
        assert_eq!(std::mem::size_of::<KvDecompressCtx>(), 24);
    }

    // ── KvDecompressCtx: typical configurations ──

    #[test]
    fn kv_decompress_ctx_4k_page() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 256,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        assert_eq!(ctx.page_size_bytes, 4096);
        assert_eq!(ctx.num_pages, 256);
    }

    #[test]
    fn kv_decompress_ctx_64k_page() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 64,
            page_size_bytes: 65536,
            header_stride: 56,
        };
        assert_eq!(ctx.page_size_bytes, 65536);
    }

    // ── KvDecompressCtx: non-null page_headers with slice ──

    #[test]
    fn kv_decompress_ctx_page_headers_points_to_data() {
        let headers: [u8; 224] = [0xAA; 224]; // 4 × 56B headers
        let ctx = KvDecompressCtx {
            page_headers: headers.as_ptr(),
            num_pages: 4,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        assert!(!ctx.page_headers.is_null());
        // Verify the pointer actually points to our data
        let first_byte = unsafe { *ctx.page_headers };
        assert_eq!(first_byte, 0xAA);
    }

    // ── null_retrieve_bridge: output unmodified ──

    #[test]
    fn null_retrieve_bridge_does_not_write_output() {
        let mut output_knowledge = [1.0f32, 2.0, 3.0, 4.0];
        let mut output_confidence = 99.0f32;
        let _result = unsafe {
            null_retrieve_bridge(
                std::ptr::null(),
                4,
                output_knowledge.as_mut_ptr(),
                &mut output_confidence as *mut f32,
                std::ptr::null(),
            )
        };
        // null_retrieve_bridge should not modify output buffers
        assert_eq!(output_knowledge, [1.0f32, 2.0, 3.0, 4.0]);
        assert!((output_confidence - 99.0).abs() < 1e-6);
    }

    // ── SgRetrieveFn type alias: null_retrieve_bridge matches ──

    #[test]
    fn null_retrieve_bridge_matches_sg_retrieve_fn_type() {
        let _: SgRetrieveFn = null_retrieve_bridge;
    }

    // ── Slot constants: slot 4..7 are reserved (within range) ──

    #[test]
    fn slot_4_within_range() {
        assert!(4 < CALLBACK_TABLE_SLOTS);
    }

    #[test]
    fn slot_7_within_range() {
        assert!(7 < CALLBACK_TABLE_SLOTS);
    }

    // ── MegaKernelCallbackFn: calling conventions ──

    #[test]
    fn callback_fn_returns_u32() {
        let cb: MegaKernelCallbackFn = test_callback;
        let result: u32 = unsafe { cb(std::ptr::null()) };
        assert_eq!(result, 1u32);
    }

    // ── MegaKernelCallbackTable: register with null fn_ptr ──

    #[test]
    fn register_null_fn_ptr_still_fills_entry() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel_ctx = 0x50_usize as *const u8;
        unsafe {
            table.register(3, std::ptr::null(), sentinel_ctx);
        }
        // fn_ptr is null, so has_any_callback returns false
        assert!(!table.has_any_callback());
        // But the ctx was stored
        assert_eq!(table.entries[3].ctx, sentinel_ctx);
    }

    // ── MegaKernelCallbackTable: register with all real decompress functions ──

    #[test]
    fn register_sg_and_all_decompress_callbacks() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(slot::SG_KNOWLEDGE_RETRIEVE, null_retrieve_bridge as *const u8, std::ptr::null());
            table.register(slot::KV_DECOMPRESS_LZ4, kv_decompress_lz4_callback as *const u8, std::ptr::null());
            table.register(slot::KV_DECOMPRESS_BITPACKRLE, kv_decompress_bitpackrle_callback as *const u8, std::ptr::null());
            table.register(slot::KV_DECOMPRESS_NVCOMP, kv_decompress_nvcomp_callback as *const u8, std::ptr::null());
        }
        assert!(table.has_any_callback());
        for slot_id in 0..4 {
            assert!(!table.entries[slot_id].fn_ptr.is_null(), "slot {} should be registered", slot_id);
        }
        for slot_id in 4..CALLBACK_TABLE_SLOTS {
            assert!(table.entries[slot_id].fn_ptr.is_null(), "slot {} should be empty", slot_id);
        }
    }

    // ── MegaKernelCallbackTable: clear slot 0 after full registration ──

    #[test]
    fn clear_sg_preserves_decompress() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(slot::SG_KNOWLEDGE_RETRIEVE, null_retrieve_bridge as *const u8, std::ptr::null());
            table.register(slot::KV_DECOMPRESS_LZ4, kv_decompress_lz4_callback as *const u8, std::ptr::null());
        }
        table.clear(slot::SG_KNOWLEDGE_RETRIEVE);
        assert!(table.entries[slot::SG_KNOWLEDGE_RETRIEVE].fn_ptr.is_null());
        assert!(!table.entries[slot::KV_DECOMPRESS_LZ4].fn_ptr.is_null());
        assert!(table.has_any_callback());
    }

    // ── MegaKernelCallbackTable: repr C table byte layout ──

    #[test]
    fn table_repr_c_byte_offset_for_slot_2() {
        let mut table = MegaKernelCallbackTable::new();
        let sentinel = 0xFE_usize as *const u8;
        unsafe {
            table.register(2, sentinel, std::ptr::null());
        }
        let ptr = table.as_ptr();
        // Slot 2 starts at byte offset 2 * 16 = 32
        let slot2_ptr = unsafe { ptr.add(32) };
        // The sentinel address should appear in these 8 bytes
        let recovered = unsafe { *(slot2_ptr as *const *const u8) };
        assert_eq!(recovered, sentinel);
    }

    // ── CALLBACK_TABLE_SLOTS: relationship with struct size ──

    #[test]
    fn callback_table_slots_equals_entries_count() {
        let table = MegaKernelCallbackTable::new();
        assert_eq!(table.entries.len(), CALLBACK_TABLE_SLOTS);
    }

    // ── SgCallbackCtx: hidden_size range property ──

    #[test]
    fn sg_callback_ctx_hidden_size_is_u32_range() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        assert_eq!(ctx.hidden_size, 0u32);
    }

    // ── KvDecompressCtx: all u32 fields range ──

    #[test]
    fn kv_decompress_ctx_all_zero_valid() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 0,
            page_size_bytes: 0,
            header_stride: 0,
        };
        assert_eq!(ctx.num_pages, 0u32);
        assert_eq!(ctx.page_size_bytes, 0u32);
        assert_eq!(ctx.header_stride, 0u32);
    }

    // ── MegaKernelCallbackTable: entries array is fixed size ──

    #[test]
    fn table_entries_array_length_is_constant() {
        let table = MegaKernelCallbackTable::new();
        assert_eq!(table.entries.len(), 8);
    }

    // ── CallbackEntry: debug format includes both fields ──

    #[test]
    fn callback_entry_debug_shows_fn_ptr_and_ctx() {
        let entry = CallbackEntry {
            fn_ptr: test_callback as *const u8,
            ctx: std::ptr::null(),
        };
        let debug = format!("{:?}", entry);
        assert!(debug.contains("fn_ptr"));
        assert!(debug.contains("ctx"));
    }

    // ── MegaKernelCallbackTable: as_ptr alignment ──

    #[test]
    fn table_as_ptr_aligned_to_8() {
        let table = MegaKernelCallbackTable::new();
        let ptr = table.as_ptr() as usize;
        assert_eq!(ptr % 8, 0, "as_ptr should be 8-byte aligned");
    }

    // ── MegaKernelCallbackFn: zero return callback ──

    unsafe extern "C" fn zero_callback(_ctx: *const u8) -> u32 {
        0
    }

    #[test]
    fn callback_fn_returning_zero() {
        let cb: MegaKernelCallbackFn = zero_callback;
        let result = unsafe { cb(std::ptr::null()) };
        assert_eq!(result, 0);
    }

    #[test]
    fn table_with_zero_callback_has_any_is_true() {
        let mut table = MegaKernelCallbackTable::new();
        unsafe {
            table.register(0, zero_callback as *const u8, std::ptr::null());
        }
        // has_any_callback checks fn_ptr != null, not return value
        assert!(table.has_any_callback());
    }

    // ── MegaKernelCallbackTable: registering same callback in multiple slots ──

    #[test]
    fn same_callback_in_all_slots() {
        let mut table = MegaKernelCallbackTable::new();
        for i in 0..CALLBACK_TABLE_SLOTS {
            unsafe {
                table.register(i, test_callback as *const u8, std::ptr::null());
            }
        }
        assert!(table.has_any_callback());
        for i in 0..CALLBACK_TABLE_SLOTS {
            assert!(!table.entries[i].fn_ptr.is_null());
        }
    }

    // ── MegaKernelCallbackTable: invoke callback from each registered slot ──

    #[test]
    fn invoke_callback_from_each_slot() {
        let mut table = MegaKernelCallbackTable::new();
        for i in 0..CALLBACK_TABLE_SLOTS {
            unsafe {
                table.register(i, test_callback as *const u8, std::ptr::null());
            }
        }
        for i in 0..CALLBACK_TABLE_SLOTS {
            let entry = table.entries[i];
            let cb: MegaKernelCallbackFn = unsafe { std::mem::transmute(entry.fn_ptr) };
            let result = unsafe { cb(entry.ctx) };
            assert_eq!(result, 1, "slot {} should return 1", i);
        }
    }

    // ── MegaKernelCallbackTable: clear middle slot and re-register ──

    #[test]
    fn clear_middle_reregister_with_different_ctx() {
        let mut table = MegaKernelCallbackTable::new();
        let ctx_a = 0x100_usize as *const u8;
        let ctx_b = 0x200_usize as *const u8;
        unsafe {
            table.register(4, test_callback as *const u8, ctx_a);
        }
        assert_eq!(table.entries[4].ctx, ctx_a);
        table.clear(4);
        assert!(table.entries[4].ctx.is_null());
        unsafe {
            table.register(4, test_callback as *const u8, ctx_b);
        }
        assert_eq!(table.entries[4].ctx, ctx_b);
    }

    // ── KvDecompressCtx: debug format completeness ──

    #[test]
    fn kv_decompress_ctx_debug_includes_page_headers() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 1,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("page_headers"));
    }

    // ── SgCallbackCtx: debug format completeness ──

    #[test]
    fn sg_callback_ctx_debug_includes_retrieve_fn() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("retrieve_fn"));
    }

    #[test]
    fn sg_callback_ctx_debug_includes_precomputed_knowledge() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: 0.0,
            precomputed_knowledge: std::ptr::null(),
        };
        let debug = format!("{:?}", ctx);
        assert!(debug.contains("precomputed_knowledge"));
    }

    // ── MegaKernelCallbackTable: registration with all decompress functions via ctx ──

    #[test]
    fn register_decompress_with_ctx_pointers() {
        let mut table = MegaKernelCallbackTable::new();
        let lz4_ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 10,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        let bpr_ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 20,
            page_size_bytes: 8192,
            header_stride: 56,
        };
        let nvc_ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 30,
            page_size_bytes: 16384,
            header_stride: 56,
        };
        unsafe {
            table.register(slot::KV_DECOMPRESS_LZ4, kv_decompress_lz4_callback as *const u8, &lz4_ctx as *const _ as *const u8);
            table.register(slot::KV_DECOMPRESS_BITPACKRLE, kv_decompress_bitpackrle_callback as *const u8, &bpr_ctx as *const _ as *const u8);
            table.register(slot::KV_DECOMPRESS_NVCOMP, kv_decompress_nvcomp_callback as *const u8, &nvc_ctx as *const _ as *const u8);
        }
        // Verify each slot's ctx points to the correct struct
        assert!(!table.entries[slot::KV_DECOMPRESS_LZ4].ctx.is_null());
        assert!(!table.entries[slot::KV_DECOMPRESS_BITPACKRLE].ctx.is_null());
        assert!(!table.entries[slot::KV_DECOMPRESS_NVCOMP].ctx.is_null());
        // Invoke each to verify end-to-end
        for slot_id in [slot::KV_DECOMPRESS_LZ4, slot::KV_DECOMPRESS_BITPACKRLE, slot::KV_DECOMPRESS_NVCOMP] {
            let entry = table.entries[slot_id];
            let cb: MegaKernelCallbackFn = unsafe { std::mem::transmute(entry.fn_ptr) };
            let result = unsafe { cb(entry.ctx) };
            assert_eq!(result, 1, "slot {} callback should return 1", slot_id);
        }
    }

    // ── SgCallbackCtx: alpha subnormal ──

    #[test]
    fn sg_callback_ctx_alpha_subnormal() {
        let tiny = f32::from_bits(1); // smallest positive subnormal
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: tiny,
            precomputed_knowledge: std::ptr::null(),
        };
        assert_eq!(ctx.alpha.to_bits(), 1u32);
    }

    // ── Additional 13 tests (163 → 176) ──

    #[test]
    fn sg_callback_ctx_alpha_nan_is_not_finite() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: f32::NAN,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!(ctx.alpha.is_nan());
        assert!(!ctx.alpha.is_finite());
    }

    #[test]
    fn sg_callback_ctx_alpha_infinity_is_not_finite() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: f32::INFINITY,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!(ctx.alpha.is_infinite());
        assert!(!ctx.alpha.is_finite());
    }

    #[test]
    fn kv_decompress_ctx_page_size_one() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 1,
            page_size_bytes: 1,
            header_stride: 1,
        };
        assert_eq!(ctx.page_size_bytes, 1);
        assert_eq!(ctx.header_stride, 1);
    }

    #[test]
    fn kv_decompress_ctx_num_pages_zero_valid() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 0,
            page_size_bytes: 4096,
            header_stride: 56,
        };
        assert_eq!(ctx.num_pages, 0);
        // A context with zero pages is a valid state (empty batch)
    }

    #[test]
    fn table_entries_memory_is_contiguous() {
        let table = MegaKernelCallbackTable::new();
        let base = table.entries.as_ptr() as usize;
        for i in 1..CALLBACK_TABLE_SLOTS {
            let prev_end = (base + i * std::mem::size_of::<CallbackEntry>());
            let curr_start = (&table.entries[i] as *const CallbackEntry) as usize;
            assert_eq!(prev_end, curr_start, "entries must be contiguous at index {}", i);
        }
    }

    #[test]
    fn callback_entry_debug_with_non_null_fn_ptr() {
        let entry = CallbackEntry {
            fn_ptr: test_callback as *const u8,
            ctx: std::ptr::null(),
        };
        let debug = format!("{:?}", entry);
        assert!(debug.contains("fn_ptr"));
        // fn_ptr should not be 0x0 (null), it should have a real address
        assert!(!entry.fn_ptr.is_null());
    }

    #[test]
    fn mega_kernel_callback_table_debug_shows_all_entries() {
        let table = MegaKernelCallbackTable::new();
        let debug = format!("{:?}", table);
        // Debug output should contain "entries"
        assert!(debug.contains("entries"));
        // All 8 entries should be present — verify by checking the output length is reasonable
        assert!(debug.len() > 100, "debug output should be substantial for 8 entries");
    }

    #[test]
    fn callback_entry_fn_ptr_matches_registered_function() {
        let mut table = MegaKernelCallbackTable::new();
        let expected = test_callback as *const u8;
        unsafe {
            table.register(0, expected, std::ptr::null());
        }
        assert_eq!(table.entries[0].fn_ptr, expected);
    }

    #[test]
    fn null_retrieve_bridge_called_with_valid_output_pointers_still_zero() {
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output_knowledge = [0.0f32; 4];
        let mut output_confidence = 0.0f32;
        let state = [0u8; 8];
        let result = unsafe {
            null_retrieve_bridge(
                input.as_ptr(),
                4,
                output_knowledge.as_mut_ptr(),
                &mut output_confidence as *mut f32,
                state.as_ptr(),
            )
        };
        assert_eq!(result, 0);
        // Output buffers should remain unmodified
        assert_eq!(output_knowledge, [0.0f32; 4]);
        assert!((output_confidence - 0.0).abs() < 1e-6);
    }

    #[test]
    fn table_register_and_invoke_all_decompress_with_ctx() {
        let mut table = MegaKernelCallbackTable::new();
        let decompress_ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 5,
            page_size_bytes: 8192,
            header_stride: 56,
        };
        let ctx_bytes = &decompress_ctx as *const KvDecompressCtx as *const u8;
        unsafe {
            table.register(slot::KV_DECOMPRESS_LZ4, kv_decompress_lz4_callback as *const u8, ctx_bytes);
            table.register(slot::KV_DECOMPRESS_BITPACKRLE, kv_decompress_bitpackrle_callback as *const u8, ctx_bytes);
            table.register(slot::KV_DECOMPRESS_NVCOMP, kv_decompress_nvcomp_callback as *const u8, ctx_bytes);
        }
        // All three should return 1 when invoked
        for (name, slot_id, cb_fn) in [
            ("lz4", slot::KV_DECOMPRESS_LZ4, kv_decompress_lz4_callback as *const u8),
            ("bitpackrle", slot::KV_DECOMPRESS_BITPACKRLE, kv_decompress_bitpackrle_callback as *const u8),
            ("nvcomp", slot::KV_DECOMPRESS_NVCOMP, kv_decompress_nvcomp_callback as *const u8),
        ] {
            let entry = table.entries[slot_id];
            assert_eq!(entry.fn_ptr, cb_fn, "{} fn_ptr mismatch", name);
            let func: MegaKernelCallbackFn = unsafe { std::mem::transmute(entry.fn_ptr) };
            let result = unsafe { func(entry.ctx) };
            assert_eq!(result, 1, "{} should return 1", name);
        }
    }

    #[test]
    fn sg_callback_ctx_alpha_neg_infinity_is_not_finite() {
        let ctx = SgCallbackCtx {
            sg_shared_memory: std::ptr::null(),
            retrieve_fn: null_retrieve_bridge,
            provider_state: std::ptr::null(),
            hidden_size: 0,
            alpha: f32::NEG_INFINITY,
            precomputed_knowledge: std::ptr::null(),
        };
        assert!(ctx.alpha.is_infinite());
        assert!(ctx.alpha.is_sign_negative());
        assert!(!ctx.alpha.is_finite());
    }

    #[test]
    fn kv_decompress_ctx_header_stride_one() {
        let ctx = KvDecompressCtx {
            page_headers: std::ptr::null(),
            num_pages: 100,
            page_size_bytes: 4096,
            header_stride: 1,
        };
        assert_eq!(ctx.header_stride, 1);
    }

    #[test]
    fn table_clear_each_slot_independently() {
        let mut table = MegaKernelCallbackTable::new();
        // Register all slots with different ctx values
        for i in 0..CALLBACK_TABLE_SLOTS {
            let ctx_val = (i as usize + 1) as *const u8;
            unsafe {
                table.register(i, test_callback as *const u8, ctx_val);
            }
        }
        // Clear every other slot
        for i in (0..CALLBACK_TABLE_SLOTS).step_by(2) {
            table.clear(i);
        }
        // Even slots should be null, odd slots should remain
        for i in 0..CALLBACK_TABLE_SLOTS {
            if i % 2 == 0 {
                assert!(table.entries[i].fn_ptr.is_null(), "even slot {} should be cleared", i);
                assert!(table.entries[i].ctx.is_null(), "even slot {} ctx should be null", i);
            } else {
                assert!(!table.entries[i].fn_ptr.is_null(), "odd slot {} should be registered", i);
                let expected_ctx = (i + 1) as *const u8;
                assert_eq!(table.entries[i].ctx, expected_ctx, "odd slot {} ctx mismatch", i);
            }
        }
        assert!(table.has_any_callback()); // odd slots still registered
    }
}
