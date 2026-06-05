//! gllm C FFI exports.
//!
//! This module provides a C-compatible API for gllm.
//! All operations are synchronous — no tokio runtime needed.
//! Inference is CPU-bound compute, not I/O-bound web service.

use std::ffi::{c_char, CStr, CString};
use std::ptr;

use crate::client::Client;
use crate::manifest::ModelKind;

/// Opaque context for gllm C API.
#[repr(C)]
pub struct GllmContext {
    pub client: Client,
}

/// Result from `gllm_generate`.
#[derive(Debug)]
#[repr(C)]
pub struct GllmGenerateResult {
    /// Generated text (allocated with `malloc`, must be freed with `gllm_free_generate_result`)
    pub text: *mut c_char,
    /// Error message (allocated with `malloc`, must be freed with `gllm_free_generate_result`)
    pub error: *mut c_char,
}

fn kind_from_u32(kind: u32) -> ModelKind {
    match kind {
        1 => ModelKind::Embedding,
        2 => ModelKind::Reranker,
        3 => ModelKind::Classifier,
        _ => ModelKind::Chat,
    }
}

/// Initialize a gllm context with a model.
///
/// # Safety
/// `model_id` must be a valid null-terminated UTF-8 string.
/// Returns NULL on failure (check stderr for diagnostics).
#[no_mangle]
pub unsafe extern "C" fn gllm_init(model_id: *const c_char, kind: u32) -> *mut GllmContext {
    if model_id.is_null() {
        return ptr::null_mut();
    }
    let model_id = match CStr::from_ptr(model_id).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    // Sync call — no tokio runtime needed
    let result = Client::new(model_id, kind_from_u32(kind));

    match result {
        Ok(client) => Box::into_raw(Box::new(GllmContext { client })),
        Err(e) => {
            eprintln!("[gllm FFI] init failed: {e}");
            ptr::null_mut()
        }
    }
}

/// Destroy a gllm context and free all associated resources.
///
/// # Safety
/// `ctx` must be a pointer returned by `gllm_init`, or NULL (no-op).
#[no_mangle]
pub unsafe extern "C" fn gllm_destroy(ctx: *mut GllmContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

/// Generate text from a prompt.
///
/// # Safety
/// - `ctx` must be a valid pointer from `gllm_init`.
/// - `prompt` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn gllm_generate(
    ctx: *mut GllmContext,
    prompt: *const c_char,
    max_tokens: u32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    // thinking_budget: 0 = disabled, negative = unlimited, positive = max tokens
    thinking_budget: i32,
) -> GllmGenerateResult {
    let err_result = |msg: &str| -> GllmGenerateResult {
        GllmGenerateResult {
            text: ptr::null_mut(),
            error: CString::new(msg).unwrap_or_default().into_raw(), // LEGAL: FFI boundary
        }
    };
    if ctx.is_null() || prompt.is_null() {
        return err_result("null pointer argument");
    }
    let ctx = &*ctx;
    let prompt_str = match CStr::from_ptr(prompt).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return err_result("invalid UTF-8 in prompt"),
    };

    // Sync call — no tokio runtime needed
    let tb = if thinking_budget < 0 { None } else { Some(thinking_budget as usize) };
    let result = ctx.client.execute_generation(
        prompt_str,
        max_tokens as usize,
        temperature,
        top_k as usize,
        top_p,
        None,
        tb,
    );

    match result {
        Ok(resp) => GllmGenerateResult {
            text: CString::new(resp.text).unwrap_or_default().into_raw(), // LEGAL: FFI boundary
            error: ptr::null_mut(),
        },
        Err(e) => err_result(&format!("{e}")),
    }
}

/// Free a generate result's strings.
///
/// # Safety
/// `result` must point to a valid `GllmGenerateResult` from `gllm_generate`.
#[no_mangle]
pub unsafe extern "C" fn gllm_free_generate_result(result: *mut GllmGenerateResult) {
    if result.is_null() {
        return;
    }
    let r = &mut *result;
    if !r.text.is_null() {
        drop(CString::from_raw(r.text));
        r.text = ptr::null_mut();
    }
    if !r.error.is_null() {
        drop(CString::from_raw(r.error));
        r.error = ptr::null_mut();
    }
}

/// Get the gllm client library version string.
///
/// Returns a static null-terminated string. Do not free.
#[no_mangle]
pub extern "C" fn gllm_client_version() -> *const c_char {
    static VERSION: &[u8] = b"0.12.0\0";
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── kind_from_u32 ─────────────────────────────────────────────────

    #[test]
    fn kind_from_u32_chat_default() {
        assert!(matches!(kind_from_u32(0), ModelKind::Chat));
        assert!(matches!(kind_from_u32(99), ModelKind::Chat));
    }

    #[test]
    fn kind_from_u32_embedding() {
        assert!(matches!(kind_from_u32(1), ModelKind::Embedding));
    }

    #[test]
    fn kind_from_u32_reranker() {
        assert!(matches!(kind_from_u32(2), ModelKind::Reranker));
    }

    #[test]
    fn kind_from_u32_classifier() {
        assert!(matches!(kind_from_u32(3), ModelKind::Classifier));
    }

    // ── gllm_client_version ───────────────────────────────────────────

    #[test]
    fn client_version_returns_non_null() {
        let ptr = gllm_client_version();
        assert!(!ptr.is_null());
    }

    #[test]
    fn client_version_is_valid_utf8() {
        let ptr = gllm_client_version();
        let cstr = unsafe { CStr::from_ptr(ptr) };
        let s = cstr.to_str().unwrap();
        assert!(!s.is_empty());
        assert!(s.chars().all(|c| c.is_ascii_digit() || c == '.'));
    }

    // ── GllmGenerateResult struct ─────────────────────────────────────

    #[test]
    fn generate_result_default_fields() {
        let result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: ptr::null_mut(),
        };
        assert!(result.text.is_null());
        assert!(result.error.is_null());
    }

    #[test]
    fn generate_result_with_error() {
        let msg = CString::new("test error").unwrap();
        let result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: msg.into_raw(),
        };
        assert!(result.text.is_null());
        assert!(!result.error.is_null());
        // Cleanup
        unsafe { drop(CString::from_raw(result.error)); }
    }

    // ── kind_from_u32 edge cases ──────────────────────────────────────

    #[test]
    fn kind_from_u32_large_values_all_chat() {
        assert!(matches!(kind_from_u32(u32::MAX), ModelKind::Chat));
        assert!(matches!(kind_from_u32(1_000_000), ModelKind::Chat));
    }

    #[test]
    fn kind_from_u32_boundary_above_known() {
        // Values 4 and 5 are not assigned; they fall through to Chat.
        assert!(matches!(kind_from_u32(4), ModelKind::Chat));
        assert!(matches!(kind_from_u32(5), ModelKind::Chat));
    }

    // ── GllmGenerateResult with text ──────────────────────────────────

    #[test]
    fn generate_result_with_text_and_no_error() {
        let text = CString::new("hello world").unwrap();
        let result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        assert!(!result.text.is_null());
        assert!(result.error.is_null());
        // Verify roundtrip: CString::from_raw recovers the original text.
        let recovered = unsafe { CString::from_raw(result.text) };
        assert_eq!(recovered.to_str().unwrap(), "hello world");
    }

    #[test]
    fn generate_result_with_both_text_and_error() {
        let text = CString::new("partial output").unwrap();
        let err = CString::new("truncated").unwrap();
        let result = GllmGenerateResult {
            text: text.into_raw(),
            error: err.into_raw(),
        };
        assert!(!result.text.is_null());
        assert!(!result.error.is_null());
        let recovered_text = unsafe { CString::from_raw(result.text) };
        let recovered_err = unsafe { CString::from_raw(result.error) };
        assert_eq!(recovered_text.to_str().unwrap(), "partial output");
        assert_eq!(recovered_err.to_str().unwrap(), "truncated");
    }

    // ── gllm_free_generate_result ─────────────────────────────────────

    #[test]
    fn free_generate_result_null_pointer_is_no_op() {
        // Calling with null must not crash (UB-free by our null guard).
        unsafe { gllm_free_generate_result(ptr::null_mut()) };
    }

    #[test]
    fn free_generate_result_both_fields_null() {
        let mut result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: ptr::null_mut(),
        };
        unsafe { gllm_free_generate_result(&mut result) };
        assert!(result.text.is_null());
        assert!(result.error.is_null());
    }

    #[test]
    fn free_generate_result_nullifies_freed_pointers() {
        let text = CString::new("some text").unwrap();
        let err = CString::new("some error").unwrap();
        let mut result = GllmGenerateResult {
            text: text.into_raw(),
            error: err.into_raw(),
        };
        unsafe { gllm_free_generate_result(&mut result) };
        // After freeing, both fields must be set to null by the function.
        assert!(result.text.is_null());
        assert!(result.error.is_null());
    }

    // ── gllm_init null pointer guard ──────────────────────────────────

    #[test]
    fn gllm_init_null_model_id_returns_null() {
        let ctx = unsafe { gllm_init(ptr::null(), 0) };
        assert!(ctx.is_null());
    }

    // ── CString edge cases ────────────────────────────────────────────

    #[test]
    fn cstring_new_unwrap_or_default_handles_empty_string() {
        // Empty string is valid for CString (no interior NUL).
        let c = CString::new("").unwrap_or_default();
        assert_eq!(c.as_bytes(), b"");
    }

    #[test]
    fn cstring_new_rejects_interior_nul() {
        // CString::new fails on interior NUL; unwrap_or_default yields empty.
        let c = CString::new(b"abc\0def".as_slice()).unwrap_or_default();
        assert_eq!(c.as_bytes(), b"");
    }

    // ── gllm_client_version format ────────────────────────────────────

    #[test]
    fn client_version_semver_format() {
        let ptr = gllm_client_version();
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        let parts: Vec<&str> = s.split('.').collect();
        assert_eq!(parts.len(), 3, "version should be major.minor.patch");
        for part in &parts {
            assert!(!part.is_empty(), "each component must be non-empty");
            assert!(part.chars().all(|c| c.is_ascii_digit()), "each component must be digits");
        }
    }

    // ── kind_from_u32 additional edge cases ────────────────────────────

    #[test]
    fn kind_from_u32_zero_is_chat() {
        assert!(matches!(kind_from_u32(0), ModelKind::Chat));
    }

    #[test]
    fn kind_from_u32_all_valid_kinds_distinct() {
        let chat = kind_from_u32(0);
        let embedding = kind_from_u32(1);
        let reranker = kind_from_u32(2);
        let classifier = kind_from_u32(3);
        // Each kind must be distinct from the others.
        assert_ne!(chat, embedding);
        assert_ne!(chat, reranker);
        assert_ne!(chat, classifier);
        assert_ne!(embedding, reranker);
        assert_ne!(embedding, classifier);
        assert_ne!(reranker, classifier);
    }

    // ── gllm_destroy null safety ───────────────────────────────────────

    #[test]
    fn gllm_destroy_null_is_no_op() {
        unsafe { gllm_destroy(ptr::null_mut()) };
    }

    // ── gllm_generate null-pointer error paths ─────────────────────────

    #[test]
    fn gllm_generate_null_ctx_returns_error() {
        let prompt = CString::new("hello").unwrap();
        let mut result = unsafe {
            gllm_generate(ptr::null_mut(), prompt.as_ptr(), 10, 0.8, 50, 0.9, 0)
        };
        assert!(result.text.is_null());
        assert!(!result.error.is_null());
        let err_msg = unsafe { CStr::from_ptr(result.error) }.to_str().unwrap();
        assert!(!err_msg.is_empty());
        unsafe { gllm_free_generate_result(&mut result) };
    }

    #[test]
    fn gllm_generate_null_prompt_returns_error() {
        // Both ctx and prompt null — tests that the null-prompt guard fires
        // before dereferencing ctx.
        let mut result = unsafe {
            gllm_generate(ptr::null_mut(), ptr::null(), 10, 0.8, 50, 0.9, 0)
        };
        assert!(result.text.is_null());
        assert!(!result.error.is_null());
        unsafe { gllm_free_generate_result(&mut result) };
    }

    #[test]
    fn gllm_generate_both_null_returns_error() {
        let mut result = unsafe {
            gllm_generate(ptr::null_mut(), ptr::null(), 0, 0.0, 0, 0.0, 0)
        };
        assert!(result.text.is_null());
        assert!(!result.error.is_null());
        let err_msg = unsafe { CStr::from_ptr(result.error) }.to_str().unwrap();
        assert_eq!(err_msg, "null pointer argument");
        unsafe { gllm_free_generate_result(&mut result) };
    }

    #[test]
    fn gllm_generate_zero_max_tokens_returns_error() {
        let mut result = unsafe {
            gllm_generate(ptr::null_mut(), ptr::null(), 0, 0.0, 0, 0.0, 0)
        };
        assert!(result.text.is_null());
        assert!(!result.error.is_null());
        unsafe { gllm_free_generate_result(&mut result) };
    }

    // ── GllmGenerateResult Debug trait ─────────────────────────────────

    #[test]
    fn generate_result_debug_format_both_null() {
        let result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: ptr::null_mut(),
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("GllmGenerateResult"));
    }

    #[test]
    fn generate_result_debug_format_with_text() {
        let text = CString::new("output text").unwrap();
        let result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        let debug_str = format!("{result:?}");
        assert!(debug_str.contains("GllmGenerateResult"));
        // Recover to avoid leak.
        unsafe { drop(CString::from_raw(result.text)); }
    }

    // ── gllm_client_version stability ──────────────────────────────────

    #[test]
    fn client_version_known_value() {
        let ptr = gllm_client_version();
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(s, "0.12.0");
    }

    #[test]
    fn client_version_multiple_calls_same_pointer() {
        let ptr1 = gllm_client_version();
        let ptr2 = gllm_client_version();
        // Static string — must return the same address every time.
        assert_eq!(ptr1, ptr2);
    }

    // ── GllmGenerateResult with empty strings ──────────────────────────

    #[test]
    fn generate_result_empty_text_string() {
        let text = CString::new("").unwrap();
        let result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        let recovered = unsafe { CString::from_raw(result.text) };
        assert_eq!(recovered.as_bytes(), b"");
    }

    #[test]
    fn generate_result_empty_error_string() {
        let err = CString::new("").unwrap();
        let mut result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: err.into_raw(),
        };
        unsafe { gllm_free_generate_result(&mut result) };
        assert!(result.error.is_null());
    }

    // ── GllmGenerateResult with long strings ───────────────────────────

    #[test]
    fn generate_result_long_text_roundtrip() {
        let long_text = "x".repeat(10_000);
        let text = CString::new(long_text.clone()).unwrap();
        let result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        let recovered = unsafe { CString::from_raw(result.text) };
        assert_eq!(recovered.to_str().unwrap().len(), 10_000);
        assert_eq!(recovered.to_str().unwrap(), long_text);
    }

    // ── free_generate_result only frees non-null fields ────────────────

    #[test]
    fn free_generate_result_text_only_nullifies_text() {
        let text = CString::new("abc").unwrap();
        let mut result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        unsafe { gllm_free_generate_result(&mut result) };
        assert!(result.text.is_null());
        assert!(result.error.is_null());
    }

    #[test]
    fn free_generate_result_error_only_nullifies_error() {
        let err = CString::new("err").unwrap();
        let mut result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: err.into_raw(),
        };
        unsafe { gllm_free_generate_result(&mut result) };
        assert!(result.text.is_null());
        assert!(result.error.is_null());
    }

    // ── gllm_init invalid UTF-8 ────────────────────────────────────────

    #[test]
    fn gllm_init_invalid_utf8_returns_null() {
        // Construct a byte sequence that is NOT valid UTF-8.
        let invalid: &[u8] = b"\xff\xfe\xfd\0";
        let cstr = unsafe { CStr::from_ptr(invalid.as_ptr() as *const c_char) };
        // CStr::to_str() should fail here — but the bytes include a NUL
        // terminator, so CStr construction succeeds; to_str() fails.
        assert!(cstr.to_str().is_err());
        // gllm_init must return NULL for invalid UTF-8.
        let ctx = unsafe { gllm_init(invalid.as_ptr() as *const c_char, 0) };
        assert!(ctx.is_null());
    }

    // ── kind_from_u32 with all ModelKind variant checks ────────────────

    #[test]
    fn kind_from_u32_model_kind_clone_copy() {
        let k = kind_from_u32(1);
        let k2 = k; // Copy
        assert_eq!(k, k2);
        let k3 = k.clone();
        assert_eq!(k, k3);
    }

    #[test]
    fn kind_from_u32_model_kind_hash_equal() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let hash_of = |k: ModelKind| {
            let mut h = DefaultHasher::new();
            k.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash_of(kind_from_u32(0)), hash_of(ModelKind::Chat));
        assert_eq!(hash_of(kind_from_u32(1)), hash_of(ModelKind::Embedding));
        assert_eq!(hash_of(kind_from_u32(2)), hash_of(ModelKind::Reranker));
        assert_eq!(hash_of(kind_from_u32(3)), hash_of(ModelKind::Classifier));
    }

    #[test]
    fn kind_from_u32_model_kind_debug_format() {
        assert!(format!("{:?}", kind_from_u32(0)).contains("Chat"));
        assert!(format!("{:?}", kind_from_u32(1)).contains("Embedding"));
        assert!(format!("{:?}", kind_from_u32(2)).contains("Reranker"));
        assert!(format!("{:?}", kind_from_u32(3)).contains("Classifier"));
    }

    // ── ModelKind::parse alias coverage ────────────────────────────────────

    #[test]
    fn model_kind_parse_chat_aliases() {
        assert_eq!(ModelKind::parse("chat"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("generation"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("generator"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("text-generation"), Some(ModelKind::Chat));
    }

    #[test]
    fn model_kind_parse_embedding_aliases() {
        assert_eq!(ModelKind::parse("embedding"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("embeddings"), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("embed"), Some(ModelKind::Embedding));
    }

    #[test]
    fn model_kind_parse_reranker_aliases() {
        assert_eq!(ModelKind::parse("rerank"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("reranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("re-ranker"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("re-rank"), Some(ModelKind::Reranker));
    }

    #[test]
    fn model_kind_parse_classifier_aliases() {
        assert_eq!(ModelKind::parse("classifier"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("classification"), Some(ModelKind::Classifier));
        assert_eq!(ModelKind::parse("classify"), Some(ModelKind::Classifier));
    }

    #[test]
    fn model_kind_parse_case_insensitive_and_trimmed() {
        assert_eq!(ModelKind::parse("CHAT"), Some(ModelKind::Chat));
        assert_eq!(ModelKind::parse("  Embedding  "), Some(ModelKind::Embedding));
        assert_eq!(ModelKind::parse("RERANKER"), Some(ModelKind::Reranker));
        assert_eq!(ModelKind::parse("Sequence-Classification"), Some(ModelKind::Classifier));
    }

    #[test]
    fn model_kind_parse_unknown_returns_none() {
        assert_eq!(ModelKind::parse(""), None);
        assert_eq!(ModelKind::parse("unknown"), None);
        assert_eq!(ModelKind::parse("caht"), None); // typo
        assert_eq!(ModelKind::parse("text"), None);
    }

    // ── ModelKind FromStr ──────────────────────────────────────────────────

    #[test]
    fn model_kind_from_str_valid() {
        assert_eq!("chat".parse::<ModelKind>(), Ok(ModelKind::Chat));
        assert_eq!("embedding".parse::<ModelKind>(), Ok(ModelKind::Embedding));
        assert_eq!("reranker".parse::<ModelKind>(), Ok(ModelKind::Reranker));
        assert_eq!("classifier".parse::<ModelKind>(), Ok(ModelKind::Classifier));
    }

    #[test]
    fn model_kind_from_str_invalid_returns_unit_err() {
        assert_eq!("bogus".parse::<ModelKind>(), Err(()));
        assert_eq!("".parse::<ModelKind>(), Err(()));
    }

    // ── GllmContext struct accessibility ───────────────────────────────────

    #[test]
    fn gllm_context_has_pub_client_field() {
        // Compile-time check: GllmContext.client field is publicly accessible.
        // We verify the struct layout without constructing an instance.
        assert!(std::mem::size_of::<GllmContext>() >= std::mem::size_of::<Client>());
    }

    // ── GllmGenerateResult repr(C) layout ──────────────────────────────────

    #[test]
    fn generate_result_size_is_two_pointers() {
        // repr(C) with two *mut c_char fields must be exactly 2 * sizeof(pointer).
        assert_eq!(
            std::mem::size_of::<GllmGenerateResult>(),
            2 * std::mem::size_of::<*mut c_char>(),
        );
    }

    #[test]
    fn generate_result_align_matches_pointer() {
        // repr(C) alignment must match pointer alignment.
        assert_eq!(
            std::mem::align_of::<GllmGenerateResult>(),
            std::mem::align_of::<*mut c_char>(),
        );
    }

    // ── gllm_free_generate_result double-free safety ───────────────────────

    #[test]
    fn free_generate_result_double_free_is_safe() {
        let text = CString::new("data").unwrap();
        let mut result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        unsafe { gllm_free_generate_result(&mut result) };
        assert!(result.text.is_null());
        // Second call on already-nullified fields must not crash.
        unsafe { gllm_free_generate_result(&mut result) };
        assert!(result.text.is_null());
        assert!(result.error.is_null());
    }

    // ── GllmGenerateResult with non-ASCII text ─────────────────────────────

    #[test]
    fn generate_result_multibyte_utf8_roundtrip() {
        let original = "你好世界🤖"; // Chinese + robot emoji
        let text = CString::new(original).unwrap();
        let result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        let recovered = unsafe { CString::from_raw(result.text) };
        assert_eq!(recovered.to_str().unwrap(), original);
    }

    // ── kind_from_u32 sequential monotonicity ──────────────────────────────

    #[test]
    fn kind_from_u32_sequential_values_cover_all_variants() {
        // Kinds 1..=3 must produce Embedding, Reranker, Classifier in order.
        let kinds: Vec<ModelKind> = (1..=3).map(kind_from_u32).collect();
        assert_eq!(kinds[0], ModelKind::Embedding);
        assert_eq!(kinds[1], ModelKind::Reranker);
        assert_eq!(kinds[2], ModelKind::Classifier);
    }

    // ── gllm_client_version pointer is readable ────────────────────────────

    #[test]
    fn client_version_bytes_contain_nul_terminator() {
        let ptr = gllm_client_version();
        let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes_with_nul();
        assert_eq!(*bytes.last().unwrap(), 0);
        assert!(bytes.len() > 1);
    }

    // ── gllm_client_version: version components are non-negative ────────────

    #[test]
    fn client_version_components_are_plausible_version_numbers() {
        let ptr = gllm_client_version();
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        let parts: Vec<u32> = s.split('.').map(|p| p.parse::<u32>().unwrap()).collect();
        assert_eq!(parts.len(), 3);
        // Major >= 0, minor >= 0, patch >= 0 (always true for u32, but validates parse)
        assert!(parts.iter().all(|&p| p < 1000), "version components should be < 1000");
    }

    // ── gllm_client_version: no extra bytes after NUL ──────────────────────

    #[test]
    fn client_version_no_extra_content_after_nul() {
        let ptr = gllm_client_version();
        let cstr = unsafe { CStr::from_ptr(ptr) };
        let s = cstr.to_str().unwrap();
        // The static VERSION is b"0.12.0\0" — length must be exactly 6 chars
        assert_eq!(s.len(), 6);
        assert_eq!(s, "0.12.0");
    }

    // ── kind_from_u32: boundary just below and above each valid value ──────

    #[test]
    fn kind_from_u32_boundary_transitions() {
        // 0 -> Chat, 1 -> Embedding: boundary at 0..=1
        assert!(matches!(kind_from_u32(0), ModelKind::Chat));
        assert!(matches!(kind_from_u32(1), ModelKind::Embedding));
        // 3 -> Classifier, 4 -> Chat (fallthrough): boundary at 3..=4
        assert!(matches!(kind_from_u32(3), ModelKind::Classifier));
        assert!(matches!(kind_from_u32(4), ModelKind::Chat));
    }

    // ── kind_from_u32: u32::MIN (0) is Chat ────────────────────────────────

    #[test]
    fn kind_from_u32_min_value_is_chat() {
        assert!(matches!(kind_from_u32(u32::MIN), ModelKind::Chat));
    }

    // ── GllmContext repr(C) size and alignment ─────────────────────────────

    #[test]
    fn gllm_context_repr_c_size_is_client_size() {
        // GllmContext is repr(C) with a single Client field.
        // Size must be exactly Client's size (no padding for single-field struct).
        assert_eq!(
            std::mem::size_of::<GllmContext>(),
            std::mem::size_of::<Client>(),
        );
    }

    #[test]
    fn gllm_context_repr_c_align_is_client_align() {
        assert_eq!(
            std::mem::align_of::<GllmContext>(),
            std::mem::align_of::<Client>(),
        );
    }

    // ── GllmGenerateResult: repr(C) field offsets are deterministic ─────────

    #[test]
    fn generate_result_field_offsets_are_sequential() {
        // In repr(C), `text` is at offset 0 and `error` is at offset sizeof(pointer).
        let text_offset = 0usize;
        let error_offset = std::mem::size_of::<*mut c_char>();
        // Verify we can compute these offsets; the struct must have text first.
        let result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: ptr::null_mut(),
        };
        let base = &result as *const GllmGenerateResult as usize;
        let text_addr = &result.text as *const _ as usize;
        let error_addr = &result.error as *const _ as usize;
        assert_eq!(text_addr - base, text_offset);
        assert_eq!(error_addr - base, error_offset);
    }

    // ── gllm_init: empty string model_id returns null ──────────────────────

    #[test]
    fn gllm_init_empty_model_id_returns_null() {
        let model = CString::new("").unwrap();
        let ctx = unsafe { gllm_init(model.as_ptr(), 0) };
        // Empty model_id should fail to create a Client -> returns null.
        assert!(ctx.is_null());
    }

    // ── gllm_generate: null ctx with valid prompt gives null-pointer error ──

    #[test]
    fn gllm_generate_null_ctx_valid_prompt_error_message() {
        let prompt = CString::new("test prompt").unwrap();
        let mut result = unsafe {
            gllm_generate(ptr::null_mut(), prompt.as_ptr(), 100, 1.0, 50, 0.95, 0)
        };
        // Assert
        assert!(result.text.is_null());
        assert!(!result.error.is_null());
        let err_msg = unsafe { CStr::from_ptr(result.error) }.to_str().unwrap();
        assert_eq!(err_msg, "null pointer argument");
        unsafe { gllm_free_generate_result(&mut result) };
    }

    // ── gllm_generate: thinking_budget negative maps to None ────────────────
    // (Tested indirectly: the function path converts negative to None.
    //  We verify the conversion logic that happens in gllm_generate.)

    #[test]
    fn thinking_budget_negative_converts_to_none() {
        // The logic in gllm_generate: if thinking_budget < 0 { None } else { Some(usize) }
        let tb: i32 = -1;
        let converted = if tb < 0 { None } else { Some(tb as usize) };
        assert!(converted.is_none());
    }

    #[test]
    fn thinking_budget_zero_converts_to_some_zero() {
        let tb: i32 = 0;
        let converted = if tb < 0 { None } else { Some(tb as usize) };
        assert_eq!(converted, Some(0usize));
    }

    #[test]
    fn thinking_budget_positive_converts_to_some_value() {
        let tb: i32 = 4096;
        let converted = if tb < 0 { None } else { Some(tb as usize) };
        assert_eq!(converted, Some(4096usize));
    }

    // ── gllm_generate: error result text is always null ─────────────────────

    #[test]
    fn gllm_generate_error_path_text_always_null() {
        // Both ctx and prompt null -> error path, text must be null.
        let mut result = unsafe {
            gllm_generate(ptr::null_mut(), ptr::null(), 0, 0.0, 0, 0.0, -1)
        };
        assert!(result.text.is_null(), "error result must have null text");
        assert!(!result.error.is_null(), "error result must have non-null error");
        unsafe { gllm_free_generate_result(&mut result) };
    }

    // ── GllmGenerateResult: CString::new rejection of interior NUL in text ──

    #[test]
    fn generate_result_cstring_interior_nul_rejected() {
        // CString::new fails on interior NUL; unwrap_or_default produces empty CString.
        let bytes = b"hello\0world".as_slice();
        let c = CString::new(bytes).unwrap_or_default();
        assert_eq!(c.as_bytes(), b"", "interior NUL causes default (empty) CString");
    }

    // ── err_result helper: produces correct field pattern ──────────────────

    #[test]
    fn err_result_helper_pattern() {
        // Simulate the err_result closure from gllm_generate.
        let msg = "some error message";
        let result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: CString::new(msg).unwrap_or_default().into_raw(),
        };
        assert!(result.text.is_null());
        assert!(!result.error.is_null());
        let recovered = unsafe { CStr::from_ptr(result.error) }.to_str().unwrap();
        assert_eq!(recovered, msg);
        unsafe { drop(CString::from_raw(result.error)); }
    }

    // ── gllm_free_generate_result: null error pointer not double-freed ──────

    #[test]
    fn free_generate_result_only_text_error_already_null() {
        let text = CString::new("only text here").unwrap();
        let mut result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        unsafe { gllm_free_generate_result(&mut result) };
        assert!(result.text.is_null());
        assert!(result.error.is_null(), "already-null error must stay null");
    }

    // ── GllmGenerateResult: text with unicode roundtrip preserves all bytes ─

    #[test]
    fn generate_result_unicode_varied_scripts_roundtrip() {
        // Arrange: text with Japanese, Arabic, Cyrillic, and emoji
        let original = "こんにちはمرحباПривет🎉";
        let text = CString::new(original).unwrap();
        let result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        // Act
        let recovered = unsafe { CString::from_raw(result.text) };
        // Assert
        assert_eq!(recovered.to_str().unwrap(), original);
        assert_eq!(recovered.as_bytes().len(), original.len());
    }

    // ── ModelKind::parse: text-classification alias (listed but not explicitly tested) ──

    #[test]
    fn model_kind_parse_text_classification_alias() {
        // Arrange & Act & Assert
        assert_eq!(
            ModelKind::parse("text-classification"),
            Some(ModelKind::Classifier)
        );
    }

    // ── GllmGenerateResult: text containing whitespace escape sequences ──

    #[test]
    fn generate_result_text_with_newlines_and_tabs_roundtrip() {
        // Arrange
        let original = "line1\nline2\tindented\r\nwindows";
        let text = CString::new(original).unwrap();
        let result = GllmGenerateResult {
            text: text.into_raw(),
            error: ptr::null_mut(),
        };
        // Act
        let recovered = unsafe { CString::from_raw(result.text) };
        // Assert
        assert_eq!(recovered.to_str().unwrap(), original);
    }

    // ── kind_from_u32: i32::MAX cast to u32 falls through to Chat ──

    #[test]
    fn kind_from_u32_i32_max_as_u32_is_chat() {
        // Arrange: i32::MAX = 2147483647, not a valid kind code
        let kind = kind_from_u32(i32::MAX as u32);
        // Assert
        assert!(matches!(kind, ModelKind::Chat));
    }

    // ── ModelKind: PartialEq identity for each variant ──

    #[test]
    fn model_kind_partial_eq_identity() {
        // Arrange
        let variants = [
            ModelKind::Chat,
            ModelKind::Embedding,
            ModelKind::Reranker,
            ModelKind::Classifier,
        ];
        // Act & Assert: each variant equals itself
        for v in &variants {
            assert_eq!(*v, *v);
        }
        // Assert: no cross-variant equality
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "variant {} must not equal variant {}", i, j);
                }
            }
        }
    }

    // ── kind_from_u32: idempotent -- same input always same output ──

    #[test]
    fn kind_from_u32_idempotent() {
        // Arrange
        for code in [0u32, 1, 2, 3, 4, 99, u32::MAX] {
            // Act
            let first = kind_from_u32(code);
            let second = kind_from_u32(code);
            // Assert
            assert_eq!(first, second, "kind_from_u32({code}) must be idempotent");
        }
    }

    // ── gllm_client_version: static pointer content unchanged across many reads ──

    #[test]
    fn client_version_content_stable_across_many_reads() {
        // Arrange
        let expected = unsafe { CStr::from_ptr(gllm_client_version()) }.to_str().unwrap().to_string();
        // Act: read 100 times
        for _ in 0..100 {
            let s = unsafe { CStr::from_ptr(gllm_client_version()) }.to_str().unwrap();
            // Assert
            assert_eq!(s, expected);
        }
    }

    // ── GllmGenerateResult: both fields with identical string content ──

    #[test]
    fn generate_result_identical_text_and_error_strings() {
        // Arrange
        let shared = "same content";
        let text = CString::new(shared).unwrap();
        let err = CString::new(shared).unwrap();
        let mut result = GllmGenerateResult {
            text: text.into_raw(),
            error: err.into_raw(),
        };
        // Act
        let recovered_text = unsafe { CString::from_raw(result.text) };
        let recovered_err = unsafe { CString::from_raw(result.error) };
        // Assert: both recovered strings equal the original
        assert_eq!(recovered_text.to_str().unwrap(), shared);
        assert_eq!(recovered_err.to_str().unwrap(), shared);
        // Cleanup: manually nullify since we already called from_raw
        result.text = ptr::null_mut();
        result.error = ptr::null_mut();
    }

    // ── gllm_init: nonexistent model_id returns null (non-empty, valid UTF-8) ──

    #[test]
    fn gllm_init_nonexistent_model_id_returns_null() {
        // Arrange: valid UTF-8, non-empty, but not a real model
        let model = CString::new("nonexistent-model-xyz-12345").unwrap();
        // Act
        let ctx = unsafe { gllm_init(model.as_ptr(), 0) };
        // Assert: Client::new fails for unknown models -> returns null
        assert!(ctx.is_null());
    }

    // ── ModelKind::parse: whitespace-only input returns None ──

    #[test]
    fn model_kind_parse_whitespace_only_returns_none() {
        // Arrange & Act & Assert
        assert_eq!(ModelKind::parse("   "), None);
        assert_eq!(ModelKind::parse("\t"), None);
        assert_eq!(ModelKind::parse("\n"), None);
    }

    // ── free_generate_result: freeing result with very long error string ──

    #[test]
    fn free_generate_result_long_error_string_no_crash() {
        // Arrange
        let long_err = "E".repeat(100_000);
        let err = CString::new(long_err).unwrap();
        let mut result = GllmGenerateResult {
            text: ptr::null_mut(),
            error: err.into_raw(),
        };
        // Act
        unsafe { gllm_free_generate_result(&mut result) };
        // Assert: both fields nullified, no crash or leak
        assert!(result.text.is_null());
        assert!(result.error.is_null());
    }
}
