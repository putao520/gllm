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
