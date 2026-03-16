//! C FFI exports for gllm inference engine.
//!
//! Provides a stable C ABI for loading models, running inference, and managing
//! the lifecycle from C/C++ callers.
//!
//! # Lifecycle
//! ```c
//! GllmContext *ctx = gllm_init("model-id", GLLM_KIND_CHAT);
//! GllmGenerateResult result = gllm_generate(ctx, "Hello", 128, 0.7, 40, 0.9);
//! // use result.text ...
//! gllm_free_generate_result(&result);
//! gllm_destroy(ctx);
//! ```

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::client::Client;
use crate::manifest::ModelKind;

/// Opaque context handle exposed to C callers.
pub struct GllmContext {
    client: Client,
}

/// Result of a text generation call.
#[repr(C)]
pub struct GllmGenerateResult {
    /// Null-terminated UTF-8 text. NULL on error.
    pub text: *mut c_char,
    /// Null-terminated error message. NULL on success.
    pub error: *mut c_char,
}

/// Model kind constants for C callers.
pub const GLLM_KIND_CHAT: u32 = 0;
pub const GLLM_KIND_EMBEDDING: u32 = 1;
pub const GLLM_KIND_RERANK: u32 = 2;

fn kind_from_u32(kind: u32) -> ModelKind {
    match kind {
        1 => ModelKind::Embedding,
        2 => ModelKind::Reranker,
        _ => ModelKind::Chat,
    }
}

/// Initialize a gllm context by loading a model.
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
    match Client::new(model_id, kind_from_u32(kind)) {
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
) -> GllmGenerateResult {
    let err_result = |msg: &str| -> GllmGenerateResult {
        GllmGenerateResult {
            text: ptr::null_mut(),
            error: CString::new(msg).unwrap_or_default().into_raw(),
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
    match ctx.client.execute_generation(
        prompt_str,
        max_tokens as usize,
        temperature,
        top_k as usize,
        top_p,
        None,
    ) {
        Ok(resp) => GllmGenerateResult {
            text: CString::new(resp.text).unwrap_or_default().into_raw(),
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
/// Note: `gllm_version()` is exported by gllm-kernels for the backend version.
#[no_mangle]
pub extern "C" fn gllm_client_version() -> *const c_char {
    static VERSION: &[u8] = b"0.11.0\0";
    VERSION.as_ptr() as *const c_char
}
