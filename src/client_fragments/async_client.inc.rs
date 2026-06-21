// ============================================================================
// AsyncClient — async wrapper with concurrent multi-prompt submission
// ============================================================================

/// Async wrapper for `Client` with concurrent batch support (SPEC/09 REQ-API-5, SPEC/20 REQ-BCI-008).
///
/// Provides an async `generate_batch` method that supports concurrent
/// multi-prompt submission by offloading sync operations to threads.
/// Multiple calls to `generate_batch` can run concurrently without
/// blocking each other, enabling true parallelism for batch inference.
///
/// # Example
///
/// ```no_run
/// use gllm::{AsyncClient, Client, GenerateRequest};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::new_empty();
/// let async_client = AsyncClient::new(client);
///
/// let requests = vec![
///     GenerateRequest {
///         request_id: 1,
///         prompt_tokens: vec![1, 2, 3],
///         max_new_tokens: 10,
///         temperature: 0.8,
///         top_k: 0,
///         top_p: 0.9,
///         session_id: None,
///         eos_token_id: 2,
///         hook_ctx_ptr: std::ptr::null(),
///         callback_table_ptr: std::ptr::null(),
///     },
/// ];
/// let results = async_client.generate_batch(&requests).await.unwrap();
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
// @trace REQ-API-5 [entity:AsyncClient] async wrapper with Arc<Client> for concurrent batch
pub struct AsyncClient {
    inner: Arc<Client>,
}

impl AsyncClient {
    /// Create a new `AsyncClient` wrapping a `Client`.
    // @trace REQ-API-5 [entity:AsyncClient] constructor — wraps Client in Arc
    pub fn new(client: Client) -> Self {
        Self {
            inner: Arc::new(client),
        }
    }

    /// Batch generate with concurrent multi-prompt submission (REQ-API-5, REQ-BCI-008).
    ///
    /// Processes all requests as a single batch, offloaded to a dedicated
    /// thread for non-blocking async execution. Multiple concurrent calls
    /// to this method run in parallel threads via independent spawns,
    /// supporting concurrent multi-prompt submission.
    // @trace REQ-API-5 [entity:AsyncClient] [api:POST /client/generate_batch] async batch generate — concurrent multi-prompt, continuous batching, KV prefix sharing
    pub async fn generate_batch(
        &self,
        requests: &[crate::engine::batch_executor::GenerateRequest],
    ) -> Result<Vec<crate::engine::batch_executor::GenerateResult>, ClientError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let requests = requests.to_vec();
        let client = self.inner.clone();
        std::thread::spawn(move || {
            client.generate_batch(&requests)
        })
        .join()
        .map_err(|_| ClientError::RuntimeError("batch thread panicked".into()))?
    }
}

fn sg_err_to_client(err: crate::semantic_gatekeeper::SemanticGatekeeperError) -> ClientError {
    ClientError::RuntimeError(format!(
        "semantic gatekeeper: precompute error: {err}"
    ))
}

/// Head Routing SDK — 将内部 `HeadRoutingError` 映射为 `ClientError`。
fn hr_err_to_client(err: crate::head_routing::HeadRoutingError) -> ClientError {
    use crate::head_routing::HeadRoutingError as HE;
    match err {
        HE::TokenNotFound(_) | HE::EmptyLabels | HE::InvalidLayerAnchor(_)
        | HE::InvalidConfig(_) | HE::MidLayerNotSupported | HE::Backend(_) => {
            ClientError::RuntimeError(format!("{err}"))
        }
    }
}

/// Resolve `text` to a single token id via the loaded tokenizer.
/// 要求 tokenize 结果恰好 1 个 token,否则 `TokenNotFound(...)`。
fn resolve_single_token(
    tokenizer: &crate::tokenizer::TokenizerHandle,
    text: &str,
) -> Result<u32, crate::head_routing::HeadRoutingError> {
    let ids = tokenizer
        .encode(text, false)
        .map_err(|e| crate::head_routing::HeadRoutingError::Backend(format!("tokenizer error: {e}")))?;
    match ids.as_slice() {
        [] => Err(crate::head_routing::HeadRoutingError::TokenNotFound(format!(
            "{text:?} tokenized to empty id list"
        ))),
        [id] => Ok(*id),
        many => Err(crate::head_routing::HeadRoutingError::TokenNotFound(format!(
            "{text:?} tokenized to {} tokens {:?}, Head Routing requires single-token labels",
            many.len(),
            many
        ))),
    }
}

// ============================================================================
// Model Info
// ============================================================================

/// Information about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub arch: String,
    pub kind: ModelKind,
}

