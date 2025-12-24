use crate::engine::{build_embedding_backend, build_rerank_backend, TokenizerAdapter, MAX_SEQ_LEN};
use crate::model::ModelManager;
use crate::types::{ClientConfig, Device, Error, RerankResponse, RerankResult, Result};
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

#[cfg(not(feature = "tokio"))]
use std::sync::mpsc;
#[cfg(feature = "tokio")]
use tokio::sync::{mpsc, oneshot};
#[cfg(feature = "tokio")]
use tokio::task;

#[cfg(not(feature = "tokio"))]
type EmbedResponseSender = mpsc::Sender<Result<Vec<f32>>>;
#[cfg(feature = "tokio")]
type EmbedResponseSender = oneshot::Sender<Result<Vec<f32>>>;

#[cfg(not(feature = "tokio"))]
type EmbedBatchResponseSender = mpsc::Sender<Result<Vec<Vec<f32>>>>;
#[cfg(feature = "tokio")]
type EmbedBatchResponseSender = oneshot::Sender<Result<Vec<Vec<f32>>>>;

#[cfg(not(feature = "tokio"))]
type RerankResponseSender = mpsc::Sender<Result<RerankResponse>>;
#[cfg(feature = "tokio")]
type RerankResponseSender = oneshot::Sender<Result<RerankResponse>>;

#[cfg(not(feature = "tokio"))]
type EmbedSender = mpsc::SyncSender<EmbedRequest>;
#[cfg(feature = "tokio")]
type EmbedSender = mpsc::Sender<EmbedRequest>;

#[cfg(not(feature = "tokio"))]
type EmbedReceiver = mpsc::Receiver<EmbedRequest>;
#[cfg(feature = "tokio")]
type EmbedReceiver = mpsc::Receiver<EmbedRequest>;

#[cfg(not(feature = "tokio"))]
type RerankSender = mpsc::SyncSender<RerankRequest>;
#[cfg(feature = "tokio")]
type RerankSender = mpsc::Sender<RerankRequest>;

#[cfg(not(feature = "tokio"))]
type RerankReceiver = mpsc::Receiver<RerankRequest>;
#[cfg(feature = "tokio")]
type RerankReceiver = mpsc::Receiver<RerankRequest>;

/// Embedder actor request.
pub enum EmbedRequest {
    Embed {
        text: String,
        respond: EmbedResponseSender,
    },
    EmbedBatch {
        texts: Vec<String>,
        respond: EmbedBatchResponseSender,
    },
    EmbedGraphBatch {
        inputs: Vec<crate::types::GraphCodeInput>,
        respond: EmbedBatchResponseSender,
    },
    Shutdown,
}

/// Reranker actor request.
pub enum RerankRequest {
    Rerank {
        query: String,
        documents: Vec<String>,
        top_n: Option<usize>,
        respond: RerankResponseSender,
    },
    Shutdown,
}

/// Shutdown guard that sends shutdown message and waits for actor thread when dropped.
/// Uses Mutex to make it Sync (required for async usage across await points).
struct ShutdownGuard {
    /// Shutdown action to send shutdown message
    shutdown_action: Mutex<Option<Box<dyn FnOnce() + Send + 'static>>>,
    /// Actor thread handle to join on drop
    thread_handle: Mutex<Option<JoinHandle<()>>>,
}

impl ShutdownGuard {
    fn new<F>(action: F, handle: JoinHandle<()>) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            shutdown_action: Mutex::new(Some(Box::new(action))),
            thread_handle: Mutex::new(Some(handle)),
        }
    }
}

impl Drop for ShutdownGuard {
    fn drop(&mut self) {
        // First, send shutdown message
        if let Ok(mut guard) = self.shutdown_action.lock() {
            if let Some(action) = guard.take() {
                action();
            }
        }
        // Then wait for the actor thread to complete
        if let Ok(mut guard) = self.thread_handle.lock() {
            if let Some(handle) = guard.take() {
                let _ = handle.join();
            }
        }
    }
}

/// Prepare model artifacts and return model directory.
fn prepare_model(model: &str) -> Result<(std::path::PathBuf, TokenizerAdapter)> {
    let manager = ModelManager::new(ClientConfig::default());
    let artifacts = manager.prepare(model)?;
    let tokenizer = artifacts.tokenizer.clone();
    Ok((artifacts.model_dir, tokenizer))
}

/// Tokenize texts for embedding.
fn tokenize_texts(tokenizer: &TokenizerAdapter, texts: &[String]) -> Vec<Vec<i64>> {
    texts
        .iter()
        .map(|t| tokenizer.encode(t, MAX_SEQ_LEN).0)
        .collect()
}

/// Tokenize query-document pairs for reranking.
fn tokenize_pairs(tokenizer: &TokenizerAdapter, query: &str, documents: &[String]) -> Vec<Vec<i64>> {
    documents
        .iter()
        .map(|doc| tokenizer.encode_pair(query, doc, MAX_SEQ_LEN).0)
        .collect()
}

#[cfg(not(feature = "tokio"))]
pub struct EmbedderHandle {
    sender: EmbedSender,
    _shutdown: ShutdownGuard,
}

#[cfg(not(feature = "tokio"))]
impl EmbedderHandle {
    pub fn new(model: &str) -> Result<Self> {
        start_embedder_actor(model.to_string()).map(|(sender, shutdown)| Self {
            sender,
            _shutdown: shutdown,
        })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(EmbedRequest::Embed {
                text: text.to_string(),
                respond: tx,
            })
            .map_err(|_| Error::InferenceError("Embedding actor is not available".into()))?;

        rx.recv()
            .map_err(|_| Error::InternalError("Embedder response channel closed".into()))?
    }

    pub fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(EmbedRequest::EmbedBatch {
                texts: texts.to_vec(),
                respond: tx,
            })
            .map_err(|_| Error::InferenceError("Embedding actor is not available".into()))?;

        rx.recv()
            .map_err(|_| Error::InternalError("Embedder response channel closed".into()))?
    }

    pub fn embed_graph_batch(&self, inputs: Vec<crate::types::GraphCodeInput>) -> Result<Vec<Vec<f32>>> {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(EmbedRequest::EmbedGraphBatch {
                inputs,
                respond: tx,
            })
            .map_err(|_| Error::InferenceError("Embedding actor is not available".into()))?;

        rx.recv()
            .map_err(|_| Error::InternalError("Embedder response channel closed".into()))?
    }
}

#[cfg(not(feature = "tokio"))]
fn start_embedder_actor(model: String) -> Result<(EmbedSender, ShutdownGuard)> {
    let (sender, receiver) = mpsc::sync_channel::<EmbedRequest>(32);
    let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();

    let handle = thread::spawn(move || embedder_loop(model, receiver, ready_tx));

    ready_rx.recv().unwrap_or_else(|_| {
        Err(Error::InternalError(
            "Embedder actor failed to start".into(),
        ))
    })?;

    let shutdown_sender = sender.clone();
    let shutdown = ShutdownGuard::new(
        move || {
            let _ = shutdown_sender.try_send(EmbedRequest::Shutdown);
        },
        handle,
    );

    Ok((sender, shutdown))
}

#[cfg(not(feature = "tokio"))]
fn embedder_loop(model: String, receiver: EmbedReceiver, ready: mpsc::Sender<Result<()>>) {
    // Prepare model and build embedding-only backend
    let (model_dir, tokenizer) = match prepare_model(&model) {
        Ok(v) => v,
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    let engine = match build_embedding_backend(&model_dir, &Device::Auto) {
        Ok(engine) => {
            let _ = ready.send(Ok(()));
            engine
        }
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    while let Ok(request) = receiver.recv() {
        match request {
            EmbedRequest::Embed { text, respond } => {
                let tokens = tokenize_texts(&tokenizer, &[text]);
                let result = engine.embed(&tokens).and_then(|mut v| {
                    v.pop()
                        .ok_or_else(|| Error::InferenceError("Missing embedding".into()))
                });
                let _ = respond.send(result);
            }
            EmbedRequest::EmbedBatch { texts, respond } => {
                let tokens = tokenize_texts(&tokenizer, &texts);
                let result = engine.embed(&tokens);
                let _ = respond.send(result);
            }
            EmbedRequest::EmbedGraphBatch { inputs, respond } => {
                // Fallback: extract code from GraphInput and tokenize as text
                // until EngineBackend supports DFG tensors.
                let texts: Vec<String> = inputs.into_iter().map(|i| i.code).collect();
                let tokens = tokenize_texts(&tokenizer, &texts);
                let result = engine.embed(&tokens);
                let _ = respond.send(result);
            }
            EmbedRequest::Shutdown => break,
        }
    }
}

#[cfg(not(feature = "tokio"))]
pub struct RerankerHandle {
    sender: RerankSender,
    _shutdown: ShutdownGuard,
}

#[cfg(not(feature = "tokio"))]
impl RerankerHandle {
    pub fn new(model: &str) -> Result<Self> {
        start_reranker_actor(model.to_string()).map(|(sender, shutdown)| Self {
            sender,
            _shutdown: shutdown,
        })
    }

    pub fn rerank(
        &self,
        query: &str,
        docs: &[String],
        top_n: Option<usize>,
    ) -> Result<RerankResponse> {
        let (tx, rx) = mpsc::channel();
        self.sender
            .send(RerankRequest::Rerank {
                query: query.to_string(),
                documents: docs.to_vec(),
                top_n,
                respond: tx,
            })
            .map_err(|_| Error::InferenceError("Reranker actor is not available".into()))?;

        rx.recv()
            .map_err(|_| Error::InternalError("Reranker response channel closed".into()))?
    }
}

#[cfg(not(feature = "tokio"))]
fn start_reranker_actor(model: String) -> Result<(RerankSender, ShutdownGuard)> {
    let (sender, receiver) = mpsc::sync_channel::<RerankRequest>(32);
    let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();

    let handle = thread::spawn(move || reranker_loop(model, receiver, ready_tx));

    ready_rx.recv().unwrap_or_else(|_| {
        Err(Error::InternalError(
            "Reranker actor failed to start".into(),
        ))
    })?;

    let shutdown_sender = sender.clone();
    let shutdown = ShutdownGuard::new(
        move || {
            let _ = shutdown_sender.try_send(RerankRequest::Shutdown);
        },
        handle,
    );

    Ok((sender, shutdown))
}

#[cfg(not(feature = "tokio"))]
fn reranker_loop(model: String, receiver: RerankReceiver, ready: mpsc::Sender<Result<()>>) {
    // Prepare model and build rerank-only backend
    let (model_dir, tokenizer) = match prepare_model(&model) {
        Ok(v) => v,
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    let engine = match build_rerank_backend(&model_dir, &Device::Auto) {
        Ok(engine) => {
            let _ = ready.send(Ok(()));
            engine
        }
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    while let Ok(request) = receiver.recv() {
        match request {
            RerankRequest::Rerank {
                query,
                documents,
                top_n,
                respond,
            } => {
                let tokens = tokenize_pairs(&tokenizer, &query, &documents);
                let result = engine.score(&tokens).map(|scores| {
                    let mut results: Vec<RerankResult> = scores
                        .into_iter()
                        .zip(documents.iter())
                        .enumerate()
                        .map(|(i, (score, doc))| RerankResult {
                            index: i,
                            score,
                            document: Some(doc.clone()),
                        })
                        .collect();
                    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some(n) = top_n {
                        results.truncate(n);
                    }
                    RerankResponse { results }
                });
                let _ = respond.send(result);
            }
            RerankRequest::Shutdown => break,
        }
    }
}

#[cfg(feature = "tokio")]
pub struct EmbedderHandle {
    sender: EmbedSender,
    _shutdown: ShutdownGuard,
}

#[cfg(feature = "tokio")]
impl EmbedderHandle {
    pub async fn new(model: &str) -> Result<Self> {
        let model = model.to_string();
        let (sender, shutdown) = task::spawn_blocking(move || start_embedder_actor_async(model))
            .await
            .map_err(|err| Error::InternalError(format!("Embedder actor init error: {err}")))??;

        Ok(Self {
            sender,
            _shutdown: shutdown,
        })
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EmbedRequest::Embed {
                text: text.to_string(),
                respond: tx,
            })
            .await
            .map_err(|_| Error::InferenceError("Embedding actor is not available".into()))?;

        rx.await
            .map_err(|_| Error::InternalError("Embedder response channel closed".into()))?
    }

    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EmbedRequest::EmbedBatch {
                texts: texts.to_vec(),
                respond: tx,
            })
            .await
            .map_err(|_| Error::InferenceError("Embedding actor is not available".into()))?;

        rx.await
            .map_err(|_| Error::InternalError("Embedder response channel closed".into()))?
    }

    pub async fn embed_graph_batch(&self, inputs: Vec<crate::types::GraphCodeInput>) -> Result<Vec<Vec<f32>>> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(EmbedRequest::EmbedGraphBatch {
                inputs,
                respond: tx,
            })
            .await
            .map_err(|_| Error::InferenceError("Embedding actor is not available".into()))?;

        rx.await
            .map_err(|_| Error::InternalError("Embedder response channel closed".into()))?
    }
}

#[cfg(feature = "tokio")]
fn start_embedder_actor_async(model: String) -> Result<(EmbedSender, ShutdownGuard)> {
    let (sender, receiver) = mpsc::channel::<EmbedRequest>(32);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

    let handle = thread::spawn(move || embedder_loop_async(model, receiver, ready_tx));

    ready_rx.recv().unwrap_or_else(|_| {
        Err(Error::InternalError(
            "Embedder actor failed to start".into(),
        ))
    })?;

    let shutdown_sender = sender.clone();
    let shutdown = ShutdownGuard::new(
        move || {
            let _ = shutdown_sender.try_send(EmbedRequest::Shutdown);
        },
        handle,
    );

    Ok((sender, shutdown))
}

#[cfg(feature = "tokio")]
fn embedder_loop_async(
    model: String,
    mut receiver: EmbedReceiver,
    ready: std::sync::mpsc::Sender<Result<()>>,
) {
    // Prepare model and build embedding-only backend
    let (model_dir, tokenizer) = match prepare_model(&model) {
        Ok(v) => v,
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    let engine = match build_embedding_backend(&model_dir, &Device::Auto) {
        Ok(engine) => {
            let _ = ready.send(Ok(()));
            engine
        }
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    while let Some(request) = receiver.blocking_recv() {
        match request {
            EmbedRequest::Embed { text, respond } => {
                let tokens = tokenize_texts(&tokenizer, &[text]);
                let result = engine.embed(&tokens).and_then(|mut v| {
                    v.pop()
                        .ok_or_else(|| Error::InferenceError("Missing embedding".into()))
                });
                let _ = respond.send(result);
            }
            EmbedRequest::EmbedBatch { texts, respond } => {
                let tokens = tokenize_texts(&tokenizer, &texts);
                let result = engine.embed(&tokens);
                let _ = respond.send(result);
            }
            EmbedRequest::EmbedGraphBatch { inputs, respond } => {
                // Fallback: extract code from GraphInput and tokenize as text.
                let texts: Vec<String> = inputs.into_iter().map(|i| i.code).collect();
                let tokens = tokenize_texts(&tokenizer, &texts);
                let result = engine.embed(&tokens);
                let _ = respond.send(result);
            }
            EmbedRequest::Shutdown => break,
        }
    }
}

#[cfg(feature = "tokio")]
pub struct RerankerHandle {
    sender: RerankSender,
    _shutdown: ShutdownGuard,
}

#[cfg(feature = "tokio")]
impl RerankerHandle {
    pub async fn new(model: &str) -> Result<Self> {
        let model = model.to_string();
        let (sender, shutdown) = task::spawn_blocking(move || start_reranker_actor_async(model))
            .await
            .map_err(|err| Error::InternalError(format!("Reranker actor init error: {err}")))??;

        Ok(Self {
            sender,
            _shutdown: shutdown,
        })
    }

    pub async fn rerank(
        &self,
        query: &str,
        docs: &[String],
        top_n: Option<usize>,
    ) -> Result<RerankResponse> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(RerankRequest::Rerank {
                query: query.to_string(),
                documents: docs.to_vec(),
                top_n,
                respond: tx,
            })
            .await
            .map_err(|_| Error::InferenceError("Reranker actor is not available".into()))?;

        rx.await
            .map_err(|_| Error::InternalError("Reranker response channel closed".into()))?
    }
}

#[cfg(feature = "tokio")]
fn start_reranker_actor_async(model: String) -> Result<(RerankSender, ShutdownGuard)> {
    let (sender, receiver) = mpsc::channel::<RerankRequest>(32);
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();

    let handle = thread::spawn(move || reranker_loop_async(model, receiver, ready_tx));

    ready_rx.recv().unwrap_or_else(|_| {
        Err(Error::InternalError(
            "Reranker actor failed to start".into(),
        ))
    })?;

    let shutdown_sender = sender.clone();
    let shutdown = ShutdownGuard::new(
        move || {
            let _ = shutdown_sender.try_send(RerankRequest::Shutdown);
        },
        handle,
    );

    Ok((sender, shutdown))
}

#[cfg(feature = "tokio")]
fn reranker_loop_async(
    model: String,
    mut receiver: RerankReceiver,
    ready: std::sync::mpsc::Sender<Result<()>>,
) {
    // Prepare model and build rerank-only backend
    let (model_dir, tokenizer) = match prepare_model(&model) {
        Ok(v) => v,
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    let engine = match build_rerank_backend(&model_dir, &Device::Auto) {
        Ok(engine) => {
            let _ = ready.send(Ok(()));
            engine
        }
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    while let Some(request) = receiver.blocking_recv() {
        match request {
            RerankRequest::Rerank {
                query,
                documents,
                top_n,
                respond,
            } => {
                let tokens = tokenize_pairs(&tokenizer, &query, &documents);
                let result = engine.score(&tokens).map(|scores| {
                    let mut results: Vec<RerankResult> = scores
                        .into_iter()
                        .zip(documents.iter())
                        .enumerate()
                        .map(|(i, (score, doc))| RerankResult {
                            index: i,
                            score,
                            document: Some(doc.clone()),
                        })
                        .collect();
                    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some(n) = top_n {
                        results.truncate(n);
                    }
                    RerankResponse { results }
                });
                let _ = respond.send(result);
            }
            RerankRequest::Shutdown => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{EmbedderHandle, RerankerHandle};

    // ==================== Sync Tests ====================

    /// Test EmbedderHandle with real model (sync).
    #[cfg(not(feature = "tokio"))]
    #[test]
    fn test_embedder_handle_sync() {
        let handle = EmbedderHandle::new("bge-small-en").expect("embedder handle");
        let vector = handle.embed("Hello world").expect("embedding");

        assert_eq!(vector.len(), 384);
        assert!(vector.iter().all(|v| v.is_finite()));
    }

    /// Test EmbedderHandle batch (sync).
    #[cfg(not(feature = "tokio"))]
    #[test]
    fn test_embedder_handle_batch_sync() {
        let handle = EmbedderHandle::new("bge-small-en").expect("embedder handle");
        let texts = vec![
            "Hello world".to_string(),
            "Machine learning".to_string(),
        ];
        let vectors = handle.embed_batch(&texts).expect("embeddings");

        assert_eq!(vectors.len(), 2);
        assert!(vectors.iter().all(|v| v.len() == 384));
    }

    /// Test RerankerHandle with real model (sync).
    /// Uses bge-reranker-base for faster CPU inference (440MB vs 2GB for v2-m3).
    #[cfg(not(feature = "tokio"))]
    #[test]
    fn test_reranker_handle_sync() {
        let handle = RerankerHandle::new("bge-reranker-base").expect("reranker handle");
        let documents = vec![
            "Paris is the capital of France".to_string(),
            "Berlin is the capital of Germany".to_string(),
        ];
        let response = handle
            .rerank("What is the capital of France?", &documents, None)
            .expect("rerank result");

        assert_eq!(response.results.len(), 2);
        assert!(response.results[0].score >= response.results[1].score);
    }

    // ==================== Async Tests ====================

    /// Test EmbedderHandle with real bge-small-en model (async).
    #[cfg(feature = "tokio")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_embedder_handle_real_model() {
        let handle = EmbedderHandle::new("bge-small-en")
            .await
            .expect("embedder handle");

        let vector = handle
            .embed("Hello world")
            .await
            .expect("embedding");

        // Real bge-small-en has 384 dimensions
        assert_eq!(vector.len(), 384);
        assert!(vector.iter().all(|v| v.is_finite()));
    }

    /// Test EmbedderHandle batch embedding with real model.
    #[cfg(feature = "tokio")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_embedder_handle_batch_real_model() {
        let handle = EmbedderHandle::new("bge-small-en")
            .await
            .expect("embedder handle");

        let texts = vec![
            "Hello world".to_string(),
            "Machine learning".to_string(),
            "Rust programming".to_string(),
        ];
        let vectors = handle.embed_batch(&texts).await.expect("embeddings");

        assert_eq!(vectors.len(), 3);
        assert!(vectors.iter().all(|v| v.len() == 384));
        assert!(vectors.iter().all(|v| v.iter().all(|x| x.is_finite())));
    }

    /// Test RerankerHandle with real bge-reranker-base model.
    /// Uses bge-reranker-base for faster CPU inference.
    #[cfg(feature = "tokio")]
    #[tokio::test(flavor = "multi_thread")]
    async fn test_reranker_handle_real_model() {
        let handle = RerankerHandle::new("bge-reranker-base")
            .await
            .expect("reranker handle");

        let documents = vec![
            "Paris is the capital of France".to_string(),
            "London is in the United Kingdom".to_string(),
            "Berlin is the capital of Germany".to_string(),
        ];
        let response = handle
            .rerank("What is the capital of France?", &documents, None)
            .await
            .expect("rerank result");

        assert_eq!(response.results.len(), 3);
        // Results should be sorted by score descending
        assert!(response.results[0].score >= response.results[1].score);
        assert!(response.results[1].score >= response.results[2].score);
        // All scores should be finite
        assert!(response.results.iter().all(|r| r.score.is_finite()));
    }
}
