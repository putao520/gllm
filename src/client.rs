//! gllm Client API — Sync-first, lock-free design (per SPEC 04-API-DESIGN).
//!
//! # Design Principles
//!
//! - **Sync-first**: All operations are synchronous. No async runtime overhead.
//!   Inference engines are CPU-bound compute, not I/O-bound web services.
//! - **Lock-free state**: `arc_swap::ArcSwapOption` for zero-overhead reads,
//!   atomic lock-free model swap. No RwLock, no poisoning, no lock contention.
//! - **Builder pattern**: Complex configuration via fluent API
//! - **Explicit types**: Strong typing over string magic values
//! - **Result-oriented**: Clear success/failure via `Result<T, GllmError>`
//!
//! 代码组织 (include! 模式 — 编译为单模块，物理分散到 5 个片段):
//! - `client_fragments/error_config.inc.rs`  — ClientError, ClientConfig, ClientState
//! - `client_fragments/builder.inc.rs`        — ClientBuilder + impl
//! - `client_fragments/client_impl.inc.rs`    — Client struct + impl (generate, embed, etc.)
//! - `client_fragments/async_client.inc.rs`   — AsyncClient
//! - `client_fragments/tests.inc.rs`          — 测试模块

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwapOption;

use crate::backend::{
    detect_backend, BackendContext, BackendContextError, BackendType,
};
use crate::engine::arbiter::InferenceMode;
use crate::embeddings::{Embedding, EmbeddingsResponse, RagResponse};
use crate::engine::executor::{BackendError, ExecutorError};
use crate::generation::GenerationResponse;
use crate::loader::{Loader, LoaderConfig, LoaderError, WeightFormat};
use crate::manifest::{
    map_architecture_token_for_kind, MoEConfig, ModelKind, ModelManifest,
    EMPTY_FILE_MAP,
};
use crate::rerank::{RerankResponse, RerankResult};
use gllm_kernels::types::DType;
use thiserror::Error;

include!("client_fragments/error_config.inc.rs");
include!("client_fragments/builder.inc.rs");
include!("client_fragments/client_impl.inc.rs");
include!("client_fragments/async_client.inc.rs");

#[cfg(test)]
include!("client_fragments/tests.inc.rs");
