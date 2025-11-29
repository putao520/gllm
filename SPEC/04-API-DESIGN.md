# gllm API Design

## Overview

Defines the public API interfaces for the gllm embedding and reranking library, using OpenAI SDK-style design.

## Revision History

| Version | Date | Description |
|---------|------|-------------|
| v0.1.0 | 2025-01-28 | Initial API design |
| v0.3.0 | 2025-01-29 | Unified sync/async API with `tokio` feature |

---

## Client API

### API-CLIENT-001: Client (Unified Client)

The same `Client` type provides both sync and async interfaces through conditional compilation.

**Sync Mode** (default, no feature flags):
```rust
/// Sync client
pub struct Client { /* ... */ }

impl Client {
    /// Create new client
    ///
    /// # Arguments
    /// * `model` - Model name (alias or HF repo ID)
    ///
    /// # Examples
    /// ```
    /// let client = Client::new("bge-small-en")?;
    /// ```
    pub fn new(model: &str) -> Result<Self>;

    /// Create client with custom config
    pub fn with_config(model: &str, config: ClientConfig) -> Result<Self>;

    /// Get Embeddings Builder
    pub fn embeddings<I, S>(&self, input: I) -> EmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;

    /// Get Rerank Builder
    pub fn rerank<I, S>(&self, query: &str, documents: I) -> RerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;
}
```

**Async Mode** (with `tokio` feature):
```rust
/// Async client (feature = "tokio")
#[cfg(feature = "tokio")]
impl Client {
    /// Create new client asynchronously
    pub async fn new(model: &str) -> Result<Self>;

    /// Create client with custom config asynchronously
    pub async fn with_config(model: &str, config: ClientConfig) -> Result<Self>;

    // embeddings() and rerank() methods are the same,
    // but the builders' generate() method returns a Future
}
```

---

## Embeddings API

### API-EMB-001: EmbeddingsBuilder

**Sync Mode** (default):
```rust
/// Embeddings request builder
pub struct EmbeddingsBuilder<'a> { /* ... */ }

impl<'a> EmbeddingsBuilder<'a> {
    /// Generate embeddings synchronously
    pub fn generate(self) -> Result<EmbeddingResponse>;
}
```

**Async Mode** (with `tokio` feature):
```rust
#[cfg(feature = "tokio")]
impl<'a> EmbeddingsBuilder<'a> {
    /// Generate embeddings asynchronously
    pub async fn generate(self) -> Result<EmbeddingResponse>;
}
```

---

## Rerank API

### API-RERANK-001: RerankBuilder

**Sync Mode** (default):
```rust
/// Rerank request builder
pub struct RerankBuilder<'a> { /* ... */ }

impl<'a> RerankBuilder<'a> {
    /// Set number of results to return
    pub fn top_n(self, n: usize) -> Self;

    /// Whether to return original documents
    pub fn return_documents(self, return_docs: bool) -> Self;

    /// Generate rerank results synchronously
    pub fn generate(self) -> Result<RerankResponse>;
}
```

**Async Mode** (with `tokio` feature):
```rust
#[cfg(feature = "tokio")]
impl<'a> RerankBuilder<'a> {
    /// Set number of results to return
    pub fn top_n(self, n: usize) -> Self;

    /// Whether to return original documents
    pub fn return_documents(self, return_docs: bool) -> Self;

    /// Generate rerank results asynchronously
    pub async fn generate(self) -> Result<RerankResponse>;
}
```

---

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wgpu` | Yes | GPU acceleration using WGPU |
| `cpu` | No | CPU-only inference using ndarray |
| `tokio` | No | Async interface (same API, add `.await`) |

---

## Usage Examples

### Embeddings (Sync)

```rust
use gllm::{Client, Result};

fn main() -> Result<()> {
    let client = Client::new("bge-small-en")?;

    let response = client
        .embeddings(["Hello world", "How are you?"])
        .generate()?;

    for emb in response.embeddings {
        println!("Index {}: {} dims", emb.index, emb.embedding.len());
    }

    Ok(())
}
```

### Embeddings (Async)

```rust
use gllm::{Client, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Same Client type, just add .await
    let client = Client::new("bge-small-en").await?;

    let response = client
        .embeddings(["Hello world", "How are you?"])
        .generate()
        .await?;

    for emb in response.embeddings {
        println!("Index {}: {} dims", emb.index, emb.embedding.len());
    }

    Ok(())
}
```

### Rerank (Sync)

```rust
use gllm::{Client, Result};

fn main() -> Result<()> {
    let client = Client::new("bge-reranker-v2")?;

    let response = client
        .rerank(
            "What is machine learning?",
            [
                "Machine learning is a subset of AI.",
                "The weather is nice today.",
                "Deep learning uses neural networks.",
            ],
        )
        .top_n(2)
        .return_documents(true)
        .generate()?;

    for result in response.results {
        println!(
            "Score: {:.4}, Doc: {}",
            result.score,
            result.document.unwrap_or_default()
        );
    }

    Ok(())
}
```

### Rerank (Async)

```rust
use gllm::{Client, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::new("bge-reranker-v2").await?;

    let response = client
        .rerank(
            "What is machine learning?",
            [
                "Machine learning is a subset of AI.",
                "The weather is nice today.",
                "Deep learning uses neural networks.",
            ],
        )
        .top_n(2)
        .return_documents(true)
        .generate()
        .await?;

    for result in response.results {
        println!(
            "Score: {:.4}, Doc: {}",
            result.score,
            result.document.unwrap_or_default()
        );
    }

    Ok(())
}
```

---

## Error Handling

All APIs return `Result<T>` type with error type `gllm::Error`:

```rust
use gllm::{Client, Error};

fn main() {
    match Client::new("unknown-model") {
        Ok(_client) => { /* use client */ }
        Err(Error::ModelNotFound(name)) => {
            eprintln!("Model '{}' not found", name);
        }
        Err(Error::DownloadError(msg)) => {
            eprintln!("Download failed: {}", msg);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}
```
