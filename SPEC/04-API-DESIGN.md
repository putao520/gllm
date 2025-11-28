# gllm API 设计

## 概述

定义 gllm 嵌入和重排序库的公共 API 接口，采用 OpenAI SDK 风格设计。

## 修订历史

| 版本 | 日期 | 描述 |
|------|------|------|
| v0.1.0 | 2025-01-28 | 初始 API 设计 |

---

## 客户端 API

### API-CLIENT-001: Client (同步客户端)

```rust
/// 同步客户端
pub struct Client { /* ... */ }

impl Client {
    /// 创建新客户端
    ///
    /// # Arguments
    /// * `model` - 模型名称 (别名或 HF repo ID)
    ///
    /// # Examples
    /// ```
    /// let client = Client::new("bge-m3")?;
    /// ```
    pub fn new(model: &str) -> Result<Self>;

    /// 使用自定义配置创建客户端
    pub fn with_config(model: &str, config: ClientConfig) -> Result<Self>;

    /// 获取 Embeddings Builder
    pub fn embeddings<I, S>(&self, input: I) -> EmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;

    /// 获取 Rerank Builder
    pub fn rerank<I, S>(&self, query: &str, documents: I) -> RerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;
}
```

### API-CLIENT-002: AsyncClient (异步客户端)

```rust
/// 异步客户端 (feature = "async")
#[cfg(feature = "async")]
pub struct AsyncClient { /* ... */ }

#[cfg(feature = "async")]
impl AsyncClient {
    /// 异步创建新客户端
    pub async fn new(model: &str) -> Result<Self>;

    /// 使用自定义配置异步创建客户端
    pub async fn with_config(model: &str, config: ClientConfig) -> Result<Self>;

    /// 获取异步 Embeddings Builder
    pub fn embeddings<I, S>(&self, input: I) -> AsyncEmbeddingsBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;

    /// 获取异步 Rerank Builder
    pub fn rerank<I, S>(&self, query: &str, documents: I) -> AsyncRerankBuilder<'_>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>;
}
```

---

## Embeddings API

### API-EMB-001: EmbeddingsBuilder

```rust
/// Embeddings 请求构建器
pub struct EmbeddingsBuilder<'a> { /* ... */ }

impl<'a> EmbeddingsBuilder<'a> {
    /// 同步生成嵌入向量
    pub fn generate(self) -> Result<EmbeddingResponse>;
}
```

### API-EMB-002: AsyncEmbeddingsBuilder

```rust
/// 异步 Embeddings 请求构建器
#[cfg(feature = "async")]
pub struct AsyncEmbeddingsBuilder<'a> { /* ... */ }

#[cfg(feature = "async")]
impl<'a> AsyncEmbeddingsBuilder<'a> {
    /// 异步生成嵌入向量
    pub async fn generate(self) -> Result<EmbeddingResponse>;
}
```

---

## Rerank API

### API-RERANK-001: RerankBuilder

```rust
/// Rerank 请求构建器
pub struct RerankBuilder<'a> { /* ... */ }

impl<'a> RerankBuilder<'a> {
    /// 设置返回结果数量
    pub fn top_n(self, n: usize) -> Self;

    /// 是否返回原始文档
    pub fn return_documents(self, return_docs: bool) -> Self;

    /// 同步生成重排序结果
    pub fn generate(self) -> Result<RerankResponse>;
}
```

### API-RERANK-002: AsyncRerankBuilder

```rust
/// 异步 Rerank 请求构建器
#[cfg(feature = "async")]
pub struct AsyncRerankBuilder<'a> { /* ... */ }

#[cfg(feature = "async")]
impl<'a> AsyncRerankBuilder<'a> {
    /// 设置返回结果数量
    pub fn top_n(self, n: usize) -> Self;

    /// 是否返回原始文档
    pub fn return_documents(self, return_docs: bool) -> Self;

    /// 异步生成重排序结果
    pub async fn generate(self) -> Result<RerankResponse>;
}
```

---

## 使用示例

### Embeddings (同步)

```rust
use gllm::{Client, Result};

fn main() -> Result<()> {
    let client = Client::new("bge-m3")?;

    let response = client
        .embeddings(["Hello world", "How are you?"])
        .generate()?;

    for emb in response.embeddings {
        println!("Index {}: {} dims", emb.index, emb.embedding.len());
    }

    Ok(())
}
```

### Embeddings (异步)

```rust
use gllm::{AsyncClient, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let client = AsyncClient::new("bge-m3").await?;

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

### Rerank (同步)

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

---

## 错误处理

所有 API 返回 `Result<T>` 类型，错误类型为 `gllm::Error`：

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
