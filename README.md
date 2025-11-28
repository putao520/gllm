# gllm: Pure Rust Local Embeddings & Reranking

[![Crates.io](https://img.shields.io/crates/v/gllm.svg)](https://crates.io/crates/gllm)
[![Documentation](https://docs.rs/gllm/badge.svg)](https://docs.rs/gllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**gllm** is a pure Rust library for local text embeddings and reranking, built on the [Burn](https://github.com/tracel-ai/burn) deep learning framework. It provides an OpenAI SDK-style API with zero external C dependencies, supporting static compilation.

## ✨ What You Can Do With gllm

- 📝 **Generate text embeddings** - Convert text into high-dimensional vectors for semantic search
- 🔄 **Rerank documents** - Sort documents by relevance to a query using cross-encoders
- ⚡ **High performance** - GPU acceleration with WGPU or CPU-only inference
- 🎯 **Production ready** - Pure Rust implementation with static compilation support
- 🚀 **Easy to use** - OpenAI-style API with builder patterns

## 🚀 Quick Start

Add gllm to your `Cargo.toml`:

```toml
[dependencies]
gllm = "0.1.0"
```

### Text Embeddings

Generate semantic embeddings for search, clustering, or similarity matching:

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client with built-in BGE-M3 model (auto-downloaded)
    let client = Client::new("bge-m3")?;

    // Generate embeddings for multiple texts
    let response = client
        .embeddings([
            "What is machine learning?",
            "How does neural network work?",
            "Python programming tutorial"
        ])
        .generate()?;

    // Process results
    for embedding in response.embeddings {
        println!("Text {}: {} dimensional vector",
                 embedding.index,
                 embedding.embedding.len());
    }

    println!("Used {} tokens", response.usage.total_tokens);
    Ok(())
}
```

### Document Reranking

Sort documents by relevance to improve search results:

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("bge-reranker-v2")?;

    let query = "What are the benefits of renewable energy?";
    let documents = vec![
        "Solar power is clean and sustainable.",
        "Yesterday's weather was sunny.",
        "Wind energy reduces carbon emissions significantly.",
        "Traditional fossil fuels cause pollution.",
        "The stock market closed higher today."
    ];

    // Rerank documents and get top 3 most relevant
    let response = client
        .rerank(query, documents)
        .top_n(3)
        .return_documents(true)  // Include original documents in response
        .generate()?;

    println!("Reranked results:");
    for result in response.results {
        println!("Score: {:.4}", result.score);
        if let Some(doc) = result.document {
            println!("Document: {}", doc);
        }
        println!();
    }

    Ok(())
}
```

### Async Support

For async applications, enable the `async` feature:

```toml
[dependencies]
gllm = { version = "0.1.0", features = ["async"] }
```

```rust
use gllm::AsyncClient;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = AsyncClient::new("bge-m3").await?;

    let response = client
        .embeddings(["Hello world", "Async programming"])
        .generate()
        .await?;

    for embedding in response.embeddings {
        println!("Vector: {} dimensions", embedding.embedding.len());
    }

    Ok(())
}
```

## 🎯 Built-in Models

gllm includes aliases for popular open-source models:

| Alias | HuggingFace Model | Type | Use Case |
|-------|------------------|------|----------|
| `bge-m3` | `BAAI/bge-m3` | Embedding | Multilingual, 8192 tokens, 1024 dims |
| `bge-large-zh` | `BAAI/bge-large-zh-v1.5` | Embedding | Chinese, 1024 dims |
| `bge-small-en` | `BAAI/bge-small-en-v1.5` | Embedding | English, 384 dims |
| `bge-reranker-v2` | `BAAI/bge-reranker-v2-m3` | Rerank | Multilingual reranking |
| `bge-reranker-large` | `BAAI/bge-reranker-large` | Rerank | High-performance reranking |

You can also use any HuggingFace model ID directly:

```rust
// Use any HuggingFace SafeTensors model
let client = Client::new("sentence-transformers/all-MiniLM-L6-v2")?;
```

## 🔧 Advanced Usage

### Backend Selection

Choose between GPU and CPU backends using feature flags:

```toml
# Default: WGPU (GPU acceleration)
gllm = "0.1.0"

# CPU-only (no GPU dependencies)
gllm = { version = "0.1.0", features = ["cpu"] }

# Both backends available
gllm = { version = "0.1.0", features = ["wgpu", "cpu"] }
```

### Custom Configuration

```rust
use gllm::{Client, ClientConfig, Device};

let config = ClientConfig {
    model_dir: Some("/custom/model/path".into()),  // Override model storage
    device: Device::Auto,                           // Auto-select GPU/CPU
};

let client = Client::with_config("bge-m3", config)?;
```

### Batch Processing

```rust
let texts: Vec<String> = vec![
    "First document".to_string(),
    "Second document".to_string(),
    // ... thousands more documents
];

let response = client.embeddings(texts).generate()?;

// Process embeddings efficiently
for embedding in response.embeddings {
    // Process each vector
}
```

### Vector Search

```rust
use std::collections::HashMap;

let query = "machine learning tutorials";
let documents = vec![
    "Introduction to machine learning",
    "Deep learning with neural networks",
    "Python basics for beginners",
    "Advanced calculus topics"
];

// Generate query embedding
let query_response = client.embeddings([query]).generate()?;
let query_vec = &query_response.embeddings[0].embedding;

// Generate document embeddings
let doc_response = client.embeddings(documents).generate()?;

// Calculate similarities and find best matches
let mut similarities = Vec::new();
for (i, doc_emb) in doc_response.embeddings.iter().enumerate() {
    let similarity = cosine_similarity(query_vec, &doc_emb.embedding);
    similarities.push((i, similarity));
}

// Sort by similarity (descending)
similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

println!("Most relevant documents:");
for (idx, similarity) in similarities.iter().take(3) {
    println!("{} (similarity: {:.4})", documents[*idx], similarity);
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}
```

## 🏗️ Architecture

### What Makes gllm Special

- **100% Pure Rust** - No C/C++ dependencies, enabling static compilation
- **Static Compilation Ready** - Build self-contained binaries with `--target x86_64-unknown-linux-musl`
- **SafeTensors Only** - Secure model format with built-in validation
- **Auto Model Management** - Download and cache models from HuggingFace automatically
- **Flexible Backends** - GPU (WGPU) or CPU inference based on your needs
- **OpenAI-Compatible API** - Familiar builder patterns and response structures

### Model Storage

Models are automatically downloaded and cached in `~/.gllm/models/`:

```
~/.gllm/models/
├── BAAI--bge-m3/
│   ├── model.safetensors          # Model weights
│   ├── config.json               # Model configuration
│   ├── tokenizer.json            # Tokenizer
│   └── tokenizer_config.json     # Tokenizer config
└── BAAI--bge-reranker-v2-m3/
    └── ...
```

## 🛠️ Installation & Requirements

### System Requirements

- **Rust 1.70+** (2021 edition)
- **GPU (Optional)** - For WGPU backend:
  - Vulkan, DirectX 12, Metal, or OpenGL 4.3+ support
- **Memory** - Minimum 2GB RAM, 4GB+ recommended for larger models

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wgpu` | ✅ | GPU acceleration using WGPU |
| `cpu` | ❌ | CPU-only inference using ndarray |
| `async` | ❌ | Async client support with tokio |

### Build Examples

```bash
# Default (GPU + CPU fallback)
cargo build --release

# CPU-only
cargo build --release --features cpu

# Async support
cargo build --release --features "wgpu,async"

# Static compilation
cargo build --release --target x86_64-unknown-linux-musl
```

## 📊 Performance

### Benchmarks (BGE-M3, 512-length text)

| Backend | Device | Throughput | Memory Usage |
|---------|--------|------------|--------------|
| WGPU | RTX 4090 | ~150 texts/sec | ~1.2GB |
| WGPU | Apple M2 | ~45 texts/sec | ~800MB |
| CPU | Intel i7-12700K | ~8 texts/sec | ~600MB |

*Results vary by model size and hardware*

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
cargo test

# Integration tests (requires CPU backend)
cargo test --features cpu --test integration

# All tests with verbose output
cargo test -- --nocapture
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-org/gllm.git
cd gllm
cargo test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Burn Framework](https://github.com/tracel-ai/burn) - Pure Rust deep learning framework
- [HuggingFace](https://huggingface.co/) - Model hosting and tokenizers
- [BGE Models](https://github.com/FlagOpen/FlagEmbedding) - High-quality embedding models
- [SafeTensors](https://github.com/huggingface/safetensors) - Secure model format

## 📚 Related Projects

- [candle](https://github.com/huggingface/candle) - Rust ML framework
- [ort](https://github.com/pykeio/ort) - ONNX Runtime bindings
- [tch](https://github.com/LaurentMazare/tch-rs) - PyTorch bindings

---

**Built with ❤️ in pure Rust** 🦀