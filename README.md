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

## 📦 Installation

### Requirements

- **Rust 1.70+** (2021 edition)
- **Memory** - Minimum 2GB RAM, 4GB+ recommended for larger models
- **GPU (Optional)** - For faster inference with WGPU backend

### Step 1: Add to Cargo.toml

```toml
[dependencies]
gllm = "0.2"
```

### Step 2: Choose Your Backend

```toml
# Option 1: Default (WGPU GPU support + CPU fallback)
gllm = "0.2"

# Option 2: CPU-only (no GPU dependencies, pure Rust)
gllm = { version = "0.2", features = ["cpu"] }

# Option 3: With async support (tokio)
gllm = { version = "0.2", features = ["tokio"] }

# Option 4: CPU-only + async
gllm = { version = "0.2", features = ["cpu", "tokio"] }
```

### Step 3: Start Using (5 minutes)

See the Quick Start section below.

## ℹ️ Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wgpu` | ✅ | GPU acceleration using WGPU (Vulkan/DX12/Metal) |
| `cpu` | ❌ | CPU-only inference using ndarray (pure Rust) |
| `tokio` | ❌ | Async interface support (same API, add `.await`) |

### GPU Support

The WGPU backend supports:
- **NVIDIA GPUs** - via Vulkan or CUDA
- **AMD GPUs** - via Vulkan or DirectX
- **Intel GPUs** - via Vulkan or DirectX
- **Apple Silicon** - via Metal
- **Intel/AMD CPUs** - Fallback to CPU compute

## 🎯 Supported Models

gllm includes built-in aliases for **26 popular models** and supports any HuggingFace SafeTensors model.

### Built-in Model Aliases (26 Models)

#### 🔄 Text Embedding Models (18 models)

| Alias | HuggingFace Model | Dimensions | Speed | Best For |
|-------|------------------|------------|-------|----------|
| **BGE Series** | | | | |
| `bge-small-zh` | `BAAI/bge-small-zh-v1.5` | 512 | Fast | 🇨🇳 Chinese, lightweight |
| `bge-small-en` | `BAAI/bge-small-en-v1.5` | 384 | Fast | 🇺🇸 English, lightweight |
| `bge-base-en` | `BAAI/bge-base-en-v1.5` | 768 | Medium | 🇺🇸 English balanced |
| `bge-large-en` | `BAAI/bge-large-en-v1.5` | 1024 | Slow | 🇺🇸 English high accuracy |

| **Sentence Transformers** | | | | |
| `all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Fast | 🎯 General purpose |
| `all-mpnet-base-v2` | `sentence-transformers/all-mpnet-base-v2` | 768 | Medium | 🎯 High quality English |
| `paraphrase-MiniLM-L6-v2` | `sentence-transformers/paraphrase-MiniLM-L6-v2` | 384 | Fast | 🔄 Paraphrase detection |
| `multi-qa-mpnet-base-dot-v1` | `sentence-transformers/multi-qa-mpnet-base-dot-v1` | 768 | Medium | ❓ Question answering |
| `all-MiniLM-L12-v2` | `sentence-transformers/all-MiniLM-L12-v2` | 384 | Fast | 🎯 General purpose (larger) |
| `all-distilroberta-v1` | `sentence-transformers/all-distilroberta-v1` | 768 | Medium | ⚡ Fast inference |

| **E5 Series** | | | | |
| `e5-large` | `intfloat/e5-large` | 1024 | Slow | 🎯 Instruction tuned |
| `e5-base` | `intfloat/e5-base` | 768 | Medium | 🎯 Instruction tuned |
| `e5-small` | `intfloat/e5-small` | 384 | Fast | ⚡ Lightweight instruction tuned |

| **JINA Embeddings** | | | | |
| `jina-embeddings-v2-base-en` | `jinaai/jina-embeddings-v2-base-en` | 768 | Medium | 🎯 Modern architecture |
| `jina-embeddings-v2-small-en` | `jinaai/jina-embeddings-v2-small-en` | 384 | Fast | ⚡ Lightweight modern |

| **Chinese Models** | | | | |
| `m3e-base` | `moka-ai/m3e-base` | 768 | Medium | 🇨🇳 Chinese, high quality |

| **Multilingual** | | | | |
| `multilingual-MiniLM-L12-v2` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Medium | 🌍 50+ languages |
| `distiluse-base-multilingual-cased-v1` | `sentence-transformers/distiluse-base-multilingual-cased-v1` | 512 | Medium | 🌍 Multilingual cased |

#### 🎯 Document Reranking Models (8 models)

| Alias | HuggingFace Model | Speed | Best For |
|-------|------------------|-------|----------|
| **BGE Rerankers** | | | |
| `bge-reranker-v2` | `BAAI/bge-reranker-v2-m3` | Medium | 🌍 Multilingual reranking |
| `bge-reranker-large` | `BAAI/bge-reranker-large` | Slow | 🎯 High accuracy |
| `bge-reranker-base` | `BAAI/bge-reranker-base` | Fast | ⚡ Fast reranking |

| **MS MARCO Rerankers** | | | |
| `ms-marco-MiniLM-L-6-v2` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast | 🎯 Search relevance |
| `ms-marco-MiniLM-L-12-v2` | `cross-encoder/ms-marco-MiniLM-L-12-v2` | Medium | 🎯 Higher accuracy search |
| `ms-marco-TinyBERT-L-2-v2` | `cross-encoder/ms-marco-TinyBERT-L-2-v2` | Very Fast | ⚡ Lightweight reranking |
| `ms-marco-electra-base` | `cross-encoder/ms-marco-electra-base` | Medium | ⚡ Efficient reranking |

| **Specialized Rerankers** | | | |
| `quora-distilroberta-base` | `cross-encoder/quora-distilroberta-base` | Medium | ❓ Question similarity |

### Using Custom Models

You can use any HuggingFace SafeTensors model directly:

```rust
// Use any HuggingFace SafeTensors model
let client = Client::new("sentence-transformers/all-MiniLM-L6-v2")?;

// Or use colon notation for shorthand
let client = Client::new("sentence-transformers:all-MiniLM-L6-v2")?;
```

### 🎛️ Model Selection Guide

#### Embedding Models - Choose Based On:

**🚀 Speed & Efficiency**
- `bge-small-en` / `e5-small` / `all-MiniLM-L6-v2` - Fastest, 384 dims
- Perfect for high-throughput applications

**⚖️ Balance of Speed & Accuracy**
- `bge-base-en` / `e5-base` / `all-mpnet-base-v2` - 768 dims
- Great general-purpose choice

**🎯 High Accuracy**
- `bge-large-en` / `e5-large` - 1024 dims
- Best for quality-critical applications

**🌍 Multilingual & Chinese Support**
- `bge-small-zh` - 512 dims, Chinese optimized
- `m3e-base` - 768 dims, Chinese high quality
- `multilingual-MiniLM-L12-v2` - 384 dims, 50+ languages

#### Reranking Models - Choose Based On:

**⚡ Fast Reranking**
- `bge-reranker-base` / `ms-marco-TinyBERT-L-2-v2`
- Best for real-time applications

**🎯 Balanced Performance**
- `bge-reranker-v2` / `ms-marco-MiniLM-L-6-v2`
- Good accuracy with reasonable speed

**🏆 High Accuracy**
- `bge-reranker-large` / `ms-marco-MiniLM-L-12-v2`
- Maximum quality for batch processing

### Model Requirements

- **Embedding Models**: BERT-style encoder models with SafeTensors weights
- **Rerank Models**: Cross-encoder models with SafeTensors weights
- **Format**: SafeTensors (`.safetensors` files)
- **Tokenizer**: HuggingFace compatible tokenizer files

## 🚀 Quick Start

### Text Embeddings

Generate semantic embeddings for search, clustering, or similarity matching:

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client with built-in model (auto-downloaded from HuggingFace)
    let client = Client::new("bge-small-en")?;

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

For async applications, enable the `tokio` feature:

```toml
[dependencies]
gllm = { version = "0.2", features = ["tokio"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

```rust
use gllm::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Same API as sync, just add .await
    let client = Client::new("bge-small-en").await?;

    // Embeddings
    let response = client
        .embeddings(["Hello world", "Async programming"])
        .generate()
        .await?;

    for embedding in response.embeddings {
        println!("Vector: {} dimensions", embedding.embedding.len());
    }

    // Reranking
    let client = Client::new("bge-reranker-v2").await?;

    let response = client
        .rerank("What is machine learning?", [
            "Machine learning is a subset of AI",
            "Python is a programming language",
            "Deep learning uses neural networks",
        ])
        .top_n(2)
        .generate()
        .await?;

    for result in response.results {
        println!("Score: {:.4}, Index: {}", result.score, result.index);
    }

    Ok(())
}
```

## 🔧 Advanced Usage

### Custom Configuration

```rust
use gllm::{Client, ClientConfig, Device};

let config = ClientConfig {
    models_dir: "/custom/model/path".into(),        // Override model storage (default: ~/.gllm/models)
    device: Device::Auto,                           // Auto-select GPU/CPU (or Device::Cpu, Device::Gpu)
};

let client = Client::with_config("bge-small-en", config)?;
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
| `tokio` | ❌ | Async interface support (same API, add `.await`) |

### Build Examples

```bash
# Default (GPU + CPU fallback)
cargo build --release

# CPU-only
cargo build --release --features cpu

# Async support
cargo build --release --features "wgpu,tokio"

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

Run the complete test suite:

```bash
# Unit tests only
cargo test --lib

# Integration tests (fast, no model downloads needed)
cargo test --test integration

# E2E tests with real model downloads and inference (requires models in ~/.gllm/models/)
cargo test --test integration -- --ignored

# All tests including E2E
cargo test -- --include-ignored

# Verbose output
cargo test -- --nocapture
```

### Test Coverage

- ✅ 10 unit tests (model configs, registry, pooling)
- ✅ 14 integration tests (API, error handling, features)
- ✅ 8 E2E tests (real model downloads and inference for all 26 models)

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