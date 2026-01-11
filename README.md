# gllm: Pure Rust Local Embeddings & Reranking

[![Crates.io](https://img.shields.io/crates/v/gllm.svg)](https://crates.io/crates/gllm)
[![Documentation](https://docs.rs/gllm/badge.svg)](https://docs.rs/gllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**gllm** is a pure Rust library for local text embeddings and reranking, built on the [Burn](https://github.com/tracel-ai/burn) deep learning framework. It provides an OpenAI SDK-style API with zero external C dependencies, supporting static compilation.

## Features

- **Text Embeddings** - Convert text into high-dimensional vectors for semantic search
- **Document Reranking** - Sort documents by relevance using cross-encoders
- **Code Embeddings** - Specialized models for code semantic similarity (CodeXEmbed)
- **GPU Acceleration** - WGPU backend with automatic GPU/CPU fallback
- **40+ Built-in Models** - BGE, E5, Sentence Transformers, Qwen3, JINA, CodeXEmbed, and more
- **Encoder & Decoder Architectures** - BERT-style encoders and Qwen2/Mistral-style decoders
- **Quantization Support** - Int4/Int8/AWQ/GPTQ/GGUF for Qwen3 series
- **Pure Rust** - Static compilation ready, no C dependencies

## Installation

```toml
[dependencies]
gllm = "0.5"
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `wgpu` | Yes | GPU acceleration (Vulkan/DX12/Metal) |
| `cpu` | No | CPU-only inference (pure Rust) |
| `tokio` | No | Async interface support |
| `wgpu-detect` | No | GPU capabilities detection (VRAM, batch size) |

```toml
# CPU-only
gllm = { version = "0.4", features = ["cpu"] }

# With async
gllm = { version = "0.4", features = ["tokio"] }

# With GPU detection
gllm = { version = "0.4", features = ["wgpu-detect"] }
```

### Requirements

- **Rust 1.70+** (2021 edition)
- **Memory**: 2GB minimum, 4GB+ recommended
- **GPU (optional)**: Vulkan, DirectX 12, Metal, or OpenGL 4.3+

## Quick Start

### Text Embeddings

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("bge-small-en")?;

    let response = client
        .embeddings(["What is machine learning?", "Neural networks explained"])
        .generate()?;

    for emb in response.embeddings {
        println!("Vector: {} dimensions", emb.embedding.len());
    }
    Ok(())
}
```

### Document Reranking

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("bge-reranker-v2")?;

    let response = client
        .rerank("What are renewable energy benefits?", [
            "Solar power is clean and sustainable.",
            "The stock market closed higher today.",
            "Wind energy reduces carbon emissions.",
        ])
        .top_n(2)
        .return_documents(true)
        .generate()?;

    for result in response.results {
        println!("Score: {:.4}", result.score);
    }
    Ok(())
}
```

### Async Usage

```toml
[dependencies]
gllm = { version = "0.4", features = ["tokio"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

```rust
use gllm::Client;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new("bge-small-en").await?;

    let response = client
        .embeddings(["Hello world"])
        .generate()
        .await?;

    Ok(())
}
```

### GPU Detection (v0.4.1+)

```rust
use gllm::{GpuCapabilities, GpuType};

// Detect GPU capabilities (cached after first call)
let caps = GpuCapabilities::detect();

println!("GPU: {} ({:?})", caps.name, caps.gpu_type);
println!("VRAM: {} MB", caps.vram_mb);
println!("Recommended batch size: {}", caps.recommended_batch_size);

if caps.gpu_available {
    println!("Using {} backend", caps.backend_name);
}
```

### FallbackEmbedder (Automatic GPU/CPU Fallback)

```rust
use gllm::FallbackEmbedder;

// Automatically falls back to CPU if GPU OOMs
let embedder = FallbackEmbedder::new("bge-small-en").await?;
let vector = embedder.embed("Hello world").await?;
```

### Code Embeddings (v0.5.0+)

CodeXEmbed models are optimized for code semantic similarity, outperforming Voyage-Code by 20%+ on CoIR benchmark.

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CodeXEmbed-400M (1024 dimensions, BERT-based)
    let client = Client::new("codexembed-400m")?;

    let code_snippets = [
        "fn add(a: i32, b: i32) -> i32 { a + b }",
        "def add(a, b): return a + b",
        "function add(a, b) { return a + b; }",
    ];

    let response = client.embeddings(code_snippets).generate()?;

    // All 3 add functions will have high similarity scores
    for emb in response.embeddings {
        println!("Vector: {} dimensions", emb.embedding.len());
    }
    Ok(())
}
```

For larger models with higher accuracy:

```rust
// CodeXEmbed-2B (1536 dimensions, Qwen2-based decoder)
let client = Client::new("codexembed-2b")?;

// CodeXEmbed-7B (4096 dimensions, Mistral-based decoder)
let client = Client::new("codexembed-7b")?;
```

## Supported Models

### Embedding Models (27)

| Model | Alias | Dimensions | Architecture | Best For |
|-------|-------|------------|--------------|----------|
| BGE Small EN | `bge-small-en` | 384 | Encoder | Fast English |
| BGE Base EN | `bge-base-en` | 768 | Encoder | Balanced English |
| BGE Large EN | `bge-large-en` | 1024 | Encoder | High accuracy |
| BGE Small ZH | `bge-small-zh` | 512 | Encoder | Chinese |
| E5 Small | `e5-small` | 384 | Encoder | Instruction tuned |
| E5 Base | `e5-base` | 768 | Encoder | Instruction tuned |
| E5 Large | `e5-large` | 1024 | Encoder | Instruction tuned |
| MiniLM L6 | `all-MiniLM-L6-v2` | 384 | Encoder | General purpose |
| MiniLM L12 | `all-MiniLM-L12-v2` | 384 | Encoder | General (larger) |
| MPNet Base | `all-mpnet-base-v2` | 768 | Encoder | High quality |
| JINA v2 Base | `jina-embeddings-v2-base-en` | 768 | Encoder | Modern arch |
| JINA v2 Small | `jina-embeddings-v2-small-en` | 384 | Encoder | Lightweight |
| JINA v4 | `jina-embeddings-v4` | 2048 | Encoder | Latest JINA |
| Qwen3 0.6B | `qwen3-embedding-0.6b` | 1024 | Encoder | Lightweight |
| Qwen3 4B | `qwen3-embedding-4b` | 2560 | Encoder | Balanced |
| Qwen3 8B | `qwen3-embedding-8b` | 4096 | Encoder | High accuracy |
| Nemotron 8B | `llama-embed-nemotron-8b` | 4096 | Encoder | State-of-the-art |
| M3E Base | `m3e-base` | 768 | Encoder | Chinese quality |
| Multilingual | `multilingual-MiniLM-L12-v2` | 384 | Encoder | 50+ languages |

### Code Embedding Models (4) - NEW in v0.5.0

| Model | Alias | Dimensions | Architecture | Best For |
|-------|-------|------------|--------------|----------|
| CodeXEmbed 400M | `codexembed-400m` | 1024 | Encoder (BERT) | Fast code search |
| CodeXEmbed 2B | `codexembed-2b` | 1536 | Decoder (Qwen2) | Balanced code |
| CodeXEmbed 7B | `codexembed-7b` | 4096 | Decoder (Mistral) | High accuracy code |
| GraphCodeBERT | `graphcodebert-base` | 768 | Encoder | Legacy code |

> **CodeXEmbed** (SFR-Embedding-Code) is the 2024 state-of-the-art for code embedding, outperforming Voyage-Code by 20%+ on CoIR benchmark.

### Reranking Models (12)

| Model | Alias | Speed | Best For |
|-------|-------|-------|----------|
| BGE Reranker v2 | `bge-reranker-v2` | Medium | Multilingual |
| BGE Reranker Large | `bge-reranker-large` | Slow | High accuracy |
| BGE Reranker Base | `bge-reranker-base` | Fast | Quick reranking |
| MS MARCO MiniLM L6 | `ms-marco-MiniLM-L-6-v2` | Fast | Search |
| MS MARCO MiniLM L12 | `ms-marco-MiniLM-L-12-v2` | Medium | Better search |
| MS MARCO TinyBERT | `ms-marco-TinyBERT-L-2-v2` | Very Fast | Lightweight |
| Qwen3 Reranker 0.6B | `qwen3-reranker-0.6b` | Fast | Lightweight |
| Qwen3 Reranker 4B | `qwen3-reranker-4b` | Medium | Balanced |
| Qwen3 Reranker 8B | `qwen3-reranker-8b` | Slow | High accuracy |
| JINA Reranker v3 | `jina-reranker-v3` | Medium | Latest JINA |

### Custom Models

```rust
// Any HuggingFace SafeTensors model
let client = Client::new("sentence-transformers/all-MiniLM-L6-v2")?;

// Or use colon notation
let client = Client::new("sentence-transformers:all-MiniLM-L6-v2")?;
```

## Quantization (Qwen3 Series)

```rust
use gllm::ModelRegistry;

let registry = ModelRegistry::new();

// Use :suffix for quantized variants
let info = registry.resolve("qwen3-embedding-8b:int4")?;  // Int4
let info = registry.resolve("qwen3-embedding-8b:awq")?;   // AWQ
let info = registry.resolve("qwen3-reranker-4b:gptq")?;   // GPTQ
```

**Supported quantization types**: `:int4`, `:int8`, `:awq`, `:gptq`, `:gguf`, `:fp8`, `:bnb4`, `:bnb8`

**Models with quantization**: Qwen3 Embedding/Reranker series, Nemotron 8B

## Advanced Usage

### Custom Configuration

```rust
use gllm::{Client, ClientConfig, Device};

let config = ClientConfig {
    models_dir: "/custom/path".into(),
    device: Device::Auto,  // or Device::Cpu, Device::Gpu
};

let client = Client::with_config("bge-small-en", config)?;
```

### Vector Search Example

```rust
let query_vec = client.embeddings(["search query"]).generate()?.embeddings[0].embedding.clone();
let doc_vecs = client.embeddings(documents).generate()?;

// Calculate cosine similarities
for (i, doc) in doc_vecs.embeddings.iter().enumerate() {
    let sim = cosine_similarity(&query_vec, &doc.embedding);
    println!("Doc {}: {:.4}", i, sim);
}
```

## Model Storage

Models are cached in `~/.gllm/models/`:

```
~/.gllm/models/
├── BAAI--bge-small-en-v1.5/
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
└── ...
```

## Performance

| Backend | Device | Throughput (512 tokens) |
|---------|--------|-------------------------|
| WGPU | RTX 4090 | ~150 texts/sec |
| WGPU | Apple M2 | ~45 texts/sec |
| CPU | Intel i7-12700K | ~8 texts/sec |

## Testing

```bash
cargo test --lib              # Unit tests
cargo test --test integration # Integration tests
cargo test -- --ignored       # E2E tests (downloads models)
```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [Burn Framework](https://github.com/tracel-ai/burn)
- [HuggingFace](https://huggingface.co/)
- [BGE Models](https://github.com/FlagOpen/FlagEmbedding)

---

**Built with Rust**
