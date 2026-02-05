# gllm: Pure Rust Next-Gen Inference Engine (2026)

[![Crates.io](https://img.shields.io/crates/v/gllm.svg)](https://crates.io/crates/gllm)
[![Documentation](https://docs.rs/gllm/badge.svg)](https://docs.rs/gllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**gllm** is a high-performance, pure Rust library for local text embeddings, reranking, and text generation. It is built on [gllm-kernels](https://github.com/anthropics/gllm-kernels) for **Driver API-based** native GPU acceleration (CUDA/Metal/ROCm) without requiring external SDKs.

> **Targeting 2026 SOTA**: Optimized for Qwen3, Llama 4, and GLM-5 architectures.

## Features

- **L3 GPU-Pure Architecture** - Zero-copy inference loop (only token IDs transfer between CPU/GPU).
- **Next-Gen Models** - Native support for **Qwen3 (Thinking)**, **Llama 4 (MoE)**, **GLM-5**, and **Mistral 3**.
- **Unified Driver API** - AOT compiled kernels (CUBIN) with no runtime compilation (JIT) overhead.
- **Tree Attention** - Native support for speculative decoding topologies (EAGLE-2 / Medusa-2).
- **Multi-Source Loader** - Auto-switching between HuggingFace and ModelScope (China mirror).
- **Quantization** - Block-wise Int4/Int8 support with hand-written SIMD kernels.
- **Pure Rust** - Static compilation ready, no `libcudart`, `libhip`, or C++ build dependencies.

## Installation

```toml
[dependencies]
gllm = "0.11"
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `tokio` | No | Async interface support |

## Quick Start

### Text Generation (Qwen3)

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Automatically downloads from HF or ModelScope
    // Qwen3-7B (2026 SOTA)
    let client = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;

    let response = client
        .generate("Explain the theory of relativity:")
        .max_tokens(512)
        .temperature(0.7)
        .generate()?;

    println!("{}", response.text);
    Ok(())
}
```

### Text Embeddings (Next-Gen)

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Qwen3-Embedding (2048 dims)
    let client = Client::new_embedding("Qwen/Qwen3-Embedding")?;

    let response = client
        .embeddings(["Future of AI", "Rust programming"])
        .generate()?;

    for emb in response.embeddings {
        println!("Vector dim: {}", emb.embedding.len());
    }
    Ok(())
}
```

### Document Reranking

```rust
use gllm::{Client, ModelKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // BGE-Reranker-v3 (XLM-R-Next architecture)
    let client = Client::new("BAAI/bge-reranker-v3", ModelKind::Reranker)?;

    let response = client
        .rerank("Efficient storage", [
            "Columnar databases compress well.",
            "Rust has zero-cost abstractions.",
        ])
        .top_n(1)
        .generate()?;

    println!("Best match score: {:.4}", response.results[0].score);
    Ok(())
}
```

## Supported Models (2026 SOTA Only)

We strictly support the latest generation of models. Legacy models (Qwen2.5, Llama 3) are deprecated.

| Category | Model ID | Architecture | Specs |
|----------|----------|--------------|-------|
| **Generator** | `qwen3-7b` | Qwen3 | 7B Dense |
| | `qwen3-moe` | Qwen3MoE | A22B (235B) |
| | `llama-4-8b` | Llama4 | 8B MoE |
| | `glm-5-9b` | GLM-5 | 9B Dense |
| | `mistral-small-3` | Mistral3 | 14B |
| **Embedding** | `qwen3-embed` | Qwen3 | 2048 dims |
| | `bge-m4` | XLM-R-Next | 1536 dims |
| **Rerank** | `qwen3-rerank` | Qwen3 | - |
| | `bge-rerank-v3` | XLM-R-Next | - |

## Backend Support

The system automatically detects the best available hardware at runtime (Zero Config):

1.  **CUDA** (NVIDIA): Direct `libcuda.so` loading. AOT Kernels for sm_80/86/89/90.
2.  **Metal** (Apple): Native `Metal.framework` binding.
3.  **ROCm** (AMD): Direct `libhsa-runtime64.so` loading.
4.  **CPU** (Fallback): `faer` (AVX-512/NEON) pure Rust SIMD.

## License

MIT License
