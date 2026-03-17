# gllm: Pure Rust JIT Inference Engine

[![Crates.io](https://img.shields.io/crates/v/gllm.svg)](https://crates.io/crates/gllm)
[![Documentation](https://docs.rs/gllm/badge.svg)](https://docs.rs/gllm)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**gllm** is a pure Rust inference library for local text generation, embeddings, and reranking. Built on [gllm-kernels](https://github.com/anthropics/gllm-kernels) — a JIT-compiled fusion kernel engine that generates hardware-optimal machine code at runtime for x86_64 (AVX2/AVX-512/AMX), AArch64 (NEON/SVE), and GPU (CUDA/ROCm/Metal).

## Features

- **JIT-First Architecture** — All operators compiled via 4-stage pipeline (Scalar → SymExec → IR → ISA Lowering), no precompiled libraries
- **Accuracy > Throughput** — Deterministic scheduling, strict causal ordering, phase-isolated prefill/decode
- **20+ Model Architectures** — Qwen3, Llama 4, GLM-5, Mistral 3, GPT-OSS, Phi-4, Gemma2, and more
- **Multi-Format Loader** — SafeTensors (zero-copy), GGUF (21 quantization types), ONNX, PyTorch
- **Multi-Source Download** — HuggingFace with automatic ModelScope fallback
- **Fused Kernels** — FlashAttention, SwiGLU, FusedQkvRope, MoE routing, FusedRMSLinear
- **PagedAttention + Continuous Batching** — KV cache as virtual memory pages with prefix sharing
- **Pure Rust** — No `libcudart`, `libhip`, or C++ build dependencies

## Installation

```toml
[dependencies]
gllm = "0.11"
```

### Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `cuda` | No | NVIDIA GPU backend (PTX JIT, sm_70/80/90/100+) |
| `hip` | No | AMD GPU backend (HIP codegen) |
| `metal` | No | Apple GPU backend (MSL codegen) |
| `tokio` | No | Async interface support |
| `paged-attention` | No | PagedAttention KV cache management |
| `flash-attention` | No | FlashAttention v2 tiled attention |

## Quick Start

### Text Generation

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

### Text Embeddings

```rust
use gllm::Client;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new_embedding("Qwen/Qwen3-Embedding")?;

    let response = client
        .embeddings(["Future of AI", "Rust programming"])
        .generate()?;

    for emb in &response.embeddings {
        println!("Vector dim: {}", emb.embedding.len());
    }
    Ok(())
}
```

### Document Reranking

```rust
use gllm::{Client, ModelKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

## Supported Models

Latest generation only. See [SPEC/SUPPORTED_MODELS.md](SPEC/SUPPORTED_MODELS.md) for full details.

| Category | Models | Architecture |
|----------|--------|--------------|
| **Generator** | Qwen3 (7B, MoE 235B, Thinking 32B), Llama 4 (8B MoE, Scout 17B), Mistral 3 (14B), GLM-4.7/5, GPT-OSS (1.5B/12B), Phi-4 (14B), SmolLM, InternLM3, Gemma2 | Dense / MoE / Thinking |
| **Embedding** | Qwen3-Embed (2048D), BGE-M3 (1024D), BGE-M4 (1536D), E5 (384/768/1024D), M3E, Jina v2/v4 | Bi-encoder |
| **Reranker** | Qwen3-Rerank, BGE-Reranker-v2-m3, BGE-Rerank-v3 | Cross-encoder |

## Backend Support

Auto-detected at runtime, zero configuration:

| Backend | Hardware | Method |
|---------|----------|--------|
| **CUDA** | NVIDIA GPU | JIT PTX codegen, SM 版本特化 (sm_70/80/90/100+) |
| **ROCm** | AMD GPU | JIT HIP codegen via `libhsa-runtime64.so` |
| **Metal** | Apple GPU | JIT MSL codegen via `Metal.framework` |
| **CPU** | x86_64 / AArch64 | JIT SIMD codegen (AVX2/AVX-512/AMX/NEON/SVE) |

## Architecture

```
┌─────────────────────────────────────────────────┐
│  Layer 1: Manifest (Static Model Definition)    │
│  config.json / GGUF metadata / ONNX graph       │
├─────────────────────────────────────────────────┤
│  Layer 2: Adapter (Logic Adaptation)            │
│  ModelAdapter / WeightMapper / TokenizerAdapter  │
├─────────────────────────────────────────────────┤
│  Layer 3: Engine (Runtime Scheduling)           │
│  Executor / PagedAttention / ContinuousBatcher   │
├─────────────────────────────────────────────────┤
│  Layer 4: Driver (Hardware Execution)           │
│  gllm-kernels JIT: Scalar→SymExec→IR→ISA        │
└─────────────────────────────────────────────────┘
```

## License

MIT License
