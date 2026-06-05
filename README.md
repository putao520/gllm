# gllm — JIT-Compiled LLM Inference Engine in Rust

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A pure Rust inference library for local text generation, embeddings, and reranking. Built on [gllm-kernels](https://github.com/putao520/gllm-kernels) — a JIT-compiled fusion kernel engine that generates hardware-optimal machine code at runtime.

## Core Philosophy

- **JIT-First**: All operators compiled via 4-stage pipeline (Scalar → SymExec → IR → ISA Lowering). No precompiled libraries, no hand-written assembly.
- **Accuracy > Throughput**: Deterministic scheduling, strict causal ordering, phase-isolated prefill/decode.
- **Zero Fallback**: No scalar fallback, no silent degradation. If JIT fails, it errors — never silently produces wrong results.

## Features

- **20+ Model Architectures** — Qwen3, Llama 4, GLM-5, Mistral 3, DeepSeek V3/R1, GPT-OSS, Phi-4, Gemma 4, and more
- **Multi-Format Loader** — SafeTensors (zero-copy), GGUF (21 quantization types), ONNX (graph pattern matching), PyTorch
- **Multi-Source Download** — HuggingFace with automatic ModelScope fallback
- **Fused Kernels** — FlashAttention, SwiGLU, FusedQkvNormRope, MoE routing, RMSNorm-into-GEMM
- **PagedAttention + Continuous Batching** — KV cache as virtual memory pages with prefix sharing
- **Quantization** — 22 quantization types (INT4/INT8/FP4/FP8/AWQ/GPTQ/MXFP4/NVFP4) with JIT dequantization fusion
- **Distributed Inference** — Multi-GPU tensor/pipeline parallelism via [gllm-nccl](https://github.com/putao520/gllm-nccl)
- **Advanced Features** — Semantic Gatekeeper, Head Routing, Guardrail, Intent Recall, CoT Reasoner, MoE, MLA, MTP, AltUp

## Quick Start

### Text Generation

```rust
use gllm::Client;

let client = Client::new_chat("Qwen/Qwen3-7B-Instruct")?;
let response = client
    .generate("Explain the theory of relativity:")
    .max_tokens(512)
    .temperature(0.7)
    .generate()?;
println!("{}", response.text);
```

### Text Embeddings

```rust
use gllm::Client;

let client = Client::new_embedding("intfloat/e5-small-v2")?;
let response = client.embed(["Hello, world!", "Test sentence"])?;
for emb in &response.embeddings {
    println!("dim: {}", emb.embedding.len());
}
```

### Document Reranking

```rust
use gllm::Client;

let client = Client::new_reranker("BAAI/bge-reranker-v3")?;
let response = client
    .rerank("Efficient storage", [
        "Columnar databases compress well.",
        "Rust has zero-cost abstractions.",
    ])
    .top_n(1)?;
println!("Best match score: {:.4}", response.results[0].score);
```

## Supported Models

| Category | Models | Architecture |
|----------|--------|--------------|
| **Generator** | Qwen3 (7B/MoE 235B/Thinking 32B), Llama 4 (8B MoE/Scout 17B), Mistral 3 (14B), GLM-4.7/5, GPT-OSS (1.5B/12B/20B), Phi-4 (14B), Gemma 4 (E2B/E4B/31B/26B-A4B), SmolLM, InternLM3 | Dense / MoE / Thinking |
| **Embedding** | Qwen3-Embed (2048D), BGE-M3 (1024D), BGE-M4 (1536D), E5 (384/768/1024D), M3E, Jina v2/v4 | Bi-encoder |
| **Reranker** | Qwen3-Rerank, BGE-Reranker-v2-m3, BGE-Rerank-v3 | Cross-encoder |

## Backend Support

Auto-detected at runtime, zero configuration:

| Backend | Hardware | JIT Method |
|---------|----------|------------|
| **CUDA** | NVIDIA GPU | PTX codegen, SM version specialized (sm_70/80/90/100+) |
| **ROCm** | AMD GPU | HIP codegen via HSA runtime |
| **Metal** | Apple GPU | MSL codegen via Metal framework |
| **CPU** | x86_64 | AVX2 / AVX-512 / AMX / VNNI / BF16 |
| **CPU** | AArch64 | NEON / SVE / SME2 |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Client API — generate / embed / rerank / classify  │
├─────────────────────────────────────────────────────┤
│  Loader — safetensors / GGUF / ONNX / PyTorch       │
├─────────────────────────────────────────────────────┤
│  Graph Optimizer — pattern fusion + HW constraints   │
├─────────────────────────────────────────────────────┤
│  Scheduler — PagedAttention + Continuous Batching    │
├─────────────────────────────────────────────────────┤
│  Executor — Mega-Kernel block routing                │
├─────────────────────────────────────────────────────┤
│  gllm-kernels — JIT: Scalar→SymExec→IR→ISA           │
├─────────────────────────────────────────────────────┤
│  gllm-nccl — Distributed: NCCL/RCCL/oneCCL          │
└─────────────────────────────────────────────────────┘
```

## Triple-Repo Architecture

| Repository | Role | Dependency |
|------------|------|------------|
| [gllm](https://github.com/putao520/gllm) | Inference client, scheduling, model loading | → gllm-kernels |
| [gllm-kernels](https://github.com/putao520/gllm-kernels) | JIT compiler, codegen, operator registry | → gllm-nccl |
| [gllm-nccl](https://github.com/putao520/gllm-nccl) | Distributed GPU communication | Standalone |

## License

MIT License
