# gllm

**Inference Client** - High-level library for model management, scheduling, and engine orchestration.

> **🚨 TABULA RASA (2026-02)**: This project has been reset.

## SPEC Location
- `./SPEC/` (Single Source of Truth)
- `../gllm-kernels/SPEC/` (Backend constraints)

## Technology Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Loader** | `hf-hub`, `safetensors` | Model fetching and zero-copy loading |
| **Tokenizer** | `tokenizers` | Text <-> ID conversion |
| **Scheduler** | Custom | PagedAttention & Continuous Batching |
| **Engine** | `gllm-kernels` | Hardware abstraction layer |

## Core Architecture

### 0. Backend Constraints (from gllm-kernels)
- **Quantization**: Template-based kernels (1/2/4/8-bit unified)
- **GPU Execution**: L3 GPU-Pure API (zero-copy generation loop)
- **AOT Only**: Pre-compiled `.cubin` files, no PTX JIT

### 1. Data Flow (Zero-Copy)
- **Loader**: Maps `safetensors` directly to memory.
- **Upload**: Pushes raw bytes to `gllm-kernels` (GPU) or maps them for CPU.
- **Inference**: Orchestrates the `gllm-kernels` L3 API.

### 2. Smart Scheduling
- **Double Buffering**: Pre-allocate next batch while current batch computes.
- **PagedAttention**: Manage KV cache as virtual memory pages.

## Directory Structure

```
src/
├── lib.rs          # Library entry
├── loader/         # Model fetching & parsing (HF/SafeTensors)
│   └── mod.rs
├── scheduler/      # Batching & KV Cache management
│   └── mod.rs
└── engine/         # Execution engine (wraps gllm-kernels)
    └── mod.rs
```

## Common Commands

```bash
cargo check
cargo test
```
