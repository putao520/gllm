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

## Cache Directory

**Model Cache Root**: `~/.gllm/models/`

下载的模型文件存储在此目录。内部子目录结构由下载库（hf-hub/ModelScope）管理。

**Environment Variables**:
- `GLLM_CACHE_DIR`: 自定义缓存路径（默认：`~/.gllm/models`）
- `HF_TOKEN`: HuggingFace 认证 token

> **自动回退**: HuggingFace 下载失败时会自动切换到 ModelScope，无需手动指定来源。

## 🚨 环境变量铁律

**禁止擅自添加环境变量**：
- ❌ AI 禁止擅自引入新的环境变量
- ❌ AI 禁止为"灵活性"或"可配置性"添加环境变量
- ✅ 新增环境变量必须由用户明确要求
- ✅ 环境变量必须有明确的、不可替代的使用场景

**理由**：环境变量是公开 API，添加容易删除难。过度使用环境变量会导致配置爆炸和用户体验下降。

## Common Commands

```bash
cargo check
cargo test
```
