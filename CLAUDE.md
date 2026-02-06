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

### 0. Core Philosophy: Accuracy First (准确度优先)
> **🚨 Critical Difference from vLLM**:
> vLLM and similar frameworks often prioritize throughput by using out-of-order execution in continuous batching, approximate attention masks, or aggressive quantization. This can degrade inference accuracy, especially in long-context or complex instruction-following scenarios.
>
> **gllm Principles**:
> 1.  **Accuracy > Throughput**: Never sacrifice calculation precision for scheduling optimization.
> 2.  **Strict Causal Ordering**: Intra-batch attention computation must guarantee strict causal masking. No out-of-order execution that risks context drift.
> 3.  **Reliability First**: Memory management (PagedAttention) must have strict boundary checks and error recovery. Prefer OOM rejection over returning corrupted results.
> 4.  **Deterministic Scheduling**: To combat floating-point non-associativity, batches must be strictly ordered (e.g., by RequestId). We prefer deterministic serial execution over messy parallel reduction.

### 1. Backend Constraints (from gllm-kernels)
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

### 3. Fused-First Architecture / 融合优先原则
- **Constraint**: 调度/执行层必须优先选择融合算子 (Fused Kernels)。仅在无法匹配融合模式时，才降级使用原子算子 (Atomic Kernels)。
- **Constraint**: ONNX Loader 必须实现 Graph Pattern Matching，将子图映射为 Fused Kernels，严禁 naive 的 1:1 翻译。

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

## 🧪 E2E 测试约束

**单线程运行强制要求**：
- ❌ E2E 测试禁止并行运行（`--test-threads=1` 强制）
- ✅ E2E 测试必须串行执行，避免资源竞争

**理由**：E2E 测试涉及真实模型下载、文件 I/O、CPU 推理，并行运行会导致：
- 磁盘 I/O 竞争
- 模型缓存冲突
- CPU 内存超限
- 测试结果不可预测

**运行命令**：
```bash
# 正确：E2E 测试单线程运行
cargo test --test test_e2e -- --test-threads=1
cargo test --test test_real_models -- --test-threads=1

# 单元测试可以并行（不涉及真实 I/O）
cargo test --lib
```
