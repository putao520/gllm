# gllm

**Inference Client** - High-level library for model management, scheduling, and engine orchestration.

> **🚨 TABULA RASA (2026-02)**: This project has been reset.

## SPEC Location
- `./SPEC/` (Single Source of Truth)
- `../gllm-kernels/SPEC/` (Backend constraints)

## Technology Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Loader** | `hf-hub`, `safetensors`, `prost`, `memmap2` | Model fetching, zero-copy loading, ONNX/GGUF parsing |
| **Tokenizer** | `tokenizers` | Text <-> ID conversion |
| **Scheduler** | Custom (HGAL) | PagedAttention & Continuous Batching |
| **Engine** | `gllm-kernels` + `compat` shim | Hardware abstraction layer (compat bridges types) |
| **Graph** | Custom DAG + `serde_yaml` | Unified OnnxGraph representation & optimization |
| **Backend** | `gllm-kernels` | Auto-detect CUDA/ROCm/Metal/CPU |

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

### 2. Data Flow (Zero-Copy)
- **Loader**: Maps `safetensors` directly to memory.
- **Upload**: Pushes raw bytes to `gllm-kernels` (GPU) or maps them for CPU.
- **Inference**: Orchestrates the `gllm-kernels` L3 API.

### 3. Smart Scheduling
- **Double Buffering**: Pre-allocate next batch while current batch computes.
- **PagedAttention**: Manage KV cache as virtual memory pages.

### 4. Fused-First Architecture / 融合优先原则
- **Constraint**: 调度/执行层必须优先选择融合算子 (Fused Kernels)。仅在无法匹配融合模式时，才降级使用原子算子 (Atomic Kernels)。
- **Constraint**: ONNX Loader 必须实现 Graph Pattern Matching，将子图映射为 Fused Kernels，严禁 naive 的 1:1 翻译。

### 5. JIT 编译融合算子推理引擎（核心技术基础）

gllm 的推理引擎以 JIT 编译为技术基础，根据当前设备最佳 ISA/ISV（Instruction Set Architecture / Vector），结合模型参数动态生成最佳性能的融合算子机器码。

**设计目标**：不依赖预编译的算子库，而是在运行时根据 (模型结构 × 硬件能力) 的笛卡尔积生成最优代码。

**四阶段管线** (`gllm-kernels/src/compiler/`):

| 阶段 | 模块 | 职责 |
|------|------|------|
| Phase 0: Scalar + SymExec | `registry.rs` + `symexec/` | 标量参考实现 → 符号执行提取 `OpTrace` + `ComputePattern` |
| Phase 1: SemanticDAG | `semantic_dag.rs` | CompilerGraph + OpTrace → OpClass 自动分类 (ElemWise/Injective/Reduction/Gemm/Opaque) |
| Phase 2: Fusion + HW | `fusion.rs` + `hw_constraints.rs` | 融合决策 (EpilogueInjection/LoopFusion/TileLevelFusion/QkvSharedInput/NormIntoGemm) + 缓存层级约束 |
| Phase 3: ISA Lowering | `codegen/x86_64.rs` / `aarch64_dynasm.rs` | DeviceProfile 驱动的 SIMD 代码生成 (AVX2/AVX-512/NEON/SVE) |

**融合模式** (gllm-kernels 层):
- `EpilogueInjection`: GEMM + 激活/bias/残差 融合到累加器寄存器
- `LoopFusion`: 逐元素算子链合并为单循环
- `TileLevelFusion`: 前驱算子嵌入 GEMM MC 循环（输出 > 75% L1 时）
- `ComputeRoot`: 前驱算子完整计算后驻留 L1/L2（输出 ≤ 75% L1 时）
- `QkvSharedInput`: Q/K/V 三个 GEMM 共享 pack_a
- `NormIntoGemm`: RmsNorm 输出直接喂入 GEMM（无中间写回）

**融合模式** (gllm 图优化层, `src/graph/optimizer/`):
- FlashAttention / GQA / FusedQkvRope / SwiGLU / MoERouting / FusedRMSLinear
- HardwareFusionPass: 硬件不支持时降级到 Atomic
- DeadCodeElimination: 移除未使用节点

**设备适配** (`DeviceProfile`):
- x86_64: SSE2 → AVX → AVX2 → AVX-512（通过 `simd_width` / `use_avx512` 参数化）
- AArch64: NEON → SVE（通过 `use_neon` / `use_sve` 参数化）
- GPU: CUDA (PTX) / HIP (AMDGPU) / Metal (AIR)（feature-gated）

**铁律**：所有算子必须走完整管线（Scalar → SymExec → IR → ISA Lowering），禁止跳过任何阶段。详见下方「JIT 编译管线（铁律）」章节。

### 5.1 JIT 编译管线实现完成度（审计 2026-03-15）

| 阶段 | 模块 | 完成度 | 状态 |
|------|------|--------|------|
| Phase 0: Scalar + SymExec | `registry.rs` + `symexec/` | 95% | ✅ 全算子注册 + x86_64 SymExec 完整；AArch64 SymExec 缺失 |
| Phase 1: SemanticDAG | `semantic_dag.rs` | 100% | ✅ OpClass 自动分类 + AI 计算 + Bottleneck 分类 |
| Phase 2: Fusion + HW | `fusion.rs` + `hw_constraints.rs` | 95% | ✅ 6 种融合模式全部实现 + 寄存器/L1 约束检查 |
| Phase 3: x86_64 | `codegen/x86_64.rs` | 100% | ✅ 生产就绪：6 种融合模式 + AVX2/AVX-512 + BLIS + Norm JIT + AMX |
| Phase 3: AArch64 | `codegen/aarch64_dynasm.rs` | 100% | ✅ GEMM + TileLevelFusion/ComputeRoot NEON/SVE norm JIT 融合 + emit_norm_row_jit() 自动分派 SVE/NEON |
| Phase 3: GPU | `codegen/gpu_ir/` | 100% | ✅ PTX/HIP/MSL 三后端 TileLevelFusion/ComputeRoot codegen 全部实现 |
| gllm 图优化 | `src/graph/optimizer/` | 100% | ✅ 6 个 Pattern Pass + HardwareFusionPass + DCE |

**全部完成** (2026-03-15 审计确认)

### 5.2 背景要求（AI 开发必读）

> **核心定位**：gllm 是一个以 JIT 编译为技术基础的融合算子推理引擎。所有性能关键路径必须通过 JIT 管线生成机器码，而非手写汇编或预编译库。
>
> **开发约束**：
> 1. 新增算子必须走完整四阶段管线（Scalar → SymExec → IR → ISA Lowering），禁止跳过
> 2. 融合决策由 Phase 2 自动完成，禁止在 codegen 层硬编码融合逻辑
> 3. ISA 选择由 `DeviceProfile` 驱动，codegen 通过 `simd_width` / `use_avx512` 等参数适配，禁止写死特定 ISA
> 4. GPU codegen 通过 `GpuDialect` trait 抽象，新增后端只需实现 trait 方法
> 5. 性能优化的优先级：融合算子 > 原子算子 > scalar fallback
> 6. 任何"绕过 JIT 管线"的 workaround（如直接调用预编译函数、静默 NOP、scalar fallback）均视为 bug

## Directory Structure

```
src/
├── lib.rs              # Library entry & public API re-exports
├── compat.rs           # Compatibility shim: re-exports gllm-kernels types (Backend, CpuBackend, Element, etc.)
├── client.rs           # Client / AsyncClient (sync/async inference API)
├── embeddings.rs       # Embeddings API
├── rerank.rs           # Rerank API
├── generation.rs       # Generation loop
├── tokenizer.rs        # Tokenizer integration
├── model_config.rs     # Tensor-driven model config (Ω1)
├── weight_loader.rs    # Weight loading utilities
├── kv_cache.rs         # KV Cache structures
├── quantization.rs     # Quantization support
├── loader/             # Model fetching & parsing
│   ├── mod.rs          # Unified loading entry (auto format detection)
│   ├── hf_hub.rs       # HuggingFace Hub downloader
│   ├── modelscope.rs   # ModelScope downloader
│   ├── downloader.rs   # Download orchestration (HF→MS fallback)
│   ├── format_detector.rs # Auto format detection
│   ├── safetensors.rs  # SafeTensors parser (zero-copy)
│   ├── adapter.rs      # GGUF→kernels type adapter
│   ├── parallel.rs     # Parallel layer loading
│   ├── pytorch.rs      # PyTorch format support
│   ├── gguf/           # GGUF parser (zero-copy, Ω1 compliant)
│   └── onnx/           # ONNX protobuf parser (prost, graph pattern matching)
├── arch/               # Architecture YAML templates → OnnxGraph
│   ├── mod.rs          # Template registry
│   ├── registry.rs     # Architecture registry
│   ├── resolve.rs      # Architecture resolution from metadata
│   └── template.rs     # YAML → OnnxGraph parser
├── graph/              # DAG optimizer (unified representation)
│   ├── mod.rs
│   ├── types.rs        # OnnxGraph extended types
│   ├── executor.rs     # FusedGraph executor
│   └── optimizer/      # Optimization passes (pattern/hardware fusion, DCE)
├── engine/             # Execution engine (wraps gllm-kernels)
│   ├── mod.rs
│   ├── executor.rs     # Executor (batch orchestration)
│   └── pipeline.rs     # Pipeline management
├── scheduler/          # Batching & KV Cache management (HGAL)
│   ├── mod.rs
│   ├── paged_scheduler.rs  # PagedAttention core
│   ├── batcher.rs      # Continuous Batching
│   ├── hgal.rs         # HGAL scheduler algorithm
│   ├── allocator.rs    # Page allocator
│   ├── memory_manager.rs   # GlobalMemoryManager
│   ├── prefix_index.rs # KvPrefixIndex (trie-based)
│   ├── sequence.rs     # SequenceGroup
│   ├── types.rs        # Scheduler types
│   ├── observer.rs     # RuntimeObserver
│   ├── policy.rs       # JIT scheduling policies
│   ├── jit_types.rs    # JIT type definitions
│   └── vllm2024.rs     # SwiftKV / legacy structures
├── backend/            # Backend detection & fallback
│   ├── mod.rs
│   ├── detection.rs    # Auto-detect CUDA→ROCm→Metal→CPU
│   └── fallback.rs     # OOM fallback (GPU→CPU)
└── manifest/           # Model manifest types
    ├── mod.rs
    └── types.rs
```

## Cache Directory

**Model Cache Root**: `~/.gllm/models/`

下载的模型文件存储在此目录。内部子目录结构由下载库（hf-hub/ModelScope）管理。

**Environment Variables**:
- `GLLM_CACHE_DIR`: 自定义缓存路径（默认：`~/.gllm/models`）— 📋 计划中，尚未实现
- `HF_TOKEN`: HuggingFace 认证 token

> **自动回退**: HuggingFace 下载失败时会自动切换到 ModelScope，无需手动指定来源。

## 🚨 环境变量铁律

**禁止擅自添加环境变量**：
- ❌ AI 禁止擅自引入新的环境变量
- ❌ AI 禁止为"灵活性"或"可配置性"添加环境变量
- ✅ 新增环境变量必须由用户明确要求
- ✅ 环境变量必须有明确的、不可替代的使用场景

**理由**：环境变量是公开 API，添加容易删除难。过度使用环境变量会导致配置爆炸和用户体验下降。

## 🚨 禁止引入推理引擎/算法库依赖 (REQ-ARCH-003)

**铁律：gllm 的计算核心完全自研，禁止引入任何外部推理引擎或深度学习算法库**：
- ❌ 禁止引入 `candle`、`candle-core`、`tch`、`torch`、`ort`、`tract`、`burn` 等推理框架
- ❌ 禁止引入 `pytorch`、`tensorflow`、`onnxruntime` 等深度学习运行时
- ❌ 禁止通过 optional feature 绕过此规则（即使是 `optional = true` 也不行）
- ✅ 计算核心必须完全由 gllm-kernels 提供
- ✅ 仅允许底层工具库：`safetensors`（格式解析）、`zip`（解压缩）、`prost`（protobuf）、`half`（f16 类型）等
- ✅ 需要的算法（如 pickle 解析、张量布局计算）必须自行实现

**理由**：gllm 定位为底层算子库，引入同类推理框架会造成依赖冲突、二进制膨胀、架构污染。

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
cargo test --test test_e2e_embedding -- --test-threads=1
cargo test --test test_e2e_generator -- --test-threads=1
cargo test --test test_e2e_reranker -- --test-threads=1

# 单元测试可以并行（不涉及真实 I/O）
cargo test --lib
```

## 🧪 测试哲学

**Pipeline 测试原则**：直接用现有 API 跑本地模型，哪里断了就是真实 bug，不做任何 workaround。

## 🚨 禁止 Fallback 绕过

**铁律：功能缺失必须补全实现，禁止用 fallback 绕过**：
- ❌ JIT 编译失败时禁止 fallback 到手写 Rust 实现
- ❌ 融合算子失败时禁止降级到原子算子拼接
- ❌ 禁止用 stub/dummy 返回值（如 `Ok(vec![0.0])`）绕过未实现的功能
- ✅ 测试失败说明功能缺失，必须补全真正的实现
- ✅ JIT/融合/算子有 bug 就修 bug，不做 workaround

**理由**：测试的目的是验证真实实现的正确性。Fallback 会掩盖真实问题，导致功能永远无法完成。

## 🚨 禁止 JIT Codegen 静默降级 (NO_SILENT_FALLBACK)

**铁律：JIT codegen 遇到无法生成代码的 OpKind 必须返回 `Err`，禁止静默 NOP**：

- ❌ `emit_nop_raw()` / `emit_nop_placeholder()` 作为未实现 op 的 catch-all
- ❌ `match _ => Ok(())` 吞掉未知 OpKind
- ❌ `eprintln!("[WARN]...")` + scalar 计算替代 JIT 编译失败
- ❌ `#[cfg(not(target_arch))]` 提供静默标量路径绕过 JIT
- ✅ 未实现的 op 必须 `Err(format!("codegen not implemented for {:?}", op_kind))`
- ✅ 仅 `Reshape` / `Transpose`（纯元数据 op，不需要计算）允许 NOP 处理

**理由**：NOP placeholder 让编译成功、测试通过，但输出是全零或内存垃圾。这是最危险的 bug 类型 — 静默产生错误结果，无法通过常规测试发现。

## 🚨 JIT 编译管线（铁律）

**所有算子必须走完整的 JIT 编译管线，无例外**：

```
算法 (Scalar Rust) → Lifting (SymExec trace) → IR (TraceOp SSA) → ISA Lowering (DeviceProfile) → 机器码
```

### 管线各阶段

| 阶段 | 输入 | 输出 | 职责 |
|------|------|------|------|
| **算法** | 数学定义 | `extern "C" fn` scalar 参考实现 | 定义算子语义，与硬件无关 |
| **Lifting** | scalar fn + `ScalarOpRegistry` | `OpTrace` + `ComputePattern` | SymExec 追踪标量执行，提取 SSA trace |
| **IR** | `TraceOp` SSA | `FusionPlan` + grouped ops | 硬件无关的算子融合、调度决策 |
| **ISA Lowering** | IR + `DeviceProfile` | x86_64 / AArch64 机器码 | 根据 `simd_width`、ISA 特性生成向量化代码 |

### 禁止事项

- ❌ 禁止跳过 Lifting 阶段直接手写 ISA 汇编（如直接写 AVX2 指令实现 Softmax）
- ❌ 禁止在 JIT 代码中通过 `call` 指令调用预编译的 Rust/C 函数
- ❌ 禁止把 JIT 当成"调度器"——只负责串联预编译函数
- ❌ 禁止写死特定 ISA（如只支持 AVX2），必须通过 `DeviceProfile` 参数化
- ✅ 新算子必须：注册 scalar 参考实现 → SymExec 提取 trace → codegen 根据硬件 lower
- ✅ codegen 层通过 `self.simd_width` / `self.use_avx512` 适配不同硬件
- ✅ 每个算子的 scalar 参考实现是 ground truth，JIT 生成的代码必须与之数值一致

### 推论

- 宁可暂时不实现某个算子（让测试失败），也不要绕过管线
- 如果现有 IR 不支持某种计算模式（如 reduction），必须先扩展 IR，再实现算子
- `ScalarOpRegistry` 是算子注册的唯一入口，不存在"特殊算子"可以绕过
