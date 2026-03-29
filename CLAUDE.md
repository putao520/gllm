# gllm

**Inference Client** — High-level library for model management, scheduling, and engine orchestration.

## SPEC Location
- `./SPEC/` (Single Source of Truth, 9 documents, 104+ REQs)
- `../gllm-kernels/SPEC/` (Backend constraints)

## SPEC Index

| Document | Content |
|----------|---------|
| `01-REQUIREMENTS.md` | 极化硅晶与通信墙生存阈值要求 (包含 TurboQuant 静态极化与 NUMA/PCIe/RDMA 硬件拓扑探测约束) |
| `02-ARCHITECTURE.md` | 4层物理架构, 彻底废弃 Native Float 软路由分流，全量拥抱 Mega-Kernel 与 Residual Bus |
| `03-DATA-STRUCTURE.md` | 终极数据容器极化洗骨 (统一物理页框 Unified Virtual Page 与双轨极化显存池 Dual-Track Pool) |
| `04-API-DESIGN.md` | 客户端公共 API (包含新加入的 Knowledge Injection & Intent SDK) |
| `06-TESTING-STRATEGY.md` | 物理路由与并发抗毁考场 (JIT 热分发并行阻击、5-Byte 原子级 Hot JMP Patching 试炼) |
| `07-OBSERVABILITY.md` | In-Kernel 核心寄生与物理灭顶法则 (STG 物理页头烙印与 Piggybacking 零通信 DMA 顺风回传) |
| `ARCH-DETAILED-DESIGNS.md` | ISV integration, quantized GEMM, GPU backend, adaptive chunking |
| `P0-P3-ROADMAP.md` | Priority roadmap (all P0-P3 completed) |
| `SUPPORTED_MODELS.md` | 20+ model architectures (generators/embeddings/rerankers) |
| `DOCS/scheduling/jit-cache-protocol.md` | JIT 编译缓存协议: 模型级缓存键, 全层融合粒度, 自适应 Tiling |
| `DOCS/scheduling/unified-jit-architecture-master.md` | (SSOT 终极蓝图) Mega-Kernel 路由, TurboQuant 双轨内存池, 知识插管 API |
| `DOCS/scheduling/ai-development-guideline.md` | 极简化内核执行底线开发思想原则 |
| `DOCS/scheduling/hgal-scheduler-algorithm.md` | HGAL 调度算法规划基准 |

## Technology Stack

| Component | Technology | Role |
|-----------|------------|------|
| **Loader** | `hf-hub`, `safetensors`, `prost`, `memmap2` | Model fetching, zero-copy loading, ONNX/GGUF/PyTorch parsing |
| **Tokenizer** | `tokenizers` | Text <-> ID conversion |
| **Scheduler** | Custom (HGAL) | PagedAttention, Continuous Batching, KvPrefixIndex |
| **Engine** | `gllm-kernels` + `compat/` shim | Hardware abstraction (compat bridges types + forward passes) |
| **Graph** | Custom DAG + `serde_yaml` | Unified OnnxGraph representation & fusion optimization |
| **Backend** | `gllm-kernels` | Auto-detect CUDA/ROCm/Metal/CPU, JIT codegen |

## Core Architecture

### 0. Core Philosophy: Accuracy First (准确度优先)
> **🚨 Critical Difference from vLLM**:
> vLLM and similar frameworks often prioritize throughput by using out-of-order execution in continuous batching, approximate attention masks, or aggressive quantization. This can degrade inference accuracy, especially in long-context or complex instruction-following scenarios.
>
> **gllm Principles**:
> 1.  **Accuracy > Throughput**: Never sacrifice calculation precision for scheduling optimization.
> 2.  **Strict Causal Ordering**: Intra-batch attention computation must guarantee strict causal masking. No out-of-order execution that risks context drift.
> 3.  **Reliability First**: Memory management (PagedAttention) must have strict boundary checks and error recovery. Prefer OOM rejection over returning corrupted results. JIT must emit hardware Trap on Out-of-Bounds memory without CPU fallback.
> 4.  **Deterministic Scheduling**: To combat floating-point non-associativity, batches must be strictly ordered (e.g., by RequestId). We prefer deterministic serial execution over messy parallel reduction.
> 5.  **JIT Hot-Repair & Block Routing (物理核内调度)**: 摒弃主机层面的复杂分流切换，一切交由运行时 JIT 发射的原子擦写指令 (Hot JMP Patching / DCE) 及根据 `SystemTopology` 切割的物理块式异构动态图处理。

### 1. Backend Constraints (from gllm-kernels)
- **Quantization**: Template-based kernels (1/2/4/8-bit unified)
- **GPU Execution**: L3 GPU-Pure API (zero-copy generation loop)
- **JIT PTX**: Phase 3 codegen 生成 PTX → cuModuleLoadData 动态加载（多版本 SM 特化，禁止 Fallback）

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
| Phase 3: ISA Lowering | `codegen/x86_64.rs` / `aarch64_dynasm.rs` / `gpu_ir/` | DeviceProfile 驱动的代码生成 (AVX2/AVX-512/NEON/SVE/PTX/HIP/MSL) |

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
- PTX 多版本调度: `PtxKernelRegistry` 按 SM 版本选择最优内核（sm_70/80/90/100+），禁止 Fallback

**铁律**：所有算子必须走完整管线（Scalar → SymExec → IR → ISA Lowering），禁止跳过任何阶段。详见下方「JIT 编译管线（铁律）」章节。

**铁律（JIT 缓存协议）**：
- **推理热路径（decode step 层循环）中禁止任何编译行为**（`InferenceCompiler::new()` / `compile_graph()` / `build_*_graph()`）。
- **编译只发生在模型加载时**。编译产物缓存于 `ModelJitCache`（三级: L1 模型级 → L2 全局 LRU → L3 磁盘）。
- **动态维度（seq_len, total_seq）通过 `SymDim::Symbolic` + `ShapeBinding` 运行时绑定**，不触发重编译。
- **缓存粒度 = 全层融合图**（`FusedAttentionLayer` / `FusedFfnLayer`），不是单算子。
- 详见 `SPEC/DOCS/scheduling/jit-cache-protocol.md`（REQ-JIT-CACHE-001~007）。

**铁律（CPU/GPU 算子与流程统一 — ARCH-CPU-GPU-UNIFIED）**：
- **CPU 和 GPU 后端共享完全一致的 `CompilerGraph` IR 和 `GraphType`**。同一个全层融合图通过 `DeviceProfile` 驱动 Phase 3 codegen 生成不同 ISA 机器码。
- **禁止出现任何以 `Cpu` / `Gpu` / 后端名称为前缀的 `GraphType` 变体**（如 ❌ `CpuDecoderLayer`、❌ `CpuKvProjection`）。后端差异由 `ModelArchKey.backend` 字段和 `DeviceProfile` 处理。
- **禁止出现 CPU 专用的图构建器**（如 ❌ `build_decoder_layer_graph()`）。CPU 的 `compile_model_graphs()` 必须调用与 GPU 相同的 `build_fused_*_symbolic()` 图构建器。
- **禁止出现子算子级 `GraphType`**（如 ❌ `KvProjection`、❌ `QRope`、❌ `Norm2`、❌ `SwiGluActivation`）。缓存粒度始终为全层融合图。
- **CPU 和 GPU 给定相同输入和权重，输出必须在浮点精度范围内一致**。P4/P5 算子（`EntropyGate`、`VRangeQuant` 等）在 CPU codegen 中可以使用简化标量实现，但必须语义正确。
- 详见 `SPEC/DOCS/scheduling/jit-cache-protocol.md`（ARCH-CPU-GPU-UNIFIED, §2.2, §2.3）。

### 5.1 JIT 编译管线实现完成度（审计 2026-03-17）

| 阶段 | 模块 | 完成度 | 状态 |
|------|------|--------|------|
| Phase 0: Scalar + SymExec | `registry.rs` + `symexec/` | 95% | ✅ 全算子注册 + x86_64 SymExec 完整；AArch64 SymExec 缺失 |
| Phase 1: SemanticDAG | `semantic_dag.rs` | 100% | ✅ OpClass 自动分类 + AI 计算 + Bottleneck 分类 |
| Phase 2: Fusion + HW | `fusion.rs` + `hw_constraints.rs` | 100% | ✅ 7 条 FusionRule + FusionEngine + 寄存器/L1 约束检查 |
| Phase 3: x86_64 | `codegen/x86_64.rs` | 100% | ✅ 生产就绪：6 种融合模式 + AVX2/AVX-512 + BLIS + Norm JIT + AMX + FMA 全指令族 + VDPBF16PS 原生 BF16 路径 |
| Phase 3: AArch64 | `codegen/aarch64_dynasm.rs` | 100% | ✅ GEMM + TileLevelFusion/ComputeRoot NEON/SVE norm JIT 融合 + FMA 全指令族 |
| Phase 3: GPU | `codegen/gpu_ir/` | 100% | ✅ PTX/HIP/MSL 三后端 TileLevelFusion/ComputeRoot codegen 全部实现 |
| gllm 图优化 | `src/graph/optimizer/` | 100% | ✅ 6 个 Pattern Pass + HardwareFusionPass + ConstantFolding + DCE |

**全部完成** (2026-03-17 审计确认)

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
├── client.rs           # Client / AsyncClient (sync/async inference API)
├── embeddings.rs       # Embeddings API
├── rerank.rs           # Rerank API
├── generation.rs       # Generation loop
├── tokenizer.rs        # Tokenizer integration
├── model_config.rs     # Tensor-driven model config (Ω1)
├── weight_loader.rs    # Weight loading utilities
├── weight_names.rs     # Canonical weight name mappings
├── kv_cache.rs         # KV Cache structures
├── quantization.rs     # Quantization support
├── ffi.rs              # C FFI exports (gllm_init/generate/destroy/client_version)
├── compat/             # gllm-kernels compatibility shim & forward passes
│   ├── mod.rs          # Re-exports gllm-kernels types (Backend, Element, etc.)
│   ├── types.rs        # Bridged type definitions
│   ├── scalar_ops.rs   # Scalar reference impls (Phase 0 only, NO runtime calls)
│   ├── jit_helpers.rs  # JIT compilation helpers
│   ├── weight_helpers.rs # Weight tensor manipulation
│   ├── bert_forward.rs # BERT/encoder forward pass
│   ├── decoder_forward.rs # Decoder (GPT-style) forward pass
│   ├── gpu_compile.rs  # GPU graph compilation & kernel launch
│   ├── gpu_helpers.rs  # GPU memory & transfer utilities
│   ├── memory.rs       # Memory management abstractions
│   ├── cpu_backend.rs  # CPU inference backend
│   ├── cuda_backend.rs # CUDA inference backend
│   ├── hip_backend.rs  # HIP (ROCm) inference backend
│   └── metal_backend.rs # Metal inference backend
├── loader/             # Model fetching & parsing
│   ├── mod.rs          # Unified loading entry (auto format detection)
│   ├── hf_hub.rs       # HuggingFace Hub downloader
│   ├── modelscope.rs   # ModelScope downloader
│   ├── downloader.rs   # Download orchestration (HF→MS fallback)
│   ├── format_detector.rs # Auto format detection
│   ├── safetensors.rs  # SafeTensors parser (zero-copy)
│   ├── adapter.rs      # 类型桥接层 (DType/KernelTensorView/GgufAdapter/QuantType 映射)
│   ├── parallel.rs     # Parallel layer loading
│   ├── pytorch.rs      # PyTorch format support
│   ├── gguf/           # GGUF parser (zero-copy, Ω1 compliant)
│   │   ├── mod.rs
│   │   ├── reader.rs   # GGUF file reader
│   │   ├── slice.rs    # Zero-copy tensor slicing
│   │   └── types.rs    # GGUF type definitions
│   └── onnx/           # ONNX protobuf parser (prost, graph pattern matching)
│       ├── mod.rs
│       ├── model.rs    # ONNX model loading
│       ├── attributes.rs # ONNX attribute parsing
│       ├── types.rs    # ONNX type mappings
│       ├── external.rs # External data loading
│       ├── pack.rs     # Tensor packing
│       ├── tests.rs    # ONNX parser tests
│       └── tensor/     # Tensor parsing
├── arch/               # Architecture YAML templates → OnnxGraph
│   ├── mod.rs          # Template registry
│   ├── registry.rs     # Architecture registry
│   ├── resolve.rs      # Architecture resolution from metadata
│   ├── template.rs     # YAML → OnnxGraph parser
│   └── templates/      # Model architecture YAML definitions
│       ├── deepseek.yaml
│       ├── gemma2.yaml
│       ├── glm4.yaml
│       ├── gpt2next.yaml
│       ├── mistral3.yaml
│       ├── phi4.yaml
│       ├── qwen3.yaml
│       └── xlmr.yaml
├── graph/              # DAG optimizer (unified representation)
│   ├── mod.rs
│   ├── types.rs        # OnnxGraph extended types
│   ├── executor.rs     # FusedGraph executor
│   └── optimizer/      # Optimization passes
│       ├── mod.rs
│       ├── pass.rs           # Pass trait & registry
│       ├── pattern_fusion.rs # Pattern-based fusion (FlashAttn, SwiGLU, etc.)
│       ├── hardware_fusion.rs # Hardware-aware fusion/degradation
│       ├── constant_folding.rs # Constant folding pass
│       └── dead_code.rs      # Dead code elimination
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
│   ├── observer.rs     # RuntimeObserver (metrics & tracing)
│   ├── policy.rs       # JIT scheduling policies (hot-switchable)
│   ├── jit_types.rs    # JIT type definitions
│   └── vllm2024.rs     # SwiftKV / Scheduler2024Config / AdaptiveChunkPolicy
├── backend/            # Backend detection & fallback
│   ├── mod.rs
│   ├── detection.rs    # Auto-detect CUDA→ROCm→Metal→CPU
│   └── fallback.rs     # OOM fallback (GPU→CPU)
├── manifest/           # Model manifest types
│   ├── mod.rs
│   └── types.rs
└── bin/                # Binary utilities
    ├── download.rs     # Model download CLI
    └── debug_shape.rs  # Shape debugging tool
```

## Cache Directory

**Model Cache Root**: `~/.gllm/models/`

下载的模型文件存储在此目录。内部子目录结构由下载库（hf-hub/ModelScope）管理。

**Environment Variables**:
- `GLLM_CACHE_DIR`: 自定义缓存路径（默认：`~/.gllm/models`）
- `HF_TOKEN`: HuggingFace 认证 token

> **自动回退**: HuggingFace 下载失败时会自动切换到 ModelScope，无需手动指定来源。

**JIT Cache**: `~/.gllm/jit_cache/`

JIT 编译的全层融合图二进制缓存（L3 磁盘层）。7 天 TTL 自动清理。
- **Debug 模式** (`cargo test` / `cargo build`)：L3 磁盘缓存完全禁用（不生产、不使用），确保每次编译产出最新代码
- **Release 模式** (`cargo build --release`)：L3 磁盘缓存正常工作，启动时自动清理超过 7 天的缓存

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

## 🚨 禁止调用外部 BLAS 库 (GPU GEMM 全 JIT)

**铁律：GPU GEMM 全部走 JIT codegen 生成设备原生二进制，禁止调用 cuBLAS/rocBLAS 等外部 BLAS 库**：
- ❌ 禁止在 GPU 推理路径中调用 cuBLAS/rocBLAS/cuDNN
- ❌ 禁止在 `dispatch_isv_sgemm` 中添加 GPU BLAS 分支
- ❌ 禁止在 `GpuIsvCapabilities` 中添加外部库可用性字段
- ✅ GPU GEMM 由 JIT 编译管线生成 PTX/HIP/MSL 原生二进制
- ✅ `GpuIsvCapabilities` 仅包含 `tensor_core_gen`（硬件矩阵单元代数），驱动 JIT 指令选择
- ✅ `GemmStrategy` GPU 路径为 `JitGpuTensorCore` / `JitGpu`，不存在 `CuBlas` / `RocBlas`
- ✅ CPU ISV（oneDNN/Accelerate）保留，因为 CPU 路径无 JIT GPU codegen

**理由**：JIT 融合算子直接驻留缓存对齐内存页，调用外部预编译库的 ROI 过低。gllm 的 JIT 可以根据 (模型结构 × 硬件能力) 生成比预编译库更优的代码。

## 🚨 严禁动态 DType 分发 (ARCH-DTYPE-ADAPTIVE 物理废弃)

**铁律：纯浮点动态调度 (`F32/F16/BF16`) 及动态计算 `dtype_size` 的架构已被全盘否决并物理销毁！系统唯一绑定物理位宽常量 `TurboQuantBits`。**

### 核心禁止规则 (Architect Veto)

1. ❌ **禁止任何基于浮点或动态 DType 的分发流**：禁止在运行时或编译时根据张量原始特征使用 `match dtype { F16 => ... }` 进行逻辑分支硬编码。
2. ❌ **禁止反量化回浮点**：不再提供 `Backend::dequantize`，严禁在运行时逆向量化至 F32 规约计算。数学运算纯基于极化微字节或定点硬派累加器实现。
3. ❌ **禁止动态寻址映射表**：原本关于 `dtype.size_bytes()` 动态寻址的复杂映射表与实现方案被永久废除。禁止实现此模块中的多态汇编下发逻辑！
4. ❌ **移除浮点混淆**：即使原始权重为 F16/BF16，所有算子统一并入静态编译位宽处理管线，精度差异化在 Load-time 被物理抹抹平。

**正确架构导向**: 系统强依赖定轨和极致物理约束（如 TurboQuant 预研架构中的 `DualTrackPool` 方案和 `Static Bit-width Execution`），将一切特征张量推回不可降级的原色矩阵运算，实现零代价极速吞吐。

## Common Commands

```bash
# 编译检查
cargo check

# 单元测试（可并行）
cargo test --lib

# E2E 测试（必须单线程）
cargo test --test test_e2e_embedding -- --test-threads=1
cargo test --test test_e2e_generator -- --test-threads=1
cargo test --test test_e2e_reranker -- --test-threads=1

# gllm-kernels 测试
cd ../gllm-kernels && cargo test --lib
cd ../gllm-kernels && cargo test --test decision_audit
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

## 🚨 Scalar 函数全面禁止 (NO_SCALAR — 零容忍)

**铁律：`scalar_ops.rs` 中的所有函数禁止在任何代码路径中被调用，包括运行时和测试代码**。

**背景**：
gllm 的核心定位是 JIT 编译融合算子推理引擎。所有计算必须通过 JIT 管线（Scalar→SymExec→IR→ISA Lowering）生成硬件最优代码。scalar 参考实现仅作为 Phase 0 的算子语义定义，供 SymExec trace 提取使用，不得在任何其他场景被调用。

**禁止的函数**（`src/compat/scalar_ops.rs`）：
- `scalar_gemm()` / `scalar_rms_norm()` / `scalar_rope()`
- `scalar_moe_gate()` / `scalar_top_k_experts()` / `scalar_expert_ffn()` / `scalar_moe_ffn()`
- `cached_gqa_attention()` / `prefill_gqa_attention()` / `swiglu_ffn()`

**禁止规则**：
- ❌ 禁止在运行时推理路径中调用任何 `scalar_*` 函数
- ❌ 禁止在单元测试中调用 `scalar_*` 函数验证算子正确性（必须用 JIT 编译的算子测试）
- ❌ 禁止以"测试需要参考实现"为借口在 `#[cfg(test)]` 中调用 scalar 函数
- ❌ 禁止以"量化 GEMM 独立优化路径"为借口调用 `scalar_gemm`
- ❌ 禁止以"某个 op 没有 JIT 实现"为借口把同层其他 op 降级为 scalar
- ❌ 禁止 `use.*scalar_ops` 出现在 `scalar_ops.rs` 以外的任何文件中
- ✅ `scalar_ops.rs` 中的函数仅供 `ScalarOpRegistry` 注册 + SymExec trace 提取
- ✅ 测试必须通过 JIT 编译图（`CompilerGraph` → `compile_and_run`）验证算子正确性
- ✅ 所有计算（GEMM、RmsNorm、RoPE、Attention、MoE）都必须走 JIT 或硬件优化路径

**理由**：scalar 实现比 JIT SIMD 代码慢 4-16 倍。允许任何 scalar 调用（即使在测试中）会掩盖 JIT 管线的真实问题，导致"测试通过但性能灾难"的假象。测试必须验证 JIT 生成代码的正确性，而非 scalar 参考实现的正确性。

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
| **ISA Lowering** | IR + `DeviceProfile` | x86_64 / AArch64 机器码 / PTX / HIP / MSL | 根据硬件特性生成最优代码（CPU: SIMD 向量化; GPU: SM 版本特化内核） |

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

## 🚨 Fallback 审计清单（2026-03-17，第二轮审计完成）

**原则：只有 SPEC 中明确记录或人类明确同意的 Fallback 才能保留。AI 无权判断 Fallback 是否可接受。**

### ✅ SPEC 已授权 Fallback（5 个）

| # | Fallback | SPEC 授权 | 实现位置 |
|---|----------|----------|---------|
| A2 | HF→ModelScope 下载源切换 | REQ-LOADER-016 | `src/loader/downloader.rs` |
| A3 | ONNX Fusion→Atomic（模式不匹配时） | ARCH-ONNX (02-ARCHITECTURE.md:445) | `src/graph/optimizer/pattern_fusion.rs` |
| A4 | HW Fusion→Standalone（硬件约束违反时） | ARCH-DETAILED-DESIGNS.md:1064,1275 | `gllm-kernels/src/compiler/hw_constraints.rs` |
| A5 | Reshape/Transpose 元数据 NOP | CLAUDE.md NO_SILENT_FALLBACK 例外 | `src/compat/gpu_compile.rs` |

### ✅ 第一轮未授权 Fallback — 已全部修复（B1-B11，2026-03-17）

| # | 文件 | 修复内容 | 提交 |
|---|------|---------|------|
| B1 | `src/client.rs` | `detect_architecture()` 失败 → 返回 `ClientError::UnknownModel` | gllm `2515c9c` |
| B2 | `src/compat/cpu_backend.rs` | `get_system_memory_pressure()` 失败 → 传播错误 | gllm `2515c9c` |
| B3 | `src/compat/cpu_backend.rs` | softmax sum=0 → 返回错误 | gllm `2515c9c` |
| B4 | `src/compat/decoder_forward.rs` | rerank token ID 缺失 → 返回错误 | gllm `2515c9c` |
| B5 | `src/compat/gpu_compile.rs` | 未知 GPU platform → 返回错误（两处） | gllm `2515c9c` |
| B6 | `src/engine/executor.rs` | 无 KV cache 输出 → 返回 `ExecutorError::OnnxPlan` | gllm `2515c9c` |
| B7 | `src/engine/executor.rs` | session position None → 返回 `ExecutorError::Scheduler` | gllm `2515c9c` |
| B8 | `src/model_config.rs` | tensor-driven 失败 → 直接传播错误，移除 fallback chain | gllm `2515c9c` |
| B9 | `gllm-kernels/.../x86_64.rs` | Paged attention on CPU → 返回错误 | gllm-kernels `74b5a687` |
| B10 | `gllm-kernels/.../trace_emitter.rs` | SM<70 PTX codegen → 返回错误 | gllm-kernels `74b5a687` |
| B11 | `gllm-kernels/.../backend.rs` | `CpuFallbackBackend` scalar 执行 → 返回错误 | gllm-kernels `74b5a687` |

### ✅ 第二轮未授权 Fallback — 已全部修复（C1-C3，2026-03-17）

| # | 文件 | 修复内容 | 提交 |
|---|------|---------|------|
| C1 | `src/backend/fallback.rs` | rerank `unwrap_or(0.0)` → 返回错误 | gllm `654642b` |
| C2 | `src/compat/gpu_compile.rs` | tensor ptr `unwrap_or(0)` NULL 指针 → 返回错误（4 处） | gllm `654642b` |
| C3 | `src/compat/memory.rs` | page_size `unwrap_or(4096)` → 传播解析错误 | gllm `654642b` |

### 🟡 已审计排除项（合法模式，非 Fallback）

| 位置 | 模式 | 排除理由 |
|------|------|---------|
| `decoder_forward.rs:291-292` | MoE config `unwrap_or(0)` | 非 MoE 模型 `moe_config` 为 None，0 表示"无 MoE"语义正确 |
| `paged_scheduler.rs:351` | `unwrap_or_default()` | 不存在的 request 返回空 block table，调用方已处理 |
| `paged_scheduler.rs:404` | `unwrap_or(0)` | 不存在的 sequence group 返回 0 used tokens，统计场景合理 |
| `cpu_backend.rs:198` | `partial_cmp().unwrap_or(Equal)` | Rust NaN 比较标准模式，非错误掩盖 |
| `ffi.rs:100` | CString `unwrap_or_default()` | FFI 边界，CString::new 仅在含 NUL 时失败，极端边界 |
| `gpu_compile.rs:2108,2138` | `partial_cmp().unwrap_or(Equal)` | 浮点排序标准模式 |
| `model_config.rs` GGUF 可选字段 | `unwrap_or(0.0/1.0/false)` | GGUF 元数据可选字段，缺失时使用行业标准默认值 |

### ❌ SPEC 明确禁止的 Fallback 模式（7 类）

| 禁止类型 | SPEC 来源 | 审计命令 |
|---------|----------|---------|
| JIT Codegen 静默 NOP | NO_SILENT_FALLBACK | `grep -rn "emit_nop" gllm-kernels/src/compiler/codegen/` |
| Scalar 函数调用 | NO_SCALAR | `grep -rn "scalar_" src/ --include="*.rs" \| grep -v scalar_ops.rs` |
| PTX SM 版本 Fallback | REQ-KERNELS-PTX-MV-003 | `grep -rn "_ =>" gllm-kernels/src/compiler/codegen/ptx/` |
| Mock/Stub 测试 | TEST-REAL-001 | `grep -rn "mock\|stub" tests/` |
| 静默错误处理 | ARCH-ERR (07-OBSERVABILITY.md) | `grep -rn "unwrap_or\|Err(_)" src/ \| grep -v test` |
| 内存压力默认值 | REQ-OBS-001 | `grep -rn "unwrap_or(0.0)" src/` |
| 元数据默认值 (Ω1) | 02-ARCHITECTURE.md §2 | `grep -rn "unwrap_or" src/model_config.rs` |
